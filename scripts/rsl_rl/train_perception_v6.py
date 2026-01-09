"""
Train Perception V6 (Auto-regressive): Previous output fed back as input
Based on ANYmal Parkour paper approach:
- Transform previous output to current robot frame
- Concatenate with current input
- No hidden state - the output IS the memory

Key differences from V5 (ConvGRU):
- Memory is explicit (previous reconstruction)
- Interpretable - can visualize what's remembered
- Requires coordinate transform between frames
"""

import os
import argparse
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from datetime import datetime
import psutil
import gc

def get_memory_usage():
    """Return RAM and VRAM usage in GB."""
    ram_gb = psutil.Process().memory_info().rss / 1024**3
    if torch.cuda.is_available():
        vram_gb = torch.cuda.memory_allocated() / 1024**3
        vram_max_gb = torch.cuda.max_memory_allocated() / 1024**3
    else:
        vram_gb = vram_max_gb = 0
    return ram_gb, vram_gb, vram_max_gb

# --- CONFIGURATION ---
MAP_SIZE = 40  
MAP_RES = 0.05 

# --- IMPORT PROJECTOR ---
try:
    from preprocess_perception import DepthProjector
except ImportError:
    raise ImportError("Could not import 'preprocess_perception.py'.")


# --- COORDINATE TRANSFORM ---
def transform_height_map(prev_map, prev_pos, prev_yaw, curr_pos, curr_yaw):
    """
    Transform previous height map from prev robot frame to current robot frame.
    
    Args:
        prev_map: (B, 1, H, W) height map in previous robot frame
        prev_pos: (B, 3) previous robot position (world frame)
        prev_yaw: (B,) previous robot yaw
        curr_pos: (B, 3) current robot position (world frame)
        curr_yaw: (B,) current robot yaw
        
    Returns:
        transformed_map: (B, 1, H, W) height map in current robot frame
    """
    B, _, H, W = prev_map.shape
    device = prev_map.device
    
    # Relative transform: how did robot move from prev to curr?
    # In world frame
    delta_pos_world = curr_pos[:, :2] - prev_pos[:, :2]  # (B, 2) - only x, y
    delta_yaw = curr_yaw - prev_yaw  # (B,)
    
    # Transform delta_pos to previous robot frame
    cos_prev = torch.cos(-prev_yaw)
    sin_prev = torch.sin(-prev_yaw)
    delta_x_robot = delta_pos_world[:, 0] * cos_prev - delta_pos_world[:, 1] * sin_prev
    delta_y_robot = delta_pos_world[:, 0] * sin_prev + delta_pos_world[:, 1] * cos_prev
    
    # Create sampling grid for grid_sample
    # We want to sample prev_map at locations that correspond to current frame grid
    
    # Grid in current robot frame: [-1, 1] normalized
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )
    grid = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)
    
    # Convert normalized coords to meters
    half_size = MAP_SIZE * MAP_RES / 2.0  # 1.0m for 40x40 at 5cm
    grid_meters = grid * half_size  # (B, H, W, 2) in meters
    
    # Transform grid from current frame to previous frame
    # 1. Rotate by -delta_yaw (undo rotation)
    cos_delta = torch.cos(-delta_yaw).view(B, 1, 1, 1)
    sin_delta = torch.sin(-delta_yaw).view(B, 1, 1, 1)
    
    grid_x_rot = grid_meters[..., 0] * cos_delta.squeeze(-1) - grid_meters[..., 1] * sin_delta.squeeze(-1)
    grid_y_rot = grid_meters[..., 0] * sin_delta.squeeze(-1) + grid_meters[..., 1] * cos_delta.squeeze(-1)
    
    # 2. Translate by -delta_pos (undo translation)
    grid_x_trans = grid_x_rot - delta_x_robot.view(B, 1, 1)
    grid_y_trans = grid_y_rot - delta_y_robot.view(B, 1, 1)
    
    # Convert back to normalized coords
    grid_transformed = torch.stack([
        grid_x_trans / half_size,
        grid_y_trans / half_size
    ], dim=-1)  # (B, H, W, 2)
    
    # Sample from previous map
    # grid_sample expects (B, C, H, W) and grid (B, H, W, 2) with values in [-1, 1]
    transformed_map = F.grid_sample(
        prev_map, 
        grid_transformed,
        mode='bilinear',
        padding_mode='zeros',  # Outside regions become 0
        align_corners=True
    )
    
    return transformed_map


def euler_from_quat(quat):
    """Extract yaw from quaternion (w, x, y, z format)."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    
    return yaw


# --- LOSS FUNCTION ---
class HybridTerrainLoss(nn.Module):
    def __init__(self, w_sparse=20.0, w_occluded=1.0, w_grad=10.0):
        super().__init__()
        self.w_sparse = w_sparse
        self.w_occluded = w_occluded
        self.w_grad = w_grad

    def forward(self, pred, target, mask):
        observed_mask = 1.0 - mask
        l1_error = torch.abs(pred - target)
        
        n_obs = observed_mask.sum() + 1e-6
        n_occ = mask.sum() + 1e-6
        
        loss_sparse = (l1_error * observed_mask).sum() / n_obs
        loss_occluded = (l1_error * mask).sum() / n_occ
        
        def get_grads(img):
            dx = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
            dy = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
            return dx, dy
        
        pred_dx, pred_dy = get_grads(pred)
        gt_dx, gt_dy = get_grads(target)
        loss_grad = F.l1_loss(pred_dx, gt_dx) + F.l1_loss(pred_dy, gt_dy)

        total_loss = (self.w_sparse * loss_sparse) + \
                     (self.w_occluded * loss_occluded) + \
                     (self.w_grad * loss_grad)

        metrics = {
            "loss_sparse": loss_sparse.item(),
            "loss_occluded": loss_occluded.item(),
            "loss_grad": loss_grad.item(),
            "l1_obs_cm": loss_sparse.item() * 100.0, 
            "l1_occ_cm": loss_occluded.item() * 100.0
        }
        return total_loss, metrics


# --- DATA AUGMENTATION ---
def augment_sequence(sparse_seq, mask_seq, target_seq, grav_seq, pos_seq, yaw_seq, p_rotate=0.5, p_flip=0.5):
    """Apply SAME augmentation to entire sequence."""
    do_rotate = torch.rand(1).item() < p_rotate
    k = torch.randint(1, 4, (1,)).item() if do_rotate else 0
    
    do_flip = torch.rand(1).item() < p_flip
    flip_horizontal = torch.rand(1).item() < 0.5 if do_flip else False
    
    T, B, C, H, W = sparse_seq.shape
    grav_out = grav_seq.clone()
    pos_out = pos_seq.clone()
    yaw_out = yaw_seq.clone()
    
    if do_rotate and k > 0:
        sparse_seq = torch.rot90(sparse_seq, k, dims=[-2, -1])
        mask_seq = torch.rot90(mask_seq, k, dims=[-2, -1])
        target_seq = torch.rot90(target_seq, k, dims=[-2, -1])
        
        # Rotate gravity and positions
        for t in range(T):
            for b in range(B):
                gx, gy = grav_out[t, b, 0].clone(), grav_out[t, b, 1].clone()
                px, py = pos_out[t, b, 0].clone(), pos_out[t, b, 1].clone()
                for _ in range(k):
                    gx_new, gy_new = -gy, gx
                    px_new, py_new = -py, px
                    gx, gy = gx_new, gy_new
                    px, py = px_new, py_new
                grav_out[t, b, 0], grav_out[t, b, 1] = gx, gy
                pos_out[t, b, 0], pos_out[t, b, 1] = px, py
                yaw_out[t, b] = yaw_out[t, b] + k * (np.pi / 2)
    
    if do_flip:
        if flip_horizontal:
            sparse_seq = torch.flip(sparse_seq, dims=[-1])
            mask_seq = torch.flip(mask_seq, dims=[-1])
            target_seq = torch.flip(target_seq, dims=[-1])
            grav_out[:, :, 1] = -grav_out[:, :, 1]
            pos_out[:, :, 1] = -pos_out[:, :, 1]
            yaw_out = -yaw_out
        else:
            sparse_seq = torch.flip(sparse_seq, dims=[-2])
            mask_seq = torch.flip(mask_seq, dims=[-2])
            target_seq = torch.flip(target_seq, dims=[-2])
            grav_out[:, :, 0] = -grav_out[:, :, 0]
            pos_out[:, :, 0] = -pos_out[:, :, 0]
            yaw_out = np.pi - yaw_out
    
    return sparse_seq, mask_seq, target_seq, grav_out, pos_out, yaw_out


# --- SEQUENTIAL DATASET ---
class MeldogSequentialDataset(Dataset):
    def __init__(self, h5_path, seq_len=16, stride=8):
        self.h5_path = h5_path
        self.seq_len = seq_len
        self.stride = stride
        self.sequences = []
        
        with h5py.File(h5_path, 'r') as f:
            for ep_name in f.keys():
                if "gt_height" not in f[ep_name]: 
                    continue
                ep_len = f[ep_name]["gt_height"].shape[0]
                
                for start in range(0, ep_len - seq_len + 1, stride):
                    self.sequences.append((ep_name, start))
        
        print(f"[INFO] Created {len(self.sequences)} sequences of length {seq_len}")
        self._h5_file = None

    def __len__(self): 
        return len(self.sequences)
    
    def _get_h5_file(self):
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, 'r', libver='latest', swmr=False)
        return self._h5_file

    def __getitem__(self, idx):
        h5_file = self._get_h5_file()
        
        ep_name, start = self.sequences[idx]
        end = start + self.seq_len
        grp = h5_file[ep_name]
        
        stack_seq = np.stack([
            np.stack([
                grp["depth_front"][start:end],
                grp["depth_rear"][start:end], 
                grp["depth_left"][start:end],
                grp["depth_right"][start:end]
            ], axis=1)
        ], axis=0).squeeze(0)
        
        quat_seq = grp["robot_quat"][start:end]
        pos_seq = grp["robot_pos"][start:end]
        gt_seq = grp["gt_height"][start:end]
        
        return stack_seq, quat_seq, pos_seq, gt_seq
    
    def __del__(self):
        if self._h5_file is not None:
            try:
                self._h5_file.close()
            except:
                pass


# --- GPU PREPROCESSOR ---
class GPUProcessor(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        self.projector = DepthProjector(map_size=MAP_SIZE, map_res=MAP_RES, device=device)

    def forward(self, stack_raw, quat_raw, target_raw):
        depth_stack = stack_raw.float() * 0.001 
        with torch.no_grad():
            sparse_map, occlusion_mask = self.projector(depth_stack, quat_raw)
        target = target_raw.float() * 0.001
        target = target.unsqueeze(1) 
        
        w, x, y, z = quat_raw[:, 0], quat_raw[:, 1], quat_raw[:, 2], quat_raw[:, 3]
        gx, gy, gz = -2*(x*z + w*y), -2*(y*z - w*x), -(1 - 2*(x*x + y*y))
        grav_vec = torch.stack([gx, gy, gz], dim=1)
        
        yaw = euler_from_quat(quat_raw)
        
        return sparse_map, occlusion_mask, grav_vec, target, yaw


# --- AUTO-REGRESSIVE MODEL ---
class SparseMapRefinerAutoReg(nn.Module):
    """
    Deep U-Net with auto-regressive feedback.
    Input: current sparse + mask + transformed previous output
    No hidden state - memory is explicit in the previous output.
    """
    def __init__(self):
        super().__init__()
        
        # Input: 2 (sparse + mask) + 1 (prev output) + 1 (prev valid mask) = 4 channels
        self.enc1 = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.ReLU()
        )
        
        # Gravity embedding
        self.mlp_gravity = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 64)
        )
        
        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128 + 64, 64, 4, 2, 1), nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, 32, 4, 2, 1), nn.ReLU()
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(32 + 32, 16, 4, 2, 1), nn.ReLU()
        )
        
        self.final_conv = nn.Conv2d(16 + 4, 1, 3, padding=1)

    def forward(self, x, mask, prev_output, prev_valid, grav):
        """
        x: (B, 1, 40, 40) sparse map
        mask: (B, 1, 40, 40) occlusion mask  
        prev_output: (B, 1, 40, 40) previous output transformed to current frame
        prev_valid: (B, 1, 40, 40) mask of where prev_output is valid (not out of bounds)
        grav: (B, 3) gravity vector
        
        returns: pred (B, 1, 40, 40)
        """
        # Concatenate all inputs
        x_in = torch.cat([x, mask, prev_output, prev_valid], dim=1)  # B, 4, 40, 40
        
        # Encode
        s1 = self.enc1(x_in)   # B, 32, 20, 20
        s2 = self.enc2(s1)     # B, 64, 10, 10
        s3 = self.enc3(s2)     # B, 128, 5, 5
        
        # Add gravity embedding
        B, _, H, W = s3.shape
        grav_embed = self.mlp_gravity(grav).view(B, 64, 1, 1).expand(B, 64, H, W)
        
        # Decode
        up1 = self.dec1(torch.cat([s3, grav_embed], dim=1))  # B, 64, 10, 10
        up2 = self.dec2(torch.cat([up1, s2], dim=1))          # B, 32, 20, 20
        up3 = self.dec3(torch.cat([up2, s1], dim=1))          # B, 16, 40, 40
        
        pred = self.final_conv(torch.cat([up3, x_in], dim=1))

        # fix nan appear during training
        pred = torch.clamp(pred, min=-5.0, max=5.0)
        
        return pred


# --- DATASET REPACKER ---
def repack_dataset(src_path, dst_path, compression='lzf'):
    if compression == 'none':
        comp_param = None
        comp_name = "uncompressed"
    elif compression == 'lzf':
        comp_param = 'lzf'
        comp_name = "LZF"
    else:
        comp_param = compression
        comp_name = compression.upper()
    
    print(f"\n[INFO] 📦 Optimizing dataset for training ({comp_name})...")
    
    with h5py.File(src_path, 'r') as src, h5py.File(dst_path, 'w') as dst:
        for ep_name in tqdm(src.keys(), desc=f"Repacking to {comp_name}"):
            src_grp = src[ep_name]
            dst_grp = dst.create_group(ep_name)
            for key in src_grp.keys():
                data = src_grp[key][:]
                if comp_param is None:
                    dst_grp.create_dataset(key, data=data, compression=None, chunks=True)
                else:
                    dst_grp.create_dataset(key, data=data, compression=comp_param, chunks=True)
    print(f"[INFO] ✅ Dataset optimization complete\n")


def get_optimized_path(original_path, compression):
    base = original_path.replace(".h5", "")
    if compression == 'none':
        return f"{base}_opt_uncompressed.h5"
    elif compression == 'lzf':
        return f"{base}_opt_lzf.h5"
    else:
        return f"{base}_opt_{compression}.h5"


# --- MAIN TRAINING ---
def main(args):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    aug_tag = "_Aug" if args.augment else ""
    tf_tag = f"_TF{int(args.teacher_forcing*100)}" if args.teacher_forcing < 1.0 else ""
    run_name = f"PM_v6_AutoReg_seq{args.seq_len}{aug_tag}{tf_tag}_{timestamp}"
    save_dir = os.path.join("logs", "perception", run_name)
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=save_dir)

    # Dataset selection
    data_path = args.data
    if data_path == "auto":
        import glob
        files = glob.glob("datasets/*/*.h5")
        data_path = sorted([f for f in files if "_opt_" not in f], key=os.path.getmtime)[-1]
    
    print(f"[INFO] 📂 Selected dataset: {data_path}")
    
    fast_data_path = get_optimized_path(data_path, args.compression)
    if os.path.exists(fast_data_path):
        print(f"[INFO] ✅ Using existing optimized dataset: {fast_data_path}")
    else:
        repack_dataset(data_path, fast_data_path, compression=args.compression)

    full_dataset = MeldogSequentialDataset(
        fast_data_path, 
        seq_len=args.seq_len, 
        stride=args.stride
    )
    
    train_size = int(0.9 * len(full_dataset))
    train_set, val_set = torch.utils.data.random_split(
        full_dataset, [train_size, len(full_dataset) - train_size]
    )
    
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, 
        num_workers=args.workers, pin_memory=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = SparseMapRefinerAutoReg().to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Model parameters: {n_params:,}")
    print(f"[INFO] Sequence length: {args.seq_len}")
    print(f"[INFO] Teacher forcing ratio: {args.teacher_forcing}")
    
    gpu_processor = GPUProcessor(device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Load checkpoint if provided
    start_epoch = 0
    if args.checkpoint:
        print(f"[INFO] Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint.get("epoch", 0)
            print(f"[INFO] Resumed from epoch {start_epoch}")
        else:
            model.load_state_dict(checkpoint)
            print(f"[INFO] Loaded model weights only")
    
    criterion = HybridTerrainLoss(
        w_sparse=args.w_sparse,
        w_occluded=args.w_occluded,
        w_grad=args.w_grad
    )
    scaler = GradScaler('cuda')

    print(f"\n[INFO] Loss weights: sparse={args.w_sparse}, occluded={args.w_occluded}, grad={args.w_grad}")
    print(f"[INFO] Augmentation: {args.augment}")
    print()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_metrics = {"loss": 0, "l1_obs": 0, "l1_occ": 0, "grad": 0}
        
        # Decay teacher forcing over training
        tf_ratio = args.teacher_forcing * (1 - epoch / args.epochs) if args.tf_decay else args.teacher_forcing
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for stack_seq, quat_seq, pos_seq, target_seq in pbar:
            # Permute to (T, B, ...)
            stack_seq = stack_seq.permute(1, 0, 2, 3, 4).to(device)
            quat_seq = quat_seq.permute(1, 0, 2).to(device)
            pos_seq = pos_seq.permute(1, 0, 2).to(device)
            target_seq = target_seq.permute(1, 0, 2, 3).to(device)
            
            T, B = stack_seq.shape[:2]
            
            # Process each timestep
            sparse_list, mask_list, grav_list, gt_list, yaw_list = [], [], [], [], []
            for t in range(T):
                sparse, mask, grav, gt, yaw = gpu_processor(
                    stack_seq[t], quat_seq[t], target_seq[t]
                )
                sparse_list.append(sparse)
                mask_list.append(mask)
                grav_list.append(grav)
                gt_list.append(gt)
                yaw_list.append(yaw)
            
            sparse_seq = torch.stack(sparse_list, dim=0)
            mask_seq = torch.stack(mask_list, dim=0)
            grav_seq = torch.stack(grav_list, dim=0)
            target_seq_proc = torch.stack(gt_list, dim=0)
            yaw_seq = torch.stack(yaw_list, dim=0)
            pos_seq_gpu = pos_seq.float()
            
            # Apply augmentation
            if args.augment:
                sparse_seq, mask_seq, target_seq_proc, grav_seq, pos_seq_gpu, yaw_seq = augment_sequence(
                    sparse_seq, mask_seq, target_seq_proc, grav_seq, pos_seq_gpu, yaw_seq
                )
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast('cuda'):
                total_loss = 0
                batch_l1_obs = 0
                batch_l1_occ = 0
                batch_grad = 0
                
                # Initialize previous output as zeros
                prev_output = torch.zeros(B, 1, MAP_SIZE, MAP_SIZE, device=device)
                prev_valid = torch.zeros(B, 1, MAP_SIZE, MAP_SIZE, device=device)
                prev_pos = pos_seq_gpu[0]
                prev_yaw = yaw_seq[0]
                
                for t in range(T):
                    curr_pos = pos_seq_gpu[t]
                    curr_yaw = yaw_seq[t]
                    
                    # Transform previous output to current frame
                    if t > 0:
                        prev_output_transformed = transform_height_map(
                            prev_output, prev_pos, prev_yaw, curr_pos, curr_yaw
                        )
                        # Create validity mask (where transform didn't go out of bounds)
                        prev_valid = (prev_output_transformed != 0).float()
                    else:
                        prev_output_transformed = prev_output
                        prev_valid = torch.zeros_like(prev_output)
                    
                    # Forward pass
                    pred = model(
                        sparse_seq[t], mask_seq[t], 
                        prev_output_transformed, prev_valid,
                        grav_seq[t]
                    )
                    
                    # Compute loss
                    loss, metrics = criterion(pred, target_seq_proc[t], mask_seq[t])
                    total_loss += loss
                    batch_l1_obs += metrics["l1_obs_cm"]
                    batch_l1_occ += metrics["l1_occ_cm"]
                    batch_grad += metrics["loss_grad"]
                    
                    # Update previous output for next timestep
                    # Teacher forcing: use GT instead of prediction sometimes
                    if torch.rand(1).item() < tf_ratio:
                        prev_output = target_seq_proc[t].detach()
                    else:
                        prev_output = pred.detach()
                    prev_pos = curr_pos
                    prev_yaw = curr_yaw
                
                total_loss /= T
                batch_l1_obs /= T
                batch_l1_occ /= T
                batch_grad /= T
            
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)  # Important when using Amp/Scaler
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_metrics["loss"] += total_loss.item()
            train_metrics["l1_obs"] += batch_l1_obs
            train_metrics["l1_occ"] += batch_l1_occ
            train_metrics["grad"] += batch_grad
            
            pbar.set_postfix({
                "L1_Obs": f"{batch_l1_obs:.2f}cm",
                "L1_Occ": f"{batch_l1_occ:.2f}cm",
                "TF": f"{tf_ratio:.2f}"
            })
        
        scheduler.step()
        
        # Validation (no teacher forcing)
        model.eval()
        val_metrics = {"loss": 0, "l1_obs": 0, "l1_occ": 0, "grad": 0}
        with torch.no_grad():
            for stack_seq, quat_seq, pos_seq, target_seq in val_loader:
                stack_seq = stack_seq.permute(1, 0, 2, 3, 4).to(device)
                quat_seq = quat_seq.permute(1, 0, 2).to(device)
                pos_seq = pos_seq.permute(1, 0, 2).to(device)
                target_seq = target_seq.permute(1, 0, 2, 3).to(device)
                
                T, B = stack_seq.shape[:2]
                
                sparse_list, mask_list, grav_list, gt_list, yaw_list = [], [], [], [], []
                for t in range(T):
                    sparse, mask, grav, gt, yaw = gpu_processor(
                        stack_seq[t], quat_seq[t], target_seq[t]
                    )
                    sparse_list.append(sparse)
                    mask_list.append(mask)
                    grav_list.append(grav)
                    gt_list.append(gt)
                    yaw_list.append(yaw)
                
                sparse_seq = torch.stack(sparse_list, dim=0)
                mask_seq = torch.stack(mask_list, dim=0)
                grav_seq = torch.stack(grav_list, dim=0)
                target_seq_proc = torch.stack(gt_list, dim=0)
                yaw_seq = torch.stack(yaw_list, dim=0)
                pos_seq_gpu = pos_seq.float()
                
                with autocast('cuda'):
                    total_loss = 0
                    batch_l1_obs = 0
                    batch_l1_occ = 0
                    batch_grad = 0
                    
                    prev_output = torch.zeros(B, 1, MAP_SIZE, MAP_SIZE, device=device)
                    prev_valid = torch.zeros(B, 1, MAP_SIZE, MAP_SIZE, device=device)
                    prev_pos = pos_seq_gpu[0]
                    prev_yaw = yaw_seq[0]
                    
                    for t in range(T):
                        curr_pos = pos_seq_gpu[t]
                        curr_yaw = yaw_seq[t]
                        
                        if t > 0:
                            prev_output_transformed = transform_height_map(
                                prev_output, prev_pos, prev_yaw, curr_pos, curr_yaw
                            )
                            prev_valid = (prev_output_transformed != 0).float()
                        else:
                            prev_output_transformed = prev_output
                            prev_valid = torch.zeros_like(prev_output)
                        
                        pred = model(
                            sparse_seq[t], mask_seq[t], 
                            prev_output_transformed, prev_valid,
                            grav_seq[t]
                        )
                        
                        loss, metrics = criterion(pred, target_seq_proc[t], mask_seq[t])
                        total_loss += loss
                        batch_l1_obs += metrics["l1_obs_cm"]
                        batch_l1_occ += metrics["l1_occ_cm"]
                        batch_grad += metrics["loss_grad"]
                        
                        # No teacher forcing in validation
                        prev_output = pred.detach()
                        prev_pos = curr_pos
                        prev_yaw = curr_yaw
                    
                    total_loss /= T
                    batch_l1_obs /= T
                    batch_l1_occ /= T
                    batch_grad /= T
                
                val_metrics["loss"] += total_loss.item()
                val_metrics["l1_obs"] += batch_l1_obs
                val_metrics["l1_occ"] += batch_l1_occ
                val_metrics["grad"] += batch_grad

        n_train = len(train_loader)
        n_val = len(val_loader)
        
        writer.add_scalar("Loss/Train_Total", train_metrics["loss"] / n_train, epoch)
        writer.add_scalar("Metrics/Train_L1_Obs_cm", train_metrics["l1_obs"] / n_train, epoch)
        writer.add_scalar("Metrics/Train_L1_Occ_cm", train_metrics["l1_occ"] / n_train, epoch)
        writer.add_scalar("Metrics/Train_Grad", train_metrics["grad"] / n_train, epoch)
        
        writer.add_scalar("Loss/Val_Total", val_metrics["loss"] / n_val, epoch)
        writer.add_scalar("Metrics/Val_L1_Obs_cm", val_metrics["l1_obs"] / n_val, epoch)
        writer.add_scalar("Metrics/Val_L1_Occ_cm", val_metrics["l1_occ"] / n_val, epoch)
        writer.add_scalar("Metrics/Val_Grad", val_metrics["grad"] / n_val, epoch)
        
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)
        writer.add_scalar("TeacherForcing", tf_ratio, epoch)
        
        checkpoint_path = os.path.join(save_dir, f"model_ep{epoch+1}.pt")
        torch.save({
            'epoch': epoch+1, 
            'model_state_dict': model.state_dict(), 
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'config': vars(args),
        }, checkpoint_path)
        
        val_obs = val_metrics['l1_obs'] / n_val
        val_occ = val_metrics['l1_occ'] / n_val
        
        # Memory monitoring
        ram_gb, vram_gb, vram_max_gb = get_memory_usage()
        writer.add_scalar("Memory/RAM_GB", ram_gb, epoch)
        writer.add_scalar("Memory/VRAM_GB", vram_gb, epoch)
        
        print(f"Epoch {epoch+1} | Val L1 Obs: {val_obs:.2f}cm | Val L1 Occ: {val_occ:.2f}cm | TF: {tf_ratio:.2f} | RAM: {ram_gb:.1f}GB | VRAM: {vram_gb:.1f}GB")
        
        # Periodic garbage collection
        if (epoch + 1) % 5 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    writer.close()
    print(f"\n[DONE] Training complete. Logs saved to: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--data", type=str, default="auto")
    parser.add_argument("--compression", type=str, default="lzf", choices=["none", "lzf", "gzip"])
    
    # Sequence parameters
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    
    # Teacher forcing
    parser.add_argument("--teacher_forcing", type=float, default=0.5,
                       help="Probability of using GT instead of prediction for previous frame")
    parser.add_argument("--tf_decay", action="store_true", default=True,
                       help="Decay teacher forcing over training")
    parser.add_argument("--no_tf_decay", action="store_false", dest="tf_decay")
    
    # Loss weights
    parser.add_argument("--w_sparse", type=float, default=20.0)
    parser.add_argument("--w_occluded", type=float, default=1.0)
    parser.add_argument("--w_grad", type=float, default=10.0)
    
    # Augmentation
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no_augment", action="store_false", dest="augment")
    
    # Resume training
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to checkpoint to resume training from")
    
    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        ram_gb, _, vram_max_gb = get_memory_usage()
        print(f"\n[FATAL ERROR] Training crashed!")
        print(f"[FATAL ERROR] RAM: {ram_gb:.1f}GB | VRAM_Max: {vram_max_gb:.1f}GB")
        print(f"[FATAL ERROR] Exception: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise