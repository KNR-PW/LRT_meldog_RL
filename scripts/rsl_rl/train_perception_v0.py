"""
Train Perception V5 (Temporal): ConvGRU for temporal consistency
Key changes from V4.1:
- ConvGRU at bottleneck to maintain temporal state
- Sequential dataset (chunks of consecutive frames)
- Hidden state passed between frames, reset between episodes

Based on Miki et al. "Learning robust perceptive locomotion":
- GRU outperforms LSTM and RNN
- 2 stacked layers work well
- Belief state captures unobservable information
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
import re

# --- CONFIGURATION ---
MAP_SIZE = 40  
MAP_RES = 0.05 

# --- IMPORT PROJECTOR ---
try:
    from preprocess_perception import DepthProjector
except ImportError:
    raise ImportError("Could not import 'preprocess_perception.py'. Ensure it is in the working directory.")


# --- CONVOLUTIONAL GRU CELL ---
class ConvGRUCell(nn.Module):
    """
    Convolutional GRU cell - maintains spatial structure unlike regular GRU.
    Based on: "Convolutional LSTM Network" (Shi et al., 2015)
    """
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        
        # Gates: reset and update
        self.conv_gates = nn.Conv2d(
            input_channels + hidden_channels, 
            2 * hidden_channels,  # reset gate + update gate
            kernel_size, padding=padding
        )
        # Candidate hidden state
        self.conv_candidate = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels,
            kernel_size, padding=padding
        )
    
    def forward(self, x, h_prev):
        """
        x: (B, C_in, H, W) - current input
        h_prev: (B, C_hidden, H, W) - previous hidden state
        returns: h_new (B, C_hidden, H, W)
        """
        if h_prev is None:
            B, _, H, W = x.shape
            h_prev = torch.zeros(B, self.hidden_channels, H, W, device=x.device, dtype=x.dtype)
        
        combined = torch.cat([x, h_prev], dim=1)
        
        # Compute gates
        gates = torch.sigmoid(self.conv_gates(combined))
        reset_gate, update_gate = gates.chunk(2, dim=1)
        
        # Compute candidate
        combined_reset = torch.cat([x, reset_gate * h_prev], dim=1)
        candidate = torch.tanh(self.conv_candidate(combined_reset))
        
        # New hidden state
        h_new = (1 - update_gate) * h_prev + update_gate * candidate
        
        return h_new


class ConvGRU(nn.Module):
    """
    Multi-layer ConvGRU (stacked cells).
    """
    def __init__(self, input_channels, hidden_channels, num_layers=2, kernel_size=3):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_ch = input_channels if i == 0 else hidden_channels
            self.cells.append(ConvGRUCell(in_ch, hidden_channels, kernel_size))
    
    def forward(self, x, h_prev=None):
        """
        x: (B, C_in, H, W)
        h_prev: list of (B, C_hidden, H, W) for each layer, or None
        returns: output (B, C_hidden, H, W), h_new (list)
        """
        if h_prev is None:
            h_prev = [None] * self.num_layers
        
        h_new = []
        current = x
        for i, cell in enumerate(self.cells):
            current = cell(current, h_prev[i])
            h_new.append(current)
        
        return current, h_new


# --- LOSS FUNCTION (same as V4.1) ---
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
        
        # Gradient Loss
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


# --- DATA AUGMENTATION (applied to whole sequence consistently) ---
def augment_sequence(sparse_seq, mask_seq, target_seq, grav_seq, p_rotate=0.5, p_flip=0.5):
    """
    Apply SAME augmentation to entire sequence for consistency.
    """
    # Decide augmentation once for whole sequence
    do_rotate = torch.rand(1).item() < p_rotate
    k = torch.randint(1, 4, (1,)).item() if do_rotate else 0
    
    do_flip = torch.rand(1).item() < p_flip
    flip_horizontal = torch.rand(1).item() < 0.5 if do_flip else False
    
    T, B, C, H, W = sparse_seq.shape
    grav_out = grav_seq.clone()
    
    if do_rotate and k > 0:
        sparse_seq = torch.rot90(sparse_seq, k, dims=[-2, -1])
        mask_seq = torch.rot90(mask_seq, k, dims=[-2, -1])
        target_seq = torch.rot90(target_seq, k, dims=[-2, -1])
        
        # Rotate gravity for each timestep
        for t in range(T):
            for b in range(B):
                gx, gy = grav_out[t, b, 0].clone(), grav_out[t, b, 1].clone()
                for _ in range(k):
                    gx_new = -gy
                    gy_new = gx
                    gx, gy = gx_new, gy_new
                grav_out[t, b, 0], grav_out[t, b, 1] = gx, gy
    
    if do_flip:
        if flip_horizontal:
            sparse_seq = torch.flip(sparse_seq, dims=[-1])
            mask_seq = torch.flip(mask_seq, dims=[-1])
            target_seq = torch.flip(target_seq, dims=[-1])
            grav_out[:, :, 1] = -grav_out[:, :, 1]
        else:
            sparse_seq = torch.flip(sparse_seq, dims=[-2])
            mask_seq = torch.flip(mask_seq, dims=[-2])
            target_seq = torch.flip(target_seq, dims=[-2])
            grav_out[:, :, 0] = -grav_out[:, :, 0]
    
    return sparse_seq, mask_seq, target_seq, grav_out


# --- SEQUENTIAL DATASET ---
class MeldogSequentialDataset(Dataset):
    """
    Returns sequences of consecutive frames from episodes.
    Each item is (stack_seq, quat_seq, gt_seq) of shape (seq_len, ...)
    """
    def __init__(self, h5_path, seq_len=16, stride=8):
        """
        seq_len: number of consecutive frames per sequence
        stride: step between sequence starts (stride < seq_len = overlapping sequences)
        """
        self.h5_path = h5_path
        self.seq_len = seq_len
        self.stride = stride
        self.sequences = []  # List of (ep_name, start_idx)
        
        with h5py.File(h5_path, 'r') as f:
            for ep_name in f.keys():
                if "gt_height" not in f[ep_name]: 
                    continue
                ep_len = f[ep_name]["gt_height"].shape[0]
                
                # Create overlapping sequences
                for start in range(0, ep_len - seq_len + 1, stride):
                    self.sequences.append((ep_name, start))
        
        print(f"[INFO] Created {len(self.sequences)} sequences of length {seq_len}")

    def __len__(self): 
        return len(self.sequences)

    def __getitem__(self, idx):
        if not hasattr(self, 'h5_file'):
            self.h5_file = h5py.File(self.h5_path, 'r', libver='latest', swmr=False)
        
        ep_name, start = self.sequences[idx]
        end = start + self.seq_len
        grp = self.h5_file[ep_name]
        
        # Load sequence of frames
        stack_seq = np.stack([
            np.stack([
                grp["depth_front"][start:end],
                grp["depth_rear"][start:end], 
                grp["depth_left"][start:end],
                grp["depth_right"][start:end]
            ], axis=1)
        ], axis=0).squeeze(0)  # (seq_len, 4, H, W)
        
        quat_seq = grp["robot_quat"][start:end]  # (seq_len, 4)
        gt_seq = grp["gt_height"][start:end]     # (seq_len, 40, 40)
        
        return stack_seq, quat_seq, gt_seq


# --- GPU PREPROCESSOR ---
class GPUProcessor(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        self.projector = DepthProjector(map_size=MAP_SIZE, map_res=MAP_RES, device=device)

    def forward(self, stack_raw, quat_raw, target_raw):
        """
        Process a single frame (called per timestep in sequence).
        """
        depth_stack = stack_raw.float() * 0.001 
        with torch.no_grad():
            sparse_map, occlusion_mask = self.projector(depth_stack, quat_raw)
        target = target_raw.float() * 0.001
        target = target.unsqueeze(1) 
        
        # Gravity vector from quaternion
        w, x, y, z = quat_raw[:, 0], quat_raw[:, 1], quat_raw[:, 2], quat_raw[:, 3]
        gx, gy, gz = -2*(x*z + w*y), -2*(y*z - w*x), -(1 - 2*(x*x + y*y))
        grav_vec = torch.stack([gx, gy, gz], dim=1)
        
        return sparse_map, occlusion_mask, grav_vec, target


# --- TEMPORAL MODEL ---
class SparseMapRefinerTemporal(nn.Module):
    """
    Deep U-Net with ConvGRU at bottleneck for temporal consistency.
    Architecture: 40->20->10->5 with ConvGRU at 5x5
    """
    def __init__(self, gru_hidden=128, gru_layers=2):
        super().__init__()
        
        # Encoder (same as Deep model)
        self.enc1 = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
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
        
        # ConvGRU at bottleneck (5x5 spatial, 128+64=192 input channels)
        self.conv_gru = ConvGRU(
            input_channels=128 + 64,  # encoder output + gravity
            hidden_channels=gru_hidden,
            num_layers=gru_layers,
            kernel_size=3
        )
        
        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(gru_hidden, 64, 4, 2, 1), nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, 32, 4, 2, 1), nn.ReLU()
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(32 + 32, 16, 4, 2, 1), nn.ReLU()
        )
        
        self.final_conv = nn.Conv2d(16 + 2, 1, 3, padding=1)

    def forward(self, x, mask, grav, h_prev=None):
        """
        x: (B, 1, 40, 40) sparse map
        mask: (B, 1, 40, 40) occlusion mask
        grav: (B, 3) gravity vector
        h_prev: list of hidden states from ConvGRU, or None
        
        returns: pred (B, 1, 40, 40), h_new (list)
        """
        x_in = torch.cat([x, mask], dim=1)  # B, 2, 40, 40
        
        # Encode
        s1 = self.enc1(x_in)   # B, 32, 20, 20
        s2 = self.enc2(s1)     # B, 64, 10, 10
        s3 = self.enc3(s2)     # B, 128, 5, 5
        
        # Add gravity embedding
        B, _, H, W = s3.shape
        grav_embed = self.mlp_gravity(grav).view(B, 64, 1, 1).expand(B, 64, H, W)
        s3_grav = torch.cat([s3, grav_embed], dim=1)  # B, 192, 5, 5
        
        # ConvGRU (temporal processing)
        gru_out, h_new = self.conv_gru(s3_grav, h_prev)  # B, gru_hidden, 5, 5
        
        # Decode
        up1 = self.dec1(gru_out)                       # B, 64, 10, 10
        up2 = self.dec2(torch.cat([up1, s2], dim=1))   # B, 32, 20, 20
        up3 = self.dec3(torch.cat([up2, s1], dim=1))   # B, 16, 40, 40
        
        pred = self.final_conv(torch.cat([up3, x_in], dim=1))
        
        return pred, h_new
    
    def forward_sequence(self, x_seq, mask_seq, grav_seq):
        """
        Process entire sequence, maintaining hidden state.
        x_seq: (T, B, 1, H, W)
        mask_seq: (T, B, 1, H, W)
        grav_seq: (T, B, 3)
        
        returns: pred_seq (T, B, 1, H, W)
        """
        T = x_seq.shape[0]
        preds = []
        h = None
        
        for t in range(T):
            pred, h = self.forward(x_seq[t], mask_seq[t], grav_seq[t], h)
            preds.append(pred)
        
        return torch.stack(preds, dim=0)


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
    run_name = f"PM_v5_Temporal_seq{args.seq_len}{aug_tag}_{timestamp}"
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

    # Sequential dataset
    full_dataset = MeldogSequentialDataset(
        fast_data_path, 
        seq_len=args.seq_len, 
        stride=args.stride
    )
    
    train_size = int(0.9 * len(full_dataset))
    train_set, val_set = torch.utils.data.random_split(
        full_dataset, [train_size, len(full_dataset) - train_size]
    )
    
    # Note: batch_size here is number of sequences, each of length seq_len
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, 
        num_workers=args.workers, pin_memory=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = SparseMapRefinerTemporal(
        gru_hidden=args.gru_hidden,
        gru_layers=args.gru_layers
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Model parameters: {n_params:,}")
    print(f"[INFO] Sequence length: {args.seq_len}, GRU hidden: {args.gru_hidden}, GRU layers: {args.gru_layers}")
    
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
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for stack_seq, quat_seq, target_seq in pbar:
            # stack_seq: (B, T, 4, H, W) -> need (T, B, 4, H, W)
            stack_seq = stack_seq.permute(1, 0, 2, 3, 4).to(device)
            quat_seq = quat_seq.permute(1, 0, 2).to(device)
            target_seq = target_seq.permute(1, 0, 2, 3).to(device)
            
            T, B = stack_seq.shape[:2]
            
            # Process each timestep through GPU processor
            sparse_list, mask_list, grav_list, gt_list = [], [], [], []
            for t in range(T):
                sparse, mask, grav, gt = gpu_processor(
                    stack_seq[t], quat_seq[t], target_seq[t]
                )
                sparse_list.append(sparse)
                mask_list.append(mask)
                grav_list.append(grav)
                gt_list.append(gt)
            
            sparse_seq = torch.stack(sparse_list, dim=0)  # (T, B, 1, H, W)
            mask_seq = torch.stack(mask_list, dim=0)
            grav_seq = torch.stack(grav_list, dim=0)
            target_seq_proc = torch.stack(gt_list, dim=0)
            
            # Apply augmentation to whole sequence
            if args.augment:
                sparse_seq, mask_seq, target_seq_proc, grav_seq = augment_sequence(
                    sparse_seq, mask_seq, target_seq_proc, grav_seq
                )
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast('cuda'):
                # Forward through sequence
                pred_seq = model.forward_sequence(sparse_seq, mask_seq, grav_seq)
                
                # Compute loss over all timesteps
                total_loss = 0
                batch_l1_obs = 0
                batch_l1_occ = 0
                batch_grad = 0
                
                for t in range(T):
                    loss, metrics = criterion(
                        pred_seq[t], target_seq_proc[t], mask_seq[t]
                    )
                    total_loss += loss
                    batch_l1_obs += metrics["l1_obs_cm"]
                    batch_l1_occ += metrics["l1_occ_cm"]
                    batch_grad += metrics["loss_grad"]
                
                total_loss /= T
                batch_l1_obs /= T
                batch_l1_occ /= T
                batch_grad /= T
            
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_metrics["loss"] += total_loss.item()
            train_metrics["l1_obs"] += batch_l1_obs
            train_metrics["l1_occ"] += batch_l1_occ
            train_metrics["grad"] += batch_grad
            
            pbar.set_postfix({
                "L1_Obs": f"{batch_l1_obs:.2f}cm",
                "L1_Occ": f"{batch_l1_occ:.2f}cm"
            })
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_metrics = {"loss": 0, "l1_obs": 0, "l1_occ": 0, "grad": 0}
        with torch.no_grad():
            for stack_seq, quat_seq, target_seq in val_loader:
                stack_seq = stack_seq.permute(1, 0, 2, 3, 4).to(device)
                quat_seq = quat_seq.permute(1, 0, 2).to(device)
                target_seq = target_seq.permute(1, 0, 2, 3).to(device)
                
                T, B = stack_seq.shape[:2]
                
                sparse_list, mask_list, grav_list, gt_list = [], [], [], []
                for t in range(T):
                    sparse, mask, grav, gt = gpu_processor(
                        stack_seq[t], quat_seq[t], target_seq[t]
                    )
                    sparse_list.append(sparse)
                    mask_list.append(mask)
                    grav_list.append(grav)
                    gt_list.append(gt)
                
                sparse_seq = torch.stack(sparse_list, dim=0)
                mask_seq = torch.stack(mask_list, dim=0)
                grav_seq = torch.stack(grav_list, dim=0)
                target_seq_proc = torch.stack(gt_list, dim=0)
                
                with autocast('cuda'):
                    pred_seq = model.forward_sequence(sparse_seq, mask_seq, grav_seq)
                    
                    total_loss = 0
                    batch_l1_obs = 0
                    batch_l1_occ = 0
                    batch_grad = 0
                    
                    for t in range(T):
                        loss, metrics = criterion(
                            pred_seq[t], target_seq_proc[t], mask_seq[t]
                        )
                        total_loss += loss
                        batch_l1_obs += metrics["l1_obs_cm"]
                        batch_l1_occ += metrics["l1_occ_cm"]
                        batch_grad += metrics["loss_grad"]
                    
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
        
        # Log metrics
        writer.add_scalar("Loss/Train_Total", train_metrics["loss"] / n_train, epoch)
        writer.add_scalar("Metrics/Train_L1_Obs_cm", train_metrics["l1_obs"] / n_train, epoch)
        writer.add_scalar("Metrics/Train_L1_Occ_cm", train_metrics["l1_occ"] / n_train, epoch)
        writer.add_scalar("Metrics/Train_Grad", train_metrics["grad"] / n_train, epoch)
        
        writer.add_scalar("Loss/Val_Total", val_metrics["loss"] / n_val, epoch)
        writer.add_scalar("Metrics/Val_L1_Obs_cm", val_metrics["l1_obs"] / n_val, epoch)
        writer.add_scalar("Metrics/Val_L1_Occ_cm", val_metrics["l1_occ"] / n_val, epoch)
        writer.add_scalar("Metrics/Val_Grad", val_metrics["grad"] / n_val, epoch)
        
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)
        
        # Save checkpoint
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
        print(f"Epoch {epoch+1} | Val L1 Obs: {val_obs:.2f}cm | Val L1 Occ: {val_occ:.2f}cm | LR: {scheduler.get_last_lr()[0]:.2e}")

    writer.close()
    print(f"\n[DONE] Training complete. Logs saved to: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16, 
                       help="Number of sequences per batch (smaller due to seq_len)")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--data", type=str, default="auto")
    parser.add_argument("--compression", type=str, default="lzf", 
                       choices=["none", "lzf", "gzip"])
    
    # Sequence parameters
    parser.add_argument("--seq_len", type=int, default=32,
                       help="Number of consecutive frames per sequence")
    parser.add_argument("--stride", type=int, default=8,
                       help="Stride between sequence starts (overlap if < seq_len)")
    
    # GRU parameters
    parser.add_argument("--gru_hidden", type=int, default=128,
                       help="Hidden channels in ConvGRU")
    parser.add_argument("--gru_layers", type=int, default=2,
                       help="Number of stacked ConvGRU layers")
    
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
    
    main(parser.parse_args())