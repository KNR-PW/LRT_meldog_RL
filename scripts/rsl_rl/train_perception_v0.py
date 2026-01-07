"""
Train Perception V4: Sparse 2.5D Map Pre-processing (Gravity Aligned)
- Architecture: U-Net with Skip Connections + Occlusion Mask Input
- Loss Function: Mask-Weighted Hybrid Loss (Sharpness Focused)
- Hardware: RTX 4090 + Ryzen 9950X
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

# --- HYBRID LOSS FUNCTION ---
class HybridTerrainLoss(nn.Module):
    def __init__(self, w_sparse=20.0, w_occluded=1.0, w_grad=10.0):
        super().__init__()
        self.w_sparse = w_sparse
        self.w_occluded = w_occluded
        self.w_grad = w_grad

    def forward(self, pred, target, mask):
        observed_mask = 1.0 - mask
        
        # 1. Use L1 (MAE) instead of MSE to keep edges sharp
        l1_error = torch.abs(pred - target)
        
        # Loss for areas where the camera actually saw something
        loss_sparse = (l1_error * observed_mask).sum() / (observed_mask.sum() + 1e-6)
        
        # Loss for "hallucinating" occluded areas
        loss_occluded = (l1_error * mask).sum() / (mask.sum() + 1e-6)
        
        # 2. Gradient Loss (Sobel-like) to force sharp transitions
        def get_grads(img):
            # Difference between adjacent pixels
            dx = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
            dy = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
            return dx, dy

        pred_dx, pred_dy = get_grads(pred)
        gt_dx, gt_dy = get_grads(target)
        
        # Penalize if the "sharpness" of the prediction doesn't match the GT
        loss_grad = F.l1_loss(pred_dx, gt_dx) + F.l1_loss(pred_dy, gt_dy)

        total_loss = (self.w_sparse * loss_sparse) + \
                     (self.w_occluded * loss_occluded) + \
                     (self.w_grad * loss_grad)

        metrics = {
            "loss_sparse": loss_sparse.item(),
            "loss_grad": loss_grad.item(),
            "l1_obs_cm": loss_sparse.item() * 100.0, 
            "l1_occ_cm": loss_occluded.item() * 100.0
        }
        return total_loss, metrics
    
# --- DATASET REPACKER WITH CONFIGURABLE COMPRESSION ---
def repack_dataset(src_path, dst_path, compression='lzf'):
    """
    Repack HDF5 dataset with specified compression.
    
    Args:
        src_path: Source HDF5 file (typically gzip compressed from data collection)
        dst_path: Destination HDF5 file
        compression: 'none', 'lzf', or 'gzip' (can also specify gzip level like 'gzip:4')
    """
    # Handle compression parameter
    if compression == 'none':
        comp_param = None
        comp_name = "uncompressed"
    elif compression == 'lzf':
        comp_param = 'lzf'
        comp_name = "LZF"
    elif compression.startswith('gzip'):
        comp_param = compression
        comp_name = compression.upper()
    else:
        comp_param = compression
        comp_name = compression
    
    print(f"\n[INFO] 📦 Optimizing dataset for training ({comp_name})...")
    print(f"[INFO] Source: {src_path}")
    print(f"[INFO] Destination: {dst_path}")
    
    try:
        with h5py.File(src_path, 'r') as src, h5py.File(dst_path, 'w') as dst:
            for ep_name in tqdm(src.keys(), desc=f"Repacking to {comp_name}"):
                src_grp = src[ep_name]
                dst_grp = dst.create_group(ep_name)
                for key in src_grp.keys():
                    data = src_grp[key][:]
                    if comp_param is None:
                        # Uncompressed - no compression, but still use chunks for flexibility
                        dst_grp.create_dataset(key, data=data, compression=None, chunks=True)
                    else:
                        dst_grp.create_dataset(key, data=data, compression=comp_param, chunks=True)
        print(f"[INFO] ✅ Dataset optimization complete: {comp_name}\n")
    except Exception as e:
        print(f"[ERROR] Failed to repack: {e}")
        if os.path.exists(dst_path): 
            os.remove(dst_path)
        raise e

def get_optimized_path(original_path, compression):
    """Generate the optimized file path based on compression type."""
    base = original_path.replace(".h5", "")
    if compression == 'none':
        return f"{base}_opt_uncompressed.h5"
    elif compression == 'lzf':
        return f"{base}_opt_lzf.h5"
    elif compression.startswith('gzip'):
        return f"{base}_opt_gzip.h5"
    else:
        return f"{base}_opt_{compression}.h5"

class MeldogDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.index_map = []
        with h5py.File(h5_path, 'r') as f:
            for ep_name in f.keys():
                if "gt_height" not in f[ep_name]: continue
                ep_len = f[ep_name]["gt_height"].shape[0]
                self.index_map.extend([(ep_name, i) for i in range(ep_len)])

    def __len__(self): return len(self.index_map)

    def __getitem__(self, idx):
        if not hasattr(self, 'h5_file'):
            self.h5_file = h5py.File(self.h5_path, 'r', libver='latest', swmr=False)
        ep_name, step_idx = self.index_map[idx]
        grp = self.h5_file[ep_name]
        stack = np.stack([grp["depth_front"][step_idx], grp["depth_rear"][step_idx], 
                          grp["depth_left"][step_idx], grp["depth_right"][step_idx]], axis=0)
        return stack, grp["robot_quat"][step_idx], grp["gt_height"][step_idx]

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
        return sparse_map, occlusion_mask, grav_vec, target

class SparseMapRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        # Level 1: 40 -> 20
        self.enc1 = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1) # Learnable downsample
        )
        # Level 2: 20 -> 10
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1)
        )
        
        self.mlp_gravity = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 32))

        # Decoder with Transpose Convs for sharpness
        self.dec1 = nn.ConvTranspose2d(64 + 32, 32, kernel_size=4, stride=2, padding=1) 
        self.dec2 = nn.ConvTranspose2d(32 + 32, 16, kernel_size=4, stride=2, padding=1)
        self.final_conv = nn.Conv2d(16 + 2, 1, kernel_size=3, padding=1)

    def forward(self, x, mask, grav):
        x_in = torch.cat([x, mask], dim=1) # B, 2, 40, 40
        s1 = self.enc1(x_in)  # B, 32, 20, 20
        s2 = self.enc2(s1)    # B, 64, 10, 10
        
        B, _, H, W = s2.shape
        grav_embed = self.mlp_gravity(grav).view(B, 32, 1, 1).expand(B, 32, H, W)
        
        up1 = self.dec1(torch.cat([s2, grav_embed], dim=1)) # B, 32, 20, 20
        up2 = self.dec2(torch.cat([up1, s1], dim=1))       # B, 16, 40, 40
        
        return self.final_conv(torch.cat([up2, x_in], dim=1))
    
def main(args):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"PM_v4_WeightedLoss_{timestamp}"
    save_dir = os.path.join("logs", "perception", run_name)
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=save_dir)

    data_path = args.data
    if data_path == "auto":
        import glob
        files = glob.glob("datasets/*/*.h5")
        # Exclude already optimized files
        data_path = sorted([f for f in files if "_opt_" not in f], key=os.path.getmtime)[-1]
    
    print(f"[INFO] 📂 Selected dataset: {data_path}")
    
    # Generate optimized path based on compression setting
    fast_data_path = get_optimized_path(data_path, args.compression)
    
    # Check if optimized file already exists
    if os.path.exists(fast_data_path):
        print(f"[INFO] ✅ Optimized dataset already exists: {fast_data_path}")
        print(f"[INFO] Skipping repacking. Delete this file to force re-optimization.")
    else:
        print(f"[INFO] 🔄 Optimized dataset not found. Creating: {fast_data_path}")
        repack_dataset(data_path, fast_data_path, compression=args.compression)

    dataset = MeldogDataset(fast_data_path)
    train_size = int(0.9 * len(dataset))
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SparseMapRefiner().to(device)
    gpu_processor = GPUProcessor(device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    start_epoch = 0
    if args.load_model:
        if os.path.exists(args.load_model):
            print(f"[INFO] 🔄 Loading checkpoint: {args.load_model}")
            checkpoint = torch.load(args.load_model, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch']
                print(f"[INFO] ✅ Resuming from Epoch {start_epoch}")
            else:
                model.load_state_dict(checkpoint)
                match = re.search(r'ep(\d+)', os.path.basename(args.load_model))
                start_epoch = int(match.group(1)) if match else 0
                print(f"[INFO] ✅ Weights loaded. Epoch set to {start_epoch}")
        else: return

    criterion = HybridTerrainLoss()
    scaler = GradScaler('cuda')

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_metrics = {"loss": 0, "l1_obs": 0, "l1_occ": 0}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for stack_raw, quat_raw, target_raw in pbar:
            stack_raw, quat_raw, target_raw = stack_raw.to(device), quat_raw.to(device), target_raw.to(device)
            sparse_in, mask, grav, target = gpu_processor(stack_raw, quat_raw, target_raw)

            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda'):
                pred = model(sparse_in, mask, grav)
                loss, batch_metrics = criterion(pred, target, mask)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_metrics["loss"] += loss.item()
            train_metrics["l1_obs"] += batch_metrics["l1_obs_cm"]
            train_metrics["l1_occ"] += batch_metrics["l1_occ_cm"]
            pbar.set_postfix({"L1_Obs_cm": f"{batch_metrics['l1_obs_cm']:.2f}"})
        
        # Validation
        model.eval()
        val_metrics = {"loss": 0, "l1_obs": 0, "l1_occ": 0}
        with torch.no_grad():
            for s_r, q_r, t_r in val_loader:
                si, m, g, t = gpu_processor(s_r.to(device), q_r.to(device), t_r.to(device))
                with autocast('cuda'):
                    pred = model(si, m, g)
                    v_loss, v_m = criterion(pred, t, m)
                    val_metrics["loss"] += v_loss.item()
                    val_metrics["l1_obs"] += v_m["l1_obs_cm"]
                    val_metrics["l1_occ"] += v_m["l1_occ_cm"]

        # --- RESTORED ALL LOGGING METRICS ---
        n_train = len(train_loader)
        n_val = len(val_loader)
        
        writer.add_scalar("Loss/Train_Total", train_metrics["loss"] / n_train, epoch)
        writer.add_scalar("Metrics/Train_L1_Obs_cm", train_metrics["l1_obs"] / n_train, epoch)
        writer.add_scalar("Metrics/Train_L1_Occ_cm", train_metrics["l1_occ"] / n_train, epoch)
        
        writer.add_scalar("Loss/Val_Total", val_metrics["loss"] / n_val, epoch)
        writer.add_scalar("Metrics/Val_L1_Obs_cm", val_metrics["l1_obs"] / n_val, epoch)
        writer.add_scalar("Metrics/Val_L1_Occ_cm", val_metrics["l1_occ"] / n_val, epoch)
        
        checkpoint_path = os.path.join(save_dir, f"model_ep{epoch+1}.pt")
        torch.save({'epoch': epoch+1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)
        print(f"Epoch {epoch+1} Done. Val L1 Obs: {val_metrics['l1_obs']/n_val:.2f} cm")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=640)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--data", type=str, default="auto")
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--compression", type=str, default="lzf", 
                       choices=["none", "lzf", "gzip"],
                       help="Compression for optimized dataset: 'none' (uncompressed, fastest), 'lzf' (fast, balanced), 'gzip' (slower, smaller)")
    main(parser.parse_args())