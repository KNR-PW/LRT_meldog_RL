"""
Train Perception V4: Sparse 2.5D Map Pre-processing (Gravity Aligned)
- Architecture: U-Net with Skip Connections + Occlusion Mask Input
- Loss Function: Hybrid Terrain Loss (MSE + TV + Gradient)
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
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from datetime import datetime
import re

# --- IMPORT PROJECTOR ---
try:
    from preprocess_perception import DepthProjector
except ImportError:
    raise ImportError("Could not import 'preprocess_perception.py'. Ensure it is in the working directory.")

# --- CONFIGURATION ---
MAP_SIZE = 40  # 40x40 Grid
MAP_RES = 0.05 # 5cm Resolution

# --- HYBRID LOSS FUNCTION ---
class HybridTerrainLoss(nn.Module):
    """
    Combines MSE for reconstruction, TV for smoothness (flat treads), 
    and Gradient Matching for sharpness (stair risers).
    """
    def __init__(self, w_mse=1.0, w_tv=0.01, w_grad=0.1):
        super().__init__()
        self.w_mse = w_mse
        self.w_tv = w_tv
        self.w_grad = w_grad

    def forward(self, pred, target):
        # 1. Reconstruction Loss (Standard Accuracy)
        loss_mse = F.mse_loss(pred, target)

        # 2. Total Variation (TV) Loss (Encourages flat plateaus/steps)
        # Difference between adjacent pixels
        tv_h = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :]).mean()
        tv_w = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1]).mean()
        loss_tv = tv_h + tv_w

        # 3. Gradient Loss (Forces sharp vertical edges)
        def get_gradients(img):
            grad_x = img[:, :, 1:, :] - img[:, :, :-1, :]
            grad_y = img[:, :, :, 1:] - img[:, :, :, :-1]
            return grad_x, grad_y

        pred_dx, pred_dy = get_gradients(pred)
        gt_dx, gt_dy = get_gradients(target)
        
        # Use L1 for gradients to prevent blurring
        loss_grad = F.l1_loss(pred_dx, gt_dx) + F.l1_loss(pred_dy, gt_dy)

        return (self.w_mse * loss_mse) + (self.w_tv * loss_tv) + (self.w_grad * loss_grad)

# --- DATASET REPACKER ---
def repack_dataset(src_path, dst_path, compression='lzf'):
    print(f"\n[INFO] 📦 Optimizing dataset for training (LZF)...")
    try:
        with h5py.File(src_path, 'r') as src, h5py.File(dst_path, 'w') as dst:
            for ep_name in tqdm(src.keys(), desc="Repacking Episodes"):
                src_grp = src[ep_name]
                dst_grp = dst.create_group(ep_name)
                for key in src_grp.keys():
                    data = src_grp[key][:]
                    dst_grp.create_dataset(key, data=data, compression=compression, chunks=True)
        print(f"[INFO] ✅ Dataset optimization complete.\n")
    except Exception as e:
        print(f"[ERROR] Failed to repack: {e}")
        if os.path.exists(dst_path): os.remove(dst_path)
        raise e

# --- DATASET ---
class MeldogDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.index_map = []
        with h5py.File(h5_path, 'r') as f:
            for ep_name in f.keys():
                if "gt_height" not in f[ep_name]: continue
                ep_len = f[ep_name]["gt_height"].shape[0]
                self.index_map.extend([(ep_name, i) for i in range(ep_len)])

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        if not hasattr(self, 'h5_file'):
            self.h5_file = h5py.File(self.h5_path, 'r', libver='latest', swmr=False)
        ep_name, step_idx = self.index_map[idx]
        grp = self.h5_file[ep_name]
        stack = np.stack([grp["depth_front"][step_idx], grp["depth_rear"][step_idx], 
                          grp["depth_left"][step_idx], grp["depth_right"][step_idx]], axis=0)
        return stack, grp["robot_quat"][step_idx], grp["gt_height"][step_idx]

# --- GPU PRE-PROCESSOR ---
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

# --- MODEL: SPARSE MAP REFINER (U-NET V4) ---
class SparseMapRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 2 Channels [Sparse Height, Occlusion Mask]
        self.enc1 = self.conv_block(2, 32)   
        self.enc2 = self.conv_block(32, 64)  
        self.enc3 = self.conv_block(64, 128) 
        
        self.mlp_gravity = nn.Sequential(
            nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 32)
        )

        # Decoder with Skip Connections
        self.dec1 = self.up_block(128 + 32, 64)  # Bottleneck + Grav
        self.dec2 = self.up_block(128, 32)       # Output dec1 + Skip enc2
        
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=3, padding=1) # Output up + Skip enc1

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, padding_mode='replicate'), 
            nn.BatchNorm2d(out_c), nn.ReLU(), nn.MaxPool2d(2)
        )

    def up_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), 
            nn.Conv2d(in_c, out_c, 3, padding=1, padding_mode='replicate'), 
            nn.BatchNorm2d(out_c), nn.ReLU()
        )

    def forward(self, x, mask, grav):
        x_in = torch.cat([x, mask], dim=1) 

        # Encoder with skip storage
        s1 = self.enc1(x_in)  # 20x20
        s2 = self.enc2(s1)    # 10x10
        s3 = self.enc3(s2)    # 5x5

        # Bottleneck + Gravity
        B, _, H, W = s3.shape
        grav_embed = self.mlp_gravity(grav).unsqueeze(-1).unsqueeze(-1).expand(B, 32, H, W)
        x = torch.cat([s3, grav_embed], dim=1) 

        # Decoder with Skip Connections (Concatenation)
        x = self.dec1(x)              # 10x10
        x = torch.cat([x, s2], dim=1) # 10x10 (64 + 64 = 128 channels)
        
        x = self.dec2(x)              # 20x20
        x = torch.cat([x, s1], dim=1) # 20x20 (32 + 32 = 64 channels)
        
        x = self.final_upsample(x)    # 40x40
        return self.final_conv(x)

# --- MAIN TRAINING LOOP ---
def main(args):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"PM_v4_Hybrid_UNet_{timestamp}"
    save_dir = os.path.join("logs", "perception", run_name)
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=save_dir)

    # Dataset Handling
    data_path = args.data
    if data_path == "auto":
        import glob
        files = glob.glob("datasets/*/*.h5")
        data_path = sorted([f for f in files if "_opt_" not in f], key=os.path.getmtime)[-1]
    
    fast_data_path = data_path.replace(".h5", f"_opt_lzf.h5")
    if not os.path.exists(fast_data_path):
        repack_dataset(data_path, fast_data_path)

    dataset = MeldogDataset(fast_data_path)
    train_size = int(0.9 * len(dataset))
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size])
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SparseMapRefiner().to(device)
    gpu_processor = GPUProcessor(device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Use Hybrid Loss instead of simple L1/MSE
    criterion = HybridTerrainLoss(w_mse=1.0, w_tv=0.01, w_grad=0.1)
    scaler = GradScaler()

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for stack_raw, quat_raw, target_raw in pbar:
            stack_raw, quat_raw, target_raw = stack_raw.to(device), quat_raw.to(device), target_raw.to(device)
            sparse_in, mask, grav, target = gpu_processor(stack_raw, quat_raw, target_raw)

            optimizer.zero_grad(set_to_none=True)
            with autocast():
                pred = model(sparse_in, mask, grav)
                loss = criterion(pred, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Val
        model.eval()
        val_l = 0
        with torch.no_grad():
            for s_r, q_r, t_r in val_loader:
                si, m, g, t = gpu_processor(s_r.to(device), q_r.to(device), t_r.to(device))
                with autocast():
                    pred = model(si, m, g)
                    val_l += criterion(pred, t).item()
        
        avg_val = val_l / len(val_loader)
        writer.add_scalar("Loss/Train", train_loss/len(train_loader), epoch)
        writer.add_scalar("Loss/Val", avg_val, epoch)
        torch.save(model.state_dict(), f"{save_dir}/model_ep{epoch+1}.pt")
        print(f"Val Loss: {avg_val:.5f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=640)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--data", type=str, default="auto")
    main(parser.parse_args())