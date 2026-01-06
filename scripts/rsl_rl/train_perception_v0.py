"""
Train Perception V1: Fixed 40x40, 5m Cap, Absolute Epochs & Replicate Padding
- Hardware: RTX 4090 + Ryzen 9950X
- Usage: python train_perception_v0.py --checkpoint "logs/.../model_ep10.pt" --epochs 50
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

# --- DIMENSIONS ---
IMG_H, IMG_W = 120, 212   
MAP_SIZE = 40  # [FIX] Strictly 40x40

# --- DATASET ---
class MeldogDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.index_map = []
        
        print(f"[INFO] Scanning dataset: {h5_path}...")
        with h5py.File(h5_path, 'r') as f:
            for ep_name in f.keys():
                if "gt_height" not in f[ep_name]: continue
                ep_len = f[ep_name]["gt_height"].shape[0]
                self.index_map.extend([(ep_name, i) for i in range(ep_len)])
        print(f"[INFO] Found {len(self.index_map)} samples.")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        # OPTIMIZATION: Open file once per worker and keep it open
        if not hasattr(self, 'h5_file'):
            self.h5_file = h5py.File(self.h5_path, 'r', swmr=True, libver='latest')

        ep_name, step_idx = self.index_map[idx]
        grp = self.h5_file[ep_name]
        
        d_front = grp["depth_front"][step_idx]
        d_rear  = grp["depth_rear"][step_idx]
        d_left  = grp["depth_left"][step_idx]
        d_right = grp["depth_right"][step_idx]
        target  = grp["gt_height"][step_idx]
        quat    = grp["robot_quat"][step_idx] 

        stack = np.stack([d_front, d_rear, d_left, d_right], axis=0)
        return stack, quat, target

# --- GPU PRE-PROCESSOR ---
class GPUProcessor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, stack_raw, quat_raw, target_raw):
        # Images: (B, 4, H, W) -> Resize -> Norm
        stack = stack_raw.float() * 0.001 
        stack = torch.clamp(stack, 0, 5.0) # Clamp max range
        stack = F.interpolate(stack, size=(IMG_H, IMG_W), mode='bilinear', align_corners=False)
        stack = stack * 0.2 # Norm 0..1

        # Target: (B, H, W) -> Resize
        target = target_raw.float() * 0.001
        target = target.unsqueeze(1) 
        if target.shape[-1] != MAP_SIZE:
            # [FIX] Use 'area' interpolation for maps to avoid aliasing/jagged edges
            target = F.interpolate(target, size=(MAP_SIZE, MAP_SIZE), mode='area')

        # Gravity: (B, 4) -> (B, 3)
        w, x, y, z = quat_raw[:, 0], quat_raw[:, 1], quat_raw[:, 2], quat_raw[:, 3]
        gx = -2 * (x*z + w*y)
        gy = -2 * (y*z - w*x)
        gz = -(1 - 2 * (x*x + y*y))
        grav_vec = torch.stack([gx, gy, gz], dim=1)

        return stack, grav_vec, target

# --- MODEL ---
class SimpleMapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = self.conv_block(4, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)
        self.mlp_gravity = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 64))
        self.dec1 = self.up_block(256, 128)
        self.dec2 = self.up_block(128, 64)
        self.dec3 = self.up_block(64, 32)
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)
        self.final_resize = nn.AdaptiveAvgPool2d((MAP_SIZE, MAP_SIZE))

    def conv_block(self, in_c, out_c):
        # [FIX] padding_mode='replicate' fixes the weird border artifacts!
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, padding_mode='replicate'), 
            nn.BatchNorm2d(out_c), 
            nn.ReLU(), 
            nn.MaxPool2d(2)
        )

    def up_block(self, in_c, out_c):
        # [FIX] padding_mode='replicate' here too
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_c, out_c, 3, padding=1, padding_mode='replicate'), 
            nn.BatchNorm2d(out_c), 
            nn.ReLU()
        )

    def forward(self, x, grav):
        x = self.enc4(self.enc3(self.enc2(self.enc1(x))))
        B, C, H, W = x.shape
        grav_embed = self.mlp_gravity(grav).unsqueeze(-1).unsqueeze(-1).expand(B, 64, H, W)
        x = torch.cat([x, grav_embed], dim=1)
        x = nn.Conv2d(256+64, 256, 1).to(x.device)(x)
        x = self.final_resize(self.final_conv(self.dec3(self.dec2(self.dec1(x)))))
        return x

# --- MAIN TRAINING LOOP ---
def main(args):
    # Logging Setup
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    prefix = "RESUME_" if args.checkpoint else ""
    run_name = f"PM_v0_{prefix}{timestamp}"
    save_dir = os.path.join("logs", "perception", run_name)
    
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=save_dir)
    print(f"[INFO] Log Directory: {save_dir}")

    # Dataset Setup
    data_path = args.data
    if data_path == "auto":
        if not os.path.exists("datasets"):
             print("[ERROR] No datasets folder found.")
             return
        import glob
        files = glob.glob("datasets/*/*.h5")
        if files:
            files.sort(key=os.path.getmtime)
            data_path = files[-1]
            print(f"[INFO] Auto-detected dataset: {data_path}")
        else:
            print("[ERROR] No .h5 files found in datasets/")
            return

    dataset = MeldogDataset(data_path)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.workers, pin_memory=True, persistent_workers=True)

    # Model Setup
    device = "cuda"
    model = SimpleMapper().to(device)
    
    # --- LOAD CHECKPOINT & PARSE EPOCH ---
    start_epoch = 0
    if args.checkpoint:
        if os.path.exists(args.checkpoint):
            print(f"[INFO] 🔄 Loading Checkpoint: {args.checkpoint}")
            state_dict = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(state_dict)
            print("[INFO] Weights loaded successfully.")
            
            # Extract epoch from filename
            match = re.search(r"model_ep(\d+).pt", args.checkpoint)
            if match:
                start_epoch = int(match.group(1))
                print(f"[INFO] Resuming from Epoch {start_epoch}")
        else:
            print(f"[WARNING] Checkpoint {args.checkpoint} not found! Starting fresh.")
    
    gpu_processor = GPUProcessor().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss() 
    
    try:
        scaler = torch.amp.GradScaler('cuda')
    except:
        scaler = GradScaler()

    # [FIX] Absolute Epoch Limit Logic
    total_target_epoch = args.epochs
    
    if start_epoch >= total_target_epoch:
        print(f"[ERROR] Target epoch {total_target_epoch} reached or exceeded (Current: {start_epoch}). Increase --epochs to continue training.")
        return

    print(f"[INFO] Training from Epoch {start_epoch} to {total_target_epoch} (Batch={args.batch_size}, Workers={args.workers})")

    for epoch in range(start_epoch, total_target_epoch):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_target_epoch}")
        for stack_raw, quat_raw, target_raw in pbar:
            stack_raw = stack_raw.to(device, non_blocking=True)
            quat_raw = quat_raw.to(device, non_blocking=True)
            target_raw = target_raw.to(device, non_blocking=True)

            with torch.no_grad():
                imgs, grav, target = gpu_processor(stack_raw, quat_raw, target_raw)

            optimizer.zero_grad(set_to_none=True)
            
            try:
                with torch.amp.autocast('cuda'):
                    pred = model(imgs, grav)
                    loss = criterion(pred, target)
            except:
                with autocast():
                    pred = model(imgs, grav)
                    loss = criterion(pred, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for stack_raw, quat_raw, target_raw in val_loader:
                stack_raw = stack_raw.to(device, non_blocking=True)
                quat_raw = quat_raw.to(device, non_blocking=True)
                target_raw = target_raw.to(device, non_blocking=True)
                
                imgs, grav, target = gpu_processor(stack_raw, quat_raw, target_raw)
                
                try:
                    with torch.amp.autocast('cuda'):
                        pred = model(imgs, grav)
                        val_loss += criterion(pred, target).item()
                except:
                    with autocast():
                        pred = model(imgs, grav)
                        val_loss += criterion(pred, target).item()
        
        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar("Loss/Val", avg_val_loss, epoch)
        writer.flush()
        
        print(f"Epoch {epoch+1}: Train {avg_train_loss:.5f} | Val {avg_val_loss:.5f}")
        
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"{save_dir}/model_ep{epoch+1}.pt")

    torch.save(model.state_dict(), f"{save_dir}/model_final.pt")
    writer.close()
    print("[INFO] Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Perception Model")
    
    # Arguments
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pt model to resume")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--workers", type=int, default=22, help="Number of dataloader workers")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Total target epochs (Absolute)")
    parser.add_argument("--data", type=str, default="auto", help="Path to h5 dataset or 'auto'")

    args = parser.parse_args()
    
    main(args)