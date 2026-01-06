"""
Train Perception V2: Sparse 2.5D Map Pre-processing
- Hardware: RTX 4090 + Ryzen 9950X
- Pipeline: Raw Depth -> DepthProjector (GPU) -> Sparse Map -> Refinement Net -> Dense Height Map
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

# Import the projector logic
# Ensure preprocess_perception.py is in the same directory
from preprocess_perception import DepthProjector 

# --- CONFIGURATION ---
MAP_SIZE = 40  # 40x40 Grid
MAP_RES = 0.05 # 5cm Resolution

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
        
        # Load Raw Depth (mm uint16) -> Convert to Meters later
        d_front = grp["depth_front"][step_idx]
        d_rear  = grp["depth_rear"][step_idx]
        d_left  = grp["depth_left"][step_idx]
        d_right = grp["depth_right"][step_idx]
        
        # Load GT Height (mm int16)
        target  = grp["gt_height"][step_idx]
        
        # Load Robot Pose (Optional, but good for gravity embedding context)
        quat = grp["robot_quat"][step_idx] 

        stack = np.stack([d_front, d_rear, d_left, d_right], axis=0)
        return stack, quat, target

# --- GPU PRE-PROCESSOR ---
class GPUProcessor(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        # Initialize the Geometric Projector
        self.projector = DepthProjector(map_size=MAP_SIZE, map_res=MAP_RES, device=device)

    def forward(self, stack_raw, quat_raw, target_raw):
        """
        Args:
            stack_raw: (B, 4, H, W) uint16 in mm
            quat_raw: (B, 4)
            target_raw: (B, 40, 40) int16 in mm
        Returns:
            sparse_map: (B, 1, 40, 40) normalized/meters
            grav_vec: (B, 3)
            target: (B, 1, 40, 40) meters
        """
        # 1. Prepare Depth Stack (mm -> meters)
        # (B, 4, H, W)
        depth_stack = stack_raw.float() * 0.001 
        
        # 2. Run Geometric Projection
        # Returns (B, 1, 40, 40) with values in meters (approx -1.0 to 1.0 relative to robot)
        # We wrap in no_grad because projection is a fixed geometric operation, not learned.
        with torch.no_grad():
            sparse_map = self.projector(depth_stack)

        # 3. Prepare Target
        target = target_raw.float() * 0.001
        target = target.unsqueeze(1) # (B, 1, 40, 40)

        # 4. Prepare Gravity Vector
        w, x, y, z = quat_raw[:, 0], quat_raw[:, 1], quat_raw[:, 2], quat_raw[:, 3]
        gx = -2 * (x*z + w*y)
        gy = -2 * (y*z - w*x)
        gz = -(1 - 2 * (x*x + y*y))
        grav_vec = torch.stack([gx, gy, gz], dim=1)

        return sparse_map, grav_vec, target

# --- MODEL: SPARSE MAP REFINER ---
# Modified to accept 1-channel Sparse Map input
class SparseMapRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder: Input 1 Channel (Sparse Map)
        self.enc1 = self.conv_block(1, 32)   # 40 -> 20
        self.enc2 = self.conv_block(32, 64)  # 20 -> 10
        self.enc3 = self.conv_block(64, 128) # 10 -> 5 (Bottleneck)
        
        # Gravity Injection Network
        self.mlp_gravity = nn.Sequential(
            nn.Linear(3, 32), 
            nn.ReLU(), 
            nn.Linear(32, 32)
        )

        # Decoder (Refinement)
        self.dec1 = self.up_block(128 + 32, 64) # +32 for gravity | 5 -> 10
        self.dec2 = self.up_block(64, 32)       # 10 -> 20
        
        # Final reconstruction
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) # 20 -> 40
        self.final_conv = nn.Conv2d(32, 1, kernel_size=3, padding=1) # 40 -> 40

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, padding_mode='replicate'), 
            nn.BatchNorm2d(out_c), 
            nn.ReLU(), 
            nn.MaxPool2d(2) # Downsample
        )

    def up_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), # Upsample
            nn.Conv2d(in_c, out_c, 3, padding=1, padding_mode='replicate'), 
            nn.BatchNorm2d(out_c), 
            nn.ReLU()
        )

    def forward(self, x, grav):
        # x: (B, 1, 40, 40)
        e1 = self.enc1(x)  # -> (32, 20, 20)
        e2 = self.enc2(e1) # -> (64, 10, 10)
        e3 = self.enc3(e2) # -> (128, 5, 5)

        # Process Gravity
        B, _, H, W = e3.shape
        grav_embed = self.mlp_gravity(grav).unsqueeze(-1).unsqueeze(-1).expand(B, 32, H, W)
        
        # Concatenate Gravity at bottleneck
        bottleneck = torch.cat([e3, grav_embed], dim=1) # (160, 5, 5)

        # Decode
        d1 = self.dec1(bottleneck) # -> (64, 10, 10)
        d2 = self.dec2(d1)         # -> (32, 20, 20)
        
        # Final Output
        out = self.final_upsample(d2) # -> (32, 40, 40)
        out = self.final_conv(out)    # -> (1, 40, 40)
        
        return out

# --- MAIN TRAINING LOOP ---
def main(args):
    # Logging Setup
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    prefix = "RESUME_" if args.checkpoint else ""
    run_name = f"PM_v2_Sparse_{prefix}{timestamp}"
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
    
    # Batch size can likely be slightly lower than before because projection adds some VRAM overhead, 
    # but since inputs to the ConvNet are tiny (40x40), it balances out.
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.workers, pin_memory=True, persistent_workers=True)

    # Model Setup
    device = "cuda"
    model = SparseMapRefiner().to(device)
    
    # --- LOAD CHECKPOINT ---
    start_epoch = 0
    if args.checkpoint:
        if os.path.exists(args.checkpoint):
            print(f"[INFO] 🔄 Loading Checkpoint: {args.checkpoint}")
            state_dict = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(state_dict)
            
            match = re.search(r"model_ep(\d+).pt", args.checkpoint)
            if match:
                start_epoch = int(match.group(1))
                print(f"[INFO] Resuming from Epoch {start_epoch}")
        else:
            print(f"[WARNING] Checkpoint {args.checkpoint} not found! Starting fresh.")
    
    # Processor on GPU (Initializes Projector)
    gpu_processor = GPUProcessor(device=device).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss() 
    
    try:
        scaler = torch.amp.GradScaler('cuda')
    except:
        scaler = GradScaler()

    total_target_epoch = args.epochs
    
    if start_epoch >= total_target_epoch:
        print(f"[ERROR] Target epoch {total_target_epoch} reached. Increase --epochs.")
        return

    print(f"[INFO] Training from Epoch {start_epoch} to {total_target_epoch}")

    for epoch in range(start_epoch, total_target_epoch):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_target_epoch}")
        for stack_raw, quat_raw, target_raw in pbar:
            stack_raw = stack_raw.to(device, non_blocking=True)
            quat_raw = quat_raw.to(device, non_blocking=True)
            target_raw = target_raw.to(device, non_blocking=True)

            with torch.no_grad():
                # 1. Project Raw Depth -> Sparse Map
                sparse_input, grav, target = gpu_processor(stack_raw, quat_raw, target_raw)

            optimizer.zero_grad(set_to_none=True)
            
            try:
                with torch.amp.autocast('cuda'):
                    # 2. Refine Sparse Map -> Dense Map
                    pred = model(sparse_input, grav)
                    loss = criterion(pred, target)
            except:
                with autocast():
                    pred = model(sparse_input, grav)
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
                
                sparse_input, grav, target = gpu_processor(stack_raw, quat_raw, target_raw)
                
                try:
                    with torch.amp.autocast('cuda'):
                        pred = model(sparse_input, grav)
                        val_loss += criterion(pred, target).item()
                except:
                    with autocast():
                        pred = model(sparse_input, grav)
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
    parser = argparse.ArgumentParser(description="Train Perception Model V2 (Sparse Projection)")
    
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pt model to resume")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--workers", type=int, default=24, help="Number of dataloader workers")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Total target epochs")
    parser.add_argument("--data", type=str, default="auto", help="Path to h5 dataset or 'auto'")

    args = parser.parse_args()
    
    main(args)