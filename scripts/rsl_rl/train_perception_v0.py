"""
Train Perception V4.1: Augmentation + Architecture Options
Changes from V4:
- Added data augmentation (rotation, flip)
- Added cosine LR schedule
- Added deeper encoder option (3 levels instead of 2)
- Added optional self-attention for global context
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


# --- LOSS FUNCTION (same as V4) ---
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
        
        # Gradient Loss (edge preservation)
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
def augment_batch(sparse_map, mask, target, grav, p_rotate=0.5, p_flip=0.5):
    """
    Augmentations for terrain maps with gravity vector handling.
    Rotations require rotating the gravity vector's x,y components too.
    """
    B = sparse_map.shape[0]
    grav_out = grav.clone()
    
    for i in range(B):
        # Random 90° rotation
        if torch.rand(1).item() < p_rotate:
            k = torch.randint(1, 4, (1,)).item()
            sparse_map[i] = torch.rot90(sparse_map[i], k, dims=[-2, -1])
            mask[i] = torch.rot90(mask[i], k, dims=[-2, -1])
            target[i] = torch.rot90(target[i], k, dims=[-2, -1])
            
            # Rotate gravity x,y components
            gx, gy = grav_out[i, 0].clone(), grav_out[i, 1].clone()
            for _ in range(k):
                gx_new = -gy
                gy_new = gx
                gx, gy = gx_new, gy_new
            grav_out[i, 0], grav_out[i, 1] = gx, gy
        
        # Random flip
        if torch.rand(1).item() < p_flip:
            if torch.rand(1).item() < 0.5:
                sparse_map[i] = torch.flip(sparse_map[i], dims=[-1])  # Horizontal
                mask[i] = torch.flip(mask[i], dims=[-1])
                target[i] = torch.flip(target[i], dims=[-1])
                grav_out[i, 1] = -grav_out[i, 1]  # Flip y component
            else:
                sparse_map[i] = torch.flip(sparse_map[i], dims=[-2])  # Vertical
                mask[i] = torch.flip(mask[i], dims=[-2])
                target[i] = torch.flip(target[i], dims=[-2])
                grav_out[i, 0] = -grav_out[i, 0]  # Flip x component
    
    return sparse_map, mask, target, grav_out


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
        
        # Gravity vector from quaternion
        w, x, y, z = quat_raw[:, 0], quat_raw[:, 1], quat_raw[:, 2], quat_raw[:, 3]
        gx, gy, gz = -2*(x*z + w*y), -2*(y*z - w*x), -(1 - 2*(x*x + y*y))
        grav_vec = torch.stack([gx, gy, gz], dim=1)
        
        return sparse_map, occlusion_mask, grav_vec, target


# --- SELF-ATTENTION BLOCK ---
class SelfAttention2D(nn.Module):
    """Self-attention for capturing global terrain patterns"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // reduction, 1)
        self.key = nn.Conv2d(channels, channels // reduction, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).view(B, -1, H*W).permute(0, 2, 1)  # B, HW, C'
        k = self.key(x).view(B, -1, H*W)                      # B, C', HW
        v = self.value(x).view(B, -1, H*W)                    # B, C, HW
        
        attn = F.softmax(torch.bmm(q, k) / (C ** 0.5), dim=-1)  # B, HW, HW
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        
        return self.gamma * out + x


# --- MODEL: ORIGINAL 2-LEVEL ---
class SparseMapRefiner(nn.Module):
    def __init__(self, use_attention=False):
        super().__init__()
        self.use_attention = use_attention
        
        # Level 1: 40 -> 20
        self.enc1 = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1)
        )
        # Level 2: 20 -> 10
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1)
        )
        
        if use_attention:
            self.attn = SelfAttention2D(64)
        
        self.mlp_gravity = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 32))

        # Decoder with Transpose Convs
        self.dec1 = nn.ConvTranspose2d(64 + 32, 32, kernel_size=4, stride=2, padding=1) 
        self.dec2 = nn.ConvTranspose2d(32 + 32, 16, kernel_size=4, stride=2, padding=1)
        self.final_conv = nn.Conv2d(16 + 2, 1, kernel_size=3, padding=1)

    def forward(self, x, mask, grav):
        x_in = torch.cat([x, mask], dim=1)
        s1 = self.enc1(x_in)
        s2 = self.enc2(s1)
        
        if self.use_attention:
            s2 = self.attn(s2)
        
        B, _, H, W = s2.shape
        grav_embed = self.mlp_gravity(grav).view(B, 32, 1, 1).expand(B, 32, H, W)
        
        up1 = self.dec1(torch.cat([s2, grav_embed], dim=1))
        up2 = self.dec2(torch.cat([up1, s1], dim=1))
        
        return self.final_conv(torch.cat([up2, x_in], dim=1))


# --- MODEL: DEEPER 3-LEVEL ---
class SparseMapRefinerDeep(nn.Module):
    """Deeper encoder for more global context (40->20->10->5)"""
    def __init__(self, use_attention=False):
        super().__init__()
        self.use_attention = use_attention
        
        # Level 1: 40 -> 20
        self.enc1 = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU()
        )
        # Level 2: 20 -> 10
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU()
        )
        # Level 3: 10 -> 5
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.ReLU()
        )
        
        if use_attention:
            self.attn = SelfAttention2D(128)
        
        # Gravity embedding at bottleneck
        self.mlp_gravity = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 64)
        )
        
        # Decoder: 5 -> 10 -> 20 -> 40
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128 + 64, 64, 4, 2, 1), nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, 32, 4, 2, 1), nn.ReLU()
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(32 + 32, 16, 4, 2, 1), nn.ReLU()
        )
        
        self.final_conv = nn.Conv2d(16 + 2, 1, 3, padding=1)

    def forward(self, x, mask, grav):
        x_in = torch.cat([x, mask], dim=1)  # B, 2, 40, 40
        
        s1 = self.enc1(x_in)   # B, 32, 20, 20
        s2 = self.enc2(s1)     # B, 64, 10, 10
        s3 = self.enc3(s2)     # B, 128, 5, 5
        
        if self.use_attention:
            s3 = self.attn(s3)
        
        B, _, H, W = s3.shape
        grav_embed = self.mlp_gravity(grav).view(B, 64, 1, 1).expand(B, 64, H, W)
        
        up1 = self.dec1(torch.cat([s3, grav_embed], dim=1))  # B, 64, 10, 10
        up2 = self.dec2(torch.cat([up1, s2], dim=1))          # B, 32, 20, 20
        up3 = self.dec3(torch.cat([up2, s1], dim=1))          # B, 16, 40, 40
        
        return self.final_conv(torch.cat([up3, x_in], dim=1))


# --- MAIN TRAINING ---
def main(args):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Build run name from config
    model_tag = "Deep" if args.deep else "Base"
    attn_tag = "_Attn" if args.attention else ""
    aug_tag = "_Aug" if args.augment else ""
    
    run_name = f"PM_v4.1_{model_tag}{attn_tag}{aug_tag}_{timestamp}"
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

    dataset = MeldogDataset(fast_data_path)
    train_size = int(0.9 * len(dataset))
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size])
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.workers, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model selection
    if args.deep:
        model = SparseMapRefinerDeep(use_attention=args.attention).to(device)
        print(f"[INFO] Using DEEP model (3-level encoder)")
    else:
        model = SparseMapRefiner(use_attention=args.attention).to(device)
        print(f"[INFO] Using BASE model (2-level encoder)")
    
    if args.attention:
        print(f"[INFO] Self-attention ENABLED")
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Model parameters: {n_params:,}")
    
    gpu_processor = GPUProcessor(device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # Cosine annealing LR schedule
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    start_epoch = 0
    if args.load_model:
        if os.path.exists(args.load_model):
            print(f"[INFO] 🔄 Loading checkpoint: {args.load_model}")
            checkpoint = torch.load(args.load_model, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint.get('epoch', 0)
                print(f"[INFO] ✅ Resuming from Epoch {start_epoch}")
            else:
                model.load_state_dict(checkpoint)
                match = re.search(r'ep(\d+)', os.path.basename(args.load_model))
                start_epoch = int(match.group(1)) if match else 0

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
        for stack_raw, quat_raw, target_raw in pbar:
            stack_raw = stack_raw.to(device)
            quat_raw = quat_raw.to(device)
            target_raw = target_raw.to(device)
            
            sparse_in, mask, grav, target = gpu_processor(stack_raw, quat_raw, target_raw)
            
            # Apply rotation/flip augmentation
            if args.augment:
                sparse_in, mask, target, grav = augment_batch(sparse_in, mask, target, grav)
            
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
            train_metrics["grad"] += batch_metrics["loss_grad"]
            
            pbar.set_postfix({
                "L1_Obs": f"{batch_metrics['l1_obs_cm']:.2f}cm",
                "L1_Occ": f"{batch_metrics['l1_occ_cm']:.2f}cm"
            })
        
        # Step LR scheduler
        scheduler.step()
        
        # Validation (no augmentation)
        model.eval()
        val_metrics = {"loss": 0, "l1_obs": 0, "l1_occ": 0, "grad": 0}
        with torch.no_grad():
            for s_r, q_r, t_r in val_loader:
                si, m, g, t = gpu_processor(s_r.to(device), q_r.to(device), t_r.to(device))
                with autocast('cuda'):
                    pred = model(si, m, g)
                    v_loss, v_m = criterion(pred, t, m)
                    val_metrics["loss"] += v_loss.item()
                    val_metrics["l1_obs"] += v_m["l1_obs_cm"]
                    val_metrics["l1_occ"] += v_m["l1_occ_cm"]
                    val_metrics["grad"] += v_m["loss_grad"]

        n_train = len(train_loader)
        n_val = len(val_loader)
        
        # Log all metrics
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
        
        val_obs = val_metrics['l1_obs']/n_val
        val_occ = val_metrics['l1_occ']/n_val
        print(f"Epoch {epoch+1} | Val L1 Obs: {val_obs:.2f}cm | Val L1 Occ: {val_occ:.2f}cm | LR: {scheduler.get_last_lr()[0]:.2e}")

    writer.close()
    print(f"\n[DONE] Training complete. Logs saved to: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=640)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--data", type=str, default="auto")
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--compression", type=str, default="lzf", 
                       choices=["none", "lzf", "gzip"])
    
    # Loss weight arguments
    parser.add_argument("--w_sparse", type=float, default=20.0, 
                       help="Weight for observed area loss")
    parser.add_argument("--w_occluded", type=float, default=1.0, 
                       help="Weight for occluded area loss")
    parser.add_argument("--w_grad", type=float, default=10.0, 
                       help="Weight for gradient (edge) loss")
    
    # Architecture flags
    parser.add_argument("--deep", action="store_true", default=False,
                       help="Use deeper 3-level encoder instead of 2-level")
    parser.add_argument("--attention", action="store_true", default=False,
                       help="Add self-attention at bottleneck")
    
    # Augmentation flag
    parser.add_argument("--augment", action="store_true", default=True,
                       help="Enable rotation/flip augmentation")
    parser.add_argument("--no_augment", action="store_false", dest="augment",
                       help="Disable augmentation")
    
    main(parser.parse_args())