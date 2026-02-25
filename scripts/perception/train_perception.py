#!/usr/bin/env python3
# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Train perception model on collected dataset.

This is a pure PyTorch training script - NO Isaac Lab required.

Supports:
- V5 (heightmap_v5): ConvGRU temporal model
- V6 (heightmap_v6): Autoregressive model with coordinate transform

Usage:
    # Train V5 (ConvGRU) model
    python scripts/perception/train_perception.py \
        --model heightmap_v5 \
        --dataset datasets/PD_rough_2026-01-06_10-00-00/dataset.h5

    # Train V6 (autoregressive) model  
    python scripts/perception/train_perception.py \
        --model heightmap_v6 \
        --dataset datasets/PD_rough_2026-01-06_10-00-00/dataset.h5

Output:
    logs/perception/PM_{model}_{terrain}_{timestamp}/
    ├── model_best.pt
    ├── model_ep{N}.pt
    └── tensorboard events
"""

import argparse
import os
import sys
import glob
import math
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "source"))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from meldog_rl.utils import make_perception_log_dir
from meldog_rl.models.perception import (
    HeightmapConvGRU,
    HeightmapAutoregressive,
    HybridTerrainLoss,
    augment_sequence,
    augment_sequence_v6,
)
from meldog_rl.datasets import (
    SequentialDatasetV5,
    SequentialDatasetV6,
    GPUProcessorV5,
    GPUProcessorV6,
    repack_dataset,
    get_optimized_path,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

MAP_SIZE = 40
MAP_RES = 0.05


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def cosine_tf_schedule(epoch: int, total_epochs: int, tf_start: float = 0.5) -> float:
    """Cosine decay for teacher forcing ratio."""
    return tf_start * 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))


def extract_terrain_from_dataset(dataset_path: str) -> str:
    """Extract terrain name from dataset path.
    
    e.g., datasets/PD_rough_2026-01-06_10-00-00/dataset.h5 -> rough
    """
    path = Path(dataset_path)
    if path.is_file():
        path = path.parent
    name = path.name
    # Format: PD_{terrain}_{timestamp}
    parts = name.split("_")
    if len(parts) >= 2:
        return parts[1]
    return "unknown"


# =============================================================================
# V5 TRAINING (ConvGRU)
# =============================================================================

def train_v5(args):
    """Train V5 ConvGRU model."""
    device = args.device
    
    # Dataset
    train_dataset = SequentialDatasetV5(args.dataset, seq_len=args.seq_len, stride=args.stride)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.workers,
        pin_memory=True
    )
    
    # Split for validation (simple 90/10 split)
    val_dataset = SequentialDatasetV5(args.dataset, seq_len=args.seq_len, stride=args.stride * 4)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    
    # Model
    model = HeightmapConvGRU(
        gru_hidden=args.gru_hidden,
        gru_layers=args.gru_layers
    ).to(device)
    
    # GPU preprocessing
    gpu_processor = GPUProcessorV5(device=device)
    
    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = HybridTerrainLoss(
        w_sparse=args.w_sparse,
        w_occluded=args.w_occluded,
        w_grad=args.w_grad
    )
    scaler = GradScaler('cuda')
    
    # Logging
    terrain = extract_terrain_from_dataset(args.dataset)
    log_dir = make_perception_log_dir("v5", terrain)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    
    print(f"[INFO] Training V5 ConvGRU model")
    print(f"[INFO] Dataset: {args.dataset}")
    print(f"[INFO] Output: {log_dir}")
    print(f"[INFO] Sequences: {len(train_dataset)}")
    
    best_val_occ = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        train_metrics = {"loss": 0, "l1_obs": 0, "l1_occ": 0, "grad": 0}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for stack_seq, quat_seq, target_seq in pbar:
            # Reshape: (B, T, ...) -> (T, B, ...)
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
            
            sparse_seq_proc = torch.stack(sparse_list, dim=0)
            mask_seq_proc = torch.stack(mask_list, dim=0)
            grav_seq_proc = torch.stack(grav_list, dim=0)
            target_seq_proc = torch.stack(gt_list, dim=0)
            
            # Augmentation
            if args.augment:
                sparse_seq_proc, mask_seq_proc, target_seq_proc, grav_seq_proc = augment_sequence(
                    sparse_seq_proc, mask_seq_proc, target_seq_proc, grav_seq_proc
                )
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast('cuda'):
                pred_seq = model.forward_sequence(sparse_seq_proc, mask_seq_proc, grav_seq_proc)
                
                total_loss = 0
                batch_l1_obs = 0
                batch_l1_occ = 0
                batch_grad = 0
                
                for t in range(T):
                    loss, metrics = criterion(pred_seq[t], target_seq_proc[t], mask_seq_proc[t])
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
            
            pbar.set_postfix({"L1_Obs": f"{batch_l1_obs:.2f}cm", "L1_Occ": f"{batch_l1_occ:.2f}cm"})
        
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
                    sparse, mask, grav, gt = gpu_processor(stack_seq[t], quat_seq[t], target_seq[t])
                    sparse_list.append(sparse)
                    mask_list.append(mask)
                    grav_list.append(grav)
                    gt_list.append(gt)
                
                sparse_seq_proc = torch.stack(sparse_list, dim=0)
                mask_seq_proc = torch.stack(mask_list, dim=0)
                grav_seq_proc = torch.stack(grav_list, dim=0)
                target_seq_proc = torch.stack(gt_list, dim=0)
                
                with autocast('cuda'):
                    pred_seq = model.forward_sequence(sparse_seq_proc, mask_seq_proc, grav_seq_proc)
                    
                    total_loss = 0
                    batch_l1_obs = 0
                    batch_l1_occ = 0
                    batch_grad = 0
                    
                    for t in range(T):
                        loss, metrics = criterion(pred_seq[t], target_seq_proc[t], mask_seq_proc[t])
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
        
        # Log metrics
        n_train = len(train_loader)
        n_val = len(val_loader)
        
        writer.add_scalar("Loss/Train", train_metrics["loss"] / n_train, epoch)
        writer.add_scalar("Metrics/Train_L1_Obs_cm", train_metrics["l1_obs"] / n_train, epoch)
        writer.add_scalar("Metrics/Train_L1_Occ_cm", train_metrics["l1_occ"] / n_train, epoch)
        
        writer.add_scalar("Loss/Val", val_metrics["loss"] / n_val, epoch)
        writer.add_scalar("Metrics/Val_L1_Obs_cm", val_metrics["l1_obs"] / n_val, epoch)
        writer.add_scalar("Metrics/Val_L1_Occ_cm", val_metrics["l1_occ"] / n_val, epoch)
        
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)
        
        val_obs = val_metrics['l1_obs'] / n_val
        val_occ = val_metrics['l1_occ'] / n_val
        
        # Save checkpoint
        checkpoint_data = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'config': vars(args),
            'val_l1_obs': val_obs,
            'val_l1_occ': val_occ,
        }
        
        torch.save(checkpoint_data, log_dir / f"model_ep{epoch+1}.pt")
        
        if val_occ < best_val_occ:
            best_val_occ = val_occ
            torch.save(checkpoint_data, log_dir / "model_best.pt")
            print(f"  -> New best model! Val L1 Occ: {val_occ:.2f}cm")
        
        print(f"Epoch {epoch+1} | Val L1 Obs: {val_obs:.2f}cm | Val L1 Occ: {val_occ:.2f}cm | LR: {scheduler.get_last_lr()[0]:.2e}")
    
    writer.close()
    print(f"\n[DONE] Training complete. Best Val L1 Occ: {best_val_occ:.2f}cm")
    print(f"[DONE] Logs saved to: {log_dir}")


# =============================================================================
# V6 TRAINING (Autoregressive)
# =============================================================================

def train_v6(args):
    """Train V6 autoregressive model."""
    device = args.device
    
    # Dataset (V6 needs position)
    train_dataset = SequentialDatasetV6(args.dataset, seq_len=args.seq_len, stride=args.stride)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )
    
    val_dataset = SequentialDatasetV6(args.dataset, seq_len=args.seq_len, stride=args.stride * 4)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    
    # Model
    model = HeightmapAutoregressive().to(device)
    
    # GPU preprocessing
    gpu_processor = GPUProcessorV6(device=device)
    
    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = HybridTerrainLoss(
        w_sparse=args.w_sparse,
        w_occluded=args.w_occluded,
        w_grad=args.w_grad
    )
    scaler = GradScaler('cuda')
    
    # Logging
    terrain = extract_terrain_from_dataset(args.dataset)
    log_dir = make_perception_log_dir("v6", terrain)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    
    print(f"[INFO] Training V6 Autoregressive model")
    print(f"[INFO] Dataset: {args.dataset}")
    print(f"[INFO] Output: {log_dir}")
    print(f"[INFO] Sequences: {len(train_dataset)}")
    
    best_val_occ = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        train_metrics = {"loss": 0, "l1_obs": 0, "l1_occ": 0, "grad": 0}
        
        # Teacher forcing schedule
        tf_ratio = cosine_tf_schedule(epoch, args.epochs, args.teacher_forcing)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for stack_seq, quat_seq, pos_seq, target_seq in pbar:
            stack_seq = stack_seq.permute(1, 0, 2, 3, 4).to(device)
            quat_seq = quat_seq.permute(1, 0, 2).to(device)
            pos_seq = pos_seq.permute(1, 0, 2).to(device)
            target_seq = target_seq.permute(1, 0, 2, 3).to(device)
            
            T, B = stack_seq.shape[:2]
            
            # Process each timestep
            sparse_list, mask_list, grav_list, gt_list, yaw_list = [], [], [], [], []
            for t in range(T):
                sparse, mask, grav, gt, yaw = gpu_processor(stack_seq[t], quat_seq[t], target_seq[t])
                sparse_list.append(sparse)
                mask_list.append(mask)
                grav_list.append(grav)
                gt_list.append(gt)
                yaw_list.append(yaw)
            
            sparse_seq_proc = torch.stack(sparse_list, dim=0)
            mask_seq_proc = torch.stack(mask_list, dim=0)
            grav_seq_proc = torch.stack(grav_list, dim=0)
            target_seq_proc = torch.stack(gt_list, dim=0)
            yaw_seq_proc = torch.stack(yaw_list, dim=0)
            pos_seq_gpu = pos_seq.float()
            
            # Augmentation
            if args.augment:
                sparse_seq_proc, mask_seq_proc, target_seq_proc, grav_seq_proc, pos_seq_gpu, yaw_seq_proc = \
                    augment_sequence_v6(
                        sparse_seq_proc, mask_seq_proc, target_seq_proc, 
                        grav_seq_proc, pos_seq_gpu, yaw_seq_proc
                    )
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast('cuda'):
                pred_seq = model.forward_sequence(
                    sparse_seq_proc, mask_seq_proc, grav_seq_proc,
                    pos_seq_gpu, yaw_seq_proc,
                    teacher_forcing_ratio=tf_ratio,
                    target_seq=target_seq_proc
                )
                
                total_loss = 0
                batch_l1_obs = 0
                batch_l1_occ = 0
                batch_grad = 0
                
                for t in range(T):
                    loss, metrics = criterion(pred_seq[t], target_seq_proc[t], mask_seq_proc[t])
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
                    sparse, mask, grav, gt, yaw = gpu_processor(stack_seq[t], quat_seq[t], target_seq[t])
                    sparse_list.append(sparse)
                    mask_list.append(mask)
                    grav_list.append(grav)
                    gt_list.append(gt)
                    yaw_list.append(yaw)
                
                sparse_seq_proc = torch.stack(sparse_list, dim=0)
                mask_seq_proc = torch.stack(mask_list, dim=0)
                grav_seq_proc = torch.stack(grav_list, dim=0)
                target_seq_proc = torch.stack(gt_list, dim=0)
                yaw_seq_proc = torch.stack(yaw_list, dim=0)
                pos_seq_gpu = pos_seq.float()
                
                with autocast('cuda'):
                    pred_seq = model.forward_sequence(
                        sparse_seq_proc, mask_seq_proc, grav_seq_proc,
                        pos_seq_gpu, yaw_seq_proc,
                        teacher_forcing_ratio=0.0,  # No TF in validation
                        target_seq=None
                    )
                    
                    total_loss = 0
                    batch_l1_obs = 0
                    batch_l1_occ = 0
                    batch_grad = 0
                    
                    for t in range(T):
                        loss, metrics = criterion(pred_seq[t], target_seq_proc[t], mask_seq_proc[t])
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
        
        # Log metrics
        n_train = len(train_loader)
        n_val = len(val_loader)
        
        writer.add_scalar("Loss/Train", train_metrics["loss"] / n_train, epoch)
        writer.add_scalar("Metrics/Train_L1_Obs_cm", train_metrics["l1_obs"] / n_train, epoch)
        writer.add_scalar("Metrics/Train_L1_Occ_cm", train_metrics["l1_occ"] / n_train, epoch)
        
        writer.add_scalar("Loss/Val", val_metrics["loss"] / n_val, epoch)
        writer.add_scalar("Metrics/Val_L1_Obs_cm", val_metrics["l1_obs"] / n_val, epoch)
        writer.add_scalar("Metrics/Val_L1_Occ_cm", val_metrics["l1_occ"] / n_val, epoch)
        
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)
        writer.add_scalar("TeacherForcing", tf_ratio, epoch)
        
        val_obs = val_metrics['l1_obs'] / n_val
        val_occ = val_metrics['l1_occ'] / n_val
        
        # Save checkpoint
        checkpoint_data = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'config': vars(args),
            'val_l1_obs': val_obs,
            'val_l1_occ': val_occ,
        }
        
        torch.save(checkpoint_data, log_dir / f"model_ep{epoch+1}.pt")
        
        if val_occ < best_val_occ:
            best_val_occ = val_occ
            torch.save(checkpoint_data, log_dir / "model_best.pt")
            print(f"  -> New best model! Val L1 Occ: {val_occ:.2f}cm")
        
        print(f"Epoch {epoch+1} | Val L1 Obs: {val_obs:.2f}cm | Val L1 Occ: {val_occ:.2f}cm | TF: {tf_ratio:.2f}")
    
    writer.close()
    print(f"\n[DONE] Training complete. Best Val L1 Occ: {best_val_occ:.2f}cm")
    print(f"[DONE] Logs saved to: {log_dir}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train perception model.")
    parser.add_argument("--model", type=str, required=True, 
                        choices=["heightmap_v5", "heightmap_v6"],
                        help="Model architecture to train.")
    parser.add_argument("--dataset", type=str, default="auto",
                        help="Path to dataset H5 file or 'auto' for latest.")
    
    # Training
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader workers.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device.")
    
    # Sequence
    parser.add_argument("--seq_len", type=int, default=16, help="Sequence length.")
    parser.add_argument("--stride", type=int, default=8, help="Sequence stride.")
    
    # V5 specific
    parser.add_argument("--gru_hidden", type=int, default=128, help="ConvGRU hidden channels.")
    parser.add_argument("--gru_layers", type=int, default=2, help="ConvGRU layers.")
    
    # V6 specific
    parser.add_argument("--teacher_forcing", type=float, default=0.5, help="Starting TF ratio.")
    
    # Loss weights
    parser.add_argument("--w_sparse", type=float, default=20.0)
    parser.add_argument("--w_occluded", type=float, default=1.0)
    parser.add_argument("--w_grad", type=float, default=10.0)
    
    # Augmentation
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no_augment", action="store_false", dest="augment")
    
    args = parser.parse_args()
    
    # Auto-select dataset
    if args.dataset == "auto":
        files = glob.glob("datasets/*/dataset.h5")
        if not files:
            print("[ERROR] No datasets found! Run collect_dataset.py first.")
            return
        args.dataset = sorted(files, key=os.path.getmtime)[-1]
        print(f"[INFO] Auto-selected dataset: {args.dataset}")
    
    # Train
    if args.model == "heightmap_v5":
        train_v5(args)
    elif args.model == "heightmap_v6":
        args.lr = 5e-5  # Lower LR for V6
        train_v6(args)
    else:
        raise ValueError(f"Unknown model: {args.model}")


if __name__ == "__main__":
    main()
