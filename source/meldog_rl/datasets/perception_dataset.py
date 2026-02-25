# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Dataset classes for perception model training.

Provides:
- SequentialDatasetV5: For ConvGRU model (sequences without position)
- SequentialDatasetV6: For autoregressive model (sequences with position/yaw)
- GPUProcessor: On-GPU preprocessing pipeline
"""

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from pathlib import Path

from ..models.perception import DepthProjector


# =============================================================================
# CONSTANTS
# =============================================================================

MAP_SIZE = 40
MAP_RES = 0.05


# =============================================================================
# SEQUENTIAL DATASETS
# =============================================================================

class SequentialDatasetV5(Dataset):
    """Dataset for V5 ConvGRU model - returns sequences without position.
    
    Each item is (stack_seq, quat_seq, gt_seq) where:
    - stack_seq: (seq_len, 4, H, W) depth images
    - quat_seq: (seq_len, 4) robot quaternions
    - gt_seq: (seq_len, 40, 40) ground truth height maps
    
    Args:
        h5_path: Path to HDF5 dataset
        seq_len: Number of consecutive frames per sequence
        stride: Step between sequence starts (stride < seq_len = overlapping)
    """
    
    def __init__(self, h5_path: str, seq_len: int = 16, stride: int = 8):
        self.h5_path = h5_path
        self.seq_len = seq_len
        self.stride = stride
        self.sequences = []  # List of (ep_name, start_idx)
        self._h5_file = None
        
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
    
    def _get_h5_file(self):
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, 'r', libver='latest', swmr=False)
        return self._h5_file

    def __getitem__(self, idx):
        h5_file = self._get_h5_file()
        
        ep_name, start = self.sequences[idx]
        end = start + self.seq_len
        grp = h5_file[ep_name]
        
        # Load sequence of frames
        stack_seq = np.stack([
            grp["depth_front"][start:end],
            grp["depth_rear"][start:end], 
            grp["depth_left"][start:end],
            grp["depth_right"][start:end]
        ], axis=1)  # (seq_len, 4, H, W)
        
        quat_seq = grp["robot_quat"][start:end]  # (seq_len, 4)
        gt_seq = grp["gt_height"][start:end]     # (seq_len, 40, 40)
        
        return stack_seq, quat_seq, gt_seq
    
    def __del__(self):
        if self._h5_file is not None:
            try:
                self._h5_file.close()
            except:
                pass


class SequentialDatasetV6(Dataset):
    """Dataset for V6 autoregressive model - includes position and yaw.
    
    Each item is (stack_seq, quat_seq, pos_seq, gt_seq) where:
    - stack_seq: (seq_len, 4, H, W) depth images
    - quat_seq: (seq_len, 4) robot quaternions
    - pos_seq: (seq_len, 3) robot positions
    - gt_seq: (seq_len, 40, 40) ground truth height maps
    
    Args:
        h5_path: Path to HDF5 dataset
        seq_len: Number of consecutive frames per sequence
        stride: Step between sequence starts
    """
    
    def __init__(self, h5_path: str, seq_len: int = 16, stride: int = 8):
        self.h5_path = h5_path
        self.seq_len = seq_len
        self.stride = stride
        self.sequences = []
        self._h5_file = None
        
        with h5py.File(h5_path, 'r') as f:
            for ep_name in f.keys():
                if "gt_height" not in f[ep_name]: 
                    continue
                ep_len = f[ep_name]["gt_height"].shape[0]
                
                for start in range(0, ep_len - seq_len + 1, stride):
                    self.sequences.append((ep_name, start))
        
        print(f"[INFO] Created {len(self.sequences)} sequences of length {seq_len}")

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
            grp["depth_front"][start:end],
            grp["depth_rear"][start:end], 
            grp["depth_left"][start:end],
            grp["depth_right"][start:end]
        ], axis=1)
        
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


# =============================================================================
# GPU PREPROCESSOR
# =============================================================================

class GPUProcessorV5(nn.Module):
    """GPU-based preprocessing for V5 model.
    
    Converts raw data to model inputs:
    - Depth images -> sparse height map + occlusion mask
    - Quaternion -> gravity vector
    - GT height (scaling)
    """
    
    def __init__(self, device: str = 'cuda'):
        super().__init__()
        self.device = device
        self.projector = DepthProjector(map_size=MAP_SIZE, map_res=MAP_RES, device=device)

    def forward(self, stack_raw, quat_raw, target_raw):
        """Process a single frame.
        
        Args:
            stack_raw: (B, 4, H, W) depth images in uint16 mm
            quat_raw: (B, 4) quaternions
            target_raw: (B, 40, 40) GT height in int16 mm
            
        Returns:
            sparse_map: (B, 1, 40, 40) sparse height map
            occlusion_mask: (B, 1, 40, 40) occlusion mask
            grav_vec: (B, 3) gravity vector
            target: (B, 1, 40, 40) GT height map
        """
        # Convert depth from mm to meters
        depth_stack = stack_raw.float() * 0.001 
        
        # Project depth to sparse map
        with torch.no_grad():
            sparse_map, occlusion_mask = self.projector(depth_stack, quat_raw)
        
        # Convert GT from mm to meters
        target = target_raw.float() * 0.001
        target = target.unsqueeze(1)  # (B, 1, 40, 40)
        
        # Compute gravity vector from quaternion
        w, x, y, z = quat_raw[:, 0], quat_raw[:, 1], quat_raw[:, 2], quat_raw[:, 3]
        gx = -2 * (x * z + w * y)
        gy = -2 * (y * z - w * x)
        gz = -(1 - 2 * (x * x + y * y))
        grav_vec = torch.stack([gx, gy, gz], dim=1)
        
        return sparse_map, occlusion_mask, grav_vec, target


class GPUProcessorV6(nn.Module):
    """GPU-based preprocessing for V6 model.
    
    Same as V5 but also extracts yaw from quaternion.
    """
    
    def __init__(self, device: str = 'cuda'):
        super().__init__()
        self.device = device
        self.projector = DepthProjector(map_size=MAP_SIZE, map_res=MAP_RES, device=device)

    def forward(self, stack_raw, quat_raw, target_raw):
        """Process a single frame.
        
        Returns:
            sparse_map, occlusion_mask, grav_vec, target, yaw
        """
        depth_stack = stack_raw.float() * 0.001 
        
        with torch.no_grad():
            sparse_map, occlusion_mask = self.projector(depth_stack, quat_raw)
        
        target = target_raw.float() * 0.001
        target = target.unsqueeze(1)
        
        w, x, y, z = quat_raw[:, 0], quat_raw[:, 1], quat_raw[:, 2], quat_raw[:, 3]
        gx = -2 * (x * z + w * y)
        gy = -2 * (y * z - w * x)
        gz = -(1 - 2 * (x * x + y * y))
        grav_vec = torch.stack([gx, gy, gz], dim=1)
        
        # Extract yaw
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        
        return sparse_map, occlusion_mask, grav_vec, target, yaw


# =============================================================================
# DATASET UTILITIES
# =============================================================================

def repack_dataset(src_path: str, dst_path: str, compression: str = 'lzf'):
    """Repack dataset with different compression for faster loading.
    
    Args:
        src_path: Source HDF5 file
        dst_path: Destination HDF5 file
        compression: 'none', 'lzf', or 'gzip'
    """
    from tqdm import tqdm
    
    if compression == 'none':
        comp_param = None
        comp_name = "uncompressed"
    elif compression == 'lzf':
        comp_param = 'lzf'
        comp_name = "LZF"
    else:
        comp_param = compression
        comp_name = compression.upper()
    
    print(f"\n[INFO] Optimizing dataset for training ({comp_name})...")
    
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
    
    print(f"[INFO] Dataset optimization complete\n")


def get_optimized_path(original_path: str, compression: str) -> str:
    """Get path for optimized dataset.
    
    Args:
        original_path: Original dataset path
        compression: Compression type
        
    Returns:
        Path to optimized dataset
    """
    base = original_path.replace(".h5", "")
    if compression == 'none':
        return f"{base}_opt_uncompressed.h5"
    elif compression == 'lzf':
        return f"{base}_opt_lzf.h5"
    else:
        return f"{base}_opt_{compression}.h5"
