# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Common components for perception models.

Includes:
- ConvGRU cells for temporal processing
- Loss functions for terrain reconstruction
- Data augmentation utilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# =============================================================================
# CONVOLUTIONAL GRU
# =============================================================================

class ConvGRUCell(nn.Module):
    """Convolutional GRU cell - maintains spatial structure unlike regular GRU.
    
    Based on: "Convolutional LSTM Network" (Shi et al., 2015)
    """
    
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
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
    
    def forward(self, x: torch.Tensor, h_prev: torch.Tensor = None) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: (B, C_in, H, W) current input
            h_prev: (B, C_hidden, H, W) previous hidden state
            
        Returns:
            h_new: (B, C_hidden, H, W) new hidden state
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
    """Multi-layer ConvGRU (stacked cells)."""
    
    def __init__(
        self, 
        input_channels: int, 
        hidden_channels: int, 
        num_layers: int = 2, 
        kernel_size: int = 3
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_ch = input_channels if i == 0 else hidden_channels
            self.cells.append(ConvGRUCell(in_ch, hidden_channels, kernel_size))
    
    def forward(
        self, 
        x: torch.Tensor, 
        h_prev: list[torch.Tensor] = None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass.
        
        Args:
            x: (B, C_in, H, W) input
            h_prev: list of (B, C_hidden, H, W) for each layer, or None
            
        Returns:
            output: (B, C_hidden, H, W) output from last layer
            h_new: list of hidden states for each layer
        """
        if h_prev is None:
            h_prev = [None] * self.num_layers
        
        h_new = []
        current = x
        for i, cell in enumerate(self.cells):
            current = cell(current, h_prev[i])
            h_new.append(current)
        
        return current, h_new


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class HybridTerrainLoss(nn.Module):
    """Hybrid loss for terrain reconstruction.
    
    Combines:
    - Sparse loss: L1 on observed (non-occluded) regions
    - Occluded loss: L1 on occluded regions (hallucination)
    - Gradient loss: L1 on spatial gradients for smoothness
    
    Args:
        w_sparse: Weight for observed region loss
        w_occluded: Weight for occluded region loss
        w_grad: Weight for gradient loss
    """
    
    def __init__(
        self, 
        w_sparse: float = 20.0, 
        w_occluded: float = 1.0, 
        w_grad: float = 10.0
    ):
        super().__init__()
        self.w_sparse = w_sparse
        self.w_occluded = w_occluded
        self.w_grad = w_grad

    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor, 
        mask: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """Compute loss.
        
        Args:
            pred: (B, 1, H, W) predicted height map
            target: (B, 1, H, W) ground truth height map
            mask: (B, 1, H, W) occlusion mask (1=occluded, 0=observed)
            
        Returns:
            total_loss: Scalar loss
            metrics: Dict with individual loss components
        """
        observed_mask = 1.0 - mask
        l1_error = torch.abs(pred - target)
        
        n_obs = observed_mask.sum() + 1e-6
        n_occ = mask.sum() + 1e-6
        
        loss_sparse = (l1_error * observed_mask).sum() / n_obs
        loss_occluded = (l1_error * mask).sum() / n_occ
        
        # Gradient loss for smoothness
        def get_grads(img):
            dx = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
            dy = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
            return dx, dy
        
        pred_dx, pred_dy = get_grads(pred)
        gt_dx, gt_dy = get_grads(target)
        loss_grad = F.l1_loss(pred_dx, gt_dx) + F.l1_loss(pred_dy, gt_dy)

        total_loss = (
            self.w_sparse * loss_sparse + 
            self.w_occluded * loss_occluded + 
            self.w_grad * loss_grad
        )

        metrics = {
            "loss_sparse": loss_sparse.item(),
            "loss_occluded": loss_occluded.item(),
            "loss_grad": loss_grad.item(),
            "l1_obs_cm": loss_sparse.item() * 100.0,  # Convert to cm
            "l1_occ_cm": loss_occluded.item() * 100.0
        }
        return total_loss, metrics


# =============================================================================
# DATA AUGMENTATION
# =============================================================================

def augment_sequence(
    sparse_seq: torch.Tensor, 
    mask_seq: torch.Tensor, 
    target_seq: torch.Tensor, 
    grav_seq: torch.Tensor,
    p_rotate: float = 0.5, 
    p_flip: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply consistent augmentation to entire sequence.
    
    Applies the same random rotation/flip to all timesteps for consistency.
    
    Args:
        sparse_seq: (T, B, 1, H, W) sparse height maps
        mask_seq: (T, B, 1, H, W) occlusion masks
        target_seq: (T, B, 1, H, W) ground truth
        grav_seq: (T, B, 3) gravity vectors
        p_rotate: Probability of rotation
        p_flip: Probability of flip
        
    Returns:
        Augmented tensors with same shapes
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


def augment_sequence_v6(
    sparse_seq: torch.Tensor,
    mask_seq: torch.Tensor,
    target_seq: torch.Tensor,
    grav_seq: torch.Tensor,
    pos_seq: torch.Tensor,
    yaw_seq: torch.Tensor,
    p_rotate: float = 0.5,
    p_flip: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Augmentation for V6 model (includes position and yaw)."""
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
