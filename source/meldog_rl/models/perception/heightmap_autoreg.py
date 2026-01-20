# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Heightmap prediction with autoregressive feedback (V6 architecture).

Key features:
- Uses previous output transformed to current frame
- Per-pixel learned gate for prev_output contribution
- Valid mask tracked separately (not height != 0)
- Coordinate transform handles robot motion between frames
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .projector import euler_from_quat


# =============================================================================
# COORDINATE TRANSFORM
# =============================================================================

MAP_SIZE = 40
MAP_RES = 0.05


def transform_height_map_with_mask(
    prev_map: torch.Tensor,
    prev_valid: torch.Tensor,
    prev_pos: torch.Tensor,
    prev_yaw: torch.Tensor,
    curr_pos: torch.Tensor,
    curr_yaw: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transform previous height map AND valid mask from prev robot frame to current.
    
    Args:
        prev_map: (B, 1, H, W) height map in previous robot frame
        prev_valid: (B, 1, H, W) validity mask (1=valid data, 0=no data)
        prev_pos: (B, 3) previous robot position (world frame)
        prev_yaw: (B,) previous robot yaw
        curr_pos: (B, 3) current robot position (world frame)
        curr_yaw: (B,) current robot yaw
        
    Returns:
        transformed_map: (B, 1, H, W) height map in current robot frame
        transformed_valid: (B, 1, H, W) validity mask in current robot frame
    """
    B, _, H, W = prev_map.shape
    device = prev_map.device
    
    # Relative transform: how did robot move from prev to curr?
    delta_pos_world = curr_pos[:, :2] - prev_pos[:, :2]
    delta_yaw = curr_yaw - prev_yaw
    
    # Transform delta_pos to previous robot frame
    cos_prev = torch.cos(-prev_yaw)
    sin_prev = torch.sin(-prev_yaw)
    delta_x_robot = delta_pos_world[:, 0] * cos_prev - delta_pos_world[:, 1] * sin_prev
    delta_y_robot = delta_pos_world[:, 0] * sin_prev + delta_pos_world[:, 1] * cos_prev
    
    # Create sampling grid
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )
    grid = torch.stack([grid_x, grid_y], dim=-1)
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1)
    
    # Convert to meters
    half_size = MAP_SIZE * MAP_RES / 2.0
    grid_meters = grid * half_size
    
    # Rotate by -delta_yaw
    cos_delta = torch.cos(-delta_yaw).view(B, 1, 1, 1)
    sin_delta = torch.sin(-delta_yaw).view(B, 1, 1, 1)
    
    grid_x_rot = grid_meters[..., 0] * cos_delta.squeeze(-1) - grid_meters[..., 1] * sin_delta.squeeze(-1)
    grid_y_rot = grid_meters[..., 0] * sin_delta.squeeze(-1) + grid_meters[..., 1] * cos_delta.squeeze(-1)
    
    # Translate by -delta_pos
    grid_x_trans = grid_x_rot - delta_x_robot.view(B, 1, 1)
    grid_y_trans = grid_y_rot - delta_y_robot.view(B, 1, 1)
    
    # Back to normalized coords
    grid_transformed = torch.stack([
        grid_x_trans / half_size,
        grid_y_trans / half_size
    ], dim=-1)
    
    # Sample height map with border padding (terrain continues at edges)
    transformed_map = F.grid_sample(
        prev_map, 
        grid_transformed,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )
    
    # Sample valid mask with zeros padding (out of bounds = invalid)
    transformed_valid = F.grid_sample(
        prev_valid,
        grid_transformed,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )
    # Threshold to binary
    transformed_valid = (transformed_valid > 0.5).float()
    
    return transformed_map, transformed_valid


# =============================================================================
# V6 MODEL
# =============================================================================

class HeightmapAutoregressive(nn.Module):
    """Deep U-Net with auto-regressive feedback and learned per-pixel gating.
    
    Key improvements over V5:
    - Uses previous output (transformed to current frame) as additional input
    - Per-pixel gate decides how much to trust previous prediction
    - Valid mask tracked properly (not height != 0)
    - No hidden state - uses explicit coordinate transform instead
    
    Input:
        - Sparse height map (projected from depth cameras)
        - Occlusion mask
        - Previous output (transformed to current frame)
        - Previous valid mask
        - Gravity vector
        
    Output:
        - Refined height map (40x40)
    """
    
    def __init__(self):
        super().__init__()
        
        # Input: 2 (sparse + mask) + 1 (prev output) + 1 (prev valid) = 4 channels
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
        
        # Per-pixel gate: learns where to trust previous prediction
        self.gate_net = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 1, 1), nn.Sigmoid()
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

    def forward(
        self, 
        x: torch.Tensor, 
        mask: torch.Tensor, 
        prev_output: torch.Tensor, 
        prev_valid: torch.Tensor, 
        grav: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: (B, 1, 40, 40) sparse map
            mask: (B, 1, 40, 40) occlusion mask  
            prev_output: (B, 1, 40, 40) previous output transformed to current frame
            prev_valid: (B, 1, 40, 40) mask of where prev_output is valid
            grav: (B, 3) gravity vector
            
        Returns:
            pred: (B, 1, 40, 40) predicted height map
        """
        # Compute per-pixel gate
        gate_input = torch.cat([prev_output, prev_valid, x, mask], dim=1)
        gate = self.gate_net(gate_input)  # (B, 1, 40, 40) in [0, 1]
        
        # Apply gate: only use prev_output where gate is high AND prev_valid
        gated_prev = prev_output * gate * prev_valid
        
        # Concatenate all inputs
        x_in = torch.cat([x, mask, gated_prev, prev_valid], dim=1)
        
        # Encode
        s1 = self.enc1(x_in)   # B, 32, 20, 20
        s2 = self.enc2(s1)     # B, 64, 10, 10
        s3 = self.enc3(s2)     # B, 128, 5, 5
        
        # Add gravity embedding
        B, _, H, W = s3.shape
        grav_embed = self.mlp_gravity(grav).view(B, 64, 1, 1).expand(B, 64, H, W)
        
        # Decode
        up1 = self.dec1(torch.cat([s3, grav_embed], dim=1))  # B, 64, 10, 10
        up2 = self.dec2(torch.cat([up1, s2], dim=1))         # B, 32, 20, 20
        up3 = self.dec3(torch.cat([up2, s1], dim=1))         # B, 16, 40, 40
        
        pred = self.final_conv(torch.cat([up3, x_in], dim=1))
        pred = torch.clamp(pred, min=-5.0, max=5.0)
        
        return pred
    
    def forward_sequence(
        self,
        sparse_seq: torch.Tensor,
        mask_seq: torch.Tensor,
        grav_seq: torch.Tensor,
        pos_seq: torch.Tensor,
        yaw_seq: torch.Tensor,
        teacher_forcing_ratio: float = 0.0,
        target_seq: torch.Tensor = None
    ) -> torch.Tensor:
        """Process entire sequence with autoregressive feedback.
        
        Args:
            sparse_seq: (T, B, 1, H, W) sparse maps
            mask_seq: (T, B, 1, H, W) occlusion masks
            grav_seq: (T, B, 3) gravity vectors
            pos_seq: (T, B, 3) robot positions
            yaw_seq: (T, B) robot yaws
            teacher_forcing_ratio: Probability of using GT instead of prediction
            target_seq: (T, B, 1, H, W) ground truth (for teacher forcing)
            
        Returns:
            pred_seq: (T, B, 1, H, W) predictions
        """
        T, B = sparse_seq.shape[:2]
        device = sparse_seq.device
        
        preds = []
        prev_output = torch.zeros(B, 1, MAP_SIZE, MAP_SIZE, device=device)
        prev_valid = torch.zeros(B, 1, MAP_SIZE, MAP_SIZE, device=device)
        prev_pos = pos_seq[0]
        prev_yaw = yaw_seq[0]
        
        for t in range(T):
            curr_pos = pos_seq[t]
            curr_yaw = yaw_seq[t]
            
            # Transform previous output to current frame
            if t > 0:
                prev_output_transformed, prev_valid_transformed = transform_height_map_with_mask(
                    prev_output, prev_valid, prev_pos, prev_yaw, curr_pos, curr_yaw
                )
            else:
                prev_output_transformed = prev_output
                prev_valid_transformed = prev_valid
            
            # Forward pass
            pred = self.forward(
                sparse_seq[t], mask_seq[t],
                prev_output_transformed, prev_valid_transformed,
                grav_seq[t]
            )
            preds.append(pred)
            
            # Update for next timestep
            use_teacher = (
                target_seq is not None and 
                torch.rand(1).item() < teacher_forcing_ratio
            )
            
            if use_teacher:
                prev_output = target_seq[t].detach()
            else:
                prev_output = pred.detach()
            
            prev_valid = torch.ones(B, 1, MAP_SIZE, MAP_SIZE, device=device)
            prev_pos = curr_pos
            prev_yaw = curr_yaw
        
        return torch.stack(preds, dim=0)
