# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Heightmap prediction with ConvGRU temporal fusion (V5 architecture).

Based on Miki et al. "Learning robust perceptive locomotion":
- GRU outperforms LSTM and RNN for temporal modeling
- 2 stacked layers work well
- Belief state captures unobservable information

Architecture:
- U-Net encoder: 40->20->10->5
- ConvGRU at bottleneck (5x5 spatial)
- U-Net decoder with skip connections
"""

import torch
import torch.nn as nn

from .common import ConvGRU


class HeightmapConvGRU(nn.Module):
    """Deep U-Net with ConvGRU at bottleneck for temporal consistency.
    
    Input:
        - Sparse height map (projected from depth cameras)
        - Occlusion mask
        - Gravity vector
        - Previous hidden state (optional)
        
    Output:
        - Refined height map (40x40)
        - New hidden state
        
    Args:
        gru_hidden: Hidden channels in ConvGRU (default 128)
        gru_layers: Number of stacked ConvGRU layers (default 2)
    """
    
    def __init__(self, gru_hidden: int = 128, gru_layers: int = 2):
        super().__init__()
        
        self.gru_hidden = gru_hidden
        self.gru_layers = gru_layers
        
        # Encoder: 40 -> 20 -> 10 -> 5
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
        
        # Decoder with skip connections: 5 -> 10 -> 20 -> 40
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(gru_hidden, 64, 4, 2, 1), nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, 32, 4, 2, 1), nn.ReLU()
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(32 + 32, 16, 4, 2, 1), nn.ReLU()
        )
        
        # Final convolution (includes input as skip connection)
        self.final_conv = nn.Conv2d(16 + 2, 1, 3, padding=1)

    def forward(
        self, 
        x: torch.Tensor, 
        mask: torch.Tensor, 
        grav: torch.Tensor, 
        h_prev: list[torch.Tensor] = None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass for single timestep.
        
        Args:
            x: (B, 1, 40, 40) sparse height map
            mask: (B, 1, 40, 40) occlusion mask
            grav: (B, 3) gravity vector
            h_prev: list of hidden states from ConvGRU, or None
            
        Returns:
            pred: (B, 1, 40, 40) predicted height map
            h_new: list of new hidden states
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
        
        # Decode with skip connections
        up1 = self.dec1(gru_out)                       # B, 64, 10, 10
        up2 = self.dec2(torch.cat([up1, s2], dim=1))   # B, 32, 20, 20
        up3 = self.dec3(torch.cat([up2, s1], dim=1))   # B, 16, 40, 40
        
        # Final prediction with input skip
        pred = self.final_conv(torch.cat([up3, x_in], dim=1))
        
        return pred, h_new
    
    def forward_sequence(
        self, 
        x_seq: torch.Tensor, 
        mask_seq: torch.Tensor, 
        grav_seq: torch.Tensor
    ) -> torch.Tensor:
        """Process entire sequence, maintaining hidden state.
        
        Args:
            x_seq: (T, B, 1, H, W) sparse maps
            mask_seq: (T, B, 1, H, W) occlusion masks
            grav_seq: (T, B, 3) gravity vectors
            
        Returns:
            pred_seq: (T, B, 1, H, W) predictions for all timesteps
        """
        T = x_seq.shape[0]
        preds = []
        h = None
        
        for t in range(T):
            pred, h = self.forward(x_seq[t], mask_seq[t], grav_seq[t], h)
            preds.append(pred)
        
        return torch.stack(preds, dim=0)
    
    def get_hidden_state_size(self) -> tuple[int, int, int]:
        """Get the expected hidden state size.
        
        Returns:
            (num_layers, hidden_channels, spatial_size) 
            where spatial_size is 5 for default 40x40 input
        """
        return (self.gru_layers, self.gru_hidden, 5)
