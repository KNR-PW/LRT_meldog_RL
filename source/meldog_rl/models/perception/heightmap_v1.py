# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Thesis model v1: single-frame U-Net height-map refiner (thesis section 2.6.1).

Copied from ``archive/perception_v2_dev`` (tip ``21dbd0a``),
``scripts/rsl_rl/train_perception_v0.py``, class ``SparseMapRefiner`` (code-era name "V4").
The layers are copied unchanged and keep their attribute names, so state_dicts saved by that
class load here. Only the class name and the docstrings differ.

Architecture:
- Input: sparse height map + occlusion mask (2 channels, 40x40) and the gravity vector
- U-Net encoder with 2 stages: 40 -> 20 -> 10 (10x10 bottleneck)
- Gravity MLP 3 -> 32 -> 32, broadcast over the bottleneck
- Transposed-conv decoder with skip connections: 10 -> 20 -> 40
- Final 3x3 conv that also sees the raw input

No temporal state: every frame is refined on its own.

Kept as in the original: the stride-2 downsampling convs and both transposed convs have no
activation after them, so the decoder is linear in its inputs.
"""

import torch
import torch.nn as nn


class HeightmapV1(nn.Module):
    """Single-frame U-Net (2 encoder stages, 10x10 bottleneck).

    Input:
        - Sparse height map (projected from depth cameras)
        - Occlusion mask
        - Gravity vector

    Output:
        - Refined height map (40x40)
    """

    def __init__(self):
        super().__init__()
        # Level 1: 40 -> 20
        self.enc1 = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),  # Learnable downsample
        )
        # Level 2: 20 -> 10
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
        )

        self.mlp_gravity = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 32))

        # Decoder with Transpose Convs for sharpness
        self.dec1 = nn.ConvTranspose2d(64 + 32, 32, kernel_size=4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(32 + 32, 16, kernel_size=4, stride=2, padding=1)
        self.final_conv = nn.Conv2d(16 + 2, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, grav: torch.Tensor) -> torch.Tensor:
        """Forward pass for a single frame.

        Args:
            x: (B, 1, 40, 40) sparse height map
            mask: (B, 1, 40, 40) occlusion mask
            grav: (B, 3) gravity vector

        Returns:
            pred: (B, 1, 40, 40) predicted height map
        """
        x_in = torch.cat([x, mask], dim=1)  # B, 2, 40, 40
        s1 = self.enc1(x_in)  # B, 32, 20, 20
        s2 = self.enc2(s1)  # B, 64, 10, 10

        B, _, H, W = s2.shape
        grav_embed = self.mlp_gravity(grav).view(B, 32, 1, 1).expand(B, 32, H, W)

        up1 = self.dec1(torch.cat([s2, grav_embed], dim=1))  # B, 32, 20, 20
        up2 = self.dec2(torch.cat([up1, s1], dim=1))  # B, 16, 40, 40

        return self.final_conv(torch.cat([up2, x_in], dim=1))
