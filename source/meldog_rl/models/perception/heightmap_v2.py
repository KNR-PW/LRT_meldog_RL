# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Thesis model v2: deeper single-frame U-Net height-map refiner (thesis section 2.6.2).

Copied from ``archive/perception_v3_dev`` (tip ``ae6c491``),
``scripts/rsl_rl/train_perception_v0.py``, class ``SparseMapRefinerDeep`` (code-era name
"V4.1", run with ``--deep``). That class had an optional ``SelfAttention2D`` block at the
bottleneck (``use_attention=True``); v2 is the plain version with attention off, so the
option and the attention branch are left out. The remaining layers are copied unchanged and keep
their attribute names, so state_dicts saved with ``use_attention=False`` load here.

Architecture:
- Input: sparse height map + occlusion mask (2 channels, 40x40) and the gravity vector
- U-Net encoder with 3 stages: 40 -> 20 -> 10 -> 5 (5x5 bottleneck)
- Gravity MLP 3 -> 64 -> 64, broadcast over the bottleneck
- Transposed-conv decoder with skip connections: 5 -> 10 -> 20 -> 40
- Final 3x3 conv that also sees the raw input

No temporal state: every frame is refined on its own. v3 is this backbone with a ConvGRU at
the bottleneck.
"""

import torch
import torch.nn as nn


class HeightmapV2(nn.Module):
    """Deeper encoder for more global context (40->20->10->5)."""

    def __init__(self):
        super().__init__()

        # Level 1: 40 -> 20
        self.enc1 = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        # Level 2: 20 -> 10
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        # Level 3: 10 -> 5
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Gravity embedding at bottleneck
        self.mlp_gravity = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 64))

        # Decoder: 5 -> 10 -> 20 -> 40
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(128 + 64, 64, 4, 2, 1), nn.ReLU())
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(64 + 64, 32, 4, 2, 1), nn.ReLU())
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(32 + 32, 16, 4, 2, 1), nn.ReLU())

        self.final_conv = nn.Conv2d(16 + 2, 1, 3, padding=1)

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
        s3 = self.enc3(s2)  # B, 128, 5, 5

        B, _, H, W = s3.shape
        grav_embed = self.mlp_gravity(grav).view(B, 64, 1, 1).expand(B, 64, H, W)

        up1 = self.dec1(torch.cat([s3, grav_embed], dim=1))  # B, 64, 10, 10
        up2 = self.dec2(torch.cat([up1, s2], dim=1))  # B, 32, 20, 20
        up3 = self.dec3(torch.cat([up2, s1], dim=1))  # B, 16, 40, 40

        return self.final_conv(torch.cat([up3, x_in], dim=1))
