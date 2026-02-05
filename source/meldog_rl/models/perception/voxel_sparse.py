# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""3D Voxel prediction using sparse convolutions.

TODO: Implement using MinkowskiEngine or TorchSparse.
"""

import torch
import torch.nn as nn


class VoxelSparse(nn.Module):
    """3D Voxel prediction using sparse convolutions.
    
    TODO: Implement based on volumetric representation papers
    """
    
    def __init__(
        self,
        num_cameras: int = 4,
        voxel_size: float = 0.05,
        grid_size: tuple[int, int, int] = (40, 40, 10),  # XYZ
    ):
        super().__init__()
        self.num_cameras = num_cameras
        self.voxel_size = voxel_size
        self.grid_size = grid_size
        
        # TODO: Implement using MinkowskiEngine or TorchSparse
        raise NotImplementedError("Implement 3D voxel model")
    
    def forward(
        self,
        depth_images: torch.Tensor,
        camera_poses: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            depth_images: (B, num_cameras, H, W) depth images
            camera_poses: (B, num_cameras, 4, 4) camera extrinsics
            
        Returns:
            voxel_grid: (B, X, Y, Z) occupancy/height voxel grid
        """
        raise NotImplementedError("Implement 3D voxel model")
