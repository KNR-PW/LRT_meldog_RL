# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Perception models for terrain reconstruction.

Available models:
- HeightmapConvGRU (V5): Temporal fusion with ConvGRU
- HeightmapAutoregressive (V6): Autoregressive temporal model  
- VoxelSparse: 3D sparse CNN for voxel prediction (TODO)
- SLAMBaseline: Classical SLAM comparison (TODO)

Supporting modules:
- DepthProjector: Projects depth images to height maps
- Common utilities: ConvGRU, loss functions, augmentation
"""

from .projector import DepthProjector, euler_from_quat
from .common import ConvGRU, ConvGRUCell, HybridTerrainLoss, augment_sequence, augment_sequence_v6
from .heightmap_convgru import HeightmapConvGRU
from .heightmap_autoreg import HeightmapAutoregressive, transform_height_map_with_mask

# TODO: Implement these
# from .voxel_sparse import VoxelSparse
# from .slam_baseline import SLAMBaseline

__all__ = [
    # Projection
    "DepthProjector",
    "euler_from_quat",
    # Models
    "HeightmapConvGRU",
    "HeightmapAutoregressive",
    # "VoxelSparse",
    # "SLAMBaseline",
    # Common
    "ConvGRU",
    "ConvGRUCell", 
    "HybridTerrainLoss",
    "augment_sequence",
    "augment_sequence_v6",
    "transform_height_map_with_mask",
]
