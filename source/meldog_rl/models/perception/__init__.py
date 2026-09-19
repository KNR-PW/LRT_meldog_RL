# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Perception models for terrain reconstruction."""

from .common import ConvGRU, ConvGRUCell, HybridTerrainLoss, augment_sequence, augment_sequence_v6
from .elevation_mapper import ElevationMapper
from .heightmap_autoreg import HeightmapAutoregressive, transform_height_map_with_mask
from .heightmap_convgru import HeightmapConvGRU
from .projector import DepthProjector, euler_from_quat
from .slam_baseline import SLAMBaseline

# TODO: VoxelSparse

__all__ = [
    # Projection
    "DepthProjector",
    "euler_from_quat",
    # Models
    "HeightmapConvGRU",
    "HeightmapAutoregressive",
    "SLAMBaseline",
    "ElevationMapper",
    # "VoxelSparse",
    # Common
    "ConvGRU",
    "ConvGRUCell",
    "HybridTerrainLoss",
    "augment_sequence",
    "augment_sequence_v6",
    "transform_height_map_with_mask",
]
