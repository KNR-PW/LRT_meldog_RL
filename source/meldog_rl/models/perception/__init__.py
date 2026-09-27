# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Perception models for terrain reconstruction.

Models are named and built through the registry (``registry.py``); print it with
``python -m meldog_rl.models.perception``.
"""

from .common import ConvGRU, ConvGRUCell, HybridTerrainLoss, augment_sequence, augment_sequence_v6
from .elevation_mapper import ElevationMapper
from .heightmap_autoreg import HeightmapAutoregressive, transform_height_map_with_mask
from .heightmap_convgru import HeightmapConvGRU
from .heightmap_v1 import HeightmapV1
from .heightmap_v2 import HeightmapV2
from .heightmap_v3 import HeightmapV3
from .projector import DepthProjector, euler_from_quat
from .registry import (
    ACTIVE_MODEL,
    AUTOREGRESSIVE,
    BASELINE,
    MODELS,
    RECURRENT,
    SINGLE_FRAME,
    ModelEntry,
    ModelRegistryError,
    build_model,
    format_table,
    print_models,
    resolve_model,
    resolve_trainable,
)
from .slam_baseline import SLAMBaseline

# TODO: VoxelSparse

__all__ = [
    # Projection
    "DepthProjector",
    "euler_from_quat",
    # Models (thesis)
    "HeightmapV1",
    "HeightmapV2",
    "HeightmapV3",
    # Models (archived, frozen)
    "HeightmapConvGRU",
    "HeightmapAutoregressive",
    # Baselines
    "SLAMBaseline",
    "ElevationMapper",
    # "VoxelSparse",
    # Registry
    "ACTIVE_MODEL",
    "AUTOREGRESSIVE",
    "BASELINE",
    "RECURRENT",
    "SINGLE_FRAME",
    "MODELS",
    "ModelEntry",
    "ModelRegistryError",
    "build_model",
    "format_table",
    "print_models",
    "resolve_model",
    "resolve_trainable",
    # Common
    "ConvGRU",
    "ConvGRUCell",
    "HybridTerrainLoss",
    "augment_sequence",
    "augment_sequence_v6",
    "transform_height_map_with_mask",
]
