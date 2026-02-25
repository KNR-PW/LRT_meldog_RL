# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Rough terrain configuration for dataset collection."""

from isaaclab.utils import configclass

from ..base_cfg import CAMERA_FRONT, CAMERA_BACK, CAMERA_LEFT, CAMERA_RIGHT, CAMERA_TOP
from ..simulation.rough_cfg import RoughSimCfg


@configclass
class RoughDatasetCfg(RoughSimCfg):
    """Rough terrain with cameras enabled for dataset collection.
    
    Inherits terrain from RoughSimCfg, enables all cameras.
    """
    
    # Enable all cameras for data collection
    tiled_camera_front = CAMERA_FRONT
    tiled_camera_rear = CAMERA_BACK
    tiled_camera_left = CAMERA_LEFT
    tiled_camera_right = CAMERA_RIGHT
    tiled_camera_top = CAMERA_TOP

    def __post_init__(self):
        super().__post_init__()
        # Override scene settings for dataset collection
        self.scene.num_envs = 64  # Fewer envs when cameras are enabled
