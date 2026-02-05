# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Flat terrain configuration for simulation training."""

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from ..base_cfg import BaseMeldogEnvCfg, BaseEventCfg


@configclass
class FlatSimCfg(BaseMeldogEnvCfg):
    """Flat terrain for simulation training.
    
    Use this for:
    - Initial policy development
    - Baseline comparisons
    - Fast iteration
    """
    
    # Flat plane terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    
    # Override base config
    flat_orientation_reward_scale = -5.0

    # Base domain randomization (minimal)
    events: BaseEventCfg = BaseEventCfg()
    
    # Cameras disabled for fast training
    tiled_camera_front = None
    tiled_camera_rear = None
    tiled_camera_left = None
    tiled_camera_right = None
    tiled_camera_top = None
