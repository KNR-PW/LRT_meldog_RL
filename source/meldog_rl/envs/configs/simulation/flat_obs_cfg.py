# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Flat terrain with obstacles configuration for simulation training.

TODO: This is a placeholder. Implement proper obstacle generation.
"""

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from ..base_cfg import BaseMeldogEnvCfg, BaseEventCfg


@configclass
class FlatObsSimCfg(BaseMeldogEnvCfg):
    """Flat terrain with obstacles for simulation training.
    
    Use this for:
    - Testing obstacle avoidance
    - Training perceptive locomotion on simple terrain
    
    TODO: Add obstacle spawning logic
    """
    
    # Flat plane terrain (obstacles will be spawned separately)
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
    
    # Base domain randomization (minimal)
    events: BaseEventCfg = BaseEventCfg()
    
    # Cameras disabled for fast training
    tiled_camera_front = None
    tiled_camera_rear = None
    tiled_camera_left = None
    tiled_camera_right = None
    tiled_camera_top = None
    
    # TODO: Add obstacle configuration
    # obstacles: ObstaclesCfg = ObstaclesCfg(
    #     num_obstacles=10,
    #     size_range=(0.1, 0.5),
    #     height_range=(0.1, 0.3),
    # )
