# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Rough terrain configuration for simulation training."""

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass

from ..base_cfg import BaseMeldogEnvCfg, BaseEventCfg


@configclass
class RoughSimCfg(BaseMeldogEnvCfg):
    """Rough terrain for simulation training.
    
    Use this for:
    - Training robust locomotion policies
    - Testing terrain adaptation
    """
    
    # Rough procedural terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=9,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
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

    def __post_init__(self):
        """Customize terrain generator parameters."""
        super().__post_init__()
        if self.terrain.terrain_generator is not None:
            # Adjust terrain difficulty
            self.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
            self.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.02, 0.06)
            self.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01
            self.terrain.terrain_generator.sub_terrains["pyramid_stairs"].step_height_range = (0.05, 0.1)
            self.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].step_height_range = (0.05, 0.1)
