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

    Curriculum Learning:
    - Enabled by default (enable_curriculum=True)
    - Progressively increases terrain difficulty based on performance
    - To disable: --enable_curriculum False

    Terrain Difficulty Progression (matches Unitree Go2):
    - Starts at level 0 (easiest): flat terrain with minimal obstacles
    - Progresses to level 9+ (hardest): complex obstacles and rough terrain
    - Box heights: 0.025m (easy) → 0.1m (hard)
    - Random roughness: 0.01m noise (easy) → 0.06m noise (hard)
    """
    
    # Rough procedural terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,  # Unitree Go2 uses 5 initial levels
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
            # Match Unitree Go2 terrain difficulty progression
            # These values define the range from easiest (start) to hardest (end) terrains
            self.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
            self.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
            self.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01
            self.terrain.terrain_generator.sub_terrains["pyramid_stairs"].step_height_range = (0.05, 0.1)
            self.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].step_height_range = (0.05, 0.1)

            # Enable curriculum learning (progressively increase difficulty)
            self.terrain.terrain_generator.curriculum = self.enable_curriculum


    # Override base config
    feet_air_time = 0.5
    action_scale = 0.3

    lin_vel_reward_scale = 1.5              # ANYmal: 1.0,   Unitree: 1.5
    yaw_rate_reward_scale = 0.7             # ANYmal: 0.5,   Unitree: 0.75
    z_vel_reward_scale = -2.0               # ANYmal: -2.0,  Unitree: -2.0
    ang_vel_reward_scale = -0.05             # ANYmal: -0.05, Unitree: -0.05
    
    joint_torque_reward_scale = -1.0e-4     # ANYmal: -2.5e-5, Unitree: -2.0e-4
    joint_accel_reward_scale = -5.0e-7     # ANYmal: -2.5e-7, Unitree: -2.5e-7
    action_rate_reward_scale = -0.01        # ANYmal: -0.01, Unitree: -0.01
    
    feet_air_time_reward_scale = 2.0        # ANYmal: 0.5,   Unitree: 0.01
    undesired_contact_reward_scale = -1.0   # ANYmal: -1.0,  Unitree: None 
    
    flat_orientation_reward_scale = -0.0    # ANYmal: 0.0,   Unitree: 0.0


@configclass
class RoughSimCfg_PLAY(RoughSimCfg):
    """Rough terrain for playing/inference (no curriculum).

    Use this for:
    - Policy evaluation
    - Recording videos
    - Real-world deployment testing

    To use: python play.py --task=Isaac-Locomotion-Rough-Meldog-Play-v0
    """

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        # Disable curriculum for play
        self.enable_curriculum = False
        if self.terrain.terrain_generator is not None:
            self.terrain.terrain_generator.curriculum = False

        # Spawn on all difficulty levels (not just easiest)
        self.terrain.max_init_terrain_level = None  # Random across all levels

        # Smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # Disable randomization for play
        self.events = None  # No domain randomization during play


