# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Rough terrain with obstacles configuration for simulation training."""

import isaaclab.sim as sim_utils
from isaaclab.terrains import (
    TerrainImporterCfg,
    MeshRepeatedBoxesTerrainCfg,
    MeshRepeatedCylindersTerrainCfg,
    MeshRepeatedPyramidsTerrainCfg,
)
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass

from ..base_cfg import BaseMeldogEnvCfg, BaseEventCfg
from .terrain_utils import make_cylinder_high_poly


@configclass
class RoughObsSimCfg(BaseMeldogEnvCfg):
    """Rough terrain with obstacles for simulation training.

    Use this for:
    - Most challenging locomotion training
    - Full perceptive locomotion pipeline

    Extends standard ROUGH_TERRAINS_CFG with repeated object obstacles
    (boxes, cylinders, pyramids/cones, walls).
    """

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
        """Customize terrain and add repeated object obstacles."""
        super().__post_init__()
        if self.terrain.terrain_generator is not None:
            tg = self.terrain.terrain_generator

            # Customize existing rough terrain parameters
            tg.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
            tg.sub_terrains["random_rough"].noise_range = (0.02, 0.06)
            tg.sub_terrains["random_rough"].noise_step = 0.01
            tg.sub_terrains["pyramid_stairs"].step_height_range = (0.05, 0.1)
            tg.sub_terrains["pyramid_stairs_inv"].step_height_range = (0.05, 0.1)

            # Reduce existing proportions to make room for repeated objects
            # Original: stairs 0.2, stairs_inv 0.2, boxes 0.2, rough 0.2, slope 0.1, slope_inv 0.1
            tg.sub_terrains["pyramid_stairs"].proportion = 0.11
            tg.sub_terrains["pyramid_stairs_inv"].proportion = 0.11
            tg.sub_terrains["boxes"].proportion = 0.11
            tg.sub_terrains["random_rough"].proportion = 0.11
            tg.sub_terrains["hf_pyramid_slope"].proportion = 0.055
            tg.sub_terrains["hf_pyramid_slope_inv"].proportion = 0.055

            # Add repeated object obstacles (total 0.45)
            tg.sub_terrains["repeated_boxes"] = MeshRepeatedBoxesTerrainCfg(
                proportion=0.18,
                object_params_start=MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                    num_objects=5,
                    height=0.1,
                    size=(0.2, 0.2),  # Small, can traverse over
                    max_yx_angle=5.0,  # Slight tilt variation
                ),
                object_params_end=MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                    num_objects=12,
                    height=1.0,
                    size=(0.8, 0.8),  # Very large, must navigate around
                    max_yx_angle=5.0,
                ),
                abs_height_noise=(0.0, 0.05),  # Add slight height variation
                platform_width=1.5,
                platform_height=0.0,  # Keep spawn platform at ground level
            )
            tg.sub_terrains["repeated_cylinders"] = MeshRepeatedCylindersTerrainCfg(
                proportion=0.15,
                object_type=make_cylinder_high_poly,  # Use high-poly cylinders (16-24 sections)
                object_params_start=MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                    num_objects=4,
                    height=0.15,
                    radius=0.08,  # Small, can traverse
                    max_yx_angle=10.0,  # More tilt variation
                ),
                object_params_end=MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                    num_objects=10,
                    height=1.2,
                    radius=0.3,  # Extra tall, must avoid
                    max_yx_angle=10.0,
                ),
                abs_height_noise=(0.0, 0.08),  # More height variation
                platform_width=1.5,
                platform_height=0.0,
            )
            tg.sub_terrains["repeated_pyramids"] = MeshRepeatedPyramidsTerrainCfg(
                proportion=0.10,
                object_params_start=MeshRepeatedPyramidsTerrainCfg.ObjectCfg(
                    num_objects=6,
                    height=0.15,
                    radius=0.1,  # Small cones
                    max_yx_angle=15.0,  # Moderate tilt
                ),
                object_params_end=MeshRepeatedPyramidsTerrainCfg.ObjectCfg(
                    num_objects=12,
                    height=1.0,
                    radius=0.35,  # Very tall cones/pyramids
                    max_yx_angle=15.0,
                ),
                abs_height_noise=(0.0, 0.06),
                platform_width=1.5,
                platform_height=0.0,
            )
            tg.sub_terrains["walls"] = MeshRepeatedBoxesTerrainCfg(
                proportion=0.07,
                object_params_start=MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                    num_objects=3,
                    height=0.5,
                    size=(0.08, 1.0),  # Thin tall boxes as walls
                    max_yx_angle=0.0,  # Keep perpendicular to ground
                ),
                object_params_end=MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                    num_objects=8,
                    height=2.0,
                    size=(0.12, 2.5),  # Very tall, longer walls
                    max_yx_angle=0.0,
                ),
                platform_width=1.5,
                platform_height=0.0,
            )

            # Enable curriculum learning
            tg.curriculum = self.enable_curriculum

    # Override base config
    feet_air_time = 0.5
    action_scale = 0.3

    lin_vel_reward_scale = 1.5
    yaw_rate_reward_scale = 0.7
    z_vel_reward_scale = -2.0
    ang_vel_reward_scale = -0.05

    joint_torque_reward_scale = -1.0e-4
    joint_accel_reward_scale = -5.0e-7
    action_rate_reward_scale = -0.01

    feet_air_time_reward_scale = 2.0
    undesired_contact_reward_scale = -1.0

    flat_orientation_reward_scale = 0.0
