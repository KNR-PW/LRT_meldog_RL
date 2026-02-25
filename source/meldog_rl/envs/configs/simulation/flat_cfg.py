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
    feet_air_time = 0.5

    lin_vel_reward_scale = 1.5              # ANYmal: 1.0,   Unitree: 1.5
    yaw_rate_reward_scale = 0.7             # ANYmal: 0.5,   Unitree: 0.75
    z_vel_reward_scale = -2.0               # ANYmal: -2.0,  Unitree: -2.0
    ang_vel_reward_scale = -0.05             # ANYmal: -0.05, Unitree: -0.05
    
    joint_torque_reward_scale = -1.0e-4     # ANYmal: -2.5e-5, Unitree: -2.0e-4
    joint_accel_reward_scale = -5.0e-7     # ANYmal: -2.5e-7, Unitree: -2.5e-7
    action_rate_reward_scale = -0.01        # ANYmal: -0.01, Unitree: -0.01
    
    feet_air_time_reward_scale = 2.0        # ANYmal: 0.5,   Unitree: 0.01
    undesired_contact_reward_scale = -1.0   # ANYmal: -1.0,  Unitree: None 
    
    flat_orientation_reward_scale = -5.0    # ANYmal: 0.0,   Unitree: 0.0

    # Base domain randomization (minimal)
    events: BaseEventCfg = BaseEventCfg()
    
    # Cameras disabled for fast training
    tiled_camera_front = None
    tiled_camera_rear = None
    tiled_camera_left = None
    tiled_camera_right = None
    tiled_camera_top = None
