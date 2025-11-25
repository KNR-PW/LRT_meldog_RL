# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg

##
# Robot Configuration
##

MELDOG_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/ubuntu/Downloads/Meldog-1.4.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, 
            solver_position_iteration_count=4, 
            solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.55),
        joint_pos={
            "LFT_joint": 0.0,
            "LFH_joint": -0.8,
            "LFK_joint": 1.6,
            "RFT_joint": 0.0,
            "RFH_joint": -0.8,
            "RFK_joint": 1.6,
            "LRT_joint": 0.0,
            "LRH_joint": -0.8,
            "LRK_joint": 1.6,
            "RRT_joint": 0.0,
            "RRH_joint": -0.8,
            "RRK_joint": 1.6,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "all_joints": DCMotorCfg(
            joint_names_expr=[".*"],
            effort_limit=34.895,
            saturation_effort=34.895, 
            stiffness=25.0,
            damping=2.5,
            velocity_limit=18.9,
        )
    },
)

##
# Environment Configuration
##

@configclass
class MeldogSimpleLocomotionPolicyEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 20.0
    
    # spaces
    action_space = 12
    observation_space = 45 # 3 (lin) + 3 (ang) + 3 (grav) + 12 (pos) + 12 (vel) + 12 (prev act)
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = MELDOG_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=True)

    ##
    # Custom Parameters
    ##
    
    @configclass
    class CustomParams:
        """Parameters for the Meldog locomotion task."""
        # --- Robot Names
        base_link_name = "trunk_link"
        foot_link_names = ["LFF_link", "RFF_link", "LRF_link", "RRF_link"]
        
        # --- Command
        class Commands:
            class Ranges:
                lin_vel_x = [-1.0, 1.0]
                lin_vel_y = [-1.0, 1.0]
                ang_vel_z = [-1.0, 1.0]
            
        # --- Reward Scales
        class RewScale:
            lin_vel_xy = 5.0
            lin_vel_y = 0.5
            ang_vel_z = 0.5
            lin_vel_z = 0.2
            ang_vel_xy = 0.05
            dof_pos_limits = 0.2
            dof_vel = 0.005
            action_rate = 0.05
            termination = 5.0
            alive = 2.0

        # --- Termination Conditions
        class Terminations:
            reset_robot_on_base_contact = True
            reset_robot_on_joint_limits = False
            reset_robot_on_bad_orientation = True
            max_roll_pitch_rad = 0.2

    params: CustomParams = CustomParams()