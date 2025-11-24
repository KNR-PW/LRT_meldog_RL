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
        # !! IMPORTANT !!
        # Update this path to point to your robot's USD file.
        # This could be an absolute path or a path relative to your IsaacLab folder.
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
        pos=(0.0, 0.0, 1.55),  # Based on your URDF's base_joint origin
        joint_pos={
            # !! IMPORTANT !!
            # These are the 12 joint names from your URDF.
            # You should change the 0.0 values to a default "standing" pose.
            "LFT_joint": 0.0,
            "LFH_joint": 0.0,
            "LFK_joint": 0.0,
            "RFT_joint": 0.0,
            "RFH_joint": 0.0,
            "RFK_joint": 0.0,
            "LRT_joint": 0.0,
            "LRH_joint": 0.0,
            "LRK_joint": 0.0,
            "RRT_joint": 0.0,
            "RRH_joint": 0.0,
            "RRK_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "all_joints": DCMotorCfg(
            # This regex ".*" matches all 12 joints
            joint_names_expr=[".*"],
            # Using the effort limit from your URDF
            effort_limit=34.895,
            # Add this line:
            saturation_effort=34.895, 
            # These are good starting values, copied from other quadrupeds
            stiffness=60.0,
            damping=1.5,
            velocity_limit=18.9, # From your URDF
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
    episode_length_s = 20.0 # Locomotion needs more time than cartpole
    
    # - spaces definition
    action_space = 12 # 12 actuated joints
    
    # This is a common observation space for locomotion:
    # 3 (base lin vel) + 3 (base ang vel) + 3 (gravity vec) + 
    # 12 (joint pos) + 12 (joint vel) + 12 (previous actions) = 45
    observation_space = 45
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot(s)
    robot_cfg: ArticulationCfg = MELDOG_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    # Quadrupeds are bigger, so increase env_spacing
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=5.0, replicate_physics=True)

    ##
    # Custom Parameters
    # (These replace all the old cartpole parameters)
    ##
    
    @configclass
    class CustomParams:
        """Parameters for the Meldog locomotion task."""
        # --- Robot Names
        base_link_name = "trunk_link" # From your URDF
        foot_link_names = ["LFF_link", "RFF_link", "LRF_link", "RRF_link"] # From your URDF
        
        # --- Command
        class Commands:
            """Commands for the robot."""
            class Ranges:
                """Ranges for the commands."""
                lin_vel_x = [-1.0, 1.0]  # min/max linear velocity in x [m/s]
                lin_vel_y = [-1.0, 1.0]  # min/max linear velocity in y [m/s]
                ang_vel_z = [-1.0, 1.0]  # min/max angular velocity in z [rad/s]
            
        # --- Reward Scales
        class RewScale:
            """Reward scales for the task."""
            # (These are all positive, the reward function will handle signs)
            lin_vel_xy = 1.0      # track linear velocity
            ang_vel_z = 0.5       # track angular velocity
            lin_vel_z = 0.2       # penalize z velocity
            ang_vel_xy = 0.05     # penalize roll/pitch velocity
            dof_pos_limits = 1.0  # penalize joint limits
            dof_vel = 0.001       # penalize high joint velocity
            action_rate = 0.01    # penalize jerky actions
            termination = 5.0     # penalize dying
            alive = 2.0           # survival bonus

        # --- Termination Conditions
        class Terminations:
            """Termination conditions for the task."""
            reset_robot_on_base_contact = True
            reset_robot_on_joint_limits = True

    # We assign our new custom parameters class to the config
    params: CustomParams = CustomParams()