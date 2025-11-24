# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.scene import InteractiveScene
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply_inverse
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.spawners.lights import DomeLightCfg
import pdb

from .meldog_simple_locomotion_policy_env_cfg import MeldogSimpleLocomotionPolicyEnvCfg

@torch.jit.script
def quat_to_euler_xyz(quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert quaternions to Euler angles (XYZ convention).
    Args:
        quat: Tensor of shape (..., 4) with (w, x, y, z) layout.
    Returns:
        tuple: (roll, pitch, yaw) tensors of shape (...).
    """
    # Unbind the quaternion components (w, x, y, z)
    w, x, y, z = quat.unbind(dim=-1)
    
    # -- Roll (x-axis rotation) --
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # -- Pitch (y-axis rotation) --
    sinp = 2 * (w * y - z * x)
    # Clamp to handle numerical errors
    pitch = torch.where(torch.abs(sinp) >= 1, torch.sign(sinp) * torch.pi / 2, torch.asin(sinp))

    # -- Yaw (z-axis rotation) --
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

class MeldogSimpleLocomotionPolicyEnv(DirectRLEnv):
    """
    DirectRLEnv class for the Meldog simple locomotion task.
    
    This class is responsible for the simulation logic, including:
    - Setting up the scene
    - Applying actions
    - Gathering observations
    - Calculating rewards
    - Checking for terminations
    - Resetting the environment
    """
    cfg: MeldogSimpleLocomotionPolicyEnvCfg

    def __init__(self, cfg: MeldogSimpleLocomotionPolicyEnvCfg, render_mode: str | None = None, **kwargs):
        # Initialize the base class
        super().__init__(cfg, render_mode, **kwargs)

        # -- Robot Data --
        # Get the default joint positions (our 'standing' pose)
        self.default_joint_pos = self.robot.data.default_joint_pos.clone()
        
        # Get the default root state (position and orientation)
        self.default_root_state = self.robot.data.default_root_state.clone()
        
        # Get the joint position and velocity buffers
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        
        # Get the robot's root state (position, orientation, velocities)
        self.root_state = self.robot.data.root_state_w
        
        # -- Task-Specific Buffers --
        # Previous actions (for action rate penalty)
        self.last_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        
        # Robot's base linear and angular velocity (in base frame)
        self.base_lin_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.base_ang_vel = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Gravity vector in the base frame
        self.gravity_vec = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Velocity commands
        self.commands = torch.zeros(self.num_envs, 3, device=self.device)

        # -- Find Key Body/Joint Indices --
        # Get the index for the base link (e.g., "trunk_link")
        self.base_link_idx, _ = self.robot.find_bodies(self.cfg.params.base_link_name)
        
        # Get indices for the feet (e.g., ["LFF_link", ...])
        self.foot_link_indices, _ = self.robot.find_bodies(self.cfg.params.foot_link_names)
        
        # Get indices for all 12 actuated joints
        self.actuated_joint_indices = self.robot.find_joints(self.robot.actuators["all_joints"].cfg.joint_names_expr)[0]


    def _setup_scene(self):
        """Set up the simulation scene."""
        # Spawn the robot
        self.robot = Articulation(self.cfg.robot_cfg)
        
        # Spawn a flat ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # Add scene lighting
        light_cfg = DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        
        # Add contact sensors to the feet
        # This allows us to check self.foot_contact_sensor.data.is_in_contact
        self.foot_contact_sensor = ContactSensor(
            cfg=ContactSensorCfg(
                prim_path="/World/envs/env_.*/Robot/meldog_core/.*F_link",
                filter_prim_paths_expr=["/World/ground"]
            ),
        )

        # Add a separate contact sensor for the base (trunk_link)
        self.base_contact_sensor = ContactSensor(
             cfg=ContactSensorCfg(
                 prim_path=f"/World/envs/env_.*/Robot/meldog_core/{self.cfg.params.base_link_name}",
                 filter_prim_paths_expr=["/World/ground"]
             ),
         )

        # Clone environments
        self.scene.clone_environments(copy_from_source=False)
        
        # Add robot and sensors to the scene
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["foot_contact_sensor"] = self.foot_contact_sensor
        
        # !! ADD THIS LINE !!
        self.scene.sensors["base_contact_sensor"] = self.base_contact_sensor


    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Apply actions to the robot before the physics step."""
        # Store the actions for observation and reward calculation
        self.actions = actions.clone()
        
    def _apply_action(self) -> None:
        """Apply actions to the robot."""
        # -- CHANGE TO POSITION CONTROL --
        
        # 1. Define a scaling factor. 
        # This determines how far the robot can move from the default pose.
        # 0.5 radians is a good range (approx 30 degrees).
        action_scale = 0.5 
        
        # 2. Compute the target joint positions
        # Target = Default Standing Pose + (Policy Output * Scale)
        current_targets = self.default_joint_pos + (self.actions * action_scale)
        
        # 3. Clip targets to safe limits (optional but recommended)
        # Assuming your robot generally operates between -3.14 and 3.14
        current_targets = torch.clamp(current_targets, -3.14, 3.14)

        # 4. Send Position Targets to the simulator
        # The internal PD controller (defined in config) will generate the torques
        self.robot.set_joint_position_target(current_targets, joint_ids=self.actuated_joint_indices)

    def _get_observations(self) -> dict:
        """Get observations for the RL policy."""
        # -- 1. Update internal state buffers --
        
        # Get the robot's root state (pos, quat, lin_vel, ang_vel)
        self.root_state = self.robot.data.root_state_w
        
        # Get base velocities in world frame
        base_lin_vel_world = self.root_state[:, 7:10]
        base_ang_vel_world = self.root_state[:, 10:13]
        
        # Get base orientation (quaternion)
        base_quat = self.root_state[:, 3:7]
        
        # Transform velocities and gravity to the robot's base frame
        # This makes the observation independent of the robot's orientation
        self.base_lin_vel = quat_apply_inverse(base_quat, base_lin_vel_world)
        self.base_ang_vel = quat_apply_inverse(base_quat, base_ang_vel_world)
        
        # Create a gravity vector [0, 0, -1] and rotate it to the base frame
        gravity_world = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)
        self.gravity_vec = quat_apply_inverse(base_quat, gravity_world)
        
        # Get joint positions and velocities
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        
        # -- 2. Assemble the observation tensor --
        # This observation is 45 dimensional, as defined in the config.
        obs_list = [
            self.base_lin_vel,                   # 3
            self.base_ang_vel,                   # 3
            self.gravity_vec,                    # 3
            self.joint_pos,                      # 12
            self.joint_vel,                      # 12
            self.last_actions                    # 12
        ]
        
        obs_buf = torch.cat(obs_list, dim=-1)

        # Return observations as a dictionary
        observations = {"policy": obs_buf}
        return observations


    def _get_rewards(self) -> torch.Tensor:
        """Get rewards for the current step."""
        # Get parameters from the config
        rew_cfg = self.cfg.params.RewScale
        
        # 1. Track linear velocity command (forward)
        # We'll use a simple command: move forward at 1.0 m/s
        target_vel_x = self.cfg.params.Commands.Ranges.lin_vel_x[1]
        rew_lin_vel_xy = torch.exp(-torch.square(self.base_lin_vel[:, 0] - target_vel_x))
        
        # --- NEW: Calculate the penalty term ---
        # We want y-velocity to be 0. We square it so positive/negative drift are both punished.
        rew_lin_vel_y = torch.square(self.base_lin_vel[:, 1])
        
        # 2. Track angular velocity command (zero)
        rew_ang_vel_z = torch.exp(-torch.square(self.base_ang_vel[:, 2]))
        
        # 3. Penalize other velocities
        rew_lin_vel_z = torch.square(self.base_lin_vel[:, 2]) # Penalize z velocity
        rew_ang_vel_xy = torch.sum(torch.square(self.base_ang_vel[:, 0:2]), dim=-1) # Penalize roll/pitch
        
        # 4. Penalize joint velocity and action rate
        rew_dof_vel = torch.sum(torch.square(self.joint_vel), dim=-1)
        rew_action_rate = torch.sum(torch.square(self.last_actions - self.actions), dim=-1)
        
        # 5. Penalize joint position limits (Soft Limits Implementation)
        # We define a "soft" limit slightly smaller than the hardware limit.
        # e.g., if range is +/- 1.5, we penalize if it goes beyond +/- 1.2

        # Define a safe range (in radians) from the default position
        soft_limit_threshold = 1.0  # Allow 1.0 radian deviation before penalizing

        # Calculate deviation from default
        deviation = torch.abs(self.joint_pos - self.default_joint_pos)

        # Only penalize the part of the deviation that exceeds the threshold
        violation = torch.maximum(deviation - soft_limit_threshold, torch.tensor(0.0, device=self.device))

        # Square the violation
        rew_dof_pos_limits = torch.sum(torch.square(violation), dim=-1)

        # 6. Survival bonus
        rew_alive = torch.ones_like(rew_lin_vel_xy)

        # 7. Total Reward
        total_reward = (
            rew_cfg.lin_vel_xy * rew_lin_vel_xy +
            - rew_cfg.lin_vel_y * rew_lin_vel_y +
            rew_cfg.ang_vel_z * rew_ang_vel_z -
            rew_cfg.lin_vel_z * rew_lin_vel_z -
            rew_cfg.ang_vel_xy * rew_ang_vel_xy -
            rew_cfg.dof_vel * rew_dof_vel -
            rew_cfg.action_rate * rew_action_rate -
            rew_cfg.dof_pos_limits * rew_dof_pos_limits +
            rew_cfg.alive * rew_alive
        )

        # 8. Add termination penalty
        total_reward = torch.where(self.reset_terminated, -rew_cfg.termination, total_reward)

        return total_reward


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get termination and timeout flags."""
        term_cfg = self.cfg.params.Terminations
        
        # -- 1. Check for timeouts --
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # -- 2. Check for terminations --
        
        # Get root state
        root_pos = self.robot.data.root_state_w[:, 0:3]
        root_quat = self.robot.data.root_state_w[:, 3:7]
        
        # a. Terminate if base is too low (fell over)
        base_height = root_pos[:, 2]
        termination_height = self.default_root_state[0, 2] - 0.3
        fell_over = base_height < termination_height
        
        # b. Terminate if orientation is bad (Roll/Pitch too high)
        if term_cfg.reset_robot_on_bad_orientation:
            # Use the correct function name here
            roll, pitch, _ = quat_to_euler_xyz(root_quat)
            
            bad_orientation = (torch.abs(roll) > term_cfg.max_roll_pitch_rad) | \
                              (torch.abs(pitch) > term_cfg.max_roll_pitch_rad)
            
            fell_over = fell_over | bad_orientation

        # c. Terminate if base hits the ground (if configured)
        if term_cfg.reset_robot_on_base_contact:
            net_forces = self.base_contact_sensor.data.net_forces_w
            force_magnitudes = torch.norm(net_forces, dim=-1)
            base_contact = torch.any(force_magnitudes > 1.0, dim=-1)
            fell_over = fell_over | base_contact

        # d. Terminate on joint limits (if configured)
        if term_cfg.reset_robot_on_joint_limits:
            joint_limits_exceeded = torch.any(
                torch.abs(self.joint_pos - self.default_joint_pos) > 0.5,
                dim=-1
            )
            terminated = fell_over | joint_limits_exceeded
        else:
            terminated = fell_over

        # --- LOGGING CUSTOM METRICS ---
        if not hasattr(self, "extras"): self.extras = {}
        
        self.extras["log"] = {
            "Episode/Vel_Linear_X": torch.mean(self.base_lin_vel[:, 0]),
            "Episode/Base_Height": torch.mean(self.root_state[:, 2]),
            "Episode/Action_Rate": torch.mean(torch.norm(self.actions - self.last_actions, dim=-1)),
            "Episode/Torque_Estimate": torch.mean(torch.norm(self.actions, dim=-1))
        }

        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset the environments specified by env_ids."""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        
        # -- 1. Reset root state --
        # Get default state for the specified envs
        root_state = self.default_root_state[env_ids]
        # Add the environment origin
        root_state[:, :3] += self.scene.env_origins[env_ids]
        
        # -- 2. Reset joint states --
        joint_pos = self.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(self.joint_vel[env_ids])

        # -- 3. Write new states to sim --
        self.robot.write_root_state_to_sim(root_state, env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # -- 4. Reset internal buffers --
        self.last_actions[env_ids] = 0.0
        self.commands[env_ids] = 0.0

        # -- 5. Call super class reset --
        # This handles the `reset_terminated` buffer
        super()._reset_idx(env_ids)