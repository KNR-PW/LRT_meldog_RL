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
import pdb

from .meldog_simple_locomotion_policy_env_cfg import MeldogSimpleLocomotionPolicyEnvCfg


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
        # The actions are target joint positions.
        # We need to scale them by the action_scale to get the desired position.
        # Here, we assume the policy outputs values in [-1, 1].
        # We scale them to be offsets from the default "standing" pose.
        
        # For a simple starter policy, we'll assume actions are *direct forces/torques*
        # (This is simpler to start with than a position controller)
        
        # Convert actions to torques (efforts)
        # We use the motor's effort limit as the action scale.
        action_scale = self.robot.actuators["all_joints"].cfg.effort_limit
        efforts = self.actions * action_scale
        
        # Apply the efforts to the correct joints
        self.robot.set_joint_effort_target(efforts, joint_ids=self.actuated_joint_indices)


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
        
        # 2. Track angular velocity command (zero)
        rew_ang_vel_z = torch.exp(-torch.square(self.base_ang_vel[:, 2]))
        
        # 3. Penalize other velocities
        rew_lin_vel_z = torch.square(self.base_lin_vel[:, 2]) # Penalize z velocity
        rew_ang_vel_xy = torch.sum(torch.square(self.base_ang_vel[:, 0:2]), dim=-1) # Penalize roll/pitch
        
        # 4. Penalize joint velocity and action rate
        rew_dof_vel = torch.sum(torch.square(self.joint_vel), dim=-1)
        rew_action_rate = torch.sum(torch.square(self.last_actions - self.actions), dim=-1)
        
        # 5. Penalize joint position limits
        # (This is a simple version, a more complex one would use the actual limits)
        rew_dof_pos_limits = torch.sum(torch.square(self.joint_pos - self.default_joint_pos), dim=-1)
        
        # 6. Survival bonus
        rew_alive = torch.ones_like(rew_lin_vel_xy)

        # 7. Total Reward
        total_reward = (
            rew_cfg.lin_vel_xy * rew_lin_vel_xy +
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
        
        # a. Terminate if base is too low (fell over)
        base_height = root_pos[:, 2]
        # We use the default height minus a threshold
        termination_height = self.default_root_state[0, 2] - 0.3
        fell_over = base_height < termination_height
        
        # b. Terminate if base hits the ground (if configured)
        if term_cfg.reset_robot_on_base_contact:
            # Access the net forces acting on the base. Shape: (num_envs, num_bodies, 3)
            net_forces = self.base_contact_sensor.data.net_forces_w
            
            # Calculate the magnitude of the force. Shape: (num_envs, num_bodies)
            # We assume the base sensor tracks only 1 body (trunk_link) per environment.
            force_magnitudes = torch.norm(net_forces, dim=-1)
            
            # Check if force is non-zero (greater than a small threshold like 1.0 Newton)
            # We use 'any' to flatten the body dimension -> Shape: (num_envs,)
            base_contact = torch.any(force_magnitudes > 1.0, dim=-1)
            
            fell_over = fell_over | base_contact

        # c. Terminate on joint limits (if configured)
        if term_cfg.reset_robot_on_joint_limits:
            # A simple check: if any joint is too far from default
            joint_limits_exceeded = torch.any(
                torch.abs(self.joint_pos - self.default_joint_pos) > 0.5, # 0.5 rad = ~30 deg
                dim=-1
            )
            terminated = fell_over | joint_limits_exceeded
        else:
            terminated = fell_over

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