# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, RayCaster

# Visualization Imports
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import quat_apply, quat_from_angle_axis

from .meldog_simple_locomotion_policy_env_cfg import MeldogSimpleLocomotionPolicyEnvCfg

class MeldogSimpleLocomotionPolicyEnv(DirectRLEnv):
    cfg: MeldogSimpleLocomotionPolicyEnvCfg

    def __init__(self, cfg: MeldogSimpleLocomotionPolicyEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Buffers
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._action_history = torch.zeros(self.num_envs, 2, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        
        # [!NEW] Command Timer & Mode Buffer
        # Timer: How long until next command switch?
        self._command_timer = torch.zeros(self.num_envs, device=self.device)
        # Mode: Integer ID for what the robot is doing (0=Stand, 1=Rot, etc.)
        # Used for split logging.
        self._command_modes = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Reward Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp", "track_ang_vel_z_exp", "lin_vel_z_l2",
                "ang_vel_xy_l2", "dof_torques_l2", "dof_acc_l2", "action_rate_l2",
                "feet_air_time", "undesired_contacts", "flat_orientation_l2",
                "alive",
                "base_height_l2",     # [!NEW]
                "joint_deviation_l2", # [!NEW]
                "action_accel_l2",
            ]
        }

        # Body IDs
        self._base_id, _ = self._contact_sensor.find_bodies("trunk_link")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*F_link")
        
        # [!CRITICAL] Dual Indices for Visualizer vs Physics
        self._undesired_sensor_ids, _ = self._contact_sensor.find_bodies(".*(H|UL|LL)_link")
        self._undesired_actor_ids, _ = self._robot.find_bodies(".*(H|UL|LL)_link")

    def _setup_scene(self):
        # 1. Robot Setup
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        
        # 2. Sensors Setup
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        
        self._height_scanner = RayCaster(self.cfg.height_scanner)
        self.scene.sensors["height_scanner"] = self._height_scanner
        
        # 3. Terrain Setup
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        
        # 4. Clone Envs & Filter Collisions
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        
        # 5. Lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        
        # Initialize Visualizers
        self.lin_visualizer = VisualizationMarkers(self.cfg.lin_vel_marker)
        self.ang_visualizer = VisualizationMarkers(self.cfg.ang_vel_marker)
        self.contact_visualizer = VisualizationMarkers(self.cfg.contact_marker)

    # [!NEW] Command Curriculum Logic
    def _sample_commands(self, env_ids: torch.Tensor):
        # Percentages:
        # 0-20%:  Stand Still (0,0,0)
        # 20-40%: Pure Rotate (Yaw)
        # 40-60%: Pure Walk X (Forward/Back)
        # 60-70%: Pure Strafe Y (Left/Right)
        # 70-100%: Omni (Mixed)
        
        r = torch.rand(len(env_ids), device=self.device)
        new_cmds = torch.zeros(len(env_ids), 3, device=self.device)
        new_modes = torch.zeros(len(env_ids), dtype=torch.long, device=self.device)

        # Mode 0: Stand Still (r < 0.2) - Default 0.0 is fine
        new_modes[r < 0.2] = 0

        # Mode 1: Pure Rotate (0.2 <= r < 0.4)
        mask = (r >= 0.2) & (r < 0.4)
        new_cmds[mask, 2] = torch.empty(mask.sum(), device=self.device).uniform_(-2.0, 2.0)
        new_modes[mask] = 1

        # Mode 2: Pure Walk X (0.4 <= r < 0.6)
        mask = (r >= 0.4) & (r < 0.6)
        new_cmds[mask, 0] = torch.empty(mask.sum(), device=self.device).uniform_(-1.0, 1.0)
        new_modes[mask] = 2

        # Mode 3: Pure Strafe Y (0.6 <= r < 0.7)
        mask = (r >= 0.6) & (r < 0.7)
        new_cmds[mask, 1] = torch.empty(mask.sum(), device=self.device).uniform_(-0.5, 0.5)
        new_modes[mask] = 3

        # Mode 4: Omni (r >= 0.7)
        mask = (r >= 0.7)
        new_cmds[mask, 0] = torch.empty(mask.sum(), device=self.device).uniform_(-1.0, 1.0)
        new_cmds[mask, 1] = torch.empty(mask.sum(), device=self.device).uniform_(-0.5, 0.5)
        new_cmds[mask, 2] = torch.empty(mask.sum(), device=self.device).uniform_(-1.0, 1.0)
        new_modes[mask] = 4

        self._commands[env_ids] = new_cmds
        self._command_modes[env_ids] = new_modes

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

        # [!NEW] Update Timers
        self._command_timer -= self.step_dt
        reset_ids = (self._command_timer <= 0).nonzero(as_tuple=False).flatten()
        if len(reset_ids) > 0:
            self._sample_commands(reset_ids)
            # Reset timer to random between 4s and 9s
            self._command_timer[reset_ids] = torch.empty(len(reset_ids), device=self.device).uniform_(4.0, 9.0)

        # action rate history
        self._action_history[:, 1] = self._action_history[:, 0]
        self._action_history[:, 0] = self._actions.clone()
        self._actions = actions.clone()

        # Visualization Logic
        if self.cfg.debug_vis:
            robot_pos = self._robot.data.root_pos_w
            robot_quat = self._robot.data.root_quat_w
            
            # 1. Command Arrows
            cmd_lin_local = torch.zeros(self.num_envs, 3, device=self.device)
            cmd_lin_local[:, :2] = self._commands[:, :2]
            cmd_lin_world = quat_apply(robot_quat, cmd_lin_local)
            
            yaw_lin = torch.atan2(cmd_lin_world[:, 1], cmd_lin_world[:, 0])
            axis_z = torch.zeros(self.num_envs, 3, device=self.device)
            axis_z[:, 2] = 1.0
            lin_arrow_quat = quat_from_angle_axis(yaw_lin, axis_z)

            lin_mag = torch.norm(self._commands[:, :2], dim=1)
            lin_arrow_scale = torch.zeros(self.num_envs, 3, device=self.device)
            lin_arrow_scale[:, 0] = torch.clamp(lin_mag, min=0.1) 
            lin_arrow_scale[:, 1] = 0.5 
            lin_arrow_scale[:, 2] = 0.5 
            
            cmd_ang_local = torch.zeros(self.num_envs, 3, device=self.device)
            cmd_ang_local[:, 1] = self._commands[:, 2]
            cmd_ang_world = quat_apply(robot_quat, cmd_ang_local)
            
            yaw_ang = torch.atan2(cmd_ang_world[:, 1], cmd_ang_world[:, 0])
            ang_arrow_quat = quat_from_angle_axis(yaw_ang, axis_z)

            ang_mag = torch.abs(self._commands[:, 2])
            ang_arrow_scale = torch.zeros(self.num_envs, 3, device=self.device)
            ang_arrow_scale[:, 0] = torch.clamp(ang_mag, min=0.1) * 1.5 
            ang_arrow_scale[:, 1] = 0.5 
            ang_arrow_scale[:, 2] = 0.5 

            self.lin_visualizer.visualize(
                robot_pos + torch.tensor([0,0,1.0], device=self.device), 
                lin_arrow_quat, 
                scales=lin_arrow_scale
            )
            self.ang_visualizer.visualize(
                robot_pos + torch.tensor([0,0,1.2], device=self.device), 
                ang_arrow_quat, 
                scales=ang_arrow_scale
            )

            # 2. Contact Markers
            raw_forces = self._contact_sensor.data.net_forces_w_history[:, :, self._undesired_sensor_ids]
            force_magnitudes = torch.max(torch.norm(raw_forces, dim=-1), dim=1)[0]
            contact_mask = force_magnitudes > 1.0
            
            undesired_body_pos = self._robot.data.body_pos_w[:, self._undesired_actor_ids, :]
            active_contact_pos = undesired_body_pos[contact_mask]
            
            if active_contact_pos.shape[0] > 0:
                self.contact_visualizer.visualize(active_contact_pos)
                self.contact_visualizer.set_visibility(True)
            else:
                self.contact_visualizer.set_visibility(False)

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        
        height_data = (
            self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - 
            self._height_scanner.data.ray_hits_w[..., 2] - 
            0.5
        ).clip(-1.0, 1.0)
        
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self._robot.data.root_lin_vel_b,
                    self._robot.data.root_ang_vel_b,
                    self._robot.data.projected_gravity_b,
                    self._commands,
                    self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                    self._robot.data.joint_vel,
                    height_data,
                    self._actions,
                )
                if tensor is not None
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # -- Tracking --
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        
        yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        
        # -- Penalties --
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        
        # Action Acceleration (2nd Derivative)
        # (Current - Prev) - (Prev - PrevPrev) = Current - 2*Prev + PrevPrev
        action_accel = torch.sum(torch.square(
            self._actions - 2*self._action_history[:, 0] + self._action_history[:, 1]
        ), dim=1)

        # -- Gait / Contacts --
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        
        # [!FIX] "Moving" = Lin > 0.1 OR Ang > 0.1
        # This ensures we don't penalize joint movement during rotation
        is_moving = (torch.norm(self._commands[:, :2], dim=1) > 0.01) | (torch.abs(self._commands[:, 2]) > 0.01)
        
        # Apply air time reward only if commanded to move
        air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1) * is_moving.float()
        
        # Undesired Contacts (Using SENSOR indices)
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self._undesired_sensor_ids], dim=-1), dim=1)[0] > 1.0
        )
        contacts = torch.sum(is_contact, dim=1)
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)

        # -- Geometric Base Height --
        # Robust calculation: Base Z - Mean Feet Z
        root_z = self._robot.data.root_pos_w[:, 2]
        feet_z = self._robot.data.body_pos_w[:, self._feet_ids, 2]
        terrain_height = torch.mean(feet_z, dim=1)
        current_base_height = root_z - terrain_height
        base_height_error = torch.square(current_base_height - self.cfg.target_base_height)

        # -- Joint Regularization --
        joint_deviation = torch.sum(torch.square(self._robot.data.joint_pos - self._robot.data.default_joint_pos), dim=1)

        alive = torch.ones(self.num_envs, device=self.device)

        rewards = {
            "alive": alive * self.cfg.alive_reward_scale * self.step_dt,
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "action_accel_l2": action_accel * self.cfg.action_accel_reward_scale * self.step_dt,
            "feet_air_time": air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "undesired_contacts": contacts * self.cfg.undesired_contact_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
            "base_height_l2": base_height_error * self.cfg.base_height_reward_scale * self.step_dt,
            "joint_deviation_l2": joint_deviation * self.cfg.joint_deviation_reward_scale * self.step_dt,
        }
        
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        
        for key, value in rewards.items():
            if key not in self._episode_sums:
                self._episode_sums[key] = torch.zeros_like(value)
            self._episode_sums[key] += value
            
        return reward
        
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        
        for key, value in rewards.items():
            if key not in self._episode_sums:
                self._episode_sums[key] = torch.zeros_like(value)
            self._episode_sums[key] += value
            
        return reward
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # Terminate if Trunk hits ground (> 50N impact)
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 50.0, dim=1)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        
        if len(env_ids) == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
            
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        
        # [!NEW] Sample curriculum commands
        self._command_timer[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(4.0, 9.0)
        self._sample_commands(env_ids)
        
        # Reset State
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
        # Logging & Extra Metrics
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
            
        # [!NEW] Mode-Specific Metrics (Average tracking error per mode)
        # We calculate this only for the resetting envs to avoid noise
        # This will show up in "log/..." in TensorBoard
        
        # 1. Stand Drift (Linear Velocity when Mode=0)
        stand_mask = (self._command_modes[env_ids] == 0)
        if stand_mask.any():
            vel_mag = torch.norm(self._robot.data.root_lin_vel_b[env_ids][stand_mask][:, :2], dim=1)
            extras["Metrics/Drift_Vel_Stand"] = torch.mean(vel_mag)
            
        # 2. Walk Tracking (Lin Vel Error when Mode=2 or 4)
        move_mask = (self._command_modes[env_ids] >= 2)
        if move_mask.any():
            lin_err = torch.norm(self._commands[env_ids][move_mask][:, :2] - self._robot.data.root_lin_vel_b[env_ids][move_mask][:, :2], dim=1)
            extras["Metrics/Tracking_Err_Move"] = torch.mean(lin_err)

        self.extras["log"] = dict()
        self.extras["log"].update(extras)