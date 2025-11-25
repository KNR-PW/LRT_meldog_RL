# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
# [!CHANGED] Removed 'import omni.debugdraw' to fix headless crash
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.scene import InteractiveScene
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_from_angle_axis
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.spawners.lights import DomeLightCfg
from isaaclab.markers import VisualizationMarkers

from .meldog_simple_locomotion_policy_env_cfg import MeldogSimpleLocomotionPolicyEnvCfg

@torch.jit.script
def quat_to_euler_xyz(quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert quaternions to Euler angles (XYZ convention)."""
    w, x, y, z = quat.unbind(dim=-1)
    
    # Roll
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch
    sinp = 2 * (w * y - z * x)
    pitch = torch.where(torch.abs(sinp) >= 1, torch.sign(sinp) * torch.pi / 2, torch.asin(sinp))

    # Yaw
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

class MeldogSimpleLocomotionPolicyEnv(DirectRLEnv):
    """
    DirectRLEnv class for the Meldog simple locomotion task.
    """
    cfg: MeldogSimpleLocomotionPolicyEnvCfg

    def __init__(self, cfg: MeldogSimpleLocomotionPolicyEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # [!CHANGED] Removed self._debug_draw initialization

        # -- Robot Data --
        self.default_joint_pos = self.robot.data.default_joint_pos.clone()
        self.default_root_state = self.robot.data.default_root_state.clone()
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        self.root_state = self.robot.data.root_state_w
        
        # -- Task-Specific Buffers --
        self.last_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.base_lin_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.base_ang_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.gravity_vec = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Command buffers and Timers
        self.commands = torch.zeros(self.num_envs, 3, device=self.device) # x, y, ang_z
        self.command_timer = torch.zeros(self.num_envs, device=self.device)

        # -- Key Indices --
        self.base_link_idx, _ = self.robot.find_bodies(self.cfg.params.base_link_name)
        self.foot_link_indices, _ = self.robot.find_bodies(self.cfg.params.foot_link_names)
        self.actuated_joint_indices = self.robot.find_joints(self.robot.actuators["all_joints"].cfg.joint_names_expr)[0]


    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        light_cfg = DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        
        self.foot_contact_sensor = ContactSensor(
            cfg=ContactSensorCfg(
                prim_path="/World/envs/env_.*/Robot/meldog_core/.*F_link",
                filter_prim_paths_expr=["/World/ground"]
            ),
        )

        self.base_contact_sensor = ContactSensor(
             cfg=ContactSensorCfg(
                 prim_path=f"/World/envs/env_.*/Robot/meldog_core/{self.cfg.params.base_link_name}",
                 filter_prim_paths_expr=["/World/ground"]
             ),
         )

        # Initialize separate visualizers
        self.lin_visualizer = VisualizationMarkers(self.cfg.lin_vel_marker)
        self.ang_visualizer = VisualizationMarkers(self.cfg.ang_vel_marker)

        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["foot_contact_sensor"] = self.foot_contact_sensor
        self.scene.sensors["base_contact_sensor"] = self.base_contact_sensor


    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        
        # --- Handle Command Resampling ---
        dt = self.cfg.sim.dt * self.cfg.decimation
        self.command_timer += dt
        
        resample_ids = (self.command_timer >= self.cfg.params.command_resampling_time).nonzero(as_tuple=False).flatten()
        if len(resample_ids) > 0:
            self._resample_commands(resample_ids)
            self.command_timer[resample_ids] = 0.0

        # --- Update Visualization (Only if enabled) ---
        if self.cfg.params.debug_vis:
            robot_pos = self.robot.data.root_pos_w
            robot_quat = self.robot.data.root_quat_w
            
            # 1. Visualize Linear Velocity (Red Arrow)
            # ---------------------------------------
            cmd_lin_local = torch.zeros(self.num_envs, 3, device=self.device)
            cmd_lin_local[:, 0] = self.commands[:, 0] # vx
            cmd_lin_local[:, 1] = self.commands[:, 1] # vy
            
            cmd_lin_world = quat_apply(robot_quat, cmd_lin_local)
            
            yaw_lin = torch.atan2(cmd_lin_world[:, 1], cmd_lin_world[:, 0])
            zeros = torch.zeros_like(yaw_lin)
            axis_z = torch.stack([zeros, zeros, torch.ones_like(yaw_lin)], dim=-1)
            lin_arrow_quat = quat_from_angle_axis(yaw_lin, axis_z)

            # Calculate Dynamic Scale for Linear Arrow
            lin_mag = torch.norm(self.commands[:, :2], dim=1)
            lin_arrow_scale = torch.zeros(self.num_envs, 3, device=self.device)
            lin_arrow_scale[:, 0] = torch.clamp(lin_mag, min=0.1) 
            lin_arrow_scale[:, 1] = 0.5  # Width (Thicker)
            lin_arrow_scale[:, 2] = 0.5  # Height (Thicker)
            
            # 2. Visualize Angular Velocity (Green Arrow)
            # ---------------------------------------
            ang_cmd_z = self.commands[:, 2]
            cmd_ang_local = torch.zeros(self.num_envs, 3, device=self.device)
            cmd_ang_local[:, 1] = ang_cmd_z 
            
            cmd_ang_world = quat_apply(robot_quat, cmd_ang_local)
            
            yaw_ang = torch.atan2(cmd_ang_world[:, 1], cmd_ang_world[:, 0])
            ang_arrow_quat = quat_from_angle_axis(yaw_ang, axis_z)

            # Calculate Dynamic Scale for Angular Arrow
            ang_mag = torch.abs(ang_cmd_z)
            ang_arrow_scale = torch.zeros(self.num_envs, 3, device=self.device)
            ang_arrow_scale[:, 0] = torch.clamp(ang_mag, min=0.1) * 1.5 
            ang_arrow_scale[:, 1] = 0.5  # Width (Thicker)
            ang_arrow_scale[:, 2] = 0.5  # Height (Thicker)

            # 3. Apply Visualization
            # ----------------------
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
         
    def _apply_action(self) -> None:
        """Apply actions to the robot using position control."""
        action_scale = 0.5 
        current_targets = self.default_joint_pos + (self.actions * action_scale)
        current_targets = torch.clamp(current_targets, -3.14, 3.14)
        self.robot.set_joint_position_target(current_targets, joint_ids=self.actuated_joint_indices)

    def _get_observations(self) -> dict:
        self.root_state = self.robot.data.root_state_w
        
        base_lin_vel_world = self.root_state[:, 7:10]
        base_ang_vel_world = self.root_state[:, 10:13]
        base_quat = self.root_state[:, 3:7]
        
        # Transform velocities and gravity to base frame
        self.base_lin_vel = quat_apply_inverse(base_quat, base_lin_vel_world)
        self.base_ang_vel = quat_apply_inverse(base_quat, base_ang_vel_world)
        
        gravity_world = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)
        self.gravity_vec = quat_apply_inverse(base_quat, gravity_world)
        
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        
        obs_buf = torch.cat([
            self.base_lin_vel,
            self.base_ang_vel,
            self.gravity_vec,
            self.joint_pos,
            self.joint_vel,
            self.last_actions,
            self.commands
        ], dim=-1)

        return {"policy": obs_buf, "critic": obs_buf}


    def _get_rewards(self) -> torch.Tensor:
        rew_cfg = self.cfg.params.RewScale
        
        # Reward Calculation based on Dynamic Commands
        
        # 1. Track linear velocity (XY)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        rew_lin_vel_xy = torch.exp(-lin_vel_error / 0.25)
        
        # 2. Track angular velocity Z
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        rew_ang_vel_z = torch.exp(-ang_vel_error / 0.25)
        
        # 3. Penalties
        rew_lin_vel_z = torch.square(self.base_lin_vel[:, 2])
        rew_ang_vel_xy = torch.sum(torch.square(self.base_ang_vel[:, 0:2]), dim=-1)
        rew_dof_vel = torch.sum(torch.square(self.joint_vel), dim=-1)
        rew_action_rate = torch.sum(torch.square(self.last_actions - self.actions), dim=-1)
        
        # 4. Joint Limits (Soft Limits)
        soft_limit_threshold = 0.0 
        deviation = torch.abs(self.joint_pos - self.default_joint_pos)
        violation = torch.maximum(deviation - soft_limit_threshold, torch.tensor(0.0, device=self.device))
        rew_dof_pos_limits = torch.sum(torch.square(violation), dim=-1)

        # 5. Survival
        rew_alive = torch.ones_like(rew_lin_vel_xy)

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

        total_reward = torch.where(self.reset_terminated, -rew_cfg.termination, total_reward)

        return total_reward

    def _resample_commands(self, env_ids: Sequence[int]):
        """Randomly sample commands for the specified environments."""
        r = self.cfg.params.Commands.Ranges
        
        # Sample x velocity
        self.commands[env_ids, 0] = torch.rand(len(env_ids), device=self.device) * (r.lin_vel_x[1] - r.lin_vel_x[0]) + r.lin_vel_x[0]
        # Sample y velocity
        self.commands[env_ids, 1] = torch.rand(len(env_ids), device=self.device) * (r.lin_vel_y[1] - r.lin_vel_y[0]) + r.lin_vel_y[0]
        # Sample ang z velocity
        self.commands[env_ids, 2] = torch.rand(len(env_ids), device=self.device) * (r.ang_vel_z[1] - r.ang_vel_z[0]) + r.ang_vel_z[0]


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        term_cfg = self.cfg.params.Terminations
        
        # -- Timeouts --
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # -- Terminations --
        root_pos = self.robot.data.root_state_w[:, 0:3]
        root_quat = self.robot.data.root_state_w[:, 3:7]
        
        # Check height
        base_height = root_pos[:, 2]
        termination_height = self.default_root_state[0, 2] - 0.3
        fell_over = base_height < termination_height
        
        # Check orientation
        if term_cfg.reset_robot_on_bad_orientation:
            roll, pitch, _ = quat_to_euler_xyz(root_quat)
            bad_orientation = (torch.abs(roll) > term_cfg.max_roll_pitch_rad) | \
                              (torch.abs(pitch) > term_cfg.max_roll_pitch_rad)
            fell_over = fell_over | bad_orientation

        # Check base contact
        if term_cfg.reset_robot_on_base_contact:
            net_forces = self.base_contact_sensor.data.net_forces_w
            force_magnitudes = torch.norm(net_forces, dim=-1)
            base_contact = torch.any(force_magnitudes > 1.0, dim=-1)
            fell_over = fell_over | base_contact

        # Check joint limits
        if term_cfg.reset_robot_on_joint_limits:
            joint_limits_exceeded = torch.any(
                torch.abs(self.joint_pos - self.default_joint_pos) > 0.5,
                dim=-1
            )
            terminated = fell_over | joint_limits_exceeded
        else:
            terminated = fell_over

        # -- Logging --
        if not hasattr(self, "extras"): self.extras = {}
        
        self.extras["log"] = {
            "Episode/Vel_Linear_X": torch.mean(self.base_lin_vel[:, 0]),
            "Episode/Base_Height": torch.mean(self.root_state[:, 2]),
            "Episode/Action_Rate": torch.mean(torch.norm(self.actions - self.last_actions, dim=-1)),
            "Episode/Torque_Estimate": torch.mean(torch.norm(self.actions, dim=-1))
        }

        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        
        # Reset state
        root_state = self.default_root_state[env_ids]
        root_state[:, :3] += self.scene.env_origins[env_ids]
        
        joint_pos = self.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(self.joint_vel[env_ids])

        self.robot.write_root_state_to_sim(root_state, env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Reset buffers
        self.last_actions[env_ids] = 0.0
        
        # Reset commands and timers
        self._resample_commands(env_ids)
        self.command_timer[env_ids] = 0.0
        
        super()._reset_idx(env_ids)