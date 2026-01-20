# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Meldog locomotion environment.

This is the main environment class used for all locomotion and dataset
collection tasks. The behavior is controlled by the configuration passed in.
"""

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors import ContactSensor, RayCaster, TiledCamera
from isaaclab.utils.math import quat_apply, quat_from_angle_axis

from .configs import BaseMeldogEnvCfg


class MeldogEnv(DirectRLEnv):
    """Meldog quadruped locomotion environment.
    
    This environment supports:
    - Multiple terrain types (flat, rough, obstacles)
    - Optional camera sensors for dataset collection
    - Domain randomization for sim-to-real transfer
    
    The specific behavior is controlled by the configuration class passed in.
    """
    
    cfg: BaseMeldogEnvCfg

    def __init__(self, cfg: BaseMeldogEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Action buffers
        self._actions = torch.zeros(
            self.num_envs, 
            gym.spaces.flatdim(self.single_action_space), 
            device=self.device
        )
        self._previous_actions = torch.zeros_like(self._actions)
        
        # Command buffer [vx, vy, yaw_rate]
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # Reward logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "feet_air_time",
                "undesired_contacts",
                "flat_orientation_l2",
            ]
        }

        # Find body indices for rewards/termination
        self._base_id, _ = self._contact_sensor.find_bodies("trunk_link")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*F_link")
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(".*(H|UL|LL)_link")
        self._undesired_actor_ids, _ = self._robot.find_bodies(".*(H|UL|LL)_link")

    def _setup_scene(self):
        """Set up the simulation scene with robot, sensors, and terrain."""
        
        # 1. Robot
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # 2. Core sensors (always present)
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        self._height_scanner = RayCaster(self.cfg.height_scanner)
        self.scene.sensors["height_scanner"] = self._height_scanner

        # GT scanner for dataset collection
        if hasattr(self.cfg, "gt_scanner") and self.cfg.gt_scanner is not None:
            self._gt_scanner = RayCaster(self.cfg.gt_scanner)
            self.scene.sensors["gt_scanner"] = self._gt_scanner
        else:
            self._gt_scanner = None

        # 3. Cameras (conditional based on config)
        self._cameras = {}
        camera_configs = [
            ("front", "tiled_camera_front"),
            ("rear", "tiled_camera_rear"),
            ("left", "tiled_camera_left"),
            ("right", "tiled_camera_right"),
            ("top", "tiled_camera_top"),
        ]
        
        for name, cfg_attr in camera_configs:
            cfg_value = getattr(self.cfg, cfg_attr, None)
            if cfg_value is not None:
                camera = TiledCamera(cfg_value)
                self.scene.sensors[cfg_attr] = camera
                self._cameras[name] = camera
            else:
                self._cameras[name] = None

        # 4. Terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # 5. Clone environments and filter collisions
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # 6. Lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # 7. Visualization markers
        self._lin_visualizer = VisualizationMarkers(self.cfg.lin_vel_marker)
        self._ang_visualizer = VisualizationMarkers(self.cfg.ang_vel_marker)
        self._contact_visualizer = VisualizationMarkers(self.cfg.contact_marker)

    def _pre_physics_step(self, actions: torch.Tensor):
        """Process actions before physics simulation."""
        self._actions = actions.clone()
        self._processed_actions = (
            self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos
        )

        # Debug visualization
        if self.cfg.debug_vis:
            self._visualize_commands()

    def _visualize_commands(self):
        """Visualize velocity commands and contact forces."""
        robot_pos = self._robot.data.root_pos_w
        robot_quat = self._robot.data.root_quat_w

        # Linear velocity command arrow (red)
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

        # Angular velocity command arrow (green)
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

        self._lin_visualizer.visualize(
            robot_pos + torch.tensor([0, 0, 1.0], device=self.device),
            lin_arrow_quat,
            scales=lin_arrow_scale,
        )
        self._ang_visualizer.visualize(
            robot_pos + torch.tensor([0, 0, 1.2], device=self.device),
            ang_arrow_quat,
            scales=ang_arrow_scale,
        )

        # Contact markers (red spheres on undesired contacts)
        raw_forces = self._contact_sensor.data.net_forces_w_history[
            :, :, self._undesired_contact_body_ids
        ]
        force_magnitudes = torch.max(torch.norm(raw_forces, dim=-1), dim=1)[0]
        contact_mask = force_magnitudes > 1.0

        undesired_body_pos = self._robot.data.body_pos_w[:, self._undesired_actor_ids, :]
        active_contact_pos = undesired_body_pos[contact_mask]

        if active_contact_pos.shape[0] > 0:
            self._contact_visualizer.visualize(active_contact_pos)
            self._contact_visualizer.set_visibility(True)
        else:
            self._contact_visualizer.set_visibility(False)

    def _apply_action(self):
        """Apply processed actions to the robot."""
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        """Compute observations for the policy."""
        self._previous_actions = self._actions.clone()

        # Height map from raycaster
        height_data = (
            self._height_scanner.data.pos_w[:, 2].unsqueeze(1)
            - self._height_scanner.data.ray_hits_w[..., 2]
            - 0.5
        ).clip(-1.0, 1.0)

        # Concatenate observation vector
        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,      # 3
                self._robot.data.root_ang_vel_b,      # 3
                self._robot.data.projected_gravity_b, # 3
                self._commands,                        # 3
                self._robot.data.joint_pos - self._robot.data.default_joint_pos,  # 12
                self._robot.data.joint_vel,           # 12
                height_data,                          # 187 (17x11)
                self._actions,                        # 12
            ],
            dim=-1,
        )
        
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards."""
        # Tracking rewards
        lin_vel_error = torch.sum(
            torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]),
            dim=1,
        )
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)

        yaw_rate_error = torch.square(
            self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2]
        )
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)

        # Penalty terms
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        ang_vel_error = torch.sum(
            torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1
        )
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        action_rate = torch.sum(
            torch.square(self._actions - self._previous_actions), dim=1
        )

        # Gait rewards
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[
            :, self._feet_ids
        ]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        air_time = torch.sum(
            (last_air_time - self.cfg.feet_air_time) * first_contact, dim=1
        ) * (torch.norm(self._commands[:, :2], dim=1) > 0.1)

        # Undesired contacts
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = (
            torch.max(
                torch.norm(
                    net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1
                ),
                dim=1,
            )[0]
            > 1.0
        )
        contacts = torch.sum(is_contact, dim=1)

        # Orientation penalty
        flat_orientation = torch.sum(
            torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1
        )

        # Compute scaled rewards
        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "feet_air_time": air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "undesired_contacts": contacts * self.cfg.undesired_contact_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Log episode sums
        for key, value in rewards.items():
            self._episode_sums[key] += value

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check termination conditions."""
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Terminate if trunk hits ground
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died = torch.any(
            torch.max(
                torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1
            )[0]
            > 1.0,
            dim=1,
        )

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset environments."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # Randomize episode length on full reset
        if len(env_ids) == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )

        # Reset buffers
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0

        # Sample new commands
        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(
            -1.0, 1.0
        )

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Log episode rewards
        extras = {}
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0

        self.extras["log"] = {}
        self.extras["log"].update(extras)

    # =========================================================================
    # Public accessors for dataset collection
    # =========================================================================
    
    @property
    def robot(self) -> Articulation:
        """Access robot articulation."""
        return self._robot
    
    @property
    def cameras(self) -> dict[str, TiledCamera | None]:
        """Access camera sensors."""
        return self._cameras
    
    @property
    def gt_scanner(self) -> RayCaster | None:
        """Access ground truth height scanner."""
        return self._gt_scanner
    
    @property
    def height_scanner(self) -> RayCaster:
        """Access height scanner for locomotion."""
        return self._height_scanner
    
    @property
    def terrain(self):
        """Access terrain."""
        return self._terrain
    
    @property
    def commands(self) -> torch.Tensor:
        """Access current velocity commands."""
        return self._commands
