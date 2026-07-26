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
from isaaclab.utils.math import quat_apply, quat_from_angle_axis, quat_mul

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

        # Command resampling tracking
        self._command_time_left = torch.zeros(self.num_envs, device=self.device)

        # Curriculum learning tracking (for terrain difficulty progression)
        self._initial_robot_pos = torch.zeros(self.num_envs, 3, device=self.device)

        # Gait phase clock (Run C): per-env phase in [0, 1), advanced each policy
        # step when cfg.gait_clock is enabled. Buffer always exists (no RNG cost);
        # advancing, resetting, and observing it are all gated on the flag.
        self._gait_phase = torch.zeros(self.num_envs, device=self.device)

        # Reward logging. The original 10 terms are always logged; each V2 term is
        # only added when its scale is non-zero, so v0 configs log exactly the
        # original 10 keys (bit-identical behavior + logs).
        reward_keys = [
            "track_lin_vel_xy_exp",
            "track_ang_vel_z_exp",
            "lin_vel_z_l2",
            "ang_vel_xy_l2",
            "dof_torques_l2",
            "dof_acc_l2",
            "action_rate_l2",
        ]
        # Legacy feet_air_time is also scale-gated (V2 replaces it with
        # air_time_mode); every v0 config has a non-zero scale, so v0 logs keep
        # all original 10 keys in their original position.
        if self.cfg.feet_air_time_reward_scale != 0.0:
            reward_keys.append("feet_air_time")
        reward_keys += [
            "undesired_contacts",
            "flat_orientation_l2",
        ]
        self._active_v2_terms = [
            key
            for key, scale_attr in (
                ("foot_slip", "foot_slip_reward_scale"),
                ("gait_sync", "gait_sync_reward_scale"),
                ("air_time_variance", "air_time_variance_reward_scale"),
                ("air_time_mode", "air_time_mode_reward_scale"),
                ("foot_clearance", "foot_clearance_reward_scale"),
                ("joint_deviation_hip", "joint_deviation_hip_reward_scale"),
                ("contact_schedule", "contact_schedule_reward_scale"),
            )
            if getattr(self.cfg, scale_attr) != 0.0
        ]
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in reward_keys + self._active_v2_terms
        }

        # Find body indices for rewards/termination
        self._base_id, _ = self._contact_sensor.find_bodies("trunk_link")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*F_link")
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(".*(H|UL|LL)_link")
        self._undesired_actor_ids, _ = self._robot.find_bodies(".*(H|UL|LL)_link")

        # Feet indices canonically ordered FL, FR, RL, RR for the V2 gait terms.
        # Body names follow "<side><end>F_link" (LFF=front-left, RRF=rear-right, ...).
        # Sensor and articulation are canonicalized independently so index i always
        # refers to the same physical foot across the two data sources.
        feet_sensor_ids, feet_sensor_names = self._contact_sensor.find_bodies(".*F_link")
        feet_robot_ids, feet_robot_names = self._robot.find_bodies(".*F_link")
        sensor_perm = self._canonical_foot_perm(feet_sensor_names)
        robot_perm = self._canonical_foot_perm(feet_robot_names)
        self._feet_sensor_ids_canon = [feet_sensor_ids[i] for i in sensor_perm]
        self._feet_robot_ids_canon = [feet_robot_ids[i] for i in robot_perm]
        # Diagonal-trot pairs (synchronized): (FL, RR) and (FR, RL).
        self._gait_pair_0 = (self._feet_sensor_ids_canon[0], self._feet_sensor_ids_canon[3])
        self._gait_pair_1 = (self._feet_sensor_ids_canon[1], self._feet_sensor_ids_canon[2])

        # Hip-abduction (T) joints for the joint_deviation_hip penalty.
        self._hip_joint_ids, _ = self._robot.find_joints(".*T_joint")

    @staticmethod
    def _canonical_foot_perm(names: list[str]) -> list[int]:
        """Permutation reordering foot body names to canonical FL, FR, RL, RR.

        Body names follow ``<side><end>...F_link`` (first char L/R side, second
        char F/R front/rear). Falls back to identity if names don't map cleanly.
        """
        desired = ["FL", "FR", "RL", "RR"]

        def label(name: str) -> str:
            side = "L" if name[0].upper() == "L" else "R"
            fb = "F" if name[1].upper() == "F" else "R"
            return fb + side

        labels = [label(n) for n in names]
        if sorted(labels) == sorted(desired):
            return [labels.index(d) for d in desired]
        return list(range(len(names)))

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

        # Command resampling during episode
        self._command_time_left -= self.step_dt
        resample_envs = self._command_time_left <= 0
        if resample_envs.any():
            self._resample_commands(resample_envs.nonzero(as_tuple=False).flatten())

        # Advance the gait phase clock (Run C)
        if self.cfg.gait_clock:
            self._gait_phase = (
                self._gait_phase + self.step_dt * self.cfg.gait_clock_freq
            ) % 1.0

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

        lin_vel = self._robot.data.root_lin_vel_b
        ang_vel = self._robot.data.root_ang_vel_b
        gravity = self._robot.data.projected_gravity_b
        joint_pos = self._robot.data.joint_pos - self._robot.data.default_joint_pos
        joint_vel = self._robot.data.joint_vel

        # Height map from raycaster
        height_data = (
            self._height_scanner.data.pos_w[:, 2].unsqueeze(1)
            - self._height_scanner.data.ray_hits_w[..., 2]
            - 0.5
        )

        # Additive uniform observation noise (Run B; Go2 rough values). Noise is
        # applied before the height clip, mirroring the manager-based pipeline
        # (func -> noise -> clip). Gated so v0 consumes no RNG and is bit-identical.
        if self.cfg.obs_noise:

            def _unoise(tensor: torch.Tensor, mag: float) -> torch.Tensor:
                return tensor + torch.empty_like(tensor).uniform_(-mag, mag)

            lin_vel = _unoise(lin_vel, 0.1)
            ang_vel = _unoise(ang_vel, 0.2)
            gravity = _unoise(gravity, 0.05)
            joint_pos = _unoise(joint_pos, 0.01)
            joint_vel = _unoise(joint_vel, 1.5)
            height_data = _unoise(height_data, 0.1)

        height_data = height_data.clip(-1.0, 1.0)

        # Concatenate observation vector
        obs_parts = [
            lin_vel,                              # 3
            ang_vel,                              # 3
            gravity,                              # 3
            self._commands,                        # 3
            joint_pos,                             # 12
            joint_vel,                             # 12
            height_data,                          # 187 (17x11)
            self._actions,                        # 12
        ]

        # Gait clock observations (Run C): [sin, cos] of the phase, appended last.
        # Configs enabling this must bump observation_space by 2 (235 -> 237).
        if self.cfg.gait_clock:
            phase_angle = 2.0 * torch.pi * self._gait_phase
            obs_parts.append(torch.sin(phase_angle).unsqueeze(1))  # 1
            obs_parts.append(torch.cos(phase_angle).unsqueeze(1))  # 1

        obs = torch.cat(obs_parts, dim=-1)

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

        # Gait rewards (legacy feet_air_time; scale-gated, V2 uses air_time_mode)
        if self.cfg.feet_air_time_reward_scale != 0.0:
            first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[
                :, self._feet_ids
            ]
            last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
            if self.cfg.air_time_gate_full_cmd:
                air_time_gate = torch.norm(self._commands, dim=1) > 0.1
            else:
                air_time_gate = torch.norm(self._commands[:, :2], dim=1) > 0.1
            air_time = torch.sum(
                (last_air_time - self.cfg.feet_air_time) * first_contact, dim=1
            ) * air_time_gate

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
        }
        if self.cfg.feet_air_time_reward_scale != 0.0:
            rewards["feet_air_time"] = (
                air_time * self.cfg.feet_air_time_reward_scale * self.step_dt
            )
        rewards["undesired_contacts"] = (
            contacts * self.cfg.undesired_contact_reward_scale * self.step_dt
        )
        rewards["flat_orientation_l2"] = (
            flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt
        )

        # V2 gait-quality terms (Spot ports). Each is only computed and logged when
        # its scale is non-zero, so v0 configs produce the original 10 terms exactly.
        if "foot_slip" in self._active_v2_terms:
            rewards["foot_slip"] = (
                self._reward_foot_slip() * self.cfg.foot_slip_reward_scale * self.step_dt
            )
        if "gait_sync" in self._active_v2_terms:
            rewards["gait_sync"] = (
                self._reward_gait_sync() * self.cfg.gait_sync_reward_scale * self.step_dt
            )
        if "air_time_variance" in self._active_v2_terms:
            rewards["air_time_variance"] = (
                self._reward_air_time_variance()
                * self.cfg.air_time_variance_reward_scale
                * self.step_dt
            )
        if "air_time_mode" in self._active_v2_terms:
            rewards["air_time_mode"] = (
                self._reward_air_time_mode()
                * self.cfg.air_time_mode_reward_scale
                * self.step_dt
            )
        if "foot_clearance" in self._active_v2_terms:
            rewards["foot_clearance"] = (
                self._reward_foot_clearance() * self.cfg.foot_clearance_reward_scale * self.step_dt
            )
        if "joint_deviation_hip" in self._active_v2_terms:
            rewards["joint_deviation_hip"] = (
                self._reward_joint_deviation_hip()
                * self.cfg.joint_deviation_hip_reward_scale
                * self.step_dt
            )
        if "contact_schedule" in self._active_v2_terms:
            rewards["contact_schedule"] = (
                self._reward_contact_schedule()
                * self.cfg.contact_schedule_reward_scale
                * self.step_dt
            )

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Log episode sums
        for key, value in rewards.items():
            self._episode_sums[key] += value

        return reward

    # =========================================================================
    # V2 gait-quality reward terms (Spot ports)
    # =========================================================================

    def _reward_foot_slip(self) -> torch.Tensor:
        """Penalize planar (xy) foot velocity while the foot is in contact.

        Port of Spot ``foot_slip_penalty``. Contact is force > 1 N (max over the
        contact-force history), matching the env's other contact checks.
        """
        net_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = (
            torch.max(
                torch.norm(net_forces[:, :, self._feet_sensor_ids_canon], dim=-1), dim=1
            )[0]
            > 1.0
        )
        foot_planar_vel = torch.norm(
            self._robot.data.body_lin_vel_w[:, self._feet_robot_ids_canon, :2], dim=2
        )
        return torch.sum(is_contact * foot_planar_vel, dim=1)

    def _reward_gait_sync(self) -> torch.Tensor:
        """Enforce a diagonal trot via the Spot ``GaitReward`` product kernel.

        Product of two "sync" terms (each synced pair's air/contact times should
        match) and four "async" terms (the two diagonals should be out of phase).
        Gated on non-zero command OR body speed above the velocity threshold.
        """
        air = self._contact_sensor.data.current_air_time
        contact = self._contact_sensor.data.current_contact_time
        max_err_sq = self.cfg.gait_sync_max_err ** 2
        std = self.cfg.gait_sync_std

        def sync(f0: int, f1: int) -> torch.Tensor:
            se_air = torch.clip(torch.square(air[:, f0] - air[:, f1]), max=max_err_sq)
            se_con = torch.clip(torch.square(contact[:, f0] - contact[:, f1]), max=max_err_sq)
            return torch.exp(-(se_air + se_con) / std)

        def async_(f0: int, f1: int) -> torch.Tensor:
            se0 = torch.clip(torch.square(air[:, f0] - contact[:, f1]), max=max_err_sq)
            se1 = torch.clip(torch.square(contact[:, f0] - air[:, f1]), max=max_err_sq)
            return torch.exp(-(se0 + se1) / std)

        p0, p1 = self._gait_pair_0, self._gait_pair_1
        sync_reward = sync(p0[0], p0[1]) * sync(p1[0], p1[1])
        async_reward = (
            async_(p0[0], p1[0])
            * async_(p0[1], p1[1])
            * async_(p0[0], p1[1])
            * async_(p1[0], p0[1])
        )
        cmd = torch.norm(self._commands, dim=1)
        body_vel = torch.norm(self._robot.data.root_lin_vel_b[:, :2], dim=1)
        gate = torch.logical_or(cmd > 0.0, body_vel > self.cfg.gait_sync_vel_threshold)
        return torch.where(gate, sync_reward * async_reward, torch.zeros_like(sync_reward))

    def _reward_air_time_variance(self) -> torch.Tensor:
        """Penalize variance in per-foot air/contact durations (Spot port)."""
        last_air = self._contact_sensor.data.last_air_time[:, self._feet_sensor_ids_canon]
        last_contact = self._contact_sensor.data.last_contact_time[:, self._feet_sensor_ids_canon]
        return torch.var(torch.clip(last_air, max=0.5), dim=1) + torch.var(
            torch.clip(last_contact, max=0.5), dim=1
        )

    def _reward_air_time_mode(self) -> torch.Tensor:
        """Reward per-foot air/contact phases approaching the gait mode time.

        Port of Spot ``air_time_reward``: while moving, each foot earns its current
        phase duration (air or contact, whichever is longer) capped at ``mode_time``
        -- so every foot must keep cycling and no foot can profit from floating or
        carrying indefinitely. When commanded to stand (and slow), it instead pays
        for contact time exceeding air time. Uses the full 3-dim command norm.
        """
        mode_time = self.cfg.air_time_mode_time
        air = self._contact_sensor.data.current_air_time[:, self._feet_sensor_ids_canon]
        contact = self._contact_sensor.data.current_contact_time[:, self._feet_sensor_ids_canon]
        t_max = torch.max(air, contact)
        t_min = torch.clip(t_max, max=mode_time)
        stance_cmd_reward = torch.clip(contact - air, -mode_time, mode_time)
        cmd = torch.norm(self._commands, dim=1).unsqueeze(1).expand(-1, 4)
        body_vel = (
            torch.norm(self._robot.data.root_lin_vel_b[:, :2], dim=1).unsqueeze(1).expand(-1, 4)
        )
        reward = torch.where(
            torch.logical_or(cmd > 0.0, body_vel > self.cfg.air_time_mode_vel_threshold),
            torch.where(t_max < mode_time, t_min, torch.zeros_like(t_min)),
            stance_cmd_reward,
        )
        return torch.sum(reward, dim=1)

    def _clock_stance_schedule(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Desired stance mask (N, 4) from the gait clock, plus the standing mask.

        Diagonal trot on the clock halves: (FL, RR) in stance while phase < 0.5,
        (FR, RL) in stance while phase >= 0.5. Standing envs (command norm < 0.1
        AND body speed < 0.5) want all four feet in stance.
        """
        phase_a = self._gait_phase < 0.5
        desired_stance = torch.stack([phase_a, ~phase_a, ~phase_a, phase_a], dim=1)
        standing = (torch.norm(self._commands, dim=1) < 0.1) & (
            torch.norm(self._robot.data.root_lin_vel_b[:, :2], dim=1) < 0.5
        )
        return desired_stance, standing

    def _reward_contact_schedule(self) -> torch.Tensor:
        """Reward feet whose contact state matches the gait-clock schedule.

        Mean over the four feet of (in_contact == desired_stance), in [0, 1].
        """
        net_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = (
            torch.max(
                torch.norm(net_forces[:, :, self._feet_sensor_ids_canon], dim=-1), dim=1
            )[0]
            > 1.0
        )
        desired_stance, standing = self._clock_stance_schedule()
        desired_stance = torch.where(
            standing.unsqueeze(1), torch.ones_like(desired_stance), desired_stance
        )
        return (is_contact == desired_stance).float().mean(dim=1)

    def _reward_foot_clearance(self) -> torch.Tensor:
        """Reward swing feet clearing a target height above the terrain.

        Terrain-relative variant of Spot ``foot_clearance_reward``: foot height is
        taken above the nearest height-scanner grid point (feet lie inside the
        1.6x1.0 m yaw-aligned scan), so it is valid on rough terrain, not just flat.

        With ``cfg.gait_clock`` (Run C) the tanh(planar-speed) weighting is dropped
        (it rewarded millimeter lifts); instead, the height error counts only for
        feet in their clock swing window, and only while the command is active.
        Without the clock, the original Spot form (tanh weighting) is used.
        """
        foot_pos = self._robot.data.body_pos_w[:, self._feet_robot_ids_canon, :]
        foot_xy = foot_pos[:, :, :2]
        foot_z = foot_pos[:, :, 2]

        hits = self._height_scanner.data.ray_hits_w
        dist = torch.cdist(foot_xy, hits[:, :, :2])  # (N, 4, num_rays)
        nearest = torch.argmin(dist, dim=2)  # (N, 4)
        terrain_z = torch.gather(hits[:, :, 2], 1, nearest)  # (N, 4)
        # Missed rays are inf in ray_hits_w; argmin already prefers valid hits, but
        # guard the all-miss case so the exp kernel never sees inf/NaN.
        terrain_z = torch.where(torch.isfinite(terrain_z), terrain_z, foot_z)
        foot_height = foot_z - terrain_z

        foot_z_target_error = torch.square(foot_height - self.cfg.foot_clearance_target)
        if self.cfg.gait_clock:
            desired_stance, standing = self._clock_stance_schedule()
            swing_mask = ~desired_stance & ~standing.unsqueeze(1)
            error = torch.sum(foot_z_target_error * swing_mask.float(), dim=1)
        else:
            foot_vel_tanh = torch.tanh(
                self.cfg.foot_clearance_tanh_mult
                * torch.norm(
                    self._robot.data.body_lin_vel_w[:, self._feet_robot_ids_canon, :2], dim=2
                )
            )
            error = torch.sum(foot_z_target_error * foot_vel_tanh, dim=1)
        return torch.exp(-error / self.cfg.foot_clearance_std)

    def _reward_joint_deviation_hip(self) -> torch.Tensor:
        """L1 deviation of the hip-abduction (T) joints from their default pose."""
        deviation = (
            self._robot.data.joint_pos[:, self._hip_joint_ids]
            - self._robot.data.default_joint_pos[:, self._hip_joint_ids]
        )
        return torch.sum(torch.abs(deviation), dim=1)

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

    def _resample_commands(self, env_ids: torch.Tensor):
        """Resample velocity commands for specified environments.

        Args:
            env_ids: Environment indices to resample commands for.
        """
        num_envs = len(env_ids)

        # Determine which environments should stand still
        standing_mask = torch.rand(num_envs, device=self.device) < self.cfg.standing_env_fraction

        # Sample random commands in [-1, 1] for all 3 axes
        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-1.0, 1.0)

        # Pure-rotation environments: zero the linear command, keep the sampled yaw
        # rate. Drawn only when enabled so the default (0.0) leaves the RNG stream --
        # and thus every v0 command sequence -- bit-identical.
        if self.cfg.pure_rotation_fraction > 0.0:
            rotation_mask = torch.rand(num_envs, device=self.device) < self.cfg.pure_rotation_fraction
            if rotation_mask.any():
                self._commands[env_ids[rotation_mask], :2] = 0.0

        # Set standing environments to zero velocity (takes precedence over rotation)
        if standing_mask.any():
            self._commands[env_ids[standing_mask]] = 0.0

        # Reset command timer for resampled environments
        self._command_time_left[env_ids] = self.cfg.command_resample_time_s

    def _update_terrain_curriculum(self, env_ids: torch.Tensor):
        """Update terrain difficulty based on robot performance.

        Robots that walk far enough progress to harder terrains.
        Robots that don't meet the target distance move to easier terrains.

        This implements the same logic as Isaac Lab's terrain_levels_vel curriculum.

        Args:
            env_ids: Environment indices being reset.
        """
        # Skip on first reset (no previous position to compare)
        if torch.all(self._initial_robot_pos[env_ids] == 0):
            return

        # Calculate distance walked (XY plane only)
        distance_walked = torch.norm(
            self._robot.data.root_pos_w[env_ids, :2] - self._initial_robot_pos[env_ids, :2],
            dim=1
        )

        # Check if terrain has a procedural generator (flat terrains often don't)
        if getattr(self._terrain.cfg, "terrain_generator", None) is None:
            return

        # Get terrain size (assuming square terrains)
        terrain_size = self._terrain.cfg.terrain_generator.size[0]

        # Move to harder terrain if robot walked more than half the terrain size
        move_up = distance_walked > (terrain_size / 2)

        # Move to easier terrain if robot walked less than half the commanded distance
        # Expected distance = commanded velocity * episode duration
        commanded_velocity = torch.norm(self._commands[env_ids, :2], dim=1)
        expected_distance = commanded_velocity * self.max_episode_length_s * 0.5
        move_down = distance_walked < expected_distance
        move_down = move_down & ~move_up  # Don't move down if already moving up

        # Update terrain levels
        self._terrain.update_env_origins(env_ids, move_up, move_down)

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

        # Random initial gait phase (Run C; gated so v0 consumes no RNG on reset)
        if self.cfg.gait_clock:
            self._gait_phase[env_ids] = torch.rand(len(env_ids), device=self.device)

        # Sample new commands
        self._resample_commands(env_ids)

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        # Run B reset randomization. Inline (not a reset EventTerm) because the
        # state writes below run after super()._reset_idx() and would overwrite
        # event-based randomization. Gated so v0 consumes no RNG on reset.
        if self.cfg.reset_randomization:
            num_resets = len(env_ids)
            # Heading: random yaw in [-pi, pi) composed onto the default quat
            yaw = torch.empty(num_resets, device=self.device).uniform_(-torch.pi, torch.pi)
            axis_z = torch.zeros(num_resets, 3, device=self.device)
            axis_z[:, 2] = 1.0
            default_root_state[:, 3:7] = quat_mul(
                quat_from_angle_axis(yaw, axis_z), default_root_state[:, 3:7]
            )
            # Root planar velocity +/-0.5 m/s
            default_root_state[:, 7:9] += torch.empty(
                num_resets, 2, device=self.device
            ).uniform_(-0.5, 0.5)
            # Joint positions +/-0.1 rad around default, clamped to soft limits
            joint_pos = joint_pos + torch.empty_like(joint_pos).uniform_(-0.1, 0.1)
            soft_limits = self._robot.data.soft_joint_pos_limits[env_ids]
            joint_pos = joint_pos.clamp(soft_limits[..., 0], soft_limits[..., 1])
            # Joint velocities +/-0.5 rad/s
            joint_vel = joint_vel + torch.empty_like(joint_vel).uniform_(-0.5, 0.5)

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Update curriculum (terrain difficulty) based on performance
        if self.cfg.enable_curriculum and hasattr(self._terrain, "update_env_origins"):
            self._update_terrain_curriculum(env_ids)

        # Store initial position for next curriculum update
        self._initial_robot_pos[env_ids] = self._robot.data.root_pos_w[env_ids, :3]

        # Log episode rewards
        extras = {}
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0

        # Log curriculum progress (terrain difficulty level)
        if hasattr(self._terrain, "terrain_levels"):
            # Mean terrain level across all environments (matches Unitree's metric name)
            mean_terrain_level = torch.mean(self._terrain.terrain_levels.float())
            extras["Curriculum/terrain_levels"] = mean_terrain_level

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
