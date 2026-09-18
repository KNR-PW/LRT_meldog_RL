# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""One interface to the locomotion envs the evaluator records.

``DirectEnvAdapter`` covers direct-workflow envs (Meldog, Isaac Lab's direct ANYmal-C).
``ManagerEnvAdapter`` covers Isaac Lab manager-based velocity tasks (ANYmal-B/C/D,
Unitree Go1/Go2/A1, Spot and extensions built the same way). The evaluator only talks
to the adapter, so every robot is recorded by the same code.

Usage: ``adapter = make_adapter(task)``, then ``adapter.configure_cfg(...)`` before
``gym.make``, then ``adapter.bind(env.unwrapped)``.
"""

from __future__ import annotations

import re

import gymnasium as gym
import torch

from isaaclab.utils.math import quat_apply_inverse

from .robot_specs import RobotSpec, canonical_order, find_robot_spec

MANAGER_ENTRY_POINT = "isaaclab.envs:ManagerBasedRLEnv"


def is_manager_based(task: str) -> bool:
    """True if the task is registered as an Isaac Lab manager-based RL env."""
    return gym.spec(task).entry_point == MANAGER_ENTRY_POINT


def load_cfgs(task: str):
    """Instantiate the env and RSL-RL runner configs registered for ``task``.

    Handles both registry styles: a config class (Meldog) and a ``"module:Class"``
    string (Isaac Lab).
    """
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    env_cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    return env_cfg, agent_cfg


def make_adapter(task: str) -> "EnvAdapter":
    """The adapter matching how ``task`` is registered."""
    spec = find_robot_spec(task)
    if is_manager_based(task):
        return ManagerEnvAdapter(task, spec)
    return DirectEnvAdapter(task, spec)


# ---------------------------------------------------------------------------
# Terrain geometry from a height scanner (same maths as MeldogEnv's helpers)
# ---------------------------------------------------------------------------
def terrain_height_under_body(scanner, radius: float = 0.3) -> torch.Tensor:
    """Mean world z of the scanner hits within ``radius`` of the scanner origin. (N,)"""
    hits = scanner.data.ray_hits_w
    origin = scanner.data.pos_w
    finite = torch.isfinite(hits).all(dim=-1)
    dist = torch.norm(hits[..., :2] - origin[:, :2].unsqueeze(1), dim=-1)
    mask = finite & (dist <= radius)
    hit_z = torch.where(finite, hits[..., 2], torch.zeros_like(hits[..., 2]))
    count = mask.sum(dim=1)
    mean_z = (hit_z * mask).sum(dim=1) / count.clamp(min=1)
    return torch.where(count > 0, mean_z, origin[:, 2])


def terrain_normal_b(scanner, root_quat_w: torch.Tensor) -> torch.Tensor:
    """Unit normal of a plane fitted to the scanner hits, in the body frame. (N, 3)"""
    hits = scanner.data.ray_hits_w
    origin = scanner.data.pos_w
    finite = torch.isfinite(hits).all(dim=-1)
    w = finite.float()
    dx = torch.where(finite, hits[..., 0] - origin[:, 0:1], torch.zeros_like(w))
    dy = torch.where(finite, hits[..., 1] - origin[:, 1:2], torch.zeros_like(w))
    z0 = terrain_height_under_body(scanner)
    dz = torch.where(finite, hits[..., 2] - z0.unsqueeze(1), torch.zeros_like(w))
    sxx, sxy, syy = (w * dx * dx).sum(1), (w * dx * dy).sum(1), (w * dy * dy).sum(1)
    sx, sy, s1 = (w * dx).sum(1), (w * dy).sum(1), w.sum(1)
    rx, ry, rz = (w * dx * dz).sum(1), (w * dy * dz).sum(1), (w * dz).sum(1)
    mat = torch.stack(
        [
            torch.stack([sxx, sxy, sx], dim=-1),
            torch.stack([sxy, syy, sy], dim=-1),
            torch.stack([sx, sy, s1], dim=-1),
        ],
        dim=-2,
    )
    eye = torch.eye(3, device=mat.device, dtype=mat.dtype).expand_as(mat)
    coeffs = torch.linalg.solve(mat + 1.0e-6 * eye, torch.stack([rx, ry, rz], dim=-1))
    normal_w = torch.stack([-coeffs[:, 0], -coeffs[:, 1], torch.ones_like(coeffs[:, 0])], dim=-1)
    normal_w = normal_w / torch.norm(normal_w, dim=-1, keepdim=True).clamp(min=1.0e-6)
    return quat_apply_inverse(root_quat_w, normal_w)


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------
class EnvAdapter:
    """Common part of both adapters. Subclasses fill in the env-specific accessors."""

    def __init__(self, task: str, spec: RobotSpec):
        self.task = task
        self.spec = spec
        self.env = None

    # -- before env creation ------------------------------------------------
    def configure_cfg(self, env_cfg, num_envs: int, seed: int, device: str, enable_cameras: bool,
                      benchmark: bool) -> None:
        env_cfg.scene.num_envs = num_envs
        env_cfg.seed = seed
        env_cfg.sim.device = device

    def apply_benchmark(self, env_cfg, profile: str, bench_terrain: str) -> list[str]:
        """Benchmark conditions (see ``profiles.py``) and optional shared terrain. Returns a log."""
        from . import profiles
        from .benchmark_terrains import make_terrain_cfg
        from .robot_specs import NOMINAL_MASS_KG

        log = profiles.apply_clean_events(env_cfg)
        log += self._disable_noise_and_curriculum(env_cfg)
        sensor_cfg = self.contact_sensor_cfg(env_cfg)
        sensor_cfg.history_length = int(env_cfg.decimation)
        log.append(f"contact sensor history = decimation = {env_cfg.decimation}")
        if bench_terrain != "task":
            holder, attr = self.terrain_cfg_holder(env_cfg)
            setattr(holder, attr, make_terrain_cfg(bench_terrain))
            log.append(f"terrain replaced by shared benchmark terrain '{bench_terrain}'")
        if profile == "real":
            log += profiles.apply_real_events(env_cfg, self, NOMINAL_MASS_KG.get(self.spec.name))
            log += self._enable_real_obs_noise(env_cfg)
        return log

    def assign_terrain_cells(self, cells) -> None:
        """Put env i on terrain cell (row, column) ``cells[i]``; takes effect at the next reset."""
        terrain = self.terrain
        levels = torch.tensor([c[0] for c in cells], device=terrain.terrain_origins.device)
        types = torch.tensor([c[1] for c in cells], device=terrain.terrain_origins.device)
        terrain.terrain_levels[:] = levels
        terrain.terrain_types[:] = types
        terrain.env_origins[:] = terrain.terrain_origins[levels, types]

    def scale_effort_limits(self, factors) -> None:
        """Multiply every actuator's effort limit per env (``factors``: (E,) tensor).

        Works for DC motors, actuator networks and lookup-table actuators, because they all clip
        their output at ``effort_limit``.
        """
        for actuator in self.robot.actuators.values():
            limit = actuator.effort_limit
            if limit is None:
                continue
            actuator.effort_limit = limit * factors.to(limit.device).view(-1, 1)

    def base_sensor_ids(self):
        """Contact-sensor ids of the trunk body (empty if the spec's name does not match)."""
        ids, _ = self.contact_sensor.find_bodies(self.spec.base_body)
        return ids

    # -- after env creation --------------------------------------------------
    def bind(self, raw_env) -> None:
        self.env = raw_env

    @property
    def robot(self):
        raise NotImplementedError

    @property
    def contact_sensor(self):
        raise NotImplementedError

    @property
    def terrain(self):
        raise NotImplementedError

    @property
    def height_scanner(self):
        raise NotImplementedError

    def get_commands(self) -> torch.Tensor:
        raise NotImplementedError

    def set_commands(self, cmd: torch.Tensor) -> None:
        raise NotImplementedError

    def freeze_command_resampling(self) -> None:
        raise NotImplementedError

    # -- shared --------------------------------------------------------------
    @property
    def step_dt(self) -> float:
        return float(self.env.step_dt)

    def _ordered_ids(self, asset, regex: str, find) -> tuple[list[int], list[str]]:
        ids, names = find(regex)
        perm = canonical_order(list(names), self.spec.name_order)
        if perm is None:
            raise RuntimeError(
                f"[{self.spec.name}] '{regex}' matched {list(names)} in {type(asset).__name__}, "
                "which does not map to FL/FR/RL/RR. Fix the robot spec."
            )
        return [ids[i] for i in perm], [names[i] for i in perm]

    def foot_ids(self):
        """(sensor ids, robot body ids, robot body names), each in FL, FR, RL, RR order."""
        sensor_ids, _ = self._ordered_ids(self.contact_sensor, self.spec.foot_regex,
                                          self.contact_sensor.find_bodies)
        robot_ids, robot_names = self._ordered_ids(self.robot, self.spec.foot_regex, self.robot.find_bodies)
        return sensor_ids, robot_ids, robot_names

    def hip_body_ids(self):
        """Robot body ids of the hip-flexion (thigh) bodies in FL, FR, RL, RR order."""
        ids, _ = self._ordered_ids(self.robot, self.spec.hip_regex, self.robot.find_bodies)
        return ids

    def shank_body_ids(self):
        """Robot body ids of the shank (lower-leg) bodies in FL, FR, RL, RR order."""
        ids, _ = self._ordered_ids(self.robot, self.spec.shank_regex, self.robot.find_bodies)
        return ids

    def leg_segment_lengths(self):
        """(thigh, shank) lengths per leg in FL, FR, RL, RR order, in metres.

        Distances between consecutive joint frames (hip-flexion body, shank body, foot body)
        of env 0. They do not depend on the pose, so any simulation step works.
        """
        pos = self.robot.data.body_pos_w[0]
        _, foot_ids, _ = self.foot_ids()
        hip, shank = self.hip_body_ids(), self.shank_body_ids()
        thigh_len = torch.norm(pos[shank] - pos[hip], dim=-1)
        shank_len = torch.norm(pos[foot_ids] - pos[shank], dim=-1)
        return thigh_len.cpu(), shank_len.cpu()

    def leg_joint_ids(self):
        """Joint ids grouped per leg in FL, FR, RL, RR order (None if the names do not group).

        Uses the same leg-label rule as the feet, so it works for every robot in ``robot_specs``.
        """
        from .robot_specs import FOOT_ORDER, leg_label

        groups = {label: [] for label in FOOT_ORDER}
        for idx, name in enumerate(self.robot.data.joint_names):
            label = leg_label(name, self.spec.name_order)
            if label in groups:
                groups[label].append(idx)
        counts = {len(v) for v in groups.values()}
        if len(counts) != 1 or counts == {0}:
            return None
        return [groups[label] for label in FOOT_ORDER]

    def knee_joint_ids(self):
        """Joint ids of the knees in FL, FR, RL, RR order."""
        ids, _ = self._ordered_ids(self.robot, self.spec.knee_regex, self.robot.find_joints)
        return ids

    def joint_limits(self):
        """Per-joint (effort limit, velocity limit) tensors of shape (J,) from the actuator models.

        Joints whose actuator has no fixed limit (e.g. Spot's knees) get ``inf``.
        """
        num_joints = self.robot.num_joints
        effort = torch.full((num_joints,), float("inf"))
        velocity = torch.full((num_joints,), float("inf"))
        for actuator in self.robot.actuators.values():
            ids = actuator.joint_indices
            if isinstance(ids, slice):
                ids = list(range(num_joints))[ids]
            effort[ids] = actuator.effort_limit[0].detach().float().cpu()
            velocity[ids] = actuator.velocity_limit[0].detach().float().cpu()
        return effort, velocity

    def has_posture(self) -> bool:
        return self.height_scanner is not None

    def terrain_height_under_body(self) -> torch.Tensor:
        return terrain_height_under_body(self.height_scanner)

    def terrain_normal_b(self) -> torch.Tensor:
        return terrain_normal_b(self.height_scanner, self.robot.data.root_quat_w)


class DirectEnvAdapter(EnvAdapter):
    """Direct-workflow envs with ``_robot``, ``_contact_sensor``, ``_terrain``, ``_commands``."""

    def configure_cfg(self, env_cfg, num_envs, seed, device, enable_cameras, benchmark):
        super().configure_cfg(env_cfg, num_envs, seed, device, enable_cameras, benchmark)
        if not enable_cameras:
            for name in ("tiled_camera_front", "tiled_camera_rear", "tiled_camera_left",
                         "tiled_camera_right", "tiled_camera_top"):
                if hasattr(env_cfg, name):
                    setattr(env_cfg, name, None)

    def terrain_cfg_holder(self, env_cfg):
        return env_cfg, "terrain"

    def contact_sensor_cfg(self, env_cfg):
        return env_cfg.contact_sensor

    def _disable_noise_and_curriculum(self, env_cfg):
        log = []
        for flag in ("obs_noise", "reset_randomization", "enable_curriculum"):
            if getattr(env_cfg, flag, False):
                setattr(env_cfg, flag, False)
                log.append(f"cfg.{flag} = False")
        for model in ("observation_noise_model", "action_noise_model"):
            if getattr(env_cfg, model, None) is not None:
                setattr(env_cfg, model, None)
                log.append(f"cfg.{model} = None")
        return log

    def _enable_real_obs_noise(self, env_cfg):
        if hasattr(env_cfg, "obs_noise"):
            env_cfg.obs_noise = True   # Meldog's inline noise uses the same magnitudes
            return ["cfg.obs_noise = True (Isaac Lab default magnitudes)"]
        return ["[WARN] this direct env has no observation-noise option; real profile without obs noise"]

    @property
    def robot(self):
        return self.env._robot

    @property
    def contact_sensor(self):
        return self.env._contact_sensor

    @property
    def terrain(self):
        return self.env._terrain

    @property
    def height_scanner(self):
        return getattr(self.env, "_height_scanner", None)

    def get_commands(self):
        return self.env._commands

    def set_commands(self, cmd):
        self.env._commands[:] = cmd

    def freeze_command_resampling(self):
        # Meldog resamples mid-episode; Isaac Lab's direct ANYmal-C only resamples on reset.
        if hasattr(self.env, "_command_time_left"):
            self.env._command_time_left[:] = 1.0e9

    def terrain_height_under_body(self):
        # Prefer the env's own helper so Meldog recordings stay bit-identical.
        if hasattr(self.env, "_terrain_height_under_body"):
            return self.env._terrain_height_under_body()
        return super().terrain_height_under_body()

    def terrain_normal_b(self):
        if hasattr(self.env, "_terrain_normal_b"):
            return self.env._terrain_normal_b()
        return super().terrain_normal_b()


class ManagerEnvAdapter(EnvAdapter):
    """Isaac Lab manager-based velocity tasks (scene entities ``robot``, ``contact_forces``)."""

    COMMAND_NAME = "base_velocity"

    def configure_cfg(self, env_cfg, num_envs, seed, device, enable_cameras, benchmark):
        super().configure_cfg(env_cfg, num_envs, seed, device, enable_cameras, benchmark)
        if benchmark:
            # Without this, heading control rewrites the yaw-rate command every step and
            # "standing" envs get zero commands, so the script would be overwritten.
            cmd = getattr(env_cfg.commands, self.COMMAND_NAME)
            cmd.heading_command = False
            cmd.rel_heading_envs = 0.0
            cmd.rel_standing_envs = 0.0
            # Extensions subclass the command term (robot_lab zeroes small commands and rewrites
            # commands on "pits" terrain); use Isaac Lab's plain term so nothing edits the script.
            from isaaclab.envs.mdp import UniformVelocityCommand

            cmd.class_type = UniformVelocityCommand

    def terrain_cfg_holder(self, env_cfg):
        return env_cfg.scene, "terrain"

    def contact_sensor_cfg(self, env_cfg):
        return env_cfg.scene.contact_forces

    def _disable_noise_and_curriculum(self, env_cfg):
        from isaaclab.managers import CurriculumTermCfg

        log = []
        env_cfg.observations.policy.enable_corruption = False
        log.append("observation noise off")
        curriculum = getattr(env_cfg, "curriculum", None)
        if curriculum is not None:
            for name, term in vars(curriculum).items():
                if isinstance(term, CurriculumTermCfg):
                    setattr(curriculum, name, None)
                    log.append(f"curriculum term '{name}' removed")
        return log

    def _enable_real_obs_noise(self, env_cfg):
        from . import profiles

        return profiles.apply_real_obs_noise(env_cfg)

    @property
    def robot(self):
        return self.env.scene["robot"]

    @property
    def contact_sensor(self):
        return self.env.scene["contact_forces"]

    @property
    def terrain(self):
        return self.env.scene.terrain

    @property
    def height_scanner(self):
        return self.env.scene.sensors.get("height_scanner")

    @property
    def _command_term(self):
        return self.env.command_manager.get_term(self.COMMAND_NAME)

    def get_commands(self):
        return self.env.command_manager.get_command(self.COMMAND_NAME)

    def set_commands(self, cmd):
        self._command_term.vel_command_b[:] = cmd

    def freeze_command_resampling(self):
        self._command_term.time_left[:] = 1.0e9


def task_tag(task: str) -> str:
    """Short folder tag for a task id, e.g. ``rough_simd1_v1`` or ``ref_go2_rough``."""
    if task.startswith("Meldog-RL-Locomotion-"):
        return task.replace("Meldog-RL-Locomotion-", "").replace("-v0", "").replace("-", "_").lower()
    spec = find_robot_spec(task)
    terrain = "flat" if "Flat" in task else "rough" if "Rough" in task else "task"
    direct = "_direct" if "Direct" in task else ""
    return f"ref_{spec.name}{direct}_{terrain}"


__all__ = [
    "DirectEnvAdapter",
    "EnvAdapter",
    "ManagerEnvAdapter",
    "is_manager_based",
    "load_cfgs",
    "make_adapter",
    "task_tag",
    "terrain_height_under_body",
    "terrain_normal_b",
]
