#!/usr/bin/env python3
# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate Meldog locomotion policy with statistics and rollout recording.

Runs a policy for a fixed number of episodes, reports survival statistics, and
(by default) records the full per-step, per-env rollout state to ``rollout.h5``
inside a fresh evaluation directory. The heavy, sim-free analysis of that file is
done offline by ``scripts/locomotion/analyze_locomotion.py`` (numpy/h5py only).

Usage:
    python scripts/locomotion/evaluate_locomotion.py \
        --task Meldog-RL-Locomotion-Rough-Sim-v0 \
        --checkpoint releases/locomotion/v1.0-rough/model.pt \
        --num_envs 16 --num_episodes 8
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "source"))

from isaaclab.app import AppLauncher


def str2bool(v):
    """Parse a boolean-ish CLI string."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


parser = argparse.ArgumentParser(description="Evaluate Meldog locomotion policy.")
parser.add_argument("--task", type=str, default="Meldog-RL-Locomotion-Rough-Sim-v0", help="Task name.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
parser.add_argument("--num_envs", type=int, default=100, help="Number of parallel environments.")
parser.add_argument("--num_episodes", type=int, default=100, help="Total episodes to evaluate.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--record", type=str2bool, default=True,
                    help="Record per-step rollout state to rollout.h5 (default: true).")
parser.add_argument(
    "--benchmark", action="store_true",
    help=(
        "Deterministic benchmark mode for comparable numbers across checkpoints. "
        "Forces seed=42 and drives a fixed, scripted per-episode command sequence "
        "applied EVERY step (episode-clock based, per env): "
        "0-5 s -> (vx=0.8, vy=0, wz=0); 5-10 s -> (0, 0, wz=0.8); "
        "10-15 s -> (vx=0.5, vy=0.3, 0); then the 15 s cycle repeats. "
        "DETERMINISTIC given the seed: the seed itself, the command script, and the "
        "terrain generation (seeded) are fixed and reproducible. NOT deterministic: "
        "PhysX contact/solver evaluation carries small run-to-run nondeterminism, so "
        "metrics still vary slightly between otherwise-identical runs."
    ),
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Benchmark mode pins the seed so terrain generation and resets are reproducible.
if args_cli.benchmark:
    args_cli.seed = 42

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ============================================================================

from datetime import datetime

import numpy as np
import torch
import gymnasium as gym
import h5py
from prettytable import PrettyTable

from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import meldog_rl
from meldog_rl import envs  # This registers the tasks
from meldog_rl import agents
from meldog_rl.utils import make_evaluation_dir
from meldog_rl.utils.git_utils import get_git_suffix


# Contact detection threshold (Newtons) -- matches the env's contact visualization.
CONTACT_FORCE_THRESHOLD = 1.0

# Benchmark scripted command sequence (see --benchmark help). Each tuple is
# (vx, vy, wz); the schedule below maps an episode-clock phase (s) to a command and
# repeats with period BENCHMARK_CYCLE_S. Applied every step, per env.
BENCHMARK_SCHEDULE = [
    (0.0, (0.8, 0.0, 0.0)),   # 0-5 s: walk forward
    (5.0, (0.0, 0.0, 0.8)),   # 5-10 s: turn in place
    (10.0, (0.5, 0.3, 0.0)),  # 10-15 s: diagonal walk
]
BENCHMARK_CYCLE_S = 15.0


def benchmark_commands(clock, device):
    """Scripted (vx, vy, wz) command per env from a per-env episode clock (seconds).

    ``clock`` is a (E,) tensor of episode-elapsed time; the schedule repeats every
    ``BENCHMARK_CYCLE_S``. Returns an (E, 3) command tensor on ``device``.
    """
    import torch as _torch
    phase = _torch.remainder(clock, BENCHMARK_CYCLE_S)
    cmd = _torch.zeros((clock.shape[0], 3), device=device)
    for start, vec in BENCHMARK_SCHEDULE:
        mask = phase >= start
        cmd[mask] = _torch.tensor(vec, device=device)
    return cmd


def canonical_foot_label(name: str) -> str:
    """Map a foot body name to a canonical FL/FR/RL/RR label.

    Meldog leg body names follow the convention ``<side><end>...F_link`` where the
    first character is the side (L/R) and the second is the end (F=front, R=rear),
    e.g. ``LFF_link`` -> front-left, ``RRF_link`` -> rear-right.

    Returns a two-letter label ordered front/rear + left/right (FL, FR, RL, RR).
    """
    side = name[0].upper()  # L / R
    end = name[1].upper()   # F / R
    fb = "F" if end == "F" else "R"
    lr = "L" if side == "L" else "R"
    return fb + lr


def build_foot_permutation(names):
    """Return a permutation reordering ``names`` to canonical FL, FR, RL, RR order.

    Falls back to identity order (with a warning) if the names do not map cleanly
    to the four canonical labels, so recording never crashes on an unexpected rig.
    """
    desired = ["FL", "FR", "RL", "RR"]
    labels = [canonical_foot_label(n) for n in names]
    if sorted(labels) == sorted(desired):
        perm = [labels.index(d) for d in desired]
        ordered_names = [names[i] for i in perm]
        return perm, ordered_names
    print(f"[WARN] Foot names {names} -> labels {labels} do not match FL/FR/RL/RR; "
          f"keeping raw order.")
    return list(range(len(names))), list(names)


def main():
    """Evaluate policy, report statistics, and record the rollout."""

    # Get configs
    env_cfg = gym.spec(args_cli.task).kwargs["env_cfg_entry_point"]()  # Instantiate!
    agent_cfg = gym.spec(args_cli.task).kwargs["rsl_rl_cfg_entry_point"]()

    # Configure for evaluation
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device else "cuda:0"

    # Disable cameras for speed (unless explicitly enabled)
    if not args_cli.enable_cameras:
        print("[INFO] Cameras disabled for faster evaluation.")
        env_cfg.tiled_camera_front = None
        env_cfg.tiled_camera_rear = None
        env_cfg.tiled_camera_left = None
        env_cfg.tiled_camera_right = None
        env_cfg.tiled_camera_top = None

    print(f"[INFO] Evaluating: {args_cli.checkpoint}")
    print(f"[INFO] Running {args_cli.num_envs} envs for {args_cli.num_episodes} total episodes.")

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    raw_env = env.unwrapped

    # Load policy
    device = args_cli.device if args_cli.device else "cuda:0"
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # ------------------------------------------------------------------
    # Recording setup
    # ------------------------------------------------------------------
    save_dir = None
    rec = None
    sensor_perm = robot_perm = None
    feet_sensor_ids = feet_robot_ids = None
    foot_names_canonical = None
    has_terrain_levels = has_terrain_types = False
    terrain = None
    if args_cli.record:
        tag = (args_cli.task
               .replace("Meldog-RL-Locomotion-", "")
               .replace("-v0", "")
               .replace("-", "_")
               .lower())
        save_dir = make_evaluation_dir("locomotion", tag)
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Recording rollout to: {save_dir / 'rollout.h5'}")

        # Foot body indices: forces come from the contact sensor, kinematics from
        # the articulation. Both are reordered independently to FL, FR, RL, RR.
        feet_sensor_ids, feet_sensor_names = raw_env._contact_sensor.find_bodies(".*F_link")
        feet_robot_ids, feet_robot_names = raw_env._robot.find_bodies(".*F_link")
        sensor_perm, sensor_names_ordered = build_foot_permutation(feet_sensor_names)
        robot_perm, robot_names_ordered = build_foot_permutation(feet_robot_names)
        foot_names_canonical = robot_names_ordered
        print(f"[INFO] Foot order (FL,FR,RL,RR) -> sensor bodies "
              f"{sensor_names_ordered}, robot bodies {robot_names_ordered}")

        rec = {k: [] for k in (
            "root_pos_w", "root_quat_w", "root_lin_vel_b", "root_ang_vel_b",
            "joint_pos", "joint_vel", "applied_torque", "actions", "commands",
            "foot_forces_w", "foot_pos_w", "foot_vel_w",
            "dones", "time_outs", "terrain_levels", "terrain_types",
        )}

        # Terrain bookkeeping availability
        terrain = raw_env._terrain
        has_terrain_levels = hasattr(terrain, "terrain_levels") and terrain.terrain_levels is not None
        has_terrain_types = hasattr(terrain, "terrain_types") and terrain.terrain_types is not None

    # Statistics trackers
    total_episodes = 0
    success_count = 0  # Timeout = survived
    failure_count = 0  # Early termination = crashed
    episode_lengths = []

    obs = env.get_observations()
    current_lengths = torch.zeros(args_cli.num_envs, device=device)

    # Benchmark mode: per-env episode clock driving the scripted command sequence.
    step_dt = float(raw_env.step_dt)
    bench_clock = torch.zeros(args_cli.num_envs, device=device)
    if args_cli.benchmark:
        print(f"[INFO] Benchmark mode ON (seed={args_cli.seed}): scripted commands "
              f"0-5s (0.8,0,0) -> 5-10s (0,0,0.8) -> 10-15s (0.5,0.3,0), repeating.")

    # Evaluation loop
    with torch.inference_mode():
        while total_episodes < args_cli.num_episodes:
            # Override commands every step from the per-env episode clock, and freeze
            # the env's own mid-episode resampling so the recorded command stays on
            # script (see benchmark_commands / --benchmark).
            if args_cli.benchmark:
                raw_env._commands[:] = benchmark_commands(bench_clock, device)
                raw_env._command_time_left[:] = 1.0e9

            actions = policy(obs)
            obs, _, dones, infos = env.step(actions)
            current_lengths += 1
            if args_cli.benchmark:
                bench_clock += step_dt
                bench_clock[dones.bool()] = 0.0  # restart clock for reset envs

            time_outs = infos.get("time_outs", torch.zeros_like(dones))

            # --- record post-step state (done steps are marked; the analyzer
            #     drops those single post-reset samples per episode) ---
            if args_cli.record:
                rec["root_pos_w"].append(raw_env._robot.data.root_pos_w.cpu().numpy())
                rec["root_quat_w"].append(raw_env._robot.data.root_quat_w.cpu().numpy())
                rec["root_lin_vel_b"].append(raw_env._robot.data.root_lin_vel_b.cpu().numpy())
                rec["root_ang_vel_b"].append(raw_env._robot.data.root_ang_vel_b.cpu().numpy())
                rec["joint_pos"].append(raw_env._robot.data.joint_pos.cpu().numpy())
                rec["joint_vel"].append(raw_env._robot.data.joint_vel.cpu().numpy())
                rec["applied_torque"].append(raw_env._robot.data.applied_torque.cpu().numpy())
                rec["actions"].append(actions.cpu().numpy())
                rec["commands"].append(raw_env.commands.cpu().numpy())
                rec["foot_forces_w"].append(
                    raw_env._contact_sensor.data.net_forces_w[:, feet_sensor_ids].cpu().numpy())
                rec["foot_pos_w"].append(
                    raw_env._robot.data.body_pos_w[:, feet_robot_ids].cpu().numpy())
                rec["foot_vel_w"].append(
                    raw_env._robot.data.body_lin_vel_w[:, feet_robot_ids].cpu().numpy())
                rec["dones"].append(dones.cpu().numpy().astype(np.uint8))
                rec["time_outs"].append(time_outs.cpu().numpy().astype(np.uint8))
                if has_terrain_levels:
                    rec["terrain_levels"].append(terrain.terrain_levels.cpu().numpy().astype(np.int16))
                if has_terrain_types:
                    rec["terrain_types"].append(terrain.terrain_types.cpu().numpy().astype(np.int16))

            if torch.any(dones):
                done_indices = torch.nonzero(dones).flatten()
                for idx in done_indices:
                    is_timeout = time_outs[idx].item()
                    if is_timeout:
                        success_count += 1
                    else:
                        failure_count += 1
                    episode_lengths.append(current_lengths[idx].item())
                    current_lengths[idx] = 0
                    total_episodes += 1

            # Progress update
            if total_episodes % 50 == 0 and total_episodes > 0:
                print(f"Progress: {total_episodes}/{args_cli.num_episodes} | "
                      f"Success: {success_count / total_episodes * 100:.1f}%", end="\r")

    # Final report
    print("\n" + "=" * 50)
    print(f"EVALUATION REPORT: {args_cli.task}")
    print("=" * 50)

    table = PrettyTable(["Metric", "Value"])
    table.add_row(["Total Episodes", total_episodes])
    table.add_row(["Successes (Survived)", success_count])
    table.add_row(["Failures (Crashed)", failure_count])

    survival_rate = (success_count / total_episodes) * 100 if total_episodes > 0 else 0.0
    table.add_row(["Survival Rate", f"{survival_rate:.2f}%"])

    avg_len = np.mean(episode_lengths) if episode_lengths else 0
    std_len = np.std(episode_lengths) if len(episode_lengths) > 1 else 0
    table.add_row(["Avg Episode Length", f"{avg_len:.1f} ± {std_len:.1f} steps"])

    print(table)
    print("=" * 50)

    # Analysis
    if survival_rate < 95.0:
        print("\n[ANALYSIS] High failure rate detected.")
        print("Possible causes:")
        print("  1. Terrain difficulty mismatch")
        print("  2. Domain randomization differences")
        print("  3. Action clipping mismatch")
    else:
        print("\n[ANALYSIS] Policy looks stable!")

    # ------------------------------------------------------------------
    # Write rollout.h5
    # ------------------------------------------------------------------
    if args_cli.record:
        _write_rollout(save_dir, rec, raw_env, env_cfg, sensor_perm, robot_perm,
                       foot_names_canonical)

    sys.stdout.flush()
    env.close()


def _write_rollout(save_dir, rec, raw_env, env_cfg, sensor_perm, robot_perm,
                   foot_names_canonical):
    """Stack recorded buffers, reorder feet to FL/FR/RL/RR, and write rollout.h5."""
    rollout_path = save_dir / "rollout.h5"
    print(f"\n[INFO] Writing rollout to {rollout_path} ...")

    def stack(key, dtype=np.float32):
        return np.asarray(np.stack(rec[key], axis=0), dtype=dtype)  # (T, E, ...)

    root_pos_w = stack("root_pos_w")
    root_quat_w = stack("root_quat_w")
    root_lin_vel_b = stack("root_lin_vel_b")
    root_ang_vel_b = stack("root_ang_vel_b")
    joint_pos = stack("joint_pos")
    joint_vel = stack("joint_vel")
    applied_torque = stack("applied_torque")
    actions = stack("actions")
    commands = stack("commands")
    dones = stack("dones", dtype=np.uint8)
    time_outs = stack("time_outs", dtype=np.uint8)

    # Feet: (T, E, 4, 3) reordered to FL, FR, RL, RR.
    foot_forces_w = stack("foot_forces_w")[:, :, sensor_perm, :]
    foot_pos_w = stack("foot_pos_w")[:, :, robot_perm, :]
    foot_vel_w = stack("foot_vel_w")[:, :, robot_perm, :]

    T, E = root_pos_w.shape[0], root_pos_w.shape[1]

    # Robot mass (actual, post-randomization if available; else default).
    try:
        masses = raw_env._robot.root_physx_view.get_masses().cpu().numpy()  # (E, nbodies)
    except Exception:
        masses = raw_env._robot.data.default_mass.cpu().numpy()
    robot_mass_per_env = masses.sum(axis=1).astype(np.float32)  # (E,)
    robot_mass = float(robot_mass_per_env.mean())

    # Joint limits (authoritative values come from the actuator config).
    actuator = list(env_cfg.robot.actuators.values())[0]
    effort_limit = float(actuator.effort_limit)
    velocity_limit = float(actuator.velocity_limit)
    try:
        joint_pos_limits = raw_env._robot.data.joint_pos_limits[0].cpu().numpy().astype(np.float32)
    except Exception:
        joint_pos_limits = None

    gzip = dict(compression="gzip")
    with h5py.File(rollout_path, "w") as f:
        f.create_dataset("root_pos_w", data=root_pos_w, **gzip)
        f.create_dataset("root_quat_w", data=root_quat_w, **gzip)
        f.create_dataset("root_lin_vel_b", data=root_lin_vel_b, **gzip)
        f.create_dataset("root_ang_vel_b", data=root_ang_vel_b, **gzip)
        f.create_dataset("joint_pos", data=joint_pos, **gzip)
        f.create_dataset("joint_vel", data=joint_vel, **gzip)
        f.create_dataset("applied_torque", data=applied_torque, **gzip)
        f.create_dataset("actions", data=actions, **gzip)
        f.create_dataset("commands", data=commands, **gzip)
        f.create_dataset("foot_forces_w", data=foot_forces_w, **gzip)
        f.create_dataset("foot_pos_w", data=foot_pos_w, **gzip)
        f.create_dataset("foot_vel_w", data=foot_vel_w, **gzip)
        f.create_dataset("dones", data=dones, **gzip)
        f.create_dataset("time_outs", data=time_outs, **gzip)

        if rec["terrain_levels"]:
            f.create_dataset("terrain_levels",
                             data=np.stack(rec["terrain_levels"], axis=0).astype(np.int16), **gzip)
        if rec["terrain_types"]:
            f.create_dataset("terrain_types",
                             data=np.stack(rec["terrain_types"], axis=0).astype(np.int16), **gzip)

        # -------- attributes --------
        a = f.attrs
        a["step_dt"] = float(raw_env.step_dt)
        a["num_steps"] = int(T)
        a["num_envs"] = int(E)
        a["contact_force_threshold"] = float(CONTACT_FORCE_THRESHOLD)
        a["gravity"] = 9.81
        a["robot_mass"] = robot_mass
        f.create_dataset("robot_mass_per_env", data=robot_mass_per_env)
        a["joint_effort_limit"] = effort_limit
        a["joint_velocity_limit"] = velocity_limit
        a["joint_names"] = list(raw_env._robot.data.joint_names)
        a["body_names"] = list(raw_env._robot.data.body_names)
        a["foot_order"] = "FL,FR,RL,RR"
        a["foot_body_names"] = list(foot_names_canonical)
        if joint_pos_limits is not None:
            f.create_dataset("joint_pos_limits", data=joint_pos_limits)  # (12, 2): lower, upper

        # meta (consumed by the analyzer for metrics.json)
        a["task"] = args_cli.task
        a["checkpoint"] = args_cli.checkpoint
        a["seed"] = int(args_cli.seed)
        a["num_episodes"] = int(args_cli.num_episodes)
        a["benchmark_mode"] = bool(args_cli.benchmark)
        a["git_commit"] = get_git_suffix()
        a["date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    size_mb = rollout_path.stat().st_size / 1e6
    print(f"[INFO] rollout.h5 written: {T} steps x {E} envs ({size_mb:.1f} MB).")


if __name__ == "__main__":
    main()
    simulation_app.close()
