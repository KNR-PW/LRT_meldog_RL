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

parser.add_argument(
    "--profile", type=str, default="clean", choices=["clean", "real"],
    help=(
        "Benchmark conditions, applied the same way to every robot (only with --benchmark). "
        "clean: no observation noise, pushes or mass randomization, nominal friction, default "
        "reset pose, curriculum off. real: clean plus observation noise, friction 0.4-1.2, "
        "trunk mass +-10 %% and pushes."
    ),
)
parser.add_argument(
    "--bench_terrain", type=str, default="task", choices=["task", "flat", "rough", "obs", "rough_obs"],
    help=(
        "Terrain for --benchmark: 'task' keeps the task's own terrain; the others are Meldog's "
        "shared terrains with a fixed terrain cell per env (difficulty rows 0, 3, 6, 9)."
    ),
)
parser.add_argument(
    "--output_root", type=str, default=None,
    help="Parent folder for the evaluation folder (default: logs/locomotion).",
)
parser.add_argument(
    "--full_episodes", action="store_true",
    help=(
        "Count only each env's first episode and start it at step 0 (direct envs such as "
        "Meldog randomize episode length on reset). Every env then runs a full-length "
        "episode unless it falls; --num_episodes is set to --num_envs."
    ),
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Benchmark mode pins the seed so terrain generation and resets are reproducible.
if args_cli.benchmark:
    args_cli.seed = 42
    args_cli.full_episodes = True  # benchmark episodes are always full length (finding E1)
if args_cli.full_episodes:
    args_cli.num_episodes = args_cli.num_envs

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

import isaaclab_tasks  # noqa: F401  (registers the Isaac Lab reference tasks)

try:
    import robot_lab.tasks  # noqa: F401  (registers robot_lab reference tasks when installed)
except ImportError:
    pass

import meldog_rl
from meldog_rl import envs  # This registers the tasks
from meldog_rl import agents
from meldog_rl.eval.adapters import load_cfgs, make_adapter, task_tag
from meldog_rl.eval.benchmark import BENCHMARK_CYCLE_S, BENCHMARK_SCHEDULE, benchmark_commands
from meldog_rl.eval.benchmark_terrains import bench_cells
from meldog_rl.utils import make_evaluation_dir
from meldog_rl.utils.git_utils import get_git_suffix


# Contact detection threshold (Newtons) -- matches the env's contact visualization.
CONTACT_FORCE_THRESHOLD = 1.0

def main():
    """Evaluate policy, report statistics, and record the rollout."""

    # Get configs (Meldog registers config classes, Isaac Lab registers "module:Class" strings)
    adapter = make_adapter(args_cli.task)
    env_cfg, agent_cfg = load_cfgs(args_cli.task)

    # Obs normalization must match the checkpoint, not the current runner cfg
    # (pre-run-B v1 checkpoints have no normalizer state)
    ckpt = torch.load(args_cli.checkpoint, map_location="cpu", weights_only=False)
    has_norm = any(k.startswith("actor_obs_normalizer.") for k in ckpt["model_state_dict"])
    agent_cfg.policy.actor_obs_normalization = has_norm
    agent_cfg.policy.critic_obs_normalization = has_norm
    del ckpt

    # Configure for evaluation (cameras off unless requested; benchmark command setup)
    device = args_cli.device if args_cli.device else "cuda:0"
    adapter.configure_cfg(env_cfg, num_envs=args_cli.num_envs, seed=args_cli.seed, device=device,
                          enable_cameras=bool(args_cli.enable_cameras), benchmark=args_cli.benchmark)
    if not args_cli.enable_cameras:
        print("[INFO] Cameras disabled for faster evaluation.")
    if args_cli.benchmark:
        for line in adapter.apply_benchmark(env_cfg, args_cli.profile, args_cli.bench_terrain):
            print(f"[BENCH] {line}")
    FALL_TILT_RAD = 1.0       # trunk tilt that counts as a fall
    FALL_BASE_FORCE_N = 1.0   # trunk contact force that counts as a fall

    print(f"[INFO] Evaluating: {args_cli.checkpoint}")
    print(f"[INFO] Running {args_cli.num_envs} envs for {args_cli.num_episodes} total episodes.")

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    raw_env = env.unwrapped
    adapter.bind(raw_env)
    cells = None
    if args_cli.benchmark and args_cli.bench_terrain not in ("task", "flat"):
        holder, attr = adapter.terrain_cfg_holder(env_cfg)
        cells = bench_cells(getattr(holder, attr).terrain_generator, args_cli.num_envs)
        adapter.assign_terrain_cells(cells)
        env.reset()
        print(f"[BENCH] fixed terrain cells: {sorted(set((c[2], c[0]) for c in cells))}")

    # Load policy
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # ------------------------------------------------------------------
    # Recording setup
    # ------------------------------------------------------------------
    save_dir = None
    rec = None
    feet_sensor_ids = feet_robot_ids = hip_robot_ids = None
    foot_names_canonical = None
    has_terrain_levels = has_terrain_types = has_posture = False
    terrain = None
    if args_cli.record:
        tag = task_tag(args_cli.task)
        if args_cli.benchmark:
            tag = f"{tag}_on_{args_cli.bench_terrain}_{args_cli.profile}_{args_cli.num_envs}envs"
        save_dir = make_evaluation_dir("locomotion", tag, base_dir=args_cli.output_root)
        # Parallel runs can start in the same second: never write into an existing folder.
        base_name, n = save_dir.name, 1
        while True:
            try:
                save_dir.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                n += 1
                save_dir = save_dir.with_name(f"{base_name}_{n}")
        print(f"[INFO] Recording rollout to: {save_dir / 'rollout.h5'}")

        # Foot body indices: forces come from the contact sensor, kinematics from
        # the articulation. Both are reordered independently to FL, FR, RL, RR.
        feet_sensor_ids, feet_robot_ids, foot_names_canonical = adapter.foot_ids()
        hip_robot_ids = adapter.hip_body_ids()
        print(f"[INFO] Robot '{adapter.spec.name}': feet (FL,FR,RL,RR) -> {foot_names_canonical}")

        rec = {k: [] for k in (
            "root_pos_w", "root_quat_w", "root_lin_vel_b", "root_ang_vel_b",
            "joint_pos", "joint_vel", "applied_torque", "computed_torque", "actions", "commands",
            "foot_forces_w", "feet_forces_max", "foot_pos_w", "foot_vel_w", "hip_pos_w",
            "dones", "time_outs", "terrain_levels", "terrain_types",
            "terrain_height_under_body", "base_height", "terrain_normal_b",
        )}

        # Posture channel (optional): needs a height scanner. Tasks without one (flat
        # tasks) simply don't record it and the analyzer skips the posture.* metrics.
        has_posture = adapter.has_posture()
        if not has_posture:
            print("[WARN] env has no height scanner; "
                  "skipping terrain_height_under_body / base_height / terrain_normal_b.")

        # Terrain bookkeeping availability
        terrain = adapter.terrain
        has_terrain_levels = getattr(terrain, "terrain_levels", None) is not None
        has_terrain_types = getattr(terrain, "terrain_types", None) is not None

    # Statistics trackers
    total_episodes = 0
    success_count = 0  # Timeout = survived
    failure_count = 0  # Early termination = crashed
    episode_lengths = []

    robot = adapter.robot
    contact = adapter.contact_sensor
    obs = env.get_observations()
    current_lengths = torch.zeros(args_cli.num_envs, device=device)
    counted = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=device)
    first_fell = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=device)
    first_survived = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=device)
    first_length = torch.zeros(args_cli.num_envs, dtype=torch.long, device=device)
    base_ids = adapter.base_sensor_ids()
    if not base_ids:
        print(f"[WARN] base body '{adapter.spec.base_body}' not in contact sensor; fall rule uses tilt only.")
    if args_cli.full_episodes:
        # Undo the reset-time episode-length randomization so every first episode is full length.
        raw_env.episode_length_buf[:] = 0
        print("[INFO] Full episodes: counting each env's first episode only.")

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
                adapter.set_commands(benchmark_commands(bench_clock, device))
                adapter.freeze_command_resampling()

            actions = policy(obs)
            obs, _, dones, infos = env.step(actions)
            current_lengths += 1
            if args_cli.benchmark:
                bench_clock += step_dt
                bench_clock[dones.bool()] = 0.0  # restart clock for reset envs

            time_outs = infos.get("time_outs", torch.zeros_like(dones))

            # Evaluator-side fall rule, identical for every robot (tasks terminate differently).
            tilt = torch.acos(torch.clamp(-robot.data.projected_gravity_b[:, 2], -1.0, 1.0))
            fallen_now = tilt > FALL_TILT_RAD
            if base_ids:
                base_force = torch.norm(contact.data.net_forces_w_history[:, :, base_ids], dim=-1)
                fallen_now |= base_force.amax(dim=(1, 2)) > FALL_BASE_FORCE_N
            first_fell |= fallen_now & ~counted & ~dones.bool()

            # --- record post-step state (done steps are marked; the analyzer
            #     drops those single post-reset samples per episode) ---
            if args_cli.record:
                rec["root_pos_w"].append(robot.data.root_pos_w.cpu().numpy())
                rec["root_quat_w"].append(robot.data.root_quat_w.cpu().numpy())
                rec["root_lin_vel_b"].append(robot.data.root_lin_vel_b.cpu().numpy())
                rec["root_ang_vel_b"].append(robot.data.root_ang_vel_b.cpu().numpy())
                rec["joint_pos"].append(robot.data.joint_pos.cpu().numpy())
                rec["joint_vel"].append(robot.data.joint_vel.cpu().numpy())
                rec["applied_torque"].append(robot.data.applied_torque.cpu().numpy())
                rec["computed_torque"].append(robot.data.computed_torque.cpu().numpy())
                rec["actions"].append(actions.cpu().numpy())
                rec["commands"].append(adapter.get_commands().cpu().numpy())
                rec["foot_forces_w"].append(
                    contact.data.net_forces_w[:, feet_sensor_ids].cpu().numpy())
                # Substep-max contact force magnitude per foot (peak transient the
                # policy-rate snapshot above misses). net_forces_w_history is
                # (E, history_len, nbodies, 3); max the per-substep magnitude over
                # the history dim -> (E, 4). See impact.peak_force_bw in the analyzer.
                rec["feet_forces_max"].append(
                    torch.max(
                        torch.norm(
                            contact.data.net_forces_w_history[:, :, feet_sensor_ids],
                            dim=-1,
                        ),
                        dim=1,
                    )[0].cpu().numpy())
                rec["foot_pos_w"].append(
                    robot.data.body_pos_w[:, feet_robot_ids].cpu().numpy())
                rec["foot_vel_w"].append(
                    robot.data.body_lin_vel_w[:, feet_robot_ids].cpu().numpy())
                rec["hip_pos_w"].append(robot.data.body_pos_w[:, hip_robot_ids].cpu().numpy())
                rec["dones"].append(dones.cpu().numpy().astype(np.uint8))
                rec["time_outs"].append(time_outs.cpu().numpy().astype(np.uint8))
                if has_terrain_levels:
                    rec["terrain_levels"].append(terrain.terrain_levels.cpu().numpy().astype(np.int16))
                if has_terrain_types:
                    rec["terrain_types"].append(terrain.terrain_types.cpu().numpy().astype(np.int16))
                if has_posture:
                    # Run D posture channel: local ground reference under the trunk,
                    # the trunk height above it, and the fitted terrain-plane normal
                    # expressed in the body frame (analyzer -> posture.*).
                    terrain_z = adapter.terrain_height_under_body()
                    rec["terrain_height_under_body"].append(terrain_z.cpu().numpy())
                    rec["base_height"].append(
                        (adapter.height_scanner.data.pos_w[:, 2] - terrain_z).cpu().numpy())
                    rec["terrain_normal_b"].append(adapter.terrain_normal_b().cpu().numpy())

            if torch.any(dones):
                done_indices = torch.nonzero(dones).flatten()
                for idx in done_indices:
                    if args_cli.full_episodes and counted[idx]:
                        continue
                    counted[idx] = True
                    first_length[idx] = int(current_lengths[idx].item())
                    first_survived[idx] = bool(time_outs[idx].item()) and not bool(first_fell[idx].item())
                    is_timeout = first_survived[idx].item() if args_cli.benchmark else time_outs[idx].item()
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
        first = dict(fell=first_fell, survived=first_survived, length=first_length, cells=cells)
        _write_rollout(save_dir, rec, raw_env, adapter, foot_names_canonical, first)

    sys.stdout.flush()
    env.close()


def _write_rollout(save_dir, rec, raw_env, adapter, foot_names_canonical, first):
    """Stack recorded buffers (feet already in FL/FR/RL/RR order) and write rollout.h5."""
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
    foot_forces_w = stack("foot_forces_w")
    foot_pos_w = stack("foot_pos_w")
    foot_vel_w = stack("foot_vel_w")
    # Substep-max force magnitude: (T, E, 4).
    feet_forces_max = stack("feet_forces_max")

    T, E = root_pos_w.shape[0], root_pos_w.shape[1]

    # Robot mass (actual, post-randomization if available; else default).
    try:
        masses = adapter.robot.root_physx_view.get_masses().cpu().numpy()  # (E, nbodies)
    except Exception:
        masses = adapter.robot.data.default_mass.cpu().numpy()
    robot_mass_per_env = masses.sum(axis=1).astype(np.float32)  # (E,)
    robot_mass = float(robot_mass_per_env.mean())

    # Joint limits per joint, from the actuator models (inf where an actuator has no fixed
    # limit, e.g. Spot's knees). The scalar attrs keep the largest finite value for older
    # analyzer versions.
    effort_limits, velocity_limits = adapter.joint_limits()
    effort_limits = effort_limits.numpy().astype(np.float32)
    velocity_limits = velocity_limits.numpy().astype(np.float32)
    finite_effort = effort_limits[np.isfinite(effort_limits)]
    finite_velocity = velocity_limits[np.isfinite(velocity_limits)]
    effort_limit = float(finite_effort.max()) if finite_effort.size else float("inf")
    velocity_limit = float(finite_velocity.max()) if finite_velocity.size else float("inf")
    robot_data = adapter.robot.data
    try:
        joint_pos_limits = robot_data.joint_pos_limits[0].cpu().numpy().astype(np.float32)
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
        f.create_dataset("feet_forces_max", data=feet_forces_max, **gzip)
        f.create_dataset("foot_pos_w", data=foot_pos_w, **gzip)
        f.create_dataset("foot_vel_w", data=foot_vel_w, **gzip)
        f.create_dataset("hip_pos_w", data=stack("hip_pos_w"), **gzip)
        f.create_dataset("computed_torque", data=stack("computed_torque"), **gzip)
        f.create_dataset("dones", data=dones, **gzip)
        f.create_dataset("time_outs", data=time_outs, **gzip)

        if rec["terrain_levels"]:
            f.create_dataset("terrain_levels",
                             data=np.stack(rec["terrain_levels"], axis=0).astype(np.int16), **gzip)
        if rec["terrain_types"]:
            f.create_dataset("terrain_types",
                             data=np.stack(rec["terrain_types"], axis=0).astype(np.int16), **gzip)

        # Run D posture channel (absent on rollouts recorded before the terrain-
        # relative helpers existed; the analyzer treats it as optional).
        for key in ("terrain_height_under_body", "base_height", "terrain_normal_b"):
            if rec[key]:
                f.create_dataset(key, data=stack(key), **gzip)

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
        a["joint_names"] = list(robot_data.joint_names)
        a["body_names"] = list(robot_data.body_names)
        a["robot_name"] = adapter.spec.name
        a["env_type"] = type(adapter).__name__
        f.create_dataset("joint_effort_limits", data=effort_limits)      # (J,)
        f.create_dataset("joint_velocity_limits", data=velocity_limits)  # (J,)
        f.create_dataset("default_joint_pos",
                         data=robot_data.default_joint_pos[0].cpu().numpy().astype(np.float32))
        f.create_dataset("knee_joint_idx", data=np.asarray(adapter.knee_joint_ids(), dtype=np.int32))
        # Robot size for interpreting results across robots (leg length = thigh + shank).
        thigh_len, shank_len = adapter.leg_segment_lengths()
        f.create_dataset("leg_thigh_lengths", data=thigh_len.numpy().astype(np.float32))  # (4,)
        f.create_dataset("leg_shank_lengths", data=shank_len.numpy().astype(np.float32))  # (4,)
        a["leg_length"] = float((thigh_len + shank_len).mean())
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
        a["full_episodes"] = bool(args_cli.full_episodes)
        a["profile"] = args_cli.profile if args_cli.benchmark else "task"
        a["bench_terrain"] = args_cli.bench_terrain if args_cli.benchmark else "task"
        a["contact_history_length"] = int(adapter.contact_sensor.cfg.history_length)
        if args_cli.full_episodes:
            f.create_dataset("first_episode_fell", data=first["fell"].cpu().numpy().astype(np.uint8))
            f.create_dataset("first_episode_survived", data=first["survived"].cpu().numpy().astype(np.uint8))
            f.create_dataset("first_episode_length", data=first["length"].cpu().numpy().astype(np.int32))
        if first["cells"] is not None:
            f.create_dataset("terrain_cell", data=np.asarray([c[:2] for c in first["cells"]], dtype=np.int16))
            a["terrain_cell_kinds"] = [c[2] for c in first["cells"]]
        a["git_commit"] = get_git_suffix()
        a["date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    size_mb = rollout_path.stat().st_size / 1e6
    print(f"[INFO] rollout.h5 written: {T} steps x {E} envs ({size_mb:.1f} MB).")


if __name__ == "__main__":
    main()
    simulation_app.close()
