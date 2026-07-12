#!/usr/bin/env python3
# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate Meldog locomotion policy with statistics.

Usage:
    python scripts/locomotion/evaluate_locomotion.py \
        --task Meldog-RL-Locomotion-Rough-Sim-v0 \
        --checkpoint logs/locomotion/LM_rough_sim_.../model_5000.pt \
        --num_episodes 100
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "source"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate Meldog locomotion policy.")
parser.add_argument("--task", type=str, default="Meldog-RL-Locomotion-Rough-Sim-v0", help="Task name.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
parser.add_argument("--num_envs", type=int, default=100, help="Number of parallel environments.")
parser.add_argument("--num_episodes", type=int, default=100, help="Total episodes to evaluate.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ============================================================================

import numpy as np
import torch
import gymnasium as gym
from prettytable import PrettyTable

from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import meldog_rl
from meldog_rl import envs  # This registers the tasks
from meldog_rl import agents
from meldog_rl.utils import make_evaluation_dir


def main():
    """Evaluate policy and report statistics."""
    
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
    
    # Load policy
    device = args_cli.device if args_cli.device else "cuda:0"
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    
    # Statistics trackers
    total_episodes = 0
    success_count = 0  # Timeout = survived
    failure_count = 0  # Early termination = crashed
    episode_lengths = []
    
    obs = env.get_observations()
    current_lengths = torch.zeros(args_cli.num_envs, device=device)
    
    # Evaluation loop
    with torch.inference_mode():
        while total_episodes < args_cli.num_episodes:
            actions = policy(obs)
            obs, _, dones, infos = env.step(actions)
            current_lengths += 1
            
            if torch.any(dones):
                done_indices = torch.nonzero(dones).flatten()
                for idx in done_indices:
                    # Check if timeout (success) or death (failure)
                    is_timeout = infos.get("time_outs", torch.zeros_like(dones))[idx].item()
                    
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
    
    sys.stdout.flush()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
