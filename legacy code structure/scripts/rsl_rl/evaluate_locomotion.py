# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Script to evaluate an RL agent with statistics."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import torch
import os
import numpy as np
from prettytable import PrettyTable

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate an RL agent with RSL-RL.")
parser.add_argument("--num_envs", type=int, default=100, help="Number of environments to simulate.")
parser.add_argument("--num_episodes", type=int, default=100, help="Total number of episodes to collect stats for.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Use the pre-trained checkpoint from Nucleus.")
# [FIX] Removed duplicate --checkpoint argument that caused the crash

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# [FIX] Removed the line that forced enable_cameras = False. 
# It now respects the command line flag.

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner, DistillationRunner
from isaaclab.envs import DirectMARLEnv, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import meldog_simple_locomotion_policy.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Evaluate with RSL-RL agent."""
    
    # 1. Configuration Setup
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 50
    env_cfg.seed = agent_cfg.seed if agent_cfg.seed is not None else 42
    
    # [FIX] Smart Camera Disabling
    # If the user did NOT run with --enable_cameras, we must remove cameras from the config
    # to prevent "RuntimeError: A camera was spawned without the --enable_cameras flag"
    if not args_cli.enable_cameras:
        print("[INFO] Cameras disabled in CLI. Removing cameras from Env Config for speed.")
        if hasattr(env_cfg, "tiled_camera_front"): env_cfg.tiled_camera_front = None
        if hasattr(env_cfg, "tiled_camera_rear"): env_cfg.tiled_camera_rear = None
        if hasattr(env_cfg, "tiled_camera_left"): env_cfg.tiled_camera_left = None
        if hasattr(env_cfg, "tiled_camera_right"): env_cfg.tiled_camera_right = None
        if hasattr(env_cfg, "tiled_camera_top"): env_cfg.tiled_camera_top = None

    # [FIX] Ensure clip_actions is applied
    clip_actions_val = 1.0
    if hasattr(agent_cfg, "clip_actions"):
        clip_actions_val = agent_cfg.clip_actions
    elif isinstance(agent_cfg, dict) and "clip_actions" in agent_cfg:
        clip_actions_val = agent_cfg["clip_actions"]

    # 2. Checkpoint Logic
    task_name = args_cli.task.split(":")[-1]
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    print(f"[INFO] Evaluating model from: {resume_path}")
    print(f"[INFO] Simulating {env_cfg.scene.num_envs} environments for {args_cli.num_episodes} total episodes.")

    # 3. Create Environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Wrap with RSL-RL wrapper
    env = RslRlVecEnvWrapper(env, clip_actions=clip_actions_val)

    # 4. Load Policy
    device = args_cli.device if args_cli.device else "cuda:0"
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # 5. Statistics Trackers
    total_episodes = 0
    success_count = 0  
    failure_count = 0 
    episode_lengths = []
    
    obs = env.get_observations()
    current_lengths = torch.zeros(env_cfg.scene.num_envs, device=device)

    # 6. Evaluation Loop
    with torch.inference_mode():
        while total_episodes < args_cli.num_episodes and simulation_app.is_running():
            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)
            current_lengths += 1
            
            if torch.any(dones):
                done_indices = torch.nonzero(dones).flatten()
                for idx in done_indices:
                    # Check for timeout (Success) or Death (Failure)
                    is_timeout = extras.get("time_outs", torch.zeros_like(dones))[idx].item()
                    
                    if is_timeout:
                        success_count += 1
                    else:
                        failure_count += 1
                        
                    episode_lengths.append(current_lengths[idx].item())
                    current_lengths[idx] = 0
                    total_episodes += 1

            if total_episodes % 50 == 0 and total_episodes > 0:
                print(f"Progress: {total_episodes}/{args_cli.num_episodes} eps | "
                      f"Current Success Rate: {success_count / total_episodes * 100:.1f}%", end="\r")

    # 7. Final Report
    print("\n" + "="*50)
    print(f"EVALUATION REPORT: {args_cli.task}")
    print("="*50)
    
    stats_table = PrettyTable(["Metric", "Value"])
    stats_table.add_row(["Total Episodes", total_episodes])
    stats_table.add_row(["Successes (Timeouts)", success_count])
    stats_table.add_row(["Failures (Crashes)", failure_count])
    
    survival_rate = (success_count / total_episodes) * 100 if total_episodes > 0 else 0.0
    stats_table.add_row(["Survival Rate", f"{survival_rate:.2f}%"])
    
    avg_len = np.mean(episode_lengths) if episode_lengths else 0
    stats_table.add_row(["Avg Episode Length", f"{avg_len:.1f} steps"])
    
    print(stats_table)
    print("="*50)
    
    if survival_rate < 95.0:
        print("[ANALYSIS] High failure rate detected.")
        print("Possible causes:")
        print("1. Terrain Difficulty: The robot might be spawning on Level 9 rough terrain immediately.")
        print("2. Action Clipping: Ensure 'clip_actions' matches training config.")
        print("3. Domain Randomization: Physics material/mass randomization might be too aggressive.")
    else:
        print("[ANALYSIS] Policy looks stable.")

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()