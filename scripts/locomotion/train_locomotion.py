#!/usr/bin/env python3
# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Train Meldog locomotion policy with RSL-RL."""

import argparse
import sys
import os

# Add source to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "source"))

from isaaclab.app import AppLauncher

# CLI argument setup
parser = argparse.ArgumentParser(description="Train Meldog locomotion policy with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=500, 
                    help="Length of recorded video in env steps (500 steps ≈ 10 seconds at 50Hz).")
parser.add_argument("--video_interval", type=int, default=50000, 
                    help="Env steps between recordings (50000 steps ≈ every 2000 iterations with num_steps_per_env=24).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Meldog-RL-Locomotion-Rough-Sim-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--distributed", action="store_true", default=False, help="Run training with multiple GPUs.")
parser.add_argument("--resume", action="store_true", default=False, help="Resume from checkpoint.")
parser.add_argument("--load_run", type=str, default=None, 
                    help="Path to checkpoint (.pt file) OR run directory name (e.g., LM_flat_sim_2026-01-19_22-20-16).")
parser.add_argument("--load_checkpoint", type=str, default=None, 
                    help="Checkpoint pattern when using directory name (e.g., model_500.pt). Ignored if load_run is a .pt file.")

# AppLauncher args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Enable cameras for video recording
if args_cli.video:
    args_cli.enable_cameras = True

# Clear sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# Launch simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Imports after AppLauncher
import gymnasium as gym
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# Import meldog_rl to register tasks (must be after AppLauncher)
import meldog_rl
from meldog_rl import envs  # This registers the tasks
from meldog_rl import agents
from meldog_rl.utils import make_locomotion_log_dir
from meldog_rl.utils.git_utils import save_git_metadata

# Enable TF32 for faster training
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def extract_terrain_domain(task_name: str) -> tuple[str, str]:
    """Extract terrain and domain from task name.
    
    Args:
        task_name: e.g., "Meldog-RL-Locomotion-Rough-Sim-v0"
        
    Returns:
        (terrain, domain): e.g., ("rough", "sim")
    """
    parts = task_name.split("-")
    # Format: Meldog-RL-Locomotion-{Terrain}-{Domain}-v0
    terrain = parts[3].lower()  # Rough -> rough
    domain = parts[4].lower()   # Sim -> sim
    return terrain, domain


def main():
    """Train locomotion policy."""
    
    # Get env and agent configs from registered task
    env_cfg = gym.spec(args_cli.task).kwargs["env_cfg_entry_point"]()  # Instantiate!
    agent_cfg = gym.spec(args_cli.task).kwargs["rsl_rl_cfg_entry_point"]()
    
    # Override with CLI arguments
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
        agent_cfg.seed = args_cli.seed
    
    # Set device
    env_cfg.sim.device = args_cli.device if args_cli.device else "cuda:0"
    
    # Multi-GPU setup
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        env_cfg.seed = agent_cfg.seed + app_launcher.local_rank
        agent_cfg.seed = env_cfg.seed
    
    # Create log directory with naming convention
    terrain, domain = extract_terrain_domain(args_cli.task)
    log_dir = make_locomotion_log_dir(terrain, domain)
    log_dir = str(log_dir.absolute())
    
    print(f"[INFO] Logging to: {log_dir}")
    
    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    # Video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    
    # Wrap for RSL-RL
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    
    # Create runner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    
    # Resume from checkpoint
    if args_cli.resume:
        # Check if load_run is a direct path to a checkpoint file
        if args_cli.load_run and args_cli.load_run.endswith(".pt"):
            resume_path = args_cli.load_run
            if not os.path.isabs(resume_path):
                resume_path = os.path.abspath(resume_path)
        else:
            # Use get_checkpoint_path for directory-based loading
            from isaaclab_tasks.utils import get_checkpoint_path
            resume_path = get_checkpoint_path(
                os.path.dirname(log_dir),
                args_cli.load_run if args_cli.load_run else ".*",
                args_cli.load_checkpoint if args_cli.load_checkpoint else "model_.*.pt",
            )
        print(f"[INFO] Loading checkpoint from: {resume_path}")
        runner.load(resume_path)
    
    # Save configs
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    save_git_metadata(log_dir)
    
    # Train
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    
    # Cleanup
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
