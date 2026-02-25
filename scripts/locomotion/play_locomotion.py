#!/usr/bin/env python3
# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Play trained Meldog locomotion policy.

Usage:
    python scripts/locomotion/play_locomotion.py \
        --task Meldog-RL-Locomotion-Rough-Sim-v0 \
        --checkpoint logs/locomotion/LM_rough_sim_.../model_5000.pt
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "source"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play trained Meldog locomotion policy.")
parser.add_argument("--task", type=str, default="Meldog-RL-Locomotion-Rough-Sim-v0", help="Task name.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # Enable for visualization

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ============================================================================

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import meldog_rl
from meldog_rl import envs  # This registers the tasks
from meldog_rl import agents


def main():
    """Play trained policy."""
    
    # Get configs
    env_cfg = gym.spec(args_cli.task).kwargs["env_cfg_entry_point"]()  # Instantiate!
    agent_cfg = gym.spec(args_cli.task).kwargs["rsl_rl_cfg_entry_point"]()
    
    # Override for play mode
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device else "cuda:0"
    
    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="human")
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    
    # Create runner and load checkpoint
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    
    # Get policy
    policy = runner.get_inference_policy(device=agent_cfg.device)
    
    # Play loop
    obs = env.get_observations()
    
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, _, _ = env.step(actions)
    
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
