#!/usr/bin/env python3
# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Debug camera setup and visualization.

Usage:
    python scripts/perception/debug_cameras.py \
        --task Meldog-RL-Dataset-Rough-v0
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "source"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Debug camera setup.")
parser.add_argument("--task", type=str, default="Meldog-RL-Dataset-Rough-v0", help="Dataset task.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ============================================================================

import torch
import gymnasium as gym
import matplotlib.pyplot as plt

import meldog_rl
from meldog_rl import envs  # This registers the tasks
from meldog_rl import agents


def main():
    """Debug cameras."""
    
    env_cfg = gym.spec(args_cli.task).kwargs["env_cfg_entry_point"]()  # Instantiate!
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device else "cuda:0"
    
    print("[INFO] Creating environment with cameras...")
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    # Access cameras
    meldog_env = env.unwrapped
    cameras = meldog_env.cameras
    
    print("\n[INFO] Camera status:")
    for name, camera in cameras.items():
        if camera is not None:
            print(f"  {name}: ENABLED ({camera.cfg.height}x{camera.cfg.width})")
        else:
            print(f"  {name}: DISABLED")
    
    # Take a few steps to get camera data
    print("\n[INFO] Stepping environment to get camera data...")
    for _ in range(10):
        obs, _, _, _, _ = env.step(torch.zeros(args_cli.num_envs, 12, device=env.unwrapped.device))
    
    # Visualize depth from each camera
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (name, camera) in enumerate(cameras.items()):
        if camera is not None and idx < len(axes):
            depth = camera.data.output.get("distance_to_image_plane")
            if depth is not None:
                depth_np = depth[0].cpu().numpy()
                axes[idx].imshow(depth_np, cmap='viridis')
                axes[idx].set_title(f"{name} (min={depth_np.min():.2f}, max={depth_np.max():.2f})")
            else:
                axes[idx].set_title(f"{name} (no depth data)")
                axes[idx].text(0.5, 0.5, "No data", ha='center', va='center')
        else:
            axes[idx].set_title(f"{name} (disabled)")
            axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig("camera_debug.png")
    print("[INFO] Saved camera_debug.png")
    
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
