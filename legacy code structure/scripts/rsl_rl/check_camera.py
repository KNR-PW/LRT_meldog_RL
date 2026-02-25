"""
Script to run the policy for a few seconds and export camera views to PNG.
"""
import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run policy and export camera images")
parser.add_argument("--task", type=str, default="Template-Meldog-Simple-Locomotion-Policy-Direct-v0")
parser.add_argument("--num_envs", type=int, default=4, help="Number of robots to spawn")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained .pt checkpoint")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

import meldog_simple_locomotion_policy.tasks 

def main():
    print(f"[INFO] Setting up task: {args_cli.task}")
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    
    env_cfg.scene.env_spacing = 8.0 
    
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    if hasattr(agent_cfg, "to_dict"):
        agent_cfg = agent_cfg.to_dict()

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    
    log_dir = os.path.dirname(args_cli.checkpoint)
    runner = OnPolicyRunner(env, agent_cfg, log_dir=log_dir, device=args_cli.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    
    obs, _ = env.reset()
    print("[INFO] Letting robots run for 5 seconds (250 steps)...")
    
    for _ in tqdm(range(250)):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

    print("[INFO] Capturing images...")
    raw_env = env.unwrapped 
    
    cameras = {
        "Front": raw_env._tiled_camera_front,
        "Rear":  raw_env._tiled_camera_rear,
        "Left":  raw_env._tiled_camera_left,
        "Right": raw_env._tiled_camera_right
    }

    timestamp = datetime.now().strftime("%H-%M-%S")
    save_dir = f"camera_check_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    for name, cam in cameras.items():
        depth_tensor = cam.data.output["distance_to_image_plane"][0].squeeze(-1) 
        
        d_min = torch.min(depth_tensor).item()
        d_max = torch.max(depth_tensor).item()
        print(f"[{name}] Range: {d_min:.2f}m to {d_max:.2f}m")

        far_plane = 5.0 
        depth_tensor = torch.nan_to_num(depth_tensor, posinf=far_plane)
        depth_tensor = torch.clamp(depth_tensor, 0.0, far_plane)
        
        filename = os.path.join(save_dir, f"{name}.png")
        plt.imsave(filename, depth_tensor.cpu().numpy(), cmap='magma', vmin=0.0, vmax=far_plane)
        print(f"   Saved {filename}")

    print(f"\n[SUCCESS] Check the folder: {save_dir}")
    
    print("[INFO] Images saved. Robots will keep walking now (Press Ctrl+C to stop)...")
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

    simulation_app.close()

if __name__ == "__main__":
    main()