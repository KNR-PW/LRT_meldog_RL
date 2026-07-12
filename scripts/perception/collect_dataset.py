#!/usr/bin/env python3
# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Collect perception dataset using trained locomotion policy.

Features:
- 4x Depth cameras (front, rear, left, right)
- Ground truth heightmap from raycast scanner
- Robot pose (position + quaternion)
- Preview video (Top RGB + Depth overlays)

Usage:
    python scripts/perception/collect_dataset.py \
        --task Meldog-RL-Dataset-Rough-v0 \
        --checkpoint logs/locomotion/LM_rough_sim_.../model_500.pt \
        --max_steps 10000

Output:
    datasets/PD_{terrain}_{timestamp}/
    ├── dataset.h5
    └── preview.mp4
"""

import argparse
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "source"))

from isaaclab.app import AppLauncher

# Arguments
parser = argparse.ArgumentParser(description="Collect perception dataset for Meldog.")
parser.add_argument("--task", type=str, default="Meldog-RL-Dataset-Rough-v0", help="Dataset task.")
parser.add_argument("--checkpoint", type=str, required=True, help="Locomotion policy checkpoint.")
parser.add_argument("--max_steps", type=int, default=10000, help="Max simulation steps.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
parser.add_argument("--min_episode_len", type=int, default=50, help="Discard episodes shorter than this.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # Dataset collection requires cameras

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ============================================================================

import gymnasium as gym
import torch
import numpy as np
import h5py
import cv2

from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import meldog_rl
from meldog_rl import envs, agents
from meldog_rl.utils import make_dataset_dir


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def process_depth(tensor_img, label, width=424, height=240):
    """Convert depth tensor to colorized image."""
    img = tensor_img.squeeze().cpu().numpy()
    img[np.isinf(img)] = 5.0
    img[img > 5.0] = 5.0
    img[img <= 0] = 5.0 
    norm_img = np.clip(img, 0, 5.0) / 5.0 * 255
    norm_img = norm_img.astype(np.uint8)
    color_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)
    color_img = cv2.resize(color_img, (width, height))
    cv2.putText(color_img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(color_img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return color_img


def process_raycast(tensor_scan, label="Raycast", target_h=240, target_w=424):
    """Convert raycast heightmap to colorized image."""
    hm = tensor_scan.squeeze().cpu().numpy()
    hm_norm = np.clip(hm, -1.0, 1.0)
    hm_norm = (hm_norm + 1.0) / 2.0 * 255.0
    hm_uint8 = hm_norm.astype(np.uint8)
    color_hm = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_VIRIDIS)
    square_img = cv2.resize(color_hm, (target_h, target_h), interpolation=cv2.INTER_NEAREST)
    pad_l = (target_w - target_h) // 2
    pad_r = target_w - target_h - pad_l
    final_img = cv2.copyMakeBorder(square_img, 0, 0, pad_l, pad_r, cv2.BORDER_CONSTANT, value=[0,0,0])
    cv2.putText(final_img, label, (pad_l + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(final_img, label, (pad_l + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return final_img


# =============================================================================
# DATA CONVERSION
# =============================================================================

def to_uint16_mm(data_list):
    """Convert depth to uint16 millimeters."""
    arr = np.array(data_list, dtype=np.float32)
    invalid_mask = np.isinf(arr) | (arr > 5.0) | (arr <= 0.0)
    arr[invalid_mask] = 5.0
    return (arr * 1000.0).astype(np.uint16)


def to_int16_mm(data_list):
    """Convert height to int16 millimeters."""
    arr = np.array(data_list, dtype=np.float32)
    return (np.clip(arr, -32.0, 32.0) * 1000.0).astype(np.int16)


# =============================================================================
# TERRAIN EXTRACTION
# =============================================================================

def extract_terrain(task_name: str) -> str:
    """Extract terrain from task name."""
    parts = task_name.split("-")
    # Format: Meldog-RL-Dataset-{Terrain}-v0
    if len(parts) >= 4:
        return parts[3].lower()
    return "unknown"


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Collect perception dataset."""
    
    # Get configs
    env_cfg = gym.spec(args_cli.task).kwargs["env_cfg_entry_point"]()
    agent_cfg = gym.spec(args_cli.task).kwargs["rsl_rl_cfg_entry_point"]()
    
    # Configure
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device else "cuda:0"
    
    print(f"[INFO] Creating environment: {args_cli.task}")
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    
    # Load policy
    print(f"[INFO] Loading policy from: {args_cli.checkpoint}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=env.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.device)
    
    # Create output directory
    terrain = extract_terrain(args_cli.task)
    output_dir = make_dataset_dir(terrain)
    output_dir.mkdir(parents=True, exist_ok=True)
    h5_path = output_dir / "dataset.h5"
    
    print(f"[INFO] Saving data to: {output_dir}")
    
    # Initialize buffers for each environment
    buffers = [
        {
            "depth_front": [], "depth_rear": [], "depth_left": [], "depth_right": [],
            "gt_height": [], "robot_pos": [], "robot_quat": []
        }
        for _ in range(args_cli.num_envs)
    ]
    
    # Video setup
    video_path = output_dir / "preview.mp4"
    video_writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), 20, (1272, 480))
    video_frames = 0
    max_video_frames = 1000 
    video_finished = False
    
    raw_env = env.unwrapped
    obs = env.get_observations()
    episode_counter = [0]
    step_count = 0
    
    with h5py.File(h5_path, 'w') as h5file:
        
        def save_episode_to_h5(env_idx, force_save=False):
            """Save episode buffer to HDF5."""
            ep_len = len(buffers[env_idx]["depth_front"])
            if ep_len >= args_cli.min_episode_len or (force_save and ep_len > 0):
                grp_name = f"episode_{episode_counter[0]}"
                grp = h5file.create_group(grp_name)
                
                grp.create_dataset("depth_front", data=to_uint16_mm(buffers[env_idx]["depth_front"]), compression="gzip")
                grp.create_dataset("depth_rear",  data=to_uint16_mm(buffers[env_idx]["depth_rear"]),  compression="gzip")
                grp.create_dataset("depth_left",  data=to_uint16_mm(buffers[env_idx]["depth_left"]),  compression="gzip")
                grp.create_dataset("depth_right", data=to_uint16_mm(buffers[env_idx]["depth_right"]), compression="gzip")
                grp.create_dataset("gt_height",   data=to_int16_mm(buffers[env_idx]["gt_height"]),    compression="gzip")
                grp.create_dataset("robot_pos",   data=np.array(buffers[env_idx]["robot_pos"], dtype=np.float32))
                grp.create_dataset("robot_quat",  data=np.array(buffers[env_idx]["robot_quat"], dtype=np.float32))
                
                print(f"[INFO] Saved Episode {episode_counter[0]} (Steps: {ep_len})")
                episode_counter[0] += 1
            
            for key in buffers[env_idx]:
                buffers[env_idx][key] = []
        
        # Verify cameras are available
        if raw_env._cameras.get("front") is None:
            print("[ERROR] Cameras not configured! Use a Dataset task (e.g., Meldog-RL-Dataset-Rough-v0)")
            print("[ERROR] or ensure your task config has camera sensors enabled.")
            env.close()
            return
        
        while True:
            with torch.inference_mode():
                actions = policy(obs)
                obs, rewards, dones, extras = env.step(actions)
            
            # Get camera data (cameras stored in _cameras dict)
            d_front = raw_env._cameras["front"].data.output["distance_to_image_plane"].clone()
            d_rear  = raw_env._cameras["rear"].data.output["distance_to_image_plane"].clone()
            d_left  = raw_env._cameras["left"].data.output["distance_to_image_plane"].clone()
            d_right = raw_env._cameras["right"].data.output["distance_to_image_plane"].clone()
            
            # RGB Top (only for video preview)
            rgb_top = None
            if raw_env._cameras.get("top") is not None:
                try:
                    rgb_top = raw_env._cameras["top"].data.output["rgb"].clone()
                except:
                    pass
            
            # Get GT height from raycast scanner
            if hasattr(raw_env, "_gt_scanner") and raw_env._gt_scanner:
                trunk_z = raw_env._gt_scanner.data.pos_w[:, 2].unsqueeze(1)
                gt_scan = raw_env._gt_scanner.data.ray_hits_w[..., 2] - trunk_z
                
                grid_side = int(np.sqrt(gt_scan.shape[1])) 
                gt_scan = gt_scan.view(args_cli.num_envs, grid_side, grid_side)
                gt_scan = torch.clamp(gt_scan, -5.0, 5.0)
                gt_scan = gt_scan.transpose(-2, -1)
                gt_scan = torch.flip(gt_scan, dims=[-2, -1])
            else:
                gt_scan = torch.zeros((args_cli.num_envs, 40, 40), device=env.device)
            
            # Get pose
            pos_w = raw_env._robot.data.root_pos_w.clone()
            quat_w = raw_env._robot.data.root_quat_w.clone()
            
            # Store data for each environment
            for i in range(args_cli.num_envs):
                buffers[i]["depth_front"].append(d_front[i].squeeze().cpu().numpy())
                buffers[i]["depth_rear"].append(d_rear[i].squeeze().cpu().numpy())
                buffers[i]["depth_left"].append(d_left[i].squeeze().cpu().numpy())
                buffers[i]["depth_right"].append(d_right[i].squeeze().cpu().numpy())
                buffers[i]["gt_height"].append(gt_scan[i].squeeze().cpu().numpy())
                buffers[i]["robot_pos"].append(pos_w[i].cpu().numpy())
                buffers[i]["robot_quat"].append(quat_w[i].cpu().numpy())
                
                # Generate preview video for env 0
                if i == 0 and not video_finished and rgb_top is not None:
                    img_front = process_depth(d_front[i], "Front", 424, 240)
                    img_rear  = process_depth(d_rear[i],  "Rear",  424, 240)
                    
                    img_top_raw = rgb_top[i].cpu().numpy()
                    img_top = cv2.cvtColor(img_top_raw, cv2.COLOR_RGB2BGR)
                    img_top = cv2.resize(img_top, (424, 240))
                    cv2.putText(img_top, "Top RGB", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    
                    img_left  = process_depth(d_left[i],  "Left",  424, 240)
                    img_right = process_depth(d_right[i], "Right", 424, 240)
                    img_ray   = process_raycast(gt_scan[i], "Raycast GT", 240, 424)
                    
                    row1 = np.hstack((img_front, img_rear, img_top))
                    row2 = np.hstack((img_left, img_right, img_ray))
                    full_frame = np.vstack((row1, row2))
                    
                    video_writer.write(full_frame)
                    video_frames += 1
                    if video_frames >= max_video_frames:
                        video_finished = True
                        video_writer.release()
                        print(f"[INFO] Preview video saved ({video_frames} frames).")
                
                # Save episode on reset
                if dones[i]:
                    save_episode_to_h5(i)
            
            step_count += 1
            if step_count % 500 == 0:
                print(f"[INFO] Progress: {step_count}/{args_cli.max_steps} steps, {episode_counter[0]} episodes saved")
            
            if step_count >= args_cli.max_steps:
                break
        
        # Save remaining data
        for i in range(args_cli.num_envs):
            save_episode_to_h5(i, force_save=True)
    
    if not video_finished and video_writer.isOpened():
        video_writer.release()
    
    env.close()
    print(f"\n[DONE] Dataset collection complete!")
    print(f"[DONE] Episodes: {episode_counter[0]}")
    print(f"[DONE] Output: {output_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
