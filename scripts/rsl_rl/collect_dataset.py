"""
Dataset Collector for Meldog
- Runs the trained policy using standard RSL-RL loading mechanisms.
- Captures: 4x Depth Cameras + Robot State + Ground Truth.
- Saves valid episodes (filtered by length) to HDF5.
- Records a FIXED LENGTH debug video (1000 frames) of Env 0.
- [FIXED] Video Aspect Ratio matches Dataset (16:9, 848x480).
- [FIXED] Depth Saved as UINT16 (mm). Sky/Inf -> 0.
"""

import argparse
import os
import sys
from datetime import datetime

# --- Isaac Lab App Launcher ---
from isaaclab.app import AppLauncher

# Argument Parsing
parser = argparse.ArgumentParser(description="Collect Perception Dataset for Meldog")
parser.add_argument("--task", type=str, default="Template-Meldog-Simple-Locomotion-Policy-Direct-v0")
parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel robots")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to policy .pt file")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL Agent config entry point")
parser.add_argument("--dataset_name", type=str, default="meldog_perception_dataset")
parser.add_argument("--max_steps", type=int, default=1000, help="Total steps per env to record")
parser.add_argument("--min_episode_len", type=int, default=50, help="Discard episodes shorter than this")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Enable cameras for rendering
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Late Imports (after app launch) ---
import gymnasium as gym
import torch
import numpy as np
import h5py
import cv2

import isaaclab_rl.rsl_rl as rsl_rl_utils
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner

import meldog_simple_locomotion_policy.tasks

def process_depth(tensor_img, label, width=424, height=240):
    """Normalize depth tensor, colorize, and add label."""
    img = tensor_img.squeeze().cpu().numpy()
    
    # Handle Sky/Infinity for VISUALIZATION ONLY (Red)
    img[np.isinf(img)] = 5.0
    img[img > 5.0] = 5.0
    img[img <= 0] = 5.0 
    
    # Normalize 0-5m -> 0-255
    norm_img = np.clip(img, 0, 5.0) / 5.0 * 255
    norm_img = norm_img.astype(np.uint8)
    
    # Colorize
    color_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)
    
    # Resize to native 16:9 resolution (424x240)
    color_img = cv2.resize(color_img, (width, height))
    
    # Add Label (Smaller font for smaller resolution)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(color_img, label, (10, 30), font, 0.6, (0, 0, 0), 3)
    cv2.putText(color_img, label, (10, 30), font, 0.6, (255, 255, 255), 1)
    return color_img

def main():
    # 1. Prepare Configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = load_cfg_from_registry(args_cli.task, args_cli.agent)
    if hasattr(agent_cfg, "to_dict"):
        agent_cfg = agent_cfg.to_dict()

    # 2. Create Environment
    print(f"[INFO] Creating environment: {args_cli.task}")
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = rsl_rl_utils.RslRlVecEnvWrapper(env)

    # 3. Initialize Runner and Load Policy
    print(f"[INFO] Loading policy from: {args_cli.checkpoint}")
    runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=args_cli.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.device)

    # 4. Prepare Data Storage
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    full_dataset_name = f"{args_cli.dataset_name}_{timestamp}"
    save_dir = os.path.join("datasets", full_dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    h5_path = os.path.join(save_dir, "dataset.h5")
    
    print(f"[INFO] Saving data to: {save_dir}")

    buffers = [
        {
            "depth_front": [], "depth_rear": [], "depth_left": [], "depth_right": [],
            "gt_height": [], "robot_vel": [], "cmd": []
        }
        for _ in range(args_cli.num_envs)
    ]

    # [FIXED] Resolution: 2x 424 width, 2x 240 height -> 848x480
    video_path = os.path.join(save_dir, "preview.mp4")
    video_writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        20, (848, 480) 
    )
    
    video_frames = 0
    max_video_frames = 1000 
    video_finished = False

    raw_env = env.unwrapped
    obs = env.get_observations()
    collected_episodes = 0
    step_count = 0
    
    with h5py.File(h5_path, 'w') as h5file:
        
        while simulation_app.is_running():
            with torch.inference_mode():
                actions = policy(obs)
                obs, rewards, dones, extras = env.step(actions)

            # -- Data Extraction --
            try:
                d_front = raw_env._tiled_camera_front.data.output["distance_to_image_plane"].clone()
                d_rear  = raw_env._tiled_camera_rear.data.output["distance_to_image_plane"].clone()
                d_left  = raw_env._tiled_camera_left.data.output["distance_to_image_plane"].clone()
                d_right = raw_env._tiled_camera_right.data.output["distance_to_image_plane"].clone()
            except AttributeError:
                print("[ERROR] Cameras not found! Check env.py _setup_scene.")
                break

            # Ground Truth
            if hasattr(raw_env, "_gt_scanner") and raw_env._gt_scanner is not None:
                ray_hits_w = raw_env._gt_scanner.data.ray_hits_w
                robot_z = raw_env._robot.data.root_pos_w[:, 2].unsqueeze(1) 
                hit_z = ray_hits_w[..., 2]                                  
                gt_scan = hit_z - robot_z 
                num_rays = gt_scan.shape[1]
                grid_side = int(np.sqrt(num_rays))
                gt_scan = gt_scan.view(args_cli.num_envs, grid_side, grid_side)
                gt_scan = torch.clamp(gt_scan, -5.0, 5.0)
            else:
                gt_scan = torch.zeros((args_cli.num_envs, 1, 1), device=env.device) 

            vel = raw_env._robot.data.root_lin_vel_b.clone() 
            cmd = raw_env._commands.clone()                  

            for i in range(args_cli.num_envs):
                buffers[i]["depth_front"].append(d_front[i].squeeze().cpu().numpy())
                buffers[i]["depth_rear"].append(d_rear[i].squeeze().cpu().numpy())
                buffers[i]["depth_left"].append(d_left[i].squeeze().cpu().numpy())
                buffers[i]["depth_right"].append(d_right[i].squeeze().cpu().numpy())
                buffers[i]["gt_height"].append(gt_scan[i].squeeze().cpu().numpy())
                buffers[i]["robot_vel"].append(vel[i].cpu().numpy())
                buffers[i]["cmd"].append(cmd[i].cpu().numpy())

                if i == 0 and not video_finished:
                    # [FIXED] Use native 424x240 resolution
                    img_front = process_depth(d_front[i], "Front", 424, 240)
                    img_rear  = process_depth(d_rear[i],  "Rear",  424, 240)
                    img_left  = process_depth(d_left[i],  "Left",  424, 240)
                    img_right = process_depth(d_right[i], "Right", 424, 240)

                    top_row = np.hstack((img_front, img_rear))
                    bot_row = np.hstack((img_left, img_right))
                    full_frame = np.vstack((top_row, bot_row))
                    
                    overlay_text = f"Dataset: {full_dataset_name}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    text_size = cv2.getTextSize(overlay_text, font, 0.6, 1)[0]
                    text_x = (full_frame.shape[1] - text_size[0]) // 2
                    
                    # [FIXED] Adjust text Y position for 480px height
                    text_y = 460 

                    cv2.putText(full_frame, overlay_text, (text_x, text_y), font, 0.6, (0, 0, 0), 3)
                    cv2.putText(full_frame, overlay_text, (text_x, text_y), font, 0.6, (255, 255, 255), 1)
                    
                    video_writer.write(full_frame)
                    video_frames += 1

                    if video_frames >= max_video_frames:
                        video_finished = True
                        video_writer.release()
                        print(f"[INFO] Preview video saved ({video_frames} frames). Stopping recording.")

                if dones[i]:
                    ep_len = len(buffers[i]["depth_front"])
                    if ep_len >= args_cli.min_episode_len:
                        grp_name = f"episode_{collected_episodes}"
                        grp = h5file.create_group(grp_name)
                        
                        # --- H5 SAVING (UInt16 + Gzip) ---
                        def to_uint16_mm(data_list):
                            arr = np.array(data_list, dtype=np.float32)
                            # Mask invalid data
                            mask_invalid = np.isinf(arr) | (arr > 20.0) | (arr <= 0.0)
                            arr = arr * 1000.0
                            arr[mask_invalid] = 0
                            return arr.astype(np.uint16)

                        def to_int16_mm(data_list):
                            arr = np.array(data_list, dtype=np.float32)
                            arr = np.clip(arr, -32.0, 32.0) * 1000.0
                            return arr.astype(np.int16)

                        grp.create_dataset("depth_front", data=to_uint16_mm(buffers[i]["depth_front"]), compression="gzip")
                        grp.create_dataset("depth_rear",  data=to_uint16_mm(buffers[i]["depth_rear"]),  compression="gzip")
                        grp.create_dataset("depth_left",  data=to_uint16_mm(buffers[i]["depth_left"]),  compression="gzip")
                        grp.create_dataset("depth_right", data=to_uint16_mm(buffers[i]["depth_right"]), compression="gzip")
                        
                        grp.create_dataset("gt_height",   data=to_int16_mm(buffers[i]["gt_height"]),    compression="gzip")
                        grp.create_dataset("velocity",    data=np.array(buffers[i]["robot_vel"], dtype=np.float32))
                        grp.create_dataset("command",     data=np.array(buffers[i]["cmd"], dtype=np.float32))
                        
                        collected_episodes += 1
                        print(f"[INFO] Saved Episode {collected_episodes} (Steps: {ep_len})")
                    
                    for key in buffers[i]:
                        buffers[i][key] = []

            step_count += 1
            if step_count >= args_cli.max_steps:
                print(f"[INFO] Max steps reached ({args_cli.max_steps}). Stopping.")
                break

    if not video_finished and 'video_writer' in locals() and video_writer.isOpened():
        video_writer.release()
        
    env.close()
    print("[DONE] Dataset collection finished.")

if __name__ == "__main__":
    main()
    simulation_app.close()