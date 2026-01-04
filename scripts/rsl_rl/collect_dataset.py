"""
Dataset Collector for Meldog
- Runs the trained policy using standard RSL-RL loading mechanisms.
- Captures: 4x Depth Cameras (Front, Rear, Left, Right) + 1x RGB Top Camera + Raycast.
- Saves valid episodes to HDF5 (UInt16 for depth).
- [UPDATED] Records a 2x3 Grid Video (RGB Top + Square Raycast).
- [UPDATED] Naming convention fixed to YYYY-MM-DD_HH-MM-SS.
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
parser.add_argument("--dataset_name", type=str, default="meldog_dataset", help="Base name (ignored for folder naming now)")
parser.add_argument("--max_steps", type=int, default=1000, help="Total steps per env to record")
parser.add_argument("--min_episode_len", type=int, default=50, help="Discard episodes shorter than this")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Enable cameras for rendering
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Late Imports ---
import gymnasium as gym
import torch
import numpy as np
import h5py
import cv2

import isaaclab_rl.rsl_rl as rsl_rl_utils
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner

import meldog_simple_locomotion_policy.tasks

# --- Visualization Helpers ---

def process_depth(tensor_img, label, width=424, height=240):
    """Normalize depth tensor, colorize, and add label."""
    img = tensor_img.squeeze().cpu().numpy()
    
    # Handle Sky/Infinity (Visuals Only)
    img[np.isinf(img)] = 5.0
    img[img > 5.0] = 5.0
    img[img <= 0] = 5.0 
    
    # Normalize 0-5m -> 0-255
    norm_img = np.clip(img, 0, 5.0) / 5.0 * 255
    norm_img = norm_img.astype(np.uint8)
    
    # Colorize (Jet: Blue=Close, Red=Far)
    color_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)
    color_img = cv2.resize(color_img, (width, height))
    
    # Label
    cv2.putText(color_img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(color_img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return color_img

def process_raycast(tensor_scan, label="Raycast", target_h=240, target_w=424):
    """Visualize the raycast height map as a SQUARE image, padded to match width."""
    # Input is relative height in meters (e.g., -0.4 to +0.2)
    # 0.0 = Robot Height. Negative = Below Robot.
    hm = tensor_scan.squeeze().cpu().numpy()
    
    # Normalize for visualization
    # We map range [-1.0m, 1.0m] to [0, 255]
    # -1.0m (Deep Hole) -> 0 (Dark)
    #  0.0m (Flat)      -> 128 (Gray)
    # +1.0m (Obstacle)  -> 255 (Bright)
    hm_norm = np.clip(hm, -1.0, 1.0)
    hm_norm = (hm_norm + 1.0) / 2.0 * 255.0
    hm_uint8 = hm_norm.astype(np.uint8)
    
    # Colorize (Viridis is good for height maps)
    color_hm = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_VIRIDIS)
    
    # Resize to Square (Height x Height) to keep 1:1 aspect ratio
    square_img = cv2.resize(color_hm, (target_h, target_h), interpolation=cv2.INTER_NEAREST)
    
    # Pad width to match other cameras (424) so video stacking works
    pad_l = (target_w - target_h) // 2
    pad_r = target_w - target_h - pad_l
    final_img = cv2.copyMakeBorder(square_img, 0, 0, pad_l, pad_r, cv2.BORDER_CONSTANT, value=[0,0,0])
    
    # Label
    cv2.putText(final_img, label, (pad_l + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(final_img, label, (pad_l + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return final_img

# --- Main ---

def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = load_cfg_from_registry(args_cli.task, args_cli.agent)
    if hasattr(agent_cfg, "to_dict"):
        agent_cfg = agent_cfg.to_dict()

    print(f"[INFO] Creating environment: {args_cli.task}")
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = rsl_rl_utils.RslRlVecEnvWrapper(env)

    print(f"[INFO] Loading policy from: {args_cli.checkpoint}")
    runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=args_cli.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.device)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    full_dataset_name = timestamp
    
    save_dir = os.path.join("datasets", full_dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    h5_path = os.path.join(save_dir, "dataset.h5")
    
    print(f"[INFO] Saving data to: {save_dir}")

    # Buffers
    buffers = [
        {
            "depth_front": [], "depth_rear": [], "depth_left": [], "depth_right": [],
            "rgb_top": [], # Kept for video, but not saved to H5
            "gt_height": [], "robot_vel": [], "cmd": []
        }
        for _ in range(args_cli.num_envs)
    ]

    # Video Writer: 3 Cols x 2 Rows (424*3 x 240*2) = 1272 x 480
    video_path = os.path.join(save_dir, "preview.mp4")
    video_writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        20, (1272, 480) 
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

            # -- 1. Data Extraction (Cameras) --
            try:
                d_front = raw_env._tiled_camera_front.data.output["distance_to_image_plane"].clone()
                d_rear  = raw_env._tiled_camera_rear.data.output["distance_to_image_plane"].clone()
                d_left  = raw_env._tiled_camera_left.data.output["distance_to_image_plane"].clone()
                d_right = raw_env._tiled_camera_right.data.output["distance_to_image_plane"].clone()
                
                # Fetch RGB for Top Camera (for video only)
                rgb_top = raw_env._tiled_camera_top.data.output["rgb"].clone()
                
            except AttributeError:
                print("[ERROR] Cameras not found! Check env config and env.py.")
                break

            # -- 2. Data Extraction (Raycast / Height) --
            if hasattr(raw_env, "_gt_scanner") and raw_env._gt_scanner is not None:
                ray_hits_w = raw_env._gt_scanner.data.ray_hits_w
                robot_z = raw_env._robot.data.root_pos_w[:, 2].unsqueeze(1) 
                hit_z = ray_hits_w[..., 2]                                  
                gt_scan = hit_z - robot_z 
                
                num_rays = gt_scan.shape[1]
                grid_side = int(np.sqrt(num_rays))
                gt_scan = gt_scan.view(args_cli.num_envs, grid_side, grid_side)
                gt_scan = torch.clamp(gt_scan, -5.0, 5.0)
                
                # --- [FIX] Transpose to swap X/Y axes ---
                # This fixes the "diagonal mismatch" that physical rotation couldn't solve.
                gt_scan = gt_scan.transpose(-2, -1) 
                
                # You noticed "Inverted Y" looked correct? 
                # Uncommenting this line usually aligns "Forward" to "Image Top"
                gt_scan = torch.flip(gt_scan, dims=[-2, -1]) 
                
            else:
                gt_scan = torch.zeros((args_cli.num_envs, 1, 1), device=env.device)

            # -- 3. Robot State --
            vel = raw_env._robot.data.root_lin_vel_b.clone() 
            cmd = raw_env._commands.clone()                  

            # -- 4. Process per Environment --
            for i in range(args_cli.num_envs):
                # Buffer
                buffers[i]["depth_front"].append(d_front[i].squeeze().cpu().numpy())
                buffers[i]["depth_rear"].append(d_rear[i].squeeze().cpu().numpy())
                buffers[i]["depth_left"].append(d_left[i].squeeze().cpu().numpy())
                buffers[i]["depth_right"].append(d_right[i].squeeze().cpu().numpy())
                
                # Append RGB Top (Used for video logic below)
                buffers[i]["rgb_top"].append(rgb_top[i].cpu().numpy()) 
                
                buffers[i]["gt_height"].append(gt_scan[i].squeeze().cpu().numpy())
                buffers[i]["robot_vel"].append(vel[i].cpu().numpy())
                buffers[i]["cmd"].append(cmd[i].cpu().numpy())

                # Video (Env 0)
                if i == 0 and not video_finished:
                    # Row 1: Front | Rear | Top (RGB)
                    img_front = process_depth(d_front[i], "Front", 424, 240)
                    img_rear  = process_depth(d_rear[i],  "Rear",  424, 240)
                    
                    # Process Top RGB
                    img_top_raw = rgb_top[i].cpu().numpy()
                    img_top = cv2.cvtColor(img_top_raw, cv2.COLOR_RGB2BGR) # Convert to BGR for OpenCV
                    img_top = cv2.resize(img_top, (424, 240))
                    cv2.putText(img_top, "Top RGB", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                    # Row 2: Left | Right | Raycast (Squared + Padded)
                    img_left  = process_depth(d_left[i],  "Left",  424, 240)
                    img_right = process_depth(d_right[i], "Right", 424, 240)
                    img_ray   = process_raycast(gt_scan[i], "Raycast GT", 240, 424)

                    # Stack
                    row1 = np.hstack((img_front, img_rear, img_top))
                    row2 = np.hstack((img_left, img_right, img_ray))
                    full_frame = np.vstack((row1, row2))
                    
                    # Overlay
                    cv2.putText(full_frame, f"Dataset: {full_dataset_name}", (350, 460), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    video_writer.write(full_frame)
                    video_frames += 1

                    if video_frames >= max_video_frames:
                        video_finished = True
                        video_writer.release()
                        print(f"[INFO] Preview video saved ({video_frames} frames).")

                # Save H5
                if dones[i]:
                    ep_len = len(buffers[i]["depth_front"])
                    if ep_len >= args_cli.min_episode_len:
                        grp_name = f"episode_{collected_episodes}"
                        grp = h5file.create_group(grp_name)
                        
                        # Helpers
                        def to_uint16_mm(data_list):
                            arr = np.array(data_list, dtype=np.float32)
                            mask_invalid = np.isinf(arr) | (arr > 20.0) | (arr <= 0.0)
                            arr = arr * 1000.0
                            arr[mask_invalid] = 0
                            return arr.astype(np.uint16)

                        def to_int16_mm(data_list):
                            arr = np.array(data_list, dtype=np.float32)
                            arr = np.clip(arr, -32.0, 32.0) * 1000.0
                            return arr.astype(np.int16)

                        # Save Images (Depth)
                        grp.create_dataset("depth_front", data=to_uint16_mm(buffers[i]["depth_front"]), compression="gzip")
                        grp.create_dataset("depth_rear",  data=to_uint16_mm(buffers[i]["depth_rear"]),  compression="gzip")
                        grp.create_dataset("depth_left",  data=to_uint16_mm(buffers[i]["depth_left"]),  compression="gzip")
                        grp.create_dataset("depth_right", data=to_uint16_mm(buffers[i]["depth_right"]), compression="gzip")

                        
                        # Save Data
                        grp.create_dataset("gt_height",   data=to_int16_mm(buffers[i]["gt_height"]),    compression="gzip")
                        grp.create_dataset("velocity",    data=np.array(buffers[i]["robot_vel"], dtype=np.float32))
                        grp.create_dataset("command",     data=np.array(buffers[i]["cmd"], dtype=np.float32))
                        
                        collected_episodes += 1
                        print(f"[INFO] Saved Episode {collected_episodes} (Steps: {ep_len})")
                    
                    for key in buffers[i]:
                        buffers[i][key] = []

            step_count += 1
            if step_count >= args_cli.max_steps:
                print(f"[INFO] Max steps reached. Stopping.")
                break

    if not video_finished and 'video_writer' in locals() and video_writer.isOpened():
        video_writer.release()
        
    env.close()
    print("[DONE] Dataset collection finished.")

if __name__ == "__main__":
    main()
    simulation_app.close()