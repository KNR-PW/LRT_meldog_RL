# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Perception Evaluation Script for Meldog (Pivot V2)
- Runs Locomotion + Perception.
- Saves HDF5 dataset in uint16/int16 format (Exact match to collect_dataset.py).
- Generates 3x3 Video.
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import cv2
import h5py
from datetime import datetime

from isaaclab.app import AppLauncher

# Argument Parsing
parser = argparse.ArgumentParser(description="Evaluate Perception Model for Meldog")
parser.add_argument("--task", type=str, default="Template-Meldog-Simple-Locomotion-Policy-Direct-v0")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel robots")
parser.add_argument("--locomotion_checkpoint", type=str, required=True, help="Path to locomotion policy .pt file")
parser.add_argument("--perception_checkpoint", type=str, default=None, help="Path to perception model (.pt).")
parser.add_argument("--video_length", type=int, default=400, help="Length of recording in steps")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL Agent config entry point")

# Append AppLauncher args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Force cameras on
args_cli.enable_cameras = True

# Launch Simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab.sim as sim_utils
import isaaclab_rl.rsl_rl as rsl_rl_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner
from isaaclab.utils.math import quat_apply

# Register custom task
import meldog_simple_locomotion_policy.tasks  

# -----------------------------------------------------------------------------
# MODEL CLASS DEFINITION
# -----------------------------------------------------------------------------
MAP_SIZE = 40
IMG_H, IMG_W = 120, 212

class SimpleMapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = self.conv_block(4, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)
        self.mlp_gravity = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 64))
        self.dec1 = self.up_block(256, 128)
        self.dec2 = self.up_block(128, 64)
        self.dec3 = self.up_block(64, 32)
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)
        self.final_resize = nn.AdaptiveAvgPool2d((MAP_SIZE, MAP_SIZE))

    def conv_block(self, in_c, out_c):
        return nn.Sequential(nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(), nn.MaxPool2d(2))

    def up_block(self, in_c, out_c):
        return nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                             nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU())

    def forward(self, x, grav):
        x = self.enc4(self.enc3(self.enc2(self.enc1(x))))
        B, C, H, W = x.shape
        grav_embed = self.mlp_gravity(grav).unsqueeze(-1).unsqueeze(-1).expand(B, 64, H, W)
        x = torch.cat([x, grav_embed], dim=1)
        x = nn.Conv2d(256+64, 256, 1).to(x.device)(x)
        x = self.final_resize(self.final_conv(self.dec3(self.dec2(self.dec1(x)))))
        return x

# -----------------------------------------------------------------------------
# PRE-PROCESSING
# -----------------------------------------------------------------------------
def process_inputs(d_front, d_rear, d_left, d_right, quat_raw):
    # Stack -> (B, 4, H, W)
    stack = torch.cat([d_front, d_rear, d_left, d_right], dim=-1).permute(0, 3, 1, 2)
    # Resize
    stack = F.interpolate(stack, size=(IMG_H, IMG_W), mode='bilinear', align_corners=False)
    # Scale (m -> normalized input)
    stack = torch.clamp(stack, 0, 5.0) * 0.2 

    # Gravity
    w, x, y, z = quat_raw[:, 0], quat_raw[:, 1], quat_raw[:, 2], quat_raw[:, 3]
    gx = -2 * (x*z + w*y)
    gy = -2 * (y*z - w*x)
    gz = -(1 - 2 * (x*x + y*y))
    grav_vec = torch.stack([gx, gy, gz], dim=1)

    return stack, grav_vec

# -----------------------------------------------------------------------------
# VISUALIZATION UTILS
# -----------------------------------------------------------------------------
TARGET_W = 424
TARGET_H = 240

def process_image(img_tensor, title, colormap=cv2.COLORMAP_JET, is_depth=True):
    img = img_tensor.squeeze().cpu().numpy()
    if is_depth:
        img[np.isinf(img)] = 5.0
        img[img > 5.0] = 5.0
        img[img <= 0] = 5.0
        norm_img = np.clip(img, 0, 5.0) / 5.0 * 255
    else:
        norm_img = np.clip(img, 0, 255) if img.max() > 1.0 else img * 255
    norm_img = norm_img.astype(np.uint8)
    if len(norm_img.shape) == 2:
        color_img = cv2.applyColorMap(norm_img, colormap)
    else:
        color_img = cv2.cvtColor(norm_img, cv2.COLOR_RGB2BGR)
    resized = cv2.resize(color_img, (TARGET_W, TARGET_H))
    cv2.putText(resized, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(resized, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return resized

def process_map_centered(map_tensor, title):
    data = map_tensor.squeeze().cpu().numpy()
    norm = np.clip((data + 0.3) / 0.6, 0.0, 1.0) * 255.0
    norm = norm.astype(np.uint8)
    color_img = cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS)
    square_size = TARGET_H 
    resized_square = cv2.resize(color_img, (square_size, square_size), interpolation=cv2.INTER_NEAREST)
    pad_total = TARGET_W - square_size
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    final_img = cv2.copyMakeBorder(resized_square, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0,0,0])
    cv2.putText(final_img, title, (pad_left + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(final_img, title, (pad_left + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return final_img

def create_info_panel(loco_name, perc_name, timestamp):
    panel = np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8)
    color_lbl = (200, 200, 200) 
    color_val = (0, 255, 0)     
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thick = 1
    def fmt_name(path):
        if not path: return "Dummy (Ground Truth)"
        try: return f"{os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}"
        except: return os.path.basename(path)
    y_start = 60
    y_step = 35
    cv2.putText(panel, "Locomotion Policy:", (20, y_start), font, scale, color_lbl, thick)
    cv2.putText(panel, fmt_name(loco_name), (20, y_start + 18), font, 0.45, color_val, thick)
    cv2.putText(panel, "Perception Model:", (20, y_start + y_step*2), font, scale, color_lbl, thick)
    cv2.putText(panel, fmt_name(perc_name), (20, y_start + y_step*2 + 18), font, 0.45, color_val, thick)
    cv2.putText(panel, "Date:", (20, y_start + y_step*4), font, scale, color_lbl, thick)
    cv2.putText(panel, timestamp, (20, y_start + y_step*4 + 18), font, 0.45, color_val, thick)
    return panel

class DummyPerceptionModel:
    def __call__(self, depth_inputs, gt_map=None):
        if gt_map is not None:
            return gt_map.clone()
        return torch.zeros((depth_inputs.shape[0], 41, 41), device=depth_inputs.device)

# -----------------------------------------------------------------------------
# SAVING HELPERS (Direct Copy from collect_dataset.py)
# -----------------------------------------------------------------------------
def to_uint16_mm(data_list):
    arr = np.array(data_list, dtype=np.float32)
    # [!] Matching collect_dataset.py strict filtering
    arr[np.isinf(arr) | (arr > 20.0) | (arr <= 0.0)] = 0
    return (arr * 1000.0).astype(np.uint16)

def to_int16_mm(data_list):
    arr = np.array(data_list, dtype=np.float32)
    return (np.clip(arr, -32.0, 32.0) * 1000.0).astype(np.int16)

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    # 1. Config
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = load_cfg_from_registry(args_cli.task, args_cli.agent)
    if hasattr(agent_cfg, "to_dict"): agent_cfg = agent_cfg.to_dict()

    print(f"[INFO] Creating environment: {args_cli.task}")
    env = gym.make(args_cli.task, cfg=env_cfg)
    clip_val = agent_cfg.get("clip_actions", 1.0)
    env = rsl_rl_utils.RslRlVecEnvWrapper(env, clip_actions=clip_val)

    # 2. Load Locomotion
    print(f"[INFO] Loading Locomotion: {args_cli.locomotion_checkpoint}")
    runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=args_cli.device)
    runner.load(args_cli.locomotion_checkpoint)
    policy = runner.get_inference_policy(device=env.device)

    # 3. Load Perception
    perception_model = None
    if args_cli.perception_checkpoint and os.path.exists(args_cli.perception_checkpoint):
        print(f"[INFO] Loading Perception: {args_cli.perception_checkpoint}")
        try:
            model = SimpleMapper()
            state_dict = torch.load(args_cli.perception_checkpoint, map_location=env.device)
            model.load_state_dict(state_dict)
            perception_model = model.to(env.device)
            perception_model.eval()
            print("[INFO] Model loaded successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            perception_model = None

    if perception_model is None:
        print("[WARN] Using Dummy Model (Ground Truth).")
        perception_model = DummyPerceptionModel()

    # 4. Setup Outputs
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    folder_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join("logs", "perception_eval", folder_ts)
    os.makedirs(save_dir, exist_ok=True)
    video_path = os.path.join(save_dir, "perception_eval.mp4")
    data_path = os.path.join(save_dir, "data.h5")
    video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (TARGET_W*3, TARGET_H*3))
    
    img_info = create_info_panel(args_cli.locomotion_checkpoint, args_cli.perception_checkpoint, timestamp_str)
    
    data_buffer = {
        "depth_front": [], "depth_rear": [], "depth_left": [], "depth_right": [],
        "gt_height": [], "pred_height": [], "diff_height": [],
        "robot_pos": [], "robot_quat": []
    }

    # 5. Markers
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/ReconstructedTerrain",
        markers={"sphere": sim_utils.SphereCfg(radius=0.02, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)))},
    )
    recon_vis = VisualizationMarkers(marker_cfg)
    recon_vis.set_visibility(True)

    grid_res = 0.05
    grid_size = 40 
    x = torch.arange(grid_size, device=env.device) * grid_res - 1.0
    y = torch.arange(grid_size, device=env.device) * grid_res - 1.0
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

    # -------------------------------------------------------------------------
    # RECORDING LOOP
    # -------------------------------------------------------------------------
    obs = env.get_observations()
    raw_env = env.unwrapped
    step = 0
    print(f"[INFO] Starting Recording ({args_cli.video_length} steps)...")
    
    with torch.inference_mode():
        while simulation_app.is_running() and step < args_cli.video_length:
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

            # Get Raw Inputs
            d_front = raw_env._tiled_camera_front.data.output["distance_to_image_plane"]
            d_rear  = raw_env._tiled_camera_rear.data.output["distance_to_image_plane"]
            d_left  = raw_env._tiled_camera_left.data.output["distance_to_image_plane"]
            d_right = raw_env._tiled_camera_right.data.output["distance_to_image_plane"]
            try: rgb_top = raw_env._tiled_camera_top.data.output["rgb"]
            except: rgb_top = torch.zeros((args_cli.num_envs, 240, 424, 3), device=env.device)

            if hasattr(raw_env, "_gt_scanner"):
                trunk_z = raw_env._gt_scanner.data.pos_w[:, 2].unsqueeze(1)
                gt_points = raw_env._gt_scanner.data.ray_hits_w[..., 2] - trunk_z
                act_size = int(np.sqrt(gt_points.shape[1]))
                gt_scan = gt_points.view(args_cli.num_envs, act_size, act_size)
                gt_scan = torch.clamp(gt_scan, -2.0, 2.0).transpose(-2, -1).flip(dims=[-2, -1])
            else:
                gt_scan = torch.zeros((args_cli.num_envs, grid_size, grid_size), device=env.device)

            # --- Inference ---
            if isinstance(perception_model, DummyPerceptionModel):
                pred_scan = perception_model(d_front, gt_map=gt_scan)
            else:
                try:
                    robot_quat = raw_env._robot.data.root_quat_w
                    stack, grav = process_inputs(d_front, d_rear, d_left, d_right, robot_quat)
                    pred_scan = perception_model(stack, grav).squeeze(1) # (B, 40, 40)
                    if pred_scan.shape[-1] != gt_scan.shape[-1]:
                         pred_scan = F.interpolate(pred_scan.unsqueeze(1), size=gt_scan.shape[-2:], mode='nearest').squeeze(1)
                except Exception as e:
                    print(f"Inference Error: {e}")
                    pred_scan = gt_scan.clone() * 0.0

            diff_scan = torch.abs(gt_scan - pred_scan)

            # [FIX] Append Data using .squeeze() (Matches collect_dataset.py)
            idx = 0
            data_buffer["depth_front"].append(d_front[idx].squeeze().cpu().numpy())
            data_buffer["depth_rear"].append(d_rear[idx].squeeze().cpu().numpy())
            data_buffer["depth_left"].append(d_left[idx].squeeze().cpu().numpy())
            data_buffer["depth_right"].append(d_right[idx].squeeze().cpu().numpy())
            
            data_buffer["gt_height"].append(gt_scan[idx].squeeze().cpu().numpy())
            data_buffer["pred_height"].append(pred_scan[idx].squeeze().cpu().numpy())
            data_buffer["diff_height"].append(diff_scan[idx].squeeze().cpu().numpy())
            
            data_buffer["robot_pos"].append(raw_env._robot.data.root_pos_w[idx].cpu().numpy())
            data_buffer["robot_quat"].append(raw_env._robot.data.root_quat_w[idx].cpu().numpy())

            # Visualization 
            if args_cli.num_envs > 0:
                robot_pos = raw_env._robot.data.root_pos_w[idx]
                robot_quat = raw_env._robot.data.root_quat_w[idx]
                curr_size = pred_scan.shape[-1]
                if grid_x.shape[0] != curr_size:
                    x = torch.arange(curr_size, device=env.device) * grid_res - 1.0
                    y = torch.arange(curr_size, device=env.device) * grid_res - 1.0
                    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
                z_local = pred_scan[idx]
                local_pts = torch.stack([grid_x.flatten(), grid_y.flatten(), z_local.flatten()], dim=-1)
                world_pts = robot_pos + quat_apply(robot_quat.repeat(local_pts.shape[0], 1), local_pts)
                recon_vis.visualize(world_pts)

            img_front = process_image(d_front[idx], "Front")
            img_rear  = process_image(d_rear[idx], "Rear")
            img_top   = process_image(rgb_top[idx], "Top", is_depth=False)
            img_left  = process_image(d_left[idx], "Left")
            img_right = process_image(d_right[idx], "Right")
            img_gt   = process_map_centered(gt_scan[idx], "GT Height")
            img_pred = process_map_centered(pred_scan[idx], "Reconstruction")
            img_diff = process_map_centered(diff_scan[idx], "Difference")

            row1 = np.hstack([img_front, img_rear, img_top])
            row2 = np.hstack([img_left, img_right, img_info])
            row3 = np.hstack([img_gt, img_pred, img_diff])
            full_frame = np.vstack([row1, row2, row3])
            
            video_writer.write(full_frame)
            step += 1
            if step % 50 == 0: print(f"Recording... {step}/{args_cli.video_length}")

    video_writer.release()
    print(f"[INFO] Video saved to {video_path}")
    
    # -------------------------------------------------------------------------
    # [FIX] SAVING LOGIC (Matching collect_dataset.py)
    # -------------------------------------------------------------------------
    print(f"[INFO] Saving HDF5 to {data_path}...")
    with h5py.File(data_path, 'w') as f:
        # 1. Depth -> UInt16 mm
        f.create_dataset("depth_front", data=to_uint16_mm(data_buffer["depth_front"]), compression="gzip")
        f.create_dataset("depth_rear",  data=to_uint16_mm(data_buffer["depth_rear"]),  compression="gzip")
        f.create_dataset("depth_left",  data=to_uint16_mm(data_buffer["depth_left"]),  compression="gzip")
        f.create_dataset("depth_right", data=to_uint16_mm(data_buffer["depth_right"]), compression="gzip")
        
        # 2. Maps -> Int16 mm
        f.create_dataset("gt_height",   data=to_int16_mm(data_buffer["gt_height"]),    compression="gzip")
        f.create_dataset("pred_height", data=to_int16_mm(data_buffer["pred_height"]),  compression="gzip")
        f.create_dataset("diff_height", data=to_int16_mm(data_buffer["diff_height"]),  compression="gzip")
        
        # 3. Pose -> Float32
        f.create_dataset("robot_pos",   data=np.array(data_buffer["robot_pos"], dtype=np.float32))
        f.create_dataset("robot_quat",  data=np.array(data_buffer["robot_quat"], dtype=np.float32))
        
    print("[INFO] Data saved.")

    print("[INFO] Entering Keep-Alive mode. Press Ctrl+C to exit.")
    with torch.inference_mode():
        while simulation_app.is_running():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()