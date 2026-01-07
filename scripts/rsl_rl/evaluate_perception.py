# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Perception Evaluation Script for Meldog (V4 - U-Net + Occlusion Mask)
- Runs Locomotion + Perception.
- Saves full dataset (all envs) to HDF5.
- Visualization: 
    - UI: Grid of cameras and height maps.
    - Isaac Sim: Green spheres (Sparse Input), Blue spheres (Model Output).
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import cv2
import h5py
from datetime import datetime

from isaaclab.app import AppLauncher

# Argument Parsing
parser = argparse.ArgumentParser(description="Evaluate Perception Model V4 for Meldog")
parser.add_argument("--task", type=str, default="Template-Meldog-Simple-Locomotion-Policy-Direct-v0")
parser.add_argument("--num_envs", type=int, default=8, help="Number of parallel robots")
parser.add_argument("--locomotion_checkpoint", type=str, required=True, help="Path to locomotion policy .pt file")
parser.add_argument("--perception_checkpoint", type=str, default=None, help="Path to perception model (.pt).")
parser.add_argument("--video_length", type=int, default=1000, help="Length of recording in steps")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL Agent config entry point")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab_rl.rsl_rl as rsl_rl_utils
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner
from isaaclab.utils.markers import VisualizationMarkers
from isaaclab.utils.markers.config import VisualizationMarkersCfg
from isaaclab.utils.math import quat_apply

# Import Projector and Utilities
import meldog_simple_locomotion_policy.tasks 
from preprocess_perception import DepthProjector

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
MAP_SIZE = 40 
MAP_RES = 0.05
TARGET_W = 424
TARGET_H = 240

# -----------------------------------------------------------------------------
# MODEL: SPARSE MAP REFINER (V4 U-Net)
# -----------------------------------------------------------------------------
class SparseMapRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = self.conv_block(2, 32)   
        self.enc2 = self.conv_block(32, 64)  
        self.enc3 = self.conv_block(64, 128) 
        
        self.mlp_gravity = nn.Sequential(
            nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 32)
        )

        self.dec1 = self.up_block(160, 64)
        self.dec2 = self.up_block(128, 32)       
        
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, padding_mode='replicate'), 
            nn.BatchNorm2d(out_c), nn.ReLU(), nn.MaxPool2d(2)
        )

    def up_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), 
            nn.Conv2d(in_c, out_c, 3, padding=1, padding_mode='replicate'), 
            nn.BatchNorm2d(out_c), nn.ReLU()
        )

    def forward(self, x, mask, grav):
        x_in = torch.cat([x, mask], dim=1) 
        s1 = self.enc1(x_in)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)

        B, _, H, W = s3.shape
        grav_embed = self.mlp_gravity(grav).unsqueeze(-1).unsqueeze(-1).expand(B, 32, H, W)
        x = torch.cat([s3, grav_embed], dim=1) 

        x = self.dec1(x)
        x = torch.cat([x, s2], dim=1)
        x = self.dec2(x)
        x = torch.cat([x, s1], dim=1)
        
        x = self.final_upsample(x)
        return self.final_conv(x)

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def to_uint16_mm(data_list):
    arr = np.array(data_list, dtype=np.float32)
    invalid_mask = np.isinf(arr) | (arr > 5.0) | (arr <= 0.0)
    arr[invalid_mask] = 5.0
    return (arr * 1000.0).astype(np.uint16)

def to_int16_mm(data_list):
    arr = np.array(data_list, dtype=np.float32)
    return (np.clip(arr, -32.0, 32.0) * 1000.0).astype(np.int16)

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

def process_map_centered(map_tensor, title, is_mask=False, is_diff=False):
    data = map_tensor.squeeze().cpu().numpy()
    
    if is_diff:
        norm = (np.clip(data / 0.2, 0.0, 1.0) * 255.0).astype(np.uint8)
        color_img = cv2.applyColorMap(norm, cv2.COLORMAP_HOT)
    elif is_mask:
        norm = (np.clip(data, 0.0, 1.0) * 255.0).astype(np.uint8)
        color_img = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
    else:
        norm = (np.clip((data + 0.5) / 1.0, 0.0, 1.0) * 255.0).astype(np.uint8)
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

def create_footer_panel(loco_name, perc_name, timestamp):
    panel_h = 60
    panel_w = TARGET_W * 3
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    
    def fmt_name(path):
        if not path: return "None"
        parts = path.replace("\\", "/").split("/")
        return f".../{parts[-2]}/{parts[-1]}" if len(parts) > 2 else path

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(panel, f"Loco: {fmt_name(loco_name)}", (20, 25), font, 0.4, (220, 220, 220), 1)
    cv2.putText(panel, f"Perc: {fmt_name(perc_name)}", (20, 45), font, 0.4, (220, 220, 220), 1)
    cv2.putText(panel, f"Timestamp: {timestamp}", (panel_w - 280, 35), font, 0.4, (220, 220, 220), 1)
    return panel

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = load_cfg_from_registry(args_cli.task, args_cli.agent)
    if hasattr(agent_cfg, "to_dict"): agent_cfg = agent_cfg.to_dict()

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = rsl_rl_utils.RslRlVecEnvWrapper(env, clip_actions=agent_cfg.get("clip_actions", 1.0))

    runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=args_cli.device)
    runner.load(args_cli.locomotion_checkpoint)
    policy = runner.get_inference_policy(device=env.device)

    projector = DepthProjector(map_size=MAP_SIZE, map_res=MAP_RES, device=env.device)

    model = SparseMapRefiner().to(env.device)
    if args_cli.perception_checkpoint:
        model.load_state_dict(torch.load(args_cli.perception_checkpoint, map_location=env.device))
        print(f"[INFO] Loaded Perception: {args_cli.perception_checkpoint}")
    model.eval()

    # --- Marker Initialization ---
    sparse_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/SparsePoints",
        markers={"sphere": VisualizationMarkersCfg.VisualAttributesCfg(radius=0.015, visual_material=VisualizationMarkersCfg.VisualAttributesCfg.MaterialCfg(diffuse_color=(0.0, 1.0, 0.0)))},
    )
    model_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/ModelOutput",
        markers={"sphere": VisualizationMarkersCfg.VisualAttributesCfg(radius=0.015, visual_material=VisualizationMarkersCfg.VisualAttributesCfg.MaterialCfg(diffuse_color=(0.0, 0.0, 1.0)))},
    )
    sparse_visualizer = VisualizationMarkers(sparse_marker_cfg)
    model_visualizer = VisualizationMarkers(model_marker_cfg)

    # Grid for markers (Local coordinates)
    x = torch.linspace(-(MAP_SIZE // 2) * MAP_RES, (MAP_SIZE // 2 - 1) * MAP_RES, MAP_SIZE, device=env.device)
    y = torch.linspace(-(MAP_SIZE // 2) * MAP_RES, (MAP_SIZE // 2 - 1) * MAP_RES, MAP_SIZE, device=env.device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    local_grid_points = torch.stack([grid_x.flatten(), grid_y.flatten(), torch.zeros_like(grid_x.flatten())], dim=1) # (1600, 3)

    save_dir = os.path.join("logs", "perception_eval", datetime.now().strftime("PE_V4_%Y-%m-%d_%H-%M-%S"))
    os.makedirs(save_dir, exist_ok=True)
    
    video_path = os.path.join(save_dir, "eval.mp4")
    data_path = os.path.join(save_dir, "data.h5")
    video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (TARGET_W*3, TARGET_H*3 + 60))
    footer_img = create_footer_panel(args_cli.locomotion_checkpoint, args_cli.perception_checkpoint, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Multi-env buffers
    buffers = [
        {
            "depth_front": [], "depth_rear": [], "depth_left": [], "depth_right": [],
            "gt_height": [], "pred_height": [], "sparse_height": [], "occlusion_mask": [],
            "robot_pos": [], "robot_quat": []
        }
        for _ in range(args_cli.num_envs)
    ]

    obs = env.get_observations()
    raw_env = env.unwrapped
    step = 0
    
    print(f"[INFO] Starting Recording ({args_cli.video_length} steps for {args_cli.num_envs} envs)...")

    with torch.inference_mode():
        while simulation_app.is_running() and step < args_cli.video_length:
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

            # 1. Get Camera Data
            d_front = raw_env._tiled_camera_front.data.output["distance_to_image_plane"]
            d_rear  = raw_env._tiled_camera_rear.data.output["distance_to_image_plane"]
            d_left  = raw_env._tiled_camera_left.data.output["distance_to_image_plane"]
            d_right = raw_env._tiled_camera_right.data.output["distance_to_image_plane"]
            
            d_stack = torch.stack([d_front.squeeze(-1), d_rear.squeeze(-1), d_left.squeeze(-1), d_right.squeeze(-1)], dim=1)
            
            try: rgb_top = raw_env._tiled_camera_top.data.output["rgb"]
            except: rgb_top = torch.zeros((args_cli.num_envs, 240, 424, 3), device=env.device)

            # 2. Get Pose & Gravity
            robot_quat = raw_env._robot.data.root_quat_w
            robot_pos = raw_env._robot.data.root_pos_w
            w, qx, qy, qz = robot_quat[:, 0], robot_quat[:, 1], robot_quat[:, 2], robot_quat[:, 3]
            grav = torch.stack([-2*(qx*qz + w*qy), -2*(qy*qz - w*qx), -(1-2*(qx*qx + qy*qy))], dim=1)

            # 3. Project & Predict
            sparse_map, occlusion_mask = projector(d_stack, robot_quat)
            pred_scan = model(sparse_map, occlusion_mask, grav)

            # 4. Markers Visualization (Env 0)
            idx = 0
            # Transform Local Map to World Markers
            # Pred Height markers
            p_heights = pred_scan[idx].flatten()
            p_local = local_grid_points.clone()
            p_local[:, 2] = p_heights
            p_world = quat_apply(robot_quat[idx].repeat(MAP_SIZE*MAP_SIZE, 1), p_local) + robot_pos[idx]
            model_visualizer.visualize(p_world)

            # Sparse Height markers (only where mask > 0)
            s_mask = occlusion_mask[idx].flatten() > 0.5
            if s_mask.any():
                s_heights = sparse_map[idx].flatten()[s_mask]
                s_local = local_grid_points[s_mask].clone()
                s_local[:, 2] = s_heights
                s_world = quat_apply(robot_quat[idx].repeat(s_local.shape[0], 1), s_local) + robot_pos[idx]
                sparse_visualizer.visualize(s_world)
            else:
                sparse_visualizer.visualize(torch.zeros((1, 3), device=env.device))

            # 5. GT Height
            if hasattr(raw_env, "_gt_scanner"):
                trunk_z = raw_env._gt_scanner.data.pos_w[:, 2].unsqueeze(1)
                gt_scan = (raw_env._gt_scanner.data.ray_hits_w[..., 2] - trunk_z).view(args_cli.num_envs, MAP_SIZE, MAP_SIZE)
                gt_scan = torch.clamp(gt_scan, -2.0, 2.0).transpose(-2, -1).flip(dims=[-2, -1])
            else:
                gt_scan = torch.zeros((args_cli.num_envs, MAP_SIZE, MAP_SIZE), device=env.device)
            
            diff_scan = torch.abs(gt_scan - pred_scan.squeeze(1))

            # Store data for ALL envs
            for i in range(args_cli.num_envs):
                buffers[i]["depth_front"].append(d_front[i].squeeze().cpu().numpy())
                buffers[i]["depth_rear"].append(d_rear[i].squeeze().cpu().numpy())
                buffers[i]["depth_left"].append(d_left[i].squeeze().cpu().numpy())
                buffers[i]["depth_right"].append(d_right[i].squeeze().cpu().numpy())
                buffers[i]["gt_height"].append(gt_scan[i].squeeze().cpu().numpy())
                buffers[i]["pred_height"].append(pred_scan[i].squeeze().cpu().numpy())
                buffers[i]["sparse_height"].append(sparse_map[i].squeeze().cpu().numpy())
                buffers[i]["occlusion_mask"].append(occlusion_mask[i].squeeze().cpu().numpy())
                buffers[i]["robot_pos"].append(robot_pos[i].cpu().numpy())
                buffers[i]["robot_quat"].append(robot_quat[i].cpu().numpy())

            # UI Visualization (Env 0)
            img_grid = [
                process_image(d_stack[idx, 0], "Front"), process_image(d_stack[idx, 1], "Rear"), process_image(rgb_top[idx], "Top", is_depth=False),
                process_image(d_stack[idx, 2], "Left"), process_image(d_stack[idx, 3], "Right"), process_map_centered(occlusion_mask[idx], "Occlusion Mask", is_mask=True),
                process_map_centered(gt_scan[idx], "GT Height"), process_map_centered(pred_scan[idx], "Model Output"), process_map_centered(diff_scan[idx], "Difference", is_diff=True)
            ]

            row1 = np.hstack(img_grid[0:3])
            row2 = np.hstack(img_grid[3:6])
            row3 = np.hstack(img_grid[6:9])
            video_writer.write(np.vstack([row1, row2, row3, footer_img]))
            
            step += 1
            if step % 50 == 0: print(f"Recording... {step}/{args_cli.video_length}")

    video_writer.release()
    print(f"[INFO] Video saved to {video_path}")

    # Save ALL envs to HDF5
    print(f"[INFO] Saving data for {args_cli.num_envs} envs to {data_path}...")
    with h5py.File(data_path, 'w') as f:
        for i in range(args_cli.num_envs):
            grp = f.create_group(f"env_{i}")
            grp.create_dataset("depth_front", data=to_uint16_mm(buffers[i]["depth_front"]), compression="gzip")
            grp.create_dataset("depth_rear",  data=to_uint16_mm(buffers[i]["depth_rear"]),  compression="gzip")
            grp.create_dataset("depth_left",  data=to_uint16_mm(buffers[i]["depth_left"]),  compression="gzip")
            grp.create_dataset("depth_right", data=to_uint16_mm(buffers[i]["depth_right"]), compression="gzip")
            
            grp.create_dataset("gt_height",      data=to_int16_mm(buffers[i]["gt_height"]),    compression="gzip")
            grp.create_dataset("pred_height",    data=to_int16_mm(buffers[i]["pred_height"]),  compression="gzip")
            grp.create_dataset("sparse_height",  data=to_int16_mm(buffers[i]["sparse_height"]),compression="gzip")
            grp.create_dataset("occlusion_mask", data=np.array(buffers[i]["occlusion_mask"], dtype=np.uint8), compression="gzip")
            
            grp.create_dataset("robot_pos",   data=np.array(buffers[i]["robot_pos"], dtype=np.float32))
            grp.create_dataset("robot_quat",  data=np.array(buffers[i]["robot_quat"], dtype=np.float32))

    print(f"[INFO] Data saved. Entering Keep-Alive mode.")
    while simulation_app.is_running():
        obs, _, _, _ = env.step(policy(obs))
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()