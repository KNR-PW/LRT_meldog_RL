# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Perception Evaluation Script for Meldog (V5 - Temporal with ConvGRU)
- Maintains hidden state across frames
- Shows temporal accumulation effect
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

def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

# Argument Parsing
parser = argparse.ArgumentParser(description="Evaluate Temporal Perception Model V5 for Meldog")
parser.add_argument("--task", type=str, default="Template-Meldog-Simple-Locomotion-Policy-Direct-v0")
parser.add_argument("--num_envs", type=int, default=8, help="Number of parallel robots")
parser.add_argument("--locomotion_checkpoint", type=str, required=True)
parser.add_argument("--perception_checkpoint", type=str, default=None)
parser.add_argument("--video_length", type=int, default=1000)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--vis_model", type=str2bool, default=True)
parser.add_argument("--vis_sparse", type=str2bool, default=False)
# GRU parameters (must match training)
parser.add_argument("--gru_hidden", type=int, default=128)
parser.add_argument("--gru_layers", type=int, default=2)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab.sim as sim_utils
import isaaclab_rl.rsl_rl as rsl_rl_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import quat_apply, euler_xyz_from_quat, quat_from_euler_xyz
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner

import meldog_simple_locomotion_policy.tasks 
from preprocess_perception import DepthProjector

MAP_SIZE = 40 
MAP_RES = 0.05
TARGET_W = 424
TARGET_H = 240

# -----------------------------------------------------------------------------
# CONVOLUTIONAL GRU (same as training)
# -----------------------------------------------------------------------------
class ConvGRUCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        self.conv_gates = nn.Conv2d(
            input_channels + hidden_channels, 
            2 * hidden_channels,
            kernel_size, padding=padding
        )
        self.conv_candidate = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels,
            kernel_size, padding=padding
        )
    
    def forward(self, x, h_prev):
        if h_prev is None:
            B, _, H, W = x.shape
            h_prev = torch.zeros(B, self.hidden_channels, H, W, device=x.device, dtype=x.dtype)
        
        combined = torch.cat([x, h_prev], dim=1)
        gates = torch.sigmoid(self.conv_gates(combined))
        reset_gate, update_gate = gates.chunk(2, dim=1)
        combined_reset = torch.cat([x, reset_gate * h_prev], dim=1)
        candidate = torch.tanh(self.conv_candidate(combined_reset))
        h_new = (1 - update_gate) * h_prev + update_gate * candidate
        return h_new


class ConvGRU(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_layers=2, kernel_size=3):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_ch = input_channels if i == 0 else hidden_channels
            self.cells.append(ConvGRUCell(in_ch, hidden_channels, kernel_size))
    
    def forward(self, x, h_prev=None):
        if h_prev is None:
            h_prev = [None] * self.num_layers
        h_new = []
        current = x
        for i, cell in enumerate(self.cells):
            current = cell(current, h_prev[i])
            h_new.append(current)
        return current, h_new


# -----------------------------------------------------------------------------
# TEMPORAL MODEL (same as training)
# -----------------------------------------------------------------------------
class SparseMapRefinerTemporal(nn.Module):
    def __init__(self, gru_hidden=128, gru_layers=2):
        super().__init__()
        
        self.enc1 = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.ReLU()
        )
        
        self.mlp_gravity = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 64)
        )
        
        self.conv_gru = ConvGRU(
            input_channels=128 + 64,
            hidden_channels=gru_hidden,
            num_layers=gru_layers,
            kernel_size=3
        )
        
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(gru_hidden, 64, 4, 2, 1), nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, 32, 4, 2, 1), nn.ReLU()
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(32 + 32, 16, 4, 2, 1), nn.ReLU()
        )
        
        self.final_conv = nn.Conv2d(16 + 2, 1, 3, padding=1)

    def forward(self, x, mask, grav, h_prev=None):
        x_in = torch.cat([x, mask], dim=1)
        
        s1 = self.enc1(x_in)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        
        B, _, H, W = s3.shape
        grav_embed = self.mlp_gravity(grav).view(B, 64, 1, 1).expand(B, 64, H, W)
        s3_grav = torch.cat([s3, grav_embed], dim=1)
        
        gru_out, h_new = self.conv_gru(s3_grav, h_prev)
        
        up1 = self.dec1(gru_out)
        up2 = self.dec2(torch.cat([up1, s2], dim=1))
        up3 = self.dec3(torch.cat([up2, s1], dim=1))
        
        pred = self.final_conv(torch.cat([up3, x_in], dim=1))
        
        return pred, h_new


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

def process_map_centered(map_tensor, title, is_diff=False):
    data = map_tensor.squeeze().cpu().numpy()
    
    if is_diff:
        norm = (np.clip(data / 0.2, 0.0, 1.0) * 255.0).astype(np.uint8)
        color_img = cv2.applyColorMap(norm, cv2.COLORMAP_HOT)
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
    footer = np.zeros((60, TARGET_W * 3, 3), dtype=np.uint8)
    cv2.putText(footer, f"Loco: {os.path.basename(loco_name) if loco_name else 'N/A'}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(footer, f"Perc: {os.path.basename(perc_name) if perc_name else 'N/A'} [TEMPORAL]", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(footer, f"Time: {timestamp}", (TARGET_W * 2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return footer

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

    # Load model config from checkpoint if available
    gru_hidden = args_cli.gru_hidden
    gru_layers = args_cli.gru_layers
    
    if args_cli.perception_checkpoint:
        checkpoint = torch.load(args_cli.perception_checkpoint, map_location=env.device)
        if isinstance(checkpoint, dict) and 'config' in checkpoint:
            saved_config = checkpoint['config']
            gru_hidden = saved_config.get('gru_hidden', gru_hidden)
            gru_layers = saved_config.get('gru_layers', gru_layers)
            print(f"[INFO] Auto-detected from checkpoint: gru_hidden={gru_hidden}, gru_layers={gru_layers}")

    model = SparseMapRefinerTemporal(
        gru_hidden=gru_hidden,
        gru_layers=gru_layers
    ).to(env.device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Model parameters: {n_params:,}")
    
    model.eval()
    
    if args_cli.perception_checkpoint:
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            epoch = checkpoint.get('epoch', 'unknown')
            print(f"[INFO] Loaded Temporal Perception Model from Epoch {epoch}")
        else:
            model.load_state_dict(checkpoint)
            print(f"[INFO] Loaded Temporal Perception Model")
    else:
        print("[WARNING] No perception checkpoint provided. Using untrained model.")

    save_dir = os.path.join("logs", "perception_eval", datetime.now().strftime("PE_V5_Temporal_%Y-%m-%d_%H-%M-%S"))
    os.makedirs(save_dir, exist_ok=True)
    
    video_path = os.path.join(save_dir, "eval.mp4")
    data_path = os.path.join(save_dir, "data.h5")
    video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (TARGET_W*3, TARGET_H*3 + 60))
    footer_img = create_footer_panel(args_cli.locomotion_checkpoint, args_cli.perception_checkpoint, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    sparse_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/SparseMap",
        markers={"sphere": sim_utils.SphereCfg(radius=0.015, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)))},
    )
    sparse_vis = VisualizationMarkers(sparse_marker_cfg)

    recon_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/ReconstructedTerrain",
        markers={"sphere": sim_utils.SphereCfg(radius=0.015, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)))},
    )
    recon_vis = VisualizationMarkers(recon_marker_cfg)

    grid_range = (MAP_SIZE * MAP_RES) / 2.0
    x_coords = torch.linspace(grid_range - MAP_RES/2, -grid_range + MAP_RES/2, MAP_SIZE, device=env.device)
    y_coords = torch.linspace(grid_range - MAP_RES/2, -grid_range + MAP_RES/2, MAP_SIZE, device=env.device)
    grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='ij')
    grid_x = grid_x.flatten()
    grid_y = grid_y.flatten()

    buffers = [
        {
            "depth_front": [], "depth_rear": [], "depth_left": [], "depth_right": [],
            "gt_height": [], "pred_height": [], "sparse_height": [], "occlusion_mask": [],
            "diff_height": [],
            "robot_pos": [], "robot_quat": []
        }
        for _ in range(args_cli.num_envs)
    ]

    obs = env.get_observations()
    raw_env = env.unwrapped
    trunk_link_idx = raw_env._robot.find_bodies("trunk_link")[0][0]

    # Initialize hidden state for each environment
    hidden_states = [None] * args_cli.num_envs
    
    step = 0
    print(f"[INFO] Starting Recording ({args_cli.video_length} steps, TEMPORAL mode)...")

    with torch.inference_mode():
        while simulation_app.is_running() and step < args_cli.video_length:
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

            # Reset hidden state for environments that terminated
            for i in range(args_cli.num_envs):
                if dones[i]:
                    hidden_states[i] = None

            d_front = raw_env._tiled_camera_front.data.output["distance_to_image_plane"]
            d_rear  = raw_env._tiled_camera_rear.data.output["distance_to_image_plane"]
            d_left  = raw_env._tiled_camera_left.data.output["distance_to_image_plane"]
            d_right = raw_env._tiled_camera_right.data.output["distance_to_image_plane"]
            
            d_stack = torch.stack([d_front.squeeze(-1), d_rear.squeeze(-1), d_left.squeeze(-1), d_right.squeeze(-1)], dim=1)
            
            try: rgb_top = raw_env._tiled_camera_top.data.output["rgb"]
            except: rgb_top = torch.zeros((args_cli.num_envs, 240, 424, 3), device=env.device)

            robot_quat = raw_env._robot.data.root_quat_w
            robot_pos = raw_env._robot.data.root_pos_w
            trunk_pos_w = raw_env._robot.data.body_pos_w[:, trunk_link_idx]
            
            _, _, yaw = euler_xyz_from_quat(robot_quat)
            yaw_quat = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)

            w, x, y, z = robot_quat[:, 0], robot_quat[:, 1], robot_quat[:, 2], robot_quat[:, 3]
            gx, gy, gz = -2*(x*z + w*y), -2*(y*z - w*x), -(1 - 2*(x*x + y*y))
            grav = torch.stack([gx, gy, gz], dim=1)

            sparse_map, occlusion_mask = projector(d_stack, robot_quat)
            
            # TEMPORAL: Run model with hidden state
            # For simplicity, we process all envs together with a single hidden state
            # (In production, you'd want per-env hidden states)
            pred_scan, new_hidden = model(sparse_map, occlusion_mask, grav, hidden_states[0])
            hidden_states[0] = new_hidden  # Update hidden state

            # GT Height
            if hasattr(raw_env, "_gt_scanner"):
                trunk_z = raw_env._gt_scanner.data.pos_w[:, 2].unsqueeze(1)
                gt_scan = (raw_env._gt_scanner.data.ray_hits_w[..., 2] - trunk_z).view(args_cli.num_envs, MAP_SIZE, MAP_SIZE)
                gt_scan = torch.clamp(gt_scan, -2.0, 2.0).transpose(-2, -1).flip(dims=[-2, -1])
            else:
                gt_scan = torch.zeros((args_cli.num_envs, MAP_SIZE, MAP_SIZE), device=env.device)
            
            diff_scan = torch.abs(gt_scan - pred_scan.squeeze(1))

            n = args_cli.num_envs
            m = MAP_SIZE * MAP_SIZE
            
            local_grid_xy = torch.stack([grid_x, grid_y, torch.zeros_like(grid_x)], dim=-1).repeat(n, 1, 1)
            rotated_grid_xy = quat_apply(yaw_quat.repeat_interleave(m, dim=0), local_grid_xy.view(-1, 3)).view(n, m, 3)

            if args_cli.vis_sparse:
                sparse_world_pts = trunk_pos_w.unsqueeze(1) + rotated_grid_xy
                sparse_world_pts[..., 2] += sparse_map.view(n, m)
                sparse_vis.visualize(sparse_world_pts.view(-1, 3))

            if args_cli.vis_model:
                recon_world_pts = trunk_pos_w.unsqueeze(1) + rotated_grid_xy
                recon_world_pts[..., 2] += pred_scan.view(n, m)
                recon_vis.visualize(recon_world_pts.view(-1, 3))

            for i in range(args_cli.num_envs):
                buffers[i]["depth_front"].append(d_front[i].squeeze().cpu().numpy())
                buffers[i]["depth_rear"].append(d_rear[i].squeeze().cpu().numpy())
                buffers[i]["depth_left"].append(d_left[i].squeeze().cpu().numpy())
                buffers[i]["depth_right"].append(d_right[i].squeeze().cpu().numpy())
                buffers[i]["gt_height"].append(gt_scan[i].squeeze().cpu().numpy())
                buffers[i]["pred_height"].append(pred_scan[i].squeeze().cpu().numpy())
                buffers[i]["sparse_height"].append(sparse_map[i].squeeze().cpu().numpy())
                buffers[i]["occlusion_mask"].append(occlusion_mask[i].squeeze().cpu().numpy())
                buffers[i]["diff_height"].append(diff_scan[i].squeeze().cpu().numpy())
                buffers[i]["robot_pos"].append(robot_pos[i].cpu().numpy())
                buffers[i]["robot_quat"].append(robot_quat[i].cpu().numpy())

            idx = 0
            img_grid = [
                process_image(d_stack[idx, 0], "Front"), 
                process_image(d_stack[idx, 1], "Rear"), 
                process_image(rgb_top[idx], "Top", is_depth=False),
                process_image(d_stack[idx, 2], "Left"), 
                process_image(d_stack[idx, 3], "Right"), 
                process_map_centered(sparse_map[idx], "Sparse Map"),
                process_map_centered(gt_scan[idx], "GT Height"), 
                process_map_centered(pred_scan[idx], "Model Output"), 
                process_map_centered(diff_scan[idx], "Difference", is_diff=True)
            ]

            row1 = np.hstack(img_grid[0:3])
            row2 = np.hstack(img_grid[3:6])
            row3 = np.hstack(img_grid[6:9])
            video_writer.write(np.vstack([row1, row2, row3, footer_img]))
            
            step += 1
            if step % 50 == 0: print(f"Recording... {step}/{args_cli.video_length}")

    video_writer.release()
    print(f"[INFO] Video saved to {video_path}")

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
            grp.create_dataset("diff_height",    data=to_int16_mm(buffers[i]["diff_height"]),  compression="gzip")
            grp.create_dataset("occlusion_mask", data=np.array(buffers[i]["occlusion_mask"], dtype=np.uint8), compression="gzip")
            
            grp.create_dataset("robot_pos",   data=np.array(buffers[i]["robot_pos"], dtype=np.float32))
            grp.create_dataset("robot_quat",  data=np.array(buffers[i]["robot_quat"], dtype=np.float32))

    print(f"[INFO] Data saved. Entering Keep-Alive mode.")
    with torch.inference_mode():
        while simulation_app.is_running():
            obs, _, _, _ = env.step(policy(obs))
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()