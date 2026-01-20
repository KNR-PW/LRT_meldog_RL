#!/usr/bin/env python3
# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate perception model with 3D visualization and video recording.

Supports:
- V5 (HeightmapConvGRU): Temporal model with hidden state
- V6 (HeightmapAutoregressive): Autoregressive model (TODO)

Features:
- 3D visualization of predicted terrain as point cloud
- Video recording with depth images, sparse map, GT, prediction
- Data saving to H5 for offline analysis

Usage:
    python scripts/perception/evaluate_perception.py \
        --task Meldog-RL-Dataset-Rough-v0 \
        --locomotion_checkpoint logs/locomotion/LM_rough_sim_.../model_500.pt \
        --perception_checkpoint logs/perception/PM_v5_rough_.../model_best.pt \
        --video_length 500

Output:
    logs/perception_eval/PE_{model}_{timestamp}/
    ├── eval.mp4
    └── data.h5
"""

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "source"))

from isaaclab.app import AppLauncher


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Arguments
parser = argparse.ArgumentParser(description="Evaluate perception model for Meldog.")
parser.add_argument("--task", type=str, default="Meldog-RL-Dataset-Rough-v0", help="Task with cameras enabled.")
parser.add_argument("--num_envs", type=int, default=8, help="Number of parallel robots.")
parser.add_argument("--locomotion_checkpoint", type=str, required=True, help="Locomotion policy checkpoint.")
parser.add_argument("--perception_checkpoint", type=str, default=None, help="Perception model checkpoint.")
parser.add_argument("--video_length", type=int, default=1000, help="Recording length in steps.")
parser.add_argument("--model", type=str, default="v5", choices=["v5", "v6"], help="Model type.")

# Visualization options
parser.add_argument("--vis_model", type=str2bool, default=True, help="Visualize model prediction as 3D points.")
parser.add_argument("--vis_sparse", type=str2bool, default=False, help="Visualize sparse map as 3D points.")

# Model parameters (auto-detected from checkpoint if available)
parser.add_argument("--gru_hidden", type=int, default=128, help="ConvGRU hidden channels (V5).")
parser.add_argument("--gru_layers", type=int, default=2, help="ConvGRU layers (V5).")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ============================================================================

import gymnasium as gym
import torch
import numpy as np
import cv2
import h5py

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import quat_apply, euler_xyz_from_quat, quat_from_euler_xyz
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

import meldog_rl
from meldog_rl import envs, agents
from meldog_rl.models.perception import HeightmapConvGRU, HeightmapAutoregressive, DepthProjector
from meldog_rl.utils import make_evaluation_dir


# =============================================================================
# CONSTANTS
# =============================================================================

MAP_SIZE = 40
MAP_RES = 0.05
TARGET_W = 424
TARGET_H = 240


# =============================================================================
# VISUALIZATION HELPERS
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


def process_image(img_tensor, title, colormap=cv2.COLORMAP_JET, is_depth=True):
    """Process depth or RGB image for video frame."""
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
    """Process heightmap for video frame with centered display."""
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


def create_footer_panel(loco_name, perc_name, model_type, timestamp):
    """Create footer with checkpoint info."""
    footer = np.zeros((60, TARGET_W * 3, 3), dtype=np.uint8)
    cv2.putText(footer, f"Loco: {os.path.basename(loco_name) if loco_name else 'N/A'}", 
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(footer, f"Perc: {os.path.basename(perc_name) if perc_name else 'N/A'} [{model_type.upper()}]", 
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(footer, f"Time: {timestamp}", 
                (TARGET_W * 2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return footer


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Evaluate perception model."""
    
    # Get configs
    env_cfg = gym.spec(args_cli.task).kwargs["env_cfg_entry_point"]()
    agent_cfg = gym.spec(args_cli.task).kwargs["rsl_rl_cfg_entry_point"]()
    
    # Configure
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device else "cuda:0"
    
    print(f"[INFO] Creating environment: {args_cli.task}")
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    
    # Load locomotion policy
    print(f"[INFO] Loading locomotion policy: {args_cli.locomotion_checkpoint}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=env.device)
    runner.load(args_cli.locomotion_checkpoint)
    policy = runner.get_inference_policy(device=env.device)
    
    # Create depth projector
    projector = DepthProjector(map_size=MAP_SIZE, map_res=MAP_RES, device=env.device)
    
    # Load model config from checkpoint if available
    gru_hidden = args_cli.gru_hidden
    gru_layers = args_cli.gru_layers
    
    if args_cli.perception_checkpoint:
        checkpoint = torch.load(args_cli.perception_checkpoint, map_location=env.device, weights_only=False)
        if isinstance(checkpoint, dict) and 'config' in checkpoint:
            saved_config = checkpoint['config']
            gru_hidden = saved_config.get('gru_hidden', gru_hidden)
            gru_layers = saved_config.get('gru_layers', gru_layers)
            print(f"[INFO] Auto-detected from checkpoint: gru_hidden={gru_hidden}, gru_layers={gru_layers}")
    
    # Create model
    if args_cli.model == "v5":
        model = HeightmapConvGRU(
            gru_hidden=gru_hidden,
            gru_layers=gru_layers
        ).to(env.device)
    else:
        model = HeightmapAutoregressive().to(env.device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Model parameters: {n_params:,}")
    
    model.eval()
    
    # Load perception weights
    if args_cli.perception_checkpoint:
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            epoch = checkpoint.get('epoch', 'unknown')
            print(f"[INFO] Loaded perception model from epoch {epoch}")
        else:
            model.load_state_dict(checkpoint)
            print(f"[INFO] Loaded perception model")
    else:
        print("[WARNING] No perception checkpoint provided. Using untrained model.")
    
    # Output directory
    save_dir = make_evaluation_dir("perception", args_cli.model)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    video_path = save_dir / "eval.mp4"
    data_path = save_dir / "data.h5"
    
    video_writer = cv2.VideoWriter(
        str(video_path), 
        cv2.VideoWriter_fourcc(*'mp4v'), 
        30.0, 
        (TARGET_W * 3, TARGET_H * 3 + 60)
    )
    footer_img = create_footer_panel(
        args_cli.locomotion_checkpoint,
        args_cli.perception_checkpoint,
        args_cli.model,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    
    print(f"[INFO] Saving to: {save_dir}")
    
    # Create visualization markers
    sparse_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/SparseMap",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.015,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0))
            )
        },
    )
    sparse_vis = VisualizationMarkers(sparse_marker_cfg)
    
    recon_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/ReconstructedTerrain",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.015,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))
            )
        },
    )
    recon_vis = VisualizationMarkers(recon_marker_cfg)
    
    # Grid for 3D visualization
    grid_range = (MAP_SIZE * MAP_RES) / 2.0
    x_coords = torch.linspace(grid_range - MAP_RES/2, -grid_range + MAP_RES/2, MAP_SIZE, device=env.device)
    y_coords = torch.linspace(grid_range - MAP_RES/2, -grid_range + MAP_RES/2, MAP_SIZE, device=env.device)
    grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='ij')
    grid_x = grid_x.flatten()
    grid_y = grid_y.flatten()
    
    # Data buffers
    buffers = [
        {
            "depth_front": [], "depth_rear": [], "depth_left": [], "depth_right": [],
            "gt_height": [], "pred_height": [], "sparse_height": [], "occlusion_mask": [],
            "diff_height": [],
            "robot_pos": [], "robot_quat": []
        }
        for _ in range(args_cli.num_envs)
    ]
    
    raw_env = env.unwrapped
    
    # Verify cameras
    if raw_env._cameras.get("front") is None:
        print("[ERROR] Cameras not configured! Use a Dataset task.")
        env.close()
        return
    
    obs = env.get_observations()
    trunk_link_idx = raw_env._robot.find_bodies("trunk_link")[0][0]
    
    # Initialize hidden states (V5 only)
    hidden_states = [None] * args_cli.num_envs
    
    # V6 state tracking
    prev_outputs = [torch.zeros(1, 1, MAP_SIZE, MAP_SIZE, device=env.device) for _ in range(args_cli.num_envs)]
    prev_valids = [torch.zeros(1, 1, MAP_SIZE, MAP_SIZE, device=env.device) for _ in range(args_cli.num_envs)]
    prev_positions = [None] * args_cli.num_envs
    prev_yaws = [None] * args_cli.num_envs
    
    step = 0
    print(f"[INFO] Starting recording ({args_cli.video_length} steps, {args_cli.model.upper()} mode)...")
    
    with torch.inference_mode():
        while simulation_app.is_running() and step < args_cli.video_length:
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            
            # Reset state for terminated environments
            for i in range(args_cli.num_envs):
                if dones[i]:
                    hidden_states[i] = None
                    prev_outputs[i] = torch.zeros(1, 1, MAP_SIZE, MAP_SIZE, device=env.device)
                    prev_valids[i] = torch.zeros(1, 1, MAP_SIZE, MAP_SIZE, device=env.device)
                    prev_positions[i] = None
                    prev_yaws[i] = None
            
            # Get camera data
            d_front = raw_env._cameras["front"].data.output["distance_to_image_plane"]
            d_rear  = raw_env._cameras["rear"].data.output["distance_to_image_plane"]
            d_left  = raw_env._cameras["left"].data.output["distance_to_image_plane"]
            d_right = raw_env._cameras["right"].data.output["distance_to_image_plane"]
            
            d_stack = torch.stack([
                d_front.squeeze(-1), d_rear.squeeze(-1), 
                d_left.squeeze(-1), d_right.squeeze(-1)
            ], dim=1)
            
            # RGB top for video
            rgb_top = None
            if raw_env._cameras.get("top") is not None:
                try:
                    rgb_top = raw_env._cameras["top"].data.output["rgb"]
                except:
                    pass
            if rgb_top is None:
                rgb_top = torch.zeros((args_cli.num_envs, TARGET_H, TARGET_W, 3), device=env.device)
            
            # Robot state
            robot_quat = raw_env._robot.data.root_quat_w
            robot_pos = raw_env._robot.data.root_pos_w
            trunk_pos_w = raw_env._robot.data.body_pos_w[:, trunk_link_idx]
            
            _, _, yaw = euler_xyz_from_quat(robot_quat)
            yaw_quat = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
            
            # Gravity vector
            w, x, y, z = robot_quat[:, 0], robot_quat[:, 1], robot_quat[:, 2], robot_quat[:, 3]
            gx = -2 * (x * z + w * y)
            gy = -2 * (y * z - w * x)
            gz = -(1 - 2 * (x * x + y * y))
            grav = torch.stack([gx, gy, gz], dim=1)
            
            # Project depth to sparse map
            sparse_map, occlusion_mask = projector(d_stack, robot_quat)
            
            # Run perception model
            if args_cli.model == "v5":
                # V5: Use hidden state
                pred_scan, new_hidden = model(sparse_map, occlusion_mask, grav, hidden_states[0])
                hidden_states[0] = new_hidden
            else:
                # V6: Use previous output (simplified - treats all envs together)
                # TODO: Proper per-env tracking with coordinate transform
                pred_scan = model(
                    sparse_map, occlusion_mask,
                    prev_outputs[0].expand(args_cli.num_envs, -1, -1, -1),
                    prev_valids[0].expand(args_cli.num_envs, -1, -1, -1),
                    grav
                )
                prev_outputs[0] = pred_scan[0:1].detach()
                prev_valids[0] = torch.ones(1, 1, MAP_SIZE, MAP_SIZE, device=env.device)
            
            # GT height
            if hasattr(raw_env, "_gt_scanner") and raw_env._gt_scanner is not None:
                trunk_z = raw_env._gt_scanner.data.pos_w[:, 2].unsqueeze(1)
                gt_scan = (raw_env._gt_scanner.data.ray_hits_w[..., 2] - trunk_z)
                gt_scan = gt_scan.view(args_cli.num_envs, MAP_SIZE, MAP_SIZE)
                gt_scan = torch.clamp(gt_scan, -2.0, 2.0).transpose(-2, -1).flip(dims=[-2, -1])
            else:
                gt_scan = torch.zeros((args_cli.num_envs, MAP_SIZE, MAP_SIZE), device=env.device)
            
            diff_scan = torch.abs(gt_scan - pred_scan.squeeze(1))
            
            # 3D Visualization
            n = args_cli.num_envs
            m = MAP_SIZE * MAP_SIZE
            
            local_grid_xy = torch.stack([grid_x, grid_y, torch.zeros_like(grid_x)], dim=-1).repeat(n, 1, 1)
            rotated_grid_xy = quat_apply(
                yaw_quat.repeat_interleave(m, dim=0), 
                local_grid_xy.view(-1, 3)
            ).view(n, m, 3)
            
            if args_cli.vis_sparse:
                sparse_world_pts = trunk_pos_w.unsqueeze(1) + rotated_grid_xy
                sparse_world_pts[..., 2] += sparse_map.view(n, m)
                sparse_vis.visualize(sparse_world_pts.view(-1, 3))
            
            if args_cli.vis_model:
                recon_world_pts = trunk_pos_w.unsqueeze(1) + rotated_grid_xy
                recon_world_pts[..., 2] += pred_scan.view(n, m)
                recon_vis.visualize(recon_world_pts.view(-1, 3))
            
            # Store data
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
            
            # Video frame (env 0)
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
            if step % 50 == 0:
                print(f"Recording... {step}/{args_cli.video_length}")
    
    video_writer.release()
    print(f"[INFO] Video saved to {video_path}")
    
    # Save data
    print(f"[INFO] Saving data to {data_path}...")
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
            
            grp.create_dataset("robot_pos",  data=np.array(buffers[i]["robot_pos"], dtype=np.float32))
            grp.create_dataset("robot_quat", data=np.array(buffers[i]["robot_quat"], dtype=np.float32))
    
    print(f"[INFO] Data saved.")
    print(f"[INFO] Entering keep-alive mode (Ctrl+C to exit)...")
    
    with torch.inference_mode():
        while simulation_app.is_running():
            obs, _, _, _ = env.step(policy(obs))
    
    env.close()
    print("[DONE] Evaluation complete.")


if __name__ == "__main__":
    main()
    simulation_app.close()
