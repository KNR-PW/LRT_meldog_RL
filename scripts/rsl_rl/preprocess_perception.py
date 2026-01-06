"""
preprocess_perception.py
Handles geometric projection from Depth Images to Top-Down Height Maps.
"""
import torch
import torch.nn as nn
import numpy as np

# --- HARDWARE CONFIGURATION ---
CAM_RES = (424, 240) # (Width, Height)
CAM_FOV = 87.0       # Horizontal FOV (Realsense D435 default)

# Extrinsics: [x, y, z] position relative to robot base, and [r, p, y] Euler angles.
CAM_EXTRINSICS = {
    "front": {"pos": [0.28, 0.0, 0.05],  "rpy": [0, 0.26, 0]},         # Front
    "rear":  {"pos": [-0.28, 0.0, 0.05], "rpy": [0, 0.26, 3.14159]},   # Rear
    "left":  {"pos": [0.0, 0.15, 0.05],  "rpy": [0, 0.26, 1.5708]},    # Left
    "right": {"pos": [0.0, -0.15, 0.05], "rpy": [0, 0.26, -1.5708]},   # Right
}

class DepthProjector(nn.Module):
    def __init__(self, map_size=40, map_res=0.05, device='cuda'):
        super().__init__()
        self.map_size = map_size 
        self.map_res = map_res   
        self.device = device
        
        # 1. Pre-compute Intrinsics (Ray Directions)
        w, h = CAM_RES
        fx = w / (2 * np.tan(np.deg2rad(CAM_FOV) / 2))
        fy = fx 
        cx, cy = w / 2, h / 2
        
        y_grid, x_grid = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        # Optical Frame: x=right, y=down, z=forward
        x_norm = (x_grid - cx) / fx
        y_norm = (y_grid - cy) / fy
        self.ray_dirs = torch.stack([x_norm, y_norm, torch.ones_like(x_norm)], dim=-1).to(device) # (H, W, 3)
        
        # 2. Pre-compute Extrinsics (Camera -> Robot Base)
        self.cam_transforms = {}
        # Matrix to align Optical Frame (Z-forward) to ROS Frame (X-forward)
        R_opt_to_ros = torch.tensor([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=torch.float32, device=device)

        for cam_name, cfg in CAM_EXTRINSICS.items():
            pos = torch.tensor(cfg['pos'], device=device, dtype=torch.float32)
            rpy = torch.tensor(cfg['rpy'], device=device, dtype=torch.float32)
            R_world = self._euler_to_mat(rpy)
            
            # Combine: Rotate Optical to ROS, then Rotate by Extrinsic
            final_R = R_world @ R_opt_to_ros
            self.cam_transforms[cam_name] = (final_R, pos)

    def _euler_to_mat(self, rpy):
        r, p, y = rpy[0], rpy[1], rpy[2]
        Rx = torch.tensor([[1,0,0],[0,torch.cos(r),-torch.sin(r)],[0,torch.sin(r),torch.cos(r)]], device=self.device)
        Ry = torch.tensor([[torch.cos(p),0,torch.sin(p)],[0,1,0],[-torch.sin(p),0,torch.cos(p)]], device=self.device)
        Rz = torch.tensor([[torch.cos(y),-torch.sin(y),0],[torch.sin(y),torch.cos(y),0],[0,0,1]], device=self.device)
        return Rz @ Ry @ Rx

    def forward(self, depth_stack):
        """
        Args:
            depth_stack: (B, 4, H, W) order [Front, Rear, Left, Right]
        Returns:
            height_map: (B, 1, 40, 40) - Sparse map with max heights
        """
        B, _, H, W = depth_stack.shape
        
        # Initialize as 1D Flat Tensor
        grid_flat = torch.full((B * self.map_size * self.map_size,), -10.0, device=self.device)
        
        cam_names = ["front", "rear", "left", "right"]
        half_range = (self.map_size * self.map_res) / 2.0 

        for i, name in enumerate(cam_names):
            depth = depth_stack[:, i, :, :] # (B, H, W)
            
            # Filter valid depth
            mask = (depth > 0.1) & (depth < 3.0) 
            if not mask.any(): continue

            # 1. Unproject
            points_opt = self.ray_dirs.unsqueeze(0) * depth.unsqueeze(-1)
            
            # 2. Transform
            R, t = self.cam_transforms[name]
            points_base = torch.matmul(points_opt, R.T) + t
            
            # 3. Flatten and Filter
            pts_flat = points_base[mask] # (N, 3)
            batch_ids = torch.arange(B, device=self.device).view(B, 1, 1).expand(B, H, W)[mask]
            
            # 4. Discretize
            # Raw indices (0 = -Range, 39 = +Range)
            u_raw = ((pts_flat[:, 0] + half_range) / self.map_res).long()
            v_raw = ((pts_flat[:, 1] + half_range) / self.map_res).long()
            
            # Boundary check on raw indices
            valid = (u_raw >= 0) & (u_raw < self.map_size) & (v_raw >= 0) & (v_raw < self.map_size)
            
            u = u_raw[valid]
            v = v_raw[valid]
            b_id = batch_ids[valid]
            z_vals = pts_flat[valid, 2]

            # [FIX] Invert indices to match Image Coordinates / GT Flip
            # +X (Front) -> Index 0 (Top)
            # +Y (Left)  -> Index 0 (Left)
            u = (self.map_size - 1) - u
            v = (self.map_size - 1) - v
            
            # 5. Scatter Max 
            flat_indices = b_id * (self.map_size**2) + u * self.map_size + v
            grid_flat.scatter_reduce_(0, flat_indices, z_vals, reduce="amax", include_self=True)

        # Reshape
        height_map = grid_flat.view(B, 1, self.map_size, self.map_size)
        
        # Clamp bottom
        height_map = torch.clamp(height_map, min=-1.0)
        
        return height_map