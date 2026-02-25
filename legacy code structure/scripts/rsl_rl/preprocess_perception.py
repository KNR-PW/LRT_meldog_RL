"""
preprocess_perception.py
Handles geometric projection from Depth Images to Top-Down Height Maps.
V4 Update: Added Occlusion Mask generation (1=No Data, 0=Data).
Standalone Version: No dependency on isaaclab or pxr (offline-safe).
"""
import torch
import torch.nn as nn
import numpy as np

# --- STANDALONE MATH UTILS (Replaces isaaclab.utils.math) ---
def quat_inv(q):
    """Inverse of quaternion (w, x, y, z)."""
    return torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], dim=-1)

def quat_apply(q, v):
    """Apply quaternion rotation to vector."""
    # q: (..., 4) [w, x, y, z], v: (..., 3)
    xyz = q[..., 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + q[..., 0:1] * t + torch.cross(xyz, t, dim=-1)

def quat_from_euler_xyz(r, p, y):
    """Create quaternion from Euler angles (roll, pitch, yaw)."""
    cx, sx = torch.cos(r / 2.0), torch.sin(r / 2.0)
    cy, sy = torch.cos(p / 2.0), torch.sin(p / 2.0)
    cz, sz = torch.cos(y / 2.0), torch.sin(y / 2.0)
    
    qw = cx * cy * cz + sx * sy * sz
    qx = sx * cy * cz - cx * sy * sz
    qy = cx * sy * cz + sx * cy * sz
    qz = cx * cy * sz - sx * sy * cz
    return torch.stack([qw, qx, qy, qz], dim=-1)

# --- HARDWARE CONFIGURATION ---
CAM_RES = (424, 240) # (Width, Height)
CAM_FOV = 87.0       # Horizontal FOV

# EXTRINSICS (V3/V4 Standard)
CAM_EXTRINSICS = {
    "front": {"pos": [0.4, 0.0, 0.04],   "rpy": [0.0, np.deg2rad(30), 0.0]},
    "rear":  {"pos": [-0.4, 0.0, 0.04],  "rpy": [0.0, np.deg2rad(30), np.pi]},
    "left":  {"pos": [0.0, 0.16, 0.05],  "rpy": [0.0, np.deg2rad(30), np.pi/2]},
    "right": {"pos": [0.0, -0.16, 0.05], "rpy": [0.0, np.deg2rad(30), -np.pi/2]},
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
        
        # Optical Frame: x=Right, y=Down, z=Forward
        x_norm = (x_grid - cx) / fx
        y_norm = (y_grid - cy) / fy
        self.ray_dirs = torch.stack([x_norm, y_norm, torch.ones_like(x_norm)], dim=-1).to(device)
        
        # 2. Pre-compute Extrinsics (Camera -> Robot Base)
        self.cam_transforms = {}
        # Transform: Optical (Z-fwd) -> ROS (X-fwd, Y-left, Z-up)
        R_opt_to_ros = torch.tensor([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=torch.float32, device=device)

        for cam_name, cfg in CAM_EXTRINSICS.items():
            pos = torch.tensor(cfg['pos'], device=device, dtype=torch.float32)
            rpy = torch.tensor(cfg['rpy'], device=device, dtype=torch.float32)
            R_world = self._euler_to_mat(rpy)
            final_R = R_world @ R_opt_to_ros
            self.cam_transforms[cam_name] = (final_R, pos)

    def _euler_to_mat(self, rpy):
        r, p, y = rpy[0], rpy[1], rpy[2]
        Rx = torch.tensor([[1,0,0],[0,torch.cos(r),-torch.sin(r)],[0,torch.sin(r),torch.cos(r)]], device=self.device)
        Ry = torch.tensor([[torch.cos(p),0,torch.sin(p)],[0,1,0],[-torch.sin(p),0,torch.cos(p)]], device=self.device)
        Rz = torch.tensor([[torch.cos(y),-torch.sin(y),0],[torch.sin(y),torch.cos(y),0],[0,0,1]], device=self.device)
        return Rz @ Ry @ Rx

    def forward(self, depth_stack, robot_quat=None):
        """
        Args:
            depth_stack: (B, 4, H, W) [Front, Rear, Left, Right] in Meters.
            robot_quat: (B, 4) [w, x, y, z] for Gravity Alignment.
        Returns:
            height_map: (B, 1, 40, 40) - Gravity Aligned Heights.
            occlusion_mask: (B, 1, 40, 40) - 1.0 where NO data exists, 0.0 where data exists.
        """
        B, _, H, W = depth_stack.shape
        
        # Initialize grids
        # Height grid starts at -5.0 (floor)
        grid_flat = torch.full((B * self.map_size * self.map_size,), -5.0, device=self.device)
        # Mask grid starts at 1.0 (all occluded/no data)
        mask_flat = torch.ones((B * self.map_size * self.map_size,), device=self.device)
        
        half_range = (self.map_size * self.map_res) / 2.0 
        cam_names = ["front", "rear", "left", "right"]
        
        for i, name in enumerate(cam_names):
            depth = depth_stack[:, i, :, :]
            
            mask = (depth > 0.1) & (depth < 3.0) 
            if not mask.any(): continue

            # 1. Unproject to Optical Frame
            points_opt = self.ray_dirs.unsqueeze(0) * depth.unsqueeze(-1)
            
            # 2. Transform to Robot Base Frame
            R, t = self.cam_transforms[name]
            points_base = torch.matmul(points_opt, R.T) + t
            
            pts_flat = points_base[mask]
            batch_ids = torch.arange(B, device=self.device).view(B, 1, 1).expand(B, H, W)[mask]

            # 3. Gravity Alignment
            if robot_quat is not None:
                q_b = robot_quat[batch_ids]
                pts_world = quat_apply(q_b, pts_flat)
                
                w, x, y, z = q_b[:, 0], q_b[:, 1], q_b[:, 2], q_b[:, 3]
                yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
                cy = torch.cos(yaw * 0.5)
                sy = torch.sin(yaw * 0.5)
                q_yaw_inv = torch.stack([cy, torch.zeros_like(cy), torch.zeros_like(cy), -sy], dim=-1)
                
                pts_horiz = quat_apply(q_yaw_inv, pts_world)
                x_vals, y_vals, z_vals = pts_horiz[:, 0], pts_horiz[:, 1], pts_horiz[:, 2]
            else:
                x_vals, y_vals, z_vals = pts_flat[:, 0], pts_flat[:, 1], pts_flat[:, 2]

            # 4. Discretize
            u_raw = ((x_vals + half_range) / self.map_res).long()
            v_raw = ((y_vals + half_range) / self.map_res).long()
            
            valid = (u_raw >= 0) & (u_raw < self.map_size) & (v_raw >= 0) & (v_raw < self.map_size)
            
            u, v, b_id, z = u_raw[valid], v_raw[valid], batch_ids[valid], z_vals[valid]

            # 5. Axis Alignment (X-Up in Grid)
            row = (self.map_size - 1) - u
            col = (self.map_size - 1) - v
            
            # 6. Scatter Max for Heights and Scatter Constant for Mask
            flat_indices = b_id * (self.map_size**2) + row * self.map_size + col
            
            grid_flat.scatter_reduce_(0, flat_indices, z, reduce="amax", include_self=True)
            # Set mask to 0.0 where we have at least one data point
            mask_flat.scatter_(0, flat_indices, 0.0)

        # Reshape and finalize
        height_map = grid_flat.view(B, 1, self.map_size, self.map_size)
        occlusion_mask = mask_flat.view(B, 1, self.map_size, self.map_size)
        
        height_map = torch.clamp(height_map, min=-2.0)
        
        return height_map, occlusion_mask