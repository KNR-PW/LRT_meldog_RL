# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Batched world-frame elevation mapping baseline.

Fankhauser-style elevation mapping ("Robot-Centric Elevation Mapping with
Uncertainty Estimates", Fankhauser et al.) with known poses, implemented in pure
PyTorch and batched over environments. Non-learned baseline for the V5/V6
perception models — same call signature as SLAMBaseline.

Key differences from the legacy shift-and-composite SLAMBaseline:
- The map is anchored in the WORLD frame with world-aligned axes; the buffer origin
  scrolls in whole-cell increments, so stored data is never resampled (no blur
  accumulation, no drift).
- Heights are stored as absolute world z and re-referenced to the robot base only
  at readout, so base bobbing/climbing cannot corrupt the memory.
- Per-cell 1D Kalman fusion (height mean + variance); measurement noise grows with
  distance from the robot, so close observations dominate.

Standalone module — no Isaac Lab dependencies (offline-safe).
"""

import torch
import torch.nn.functional as F

from .slam_baseline import fill_unobserved


class ElevationMapper:
    """Height map reconstruction via world-frame elevation mapping with known poses.

    Each timestep:
    1. Scroll the world-anchored buffer (whole cells) to keep the robot centered.
    2. Transform observed sparse-map cells to world frame and fuse them into the
       buffer with a per-cell Kalman update.
    3. Read out the robot-centric, yaw-aligned local map with a single bilinear
       sample (normalized-convolution style to avoid bleeding across the
       valid/invalid boundary), re-referenced to the current base height.

    Args:
        map_size: Local output grid dimension (default 40 -> 40x40).
        map_res: Grid resolution in meters (default 0.05 m).
        device: Torch device.
        world_size: World buffer dimension in cells (default 160 -> 8 m x 8 m).
        meas_var_base: Measurement variance at zero distance (m^2).
        meas_var_dist: Variance growth per squared meter of distance (m^2 / m^2).
        process_var: Per-step variance inflation of stored cells (m^2) — lets the
            map adapt to changes and lets newer data win over stale data.
        init_var: Variance assigned to never-seen cells (first update ~= overwrite).
    """

    def __init__(
        self,
        map_size: int = 40,
        map_res: float = 0.05,
        device: str = "cuda",
        world_size: int = 160,
        meas_var_base: float = 1e-4,
        meas_var_dist: float = 1e-4,
        process_var: float = 1e-6,
        init_var: float = 1e6,
        **kwargs,
    ):
        self.map_size = map_size
        self.map_res = map_res
        self.device = device
        self.world_size = world_size
        self.meas_var_base = meas_var_base
        self.meas_var_dist = meas_var_dist
        self.process_var = process_var
        self.init_var = init_var
        self.batch_size = 0

        # Robot-frame cell-center coordinates of the local map (row 0 = front +x,
        # col 0 = left +y), matching DepthProjector's convention.
        half = map_size * map_res / 2.0
        centers = half - (torch.arange(map_size, device=device) + 0.5) * map_res
        self.local_x = centers.view(-1, 1).expand(map_size, map_size)  # rows
        self.local_y = centers.view(1, -1).expand(map_size, map_size)  # cols
        # Squared distance from robot center -> per-cell measurement variance.
        d2 = self.local_x**2 + self.local_y**2
        self.meas_var = meas_var_base + meas_var_dist * d2  # (N, N)

    def reset(self, batch_size: int):
        """Allocate and zero all per-environment state."""
        self.batch_size = batch_size
        ws = self.world_size
        self.world_mean = torch.zeros(batch_size, 1, ws, ws, device=self.device)
        self.world_var = torch.full((batch_size, 1, ws, ws), self.init_var, device=self.device)
        self.world_valid = torch.zeros(batch_size, 1, ws, ws, device=self.device)
        # World cell index of buffer cell (0, 0); set on first update per env.
        self.origin = torch.zeros(batch_size, 2, dtype=torch.long, device=self.device)
        self.initialized = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

    def reset_env(self, env_indices: torch.Tensor):
        """Reset state for specific environments (episode boundary)."""
        self.world_mean[env_indices] = 0.0
        self.world_var[env_indices] = self.init_var
        self.world_valid[env_indices] = 0.0
        self.initialized[env_indices] = False

    def _scroll(self, robot_cell: torch.Tensor):
        """Re-center buffers on the robot in whole-cell steps (no resampling)."""
        ws = self.world_size
        desired_origin = robot_cell - ws // 2
        # Initialize origins on first use.
        if not self.initialized.all():
            fresh = ~self.initialized
            self.origin[fresh] = desired_origin[fresh]
            self.initialized[fresh] = True
        offset = desired_origin - self.origin  # (B, 2)
        need = (offset.abs() > ws // 4).any(dim=1)
        for i in need.nonzero(as_tuple=False).squeeze(-1).tolist():
            sx, sy = offset[i, 0].item(), offset[i, 1].item()
            self.world_mean[i] = torch.roll(self.world_mean[i], shifts=(-sx, -sy), dims=(-2, -1))
            self.world_var[i] = torch.roll(self.world_var[i], shifts=(-sx, -sy), dims=(-2, -1))
            self.world_valid[i] = torch.roll(self.world_valid[i], shifts=(-sx, -sy), dims=(-2, -1))
            # Cells that wrapped around carry stale data — invalidate them.
            for dim, s in ((2, sx), (3, sy)):
                if s == 0:
                    continue
                idx = [slice(None)] * 4
                idx[dim] = slice(ws - s, ws) if s > 0 else slice(0, -s)
                self.world_mean[i][tuple(idx[1:])] = 0.0
                self.world_var[i][tuple(idx[1:])] = self.init_var
                self.world_valid[i][tuple(idx[1:])] = 0.0
            self.origin[i] += offset[i]

    @torch.no_grad()
    def __call__(
        self,
        sparse_map: torch.Tensor,
        occlusion_mask: torch.Tensor,
        robot_pos: torch.Tensor,
        robot_yaw: torch.Tensor,
        grav: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fuse new sparse observations and return the local height map estimate.

        Args:
            sparse_map: (B, 1, N, N) sparse height map from DepthProjector,
                heights relative to robot base z.
            occlusion_mask: (B, 1, N, N) 1=no data, 0=has data.
            robot_pos: (B, 3) world-frame robot base position.
            robot_yaw: (B,) world-frame robot yaw.
            grav: unused, accepted for API compat.

        Returns:
            result: (B, 1, N, N) height map estimate relative to robot base z.
        """
        B, _, N, _ = sparse_map.shape
        ws = self.world_size
        res = self.map_res

        robot_cell = torch.floor(robot_pos[:, :2] / res).long()
        self._scroll(robot_cell)

        # --- Time update: stored estimates slowly lose confidence.
        self.world_var += self.process_var

        # --- Measurement update.
        cos_y = torch.cos(robot_yaw).view(B, 1, 1)
        sin_y = torch.sin(robot_yaw).view(B, 1, 1)
        lx = self.local_x.unsqueeze(0)  # (1, N, N)
        ly = self.local_y.unsqueeze(0)
        xw = cos_y * lx - sin_y * ly + robot_pos[:, 0].view(B, 1, 1)
        yw = sin_y * lx + cos_y * ly + robot_pos[:, 1].view(B, 1, 1)
        zw = sparse_map.squeeze(1) + robot_pos[:, 2].view(B, 1, 1)

        bi = torch.floor(xw / res).long() - self.origin[:, 0].view(B, 1, 1)
        bj = torch.floor(yw / res).long() - self.origin[:, 1].view(B, 1, 1)

        has_data = occlusion_mask.squeeze(1) < 0.5
        in_bounds = (bi >= 0) & (bi < ws) & (bj >= 0) & (bj < ws)
        obs = has_data & in_bounds

        env_id = torch.arange(B, device=self.device).view(B, 1, 1).expand(B, N, N)
        flat_idx = (env_id * ws * ws + bi.clamp(0, ws - 1) * ws + bj.clamp(0, ws - 1))[obs]

        # Collapse same-frame collisions: max height (projector convention),
        # smallest variance (closest measurement) per world cell.
        meas_z = torch.full((B * ws * ws,), -1e9, device=self.device)
        meas_z.scatter_reduce_(0, flat_idx, zw[obs], reduce="amax", include_self=True)
        meas_v = torch.full((B * ws * ws,), 1e9, device=self.device)
        var_src = self.meas_var.unsqueeze(0).expand(B, N, N)
        meas_v.scatter_reduce_(0, flat_idx, var_src[obs], reduce="amin", include_self=True)
        meas_seen = meas_z > -1e8

        mean_flat = self.world_mean.view(-1)
        var_flat = self.world_var.view(-1)
        valid_flat = self.world_valid.view(-1)

        gain = var_flat / (var_flat + meas_v)
        new_mean = mean_flat + gain * (meas_z - mean_flat)
        new_var = (1.0 - gain) * var_flat
        self.world_mean = torch.where(meas_seen, new_mean, mean_flat).view(B, 1, ws, ws)
        self.world_var = torch.where(meas_seen, new_var, var_flat).view(B, 1, ws, ws)
        self.world_valid = torch.clamp(valid_flat + meas_seen.float(), 0, 1).view(B, 1, ws, ws)

        # --- Readout: single bilinear sample of the world buffer at the local
        # cell centers (normalized convolution avoids valid-boundary bleed).
        gi = xw / res - self.origin[:, 0].view(B, 1, 1).float() - 0.5  # buffer row
        gj = yw / res - self.origin[:, 1].view(B, 1, 1).float() - 0.5  # buffer col
        grid = torch.stack(
            [2.0 * gj / (ws - 1) - 1.0, 2.0 * gi / (ws - 1) - 1.0], dim=-1
        )  # (B, N, N, 2), gx=cols, gy=rows

        num = F.grid_sample(
            self.world_mean * self.world_valid,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        den = F.grid_sample(
            self.world_valid,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        valid_out = (den > 0.25).float()
        heights = num / den.clamp(min=1e-6) - robot_pos[:, 2].view(B, 1, 1, 1)
        heights = heights * valid_out

        # Fill never-seen cells from valid neighbors (same as legacy baseline).
        return fill_unobserved(heights, valid_out)
