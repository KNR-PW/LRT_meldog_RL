# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Classical SLAM baseline for height map reconstruction.

Simple shift-and-composite approach:
1. Project depth cameras to sparse heightmap (existing DepthProjector)
2. Shift previous accumulated map by robot body movement
3. Use new measurements where available, shifted previous elsewhere

This provides a non-learned baseline directly comparable to V5 (ConvGRU) and
V6 (Autoregressive) perception models.
"""

import torch
import torch.nn.functional as F

from .heightmap_autoreg import transform_height_map_with_mask


def fill_unobserved(
    height_map: torch.Tensor,
    valid_mask: torch.Tensor,
    iterations: int = 20,
) -> torch.Tensor:
    """Fill unobserved cells by propagating from valid neighbors.

    Args:
        height_map: (B, 1, H, W) height values.
        valid_mask: (B, 1, H, W) 1=valid, 0=unobserved.
        iterations: Max fill rounds.

    Returns:
        filled: (B, 1, H, W) height map with gaps filled.
    """
    filled = height_map.clone()
    valid = valid_mask.clone()
    kernel = torch.ones(1, 1, 3, 3, device=height_map.device)

    for _ in range(iterations):
        neighbor_sum = F.conv2d(filled * valid, kernel, padding=1)
        neighbor_count = F.conv2d(valid, kernel, padding=1)

        can_fill = (valid < 0.5) & (neighbor_count > 0.5)
        new_vals = neighbor_sum / neighbor_count.clamp(min=1e-6)
        filled = torch.where(can_fill, new_vals, filled)
        valid = torch.where(can_fill, torch.ones_like(valid), valid)

        if valid.min() > 0.5:
            break

    return filled


class SLAMBaseline:
    """Height map reconstruction via shift-and-composite.

    Each timestep:
    1. Shift previous result to current robot frame
    2. Use new camera data where available, shifted previous where occluded
    3. Fill any remaining tiny gaps via neighbor propagation

    Args:
        map_size: Grid dimension (default 40 -> 40x40 grid).
        map_res: Grid resolution in meters (default 0.05m).
        device: Torch device.
    """

    def __init__(
        self,
        map_size: int = 40,
        map_res: float = 0.05,
        device: str = "cuda",
        **kwargs,
    ):
        self.map_size = map_size
        self.map_res = map_res
        self.device = device
        self.batch_size = 0

    def reset(self, batch_size: int):
        """Allocate and zero all per-environment state."""
        self.batch_size = batch_size
        self.accumulated_map = torch.zeros(
            batch_size, 1, self.map_size, self.map_size, device=self.device
        )
        self.accumulated_valid = torch.zeros(
            batch_size, 1, self.map_size, self.map_size, device=self.device
        )
        self.prev_pos = torch.zeros(batch_size, 3, device=self.device)
        self.prev_yaw = torch.zeros(batch_size, device=self.device)
        self.initialized = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

    def reset_env(self, env_indices: torch.Tensor):
        """Reset state for specific environments (episode boundary)."""
        self.accumulated_map[env_indices] = 0.0
        self.accumulated_valid[env_indices] = 0.0
        self.prev_pos[env_indices] = 0.0
        self.prev_yaw[env_indices] = 0.0
        self.initialized[env_indices] = False

    @torch.no_grad()
    def __call__(
        self,
        sparse_map: torch.Tensor,
        occlusion_mask: torch.Tensor,
        robot_pos: torch.Tensor,
        robot_yaw: torch.Tensor,
        grav: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Update accumulated map with new sparse observations.

        Args:
            sparse_map: (B, 1, H, W) sparse height map from DepthProjector.
            occlusion_mask: (B, 1, H, W) 1=no data, 0=has data.
            robot_pos: (B, 3) world-frame robot position.
            robot_yaw: (B,) world-frame robot yaw.
            grav: unused, accepted for API compat.

        Returns:
            result: (B, 1, H, W) accumulated height map estimate.
        """
        has_data = 1.0 - occlusion_mask  # 1 where camera sees

        # Shift previous accumulated map to current robot frame
        shifted_map, shifted_valid = transform_height_map_with_mask(
            self.accumulated_map,
            self.accumulated_valid,
            self.prev_pos,
            self.prev_yaw,
            robot_pos,
            robot_yaw,
        )

        # Simple composite: new data where available, shifted previous where occluded
        result = torch.where(has_data > 0.5, sparse_map, shifted_map)
        valid = torch.clamp(has_data + shifted_valid, 0.0, 1.0)

        # First frame: no prior to shift, just use sparse data
        not_init = (~self.initialized).view(-1, 1, 1, 1)
        result = torch.where(not_init, sparse_map, result)
        valid = torch.where(not_init, has_data, valid)

        # Update state
        self.accumulated_map = result.clone()
        self.accumulated_valid = valid.clone()
        self.prev_pos = robot_pos.clone()
        self.prev_yaw = robot_yaw.clone()
        self.initialized[:] = True

        # Fill remaining small gaps (cells never seen by any camera)
        return fill_unobserved(result, valid)
