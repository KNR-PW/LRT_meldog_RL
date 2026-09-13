# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Synthetic end-to-end test for the ElevationMapper baseline.

Scenario: a robot walks past a 0.3 m box while only seeing a ring in FRONT of
itself (0.25–0.9 m), with the base height bobbing like a gait. At the end, the box
is entirely behind the robot, so reconstructing it exercises pure map memory —
exactly what the legacy shift-and-composite baseline destroyed (blur + z-drift).

No Isaac imports — runs anywhere:  python tests/test_elevation_mapper.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))

from meldog_rl.models.perception.elevation_mapper import ElevationMapper  # noqa: E402
from meldog_rl.models.perception.slam_baseline import SLAMBaseline  # noqa: E402

N, RES = 40, 0.05
HALF = N * RES / 2.0
DEVICE = "cpu"

# Box: world x in [0.8, 1.3], y in [-0.25, 0.25], height 0.3 (cell-aligned edges).
BOX = (0.8, 1.3, -0.25, 0.25, 0.3)


def gt_height(xw: torch.Tensor, yw: torch.Tensor) -> torch.Tensor:
    x0, x1, y0, y1, h = BOX
    inside = (xw >= x0) & (xw <= x1) & (yw >= y0) & (yw <= y1)
    return torch.where(inside, torch.full_like(xw, h), torch.zeros_like(xw))


def local_grid():
    """Robot-frame cell centers, DepthProjector convention (row0=+x, col0=+y)."""
    c = HALF - (torch.arange(N) + 0.5) * RES
    lx = c.view(-1, 1).expand(N, N)
    ly = c.view(1, -1).expand(N, N)
    return lx, ly


def observe(pos, yaw):
    """Simulated projector output at a pose: heights rel. base + occlusion mask.

    Visibility: ring 0.25 < d < 0.9 in the robot's FRONT half-plane only.
    """
    lx, ly = local_grid()
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
    xw = cos_y * lx - sin_y * ly + pos[0]
    yw = sin_y * lx + cos_y * ly + pos[1]
    h = gt_height(xw, yw) - pos[2]
    d = (lx**2 + ly**2).sqrt()
    visible = (d > 0.25) & (d < 0.9) & (lx > 0.0)
    sparse = torch.where(visible, h, torch.full_like(h, -2.0))
    occlusion = (~visible).float()
    return sparse.view(1, 1, N, N), occlusion.view(1, 1, N, N)


def readout_error(pred, pos, yaw):
    """(all-cells RMSE, box-region RMSE) of a readout vs GT at the given pose."""
    lx, ly = local_grid()
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
    xw = cos_y * lx - sin_y * ly + pos[0]
    yw = sin_y * lx + cos_y * ly + pos[1]
    gt = gt_height(xw, yw) - pos[2]
    err = pred.view(N, N) - gt
    x0, x1, y0, y1, _ = BOX
    pad = 0.1
    box_region = (xw > x0 - pad) & (xw < x1 + pad) & (yw > y0 - pad) & (yw < y1 + pad)
    rmse_all = err.pow(2).mean().sqrt().item()
    rmse_box = err[box_region].pow(2).mean().sqrt().item()
    return rmse_all, rmse_box


def walk_trajectory():
    """Walk from x=0 to x=2 with gait-like base bobbing, then turn 90 deg CCW."""
    poses = []
    steps = 100
    for t in range(steps):
        x = 2.0 * t / (steps - 1)
        z = 0.35 + 0.05 * torch.sin(torch.tensor(t * 0.5)).item()
        poses.append((torch.tensor([x, 0.0, z]), torch.tensor(0.0)))
    for t in range(20):
        yaw = (t + 1) / 20 * torch.pi / 2
        poses.append((torch.tensor([2.0, 0.0, 0.35]), torch.tensor(yaw)))
    return poses


def run_mapper(mapper):
    mapper.reset(1)
    poses = walk_trajectory()
    pred = None
    for pos, yaw in poses[:100]:  # walk phase
        sparse, occ = observe(pos, yaw)
        pred = mapper(sparse, occ, pos.view(1, 3), yaw.view(1))
    walk_err = readout_error(pred, *poses[99])
    for pos, yaw in poses[100:]:  # turn-in-place phase (box never visible again)
        sparse, occ = observe(pos, yaw)
        pred = mapper(sparse, occ, pos.view(1, 3), yaw.view(1))
    turn_err = readout_error(pred, *poses[-1])
    return walk_err, turn_err


def test_elevation_mapper_memory():
    walk, turn = run_mapper(ElevationMapper(map_size=N, map_res=RES, device=DEVICE))
    print(f"  elevation: after walk rmse_all={walk[0]:.4f} rmse_box={walk[1]:.4f}")
    print(f"  elevation: after turn rmse_all={turn[0]:.4f} rmse_box={turn[1]:.4f}")
    # Box is entirely behind the robot at the end of the walk -> pure memory.
    assert walk[1] < 0.03, f"box memory RMSE too high after walk: {walk[1]:.4f}"
    # After turning in place the box must survive the frame change on the left.
    assert turn[1] < 0.03, f"box memory RMSE too high after turn: {turn[1]:.4f}"
    assert walk[0] < 0.05 and turn[0] < 0.05


def test_reset_clears_memory():
    mapper = ElevationMapper(map_size=N, map_res=RES, device=DEVICE)
    mapper.reset(1)
    poses = walk_trajectory()
    for pos, yaw in poses[:100]:
        sparse, occ = observe(pos, yaw)
        mapper(sparse, occ, pos.view(1, 3), yaw.view(1))
    mapper.reset_env(torch.tensor([0]))
    assert mapper.world_valid.sum().item() == 0.0, "reset_env left valid cells"


def compare_legacy():
    """Informational: legacy shift-and-composite on the identical trajectory."""
    walk, turn = run_mapper(SLAMBaseline(map_size=N, map_res=RES, device=DEVICE))
    print(f"  legacy:    after walk rmse_all={walk[0]:.4f} rmse_box={walk[1]:.4f}")
    print(f"  legacy:    after turn rmse_all={turn[0]:.4f} rmse_box={turn[1]:.4f}")


if __name__ == "__main__":
    test_elevation_mapper_memory()
    test_reset_clears_memory()
    compare_legacy()
    print("All elevation mapper tests passed.")
