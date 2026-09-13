# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Geometric regression tests for transform_height_map_with_mask.

Guards against the 2026-07-12 translation-axis bug: the sampling grid's W axis was
treated as robot +x, but the DepthProjector grid convention is row = -x, col = -y,
so walking forward smeared the remembered map sideways.

No Isaac imports — runs anywhere:  python tests/test_heightmap_transform.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))

from meldog_rl.models.perception.heightmap_autoreg import (  # noqa: E402
    MAP_RES,
    MAP_SIZE,
    transform_height_map_with_mask,
)

HALF = MAP_SIZE * MAP_RES / 2.0


def rc_from_xy(x, y):
    """Robot-frame (x, y) -> (row, col), matching DepthProjector's discretization."""
    u = int((x + HALF) / MAP_RES)
    v = int((y + HALF) / MAP_RES)
    return (MAP_SIZE - 1) - u, (MAP_SIZE - 1) - v


def shift_bump(prev_pos, prev_yaw, curr_pos, curr_yaw, bump_xy, bump_h=1.0):
    """Place a bump in the prev-frame map, run the transform, return (map, peak rc)."""
    prev = torch.zeros(1, 1, MAP_SIZE, MAP_SIZE)
    valid = torch.ones(1, 1, MAP_SIZE, MAP_SIZE)
    r, c = rc_from_xy(*bump_xy)
    prev[0, 0, r, c] = bump_h
    out, _ = transform_height_map_with_mask(
        prev, valid,
        torch.tensor([prev_pos]), torch.tensor([prev_yaw]),
        torch.tensor([curr_pos]), torch.tensor([curr_yaw]),
    )
    idx = out[0, 0].view(-1).argmax().item()
    return out, (idx // MAP_SIZE, idx % MAP_SIZE)


def assert_peak(got, expect_xy, name):
    exp = rc_from_xy(*expect_xy)
    ok = abs(got[0] - exp[0]) <= 1 and abs(got[1] - exp[1]) <= 1
    assert ok, f"{name}: peak at rc{got}, expected rc{exp} (xy={expect_xy})"
    print(f"  OK  {name}: rc{got} ~ rc{exp}")


def test_walk_forward():
    # Robot walks +x 0.5 m: a bump 0.5 m ahead must end up under the robot.
    _, got = shift_bump((0, 0, 0), 0.0, (0.5, 0, 0), 0.0, (0.5, 0.0))
    assert_peak(got, (0.0, 0.0), "walk fwd 0.5m")


def test_strafe_left():
    _, got = shift_bump((0, 0, 0), 0.0, (0, 0.5, 0), 0.0, (0.0, 0.5))
    assert_peak(got, (0.0, 0.0), "strafe left 0.5m")


def test_rotate_ccw():
    # Robot yaws +90 deg: a bump ahead must appear on the robot's right (y = -0.5).
    _, got = shift_bump((0, 0, 0), 0.0, (0, 0, 0), torch.pi / 2, (0.5, 0.0))
    assert_peak(got, (0.0, -0.5), "rotate +90deg")


def test_yawed_walk():
    # At yaw=90 the robot's +x is world +y; walking world +y brings the bump home.
    _, got = shift_bump((0, 0, 0), torch.pi / 2, (0, 0.5, 0), torch.pi / 2, (0.5, 0.0))
    assert_peak(got, (0.0, 0.0), "yaw=90, walk world +y")


def test_z_compensation():
    # Base rises 0.3 m: remembered heights must drop by 0.3 m relative to base.
    out, got = shift_bump((0, 0, 0), 0.0, (0, 0, 0.3), 0.0, (0.5, 0.0), bump_h=1.0)
    assert_peak(got, (0.5, 0.0), "z-comp position")
    peak = out[0, 0, got[0], got[1]].item()
    assert abs(peak - 0.7) < 0.02, f"z-comp height: got {peak:.3f}, expected 0.7"
    print(f"  OK  z-comp height: {peak:.3f} ~ 0.7")


if __name__ == "__main__":
    test_walk_forward()
    test_strafe_left()
    test_rotate_ccw()
    test_yawed_walk()
    test_z_compensation()
    print("All transform tests passed.")
