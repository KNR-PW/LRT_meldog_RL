# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Four chase cameras tiled into one benchmark video.

One long video of a single robot shows one scenario. Four cameras, each following a robot on a
different kind of terrain, put four scenarios in the same file and the same running time.

The cameras are world-anchored and created after the env, so the same code works for both env
styles; every recorded frame they are placed behind and above the robot they follow.
"""

from __future__ import annotations

import numpy as np
import torch

CAMERA_PREFIX = "bench_cam_"
DEFAULT_VIEWS = 4
# Chase camera placement in the robot's yaw frame: behind, to the side, above, in metres.
CHASE_OFFSET = (-1.9, -1.3, 1.0)
LOOK_AT_HEIGHT = 0.25


def make_cameras(num_views: int = DEFAULT_VIEWS, width: int = 640, height: int = 360):
    """Create ``num_views`` RGB cameras after the env exists, outside its scene.

    They are deliberately not scene sensors: the scene resets sensors with per-env indices, and
    these are single world-anchored cameras, not one per env.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import Camera, CameraCfg

    cameras = []
    for i in range(num_views):
        camera = Camera(
            CameraCfg(
                prim_path=f"/World/{CAMERA_PREFIX}{i}",
                update_period=0.0,
                width=width,
                height=height,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.1, 200.0)),
            )
        )
        # The simulation is already playing, so the play callback that normally initializes a
        # sensor has been and gone; initialize it here instead.
        camera._initialize_callback(None)
        cameras.append(camera)
    return cameras


def pick_view_envs(cells, num_envs: int, num_views: int = DEFAULT_VIEWS) -> list[int]:
    """Env indices to follow: one per terrain kind where the benchmark assigns fixed cells.

    Picks the hardest difficulty row available for each kind, so the video shows the interesting
    cases rather than four robots on flat ground.
    """
    if not cells:
        return list(range(min(num_views, num_envs)))
    best: dict[str, tuple[int, int]] = {}
    for env_index, (row, _col, kind) in enumerate(cells):
        if kind not in best or row > best[kind][0]:
            best[kind] = (row, env_index)
    picks = [env_index for _, (_, env_index) in sorted(best.items())]
    step = max(1, len(picks) // num_views)
    return picks[::step][:num_views] or list(range(min(num_views, num_envs)))


class BenchmarkVideo:
    """Writes a tiled chase-camera video while the benchmark runs."""

    def __init__(self, robot, env_indices, path, fps: int = 25, every_n_steps: int = 2):
        self.cameras = make_cameras(len(env_indices))
        self.robot = robot
        self.env_indices = list(env_indices)
        self.every_n_steps = max(1, every_n_steps)
        self.frames = 0
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        import imageio.v2 as imageio

        self.writer = imageio.get_writer(str(path), fps=fps, macro_block_size=None)

    def _chase_pose(self, env_index: int):
        """(eye, target) for the camera following ``env_index``, in world coordinates."""
        pos = self.robot.data.root_pos_w[env_index]
        quat = self.robot.data.root_quat_w[env_index]
        yaw = torch.atan2(
            2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
            1.0 - 2.0 * (quat[2] ** 2 + quat[3] ** 2),
        )
        cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
        dx, dy, dz = CHASE_OFFSET
        eye = pos + torch.tensor(
            [dx * cos_yaw - dy * sin_yaw, dx * sin_yaw + dy * cos_yaw, dz], device=pos.device
        )
        target = pos + torch.tensor([0.0, 0.0, LOOK_AT_HEIGHT], device=pos.device)
        return eye, target

    def capture(self, step: int) -> None:
        """Move the cameras onto their robots and append one tiled frame."""
        if step % self.every_n_steps:
            return
        images = []
        for camera, env_index in zip(self.cameras, self.env_indices):
            eye, target = self._chase_pose(env_index)
            camera.set_world_poses_from_view(eye.unsqueeze(0), target.unsqueeze(0))
            camera.update(dt=0.0, force_recompute=True)
            rgb = camera.data.output["rgb"][0, ..., :3]
            images.append(rgb.detach().cpu().numpy().astype(np.uint8))
        self.writer.append_data(self._tile(images))
        self.frames += 1

    @staticmethod
    def _tile(images) -> np.ndarray:
        """2x2 grid (or a single image for one view), padded if a view is missing."""
        if len(images) == 1:
            return images[0]
        while len(images) < 4:
            images.append(np.zeros_like(images[0]))
        top = np.concatenate(images[:2], axis=1)
        bottom = np.concatenate(images[2:4], axis=1)
        return np.concatenate([top, bottom], axis=0)

    def close(self) -> None:
        self.writer.close()
