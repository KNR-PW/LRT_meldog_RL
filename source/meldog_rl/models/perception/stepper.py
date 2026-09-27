# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Run a learned perception model one step at a time over many robots.

Both the live evaluator (evaluate_perception.py) and the offline replay
(run_model_offline.py) step models through this class, so a recording replayed
offline gives the same prediction the model gave live.

Every robot keeps its own memory:

- recurrent models (v3, v5_archived): one row of each ConvGRU hidden state per robot;
- autoregressive models (v6_archived): one previous prediction per robot, warped into
  the robot's current frame with its own pose change;
- single-frame models (v1, v2): no memory.

A reset clears only the robots it names. ``dones[t]`` follows Isaac Lab's auto-reset:
the frame at step ``t`` already belongs to the new episode, so memory is cleared before
that frame is processed.

No Isaac imports — runs anywhere with torch.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from .heightmap_autoreg import transform_height_map_with_mask
from .projector import euler_from_quat
from .registry import AUTOREGRESSIVE, RECURRENT, SINGLE_FRAME, ModelEntry, build_model

MAP_SIZE = 40


def gravity_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """Gravity direction in the body frame, (B, 3), from (B, 4) quaternions [w, x, y, z]."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    gx = -2 * (x * z + w * y)
    gy = -2 * (y * z - w * x)
    gz = -(1 - 2 * (x * x + y * y))
    return torch.stack([gx, gy, gz], dim=1)


def file_sha256(path: str | Path) -> str:
    """Hex sha256 of a file, read in 1 MB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_perception_model(
    entry: ModelEntry,
    checkpoint_path: str | Path | None,
    device: str | torch.device,
    gru_hidden: int = 128,
    gru_layers: int = 2,
) -> tuple[torch.nn.Module, dict]:
    """Build ``entry``'s model, load a checkpoint into it and put it in eval mode.

    ConvGRU sizes are read from the checkpoint's ``config`` when present. Without a
    checkpoint the weights stay random (enough for timing, meaningless for accuracy).

    Returns:
        (model, info) where info holds ``epoch``, ``gru_hidden``, ``gru_layers``,
        ``checkpoint`` and ``checkpoint_sha256`` (None without a checkpoint).
    """
    checkpoint = None
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "config" in checkpoint:
            gru_hidden = checkpoint["config"].get("gru_hidden", gru_hidden)
            gru_layers = checkpoint["config"].get("gru_layers", gru_layers)

    model = build_model(entry, gru_hidden=gru_hidden, gru_layers=gru_layers).to(device)
    info = {
        "epoch": None,
        "gru_hidden": gru_hidden,
        "gru_layers": gru_layers,
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_sha256": file_sha256(checkpoint_path) if checkpoint_path else None,
    }
    if checkpoint is not None:
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            info["epoch"] = checkpoint.get("epoch")
        else:
            model.load_state_dict(checkpoint)
    model.eval()
    return model, info


class PerceptionStepper:
    """Step a learned perception model over ``num_envs`` robots with per-robot memory."""

    def __init__(
        self,
        model: torch.nn.Module,
        kind: str,
        num_envs: int,
        device: str | torch.device,
        map_size: int = MAP_SIZE,
    ):
        if kind not in (SINGLE_FRAME, RECURRENT, AUTOREGRESSIVE):
            raise ValueError(f"PerceptionStepper cannot step a '{kind}' model")
        self.model = model
        self.kind = kind
        self.num_envs = num_envs
        self.device = torch.device(device)

        shape = (num_envs, 1, map_size, map_size)
        self.hidden: list[torch.Tensor] | None = None  # recurrent
        self.prev_output = torch.zeros(shape, device=self.device)  # autoregressive
        self.prev_valid = torch.zeros(shape, device=self.device)
        self.prev_pos = torch.zeros(num_envs, 3, device=self.device)
        self.prev_yaw = torch.zeros(num_envs, device=self.device)
        # A fresh robot has no previous frame: its (zero) memory is used without warping.
        self.fresh = torch.ones(num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Clear the memory of the robots in ``env_ids`` (all robots when None)."""
        reset = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if env_ids is None:
            reset[:] = True
        else:
            reset[torch.as_tensor(env_ids, device=self.device).long()] = True
        if not reset.any():
            return

        if self.hidden is not None:
            self.hidden = [torch.where(reset.view(-1, 1, 1, 1), 0.0, h) for h in self.hidden]
        self.prev_output = torch.where(reset.view(-1, 1, 1, 1), 0.0, self.prev_output)
        self.prev_valid = torch.where(reset.view(-1, 1, 1, 1), 0.0, self.prev_valid)
        self.fresh = self.fresh | reset

    def step(
        self,
        sparse: torch.Tensor,
        mask: torch.Tensor,
        robot_pos: torch.Tensor,
        robot_quat: torch.Tensor,
        dones: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict one frame for every robot.

        Args:
            sparse: (B, 1, H, W) sparse height map
            mask: (B, 1, H, W) occlusion mask (1 = no data)
            robot_pos: (B, 3) base position, world frame
            robot_quat: (B, 4) base orientation [w, x, y, z], world frame
            dones: (B,) robots reset during this step; their memory is cleared first

        Returns:
            (B, 1, H, W) predicted height map
        """
        if dones is not None:
            done_ids = torch.as_tensor(dones, device=self.device).bool().nonzero().flatten()
            if done_ids.numel() > 0:
                self.reset(done_ids)

        grav = gravity_from_quat(robot_quat)

        if self.kind == SINGLE_FRAME:
            pred = self.model(sparse, mask, grav)
        elif self.kind == RECURRENT:
            pred, self.hidden = self.model(sparse, mask, grav, self.hidden)
        else:
            yaw = euler_from_quat(robot_quat)
            fresh = self.fresh
            prev_pos = torch.where(fresh.view(-1, 1), robot_pos, self.prev_pos)
            prev_yaw = torch.where(fresh, yaw, self.prev_yaw)
            prev_output, prev_valid = transform_height_map_with_mask(
                self.prev_output, self.prev_valid, prev_pos, prev_yaw, robot_pos, yaw
            )
            pred = self.model(sparse, mask, prev_output, prev_valid, grav)
            self.prev_output = pred.detach()
            self.prev_valid = torch.ones_like(self.prev_valid)
            self.prev_pos = robot_pos.detach().clone()
            self.prev_yaw = yaw.detach().clone()

        self.fresh = torch.zeros_like(self.fresh)
        return pred
