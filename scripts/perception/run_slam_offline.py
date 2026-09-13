#!/usr/bin/env python3
# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Replay a SLAM baseline offline on a recorded perception run (data.h5).

The SLAM baselines are sim-free, so they can be re-run on the sparse maps + poses
already stored in any evaluation dump. This produces a SLAM data.h5 on the
IDENTICAL trajectory as the source run — the same-trajectory model-vs-SLAM
comparison that docs/evaluation.md marks mandatory for the thesis claims:

    # 1. record a model run
    evaluate_perception.py --method model --perception_checkpoint ... -> MODEL/data.h5
    # 2. replay SLAM on it (no Isaac, seconds)
    run_slam_offline.py MODEL/data.h5 --variant elevation
    # 3. compare on identical inputs
    analyze_perception.py MODEL/data.h5 REPLAY/data.h5 --labels model slam

No Isaac imports — runs anywhere with torch + h5py.
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "source"))

from meldog_rl.models.perception import ElevationMapper, SLAMBaseline, euler_from_quat  # noqa: E402
from meldog_rl.utils import make_evaluation_dir  # noqa: E402
from meldog_rl.utils.git_utils import get_git_suffix  # noqa: E402

MAP_SIZE = 40
MAP_RES = 0.05


def to_int16_mm(arr):
    return (np.clip(arr, -32.0, 32.0) * 1000.0).astype(np.int16)


def main():
    parser = argparse.ArgumentParser(description="Replay a SLAM baseline on a recorded data.h5.")
    parser.add_argument("input", type=str, help="Source data.h5 (from evaluate_perception.py).")
    parser.add_argument("--variant", type=str, default="elevation", choices=["legacy", "elevation"],
                        help="SLAM baseline variant.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default=None,
                        help="Output dir (default: new logs/perception/PE_slam-<variant>-replay_* dir).")
    args = parser.parse_args()

    device = args.device
    f_in = h5py.File(args.input, "r")
    env_keys = sorted([k for k in f_in.keys() if k.startswith("env_")],
                      key=lambda k: int(k.split("_")[1]))
    B = len(env_keys)

    # Load per-env sequences and stack to (T, B, ...).
    sparse = np.stack([f_in[k]["sparse_height"][:] for k in env_keys], axis=1).astype(np.float32) / 1000.0
    occ = np.stack([f_in[k]["occlusion_mask"][:] for k in env_keys], axis=1).astype(np.float32)
    pos = np.stack([f_in[k]["robot_pos"][:] for k in env_keys], axis=1).astype(np.float32)
    quat = np.stack([f_in[k]["robot_quat"][:] for k in env_keys], axis=1).astype(np.float32)
    gt = np.stack([f_in[k]["gt_height"][:] for k in env_keys], axis=1).astype(np.float32) / 1000.0
    if "dones" in f_in[env_keys[0]]:
        dones = np.stack([f_in[k]["dones"][:] for k in env_keys], axis=1).astype(bool)
    else:
        # Old dumps lack dones: infer resets from robot position jumps (>1 m/step).
        jumps = np.linalg.norm(np.diff(pos[:, :, :2], axis=0), axis=-1) > 1.0
        dones = np.concatenate([np.zeros((1, B), dtype=bool), jumps], axis=0)
        print(f"No dones in source; inferred {int(dones.sum())} resets from position jumps")
    T = sparse.shape[0]
    print(f"Replaying {args.variant} SLAM on {args.input}: {T} steps x {B} envs ({device})")

    slam_cls = SLAMBaseline if args.variant == "legacy" else ElevationMapper
    slam = slam_cls(map_size=MAP_SIZE, map_res=MAP_RES, device=device)
    slam.reset(B)

    yaw_all = euler_from_quat(torch.from_numpy(quat.reshape(-1, 4))).view(T, B).to(device)
    preds = np.empty((T, B, MAP_SIZE, MAP_SIZE), dtype=np.float32)
    with torch.no_grad():
        for t in range(T):
            done_idx = torch.from_numpy(np.nonzero(dones[t])[0]).long().to(device)
            if done_idx.numel() > 0:
                slam.reset_env(done_idx)
            pred = slam(
                torch.from_numpy(sparse[t]).unsqueeze(1).to(device),
                torch.from_numpy(occ[t]).unsqueeze(1).to(device),
                torch.from_numpy(pos[t]).to(device),
                yaw_all[t],
            )
            preds[t] = pred.squeeze(1).cpu().numpy()
            if (t + 1) % 200 == 0:
                print(f"  {t + 1}/{T}")

    if args.output:
        save_dir = Path(args.output)
    else:
        save_dir = make_evaluation_dir("perception", f"slam-{args.variant}-replay")
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "data.h5"

    print(f"Writing {out_path}")
    with h5py.File(out_path, "w") as f_out:
        for i, k in enumerate(env_keys):
            f_in.copy(k, f_out)  # keeps depth/gt/sparse/occlusion/pose/dones as-is
            for name in ("pred_height", "diff_height"):
                if name in f_out[k]:
                    del f_out[k][name]
            f_out[k].create_dataset("pred_height", data=to_int16_mm(preds[:, i]), compression="gzip")
            f_out[k].create_dataset("diff_height", data=to_int16_mm(np.abs(gt[:, i] - preds[:, i])),
                                    compression="gzip")

        for key, val in f_in.attrs.items():
            f_out.attrs[key] = val
        f_out.attrs["method"] = "slam"
        f_out.attrs["slam_variant"] = args.variant
        f_out.attrs["perception_checkpoint"] = f"slam:{args.variant}"
        f_out.attrs["replay_of"] = os.path.abspath(args.input)
        f_out.attrs["date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f_out.attrs["git_commit"] = get_git_suffix()

    f_in.close()
    print("Done")


if __name__ == "__main__":
    main()
