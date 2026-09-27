#!/usr/bin/env python3
# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Replay a learned perception model offline on a recorded perception run (data.h5).

Record once in Isaac Sim, then run any number of models on the identical frames:

    # 1. record a run (Isaac Sim)
    evaluate_perception.py ... -> RUN/data.h5
    # 2. replay each model and baseline on it (no Isaac)
    run_model_offline.py RUN/data.h5 --model v3 --checkpoint PM_v3_.../model_best.pt
    run_slam_offline.py RUN/data.h5 --variant elevation
    # 3. compare on identical inputs
    analyze_perception.py V3_REPLAY/data.h5 SLAM_REPLAY/data.h5 --labels v3 slam

Every robot keeps its own model memory, cleared at the recorded ``dones``. The output
copies the source run (without the raw depth images unless --copy_depth), replaces
``pred_height`` / ``diff_height`` and records the model, the checkpoint and the source
file (``replay_of``). ``complete`` is written last, so a replay that crashed is refused
by the analyzer.

The per-frame cost is measured too: one robot (batch 1), the model step and, separately,
the depth projector that produces its input. It is stored as the JSON attr ``timing``.
--timing_only measures without replaying (no checkpoint needed: cost does not depend on
the weights).

No Isaac imports — runs anywhere with torch + h5py.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "source"))

from meldog_rl.models.perception import (  # noqa: E402
    BASELINE,
    DepthProjector,
    ModelRegistryError,
    PerceptionStepper,
    file_sha256,
    load_perception_model,
    resolve_model,
)
from meldog_rl.utils import make_evaluation_dir  # noqa: E402
from meldog_rl.utils.git_utils import get_git_suffix  # noqa: E402
from meldog_rl.utils.timing import device_name, time_callable  # noqa: E402

MAP_SIZE = 40
MAP_RES = 0.05
DEPTH_KEYS = ("depth_front", "depth_rear", "depth_left", "depth_right")
RESET_JUMP_M = 1.0


def to_meters(arr: np.ndarray) -> np.ndarray:
    """Height dataset to float32 meters: integer datasets are int16 millimeters."""
    arr = np.asarray(arr)
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.float32) / 1000.0
    return arr.astype(np.float32)


def to_int16_mm(arr: np.ndarray) -> np.ndarray:
    return (np.clip(arr, -32.0, 32.0) * 1000.0).astype(np.int16)


def env_keys_of(f: h5py.File) -> list[str]:
    return sorted([k for k in f.keys() if k.startswith("env_")], key=lambda k: int(k.split("_")[1]))


def load_recording(path: str | Path) -> dict:
    """Load a recording's model inputs, stacked as (T, B, ...)."""
    with h5py.File(path, "r") as f:
        keys = env_keys_of(f)
        if not keys:
            raise ValueError(f"{path}: no env_* groups")

        def stack(name, conv=np.asarray):
            return np.stack([conv(f[k][name][:]) for k in keys], axis=1)

        rec = {
            "env_keys": keys,
            "sparse": stack("sparse_height", to_meters),
            "occ": stack("occlusion_mask").astype(np.float32),
            "pos": stack("robot_pos").astype(np.float32),
            "quat": stack("robot_quat").astype(np.float32),
            "gt": stack("gt_height", to_meters),
        }
        if "dones" in f[keys[0]]:
            rec["dones"] = stack("dones").astype(bool)
            rec["dones_source"] = "dones"
        else:
            # Old dumps lack dones: infer resets from robot position jumps.
            jumps = np.linalg.norm(np.diff(rec["pos"][:, :, :2], axis=0), axis=-1) > RESET_JUMP_M
            rec["dones"] = np.concatenate([np.zeros((1, len(keys)), dtype=bool), jumps], axis=0)
            rec["dones_source"] = "robot_pos_jump"
        rec["depth"] = (
            np.stack([f[keys[0]][d][0] for d in DEPTH_KEYS]).astype(np.float32) / 1000.0
            if all(d in f[keys[0]] for d in DEPTH_KEYS)
            else None
        )
    return rec


def replay(stepper: PerceptionStepper, rec: dict, device: str, log_every: int = 200) -> np.ndarray:
    """Step the model over the whole recording; returns predictions (T, B, H, W) in meters."""
    T, B = rec["sparse"].shape[:2]
    preds = np.empty((T, B, MAP_SIZE, MAP_SIZE), dtype=np.float32)
    with torch.inference_mode():
        for t in range(T):
            pred = stepper.step(
                torch.from_numpy(rec["sparse"][t]).unsqueeze(1).to(device),
                torch.from_numpy(rec["occ"][t]).unsqueeze(1).to(device),
                torch.from_numpy(rec["pos"][t]).to(device),
                torch.from_numpy(rec["quat"][t]).to(device),
                torch.from_numpy(rec["dones"][t]).to(device),
            )
            preds[t] = pred.squeeze(1).cpu().numpy()
            if log_every and (t + 1) % log_every == 0:
                print(f"  {t + 1}/{T}")
    return preds


def measure_cost(model, entry, rec, device, warmup, steps) -> dict:
    """Per-frame cost for one robot: the model step and the depth projector."""
    one = lambda a: torch.from_numpy(np.ascontiguousarray(a[0, :1])).to(device)  # noqa: E731
    sparse, occ = one(rec["sparse"]).unsqueeze(1), one(rec["occ"]).unsqueeze(1)
    pos, quat = one(rec["pos"]), one(rec["quat"])

    stepper = PerceptionStepper(model, entry.kind, 1, device)
    with torch.inference_mode():
        model_cost = time_callable(
            lambda: stepper.step(sparse, occ, pos, quat), device, warmup=warmup, steps=steps
        )

    projector = DepthProjector(map_size=MAP_SIZE, map_res=MAP_RES, device=device)
    if rec["depth"] is not None:
        depth = torch.from_numpy(rec["depth"]).unsqueeze(0).to(device)
        depth_source = "recorded frame 0 of env_0"
    else:
        depth = torch.full((1, 4, 240, 424), 1.5, device=device)
        depth_source = "synthetic 1.5 m, 4 x 240 x 424"
    with torch.inference_mode():
        projector_cost = time_callable(
            lambda: projector(depth, quat), device, warmup=warmup, steps=steps
        )

    return {
        "device": device_name(device),
        "batch": 1,
        "model": {
            "name": entry.name,
            "params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            **model_cost,
        },
        "projector": {"input": depth_source, **projector_cost},
    }


def print_cost(cost: dict) -> None:
    for part in ("model", "projector"):
        c = cost[part]
        mem = f"{c['peak_mem_mb']:.0f} MB" if c["peak_mem_mb"] is not None else "n/a"
        print(
            f"  {part:9s} p50 {c['ms_p50']:.3f} ms  p95 {c['ms_p95']:.3f} ms  "
            f"{c['fps']:.0f} FPS  peak mem {mem}"
        )
    print(f"  device: {cost['device']}, batch 1")


def write_output(out_path, src_path, rec, preds, attrs, copy_depth) -> None:
    """Copy the source run, swap in the predictions, set attrs, and mark it complete last."""
    with h5py.File(src_path, "r") as f_in, h5py.File(out_path, "w") as f_out:
        for i, k in enumerate(rec["env_keys"]):
            grp = f_out.create_group(k)
            for name in f_in[k]:
                if name in ("pred_height", "diff_height"):
                    continue
                if name in DEPTH_KEYS and not copy_depth:
                    continue
                f_in[k].copy(name, grp)
            grp.create_dataset("pred_height", data=to_int16_mm(preds[:, i]), compression="gzip")
            grp.create_dataset(
                "diff_height",
                data=to_int16_mm(np.abs(rec["gt"][:, i] - preds[:, i])),
                compression="gzip",
            )

        # The source's own completion marker and timing describe the source, not this file.
        for key, val in f_in.attrs.items():
            if key not in ("complete", "timing"):
                f_out.attrs[key] = val
        for key, val in attrs.items():
            f_out.attrs[key] = val
        f_out.flush()
        f_out.attrs["complete"] = True


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Replay a learned perception model on a recorded data.h5."
    )
    parser.add_argument("input", type=str, help="Source data.h5 (from evaluate_perception.py).")
    parser.add_argument(
        "--model", type=str, required=True, help="Registry name, e.g. v3 or v6_archived."
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Model checkpoint (.pt); required to replay."
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output dir (default: new logs/perception/PE_<model>-replay_* dir).",
    )
    parser.add_argument(
        "--copy_depth",
        action="store_true",
        help="Also copy the raw depth images (large; the source file keeps them).",
    )
    parser.add_argument(
        "--timing_only", action="store_true", help="Measure the per-frame cost, do not replay."
    )
    parser.add_argument("--no_timing", action="store_true", help="Skip the cost measurement.")
    parser.add_argument("--timing_steps", type=int, default=1000, help="Timed steps.")
    parser.add_argument("--timing_warmup", type=int, default=50, help="Untimed warm-up steps.")
    args = parser.parse_args(argv)

    try:
        entry = resolve_model(args.model)
    except ModelRegistryError as err:
        parser.error(str(err))
    if entry.kind == BASELINE:
        parser.error(f"'{entry.name}' is a non-learned baseline; use run_slam_offline.py")
    if not args.checkpoint and not args.timing_only:
        parser.error("--checkpoint is required to replay (only --timing_only runs without one)")
    if args.timing_only and args.no_timing:
        parser.error("--timing_only and --no_timing exclude each other")

    # Same input, same output: no autotuned kernel choice between runs.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = args.device
    model, info = load_perception_model(entry, args.checkpoint, device)
    rec = load_recording(args.input)
    T, B = rec["sparse"].shape[:2]

    cost = None
    if not args.no_timing:
        print(f"Measuring per-frame cost of {entry.name} ({args.timing_steps} steps, batch 1)")
        cost = measure_cost(model, entry, rec, device, args.timing_warmup, args.timing_steps)
        print_cost(cost)

    if args.output:
        save_dir = Path(args.output)
    else:
        save_dir = make_evaluation_dir("perception", f"{entry.name}-replay")
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.timing_only:
        with open(save_dir / "timing.json", "w") as f:
            json.dump(cost, f, indent=2)
        print(f"Wrote {save_dir / 'timing.json'}")
        return save_dir

    print(
        f"Replaying {entry.name} on {args.input}: {T} steps x {B} envs ({device}), "
        f"resets from {rec['dones_source']}"
    )
    stepper = PerceptionStepper(model, entry.kind, B, device)
    preds = replay(stepper, rec, device)

    out_path = save_dir / "data.h5"
    print(f"Writing {out_path}")
    attrs = {
        "method": "model",
        "model": entry.name,
        "model_class": entry.class_path,
        "model_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "perception_checkpoint": info["checkpoint"],
        "checkpoint_sha256": info["checkpoint_sha256"],
        "checkpoint_epoch": -1 if info["epoch"] is None else int(info["epoch"]),
        "replay_of": os.path.abspath(args.input),
        "replay_of_sha256": file_sha256(args.input),
        "resets_from": rec["dones_source"],
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": get_git_suffix(),
    }
    if cost is not None:
        attrs["timing"] = json.dumps(cost)
    write_output(out_path, args.input, rec, preds, attrs, args.copy_depth)
    print("Done")
    return save_dir


if __name__ == "__main__":
    main()
