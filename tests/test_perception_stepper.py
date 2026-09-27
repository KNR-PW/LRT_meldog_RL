# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Per-robot model memory, offline replay and the completion marker.

Several robots are stepped through one model in one batch. Each must get exactly what it
would get alone: its own ConvGRU hidden state (recurrent models), its own previous
prediction warped by its own motion (autoregressive models), and a reset that clears only
the robots it names. The offline replay must give the same predictions every run, and the
analyzer must refuse a file whose writer did not finish.

No Isaac imports — runs anywhere with torch:  python tests/test_perception_stepper.py
"""

import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile

import h5py
import numpy as np
import torch

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(REPO, "source"))

from meldog_rl.models.perception import (  # noqa: E402
    PerceptionStepper,
    build_model,
    gravity_from_quat,
    resolve_model,
)

REPLAY_SCRIPT = os.path.join(REPO, "scripts", "perception", "run_model_offline.py")
ANALYZER_SCRIPT = os.path.join(REPO, "scripts", "perception", "analyze_perception.py")
T, H = 8, 40
ATOL = 1e-5


def _load_script(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _model(name):
    torch.manual_seed(0)
    entry = resolve_model(name)
    return build_model(entry).eval(), entry.kind


def _yaw_quat(yaw):
    return torch.stack(
        [torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)], -1
    )


def _sequences(num_envs, seed=1):
    """Per-robot inputs (T, B, ...): robot 0 walks and turns, the others move differently."""
    g = torch.Generator().manual_seed(seed)
    sparse = torch.randn(T, num_envs, 1, H, H, generator=g) * 0.1
    mask = (torch.rand(T, num_envs, 1, H, H, generator=g) > 0.6).float()
    steps = torch.arange(T, dtype=torch.float32).view(T, 1)
    speed = torch.tensor([0.02 * (i + 1) for i in range(num_envs)]).view(1, num_envs)
    turn = torch.tensor([0.05 * (-1) ** i for i in range(num_envs)]).view(1, num_envs)
    yaw = steps * turn
    pos = torch.stack([steps * speed, steps * speed * 0.3, 0.4 + 0.01 * steps * speed], -1)
    quat = _yaw_quat(yaw)
    return sparse, mask, pos, quat


def _run(model, kind, sparse, mask, pos, quat, dones=None):
    """Step a batch through a fresh stepper; returns (T, B, 1, H, W)."""
    B = sparse.shape[1]
    stepper = PerceptionStepper(model, kind, B, "cpu")
    out = []
    with torch.inference_mode():
        for t in range(sparse.shape[0]):
            d = None if dones is None else dones[t]
            out.append(stepper.step(sparse[t], mask[t], pos[t], quat[t], d))
    return torch.stack(out)


def test_gravity_from_quat():
    level = gravity_from_quat(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    assert torch.allclose(level, torch.tensor([[0.0, 0.0, -1.0]]))
    yawed = gravity_from_quat(_yaw_quat(torch.tensor([1.3])))
    assert torch.allclose(yawed, torch.tensor([[0.0, 0.0, -1.0]]), atol=1e-6), "yaw moves gravity"
    print("  OK  gravity_from_quat")


def test_batch_equals_alone():
    """The B2 regression: robots stepped together get exactly what they get alone."""
    sparse, mask, pos, quat = _sequences(3)
    for name in ("v3", "v6_archived", "v1"):
        model, kind = _model(name)
        together = _run(model, kind, sparse, mask, pos, quat)
        for i in range(3):
            s = slice(i, i + 1)
            alone = _run(model, kind, sparse[:, s], mask[:, s], pos[:, s], quat[:, s])
            diff = (together[:, s] - alone).abs().max().item()
            assert diff < ATOL, f"{name}: robot {i} differs by {diff} when batched"
        print(f"  OK  {name}: 3 robots in one batch == each alone")


def test_memory_matters():
    """Guard for the test above: a recurrent/autoregressive model must actually use memory."""
    sparse, mask, pos, quat = _sequences(1)
    for name in ("v3", "v6_archived"):
        model, kind = _model(name)
        seq = _run(model, kind, sparse, mask, pos, quat)
        last_alone = _run(model, kind, sparse[-1:], mask[-1:], pos[-1:], quat[-1:])
        diff = (seq[-1] - last_alone[0]).abs().max().item()
        assert diff > 1e-4, f"{name}: output ignores its memory (diff {diff})"
    print("  OK  v3 and v6_archived outputs depend on their memory")


def test_reset_clears_only_named_robot():
    """Robot 1 resets at t=4: it restarts from scratch, robot 0 carries on untouched."""
    sparse, mask, pos, quat = _sequences(2)
    t_reset = 4
    dones = torch.zeros(T, 2, dtype=torch.bool)
    dones[t_reset, 1] = True
    for name in ("v3", "v6_archived"):
        model, kind = _model(name)
        out = _run(model, kind, sparse, mask, pos, quat, dones)
        r0 = _run(model, kind, sparse[:, :1], mask[:, :1], pos[:, :1], quat[:, :1])
        r1 = _run(
            model,
            kind,
            sparse[t_reset:, 1:],
            mask[t_reset:, 1:],
            pos[t_reset:, 1:],
            quat[t_reset:, 1:],
        )
        assert (out[:, 0] - r0[:, 0]).abs().max().item() < ATOL, f"{name}: robot 0 was disturbed"
        assert (out[t_reset:, 1] - r1[:, 0]).abs().max().item() < ATOL, f"{name}: bad reset"
        print(f"  OK  {name}: reset of robot 1 leaves robot 0 alone and restarts robot 1")


def _write_recording(path, num_envs=2, depth=False):
    sparse, mask, pos, quat = _sequences(num_envs, seed=3)
    gt = torch.randn(T, num_envs, H, H, generator=torch.Generator().manual_seed(4)) * 0.1
    mm = lambda a: (np.clip(a, -32, 32) * 1000).astype(np.int16)  # noqa: E731
    with h5py.File(path, "w") as f:
        for i in range(num_envs):
            g = f.create_group(f"env_{i}")
            g["sparse_height"] = mm(sparse[:, i, 0].numpy())
            g["occlusion_mask"] = mask[:, i, 0].numpy().astype(np.uint8)
            g["gt_height"] = mm(gt[:, i].numpy())
            g["pred_height"] = np.zeros((T, H, H), dtype=np.int16)
            g["diff_height"] = np.zeros((T, H, H), dtype=np.int16)
            g["robot_pos"] = pos[:, i].numpy()
            g["robot_quat"] = quat[:, i].numpy()
            dones = np.zeros(T, dtype=np.uint8)
            dones[5] = i == 1
            g["dones"] = dones
            if depth:
                for d in ("depth_front", "depth_rear", "depth_left", "depth_right"):
                    g[d] = np.full((T, 24, 42), 1500, dtype=np.uint16)
        f.attrs["task"] = "synthetic"
        f.attrs["complete"] = True


def test_replay_is_deterministic():
    replay = _load_script(REPLAY_SCRIPT, "run_model_offline")
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "src.h5")
        _write_recording(src, depth=True)
        model, _ = _model("v3")
        ckpt = os.path.join(tmp, "model.pt")
        torch.save({"model_state_dict": model.state_dict(), "epoch": 7}, ckpt)

        outs = []
        for run in ("a", "b"):
            out_dir = os.path.join(tmp, run)
            argv = [src, "--model", "v3", "--checkpoint", ckpt, "--device", "cpu"]
            replay.main(argv + ["--output", out_dir, "--no_timing"])
            outs.append(os.path.join(out_dir, "data.h5"))

        with h5py.File(outs[0], "r") as a, h5py.File(outs[1], "r") as b:
            for k in ("env_0", "env_1"):
                assert np.array_equal(a[k]["pred_height"][:], b[k]["pred_height"][:])
                assert "depth_front" not in a[k], "raw depth copied without --copy_depth"
                assert np.array_equal(a[k]["gt_height"][:], b[k]["gt_height"][:])
            assert a.attrs["complete"] and a.attrs["model"] == "v3"
            assert a.attrs["replay_of"] == os.path.abspath(src)
            assert int(a.attrs["checkpoint_epoch"]) == 7
            assert not np.all(a["env_0"]["pred_height"][:] == 0), "predictions not written"

        # The replay must equal stepping the model directly on the same (quantized) inputs.
        rec = replay.load_recording(src)
        direct = replay.replay(PerceptionStepper(model, "recurrent", 2, "cpu"), rec, "cpu", 0)
        with h5py.File(outs[0], "r") as a:
            stored = a["env_1"]["pred_height"][:]
        assert np.array_equal(stored, replay.to_int16_mm(direct[:, 1])), "replay != direct stepping"
    print("  OK  replay: two runs identical, depth skipped, provenance and complete written")


def test_timing_only():
    replay = _load_script(REPLAY_SCRIPT, "run_model_offline")
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "src.h5")
        _write_recording(src)
        out_dir = os.path.join(tmp, "t")
        argv = [src, "--model", "v6_archived", "--timing_only", "--device", "cpu"]
        replay.main(argv + ["--timing_steps", "3", "--timing_warmup", "1", "--output", out_dir])
        with open(os.path.join(out_dir, "timing.json")) as f:
            cost = json.load(f)
        assert cost["model"]["params"] == 573910 and cost["batch"] == 1
        for part in ("model", "projector"):
            assert cost[part]["ms_p50"] > 0 and math.isfinite(cost[part]["fps"])
        assert not os.path.exists(os.path.join(out_dir, "data.h5"))
    print("  OK  --timing_only writes timing.json for the model and the projector")


def test_analyzer_refuses_incomplete():
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "run", "data.h5")
        os.makedirs(os.path.dirname(src))
        _write_recording(src)
        with h5py.File(src, "a") as f:
            del f.attrs["complete"]

        cmd = [sys.executable, ANALYZER_SCRIPT, src, "-o", os.path.join(tmp, "out")]
        refused = subprocess.run(cmd, capture_output=True, text=True)
        assert refused.returncode != 0 and "complete=True" in refused.stderr, refused.stderr

        allowed = subprocess.run(cmd + ["--allow_incomplete"], capture_output=True, text=True)
        assert allowed.returncode == 0, allowed.stderr
        with open(os.path.join(tmp, "out", "metrics.json")) as f:
            meta = json.load(f)["meta"]
        assert meta["incomplete_input"] is True and meta["incomplete_inputs"] == [src]
    print("  OK  analyzer refuses a file without complete=True unless --allow_incomplete")


if __name__ == "__main__":
    test_gravity_from_quat()
    test_batch_equals_alone()
    test_memory_matters()
    test_reset_clears_only_named_robot()
    test_replay_is_deterministic()
    test_timing_only()
    test_analyzer_refuses_incomplete()
    print("All perception stepper tests passed.")
