# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Perception model registry tests.

The registry maps thesis names (v1, v2, v3) and archived code-era names (v5_archived, ...)
to model classes. These tests pin the roster, the parameter counts, the error messages for
retired names, and that thesis v3 is the same network as the frozen v5_archived.

No Isaac imports — runs anywhere with torch:  python tests/test_model_registry.py
"""

import importlib.util
import os
import subprocess
import sys

import torch

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(REPO, "source"))

from meldog_rl.models.perception import (  # noqa: E402
    ACTIVE_MODEL,
    MODELS,
    HeightmapConvGRU,
    HeightmapV1,
    HeightmapV2,
    HeightmapV3,
    ModelRegistryError,
    build_model,
    format_table,
    resolve_model,
    resolve_trainable,
)

REGISTRY_FILE = os.path.join(REPO, "source", "meldog_rl", "models", "perception", "registry.py")

# name: (status, kind, runnable, trainable, thesis section)
ROSTER = {
    "v1": ("thesis", "single_frame", True, False, "2.6.1"),
    "v2": ("thesis", "single_frame", True, False, "2.6.2"),
    "v3": ("active", "recurrent", True, True, "2.6.3"),
    "v5_archived": ("archived", "recurrent", True, False, None),
    "v6_archived": ("archived", "autoregressive", True, False, None),
    "v1_archived": ("archived", None, False, False, None),
    "elevation": ("baseline", "baseline", True, False, None),
    "legacy_slam": ("baseline", "baseline", True, False, None),
}

PARAMS = {
    "v1": 132_403,
    "v2": 570_387,
    "v3": 2_496_275,
    "v5_archived": 2_496_275,
    "v6_archived": 573_910,
}


def expect_error(fn, *needles):
    """Call fn, require a ModelRegistryError whose message contains every needle."""
    try:
        fn()
    except ModelRegistryError as err:
        msg = str(err)
        for needle in needles:
            assert needle in msg, f"{needle!r} not in {msg!r}"
        return msg
    raise AssertionError(f"no ModelRegistryError (expected {needles})")


def test_roster():
    assert list(MODELS) == list(ROSTER), list(MODELS)
    for name, (status, kind, runnable, trainable, thesis) in ROSTER.items():
        e = MODELS[name]
        assert e.name == name
        assert (e.status, e.kind, e.runnable, e.trainable, e.thesis) == (
            status,
            kind,
            runnable,
            trainable,
            thesis,
        ), name
        assert e.summary and e.source, name
        if e.file is not None:
            assert os.path.isfile(os.path.join(REPO, e.file)), e.file
    assert MODELS["v1_archived"].class_path is None and MODELS["v1_archived"].params == 781_857
    assert ACTIVE_MODEL == "v3" and MODELS[ACTIVE_MODEL].status == "active"
    assert [e.name for e in MODELS.values() if e.trainable] == ["v3"]
    assert MODELS["v3"].load_class() is HeightmapV3
    assert MODELS["v5_archived"].load_class() is HeightmapConvGRU
    assert MODELS["v1"].load_class() is HeightmapV1 and MODELS["v2"].load_class() is HeightmapV2
    table = format_table()
    for name in ROSTER:
        assert name in table, name
    print("  OK  roster, statuses, kinds and flags")


def test_param_counts():
    for name, expected in PARAMS.items():
        model = build_model(name)
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert n == expected == MODELS[name].params, (name, n, expected)
        print(f"  OK  {name:12s} {n:>10,} params")


def test_retired_names():
    for old in ("v5", "heightmap_v5", "hmv5", "V5"):
        expect_error(lambda old=old: resolve_model(old), f"'{old}'", "v5_archived", "'v3'")
    for old in ("v6", "heightmap_v6", "hmv6"):
        expect_error(lambda old=old: resolve_model(old), f"'{old}'", "v6_archived")
    print("  OK  v5/v6 names point to v5_archived/v6_archived")


def test_unknown_and_record_only():
    msg = expect_error(lambda: resolve_model("v9"), "unknown model 'v9'")
    for name in ROSTER:
        assert name in msg, name
    expect_error(lambda: resolve_model("v1_archived"), "not runnable", "SimpleMapper")
    expect_error(lambda: build_model(MODELS["v1_archived"]), "not runnable")
    expect_error(lambda: build_model("elevation"), "non-learned baseline")
    assert resolve_model(" V3 ").name == "v3"
    print("  OK  unknown names list the roster; v1_archived is not runnable")


def test_trainability():
    assert resolve_trainable("v3") is MODELS["v3"]
    for name in ("v1", "v2"):
        expect_error(lambda name=name: resolve_trainable(name), "later step")
    for name in ("v5_archived", "v6_archived"):
        expect_error(lambda name=name: resolve_trainable(name), "frozen", "train v3")
    for name in ("elevation", "legacy_slam"):
        expect_error(lambda name=name: resolve_trainable(name), "baseline")
    expect_error(lambda: resolve_trainable("heightmap_v5"), "v5_archived")
    print("  OK  only v3 is trainable, each refusal says why")


def test_v3_is_v5_archived():
    torch.manual_seed(0)
    archived = build_model("v5_archived")
    # A few train-mode passes so BatchNorm running stats are not the defaults.
    archived.train()
    with torch.no_grad():
        for _ in range(3):
            archived(torch.randn(4, 1, 40, 40), torch.rand(4, 1, 40, 40).round(), torch.randn(4, 3))

    v3 = build_model("v3")
    assert list(v3.state_dict()) == list(archived.state_dict())
    v3.load_state_dict(archived.state_dict(), strict=True)
    archived.load_state_dict(v3.state_dict(), strict=True)  # and back
    archived.eval()
    v3.eval()

    B, T = 3, 5
    x = torch.randn(B, 1, 40, 40)
    mask = torch.rand(B, 1, 40, 40).round()
    grav = torch.randn(B, 3)
    h = [torch.randn(B, 128, 5, 5) for _ in range(2)]
    seq = (torch.randn(T, B, 1, 40, 40), torch.rand(T, B, 1, 40, 40).round(), torch.randn(T, B, 3))
    with torch.no_grad():
        pred_a, h_a = archived(x, mask, grav, h)
        pred_b, h_b = v3(x, mask, grav, h)
        seq_a = archived.forward_sequence(*seq)
        seq_b = v3.forward_sequence(*seq)
    assert torch.equal(pred_a, pred_b)
    assert len(h_a) == len(h_b) == 2 and all(torch.equal(a, b) for a, b in zip(h_a, h_b))
    assert torch.equal(seq_a, seq_b)
    print("  OK  v3 loads a v5_archived state_dict (strict) and gives bitwise-identical output")


def test_single_frame_forward():
    B = 2
    x = torch.randn(B, 1, 40, 40)
    mask = torch.rand(B, 1, 40, 40).round()
    grav = torch.randn(B, 3)
    for name in ("v1", "v2"):
        model = build_model(name).eval()
        with torch.no_grad():
            pred = model(x, mask, grav)
        assert pred.shape == (B, 1, 40, 40), (name, pred.shape)
        assert torch.isfinite(pred).all(), name
    print("  OK  v1 / v2: (B,1,40,40) sparse + mask + (B,3) gravity -> (B,1,40,40)")


def test_build_model_kwargs():
    m = build_model("v3", gru_hidden=64, gru_layers=1)
    assert (m.gru_hidden, m.gru_layers) == (64, 1)
    m = build_model("v6_archived", gru_hidden=64, gru_layers=1)  # ignored: no GRU
    assert sum(p.numel() for p in m.parameters()) == PARAMS["v6_archived"]
    print("  OK  build_model passes gru_* only to models that take them")


def test_registry_file_needs_no_torch():
    """evaluate_perception.py loads registry.py on its own before Isaac Sim starts."""
    code = (
        "import importlib.util, sys\n"
        "sys.modules['torch'] = None  # any torch import now fails\n"
        f"spec = importlib.util.spec_from_file_location('reg', {REGISTRY_FILE!r})\n"
        "reg = importlib.util.module_from_spec(spec)\n"
        "sys.modules['reg'] = reg\n"
        "spec.loader.exec_module(reg)\n"
        "print(reg.resolve_model('v3').name, len(reg.MODELS))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["v3", str(len(ROSTER))], out.stdout
    assert importlib.util.find_spec("meldog_rl.models.perception.registry") is not None
    print("  OK  registry.py loads standalone without torch")


if __name__ == "__main__":
    test_roster()
    test_param_counts()
    test_retired_names()
    test_unknown_and_record_only()
    test_trainability()
    test_v3_is_v5_archived()
    test_single_frame_forward()
    test_build_model_kwargs()
    test_registry_file_needs_no_torch()
    print("All model registry tests passed.")
