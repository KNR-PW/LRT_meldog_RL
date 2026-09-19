#!/usr/bin/env python3
# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Offline, sim-free analyzer for perception evaluation rollouts.

Consumes the ``data.h5`` dumped by ``evaluate_perception.py`` / ``evaluate_slam.py``
and produces machine-readable metrics + static plots that can be inspected directly,
without relaunching Isaac Sim. Runs anywhere in seconds (numpy + h5py + matplotlib,
**no isaaclab / torch imports**).

Occlusion mask semantics (verified in
``source/meldog_rl/models/perception/projector.py``):
    The projector documents and produces ``occlusion_mask`` where
        1.0 == NO data  (cell is occluded / never observed by the depth cameras)
        0.0 == data      (cell was observed this step)
    Therefore, in this analyzer:
        visible  cells := occlusion_mask == 0   -> ``rmse_visible``
        occluded cells := occlusion_mask == 1   -> ``rmse_occluded``  (thesis-critical:
            reconstruction quality where the robot cannot currently see).

Height storage:
    Height maps (gt_height / pred_height / sparse_height / diff_height) are stored as
    int16 millimeters (see ``to_int16_mm`` in evaluate_perception.py); this script
    converts them to meters (value / 1000.0). Float datasets are assumed to already be
    in meters.

No-data sentinel rule:
    The projector initializes unobserved cells to -5.0 m and clamps to -2.0 m, so
    sparse_height (and SLAM pred_height in never-observed cells) carries a ~-2.0 m
    sentinel that encodes *absence of data*, not a height estimate. Including it
    would let the arbitrary sentinel magnitude dominate the error metrics. Rule:
        cells where the prediction's own value < SENTINEL_THRESH (-1.9 m) are
        treated as "no data" and EXCLUDED from rmse_all, mae_all,
        rmse_visible/occluded, rmse_radial and the error-vs-time curves.
    ``coverage`` reports the fraction of non-sentinel cells (dense model outputs
    -> 1.0; SLAM / sparse input -> below 1.0), so a sparse predictor cannot win
    the RMSE comparison simply by filling fewer cells. For ``edge_mae`` the
    sentinel mask is dilated by 1 cell before exclusion — Sobel responses across
    a data/sentinel boundary measure the sentinel step, not real edges. In the
    panel plots, sentinel cells render gray so they don't read as -0.5 m terrain.

Episode resets:
    The perception dump has no ``dones`` dataset (known recorder gap — to be added
    to the recorder later), so resets are detected offline: a jump in ``robot_pos``
    of > RESET_JUMP_M (0.5 m) between consecutive steps marks a reset (env teleport,
    stale remembered map -> error spike). The raw ``error_vs_time`` curve keeps the
    spikes but they are marked in the plot; ``error_vs_time_since_reset`` averages
    per-step RMSE across all episode segments aligned at reset (NaN-pad + nanmean)
    — the actual memory-accumulation curve.

Schema tolerance:
    The h5 may be a per-env layout (``env_0/``, ``env_1/`` ... groups) or a flat layout
    (height datasets at the file root). Missing optional datasets degrade gracefully:
    without ``occlusion_mask`` the visible/occluded split is reported ``null``; without
    ``sparse_height`` the sparse panel is omitted; without ``robot_pos`` reset
    detection is skipped. All detected schema differences are recorded in
    ``meta.schema_notes`` and echoed in ``report.md``.

Usage:
    python scripts/perception/analyze_perception.py MODEL.h5
    python scripts/perception/analyze_perception.py MODEL.h5 SLAM.h5 --labels model slam
"""

import argparse
import json
import subprocess
import warnings
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MAP_SIZE = 40
MAP_RES = 0.05  # meters / cell

# Fixed display scales (shared with the eval-video panels for visual consistency).
HEIGHT_CLIP = 0.5  # viridis, +/- 0.5 m
ERROR_CLIP = 0.2  # hot, 0 .. 0.2 m
DEFAULT_TIMESTEPS = [50, 200, 500, 950]

# Values below this are the projector's no-data sentinel (clamp floor -2.0 m), not
# height estimates — see "No-data sentinel rule" in the module docstring.
SENTINEL_THRESH = -1.9  # meters

# robot_pos jump between consecutive steps larger than this = episode reset.
RESET_JUMP_M = 0.5  # meters

# --------------------------------------------------------------------------- #
# Benchmark reference ranges
# --------------------------------------------------------------------------- #
# Thresholds copied from docs/evaluation.md (the human-readable source of truth;
# do NOT parse that markdown at runtime — keep these in sync by hand). Perception
# bands are thesis-internal orderings, so we only flag what a single run can
# support (rmse_visible absolute floor; occluded-vs-visible ratio). The model<SLAM
# comparison checks are emitted only when two inputs are supplied.
FLAG_EMOJI = {"good": "✅", "acceptable": "⚠️", "investigate": "❌"}
RMSE_VISIBLE_GOOD = 0.05  # m, evaluation.md "rmse_visible < 0.05 m"
RMSE_VISIBLE_ACCEPTABLE = 0.10  # 2x soft margin -> ⚠️ tier
OCCLUDED_VS_VISIBLE_MAX = 4.0  # evaluation.md "rmse_occluded <= 4x rmse_visible"


def _finite(x):
    """Return float(x) if finite, else None (None/NaN/inf -> None)."""
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


def compute_perception_flags(results):
    """Map metric keys -> good/acceptable/investigate per docs/evaluation.md.

    ``results`` is a list of (label, metrics, schema). Single-run flags come from the
    primary (first) input; model<SLAM comparison flags are added only when exactly two
    inputs are given (order: [model, slam]). Null/NaN metrics get no flag.
    """
    flags = {}
    primary = results[0][1]

    rv = _finite(primary.get("rmse_visible"))
    if rv is not None:
        flags["rmse_visible"] = (
            "good"
            if rv < RMSE_VISIBLE_GOOD
            else "acceptable" if rv < RMSE_VISIBLE_ACCEPTABLE else "investigate"
        )
    ro = _finite(primary.get("rmse_occluded"))
    if rv is not None and ro is not None and rv > 0:
        flags["rmse_occluded_vs_visible"] = (
            "good" if (ro / rv) <= OCCLUDED_VS_VISIBLE_MAX else "investigate"
        )

    if len(results) == 2:
        model, slam = results[0][1], results[1][1]

        def _cmp(mk, allow_equal=False):
            a, b = _finite(model.get(mk)), _finite(slam.get(mk))
            if a is None or b is None:
                return None
            ok = (a <= b) if allow_equal else (a < b)
            return "good" if ok else "investigate"

        for mk, fk, eq in [
            ("rmse_all", "rmse_all_model_lt_slam", False),
            ("rmse_occluded", "rmse_occluded_model_lt_slam", False),
            ("edge_mae", "edge_mae_model_le_slam", True),
        ]:
            v = _cmp(mk, eq)
            if v is not None:
                flags[fk] = v
    return flags


# --------------------------------------------------------------------------- #
# Loading / schema handling
# --------------------------------------------------------------------------- #
def _to_meters(arr: np.ndarray) -> np.ndarray:
    """Convert a height dataset to float32 meters.

    Integer datasets are int16 millimeters (see ``to_int16_mm``); float datasets are
    assumed already in meters.
    """
    arr = np.asarray(arr)
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.float32) / 1000.0
    return arr.astype(np.float32)


def _decode_attr(v):
    """Normalize an h5 attr to a JSON-friendly python scalar/str."""
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.generic):
        return v.item()
    return v


def _iter_env_groups(f: h5py.File):
    """Yield (env_name, accessor) pairs, tolerating per-env and flat layouts."""
    env_names = sorted(
        (k for k in f.keys() if k.startswith("env_") and isinstance(f[k], h5py.Group)),
        key=lambda s: int(s.split("_")[1]) if s.split("_")[1].isdigit() else s,
    )
    if env_names:
        for name in env_names:
            yield name, f[name]
    elif "gt_height" in f:
        # Flat layout: treat the file root as a single env.
        yield "env_0", f
    else:
        raise ValueError(
            "Unrecognized h5 layout: no 'env_*' groups and no root 'gt_height' dataset."
        )


def load_h5(path: Path):
    """Load height stacks from a perception ``data.h5``.

    Returns a dict with per-quantity arrays stacked as (num_envs, T, H, W) in meters,
    ``occlusion_mask`` (num_envs, T, H, W) uint8 or ``None``, ``robot_pos``
    (num_envs, T, 3) or ``None``, and a ``schema`` report describing which datasets
    were present.
    """
    gt_list, pred_list, sparse_list, mask_list, pos_list, dones_list = [], [], [], [], [], []
    present = None
    env_count = 0
    grouped = False
    file_attrs = {}

    with h5py.File(path, "r") as f:
        grouped = any(k.startswith("env_") and isinstance(f[k], h5py.Group) for k in f.keys())
        file_attrs = {k: _decode_attr(f.attrs[k]) for k in f.attrs}
        for _name, grp in _iter_env_groups(f):
            keys = set(grp.keys())
            if present is None:
                present = keys
            if "gt_height" not in keys or "pred_height" not in keys:
                raise ValueError(
                    f"{path}: env group missing required gt_height/pred_height "
                    f"(has {sorted(keys)})."
                )
            gt_list.append(_to_meters(grp["gt_height"][:]))
            pred_list.append(_to_meters(grp["pred_height"][:]))
            sparse_list.append(
                _to_meters(grp["sparse_height"][:]) if "sparse_height" in keys else None
            )
            mask_list.append(
                np.asarray(grp["occlusion_mask"][:]) if "occlusion_mask" in keys else None
            )
            pos_list.append(
                np.asarray(grp["robot_pos"][:], dtype=np.float32) if "robot_pos" in keys else None
            )
            dones_list.append(
                np.asarray(grp["dones"][:], dtype=np.uint8) if "dones" in keys else None
            )
            env_count += 1

    # Align to the shortest sequence length across envs (robustness; normally equal).
    t_min = min(a.shape[0] for a in gt_list)
    gt = np.stack([a[:t_min] for a in gt_list], axis=0)
    pred = np.stack([a[:t_min] for a in pred_list], axis=0)

    has_sparse = all(a is not None for a in sparse_list)
    sparse = np.stack([a[:t_min] for a in sparse_list], axis=0) if has_sparse else None

    has_mask = all(a is not None for a in mask_list)
    mask = np.stack([a[:t_min] for a in mask_list], axis=0) if has_mask else None

    has_pos = all(a is not None for a in pos_list)
    robot_pos = np.stack([a[:t_min] for a in pos_list], axis=0) if has_pos else None

    has_dones = all(a is not None for a in dones_list)
    dones = np.stack([a[:t_min] for a in dones_list], axis=0) if has_dones else None

    schema = {
        "layout": "per_env_groups" if grouped else "flat",
        "num_envs": env_count,
        "num_steps": int(t_min),
        "map_shape": list(gt.shape[-2:]),
        "datasets_present": sorted(present) if present else [],
        "has_occlusion_mask": bool(has_mask),
        "has_sparse_height": bool(has_sparse),
        "has_robot_pos": bool(has_pos),
        "has_dones": bool(has_dones),
        "reset_source": "dones" if has_dones else ("robot_pos_jump" if has_pos else None),
        "notes": [],
    }
    if not has_mask:
        schema["notes"].append(
            "occlusion_mask absent -> rmse_visible/rmse_occluded reported as null."
        )
    if not has_sparse:
        schema["notes"].append("sparse_height absent -> sparse panel omitted in plots.")
    if has_dones:
        schema["notes"].append("episode resets read from the real `dones` dataset.")
    elif has_pos:
        schema["notes"].append(
            "dones absent (older dump) -> episode resets inferred from robot_pos "
            f"jumps > {RESET_JUMP_M} m (fallback heuristic)."
        )
    else:
        schema["notes"].append(
            "no dones and no robot_pos -> reset detection skipped, "
            "error_vs_time_since_reset reported as null."
        )

    return {
        "gt": gt,
        "pred": pred,
        "sparse": sparse,
        "mask": mask,
        "robot_pos": robot_pos,
        "dones": dones,
        "attrs": file_attrs,
        "schema": schema,
    }


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _rmse(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(err)))) if err.size else float("nan")


def _radial_bins(h: int, w: int, n_bins: int = 4):
    """Per-cell radial distance from map center (meters) and the bin index per cell."""
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    dist_m = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) * MAP_RES
    max_r = (min(h, w) / 2.0) * MAP_RES  # half-extent (= 1.0 m for a 40x40 @ 0.05 grid)
    edges = np.linspace(0.0, max_r, n_bins + 1)
    # Cells beyond the half-extent (corners) fall into the outer bin.
    bin_idx = np.clip(np.digitize(dist_m, edges[1:-1]), 0, n_bins - 1)
    return bin_idx, edges


def _sobel_grad_mag(frame: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude of a 2D array, numpy-only (no cv2/scipy)."""
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = kx.T
    p = np.pad(frame, 1, mode="edge")
    gx = (
        kx[0, 0] * p[:-2, :-2]
        + kx[0, 2] * p[:-2, 2:]
        + kx[1, 0] * p[1:-1, :-2]
        + kx[1, 2] * p[1:-1, 2:]
        + kx[2, 0] * p[2:, :-2]
        + kx[2, 2] * p[2:, 2:]
    )
    gy = (
        ky[0, 0] * p[:-2, :-2]
        + ky[0, 1] * p[:-2, 1:-1]
        + ky[0, 2] * p[:-2, 2:]
        + ky[2, 0] * p[2:, :-2]
        + ky[2, 1] * p[2:, 1:-1]
        + ky[2, 2] * p[2:, 2:]
    )
    return np.sqrt(gx * gx + gy * gy)


def _dilate3x3(m: np.ndarray) -> np.ndarray:
    """Binary dilation of a 2D bool mask with a 3x3 structuring element."""
    p = np.pad(m, 1, mode="constant", constant_values=False)
    out = np.zeros_like(m)
    for dy in range(3):
        for dx in range(3):
            out |= p[dy : dy + m.shape[0], dx : dx + m.shape[1]]
    return out


def detect_resets(robot_pos: np.ndarray) -> list:
    """Detect episode resets per env from teleport-sized robot_pos jumps.

    The dump carries no ``dones`` (known recorder gap), so a between-step position
    jump > RESET_JUMP_M is treated as a reset.

    Args:
        robot_pos: (N, T, 3) world positions.

    Returns:
        Per-env list of reset step indices (the first step *after* the jump).
    """
    jumps = np.linalg.norm(np.diff(robot_pos, axis=1), axis=-1)  # (N, T-1)
    return [
        [int(t) + 1 for t in np.nonzero(jumps[n] > RESET_JUMP_M)[0]]
        for n in range(robot_pos.shape[0])
    ]


def _nan_to_none(arr) -> list:
    """JSON-safe list: NaN/inf -> null."""
    return [float(v) if np.isfinite(v) else None for v in np.asarray(arr, dtype=float)]


def compute_metrics(data: dict, n_radial: int = 4) -> dict:
    """Compute the perception metric set. Errors are in meters.

    All error metrics exclude no-data sentinel cells (pred < SENTINEL_THRESH);
    ``coverage`` reports the fraction of cells that carry a real estimate.
    """
    gt, pred, mask = data["gt"], data["pred"], data["mask"]
    err = pred - gt  # (N, T, H, W) signed error, meters
    abs_err = np.abs(err)
    N, T, H, W = gt.shape

    # No-data sentinel: the prediction's own value marks absence of an estimate.
    valid = pred >= SENTINEL_THRESH  # (N, T, H, W) True = real estimate

    metrics = {
        "rmse_all": _rmse(err[valid]),
        "mae_all": float(np.mean(abs_err[valid])) if valid.any() else float("nan"),
        "coverage": float(valid.mean()),
    }

    # Visible / occluded split (thesis-critical pair), sentinel-excluded.
    if mask is not None:
        occ = mask.astype(bool)  # True where occluded (mask == 1)
        vis_valid = ~occ & valid
        occ_valid = occ & valid
        metrics["rmse_visible"] = _rmse(err[vis_valid]) if vis_valid.any() else None
        metrics["rmse_occluded"] = _rmse(err[occ_valid]) if occ_valid.any() else None
    else:
        metrics["rmse_visible"] = None
        metrics["rmse_occluded"] = None

    # Radial bins (error vs distance from robot center), sentinel-excluded.
    bin_idx, edges = _radial_bins(H, W, n_radial)
    radial = []
    for b in range(n_radial):
        cells = valid & (bin_idx == b)  # (H, W) broadcast over (N, T, H, W)
        radial.append(_rmse(err[cells]) if cells.any() else float("nan"))
    metrics["rmse_radial"] = [float(v) for v in radial]
    metrics["radial_bin_edges_m"] = [float(e) for e in edges]

    # Edge MAE (Sobel gradient magnitude, GT vs pred) — obstacle-boundary
    # preservation. Sentinel cells dilated by 1 are excluded: Sobel across a
    # data/sentinel boundary measures the sentinel step, not a real edge.
    edge_diffs = []
    for n in range(N):
        for t in range(T):
            keep = ~_dilate3x3(~valid[n, t])
            if not keep.any():
                continue
            g_gt = _sobel_grad_mag(gt[n, t])
            g_pred = _sobel_grad_mag(pred[n, t])
            edge_diffs.append(np.mean(np.abs(g_gt - g_pred)[keep]))
    metrics["edge_mae"] = float(np.mean(edge_diffs)) if edge_diffs else float("nan")

    # Error vs time: per-step rmse averaged across envs (sentinel-excluded).
    err2 = np.where(valid, np.square(err), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN steps
        per_step = np.sqrt(np.nanmean(err2, axis=(0, 2, 3)))  # (T,)
        per_env_step = np.sqrt(np.nanmean(err2, axis=(2, 3)))  # (N, T)
    metrics["error_vs_time"] = _nan_to_none(per_step)

    # Episode resets: prefer the real `dones` dataset; fall back to robot_pos jumps
    # for older dumps that predate the recorder gap fix.
    dones = data.get("dones")
    robot_pos = data.get("robot_pos")
    if dones is not None:
        resets = [[int(t) for t in np.nonzero(dones[n] == 1)[0]] for n in range(dones.shape[0])]
        metrics["reset_source"] = "dones"
    elif robot_pos is not None:
        resets = detect_resets(robot_pos)
        metrics["reset_source"] = "robot_pos_jump"
    else:
        resets = None
        metrics["reset_source"] = None

    if resets is not None:
        metrics["reset_steps"] = resets
        metrics["num_resets"] = int(sum(len(r) for r in resets))

        # Reset-aligned memory-accumulation curve: split each env's per-step RMSE
        # at resets, align segments at t=0, NaN-pad to the longest, nanmean.
        segments = []
        for n in range(N):
            bounds = [0] + resets[n] + [T]
            for s, e in zip(bounds[:-1], bounds[1:]):
                if e - s >= 2:
                    segments.append(per_env_step[n, s:e])
        if segments:
            max_len = max(len(s) for s in segments)
            mat = np.full((len(segments), max_len), np.nan)
            for i, s in enumerate(segments):
                mat[i, : len(s)] = s
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                aligned = np.nanmean(mat, axis=0)
            metrics["error_vs_time_since_reset"] = _nan_to_none(aligned)
        else:
            metrics["error_vs_time_since_reset"] = None
    else:
        metrics["reset_steps"] = None
        metrics["num_resets"] = None
        metrics["error_vs_time_since_reset"] = None

    return metrics


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _imshow_panel(ax, arr, title, cmap, vmin, vmax, cbar_label):
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="0.55")  # sentinel / no-data cells render gray
    im = ax.imshow(
        arr, cmap=cmap_obj, vmin=vmin, vmax=vmax, origin="upper", interpolation="nearest"
    )
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    # Robot is at map center; mark it for orientation.
    h, w = arr.shape
    ax.plot((w - 1) / 2.0, (h - 1) / 2.0, marker="+", color="red", markersize=8, mew=1.5)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label, fontsize=8)
    cb.ax.tick_params(labelsize=7)


def _mask_sentinel(frame: np.ndarray) -> np.ma.MaskedArray:
    """Mask no-data sentinel cells so they render gray instead of as terrain."""
    return np.ma.masked_less(frame, SENTINEL_THRESH)


def plot_panels(data: dict, out_dir: Path, timesteps, env: int, label: str):
    """One PNG per timestep: GT | prediction | sparse input | error, fixed scales.

    Sentinel (no-data) cells render gray in the prediction/sparse/error panels.
    """
    gt, pred, sparse = data["gt"], data["pred"], data["sparse"]
    T = gt.shape[1]
    env = min(env, gt.shape[0] - 1)
    written = []
    seen = set()

    for t_req in timesteps:
        t = min(int(t_req), T - 1)
        if t in seen:
            continue
        seen.add(t)

        cols = 4 if sparse is not None else 3
        fig, axes = plt.subplots(1, cols, figsize=(4.0 * cols, 4.2))
        gt_f = gt[env, t]
        pred_f = _mask_sentinel(pred[env, t])
        err_f = np.ma.masked_array(np.abs(pred[env, t] - gt_f), mask=pred_f.mask)

        ci = 0
        _imshow_panel(
            axes[ci], gt_f, "GT height", "viridis", -HEIGHT_CLIP, HEIGHT_CLIP, "height [m]"
        )
        ci += 1
        _imshow_panel(
            axes[ci], pred_f, "Prediction", "viridis", -HEIGHT_CLIP, HEIGHT_CLIP, "height [m]"
        )
        ci += 1
        if sparse is not None:
            _imshow_panel(
                axes[ci],
                _mask_sentinel(sparse[env, t]),
                "Sparse input",
                "viridis",
                -HEIGHT_CLIP,
                HEIGHT_CLIP,
                "height [m]",
            )
            ci += 1
        _imshow_panel(axes[ci], err_f, "Abs error", "hot", 0.0, ERROR_CLIP, "|err| [m]")

        fig.suptitle(f"{label} — env {env}, step {t} (gray = no data)", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out = out_dir / f"panels_t{t:03d}.png"
        fig.savefig(out, dpi=90)
        plt.close(fig)
        written.append(out.name)
    return written


def plot_error_over_time(results, out_dir: Path):
    """Two stacked subplots: raw per-step RMSE with detected resets marked, and the
    reset-aligned memory-accumulation curve. Overlays multiple inputs when provided.
    """
    fig, (ax_raw, ax_aligned) = plt.subplots(2, 1, figsize=(9, 8))

    for i, (label, metrics) in enumerate(results):
        color = f"C{i}"
        curve = np.array(
            [np.nan if v is None else v for v in metrics["error_vs_time"]], dtype=float
        )
        ax_raw.plot(np.arange(len(curve)), curve, label=label, linewidth=1.5, color=color)
        # Mark detected resets (union across envs) in the curve's color.
        reset_steps = metrics.get("reset_steps") or []
        all_resets = sorted({t for env_resets in reset_steps for t in env_resets})
        for j, t in enumerate(all_resets):
            ax_raw.axvline(
                t,
                color=color,
                linestyle="--",
                linewidth=0.8,
                alpha=0.4,
                label=f"{label} resets" if j == 0 else None,
            )

        aligned = metrics.get("error_vs_time_since_reset")
        if aligned is not None:
            aligned = np.array([np.nan if v is None else v for v in aligned], dtype=float)
            ax_aligned.plot(
                np.arange(len(aligned)), aligned, label=label, linewidth=1.5, color=color
            )

    ax_raw.set_xlabel("timestep")
    ax_raw.set_ylabel("RMSE (non-sentinel cells) [m]")
    ax_raw.set_title("Reconstruction error over time (dashed = detected episode resets)")
    ax_raw.grid(True, alpha=0.3)
    ax_raw.legend(fontsize=9)

    ax_aligned.set_xlabel("steps since episode reset")
    ax_aligned.set_ylabel("RMSE (non-sentinel cells) [m]")
    ax_aligned.set_title("Error vs time since reset (memory-accumulation curve)")
    ax_aligned.grid(True, alpha=0.3)
    ax_aligned.legend(fontsize=9)

    fig.tight_layout()
    out = out_dir / "error_over_time.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out.name


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, float) and (v != v):  # NaN
        return "nan"
    return f"{v:.4f}" if isinstance(v, float) else str(v)


SCALAR_KEYS = [
    ("rmse_all", "RMSE all cells [m]"),
    ("mae_all", "MAE all cells [m]"),
    ("coverage", "Coverage (non-sentinel fraction)"),
    ("rmse_visible", "RMSE visible (observed) [m]"),
    ("rmse_occluded", "RMSE occluded [m]"),
    ("edge_mae", "Edge MAE (Sobel) [m]"),
    ("num_resets", "Episode resets detected"),
]


def write_report(out_dir: Path, results, meta: dict, panel_names, err_plot_name, flags=None):
    flags = flags or {}

    def fe(key):
        """Flag emoji cell for a metric key (blank when un-banded / null)."""
        return FLAG_EMOJI.get(flags.get(key), "")

    lines = ["# Perception Evaluation Report", ""]
    lines.append(f"- Generated: {meta['date']}")
    if meta.get("task"):
        lines.append(f"- Task: {meta['task']}")
    if meta.get("checkpoint"):
        lines.append(f"- Checkpoint: {meta['checkpoint']}")
    lines.append(f"- Git commit: {meta.get('git_commit') or 'unknown'}")
    lines.append(f"- Inputs: {meta['num_episodes']} env(s), {meta['num_steps']} steps each")
    lines.append("")

    # Schema notes.
    lines.append("## Schema")
    for label, _m, schema in results:
        notes = "; ".join(schema.get("notes", [])) or "matches current dump format"
        lines.append(
            f"- **{label}** (`{schema['input_h5']}`): layout={schema['layout']}, "
            f"occlusion_mask={schema['has_occlusion_mask']}, "
            f"sparse_height={schema['has_sparse_height']}, "
            f"reset_source={schema.get('reset_source')} — {notes}"
        )
    lines.append("")

    # Metrics table (single or side-by-side).
    lines.append("## Metrics")
    lines.append(
        "Flags vs `docs/evaluation.md`: ✅ good · ⚠️ acceptable · ❌ investigate. "
        "Perception bands are thesis-internal; only single-run-computable checks are "
        "flagged here (model<SLAM comparisons appear only with two inputs).\n"
    )
    if len(results) == 1:
        label, m, _s = results[0]
        lines.append("| metric | value | flag |")
        lines.append("|---|---|---|")
        single_flag = {"rmse_visible": "rmse_visible"}
        for key, name in SCALAR_KEYS:
            lines.append(f"| {name} | {_fmt(m.get(key))} | {fe(single_flag.get(key, ''))} |")
        # Derived single-run check: occluded error should stay within 4x visible.
        rv, ro = m.get("rmse_visible"), m.get("rmse_occluded")
        ratio = f"{ro / rv:.2f}x" if (rv and ro and rv > 0) else "n/a"
        lines.append(
            f"| RMSE occluded / visible (<= 4x) | {ratio} | {fe('rmse_occluded_vs_visible')} |"
        )
        edges = m["radial_bin_edges_m"]
        for i, v in enumerate(m["rmse_radial"]):
            lines.append(f"| RMSE radial [{edges[i]:.2f}-{edges[i+1]:.2f} m] | {_fmt(v)} |  |")
    else:
        labels = [r[0] for r in results]
        cmp_flag_key = {
            "rmse_all": "rmse_all_model_lt_slam",
            "rmse_occluded": "rmse_occluded_model_lt_slam",
            "edge_mae": "edge_mae_model_le_slam",
        }
        lines.append("| metric | " + " | ".join(labels) + " | flag |")
        lines.append("|---|" + "|".join(["---"] * len(labels)) + "|---|")
        for key, name in SCALAR_KEYS:
            row = [_fmt(r[1].get(key)) for r in results]
            lines.append(f"| {name} | " + " | ".join(row) + f" | {fe(cmp_flag_key.get(key, ''))} |")
        edges = results[0][1]["radial_bin_edges_m"]
        n_bins = len(results[0][1]["rmse_radial"])
        for i in range(n_bins):
            row = [_fmt(r[1]["rmse_radial"][i]) for r in results]
            lines.append(
                f"| RMSE radial [{edges[i]:.2f}-{edges[i+1]:.2f} m] | " + " | ".join(row) + " |  |"
            )
        lines.append("")
        lines.append(
            "> Comparison flags read the first input as `model`, the second as "
            "`slam`: ✅ = model beats slam (rmse_all, rmse_occluded, edge_mae)."
        )
    lines.append("")
    lines.append(
        "> Occlusion split uses projector semantics: mask==0 observed (visible), "
        "mask==1 occluded. `rmse_occluded` is the thesis-critical reconstruction "
        "quality where the robot cannot currently see."
    )
    lines.append(
        "> All error metrics exclude no-data sentinel cells (prediction < "
        f"{SENTINEL_THRESH} m). Read RMSE together with `coverage`: a sparse "
        "predictor scores only on the cells it dares to fill."
    )
    reset_src = results[0][2].get("reset_source")
    if reset_src == "dones":
        lines.append("> Episode resets read from the real `dones` dataset.")
    elif reset_src == "robot_pos_jump":
        lines.append(
            "> Episode resets inferred from robot_pos jumps "
            f"(> {RESET_JUMP_M} m/step) — this dump predates the `dones` recorder fix."
        )
    else:
        lines.append("> Episode resets unavailable (no `dones`, no `robot_pos`).")
    lines.append("")

    # Plots.
    lines.append("## Plots")
    lines.append(
        f"- `plots/{err_plot_name}` — raw RMSE over time (resets marked) + "
        "reset-aligned memory curve" + (" (all inputs overlaid)" if len(results) > 1 else "")
    )
    for name in panel_names:
        lines.append(
            f"- `plots/{name}` — GT | prediction | sparse | error panels " "(gray = no data)"
        )
    lines.append("")

    (out_dir / "report.md").write_text("\n".join(lines))


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return ""


def main():
    parser = argparse.ArgumentParser(description="Sim-free perception evaluation analyzer.")
    parser.add_argument("inputs", nargs="+", help="One or two perception data.h5 files.")
    parser.add_argument(
        "-o", "--output", default=None, help="Output dir (default: alongside the first input h5)."
    )
    parser.add_argument(
        "--labels", nargs="+", default=None, help="Labels for the inputs (e.g. model slam)."
    )
    parser.add_argument(
        "--env", type=int, default=0, help="Env index used for the panel plots (default 0)."
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        nargs="+",
        default=DEFAULT_TIMESTEPS,
        help="Timesteps for panel plots (clamped to sequence length).",
    )
    args = parser.parse_args()

    if len(args.inputs) > 2:
        parser.error("Provide at most two h5 inputs (e.g. model vs SLAM).")

    input_paths = [Path(p) for p in args.inputs]
    for p in input_paths:
        if not p.exists():
            parser.error(f"Input not found: {p}")

    # Labels: explicit, else derived from parent directory names.
    if args.labels and len(args.labels) == len(input_paths):
        labels = args.labels
    else:
        labels = [p.parent.name or p.stem for p in input_paths]

    out_dir = Path(args.output) if args.output else input_paths[0].parent
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load + compute per input.
    results = []  # (label, metrics, schema)
    loaded = []  # (label, data)
    for label, path in zip(labels, input_paths):
        data = load_h5(path)
        metrics = compute_metrics(data)
        schema = dict(data["schema"])
        schema["input_h5"] = str(path)
        results.append((label, metrics, schema))
        loaded.append((label, data))
        print(
            f"[{label}] {path}: rmse_all={metrics['rmse_all']:.4f} m "
            f"coverage={metrics['coverage']:.3f} "
            f"visible={_fmt(metrics['rmse_visible'])} "
            f"occluded={_fmt(metrics['rmse_occluded'])} "
            f"resets={_fmt(metrics['num_resets'])}"
        )

    # Panels: from the primary (first) input; error curves overlay all inputs.
    primary_label, primary_data = loaded[0]
    panel_names = plot_panels(primary_data, plots_dir, args.timesteps, args.env, primary_label)
    err_plot_name = plot_error_over_time([(lbl, m) for lbl, m, _s in results], plots_dir)

    # metrics.json (stable meta + perception schema).
    primary_schema = results[0][2]
    # File-level provenance attrs (present in dumps produced after the recorder-gap
    # fix; absent -> None, preserving backward compatibility).
    primary_attrs = loaded[0][1].get("attrs", {})
    checkpoint = primary_attrs.get("perception_checkpoint") or None

    flags = compute_perception_flags(results)

    meta = {
        "task": primary_attrs.get("task"),
        "checkpoint": checkpoint,
        "locomotion_checkpoint": primary_attrs.get("locomotion_checkpoint") or None,
        "method": primary_attrs.get("method"),
        "seed": None,
        "git_commit": primary_attrs.get("git_commit") or _git_commit(),
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_date": primary_attrs.get("date"),
        "num_episodes": primary_schema["num_envs"],
        "num_steps": primary_schema["num_steps"],
        "map_size": primary_schema["map_shape"],
        "benchmark_mode": False,
        "inputs": [str(p) for p in input_paths],
        "labels": labels,
        "sentinel_thresh_m": SENTINEL_THRESH,
        "reset_jump_m": RESET_JUMP_M,
        "schema_notes": {r[0]: r[2] for r in results},
    }
    out = {"meta": meta, "perception": results[0][1], "flags": flags}
    if len(results) > 1:
        out["comparison"] = {
            "labels": labels,
            **{r[0]: r[1] for r in results},
        }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(out, f, indent=2)

    write_report(out_dir, results, meta, panel_names, err_plot_name, flags)

    print(f"Wrote: {out_dir/'metrics.json'}")
    print(f"Wrote: {out_dir/'report.md'}")
    print(f"Wrote: {len(panel_names)} panel plot(s) + {err_plot_name} in {plots_dir}")


if __name__ == "__main__":
    main()
