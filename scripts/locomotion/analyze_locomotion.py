#!/usr/bin/env python3
# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Offline, sim-free analysis of a locomotion rollout.

Consumes ``rollout.h5`` (produced by ``evaluate_locomotion.py --record``) and emits
the evaluation artifact set into the same evaluation directory:

    metrics.json   # stable schema, per-episode metrics -> mean +/- std
    report.md      # skimmable rendering of the metric table
    plots/*.png    # gait_diagram, tracking, attitude, actions

Uses only numpy / h5py / matplotlib -- no isaaclab, no torch -- so it runs
anywhere in seconds and can be re-run on old rollouts without a GPU.

Usage:
    python scripts/locomotion/analyze_locomotion.py \
        --rollout logs/locomotion/LE_rough_sim_.../rollout.h5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import h5py

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


FOOT_LABELS = ["FL", "FR", "RL", "RR"]
PHASE_LABELS = ["FR", "RL", "RR"]  # offsets relative to FL
GRAVITY = 9.81

# Default knee (K) joint angle from base_cfg init_state (T=0.0, H=-0.6, K=1.3 rad).
# Used as the rest reference for the swing-phase knee-excursion ("curling") metric.
DEFAULT_KNEE_ANGLE = 1.3
# Documented DOF-grouped joint order fallback: joint_names group by DOF type not by
# leg (T[0:4], H[4:8], K[8:12], each ordered LF,LR,RF,RR). K joints mapped to feet in
# FL,FR,RL,RR order are therefore LFK(8), RFK(10), LRK(9), RRK(11).
K_JOINT_IDX_FALLBACK = [8, 10, 9, 11]

# Minimum usable segment length (steps) for continuous metrics.
MIN_SEGMENT_LEN = 25

# Autocorrelation gait-cycle detection bounds.
MIN_PERIOD_S = 0.2
MAX_PERIOD_S = 2.0
MIN_AUTOCORR_PEAK = 0.25

# ---------------------------------------------------------------------------
# Benchmark reference ranges
# ---------------------------------------------------------------------------
# Thresholds are copied from docs/evaluation.md -- that markdown file is the
# human-readable source of truth; this dict must be kept in sync with it by hand
# (do NOT parse the markdown at runtime). Each "lt"/"abs_lt" band is (good_cut,
# investigate_cut): value < good_cut -> good, < investigate_cut -> acceptable, else
# investigate. "gt" is (good_cut, acceptable_cut) with larger = better. "range" is
# (lo_good, hi_good, lo_acc, hi_acc). Phase offset and duty-factor arrays use bespoke
# checks below. Flags are advisory ("look at this"), not pass/fail; attitude and
# impact bands assume the flat/benchmark scenario (loosen on random rough terrain).
FLAG_EMOJI = {"good": "✅", "acceptable": "⚠️", "investigate": "❌"}
FLAG_ORDER = {"good": 0, "acceptable": 1, "investigate": 2, None: -1}

# Phase-offset trot targets (FR, RL, RR relative to FL), in cycle fractions.
PHASE_OFFSET_TARGETS = (0.5, 0.5, 0.0)


def _num(x):
    """Coerce to a plain float, or None for None/NaN."""
    if x is None:
        return None
    if isinstance(x, dict):
        x = x.get("mean")
        if x is None:
            return None
    x = float(x)
    return None if np.isnan(x) else x


def _flag_lt(x, good_cut, invest_cut):
    x = _num(x)
    if x is None:
        return None
    if x < good_cut:
        return "good"
    if x < invest_cut:
        return "acceptable"
    return "investigate"


def _flag_abs_lt(x, good_cut, invest_cut):
    x = _num(x)
    return None if x is None else _flag_lt(abs(x), good_cut, invest_cut)


def _flag_gt(x, good_cut, acc_cut):
    x = _num(x)
    if x is None:
        return None
    if x > good_cut:
        return "good"
    if x > acc_cut:
        return "acceptable"
    return "investigate"


def _flag_range(x, lo_g, hi_g, lo_a, hi_a):
    x = _num(x)
    if x is None:
        return None
    if lo_g <= x <= hi_g:
        return "good"
    if lo_a <= x <= hi_a:
        return "acceptable"
    return "investigate"


def _worst(a, b):
    return a if FLAG_ORDER[a] >= FLAG_ORDER[b] else b


def _flag_duty_factor(agg):
    """Worst per-foot flag against the 0.50-0.65 (good) / 0.45-0.75 (acc) band."""
    if agg is None:
        return None
    worst = None
    for mo in agg["mean"]:
        f = _flag_range(mo, 0.50, 0.65, 0.45, 0.75)
        worst = f if worst is None else _worst(worst, f)
    return worst


def _flag_phase_offset(agg):
    """Trot signature check: FR,RL,RR near 0.5,0.5,0.0 (circular), +/-0.10 good / 0.15 acc."""
    if agg is None:
        return None
    worst = None
    for mo, tgt in zip(agg["mean"], PHASE_OFFSET_TARGETS):
        d = abs(float(mo) - tgt)
        d = min(d, 1.0 - d)  # phase is cyclic in [0,1)
        f = "good" if d <= 0.10 else ("acceptable" if d <= 0.15 else "investigate")
        worst = f if worst is None else _worst(worst, f)
    return worst


def compute_flags(meta, loco):
    """Map metric keys -> 'good'|'acceptable'|'investigate' per docs/evaluation.md.

    Only metrics with a band and a non-null value get a flag. Trend metrics
    (smooth.*) and un-banded metrics are omitted.
    """
    task = (meta.get("task") or "").lower()
    is_flat = "flat" in task and "rough" not in task
    flags = {}

    def put(key, val):
        if val is not None:
            flags[key] = val

    sr = loco.get("survival_rate")
    put("survival_rate", _flag_gt(sr, 0.99, 0.95) if is_flat else _flag_gt(sr, 0.95, 0.85))

    tr = loco["tracking"]
    put("tracking.lin_err", _flag_lt(tr["lin_err"], 0.15, 0.30))
    put("tracking.ang_err", _flag_lt(tr["ang_err"], 0.20, 0.40))

    at = loco["attitude"]
    put("attitude.roll_mean", _flag_abs_lt(at["roll_mean"], 0.03, 0.07))
    put("attitude.pitch_mean", _flag_abs_lt(at["pitch_mean"], 0.03, 0.07))
    put("attitude.roll_std", _flag_lt(at["roll_std"], 0.05, 0.10))
    put("attitude.pitch_std", _flag_lt(at["pitch_std"], 0.05, 0.10))

    ga = loco["gait"]
    put("gait.duty_factor", _flag_duty_factor(ga["duty_factor"]))
    put("gait.duty_factor_spread", _flag_lt(ga.get("duty_factor_spread"), 0.05, 0.10))
    put("gait.phase_offset", _flag_phase_offset(ga["phase_offset"]))
    put("gait.stride_freq", _flag_range(ga["stride_freq"], 1.0, 2.5, 0.8, 3.0))

    put("slip.mean_vel", _flag_lt(loco["slip"]["mean_vel"], 0.05, 0.20))

    im = loco["impact"]
    put("impact.peak_force_bw", _flag_lt(im["peak_force_bw"], 2.0, 3.0))
    put("impact.touchdown_vel", _flag_lt(im["touchdown_vel"], 0.3, 0.5))

    ac = loco["actuator"]
    put("actuator.torque_sat_pct", _flag_lt(ac["torque_sat_pct"], 5.0, 20.0))
    put("actuator.vel_sat_pct", _flag_lt(ac["vel_sat_pct"], 1.0, 5.0))

    put("energy.cost_of_transport", _flag_lt(loco["energy"]["cost_of_transport"], 1.0, 2.0))
    return flags


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def quat_to_roll_pitch(quat: np.ndarray):
    """Roll and pitch (rad) from a (N, 4) w,x,y,z quaternion array."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return roll, pitch


def contact_runs(contact_bool: np.ndarray):
    """Return [(start, end_exclusive)] contiguous True runs of a 1-D bool array."""
    c = contact_bool.astype(int)
    diff = np.diff(np.concatenate([[0], c, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return list(zip(starts, ends))


def contact_onsets(contact_bool: np.ndarray):
    """Indices where contact transitions from 0 -> 1."""
    c = contact_bool.astype(int)
    return np.where((c[1:] == 1) & (c[:-1] == 0))[0] + 1


def _decode_str_list(v):
    """Decode an h5 string-attr (bytes / numpy str array) to a plain list of str."""
    if v is None:
        return None
    return [x.decode() if isinstance(x, bytes) else str(x) for x in np.atleast_1d(v)]


def _canonical_foot_from_joint(name: str) -> str:
    """Foot label (FL/FR/RL/RR) for a joint named ``<side><end><type>_joint``.

    e.g. ``LFK_joint`` -> front-left -> ``FL``; ``RRK_joint`` -> rear-right -> ``RR``.
    Mirrors the recorder's ``canonical_foot_label`` so K joints map onto the same
    canonicalized FL,FR,RL,RR foot order used everywhere else in the analyzer.
    """
    base = name.split("_")[0]
    side = base[0].upper()  # L / R
    end = base[1].upper()   # F / R
    fb = "F" if end == "F" else "R"
    return fb + side


def k_joint_indices_by_foot(joint_names):
    """Map the four knee (K) joints onto feet, returning [FL, FR, RL, RR] indices.

    Returns None if the names do not resolve cleanly to four K joints (caller falls
    back to the documented DOF-grouped order).
    """
    if not joint_names:
        return None
    kmap = {}
    for i, n in enumerate(joint_names):
        base = n.split("_")[0]
        if len(base) >= 3 and base[-1].upper() == "K":
            kmap[_canonical_foot_from_joint(n)] = i
    try:
        return [kmap[label] for label in FOOT_LABELS]
    except KeyError:
        return None


def resolve_k_idx(attrs, data=None):
    """Knee joint indices in FL,FR,RL,RR order.

    Newer recordings store them (``knee_joint_idx``, any robot); older Meldog recordings
    fall back to the Meldog joint-name rule, then to the documented DOF-grouped order.
    """
    if data is not None and "knee_joint_idx" in data:
        return [int(i) for i in data["knee_joint_idx"]]
    idx = k_joint_indices_by_foot(_decode_str_list(attrs.get("joint_names")))
    return idx if idx is not None else list(K_JOINT_IDX_FALLBACK)


def resolve_knee_defaults(data, k_idx):
    """Default knee angle per foot (FL,FR,RL,RR); Meldog's 1.3 rad for older recordings."""
    if "default_joint_pos" in data:
        return np.asarray(data["default_joint_pos"], dtype=float)[k_idx]
    return np.full(4, DEFAULT_KNEE_ANGLE)


def detect_period_steps(sig: np.ndarray, dt: float):
    """Fundamental period (in steps) of a binary contact signal via autocorrelation.

    Returns None if no stable cycle is detectable (short signal, no periodicity, or
    peak below threshold) -- never returns a garbage number.
    """
    x = sig.astype(float)
    if len(x) < int(2 * MIN_PERIOD_S / dt) + 2:
        return None
    x = x - x.mean()
    denom = np.sum(x * x)
    if denom <= 1e-9:
        return None
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac = ac / ac[0]
    lo = max(1, int(round(MIN_PERIOD_S / dt)))
    hi = min(len(ac) - 2, int(round(MAX_PERIOD_S / dt)))
    if hi <= lo:
        return None
    # First prominent local maximum (the fundamental) above threshold.
    for i in range(lo, hi + 1):
        if ac[i] > ac[i - 1] and ac[i] >= ac[i + 1] and ac[i] > MIN_AUTOCORR_PEAK:
            return i
    return None


def phase_offset_xcorr(ref: np.ndarray, other: np.ndarray, period_steps: int):
    """Phase of ``other`` relative to ``ref`` in cycle fractions [0, 1)."""
    a = ref.astype(float) - ref.mean()
    b = other.astype(float) - other.mean()
    n = len(a)
    P = int(round(period_steps))
    if P < 2 or n <= P:
        return None
    best_lag, best_val = 0, -np.inf
    for lag in range(0, P):
        v = np.sum(a[: n - lag] * b[lag:]) if lag > 0 else np.sum(a * b)
        if v > best_val:
            best_val, best_lag = v, lag
    return (best_lag % P) / P


def aggregate(values):
    """{'mean','std'} over a list of scalars, skipping None/NaN. None if empty."""
    arr = np.array([v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))],
                   dtype=float)
    if arr.size == 0:
        return None
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr))}


def aggregate_array(values, length):
    """{'mean':[..],'std':[..]} over a list of equal-length arrays (skipping None)."""
    rows = [np.asarray(v, dtype=float) for v in values if v is not None]
    if not rows:
        return None
    mat = np.vstack(rows)
    return {"mean": [float(x) for x in np.nanmean(mat, axis=0)],
            "std": [float(x) for x in np.nanstd(mat, axis=0)]}


# ---------------------------------------------------------------------------
# Rollout loading / segmentation
# ---------------------------------------------------------------------------
def load_rollout(path: Path):
    """Load all datasets + attrs from rollout.h5 into a plain dict."""
    data = {}
    with h5py.File(path, "r") as f:
        for k in f.keys():
            data[k] = f[k][()]
        data["_attrs"] = {k: f.attrs[k] for k in f.attrs}
    return data


def extract_segments(dones, min_len=MIN_SEGMENT_LEN):
    """Contiguous, reset-free segments usable for continuous metrics.

    A ``done`` at step t closes an episode; that single post-reset sample at t is
    excluded (the segment ends at t). The trailing run (no terminating done) is
    kept too. Returns list of dicts with env index and [start, end) slice.
    """
    T, E = dones.shape
    segments = []
    for e in range(E):
        d = dones[:, e]
        done_idx = np.where(d == 1)[0]
        start = 0
        for di in done_idx:
            if di - start >= min_len:
                segments.append({"env": e, "start": int(start), "end": int(di)})
            start = di + 1
        if T - start >= min_len:
            segments.append({"env": e, "start": int(start), "end": int(T)})
    return segments


def survival_stats(dones, time_outs):
    """(#terminated, #survived) counted exactly like evaluate_locomotion.py."""
    done_mask = dones == 1
    n_term = int(done_mask.sum())
    n_surv = int((time_outs[done_mask] == 1).sum())
    return n_term, n_surv


# ---------------------------------------------------------------------------
# Per-episode metrics
# ---------------------------------------------------------------------------
def episode_metrics(data, seg, effort_limit, vel_limit, dt, k_idx):
    """Compute all per-episode metrics for one contiguous segment.

    ``k_idx`` maps feet FL,FR,RL,RR onto knee-joint columns (see resolve_k_idx).
    """
    e, s, end = seg["env"], seg["start"], seg["end"]
    m = {}

    linb = data["root_lin_vel_b"][s:end, e]      # (L,3)
    angb = data["root_ang_vel_b"][s:end, e]      # (L,3)
    cmd = data["commands"][s:end, e]             # (L,3)
    quat = data["root_quat_w"][s:end, e]         # (L,4)
    pos = data["root_pos_w"][s:end, e]           # (L,3)
    jpos = data["joint_pos"][s:end, e]           # (L,12)
    jvel = data["joint_vel"][s:end, e]           # (L,12)
    torque = data["applied_torque"][s:end, e]    # (L,12)
    act = data["actions"][s:end, e]              # (L,12)
    fforce = data["foot_forces_w"][s:end, e]     # (L,4,3)
    fpos = data["foot_pos_w"][s:end, e]          # (L,4,3)
    fvel = data["foot_vel_w"][s:end, e]          # (L,4,3)

    thr = float(data["_attrs"].get("contact_force_threshold", 1.0))
    mass = float(data["robot_mass_per_env"][e]) if "robot_mass_per_env" in data \
        else float(data["_attrs"]["robot_mass"])

    # --- tracking ---
    m["lin_err"] = float(np.mean(np.abs(cmd[:, :2] - linb[:, :2])))
    m["lin_err_vx"] = float(np.mean(np.abs(cmd[:, 0] - linb[:, 0])))
    m["lin_err_vy"] = float(np.mean(np.abs(cmd[:, 1] - linb[:, 1])))
    m["ang_err"] = float(np.mean(np.abs(cmd[:, 2] - angb[:, 2])))

    # --- attitude ---
    roll, pitch = quat_to_roll_pitch(quat)
    m["roll_mean"] = float(np.mean(roll))
    m["roll_std"] = float(np.std(roll))
    m["pitch_mean"] = float(np.mean(pitch))
    m["pitch_std"] = float(np.std(pitch))

    # --- gait ---
    fmag = np.linalg.norm(fforce, axis=-1)       # (L,4)
    contact = fmag > thr                         # (L,4) bool
    m["duty_factor"] = np.mean(contact, axis=0)  # (4,)
    m["duty_factor_spread"] = float(np.max(m["duty_factor"]) - np.min(m["duty_factor"]))

    period = detect_period_steps(contact[:, 0].astype(float), dt)  # FL foot
    if period is not None:
        m["stride_freq"] = 1.0 / (period * dt)
        offs = []
        for fi in (1, 2, 3):  # FR, RL, RR
            po = phase_offset_xcorr(contact[:, 0].astype(float), contact[:, fi].astype(float), period)
            offs.append(po)
        m["phase_offset"] = None if any(o is None for o in offs) else np.array(offs)
    else:
        m["stride_freq"] = None
        m["phase_offset"] = None

    # --- slip (horizontal foot speed while in contact) ---
    # Touchdown/liftoff frames carry contact-transition velocity artifacts that
    # inflate mean slip (D2 calibration: eye saw no slip, metric read high). The
    # banded value ``slip_mean_vel`` excludes the first & last sample of every
    # contact phase; ``slip_mean_vel_raw`` keeps the old all-in-contact value.
    hspeed = np.linalg.norm(fvel[:, :, :2], axis=-1)  # (L,4)
    in_contact = contact
    m["slip_mean_vel_raw"] = float(np.mean(hspeed[in_contact])) if in_contact.any() else None

    interior = np.zeros_like(contact)  # (L,4) bool: contact minus transition frames
    phase_slips, phase_slips_raw = [], []
    for fi in range(4):
        for (rs, re) in contact_runs(contact[:, fi]):
            phase_slips_raw.append(float(np.sum(hspeed[rs:re, fi]) * dt))
            if re - rs > 2:  # need >=1 interior sample after dropping touchdown+liftoff
                interior[rs + 1:re - 1, fi] = True
                phase_slips.append(float(np.sum(hspeed[rs + 1:re - 1, fi]) * dt))
    m["slip_mean_vel"] = float(np.mean(hspeed[interior])) if interior.any() else None
    m["slip_dist_per_step"] = float(np.mean(phase_slips)) if phase_slips else None
    m["slip_dist_per_step_raw"] = float(np.mean(phase_slips_raw)) if phase_slips_raw else None

    # --- impact (per touchdown) ---
    # Peak force prefers the per-step substep maximum (feet_forces_max, magnitude
    # over net_forces_w_history) when the recorder provides it; otherwise it falls
    # back to the policy-rate snapshot magnitude (under-samples true transients).
    fmag_peak = data["feet_forces_max"][s:end, e] if "feet_forces_max" in data else fmag
    peak_bw, td_vel = [], []
    for fi in range(4):
        runs = contact_runs(contact[:, fi])
        onsets = set(contact_onsets(contact[:, fi]).tolist())
        for (rs, re) in runs:
            if rs in onsets:  # genuine touchdown (not contact at t=0)
                peak_bw.append(float(np.max(fmag_peak[rs:re, fi]) / (mass * GRAVITY)))
                td_vel.append(float(abs(fvel[rs, fi, 2])))
    m["peak_force_bw"] = float(np.mean(peak_bw)) if peak_bw else None
    m["touchdown_vel"] = float(np.mean(td_vel)) if td_vel else None

    # --- swing-phase kinematics (trend metrics, per foot FL,FR,RL,RR) ---
    # apex_height: per swing (non-contact phase) max foot-z rise above liftoff z.
    # knee_excursion: per swing max |K-joint angle - default|, a "limb curling"
    # detector. Both averaged over the episode's genuine swings (must start from a
    # real liftoff, i.e. preceded by contact -> swing run start > 0).
    foot_z = fpos[:, :, 2]  # (L,4) world z
    knee_default = resolve_knee_defaults(data, k_idx)
    swing_apex = np.full(4, np.nan)
    swing_knee = np.full(4, np.nan)
    for fi in range(4):
        apex_vals, knee_vals = [], []
        for (rs, re) in contact_runs(~contact[:, fi]):
            if rs == 0 or re - rs < 2:  # skip pre-existing air phase / too-short swings
                continue
            apex_vals.append(float(np.max(foot_z[rs:re, fi]) - foot_z[rs, fi]))
            kj = jpos[rs:re, k_idx[fi]]
            knee_vals.append(float(np.max(np.abs(kj - knee_default[fi]))))
        if apex_vals:
            swing_apex[fi] = float(np.mean(apex_vals))
        if knee_vals:
            swing_knee[fi] = float(np.mean(knee_vals))
    m["swing_apex"] = swing_apex
    m["swing_knee"] = swing_knee

    # --- posture: trunk height above terrain + attitude vs the terrain plane ---
    # Optional Run D channel. Rollouts recorded before it exists simply leave these
    # None, and every posture.* metric then aggregates to null.
    # ``terrain_normal_b`` is the local terrain-plane normal in body coordinates; the
    # tilt of the body relative to that plane decomposes exactly like world roll/pitch
    # does against (0,0,1), so on flat ground these reproduce attitude.roll/pitch.
    m["base_height"] = m["base_height_std"] = None
    m["pitch_terrain_rel_mean"] = m["pitch_terrain_rel_std"] = None
    m["roll_terrain_rel_mean"] = m["roll_terrain_rel_std"] = None
    if "base_height" in data:
        bh = np.asarray(data["base_height"][s:end, e], dtype=float)
        bh = bh[np.isfinite(bh)]
        if bh.size:
            m["base_height"] = float(np.mean(bh))
            m["base_height_std"] = float(np.std(bh))
    if "terrain_normal_b" in data:
        nb = np.asarray(data["terrain_normal_b"][s:end, e], dtype=float)  # (L,3)
        nb = nb[np.isfinite(nb).all(axis=1)]
        if nb.size:
            pitch_rel = np.arcsin(np.clip(-nb[:, 0], -1.0, 1.0))
            roll_rel = np.arctan2(nb[:, 1], nb[:, 2])
            m["pitch_terrain_rel_mean"] = float(np.mean(pitch_rel))
            m["pitch_terrain_rel_std"] = float(np.std(pitch_rel))
            m["roll_terrain_rel_mean"] = float(np.mean(roll_rel))
            m["roll_terrain_rel_std"] = float(np.std(roll_rel))

    # --- smoothness ---
    if len(act) > 1:
        m["action_rate"] = float(np.mean(np.abs(np.diff(act, axis=0))))
        m["joint_acc"] = float(np.mean(np.abs(np.diff(jvel, axis=0) / dt)))
    else:
        m["action_rate"] = None
        m["joint_acc"] = None

    # --- actuator saturation ---
    m["torque_sat_pct"] = float(100.0 * np.mean(np.abs(torque) > 0.9 * effort_limit))
    m["vel_sat_pct"] = float(100.0 * np.mean(np.abs(jvel) > 0.9 * vel_limit))

    # --- energy / cost of transport ---
    power = np.sum(np.abs(torque * jvel), axis=1)       # (L,)
    energy = float(np.sum(power) * dt)
    distance = float(np.sum(np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1)))
    m["cost_of_transport"] = energy / (mass * GRAVITY * distance) if distance > 0.05 else None

    return m


# ---------------------------------------------------------------------------
# Metric aggregation into the metrics.json schema
# ---------------------------------------------------------------------------
def build_metrics(data, segments):
    dt = float(data["_attrs"]["step_dt"])
    # Per-joint limits when recorded (inf = no fixed limit, never counted as saturated).
    effort_limit = (np.asarray(data["joint_effort_limits"], dtype=float) if "joint_effort_limits" in data
                    else float(data["_attrs"]["joint_effort_limit"]))
    vel_limit = (np.asarray(data["joint_velocity_limits"], dtype=float) if "joint_velocity_limits" in data
                 else float(data["_attrs"]["joint_velocity_limit"]))

    k_idx = resolve_k_idx(data["_attrs"], data)

    n_term, n_surv = survival_stats(data["dones"], data["time_outs"])
    per_ep = [episode_metrics(data, seg, effort_limit, vel_limit, dt, k_idx) for seg in segments]

    def col(key):
        return [ep[key] for ep in per_ep]

    n_cycle = sum(1 for ep in per_ep if ep["stride_freq"] is not None)

    loco = {
        "survival_rate": (n_surv / n_term) if n_term > 0 else None,
        "n_terminated": n_term,
        "n_survived": n_surv,
        "num_usable_episodes": len(per_ep),
        "tracking": {
            "lin_err": aggregate(col("lin_err")),
            "lin_err_vx": aggregate(col("lin_err_vx")),
            "lin_err_vy": aggregate(col("lin_err_vy")),
            "ang_err": aggregate(col("ang_err")),
        },
        "attitude": {
            "roll_mean": aggregate(col("roll_mean")),
            "roll_std": aggregate(col("roll_std")),
            "pitch_mean": aggregate(col("pitch_mean")),
            "pitch_std": aggregate(col("pitch_std")),
        },
        "gait": {
            "foot_order": "FL,FR,RL,RR",
            "duty_factor": aggregate_array(col("duty_factor"), 4),
            "duty_factor_spread": aggregate(col("duty_factor_spread")),
            "phase_offset_labels": PHASE_LABELS,
            "phase_offset": aggregate_array(col("phase_offset"), 3),
            "stride_freq": aggregate(col("stride_freq")),
            "cycle_detected_frac": (n_cycle / len(per_ep)) if per_ep else None,
        },
        "slip": {
            "mean_vel": aggregate(col("slip_mean_vel")),
            "mean_vel_raw": aggregate(col("slip_mean_vel_raw")),
            "dist_per_step": aggregate(col("slip_dist_per_step")),
            "dist_per_step_raw": aggregate(col("slip_dist_per_step_raw")),
        },
        "impact": {
            "peak_force_bw": aggregate(col("peak_force_bw")),
            "touchdown_vel": aggregate(col("touchdown_vel")),
        },
        "swing": {
            "foot_order": "FL,FR,RL,RR",
            "apex_height": aggregate_array(col("swing_apex"), 4),
            "knee_excursion": aggregate_array(col("swing_knee"), 4),
        },
        # Trend metrics (no bands). Null on rollouts without the Run D datasets.
        "posture": {
            "base_height": aggregate(col("base_height")),
            "base_height_std": aggregate(col("base_height_std")),
            "pitch_terrain_rel_mean": aggregate(col("pitch_terrain_rel_mean")),
            "pitch_terrain_rel_std": aggregate(col("pitch_terrain_rel_std")),
            "roll_terrain_rel_mean": aggregate(col("roll_terrain_rel_mean")),
            "roll_terrain_rel_std": aggregate(col("roll_terrain_rel_std")),
        },
        "smooth": {
            "action_rate": aggregate(col("action_rate")),
            "joint_acc": aggregate(col("joint_acc")),
        },
        "actuator": {
            "torque_sat_pct": aggregate(col("torque_sat_pct")),
            "vel_sat_pct": aggregate(col("vel_sat_pct")),
        },
        "energy": {
            "cost_of_transport": aggregate(col("cost_of_transport")),
        },
    }

    attrs = data["_attrs"]

    def a(key, default=None):
        v = attrs.get(key, default)
        if isinstance(v, bytes):
            return v.decode()
        if isinstance(v, np.generic):
            return v.item()
        return v

    meta = {
        "task": a("task"),
        "checkpoint": a("checkpoint"),
        "seed": a("seed"),
        "git_commit": a("git_commit"),
        "date": a("date"),
        "num_episodes": a("num_episodes"),
        "benchmark_mode": bool(a("benchmark_mode", False)),
        "num_envs": int(a("num_envs", data["dones"].shape[1])),
        "num_steps": int(a("num_steps", data["dones"].shape[0])),
        "robot_mass": a("robot_mass"),
        "robot_name": a("robot_name", "meldog"),
        "robot_leg_length": a("leg_length"),
        "impact_force_source": "substep_max" if "feet_forces_max" in data else "snapshot",
    }
    flags = compute_flags(meta, loco)
    return {"meta": meta, "locomotion": loco, "flags": flags}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _pick_segment(segments):
    """Longest usable segment -- the representative episode for trace plots."""
    return max(segments, key=lambda s: s["end"] - s["start"])


def plot_gait_diagram(data, seg, dt, out_path, window_s=10.0):
    e, s, end = seg["env"], seg["start"], seg["end"]
    thr = float(data["_attrs"].get("contact_force_threshold", 1.0))
    n = min(end - s, int(round(window_s / dt)))
    fforce = data["foot_forces_w"][s:s + n, e]
    contact = np.linalg.norm(fforce, axis=-1) > thr  # (n,4)

    fig, ax = plt.subplots(figsize=(11, 3.2))
    colors = ["#1b7837", "#762a83", "#2166ac", "#b2182b"]
    for i, label in enumerate(FOOT_LABELS):
        for (rs, re) in contact_runs(contact[:, i]):
            ax.broken_barh([(rs * dt, (re - rs) * dt)], (i - 0.4, 0.8),
                           facecolors=colors[i], edgecolor="none")
    ax.set_yticks(range(4))
    ax.set_yticklabels(FOOT_LABELS)
    ax.set_ylim(-0.6, 3.6)
    ax.invert_yaxis()
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Foot")
    ax.set_xlim(0, n * dt)
    ax.set_title(f"Gait diagram  (env {e}, contact = |F| > {thr:g} N)")
    ax.legend(handles=[Patch(facecolor="0.3", label="stance (in contact)")],
              loc="upper right", fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_tracking(data, seg, dt, out_path):
    e, s, end = seg["env"], seg["start"], seg["end"]
    t = np.arange(end - s) * dt
    linb = data["root_lin_vel_b"][s:end, e]
    angb = data["root_ang_vel_b"][s:end, e]
    cmd = data["commands"][s:end, e]

    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    series = [
        ("v_x (m/s)", cmd[:, 0], linb[:, 0]),
        ("v_y (m/s)", cmd[:, 1], linb[:, 1]),
        ("w_z (rad/s)", cmd[:, 2], angb[:, 2]),
    ]
    for ax, (ylabel, c, actual) in zip(axes, series):
        ax.plot(t, c, color="#b2182b", lw=1.6, ls="--", label="command")
        ax.plot(t, actual, color="#2166ac", lw=1.2, label="actual")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title(f"Command tracking  (env {e})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_attitude(data, seg, dt, out_path):
    e, s, end = seg["env"], seg["start"], seg["end"]
    t = np.arange(end - s) * dt
    roll, pitch = quat_to_roll_pitch(data["root_quat_w"][s:end, e])

    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(t, roll, color="#762a83", lw=1.2, label="roll")
    ax.plot(t, pitch, color="#1b7837", lw=1.2, label="pitch")
    ax.axhline(np.mean(roll), color="#762a83", ls=":", lw=1.0,
               label=f"roll mean {np.mean(roll):+.3f}")
    ax.axhline(np.mean(pitch), color="#1b7837", ls=":", lw=1.0,
               label=f"pitch mean {np.mean(pitch):+.3f}")
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (rad)")
    ax.set_title(f"Body attitude  (env {e}; nonzero mean = persistent lean)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_actions(data, seg, dt, out_path):
    e, s, end = seg["env"], seg["start"], seg["end"]
    t = np.arange(end - s) * dt
    act = data["actions"][s:end, e]  # (L,12)
    joint_names = data["_attrs"].get("joint_names")
    if joint_names is not None:
        joint_names = [j.decode() if isinstance(j, bytes) else str(j) for j in joint_names]
    else:
        joint_names = [f"joint_{i}" for i in range(act.shape[1])]

    # Select the 3 joints of a single leg by name prefix (Isaac Lab groups joints
    # by DOF type, not by leg, so consecutive indices are different legs).
    leg_prefixes = ["LF", "RF", "LR", "RR"]
    leg_idx = []
    for pfx in leg_prefixes:
        leg_idx = [i for i, n in enumerate(joint_names) if n.upper().startswith(pfx)]
        if len(leg_idx) >= 2:
            break
    if len(leg_idx) < 2:
        leg_idx = list(range(min(3, act.shape[1])))
    leg_tag = joint_names[leg_idx[0]][:2] if joint_names else ""

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = ["#1b7837", "#762a83", "#2166ac", "#b2182b"]
    for c, j in enumerate(leg_idx):
        axes[0].plot(t, act[:, j], color=colors[c % len(colors)], lw=1.1, label=joint_names[j])
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Action (normalized joint target)")
    axes[0].set_title(f"Action traces, one leg [{leg_tag}]  (env {e})")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)

    rate = np.abs(np.diff(act, axis=0)).ravel()  # all joints
    axes[1].hist(rate, bins=50, color="#2166ac", alpha=0.85)
    axes[1].set_xlabel("Action rate  |a_t - a_{t-1}|")
    axes[1].set_ylabel("Count (all joints x steps)")
    axes[1].set_title("Action-rate distribution")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_swing(data, seg, dt, out_path, k_idx):
    """Per-foot swing apex height and knee excursion (trend metrics) for one segment."""
    e, s, end = seg["env"], seg["start"], seg["end"]
    thr = float(data["_attrs"].get("contact_force_threshold", 1.0))
    contact = np.linalg.norm(data["foot_forces_w"][s:end, e], axis=-1) > thr  # (L,4)
    foot_z = data["foot_pos_w"][s:end, e, :, 2]  # (L,4)
    jpos = data["joint_pos"][s:end, e]           # (L,12)

    apex_means, knee_means = [], []
    for fi in range(4):
        apex_vals, knee_vals = [], []
        for (rs, re) in contact_runs(~contact[:, fi]):
            if rs == 0 or re - rs < 2:
                continue
            apex_vals.append(float(np.max(foot_z[rs:re, fi]) - foot_z[rs, fi]))
            knee_vals.append(float(np.max(np.abs(jpos[rs:re, k_idx[fi]] - DEFAULT_KNEE_ANGLE))))
        apex_means.append(float(np.mean(apex_vals)) if apex_vals else 0.0)
        knee_means.append(float(np.mean(knee_vals)) if knee_vals else 0.0)

    colors = ["#1b7837", "#762a83", "#2166ac", "#b2182b"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].bar(FOOT_LABELS, apex_means, color=colors)
    axes[0].set_ylabel("Swing apex height (m)")
    axes[0].set_title("Apex: max foot-z rise above liftoff")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar(FOOT_LABELS, knee_means, color=colors)
    axes[1].set_ylabel("max |K - default| (rad)")
    axes[1].set_title(f"Knee excursion (curling; K default {DEFAULT_KNEE_ANGLE:g})")
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.suptitle(f"Swing kinematics  (env {e}, trend metrics)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def has_posture(data):
    """True when the rollout carries the Run D terrain-relative posture datasets."""
    return "base_height" in data and "terrain_normal_b" in data


def plot_posture(data, seg, dt, out_path):
    """Trunk height above terrain and body attitude relative to the terrain plane."""
    e, s, end = seg["env"], seg["start"], seg["end"]
    t = np.arange(end - s) * dt
    bh = np.asarray(data["base_height"][s:end, e], dtype=float)
    nb = np.asarray(data["terrain_normal_b"][s:end, e], dtype=float)
    pitch_rel = np.arcsin(np.clip(-nb[:, 0], -1.0, 1.0))
    roll_rel = np.arctan2(nb[:, 1], nb[:, 2])

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(t, bh, color="#2166ac", lw=1.2, label="trunk height above terrain")
    axes[0].axhline(np.mean(bh), color="#2166ac", ls=":", lw=1.0,
                    label=f"mean {np.mean(bh):.3f} m")
    axes[0].set_ylabel("Height (m)")
    axes[0].set_title(f"Posture  (env {e}; terrain-relative, trend metrics)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].plot(t, roll_rel, color="#762a83", lw=1.2, label="roll vs terrain")
    axes[1].plot(t, pitch_rel, color="#1b7837", lw=1.2, label="pitch vs terrain")
    axes[1].axhline(np.mean(roll_rel), color="#762a83", ls=":", lw=1.0,
                    label=f"roll mean {np.mean(roll_rel):+.3f}")
    axes[1].axhline(np.mean(pitch_rel), color="#1b7837", ls=":", lw=1.0,
                    label=f"pitch mean {np.mean(pitch_rel):+.3f}")
    axes[1].axhline(0.0, color="0.6", lw=0.8)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Angle (rad)")
    axes[1].set_title("Attitude relative to the fitted terrain plane "
                      "(0 = trunk parallel to local ground)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# report.md
# ---------------------------------------------------------------------------
def fmt(agg, scale=1.0, unit=""):
    if agg is None:
        return "n/a"
    if isinstance(agg, dict) and "mean" in agg and np.isscalar(agg["mean"]):
        return f"{agg['mean'] * scale:.4g} +/- {agg['std'] * scale:.3g}{unit}"
    return str(agg)


def fmt_array(agg, unit=""):
    if agg is None:
        return "n/a"
    means = agg["mean"]
    stds = agg["std"]
    return ", ".join(f"{mo:.3g}+/-{so:.2g}" for mo, so in zip(means, stds)) + unit


def write_report(metrics, out_path):
    meta = metrics["meta"]
    L = metrics["locomotion"]
    flags = metrics.get("flags", {})

    def fl(key):
        """Flag emoji cell for a metric key (blank when un-banded / null)."""
        return FLAG_EMOJI.get(flags.get(key), "")

    lines = []
    lines.append("# Locomotion evaluation report\n")
    lines.append(f"- **task**: {meta['task']}")
    lines.append(f"- **checkpoint**: {meta['checkpoint']}")
    lines.append(f"- **date**: {meta['date']}  ·  **git**: {meta['git_commit']}  ·  "
                 f"**seed**: {meta['seed']}  ·  **benchmark_mode**: {meta['benchmark_mode']}")
    lines.append(f"- **envs x steps**: {meta['num_envs']} x {meta['num_steps']}  ·  "
                 f"**usable episodes**: {L['num_usable_episodes']}  ·  "
                 f"**robot mass**: {meta['robot_mass']:.2f} kg"
                 + (f"  ·  **robot**: {meta['robot_name']}, leg length {meta['robot_leg_length']:.3f} m"
                    if meta.get("robot_leg_length") else "") + "\n")

    sr = L["survival_rate"]
    lines.append("## Survival")
    lines.append(
        f"- survival_rate: {sr*100:.1f}%  ({L['n_survived']}/{L['n_terminated']} "
        f"episodes) {fl('survival_rate')}\n"
        if sr is not None else "- survival_rate: n/a\n")

    lines.append("## Metrics (mean +/- std across episodes)\n")
    lines.append("Flags vs `docs/evaluation.md`: ✅ good · ⚠️ acceptable · ❌ investigate "
                 "(advisory — attitude/impact bands assume the flat/benchmark scenario). "
                 "Trend metrics (smooth.*, swing.*, posture.*) and un-banded rows (marked "
                 "`trend`) carry no flag.\n")
    lines.append("| metric | value | flag |")
    lines.append("|---|---|---|")
    lines.append(f"| tracking.lin_err (vx,vy) | {fmt(L['tracking']['lin_err'], unit=' m/s')} | {fl('tracking.lin_err')} |")
    lines.append(f"| tracking.lin_err_vx | {fmt(L['tracking']['lin_err_vx'], unit=' m/s')} |  |")
    lines.append(f"| tracking.lin_err_vy | {fmt(L['tracking']['lin_err_vy'], unit=' m/s')} |  |")
    lines.append(f"| tracking.ang_err (wz) | {fmt(L['tracking']['ang_err'], unit=' rad/s')} | {fl('tracking.ang_err')} |")
    lines.append(f"| attitude.roll_mean | {fmt(L['attitude']['roll_mean'], unit=' rad')} | {fl('attitude.roll_mean')} |")
    lines.append(f"| attitude.roll_std | {fmt(L['attitude']['roll_std'], unit=' rad')} | {fl('attitude.roll_std')} |")
    lines.append(f"| attitude.pitch_mean | {fmt(L['attitude']['pitch_mean'], unit=' rad')} | {fl('attitude.pitch_mean')} |")
    lines.append(f"| attitude.pitch_std | {fmt(L['attitude']['pitch_std'], unit=' rad')} | {fl('attitude.pitch_std')} |")
    lines.append(f"| gait.duty_factor [FL,FR,RL,RR] | {fmt_array(L['gait']['duty_factor'])} | {fl('gait.duty_factor')} |")
    lines.append(f"| gait.duty_factor_spread | {fmt(L['gait']['duty_factor_spread'])} | {fl('gait.duty_factor_spread')} |")
    lines.append(f"| gait.phase_offset [FR,RL,RR] | {fmt_array(L['gait']['phase_offset'], unit=' cyc')} | {fl('gait.phase_offset')} |")
    lines.append(f"| gait.stride_freq | {fmt(L['gait']['stride_freq'], unit=' Hz')} | {fl('gait.stride_freq')} |")
    lines.append(f"| gait.cycle_detected_frac | {L['gait']['cycle_detected_frac']} |  |")
    lines.append(f"| slip.mean_vel (transition-filtered) | {fmt(L['slip']['mean_vel'], unit=' m/s')} | {fl('slip.mean_vel')} |")
    lines.append(f"| slip.mean_vel_raw (unfiltered) | {fmt(L['slip']['mean_vel_raw'], unit=' m/s')} | trend |")
    lines.append(f"| slip.dist_per_step | {fmt(L['slip']['dist_per_step'], unit=' m')} |  |")
    lines.append(f"| impact.peak_force_bw ({meta.get('impact_force_source', 'snapshot')}) | {fmt(L['impact']['peak_force_bw'], unit=' BW')} | {fl('impact.peak_force_bw')} |")
    lines.append(f"| impact.touchdown_vel | {fmt(L['impact']['touchdown_vel'], unit=' m/s')} | {fl('impact.touchdown_vel')} |")
    lines.append(f"| swing.apex_height [FL,FR,RL,RR] | {fmt_array(L['swing']['apex_height'], unit=' m')} | trend |")
    lines.append(f"| swing.knee_excursion [FL,FR,RL,RR] | {fmt_array(L['swing']['knee_excursion'], unit=' rad')} | trend |")
    P = L.get("posture") or {}
    if P.get("base_height") is not None or P.get("pitch_terrain_rel_mean") is not None:
        lines.append(f"| posture.base_height (trunk above terrain) | {fmt(P['base_height'], unit=' m')} | trend |")
        lines.append(f"| posture.base_height_std (within episode) | {fmt(P['base_height_std'], unit=' m')} | trend |")
        lines.append(f"| posture.pitch_terrain_rel_mean | {fmt(P['pitch_terrain_rel_mean'], unit=' rad')} | trend |")
        lines.append(f"| posture.pitch_terrain_rel_std | {fmt(P['pitch_terrain_rel_std'], unit=' rad')} | trend |")
        lines.append(f"| posture.roll_terrain_rel_mean | {fmt(P['roll_terrain_rel_mean'], unit=' rad')} | trend |")
        lines.append(f"| posture.roll_terrain_rel_std | {fmt(P['roll_terrain_rel_std'], unit=' rad')} | trend |")
    lines.append(f"| smooth.action_rate | {fmt(L['smooth']['action_rate'])} |  |")
    lines.append(f"| smooth.joint_acc | {fmt(L['smooth']['joint_acc'], unit=' rad/s^2')} |  |")
    lines.append(f"| actuator.torque_sat_pct | {fmt(L['actuator']['torque_sat_pct'], unit=' %')} | {fl('actuator.torque_sat_pct')} |")
    lines.append(f"| actuator.vel_sat_pct | {fmt(L['actuator']['vel_sat_pct'], unit=' %')} | {fl('actuator.vel_sat_pct')} |")
    lines.append(f"| energy.cost_of_transport | {fmt(L['energy']['cost_of_transport'])} | {fl('energy.cost_of_transport')} |")
    lines.append("\n## Plots\n")
    plots = ["gait_diagram", "tracking", "attitude", "actions", "swing"]
    if P.get("base_height") is not None:
        plots.append("posture")
    for p in plots:
        lines.append(f"![{p}](plots/{p}.png)\n")

    out_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# JSON sanitize
# ---------------------------------------------------------------------------
def sanitize(obj):
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return [sanitize(v) for v in obj.tolist()]
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def main():
    parser = argparse.ArgumentParser(description="Analyze a locomotion rollout.h5 (sim-free).")
    parser.add_argument("--rollout", type=str, required=True, help="Path to rollout.h5.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output dir (default: the rollout's directory).")
    args = parser.parse_args()

    rollout_path = Path(args.rollout)
    if not rollout_path.exists():
        raise FileNotFoundError(rollout_path)
    out_dir = Path(args.output) if args.output else rollout_path.parent
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading {rollout_path}")
    data = load_rollout(rollout_path)
    dt = float(data["_attrs"]["step_dt"])
    T, E = data["dones"].shape
    print(f"[INFO] {T} steps x {E} envs, dt={dt:g}s")

    segments = extract_segments(data["dones"])
    print(f"[INFO] {len(segments)} usable segments (>= {MIN_SEGMENT_LEN} steps)")
    if not segments:
        raise RuntimeError("No usable segments found -- rollout too short.")

    metrics = build_metrics(data, segments)

    (out_dir / "metrics.json").write_text(json.dumps(sanitize(metrics), indent=2))
    print(f"[INFO] Wrote {out_dir / 'metrics.json'}")

    seg = _pick_segment(segments)
    print(f"[INFO] Representative segment: env {seg['env']}, "
          f"{seg['end'] - seg['start']} steps ({(seg['end']-seg['start'])*dt:.1f}s)")
    plot_gait_diagram(data, seg, dt, plots_dir / "gait_diagram.png")
    plot_tracking(data, seg, dt, plots_dir / "tracking.png")
    plot_attitude(data, seg, dt, plots_dir / "attitude.png")
    plot_actions(data, seg, dt, plots_dir / "actions.png")
    plot_swing(data, seg, dt, plots_dir / "swing.png", resolve_k_idx(data["_attrs"], data))
    if has_posture(data):
        plot_posture(data, seg, dt, plots_dir / "posture.png")
    else:
        print("[INFO] No terrain-relative posture datasets in this rollout "
              "(no height scanner, or a pre-Run-D recording); skipping posture metrics and plot.")
    print(f"[INFO] Wrote plots to {plots_dir}")

    write_report(metrics, out_dir / "report.md")
    print(f"[INFO] Wrote {out_dir / 'report.md'}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
