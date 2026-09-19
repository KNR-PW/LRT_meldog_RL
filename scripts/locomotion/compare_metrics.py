#!/usr/bin/env python3
# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Compare locomotion evaluations side by side.

Reads two or more ``metrics.json`` files written by ``analyze_locomotion.py`` (or the
evaluation folders containing them) and prints one Markdown table with the robot, its
size and the key metrics, each with its ✅/⚠️/❌ flag. Rows are sorted by robot mass so
results of similarly sized robots sit next to each other.

Uses only the standard library, so it runs anywhere.

Usage:
    python scripts/locomotion/compare_metrics.py logs/locomotion/LE_a logs/locomotion/LE_b \
        [--output comparison.md] [--csv comparison.csv] [--bands meldog]
"""

import argparse
import csv
import json
from pathlib import Path

FLAG_EMOJI = {"good": "✅", "acceptable": "⚠️", "investigate": "❌"}

# (column title, path into the "locomotion" block, flag key, number format)
COLUMNS = [
    ("survival", ("survival_rate",), "survival_rate", "{:.2f}"),
    ("lin_err m/s", ("tracking", "lin_err", "mean"), "tracking.lin_err", "{:.3f}"),
    ("ang_err rad/s", ("tracking", "ang_err", "mean"), "tracking.ang_err", "{:.3f}"),
    ("duty factor", ("gait", "duty_factor", "mean"), "gait.duty_factor", "range"),
    ("phase FR,RL,RR", ("gait", "phase_offset", "mean"), "gait.phase_offset", "list"),
    ("stride Hz", ("gait", "stride_freq", "mean"), "gait.stride_freq", "{:.2f}"),
    ("slip m/s", ("slip", "mean_vel", "mean"), "slip.mean_vel", "{:.3f}"),
    (
        "impact p95 BW",
        ("impact", "peak_force_bw_p95", "mean"),
        "impact.peak_force_bw_p95",
        "{:.2f}",
    ),
    ("touchdown m/s", ("impact", "touchdown_vel", "mean"), "impact.touchdown_vel", "{:.2f}"),
    (
        "pitch rel. terrain std rad",
        ("posture", "pitch_terrain_rel_std", "mean"),
        "posture.pitch_terrain_rel_std",
        "{:.3f}",
    ),
    ("trunk height m", ("posture", "base_height", "mean"), None, "{:.3f}"),
    ("torque sat %", ("actuator", "torque_sat_pct", "mean"), "actuator.torque_sat_pct", "{:.1f}"),
    ("vel sat %", ("actuator", "vel_sat_pct", "mean"), "actuator.vel_sat_pct", "{:.1f}"),
    ("CoT", ("energy", "cost_of_transport", "mean"), "energy.cost_of_transport", "{:.2f}"),
]


def load(path: Path) -> dict:
    """Load metrics.json from a file path or an evaluation folder."""
    file = path / "metrics.json" if path.is_dir() else path
    with open(file) as f:
        metrics = json.load(f)
    metrics["_path"] = file.parent
    return metrics


def dig(tree, keys):
    for key in keys:
        if not isinstance(tree, dict):
            return None
        tree = tree.get(key)
    return tree


def fmt(value, style: str) -> str:
    if value is None:
        return "–"
    if style == "range":
        return f"{min(value):.2f}-{max(value):.2f}"
    if style == "list":
        return ",".join(f"{v:.2f}" for v in value)
    return style.format(value)


def rows_for(runs):
    header = ["run", "robot", "mass kg", "leg m", "terrain", "profile"] + [c[0] for c in COLUMNS]
    rows = []
    for m in sorted(runs, key=lambda r: r["meta"].get("robot_mass") or 0.0):
        meta, loco, flags = m["meta"], m["locomotion"], m.get("flags", {})
        leg = meta.get("robot_leg_length")
        task = meta.get("task") or ""
        terrain = meta.get("bench_terrain") or "task"
        if terrain == "task":
            terrain = (
                "flat (task)"
                if "flat" in task.lower()
                else "rough (task)" if "rough" in task.lower() else "task"
            )
        row = [
            m["_path"].name,
            meta.get("robot_name") or "meldog",
            f"{meta['robot_mass']:.1f}" if meta.get("robot_mass") else "–",
            f"{leg:.3f}" if leg else "–",
            terrain,
            meta.get("profile") or "task",
        ]
        for _, keys, flag_key, style in COLUMNS:
            text = fmt(dig(loco, keys), style)
            flag = FLAG_EMOJI.get(flags.get(flag_key)) if flag_key else None
            row.append(f"{text} {flag}" if flag else text)
        rows.append(row)
    return header, rows


def _scalar_values(value):
    """Numbers inside a metric value (a scalar, or a per-foot / per-offset list)."""
    if value is None:
        return []
    return [float(v) for v in value] if isinstance(value, list) else [float(value)]


def reference_bands(runs, focus_robot: str):
    """Per (terrain, profile): min-max of every column over the non-focus robots, and the focus
    robot's values outside that band. Returns Markdown text."""
    groups = {}
    for m in runs:
        key = (m["meta"].get("bench_terrain") or "task", m["meta"].get("profile") or "task")
        groups.setdefault(key, []).append(m)
    out = []
    for (terrain, profile), members in sorted(groups.items()):
        refs = [m for m in members if (m["meta"].get("robot_name") or "meldog") != focus_robot]
        focus = [m for m in members if (m["meta"].get("robot_name") or "meldog") == focus_robot]
        if not refs:
            continue
        names = ", ".join(sorted({m["meta"].get("robot_name") for m in refs}))
        out.append(
            f"\n### Reference band: terrain `{terrain}`, profile `{profile}` ({len(refs)} runs: {names})\n"
        )
        out.append(
            "Bold values lie outside the reference min-max: ↓ below, ↑ above. Whether that is "
            "better or worse depends on the metric.\n"
        )
        header = ["metric", "reference min", "reference max"] + [m["_path"].name for m in focus]
        out.append("| " + " | ".join(header) + " |")
        out.append("|" + "---|" * len(header))
        for title, keys, _, _ in COLUMNS:
            ref_lists = [_scalar_values(dig(m["locomotion"], keys)) for m in refs]
            ref_lists = [v for v in ref_lists if v]
            if not ref_lists:
                continue
            width = max(len(v) for v in ref_lists)
            # Lists (per foot / per phase offset) get a band per position, scalars one band.
            lo = [min(v[i] for v in ref_lists if len(v) > i) for i in range(width)]
            hi = [max(v[i] for v in ref_lists if len(v) > i) for i in range(width)]
            cells = [title, ",".join(f"{x:.3f}" for x in lo), ",".join(f"{x:.3f}" for x in hi)]
            for m in focus:
                vals = _scalar_values(dig(m["locomotion"], keys))
                if not vals:
                    cells.append("–")
                    continue
                parts = []
                for i, v in enumerate(vals):
                    j = min(i, width - 1)
                    below, above = v < lo[j], v > hi[j]
                    parts.append(
                        f"**{v:.3f}↓**" if below else f"**{v:.3f}↑**" if above else f"{v:.3f}"
                    )
                cells.append(",".join(parts))
            out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def to_markdown(header, rows) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Compare locomotion metrics.json files in one table."
    )
    parser.add_argument(
        "inputs", nargs="+", type=Path, help="metrics.json files or evaluation folders."
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Also write the Markdown table here."
    )
    parser.add_argument(
        "--csv", type=Path, default=None, help="Also write the table as CSV (no flag emoji)."
    )
    parser.add_argument(
        "--bands",
        type=str,
        default=None,
        metavar="ROBOT",
        help="Also print, per terrain and profile, the min-max of each metric over all other robots and "
        "mark where ROBOT's runs fall outside it (e.g. --bands meldog).",
    )
    args = parser.parse_args()

    runs = []
    for p in args.inputs:
        try:
            runs.append(load(p))
        except FileNotFoundError:
            print(f"[WARN] no metrics.json in {p}, skipped")
    header, rows = rows_for(runs)
    table = to_markdown(header, rows)
    if args.bands:
        table += reference_bands(runs, args.bands)
    print(table)
    if args.output:
        args.output.write_text(table)
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for row in rows:
                writer.writerow([cell.split(" ")[0] for cell in row])


if __name__ == "__main__":
    main()
