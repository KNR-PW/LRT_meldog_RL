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
        [--output comparison.md] [--csv comparison.csv]
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
    ("impact BW", ("impact", "peak_force_bw", "mean"), "impact.peak_force_bw", "{:.2f}"),
    ("touchdown m/s", ("impact", "touchdown_vel", "mean"), "impact.touchdown_vel", "{:.2f}"),
    ("pitch std rad", ("attitude", "pitch_std", "mean"), "attitude.pitch_std", "{:.3f}"),
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
    header = ["run", "robot", "mass kg", "leg m", "terrain"] + [c[0] for c in COLUMNS]
    rows = []
    for m in sorted(runs, key=lambda r: r["meta"].get("robot_mass") or 0.0):
        meta, loco, flags = m["meta"], m["locomotion"], m.get("flags", {})
        leg = meta.get("robot_leg_length")
        task = meta.get("task") or ""
        terrain = "flat" if "flat" in task.lower() else "rough" if "rough" in task.lower() else "?"
        row = [
            m["_path"].name,
            meta.get("robot_name") or "meldog",
            f"{meta['robot_mass']:.1f}" if meta.get("robot_mass") else "–",
            f"{leg:.3f}" if leg else "–",
            terrain,
        ]
        for _, keys, flag_key, style in COLUMNS:
            text = fmt(dig(loco, keys), style)
            flag = FLAG_EMOJI.get(flags.get(flag_key)) if flag_key else None
            row.append(f"{text} {flag}" if flag else text)
        rows.append(row)
    return header, rows


def to_markdown(header, rows) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Compare locomotion metrics.json files in one table.")
    parser.add_argument("inputs", nargs="+", type=Path, help="metrics.json files or evaluation folders.")
    parser.add_argument("--output", type=Path, default=None, help="Also write the Markdown table here.")
    parser.add_argument("--csv", type=Path, default=None, help="Also write the table as CSV (no flag emoji).")
    args = parser.parse_args()

    runs = [load(p) for p in args.inputs]
    header, rows = rows_for(runs)
    table = to_markdown(header, rows)
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
