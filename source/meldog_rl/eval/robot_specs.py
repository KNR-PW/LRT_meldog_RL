# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Per-robot naming conventions needed to evaluate a locomotion policy.

Each quadruped names its feet and joints differently. A :class:`RobotSpec` holds the
regular expressions that find the base, feet, knees and hip-flexion bodies, and how to
read the leg label (front/rear, left/right) from a body or joint name. Everything here
is plain Python so it can be tested without Isaac Sim.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

FOOT_ORDER = ("FL", "FR", "RL", "RR")


@dataclass(frozen=True)
class RobotSpec:
    """Naming conventions of one robot.

    Attributes:
        name: Short robot name used in output folders and metadata.
        task_regex: Regular expression matched against the gym task id.
        base_body: Name of the trunk body (base contact, posture).
        foot_regex: Regular expression for the four foot bodies.
        knee_regex: Regular expression for the four knee joints.
        hip_regex: Regular expression for the four hip-flexion (thigh) bodies.
        shank_regex: Regular expression for the four shank (lower-leg) bodies. With
            ``hip_regex`` and ``foot_regex`` it gives thigh and shank lengths.
        name_order: How the leg label is written at the start of a name:
            ``"side_end"`` (``LF``, ``RH``: ANYmal, Meldog) or
            ``"end_side"`` (``FL``, ``RR``, ``HL``: Unitree, Spot, most others).
    """

    name: str
    task_regex: str
    base_body: str
    foot_regex: str
    knee_regex: str
    hip_regex: str
    shank_regex: str
    name_order: str


def leg_label(name: str, name_order: str) -> str | None:
    """Canonical leg label (FL, FR, RL, RR) from a body or joint name, or None.

    ``side_end`` reads the first two letters (``LFK_joint`` -> L, F). ``end_side`` reads the first
    and the last letter of the first ``_``-separated token, so both ``FL_foot`` and Zsibot's
    ``FBL_KNEE_JOINT`` give F, L. Hind (``H``) counts as rear.
    """
    token = re.sub(r"[^A-Za-z]", "", name.split("_")[0]).upper()
    if len(token) < 2:
        return None
    if name_order == "side_end":
        side, end = token[0], token[1]
    elif name_order == "end_side":
        end, side = token[0], token[-1]
    else:
        raise ValueError(f"unknown name_order: {name_order}")
    if side not in ("L", "R") or end not in ("F", "R", "H"):
        return None
    return ("F" if end == "F" else "R") + side


def canonical_order(names: list[str], name_order: str) -> list[int] | None:
    """Permutation that puts four leg-labelled names into FL, FR, RL, RR order.

    Returns None if the names do not map to exactly those four labels.
    """
    labels = [leg_label(n, name_order) for n in names]
    if len(labels) != 4 or sorted(l for l in labels if l) != sorted(FOOT_ORDER):
        return None
    return [labels.index(label) for label in FOOT_ORDER]


ROBOT_SPECS: tuple[RobotSpec, ...] = (
    RobotSpec(
        "meldog",
        r"^Meldog-",
        "trunk_link",
        r".*F_link",
        r".*K_joint",
        r".*UL_link",
        r".*LL_link",
        "side_end",
    ),
    RobotSpec(
        "anymal_b", r"Anymal-B", "base", r".*_FOOT", r".*_KFE", r".*_THIGH", r".*_SHANK", "side_end"
    ),
    RobotSpec(
        "anymal_c", r"Anymal-C", "base", r".*_FOOT", r".*_KFE", r".*_THIGH", r".*_SHANK", "side_end"
    ),
    RobotSpec(
        "anymal_d", r"Anymal-D", "base", r".*_FOOT", r".*_KFE", r".*_THIGH", r".*_SHANK", "side_end"
    ),
    RobotSpec(
        "go1",
        r"Unitree-Go1",
        "(trunk|base)",
        r".*_foot",
        r".*_calf_joint",
        r".*_thigh",
        r".*_calf",
        "end_side",
    ),
    RobotSpec(
        "go2",
        r"Unitree-Go2",
        "base",
        r".*_foot",
        r".*_calf_joint",
        r".*_thigh",
        r".*_calf",
        "end_side",
    ),
    RobotSpec(
        "a1",
        r"Unitree-A1",
        "(trunk|base)",
        r".*_foot",
        r".*_calf_joint",
        r".*_thigh",
        r".*_calf",
        "end_side",
    ),
    RobotSpec("spot", r"-Spot-", "body", r".*_foot", r".*_kn", r".*_uleg", r".*_lleg", "end_side"),
    # robot_lab quadrupeds (URDF link names)
    RobotSpec(
        "b2",
        r"Unitree-B2",
        "base_link",
        r".*_foot",
        r".*_calf_joint",
        r".*_thigh",
        r".*_calf",
        "end_side",
    ),
    RobotSpec(
        "lite3",
        r"Deeprobotics-Lite3",
        "TORSO",
        r".*_FOOT",
        r".*_Knee_joint",
        r".*_THIGH",
        r".*_SHANK",
        "end_side",
    ),
    RobotSpec(
        "d1",
        r"Agibot-D1",
        "BASE_LINK",
        r".*_FOOT_LINK",
        r".*_KNEE_JOINT",
        r".*_HIP_LINK",
        r".*_KNEE_LINK",
        "end_side",
    ),
    RobotSpec(
        "zsl1",
        r"Zsibot-ZSL1",
        "BASE_LINK",
        r".*_FOOT_LINK",
        r".*_KNEE_JOINT",
        r".*_HIP_LINK",
        r".*_KNEE_LINK",
        "end_side",
    ),
    RobotSpec(
        "magicdog",
        r"MagicLab-Dog",
        "base",
        r".*_foot",
        r".*_calf_joint",
        r".*_thigh",
        r".*_calf",
        "end_side",
    ),
)


# Approximate nominal masses (model files), used only to size the ``real`` profile's mass change.
NOMINAL_MASS_KG = {
    "meldog": 21.5,
    "anymal_b": 30.4,
    "anymal_c": 52.1,
    "anymal_d": 57.0,
    "go1": 13.1,
    "go2": 15.5,
    "a1": 13.7,
    "spot": 32.5,
    "b2": 74.6,
    "lite3": 11.9,
    "d1": 15.2,
    "zsl1": 15.2,
    "magicdog": 15.9,
}


def find_robot_spec(task: str) -> RobotSpec:
    """The spec whose ``task_regex`` matches ``task``."""
    for spec in ROBOT_SPECS:
        if re.search(spec.task_regex, task):
            return spec
    raise KeyError(
        f"No robot spec matches task '{task}'. Add one to meldog_rl/eval/robot_specs.py."
    )
