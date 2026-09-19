# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Leg-label mapping tests for the evaluator's robot specs.

Every robot names its feet and knees differently (``LFF_link``, ``FL_foot``, ``LH_FOOT``,
``hr_foot``). The evaluator must put all of them into FL, FR, RL, RR order, or gait
phase offsets and per-foot metrics are silently wrong.

No Isaac imports — runs anywhere:  python tests/test_robot_specs.py
"""

import importlib.util
import os
import sys

# Load the module file directly: importing the meldog_rl package would pull in Isaac Lab.
_PATH = os.path.join(
    os.path.dirname(__file__), "..", "source", "meldog_rl", "eval", "robot_specs.py"
)
_spec = importlib.util.spec_from_file_location("robot_specs", _PATH)
robot_specs = importlib.util.module_from_spec(_spec)
sys.modules["robot_specs"] = robot_specs
_spec.loader.exec_module(robot_specs)

# (task id, body or joint names in the order an asset may list them, expected FL, FR, RL, RR)
CASES = [
    (
        "Meldog-RL-Locomotion-Rough-SimD1-v1",
        ["LFF_link", "LRF_link", "RFF_link", "RRF_link"],
        ["LFF_link", "RFF_link", "LRF_link", "RRF_link"],
    ),
    (
        "Meldog-RL-Locomotion-Rough-SimD1-v1",
        ["LFK_joint", "LRK_joint", "RFK_joint", "RRK_joint"],
        ["LFK_joint", "RFK_joint", "LRK_joint", "RRK_joint"],
    ),
    (
        "Isaac-Velocity-Rough-Anymal-C-v0",
        ["LF_FOOT", "LH_FOOT", "RF_FOOT", "RH_FOOT"],
        ["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"],
    ),
    (
        "Isaac-Velocity-Rough-Unitree-Go2-v0",
        ["FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"],
        ["FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"],
    ),
    (
        "Isaac-Velocity-Rough-Unitree-A1-v0",
        ["RR_foot", "RL_foot", "FR_foot", "FL_foot"],
        ["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
    ),
    (
        "Isaac-Velocity-Flat-Spot-v0",
        ["fl_foot", "fr_foot", "hl_foot", "hr_foot"],
        ["fl_foot", "fr_foot", "hl_foot", "hr_foot"],
    ),
    (
        "RobotLab-Isaac-Velocity-Rough-Deeprobotics-Lite3-v0",
        ["HR_Knee_joint", "HL_Knee_joint", "FR_Knee_joint", "FL_Knee_joint"],
        ["FL_Knee_joint", "FR_Knee_joint", "HL_Knee_joint", "HR_Knee_joint"],
    ),
    (
        "RobotLab-Isaac-Velocity-Rough-Zsibot-ZSL1-v0",
        ["FBL_KNEE_JOINT", "FAR_KNEE_JOINT", "RBL_KNEE_JOINT", "RAR_KNEE_JOINT"],
        ["FBL_KNEE_JOINT", "FAR_KNEE_JOINT", "RBL_KNEE_JOINT", "RAR_KNEE_JOINT"],
    ),
    (
        "RobotLab-Isaac-Velocity-Rough-Agibot-D1-v0",
        ["FL_FOOT_LINK", "FR_FOOT_LINK", "RR_FOOT_LINK", "RL_FOOT_LINK"],
        ["FL_FOOT_LINK", "FR_FOOT_LINK", "RL_FOOT_LINK", "RR_FOOT_LINK"],
    ),
    (
        "RobotLab-Isaac-Velocity-Rough-Unitree-B2-v0",
        ["FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"],
        ["FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"],
    ),
]


def test_canonical_order():
    for task, names, expected in CASES:
        spec = robot_specs.find_robot_spec(task)
        perm = robot_specs.canonical_order(names, spec.name_order)
        got = [names[i] for i in perm]
        assert got == expected, f"{spec.name}: {names} -> {got}, expected {expected}"
        print(f"  OK  {spec.name}: {got}")


def test_rejects_incomplete_or_wrong_convention():
    assert robot_specs.canonical_order(["FL_foot", "FR_foot"], "end_side") is None
    # Unitree names read side-first give no valid set of four labels.
    assert (
        robot_specs.canonical_order(["FL_foot", "FR_foot", "RL_foot", "RR_foot"], "side_end")
        is None
    )
    print("  OK  incomplete or wrong-convention names are rejected")


def test_task_lookup():
    expected = {
        "Meldog-RL-Locomotion-Flat-Sim-v0": "meldog",
        "Isaac-Velocity-Rough-Anymal-C-Direct-v0": "anymal_c",
        "Isaac-Velocity-Flat-Anymal-D-v0": "anymal_d",
        "Isaac-Velocity-Rough-Unitree-Go1-v0": "go1",
        "Isaac-Velocity-Flat-Spot-v0": "spot",
        "RobotLab-Isaac-Velocity-Rough-Unitree-Go2-v0": "go2",
        "RobotLab-Isaac-Velocity-Rough-MagicLab-Dog-Hip0125-v0": "magicdog",
        "RobotLab-Isaac-Velocity-Flat-Unitree-B2-v0": "b2",
    }
    for task, name in expected.items():
        assert robot_specs.find_robot_spec(task).name == name, task
    print("  OK  task ids resolve to robot specs")


if __name__ == "__main__":
    test_canonical_order()
    test_rejects_incomplete_or_wrong_convention()
    test_task_lookup()
    print("All robot spec tests passed.")
