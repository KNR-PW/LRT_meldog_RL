#!/usr/bin/env python3
# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""DEPRECATED shim — use ``evaluate_perception.py --method slam`` instead.

Kept so old commands keep working: forwards all arguments unchanged to
evaluate_perception.py, appending ``--method slam`` (variant defaults to
``elevation``; pass ``--slam_variant legacy`` for the old shift-and-composite).
"""

import os
import sys

target = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluate_perception.py")
argv = [sys.executable, target, *sys.argv[1:]]
if "--method" not in sys.argv:
    argv += ["--method", "slam"]

print("[DEPRECATED] evaluate_slam.py now forwards to:")
print("  " + " ".join(argv[1:]))
os.execv(sys.executable, argv)
