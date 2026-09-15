# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark-mode command script shared by every robot.

Each tuple is (start time in s, (vx, vy, wz)); the schedule maps an episode-clock phase
to a command and repeats with period ``BENCHMARK_CYCLE_S``. Applied every step, per env.
"""

import torch

BENCHMARK_SCHEDULE = [
    (0.0, (0.8, 0.0, 0.0)),   # 0-5 s: walk forward
    (5.0, (0.0, 0.0, 0.8)),   # 5-10 s: turn in place
    (10.0, (0.5, 0.3, 0.0)),  # 10-15 s: diagonal walk
]
BENCHMARK_CYCLE_S = 15.0


def benchmark_commands(clock: torch.Tensor, device) -> torch.Tensor:
    """Scripted (vx, vy, wz) command per env from a per-env episode clock (seconds).

    ``clock`` is a (E,) tensor of episode-elapsed time. Returns an (E, 3) tensor.
    """
    phase = torch.remainder(clock, BENCHMARK_CYCLE_S)
    cmd = torch.zeros((clock.shape[0], 3), device=device)
    for start, vec in BENCHMARK_SCHEDULE:
        cmd[phase >= start] = torch.tensor(vec, device=device)
    return cmd
