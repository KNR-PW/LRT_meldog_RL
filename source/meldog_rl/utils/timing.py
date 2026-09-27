# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Wall-clock cost of one call, for the timing block next to accuracy in metrics.json."""

from __future__ import annotations

import platform
import time
from collections.abc import Callable

import numpy as np
import torch


def device_name(device: str | torch.device) -> str:
    """Readable name of a torch device: the GPU model, or the CPU."""
    dev = torch.device(device)
    if dev.type == "cuda":
        return torch.cuda.get_device_name(dev)
    return platform.processor() or platform.machine() or "cpu"


def time_callable(
    fn: Callable[[], object], device: str | torch.device, warmup: int = 50, steps: int = 1000
) -> dict:
    """Time ``fn()`` per call after ``warmup`` untimed calls.

    On CUDA each call is bracketed by ``torch.cuda.synchronize`` so the time covers the
    GPU work, not just the kernel launches. ``peak_mem_mb`` is the peak memory allocated
    by torch on the device during the timed calls, including everything already resident
    (weights, buffers); None on CPU.

    Returns:
        dict with ms_p50, ms_p95, fps (= 1000 / ms_p50), peak_mem_mb, device, warmup, steps.
    """
    dev = torch.device(device)
    cuda = dev.type == "cuda"

    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)

    ms = np.empty(steps, dtype=np.float64)
    for i in range(steps):
        if cuda:
            torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize(dev)
        ms[i] = (time.perf_counter() - t0) * 1000.0

    p50 = float(np.percentile(ms, 50))
    return {
        "ms_p50": p50,
        "ms_p95": float(np.percentile(ms, 95)),
        "fps": 1000.0 / p50 if p50 > 0 else None,
        "peak_mem_mb": torch.cuda.max_memory_allocated(dev) / 2**20 if cuda else None,
        "device": device_name(dev),
        "warmup": warmup,
        "steps": steps,
    }
