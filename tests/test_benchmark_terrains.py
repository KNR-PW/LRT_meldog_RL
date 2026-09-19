# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Fixed benchmark terrain cells.

The benchmark puts env i on the same terrain cell for every robot. The column of each
sub-terrain kind must follow Isaac Lab's TerrainGenerator rule, otherwise the report would
label cells with the wrong kind.

No Isaac imports — runs anywhere:  python tests/test_benchmark_terrains.py
"""

import importlib.util
import os
import sys
from types import SimpleNamespace

_PATH = os.path.join(
    os.path.dirname(__file__), "..", "source", "meldog_rl", "eval", "benchmark_terrains.py"
)
_spec = importlib.util.spec_from_file_location("benchmark_terrains", _PATH)
bt = importlib.util.module_from_spec(_spec)
sys.modules["benchmark_terrains"] = bt
_spec.loader.exec_module(bt)


def rough_like_generator():
    """Proportions of Isaac Lab's ROUGH_TERRAINS_CFG (as used by Meldog's rough task)."""
    subs = {
        "pyramid_stairs": 0.2,
        "pyramid_stairs_inv": 0.2,
        "boxes": 0.2,
        "random_rough": 0.2,
        "hf_pyramid_slope": 0.1,
        "hf_pyramid_slope_inv": 0.1,
    }
    return SimpleNamespace(
        num_rows=10,
        num_cols=20,
        sub_terrains={k: SimpleNamespace(proportion=v) for k, v in subs.items()},
    )


def test_column_kinds():
    kinds = bt.column_kinds(rough_like_generator())
    expected = (
        ["pyramid_stairs"] * 4
        + ["pyramid_stairs_inv"] * 4
        + ["boxes"] * 4
        + ["random_rough"] * 4
        + ["hf_pyramid_slope"] * 2
        + ["hf_pyramid_slope_inv"] * 2
    )
    assert kinds == expected, kinds
    print("  OK  column kinds follow the proportion rule")


def test_bench_cells():
    gen = rough_like_generator()
    for num_envs in (16, 48, 192):
        cells = bt.bench_cells(gen, num_envs)
        assert len(cells) == num_envs
        # Every env needs its own cell: cells share one spawn platform, so robots on the same cell
        # start on the same spot (stacked) instead of each on its own obstacle.
        assert len({(row, col) for row, col, _ in cells}) == num_envs, num_envs
        kinds = bt.column_kinds(gen)
        for row, col, kind in cells:
            assert kinds[col] == kind
        counts = [sum(1 for c in cells if c[2] == kind) for kind in set(kinds)]
        assert max(counts) - min(counts) <= max(counts) / 2, counts  # roughly balanced over kinds
    hard_rows = {row for row, _, _ in bt.bench_cells(gen, 16)}
    assert max(hard_rows) >= gen.num_rows - 2, "a small run must still see the hardest rows"
    print("  OK  16 / 48 / 192 envs each get their own cell, balanced over kinds and difficulty")


if __name__ == "__main__":
    test_column_kinds()
    test_bench_cells()
    print("All benchmark terrain tests passed.")
