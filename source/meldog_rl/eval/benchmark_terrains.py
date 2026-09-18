# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Shared benchmark terrains and fixed terrain cells per env.

Every robot walks on the same terrain: Meldog's own flat, rough, flat-with-obstacles and
rough-with-obstacles generators (seed 42). Each env gets a fixed cell (difficulty row, column)
instead of a random level, so env ``i`` sees the same ground for every robot.

Cells: one column per sub-terrain kind, difficulty rows 0, 3, 6 and 9, repeated to fill the
envs. Columns follow Isaac Lab's ``TerrainGenerator`` rule for mapping sub-terrain proportions
to columns.
"""

from __future__ import annotations

import copy

import numpy as np

BENCH_TERRAINS = ("task", "flat", "rough", "obs", "rough_obs")
BENCH_ROWS = (0, 3, 6, 9)
BENCH_SEED = 42


def make_terrain_cfg(name: str):
    """Terrain importer config for a shared benchmark terrain (Meldog's generators)."""
    import isaaclab.sim as sim_utils
    from isaaclab.terrains import TerrainImporterCfg

    from meldog_rl.envs.configs.simulation import FlatSimCfg, RoughSimCfg
    from meldog_rl.envs.configs.simulation.flat_obs_cfg import FlatObsSimCfg
    from meldog_rl.envs.configs.simulation.rough_obs_cfg import RoughObsSimCfg

    if name == "flat":
        cfg = copy.deepcopy(FlatSimCfg().terrain)
        if cfg.terrain_type != "plane":
            cfg = TerrainImporterCfg(
                prim_path="/World/ground",
                terrain_type="plane",
                collision_group=-1,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply", restitution_combine_mode="multiply",
                    static_friction=1.0, dynamic_friction=1.0,
                ),
                debug_vis=False,
            )
        return cfg
    source = {"rough": RoughSimCfg, "obs": FlatObsSimCfg, "rough_obs": RoughObsSimCfg}[name]
    cfg = copy.deepcopy(source().terrain)
    gen = cfg.terrain_generator
    gen.curriculum = True      # rows are difficulty levels
    gen.seed = BENCH_SEED
    gen.use_cache = False
    cfg.max_init_terrain_level = None
    return cfg


def column_kinds(generator_cfg) -> list[str]:
    """Sub-terrain name of each generator column (Isaac Lab's proportion-to-column rule)."""
    names = list(generator_cfg.sub_terrains.keys())
    proportions = np.array([sub.proportion for sub in generator_cfg.sub_terrains.values()], dtype=float)
    proportions /= proportions.sum()
    cumsum = np.cumsum(proportions)
    return [names[int(np.min(np.where(col / generator_cfg.num_cols + 0.001 < cumsum)[0]))]
            for col in range(generator_cfg.num_cols)]


def spread_order(count: int) -> list[int]:
    """Row indices ordered so any prefix covers the whole difficulty range.

    ``spread_order(10)`` starts 0, 9, 4, 2, 7, ... instead of 0, 1, 2, ..., so a run with few envs
    still sees easy and hard terrain rather than only the easiest rows.
    """
    if count <= 2:
        return list(range(count))
    order = [0, count - 1]
    while len(order) < count:
        remaining = [r for r in range(count) if r not in order]
        # Pick the row furthest from every row already chosen.
        order.append(max(remaining, key=lambda r: min(abs(r - o) for o in order)))
    return order


def bench_cells(generator_cfg, num_envs: int) -> list[tuple[int, int, str]]:
    """A different terrain cell (row, column, kind) for every env, balanced over kinds.

    Each cell has its own flat spawn platform, which is where Isaac Lab puts the robot, so giving
    every env its own cell means: no robots stacked on one spot, no robot spawning inside a stair
    or below the surface, and every env on a different obstacle. Kinds are filled round-robin and
    rows (difficulty) are cycled, so the set spans easy to hard evenly.

    Falls back to repeating cells (with a warning) if there are more envs than cells.
    """
    kinds = column_kinds(generator_cfg)
    rows = spread_order(generator_cfg.num_rows)
    per_kind: dict[str, list[tuple[int, int, str]]] = {}
    for kind in dict.fromkeys(kinds):
        cols = [c for c, k in enumerate(kinds) if k == kind]
        # Rows change fastest and in spread order, so even a few envs cover easy to hard.
        per_kind[kind] = [(rows[i % len(rows)], cols[(i // len(rows)) % len(cols)], kind)
                          for i in range(len(rows) * len(cols))]

    cells: list[tuple[int, int, str]] = []
    cursors = {kind: 0 for kind in per_kind}
    while len(cells) < num_envs:
        progressed = False
        for kind, pool in per_kind.items():
            if len(cells) >= num_envs:
                break
            index = cursors[kind]
            if index < len(pool):
                cells.append(pool[index])
                cursors[kind] = index + 1
                progressed = True
        if not progressed:  # more envs than cells: repeat from the start
            print(f"[WARN] {num_envs} envs but only {len(cells)} terrain cells; cells are reused "
                  "and those robots spawn on the same spot.")
            cells += [cells[i % len(cells)] for i in range(num_envs - len(cells))]
            break
    return cells[:num_envs]
