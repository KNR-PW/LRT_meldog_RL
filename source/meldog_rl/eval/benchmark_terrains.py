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


def bench_cells(generator_cfg, num_envs: int) -> list[tuple[int, int, str]]:
    """Fixed (row, column, kind) per env: every kind at every benchmark row, repeated.

    The column used for a kind is the middle one of the columns holding it.
    """
    kinds = column_kinds(generator_cfg)
    rows = [r for r in BENCH_ROWS if r < generator_cfg.num_rows]
    unique = list(dict.fromkeys(kinds))
    kind_col = {}
    for kind in unique:
        cols = [c for c, k in enumerate(kinds) if k == kind]
        kind_col[kind] = cols[len(cols) // 2]
    base = [(row, kind_col[kind], kind) for kind in unique for row in rows]
    return [base[i % len(base)] for i in range(num_envs)]
