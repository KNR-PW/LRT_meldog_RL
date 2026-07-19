# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Simulation environment configs (fast training, minimal noise)."""

from .flat_cfg import FlatSimCfg
from .rough_cfg import RoughSimCfg
from .flat_obs_cfg import FlatObsSimCfg
from .rough_obs_cfg import RoughObsSimCfg
from .flat_v2_cfg import FlatSimV2Cfg, FlatSimV2Cfg_PLAY
from .rough_v2_cfg import RoughSimV2Cfg, RoughSimV2Cfg_PLAY

__all__ = [
    "FlatSimCfg",
    "RoughSimCfg",
    "FlatObsSimCfg",
    "RoughObsSimCfg",
    "FlatSimV2Cfg",
    "FlatSimV2Cfg_PLAY",
    "RoughSimV2Cfg",
    "RoughSimV2Cfg_PLAY",
]
