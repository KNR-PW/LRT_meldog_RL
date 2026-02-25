# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Sim-to-Real environment configs (aggressive domain randomization)."""

from .flat_cfg import FlatRealCfg
from .rough_cfg import RoughRealCfg
from .flat_obs_cfg import FlatObsRealCfg
from .rough_obs_cfg import RoughObsRealCfg

__all__ = ["FlatRealCfg", "RoughRealCfg", "FlatObsRealCfg", "RoughObsRealCfg"]
