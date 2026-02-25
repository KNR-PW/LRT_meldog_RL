# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Dataset collection environment configs (cameras enabled)."""

from .flat_cfg import FlatDatasetCfg
from .rough_cfg import RoughDatasetCfg
from .flat_obs_cfg import FlatObsDatasetCfg
from .rough_obs_cfg import RoughObsDatasetCfg

__all__ = ["FlatDatasetCfg", "RoughDatasetCfg", "FlatObsDatasetCfg", "RoughObsDatasetCfg"]
