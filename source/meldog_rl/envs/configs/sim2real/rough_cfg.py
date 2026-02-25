# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Rough terrain configuration for sim-to-real transfer."""

from isaaclab.utils import configclass

from ..base_cfg import Sim2RealEventCfg
from ..simulation.rough_cfg import RoughSimCfg


@configclass
class RoughRealCfg(RoughSimCfg):
    """Rough terrain with domain randomization for sim-to-real transfer.
    
    Inherits terrain from RoughSimCfg, adds aggressive randomization.
    """
    
    # Override with aggressive domain randomization
    events: Sim2RealEventCfg = Sim2RealEventCfg()
