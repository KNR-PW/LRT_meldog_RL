# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Flat terrain configuration for sim-to-real transfer."""

from isaaclab.utils import configclass

from ..base_cfg import Sim2RealEventCfg
from ..simulation.flat_cfg import FlatSimCfg


@configclass
class FlatRealCfg(FlatSimCfg):
    """Flat terrain with domain randomization for sim-to-real transfer.
    
    Inherits terrain from FlatSimCfg, adds aggressive randomization.
    """
    
    # Override with aggressive domain randomization
    events: Sim2RealEventCfg = Sim2RealEventCfg()
