# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Flat terrain with obstacles configuration for sim-to-real transfer."""

from isaaclab.utils import configclass

from ..base_cfg import Sim2RealEventCfg
from ..simulation.flat_obs_cfg import FlatObsSimCfg


@configclass
class FlatObsRealCfg(FlatObsSimCfg):
    """Flat terrain + obstacles with domain randomization for sim-to-real.
    
    Inherits terrain from FlatObsSimCfg, adds aggressive randomization.
    """
    
    # Override with aggressive domain randomization
    events: Sim2RealEventCfg = Sim2RealEventCfg()
