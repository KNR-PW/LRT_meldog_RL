# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Meldog RL environment registration.

This module registers all Meldog tasks with gymnasium:

Locomotion - Simulation (fast training):
- Meldog-RL-Locomotion-Flat-Sim-v0
- Meldog-RL-Locomotion-Rough-Sim-v0
- Meldog-RL-Locomotion-FlatObs-Sim-v0
- Meldog-RL-Locomotion-RoughObs-Sim-v0

Locomotion - Simulation V2 (gait-quality reward package):
- Meldog-RL-Locomotion-Flat-Sim-v1
- Meldog-RL-Locomotion-Rough-Sim-v1
- Meldog-RL-Locomotion-Rough-SimD1-v1

Locomotion - Sim2Real (domain randomization):
- Meldog-RL-Locomotion-Flat-Real-v0
- Meldog-RL-Locomotion-Rough-Real-v0
- Meldog-RL-Locomotion-FlatObs-Real-v0
- Meldog-RL-Locomotion-RoughObs-Real-v0

Dataset Collection (cameras enabled):
- Meldog-RL-Dataset-Flat-v0
- Meldog-RL-Dataset-Rough-v0
- Meldog-RL-Dataset-FlatObs-v0
- Meldog-RL-Dataset-RoughObs-v0
"""

import gymnasium as gym

# Agent configs
from ..agents import (
    MeldogFlatPPORunnerCfg,
    MeldogFlatV2PPORunnerCfg,
    MeldogRoughPPORunnerCfg,
    MeldogRoughV2PPORunnerCfg,
)
from .configs import (  # Simulation; Simulation V2 (gait-quality); Sim2Real; Dataset
    FlatDatasetCfg,
    FlatObsDatasetCfg,
    FlatObsRealCfg,
    FlatObsSimCfg,
    FlatRealCfg,
    FlatSimCfg,
    FlatSimV2Cfg,
    RoughDatasetCfg,
    RoughObsDatasetCfg,
    RoughObsRealCfg,
    RoughObsSimCfg,
    RoughRealCfg,
    RoughSimCfg,
    RoughSimV2Cfg,
    RoughSimV2D1Cfg,
)
from .meldog_env import MeldogEnv

# Locomotion - Simulation
gym.register(
    id="Meldog-RL-Locomotion-Flat-Sim-v0",
    entry_point="meldog_rl.envs:MeldogEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FlatSimCfg,
        "rsl_rl_cfg_entry_point": MeldogFlatPPORunnerCfg,
    },
)

gym.register(
    id="Meldog-RL-Locomotion-Rough-Sim-v0",
    entry_point="meldog_rl.envs:MeldogEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": RoughSimCfg,
        "rsl_rl_cfg_entry_point": MeldogRoughPPORunnerCfg,
    },
)

gym.register(
    id="Meldog-RL-Locomotion-FlatObs-Sim-v0",
    entry_point="meldog_rl.envs:MeldogEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FlatObsSimCfg,
        "rsl_rl_cfg_entry_point": MeldogFlatPPORunnerCfg,
    },
)

gym.register(
    id="Meldog-RL-Locomotion-RoughObs-Sim-v0",
    entry_point="meldog_rl.envs:MeldogEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": RoughObsSimCfg,
        "rsl_rl_cfg_entry_point": MeldogRoughPPORunnerCfg,
    },
)


# Locomotion - Simulation V2 (gait-quality reward package)
gym.register(
    id="Meldog-RL-Locomotion-Flat-Sim-v1",
    entry_point="meldog_rl.envs:MeldogEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FlatSimV2Cfg,
        "rsl_rl_cfg_entry_point": MeldogFlatV2PPORunnerCfg,
    },
)

gym.register(
    id="Meldog-RL-Locomotion-Rough-Sim-v1",
    entry_point="meldog_rl.envs:MeldogEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": RoughSimV2Cfg,
        "rsl_rl_cfg_entry_point": MeldogRoughV2PPORunnerCfg,
    },
)

gym.register(
    id="Meldog-RL-Locomotion-Rough-SimD1-v1",
    entry_point="meldog_rl.envs:MeldogEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": RoughSimV2D1Cfg,
        "rsl_rl_cfg_entry_point": MeldogRoughV2PPORunnerCfg,
    },
)


# Locomotion - Sim2Real
gym.register(
    id="Meldog-RL-Locomotion-Flat-Real-v0",
    entry_point="meldog_rl.envs:MeldogEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FlatRealCfg,
        "rsl_rl_cfg_entry_point": MeldogFlatPPORunnerCfg,
    },
)

gym.register(
    id="Meldog-RL-Locomotion-Rough-Real-v0",
    entry_point="meldog_rl.envs:MeldogEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": RoughRealCfg,
        "rsl_rl_cfg_entry_point": MeldogRoughPPORunnerCfg,
    },
)

gym.register(
    id="Meldog-RL-Locomotion-FlatObs-Real-v0",
    entry_point="meldog_rl.envs:MeldogEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FlatObsRealCfg,
        "rsl_rl_cfg_entry_point": MeldogFlatPPORunnerCfg,
    },
)

gym.register(
    id="Meldog-RL-Locomotion-RoughObs-Real-v0",
    entry_point="meldog_rl.envs:MeldogEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": RoughObsRealCfg,
        "rsl_rl_cfg_entry_point": MeldogRoughPPORunnerCfg,
    },
)


# Dataset Collection
gym.register(
    id="Meldog-RL-Dataset-Flat-v0",
    entry_point="meldog_rl.envs:MeldogEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FlatDatasetCfg,
        "rsl_rl_cfg_entry_point": MeldogFlatPPORunnerCfg,
    },
)

gym.register(
    id="Meldog-RL-Dataset-Rough-v0",
    entry_point="meldog_rl.envs:MeldogEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": RoughDatasetCfg,
        "rsl_rl_cfg_entry_point": MeldogRoughPPORunnerCfg,
    },
)

gym.register(
    id="Meldog-RL-Dataset-FlatObs-v0",
    entry_point="meldog_rl.envs:MeldogEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FlatObsDatasetCfg,
        "rsl_rl_cfg_entry_point": MeldogFlatPPORunnerCfg,
    },
)

gym.register(
    id="Meldog-RL-Dataset-RoughObs-v0",
    entry_point="meldog_rl.envs:MeldogEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": RoughObsDatasetCfg,
        "rsl_rl_cfg_entry_point": MeldogRoughPPORunnerCfg,
    },
)


# Export classes
__all__ = [
    "MeldogEnv",
    # Simulation configs
    "FlatSimCfg",
    "RoughSimCfg",
    "FlatObsSimCfg",
    "RoughObsSimCfg",
    # Simulation V2 configs
    "FlatSimV2Cfg",
    "RoughSimV2Cfg",
    # Sim2Real configs
    "FlatRealCfg",
    "RoughRealCfg",
    "FlatObsRealCfg",
    "RoughObsRealCfg",
    # Dataset configs
    "FlatDatasetCfg",
    "RoughDatasetCfg",
    "FlatObsDatasetCfg",
    "RoughObsDatasetCfg",
]
