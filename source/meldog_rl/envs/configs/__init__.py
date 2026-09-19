# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configurations for Meldog RL tasks."""

from .base_cfg import (
    CAMERA_BACK,
    CAMERA_FRONT,
    CAMERA_LEFT,
    CAMERA_RIGHT,
    CAMERA_TOP,
    CONTACT_SENSOR_CFG,
    GT_SCANNER_CFG,
    HEIGHT_SCANNER_CFG,
    MELDOG_CFG,
    MELDOG_USD_PATH,
    BaseEventCfg,
    BaseMeldogEnvCfg,
    Sim2RealEventCfg,
    V2EventCfg,
)

# Dataset collection configs
from .dataset import FlatDatasetCfg, FlatObsDatasetCfg, RoughDatasetCfg, RoughObsDatasetCfg

# Sim2Real configs
from .sim2real import FlatObsRealCfg, FlatRealCfg, RoughObsRealCfg, RoughRealCfg

# Simulation configs
from .simulation import (
    FlatObsSimCfg,
    FlatSimCfg,
    FlatSimV2Cfg,
    FlatSimV2Cfg_PLAY,
    RoughObsSimCfg,
    RoughSimCfg,
    RoughSimV2Cfg,
    RoughSimV2Cfg_PLAY,
    RoughSimV2D1Cfg,
    RoughSimV2D1Cfg_PLAY,
)

__all__ = [
    # Base
    "BaseMeldogEnvCfg",
    "BaseEventCfg",
    "V2EventCfg",
    "Sim2RealEventCfg",
    "MELDOG_CFG",
    "MELDOG_USD_PATH",
    # Cameras
    "CAMERA_FRONT",
    "CAMERA_BACK",
    "CAMERA_LEFT",
    "CAMERA_RIGHT",
    "CAMERA_TOP",
    # Sensors
    "CONTACT_SENSOR_CFG",
    "HEIGHT_SCANNER_CFG",
    "GT_SCANNER_CFG",
    # Simulation
    "FlatSimCfg",
    "RoughSimCfg",
    "FlatObsSimCfg",
    "RoughObsSimCfg",
    "FlatSimV2Cfg",
    "FlatSimV2Cfg_PLAY",
    "RoughSimV2Cfg",
    "RoughSimV2D1Cfg",
    "RoughSimV2D1Cfg_PLAY",
    "RoughSimV2Cfg_PLAY",
    # Sim2Real
    "FlatRealCfg",
    "RoughRealCfg",
    "FlatObsRealCfg",
    "RoughObsRealCfg",
    # Dataset
    "FlatDatasetCfg",
    "RoughDatasetCfg",
    "FlatObsDatasetCfg",
    "RoughObsDatasetCfg",
]
