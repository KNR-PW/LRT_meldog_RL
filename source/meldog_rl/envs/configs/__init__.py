# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configurations for Meldog RL tasks."""

from .base_cfg import (
    BaseMeldogEnvCfg,
    BaseEventCfg,
    Sim2RealEventCfg,
    MELDOG_CFG,
    MELDOG_USD_PATH,
    CAMERA_FRONT,
    CAMERA_BACK,
    CAMERA_LEFT,
    CAMERA_RIGHT,
    CAMERA_TOP,
    CONTACT_SENSOR_CFG,
    HEIGHT_SCANNER_CFG,
    GT_SCANNER_CFG,
)

# Simulation configs
from .simulation import (
    FlatSimCfg,
    RoughSimCfg,
    FlatObsSimCfg,
    RoughObsSimCfg,
)

# Sim2Real configs
from .sim2real import (
    FlatRealCfg,
    RoughRealCfg,
    FlatObsRealCfg,
    RoughObsRealCfg,
)

# Dataset collection configs
from .dataset import (
    FlatDatasetCfg,
    RoughDatasetCfg,
    FlatObsDatasetCfg,
    RoughObsDatasetCfg,
)

__all__ = [
    # Base
    "BaseMeldogEnvCfg",
    "BaseEventCfg",
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
