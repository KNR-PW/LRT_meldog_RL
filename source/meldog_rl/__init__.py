# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Meldog RL - Quadruped Robot Locomotion with Reinforcement Learning.

This package provides:
- Locomotion environments for training (flat, rough, obstacles)
- Sim-to-real configurations with domain randomization
- Dataset collection environments for perception training
- Perception models (heightmap, voxel, SLAM)

Task naming convention:
    Meldog-RL-{Purpose}-{Terrain}-{Domain}-v{Version}
    
    Purpose: Locomotion, Dataset
    Terrain: Flat, Rough, FlatObs, RoughObs
    Domain:  Sim (simulation), Real (sim2real)

Output naming convention:
    {PREFIX}_{details}_{YYYY-MM-DD}_{HH-MM-SS}
    
    LM = Locomotion Model
    LE = Locomotion Evaluation
    PM = Perception Model
    PE = Perception Evaluation
    PD = Perception Dataset
"""

__version__ = "0.1.0"


def get_version():
    """Return package version."""
    return __version__


# Lazy imports - only load Isaac Lab stuff when needed
def _register_tasks():
    """Register tasks with gymnasium. Call this after Isaac Lab is initialized."""
    from . import envs  # noqa: F401
    from . import agents  # noqa: F401


# Check if we're running inside Isaac Lab (pxr available)
try:
    import pxr  # noqa: F401
    # Inside Isaac Lab - register tasks immediately
    _register_tasks()
except ImportError:
    # Outside Isaac Lab - tasks will be registered when scripts import after AppLauncher
    pass
