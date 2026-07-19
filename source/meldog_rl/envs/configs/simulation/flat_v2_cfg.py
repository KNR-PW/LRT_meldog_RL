# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Flat terrain V2 configuration (gait-quality reward package, Run A).

Adds the Spot-ported gait-quality rewards and the retuned command/gait shaping
on top of ``FlatSimCfg``. Actuator params, action_scale and obs noise are left
untouched here -- those belong to Run B.
"""

from isaaclab.utils import configclass

from .flat_cfg import FlatSimCfg


@configclass
class FlatSimV2Cfg(FlatSimCfg):
    """Flat terrain, V2 gait-quality package (Locomotion Run A).

    Use this for:
    - Flat sanity + strict-attitude benchmark of the V2 reward package
    """

    # V2 gait-quality rewards (Spot ports)
    foot_slip_reward_scale = -0.5
    gait_sync_reward_scale = 2.0
    air_time_variance_reward_scale = -1.0
    foot_clearance_reward_scale = 0.5          # terrain-relative clearance
    joint_deviation_hip_reward_scale = -0.1

    # Retuned command / gait shaping
    feet_air_time_reward_scale = 0.5           # was 2.0 (rear-pair bounding exploit)
    yaw_rate_reward_scale = 1.5                # was 0.7 (parity with linear tracking)

    # Behavior flags
    air_time_gate_full_cmd = True              # gate air-time on the full 3-dim command
    pure_rotation_fraction = 0.2               # 20% of resamples are turn-in-place


@configclass
class FlatSimV2Cfg_PLAY(FlatSimV2Cfg):
    """Flat V2 for play/inference (small scene, no randomization)."""

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        # Smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # Disable randomization for play
        self.events = None
