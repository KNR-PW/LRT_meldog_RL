# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Flat terrain V2 configuration (gait-quality + robustness package).

Adds the Spot-ported gait-quality rewards and the retuned command/gait shaping
on top of ``FlatSimCfg`` (Run A/A2), plus the Run B robustness package: reset
randomization, periodic pushes, observation noise, and the tamed actuator
(velocity_limit 12.0, action_scale 0.25).
"""

from isaaclab.utils import configclass

from ..base_cfg import V2EventCfg
from .flat_cfg import FlatSimCfg


@configclass
class FlatSimV2Cfg(FlatSimCfg):
    """Flat terrain, V2 gait-quality + robustness package (Locomotion Run B).

    Use this for:
    - Flat sanity + strict-attitude benchmark of the V2 reward package
    """

    # V2 gait-quality rewards (Spot ports)
    foot_slip_reward_scale = -0.5
    gait_sync_reward_scale = 2.0
    gait_sync_max_err = 0.5                    # Run B: was 0.2 (restore gradient in the
                                               # degenerate one-diagonal-carries region)
    air_time_variance_reward_scale = -1.0
    air_time_mode_reward_scale = 1.0           # Run A2: replaces legacy feet_air_time
    foot_clearance_reward_scale = 0.5          # terrain-relative clearance
    joint_deviation_hip_reward_scale = -0.1

    # Retuned command / gait shaping
    feet_air_time_reward_scale = 0.0           # Run A2: was 0.5 (subsidized diagonal-float exploit)
    yaw_rate_reward_scale = 1.5                # was 0.7 (parity with linear tracking)

    # Behavior flags
    air_time_gate_full_cmd = True              # gate air-time on the full 3-dim command
    pure_rotation_fraction = 0.2               # 20% of resamples are turn-in-place

    # Run B robustness package
    reset_randomization = True                 # yaw/joint/velocity noise at reset (inline)
    obs_noise = True                           # additive uniform obs noise (Go2 values)
    events: V2EventCfg = V2EventCfg()          # base randomization + periodic pushes
    action_scale = 0.25                        # was 0.3

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        # Run B actuator taming (V2 only; MELDOG_CFG / v0 untouched)
        self.robot.actuators["all_joints"].velocity_limit = 12.0


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
        self.reset_randomization = False
        self.obs_noise = False
