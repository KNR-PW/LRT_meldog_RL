# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Rough terrain V2 configuration (gait-quality + robustness package).

Adds the Spot-ported gait-quality rewards and the retuned command/gait shaping
on top of ``RoughSimCfg`` (Run A/A2), plus the Run B robustness package: reset
randomization, periodic pushes, observation noise, and the tamed actuator
(velocity_limit 12.0, action_scale 0.25). Foot clearance uses the
terrain-relative variant (height above the nearest height-scanner grid point),
so it is enabled on rough.
"""

from isaaclab.utils import configclass

from ..base_cfg import V2EventCfg
from .rough_cfg import RoughSimCfg


@configclass
class RoughSimV2Cfg(RoughSimCfg):
    """Rough terrain, V2 gait-quality + robustness package (Locomotion Run B).

    Use this for:
    - Full 1500-iter V2 rough training and rough benchmark
    """

    # V2 gait-quality rewards (Spot ports)
    foot_slip_reward_scale = -0.5
    gait_sync_reward_scale = 2.0
    gait_sync_max_err = 0.5                    # Run B: was 0.2 (restore gradient in the
                                               # degenerate one-diagonal-carries region)
    air_time_variance_reward_scale = -1.0
    air_time_mode_reward_scale = 1.0           # Run A2: replaces legacy feet_air_time
    foot_clearance_reward_scale = 0.5          # terrain-relative clearance (valid on rough)
    joint_deviation_hip_reward_scale = -0.1

    # Retuned command / gait shaping
    feet_air_time_reward_scale = 0.0           # Run A2: was 0.5 (subsidized diagonal-float exploit)
    yaw_rate_reward_scale = 1.5                # was 0.7 (parity with linear tracking)
    flat_orientation_reward_scale = -1.0       # was 0.0 (fights forward lean)

    # Behavior flags
    air_time_gate_full_cmd = True              # gate air-time on the full 3-dim command
    pure_rotation_fraction = 0.2               # 20% of resamples are turn-in-place

    # Run B robustness package
    reset_randomization = True                 # yaw/joint/velocity noise at reset (inline)
    obs_noise = True                           # additive uniform obs noise (Go2 values)
    events: V2EventCfg = V2EventCfg()          # base randomization + periodic pushes
    action_scale = 0.25                        # was 0.3

    def __post_init__(self):
        # Post init of parent (terrain generator tuning)
        super().__post_init__()

        # Run B actuator taming (V2 only; MELDOG_CFG / v0 untouched)
        self.robot.actuators["all_joints"].velocity_limit = 12.0


@configclass
class RoughSimV2Cfg_PLAY(RoughSimV2Cfg):
    """Rough V2 for play/inference (no curriculum, small scene, no randomization).

    Mirrors ``RoughSimCfg_PLAY``'s overrides.
    """

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        # Disable curriculum for play
        self.enable_curriculum = False
        if self.terrain.terrain_generator is not None:
            self.terrain.terrain_generator.curriculum = False

        # Spawn on all difficulty levels (not just easiest)
        self.terrain.max_init_terrain_level = None  # Random across all levels

        # Smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # Disable randomization for play
        self.events = None  # No domain randomization during play
        self.reset_randomization = False
        self.obs_noise = False
