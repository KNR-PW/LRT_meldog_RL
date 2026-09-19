# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Flat terrain V2 configuration (phase-clock gait + robustness package).

On top of ``FlatSimCfg``: the Run C phase-clock gait (clock observations +
contact_schedule reward + swing-window foot clearance), the retuned command
shaping (Run A), and the Run B robustness package: reset randomization,
periodic pushes, observation noise, and the tamed actuator (velocity_limit
12.0, action_scale 0.25).
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

    # V2 gait-quality rewards. Run C: the phase-clock contact_schedule term
    # replaces the timing-statistics terms (gait_sync / air_time_mode /
    # air_time_variance), whose kernels kept finding degenerate optima
    # (diagonal limp, shuffle-in-place).
    foot_slip_reward_scale = -0.5
    contact_schedule_reward_scale = 2.0  # Run C: match contacts to the clock
    foot_clearance_reward_scale = 2.0  # terrain-relative, swing-window form
    joint_deviation_hip_reward_scale = -0.1

    # Run D posture package: stand at the measured nominal height, and tilt WITH the
    # terrain instead of holding gravity-level (the legacy gravity term is switched
    # off; on flat ground the terrain-relative term reduces to it exactly).
    # Run D3: base-height reward OFF, foot clearance up — see rough_v2_cfg.
    base_height_reward_scale = 0.0  # target = cfg.base_height_target (0.34 m)
    # Run D3: -8.0 on both terrains — see rough_v2_cfg for why D2's -20.0 was a
    # freeze incentive. (FlatSimCfg's legacy gravity term was -5.0; on flat ground
    # the two are mathematically identical, so -8.0 is slightly stronger than the
    # original level-keeping weight, not weaker.)
    flat_orientation_terrain_reward_scale = -8.0
    # Foot-lift target is the foot BODY ORIGIN height above terrain, and that origin
    # sits ~3.6 cm above ground when in contact (collision radius), so 0.08 only
    # asked for ~4.4 cm of real clearance. 0.13 -> ~9.4 cm, near v1.0-rough's ~10.7.
    foot_clearance_target = 0.13

    # Retuned command / gait shaping
    feet_air_time_reward_scale = 0.0  # Run A2: was 0.5 (subsidized diagonal-float exploit)
    yaw_rate_reward_scale = 1.5  # was 0.7 (parity with linear tracking)
    flat_orientation_reward_scale = 0.0  # Run D: replaced by flat_orientation_terrain (was -5.0)

    # Behavior flags
    air_time_gate_full_cmd = True  # gate air-time on the full 3-dim command
    pure_rotation_fraction = 0.2  # 20% of resamples are turn-in-place
    gait_clock = True  # Run C: phase clock + [sin, cos] obs
    observation_space = 237  # 235 + 2 clock observations

    # Run B robustness package
    reset_randomization = True  # yaw/joint/velocity noise at reset (inline)
    obs_noise = True  # additive uniform obs noise (Go2 values)
    events: V2EventCfg = V2EventCfg()  # base randomization + periodic pushes
    action_scale = 0.25  # was 0.3

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
