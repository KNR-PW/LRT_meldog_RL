# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Rough terrain V2 configuration (phase-clock gait + robustness package).

On top of ``RoughSimCfg``: the Run C phase-clock gait (clock observations +
contact_schedule reward + swing-window foot clearance), the retuned command
shaping (Run A), and the Run B robustness package: reset randomization,
periodic pushes, observation noise, and the tamed actuator (velocity_limit
12.0, action_scale 0.25). Foot clearance uses the terrain-relative variant
(height above the nearest height-scanner grid point), so it is enabled on rough.
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

    # V2 gait-quality rewards. Run C: the phase-clock contact_schedule term
    # replaces the timing-statistics terms (gait_sync / air_time_mode /
    # air_time_variance), whose kernels kept finding degenerate optima
    # (diagonal limp, shuffle-in-place).
    foot_slip_reward_scale = -0.5
    contact_schedule_reward_scale = 2.0        # Run C: match contacts to the clock
    # Run D3: raise foot lift. Measurement showed the human-visible "low to the
    # ground" defect is FOOT LIFT, not body height -- v1.0-rough lifts its feet
    # 14.3 cm (p50) vs run C's 5-8 cm apex, while both stand at the same 0.30 m.
    # Scale 0.5 -> 2.0 with std 0.05 makes a 5 cm clearance shortfall cost ~0.2/step,
    # ~15 % of the +1.3/step contact_schedule (see the reward-magnitude rule).
    foot_clearance_reward_scale = 2.0          # terrain-relative, swing-window form
    # Target is the foot BODY ORIGIN height above terrain; that origin sits ~3.6 cm
    # above ground in contact (collision radius), so the old 0.08 asked for only
    # ~4.4 cm of real clearance. 0.13 -> ~9.4 cm, near v1.0-rough's measured ~10.7.
    foot_clearance_target = 0.13
    joint_deviation_hip_reward_scale = -0.1

    # Terrain-relative orientation: tilt WITH the terrain instead of holding
    # gravity-level (the legacy gravity term is off; on flat ground the
    # terrain-relative term reduces to it exactly).
    # Run D3: -20.0 (D2) drove pitch_terrain_rel 0.087 -> 0.021 but is also a
    # freeze incentive -- a SQUARED attitude penalty punishes the pitch oscillation
    # every trot produces, and D2 duly braced (pitch_std 0.139 -> 0.063, rear feet
    # planted). -8.0 is ~5 % of the gait term at a 5 deg error: enough to correct
    # (D's -1.0 was 0.58 %, i.e. nothing) without paying to stand still.
    flat_orientation_terrain_reward_scale = -8.0

    # Run D3: base-height reward OFF. Run D2's -40.0 fixed posture by making the
    # robot squat and plant both rear feet (duty 0.98/0.99, apex 2-5 mm) -- an
    # instantaneous height penalty fights a dynamic gait, whose body height must
    # oscillate. It was also solving a non-problem: run C already matched the
    # v1.0-rough baseline height (0.300 vs 0.299).
    base_height_reward_scale = 0.0             # target = cfg.base_height_target (0.34 m)

    # Retuned command / gait shaping
    feet_air_time_reward_scale = 0.0           # Run A2: was 0.5 (subsidized diagonal-float exploit)
    yaw_rate_reward_scale = 1.5                # was 0.7 (parity with linear tracking)
    flat_orientation_reward_scale = 0.0        # Run D: replaced by flat_orientation_terrain

    # Behavior flags
    air_time_gate_full_cmd = True              # gate air-time on the full 3-dim command
    pure_rotation_fraction = 0.2               # 20% of resamples are turn-in-place
    gait_clock = True                          # Run C: phase clock + [sin, cos] obs
    observation_space = 237                    # 235 + 2 clock observations

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
