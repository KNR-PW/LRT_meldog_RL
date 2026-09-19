# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark conditions applied to any locomotion env config before the env is created.

Every robot is evaluated under the same conditions, independent of how its task was
trained:

- ``clean``: no observation noise, no pushes or external forces, no mass or centre-of-mass
  randomization, nominal friction (0.8 static / 0.6 dynamic), reset at the default pose with
  zero velocity and yaw 0, terrain curriculum off.
- ``real``: ``clean`` plus uniform observation noise in physical units (Isaac Lab velocity-task
  defaults), friction 0.4-1.2 / 0.3-1.0, added trunk mass of ±10 % of the robot's own mass and
  velocity pushes of ±0.5 m/s every 10-15 s.

Event terms are recognized by the name of their function, so the same code works for Meldog,
Isaac Lab and robot_lab configs.
"""

from __future__ import annotations

PROFILES = ("clean", "real")

# Randomization events removed in every profile (``real`` adds its own versions back).
_REMOVE = {
    "push_by_setting_velocity",
    "apply_external_force_torque",
    "randomize_rigid_body_mass",
    "randomize_rigid_body_com",
    "randomize_actuator_gains",
    "randomize_joint_parameters",
    "randomize_fixed_tendon_parameters",
    "randomize_physics_scene_gravity",
}

# Uniform observation noise of the ``real`` profile (half-width), in physical units.
REAL_OBS_NOISE = {
    "base_lin_vel": 0.1,
    "base_ang_vel": 0.2,
    "projected_gravity": 0.05,
    "joint_pos": 0.01,
    "joint_vel": 1.5,
    "height_scan": 0.1,
}
REAL_FRICTION = {"static_friction_range": (0.4, 1.2), "dynamic_friction_range": (0.3, 1.0)}
REAL_MASS_FRACTION = 0.10
REAL_PUSH = {
    "interval_range_s": (10.0, 15.0),
    "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
}


def _func_name(term) -> str:
    func = term.func
    return getattr(func, "__name__", type(func).__name__)


def _event_terms(events):
    """(name, term) pairs of all event terms set on an events config."""
    from isaaclab.managers import EventTermCfg

    if events is None:
        return []
    return [(name, term) for name, term in vars(events).items() if isinstance(term, EventTermCfg)]


def apply_clean_events(env_cfg) -> list[str]:
    """Remove randomization events and pin reset events to the nominal state. Returns a log."""
    log = []
    events = getattr(env_cfg, "events", None)
    for name, term in _event_terms(events):
        fn = _func_name(term)
        params = term.params
        if fn in _REMOVE:
            setattr(events, name, None)
            log.append(f"removed event '{name}' ({fn})")
        elif fn == "randomize_rigid_body_material":
            params["static_friction_range"] = (0.8, 0.8)
            params["dynamic_friction_range"] = (0.6, 0.6)
            params["restitution_range"] = (0.0, 0.0)
            log.append(f"event '{name}': nominal friction 0.8 / 0.6")
        elif fn == "reset_root_state_uniform":
            params["pose_range"] = {}
            params["velocity_range"] = {}
            log.append(f"event '{name}': reset at origin, yaw 0, zero velocity")
        elif fn == "reset_joints_by_scale":
            params["position_range"] = (1.0, 1.0)
            params["velocity_range"] = (0.0, 0.0)
            log.append(f"event '{name}': default joint pose")
        elif fn in ("reset_joints_by_offset", "reset_joints_around_default"):
            params["position_range"] = (0.0, 0.0)
            params["velocity_range"] = (0.0, 0.0)
            log.append(f"event '{name}': default joint pose")
        elif term.mode == "interval":
            setattr(events, name, None)
            log.append(f"removed unknown interval event '{name}' ({fn})")
    return log


def apply_real_events(env_cfg, adapter, robot_mass: float | None) -> list[str]:
    """Add the ``real`` profile perturbations on top of ``clean`` events."""
    from isaaclab.envs import mdp
    from isaaclab.managers import EventTermCfg, SceneEntityCfg

    log = []
    events = env_cfg.events
    for name, term in _event_terms(events):
        if _func_name(term) == "randomize_rigid_body_material":
            term.params.update(REAL_FRICTION)
            log.append(f"event '{name}': friction 0.4-1.2 / 0.3-1.0")
    mass = REAL_MASS_FRACTION * (robot_mass if robot_mass else 20.0)
    events.bench_trunk_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=adapter.spec.base_body),
            "mass_distribution_params": (-mass, mass),
            "operation": "add",
        },
    )
    events.bench_push = EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=REAL_PUSH["interval_range_s"],
        params={"velocity_range": REAL_PUSH["velocity_range"]},
    )
    log.append(f"added trunk mass ±{mass:.2f} kg and pushes ±0.5 m/s every 10-15 s")
    return log


def apply_real_obs_noise(env_cfg) -> list[str]:
    """Manager-based tasks: uniform noise on the policy observation terms in physical units."""
    from isaaclab.utils.noise import AdditiveUniformNoiseCfg

    log = []
    policy = env_cfg.observations.policy
    policy.enable_corruption = True
    for term_name, term in vars(policy).items():
        if not hasattr(term, "noise"):
            continue
        magnitude = REAL_OBS_NOISE.get(term_name)
        term.noise = (
            None
            if magnitude is None
            else AdditiveUniformNoiseCfg(n_min=-magnitude, n_max=magnitude)
        )
        if magnitude is not None:
            log.append(f"obs '{term_name}': ±{magnitude}")
    return log
