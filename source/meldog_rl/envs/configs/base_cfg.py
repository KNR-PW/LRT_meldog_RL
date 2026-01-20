# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Base environment configuration shared across all Meldog tasks.

This module defines the robot, sensors, cameras, and reward scales that are
common to all locomotion and dataset collection tasks.
"""

from __future__ import annotations

import copy
import math
import os

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, TiledCameraCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


# =============================================================================
# USD Path Configuration
# =============================================================================
# Set MELDOG_USD_PATH environment variable or place USD in assets/robots/meldog/
def _get_usd_path() -> str:
    """Get robot USD path with fallback options."""
    # Option 1: Environment variable (highest priority)
    if "MELDOG_USD_PATH" in os.environ:
        path = os.environ["MELDOG_USD_PATH"]
        if os.path.exists(path):
            return path
        print(f"[WARNING] MELDOG_USD_PATH set but file not found: {path}")
    
    # Option 2: Check common locations
    possible_paths = [
        # User's known location
        "/home/frydjak/Downloads/Meldog-1.4-no-ground-plane.usd",
        # Relative to this file
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "assets", "robots", "meldog", "Meldog-1.4-no-ground-plane.usd"),
        # Relative to CWD
        "assets/robots/meldog/Meldog-1.4-no-ground-plane.usd",
    ]
    
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            print(f"[INFO] Found robot USD at: {abs_path}")
            return abs_path
    
    # Fallback - will fail later with clear error
    print("[ERROR] Robot USD not found! Set MELDOG_USD_PATH or copy USD to assets/robots/meldog/")
    return "/home/frydjak/Downloads/Meldog-1.4-no-ground-plane.usd"

MELDOG_USD_PATH = _get_usd_path()


# =============================================================================
# Robot Definition
# =============================================================================
MELDOG_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=MELDOG_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, -0.1),  # Negative because of USD model origin
        joint_pos={
            "LFT_joint": 0.0, "LFH_joint": -0.6, "LFK_joint": 1.3,
            "RFT_joint": 0.0, "RFH_joint": -0.6, "RFK_joint": 1.3,
            "LRT_joint": 0.0, "LRH_joint": -0.6, "LRK_joint": 1.3,
            "RRT_joint": 0.0, "RRH_joint": -0.6, "RRK_joint": 1.3,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "all_joints": DCMotorCfg(
            joint_names_expr=[".*"],
            effort_limit=35.0,
            saturation_effort=35.0,
            stiffness=40.0,
            damping=1.0,
            velocity_limit=18.9,
        ),
    },
)


# =============================================================================
# Domain Randomization - Base (Minimal)
# =============================================================================
@configclass
class BaseEventCfg:
    """Minimal domain randomization for simulation training."""
    
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk_link"),
            "mass_distribution_params": (-1.0, 1.0),
            "operation": "add",
        },
    )


# =============================================================================
# Domain Randomization - Sim2Real (Aggressive)
# =============================================================================
@configclass
class Sim2RealEventCfg(BaseEventCfg):
    """Aggressive domain randomization for sim-to-real transfer."""
    
    # Override with wider ranges
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.2),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.2),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk_link"),
            "mass_distribution_params": (-3.0, 3.0),  # Wider range
            "operation": "add",
        },
    )
    
    # TODO: Add more randomization for sim2real:
    # - Motor strength randomization
    # - Observation noise
    # - Action delay
    # - Joint friction


# =============================================================================
# Camera Definitions
# =============================================================================
_DEPTH_CAMERA_COMMON = TiledCameraCfg(
    prim_path="/World/envs/env_.*/Robot/meldog_core/trunk_link/camera_.*",
    update_period=0.05,
    height=240,
    width=424,
    data_types=["distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=1.93,
        focus_distance=4.0,
        horizontal_aperture=3.8,
        clipping_range=(0.01, 5.0),
        visible=True,
    ),
)

CAMERA_FRONT = copy.deepcopy(_DEPTH_CAMERA_COMMON)
CAMERA_FRONT.prim_path = "/World/envs/env_.*/Robot/meldog_core/trunk_link/camera_front"
CAMERA_FRONT.offset = TiledCameraCfg.OffsetCfg(
    pos=(0.4, 0.0, 0.04),
    rot=(0.9659, 0.0, 0.2588, 0.0),  # Euler (0, 30, 0)
    convention="world",
)

CAMERA_BACK = copy.deepcopy(_DEPTH_CAMERA_COMMON)
CAMERA_BACK.prim_path = "/World/envs/env_.*/Robot/meldog_core/trunk_link/camera_rear"
CAMERA_BACK.offset = TiledCameraCfg.OffsetCfg(
    pos=(-0.4, 0.0, 0.04),
    rot=(0.0, -0.2588, 0.0, 0.9659),  # Euler (0, -30, 180)
    convention="world",
)

CAMERA_LEFT = copy.deepcopy(_DEPTH_CAMERA_COMMON)
CAMERA_LEFT.prim_path = "/World/envs/env_.*/Robot/meldog_core/trunk_link/camera_left"
CAMERA_LEFT.offset = TiledCameraCfg.OffsetCfg(
    pos=(0.0, 0.16, 0.05),
    rot=(0.683, -0.183, 0.183, 0.683),  # Euler (-30, 0, 90)
    convention="world",
)

CAMERA_RIGHT = copy.deepcopy(_DEPTH_CAMERA_COMMON)
CAMERA_RIGHT.prim_path = "/World/envs/env_.*/Robot/meldog_core/trunk_link/camera_right"
CAMERA_RIGHT.offset = TiledCameraCfg.OffsetCfg(
    pos=(0.0, -0.16, 0.05),
    rot=(0.683, 0.183, 0.183, -0.683),  # Euler (30, 0, -90)
    convention="world",
)

CAMERA_TOP = copy.deepcopy(_DEPTH_CAMERA_COMMON)
CAMERA_TOP.prim_path = "/World/envs/env_.*/Robot/meldog_core/trunk_link/camera_top"
CAMERA_TOP.data_types = ["rgb"]
CAMERA_TOP.offset = TiledCameraCfg.OffsetCfg(
    pos=(0.0, 0.0, 2.0),
    rot=(0.7071, 0.0, 0.7071, 0.0),  # Pitch 90 degrees down
    convention="world",
)


# =============================================================================
# Sensor Definitions
# =============================================================================
CONTACT_SENSOR_CFG = ContactSensorCfg(
    prim_path="/World/envs/env_.*/Robot/meldog_core/.*link",
    history_length=3,
    update_period=0.005,
    track_air_time=True,
    force_threshold=0.5,
)

# Height scanner for locomotion policy (17x11 grid = 187 points)
HEIGHT_SCANNER_CFG = RayCasterCfg(
    prim_path="/World/envs/env_.*/Robot/meldog_core/trunk_link",
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    ray_alignment="yaw",
    pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
    debug_vis=False,
    mesh_prim_paths=["/World/ground"],
)

# Ground truth scanner for dataset collection (40x40 grid = 1600 points)
GT_SCANNER_CFG = RayCasterCfg(
    prim_path="/World/envs/env_.*/Robot/meldog_core/trunk_link",
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 10.0)),
    ray_alignment="yaw",
    pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(1.95, 1.95)),
    debug_vis=False,
    mesh_prim_paths=["/World/ground"],
)


# =============================================================================
# Visualization Markers
# =============================================================================
LIN_VEL_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/LinArrow",
    markers={
        "arrow": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
            scale=(0.2, 0.2, 0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    },
)

ANG_VEL_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/AngArrow",
    markers={
        "arrow": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
            scale=(0.2, 0.2, 0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
    },
)

CONTACT_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/ContactMarker",
    markers={
        "sphere": sim_utils.SphereCfg(
            radius=0.1,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), opacity=0.8),
        ),
    },
)


# =============================================================================
# Base Environment Config
# =============================================================================
@configclass
class BaseMeldogEnvCfg(DirectRLEnvCfg):
    """Base configuration shared by all Meldog environments.
    
    Subclasses should override:
    - terrain: Different terrain types
    - events: Different domain randomization
    - Camera configs: Enable/disable cameras
    """
    
    # Environment settings
    episode_length_s = 20.0
    decimation = 4
    action_scale = 0.35
    action_space = 12
    observation_space = 235  # Will be recalculated based on sensors
    state_space = 0
    debug_vis = True
    
    # Simulation settings
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=4,  # Must match decimation
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=sim_utils.PhysxCfg(
            gpu_max_rigid_patch_count=5 * 2**17,
            gpu_max_rigid_contact_count=2**24,
        ),
    )
    
    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=3.0,
        replicate_physics=True,
    )
    
    # Robot
    robot: ArticulationCfg = MELDOG_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    
    # Sensors (always present)
    contact_sensor: ContactSensorCfg = CONTACT_SENSOR_CFG
    height_scanner: RayCasterCfg = HEIGHT_SCANNER_CFG
    gt_scanner: RayCasterCfg = GT_SCANNER_CFG
    
    # Cameras (None = disabled, set in subclass to enable)
    tiled_camera_front: TiledCameraCfg | None = None
    tiled_camera_rear: TiledCameraCfg | None = None
    tiled_camera_left: TiledCameraCfg | None = None
    tiled_camera_right: TiledCameraCfg | None = None
    tiled_camera_top: TiledCameraCfg | None = None
    
    # Visualization markers
    lin_vel_marker: VisualizationMarkersCfg = LIN_VEL_MARKER_CFG
    ang_vel_marker: VisualizationMarkersCfg = ANG_VEL_MARKER_CFG
    contact_marker: VisualizationMarkersCfg = CONTACT_MARKER_CFG
    
    # Domain randomization (override in subclass)
    events: BaseEventCfg = BaseEventCfg()
    
    # Terrain (override in subclass)
    terrain: TerrainImporterCfg = None  # Must be set by subclass
    
    # Gait parameters
    feet_air_time = 0.3
    
    # Reward scales (tuned for Meldog)
    lin_vel_reward_scale = 1.5
    yaw_rate_reward_scale = 0.7
    z_vel_reward_scale = -2.0
    ang_vel_reward_scale = -0.05
    joint_torque_reward_scale = -1.0e-4
    joint_accel_reward_scale = -2.5e-7
    action_rate_reward_scale = -0.01
    feet_air_time_reward_scale = 0.3
    undesired_contact_reward_scale = -1.0
    flat_orientation_reward_scale = -0.0
