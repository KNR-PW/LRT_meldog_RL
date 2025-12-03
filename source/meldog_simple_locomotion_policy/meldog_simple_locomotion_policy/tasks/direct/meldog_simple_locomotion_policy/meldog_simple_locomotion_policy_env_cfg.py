# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR 

# Imports for Visualization
from isaaclab.markers import VisualizationMarkersCfg 

from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG

##
# Robot Definition
##
MELDOG_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/ubuntu/Downloads/Meldog-1.4-no-ground-plane.usd",
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
            enabled_self_collisions=False, # self colisions are unreliable
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Spawn height negative, might be related to created USD and its spawn height
        pos=(0.0, 0.0, -0.12), 
        
        # Taller Stance to prevent immediate collapse
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
        )
    },
)

##
# Domain Randomization
##
@configclass
class EventCfg:
    """Configuration for randomization."""
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.25), 
            "dynamic_friction_range": (0.6, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk_link"),
            "mass_distribution_params": (-2.0, 3.0),
            "operation": "add",
        },
    )

##
# Env Config
##
@configclass
class MeldogSimpleLocomotionPolicyEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 4
    action_scale = 0.5
    action_space = 12
    observation_space = 235
    state_space = 0
    debug_vis = True 

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200, 
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # Terrain - Rough Terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=9,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,

    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=True)

    # events
    events: EventCfg = EventCfg()

    # robot
    robot: ArticulationCfg = MELDOG_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    
    # sensors
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/meldog_core/.*link", 
        history_length=3, 
        update_period=0.005, 
        track_air_time=True
    )

    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/meldog_core/trunk_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # Visualization Markers
    lin_vel_marker: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/LinArrow",
        markers={
            "arrow": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.2, 0.2, 0.5),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)), # RED
            ),
        },
    )
    
    ang_vel_marker: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/AngArrow",
        markers={
            "arrow": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.2, 0.2, 0.5),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)), # GREEN
            ),
        },
    )

    contact_marker: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/ContactMarker",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.1, # red ball
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), opacity=0.8),
            ),
        },
    )

    target_base_height = 0.35

    # Reward Scales
    alive_reward_scale = 1.0 * 0

    lin_vel_reward_scale = 1.0
    yaw_rate_reward_scale = 0.5
    z_vel_reward_scale = -2.0
    ang_vel_reward_scale = -0.05
    
    joint_torque_reward_scale = -2.5e-6 
    joint_accel_reward_scale = -2.0e-8  
    action_rate_reward_scale = -0.005
    action_accel_reward_scale = -0.0025

    
    feet_air_time_reward_scale = 0.5
    undesired_contact_reward_scale = -1.0
    flat_orientation_reward_scale = -0.3

    base_height_reward_scale = -1.0 *0
    joint_deviation_reward_scale = -0.1