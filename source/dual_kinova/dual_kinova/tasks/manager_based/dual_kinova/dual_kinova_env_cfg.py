# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import torch
from torchvision.utils import save_image
import math

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, RigidObject
from isaaclab.sensors import Camera, CameraCfg, RayCasterCamera, TiledCameraCfg, TiledCamera
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.lift import mdp

##
# Pre-defined configs
##

from isaaclab_assets.robots.kinova import DUAL_KINOVA_CFG  # isort:skip

##
# Scene definition
##


@configclass
class DualKinovaSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # robot
    robot: ArticulationCfg = DUAL_KINOVA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/cam",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.5), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )
    
    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    # cube
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 1.0, 0.1), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),   
        )
    )
##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

        # Set actions for the specific robot type (franka)
    left_arm = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["left_joint_.*"],
        body_name="left_end_effector_link",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        scale=0.5,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.15)),
    )
    left_gripper = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["left_robotiq_85_left_knuckle_joint"],
            open_command_expr={"left_robotiq_85_left_knuckle_joint": 0.78},
            close_command_expr={"left_robotiq_85_left_knuckle_joint": 0.0},
        )
    right_arm = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["right_joint_.*"],
        body_name="right_end_effector_link",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        scale=0.5,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.15)),
    )
    right_gripper = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["right_robotiq_85_left_knuckle_joint"],
            open_command_expr={"right_robotiq_85_left_knuckle_joint": 0.78},
            close_command_expr={"right_robotiq_85_left_knuckle_joint": 0.0},
        )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        """Observations for policy group with RGB images."""

        rgb = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("camera"),
                "data_type": "rgb",
                "normalize": True,
            },
        )
        
        depth = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("camera"),
                "data_type": "distance_to_image_plane",
                "normalize": True,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    rgbd_camera: RGBCameraPolicyCfg = RGBCameraPolicyCfg()

@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # # (3) Primary task: keep pole upright
    # pole_pos = RewTerm(
    #     func=mdp.joint_pos_target_l2,
    #     weight=-1.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]), "target": 0.0},
    # )
    # # (4) Shaping tasks: lower cart velocity
    # cart_vel = RewTerm(
    #     func=mdp.joint_vel_l1,
    #     weight=-0.01,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
    # )
    # # (5) Shaping tasks: lower pole angular velocity
    # pole_vel = RewTerm(
    #     func=mdp.joint_vel_l1,
    #     weight=-0.005,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"])},
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Cart out of bounds
    # cart_out_of_bounds = DoneTerm(
    #     func=mdp.joint_pos_out_of_manual_limit,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]), "bounds": (-3.0, 3.0)},
    # )
    # (3) Velocity out of bounds
    velocity_out_of_bounds = DoneTerm(
        func=mdp.joint_vel_out_of_limit,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])}
    )
    

##
# Environment configuration
##


@configclass
class DualKinovaRLEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: DualKinovaSceneCfg = DualKinovaSceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 10
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation