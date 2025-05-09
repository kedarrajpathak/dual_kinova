# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to run the RL environment for the cartpole balancing task.

.. code-block:: bash

    ./isaaclab.sh -p scripts/tutorials/03_envs/run_dual_kinova_rl_env.py --task Isaac-Dual-Kinova-IK-v0 --num_envs 1 --enable_cameras -k calibration_matrix.npy -d distortion_coefficients.npy -t DICT_4X4_50 -l 0.053

"""

"""Launch Isaac Sim Simulator first."""

import cv2
import sys
import numpy as np
import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on running the cartpole RL environment.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# parser.add_argument("-k", "--K_Matrix", required=True, help="Path to calibration matrix (numpy file)")
# parser.add_argument("-d", "--D_Coeff", required=True, help="Path to distortion coefficients (numpy file)")
# parser.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="Type of ArUCo tag to detect")
# parser.add_argument("-l", "--length", type=float, default=0.1, help="Length of the marker's side in meters")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_tasks.manager_based.manipulation.dual_kinova.dual_kinova_ik_env_cfg import DualKinovaRLEnvCfg
import isaaclab.utils.math as math_utils
from isaaclab_tasks.utils import parse_env_cfg

from pose_estimation import pose_esitmation
from cube_tracking import KalmanFilter3D, ArucoCubeTracking

aruco_cube_tracker = ArucoCubeTracking(calibration_matrix_path="/workspace/isaaclab/scripts/tutorials/03_envs/calibration_matrix.npy",
                                       distortion_coefficients_path="/workspace/isaaclab/scripts/tutorials/03_envs/distortion_coefficients.npy")

def process_aruco_poses(rvec, tvec, idx):
    quats = []
    for r in rvec:
        quat = math_utils.quat_from_euler_xyz(roll=torch.tensor([r[0]],device="cuda:0"), 
                                              pitch=torch.tensor([r[1]],device="cuda:0"), 
                                              yaw=torch.tensor([r[2]],device="cuda:0"))
        quats.append(quat[0])
    left_arm_pose = torch.concatenate((torch.tensor(tvec[0], dtype=torch.float32, device="cuda:0"),quats[0]))
    right_arm_pose = torch.concatenate((torch.tensor(tvec[1], dtype=torch.float32, device="cuda:0"),quats[1]))
    left_gripper_pose = torch.tensor([0.78], dtype=torch.float32, device="cuda:0")
    right_gripper_pose = torch.tensor([0.78], dtype=torch.float32, device="cuda:0")
    actions_list = torch.concatenate((left_arm_pose, left_gripper_pose, right_arm_pose, right_gripper_pose))
    actions = torch.tensor(actions_list, dtype=torch.float32, device="cuda:0").unsqueeze(0)
    return actions

def main():
    """Main function."""
    
    # Load camera calibration data
    calibration_matrix_path = "/workspace/isaaclab/scripts/tutorials/03_envs/calibration_matrix.npy"  # Update with the correct path
    distortion_coefficients_path = "/workspace/isaaclab/scripts/tutorials/03_envs/distortion_coefficients.npy"  # Update with the correct path
    k = np.load(calibration_matrix_path)
    d = np.load(distortion_coefficients_path)

    # Set ArUco marker type and marker length
    aruco_dict_type = cv2.aruco.DICT_4X4_50  # Update with the desired ArUco dictionary
    marker_length = 0.053  # Length of the marker's side in meters

    # Initialize webcam
    video = cv2.VideoCapture(0)

    """Main function."""
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment configuration
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    left_arm_pose = [0.5, -0.5, 0.5, 0.0, 1.0, 0.0, 0.0] # position:xyz orientation:wxyz
    left_gripper_pose = [1.0]
    right_arm_pose = [0.5, 0.5, 0.5, 0.0, 1.0, 0.0, 0.0]
    right_gripper_pose = [1.0]
    actions_list = left_arm_pose + left_gripper_pose + right_arm_pose + right_gripper_pose
    # left_arm_pose = [0.01, 0.0, 0.0, 0.0, 0.05, 0.0] # position:xyz orientation:wxyz
    # left_gripper_pose = [0.78]
    # right_arm_pose = [0.01, 0.0, 0.0, 0.0, 0.05, 0.0]
    # right_gripper_pose = [0.78]
    # actions_list = left_arm_pose + left_gripper_pose + right_arm_pose + right_gripper_pose
    
    actions = torch.tensor(actions_list, dtype=torch.float32, device=env.device).repeat(env.num_envs, 1)
    
    print("actions: ", actions)
    left_cube = ([0.5, -0.5, 0.5], [0.0, 1.0, 0.0, 0.0])
    right_cube = ([0.5, 0.5, 0.5], [0.0, 1.0, 0.0, 0.0])
            
    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # # Capture frame from webcam
            # ret, frame = video.read()
            # if not ret:
            #     print("[ERROR]: Unable to capture video frame.")
            #     break

            # reset
            if count % 1000 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
                
            # # Estimate poses of ArUco markers
            # _, rvec, tvec, idx = pose_esitmation(frame, aruco_dict_type, marker_length, k, d)
            output_frame, left_cube, right_cube = aruco_cube_tracker.process_frame()
            left_pos, left_quat = left_cube
            right_pos, right_quat = right_cube
            # actions_cube = torch.tensor([left_pos[0], left_pos[1], left_pos[2], left_quat[0], left_quat[1], left_quat[2], left_quat[3],
            #                              0.78, right_pos[0], right_pos[1], right_pos[2], right_quat[0], right_quat[1], right_quat[2], right_quat[3],
            #                              0.78], dtype=torch.float32, device="cuda:0").unsqueeze(0)
            cv2.imshow("Output Frame", output_frame)
            cv2.waitKey(1)
            actions_cube = torch.tensor([1-left_pos[2], left_pos[0]*3, -left_pos[1]+0.3, 0.0, 1.0, 0.0, 0.0,
                                         0.78, 1-right_pos[2], right_pos[0]*3, -right_pos[1]+0.3, 0.0, 1.0, 0.0, 0.0,
                                         0.78], dtype=torch.float32, device="cuda:0").unsqueeze(0)
            print(f"actions_cube: {actions_cube[0][:3]}, {actions_cube[0][7:10]}")
            obs, rew, terminated, truncated, info = env.step(actions_cube)
            
            # if len(idx) == 2:
            #     actions_aruco = process_aruco_poses(rvec, tvec, idx)
            #     print(f"actions_aruco: {actions_aruco}")
            #     # step the environment
            #     obs, rew, terminated, truncated, info = env.step(actions_aruco)
            
            # else:
            #     # step the environment
            #     obs, rew, terminated, truncated, info = env.step(actions)
                
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
