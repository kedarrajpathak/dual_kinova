# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to run the RL environment for the cartpole balancing task.

.. code-block:: bash

    ./isaaclab.sh -p scripts/tutorials/03_envs/run_dual_kinova_rl_env.py --task Isaac-Dual-Kinova-v0 --num_envs 1 --enable_cameras

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on running the cartpole RL environment.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")

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
from isaaclab_tasks.utils import parse_env_cfg



def main():
    """Main function."""
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment configuration
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    # env_cfg = DualKinovaRLEnvCfg()
    # env_cfg.scene.num_envs = args_cli.num_envs
    # # setup RL environment
    # env = ManagerBasedRLEnv(cfg=env_cfg)

    # left_arm_pose = [0.5, -0.5, 0.5, 0.0, 1.0, 0.0, 0.0] # position:xyz orientation:wxyz
    # left_gripper_pose = [1.0]
    # right_arm_pose = [0.5, 0.5, 0.5, 0.0, 1.0, 0.0, 0.0]
    # right_gripper_pose = [1.0]
    # actions_list = left_arm_pose + left_gripper_pose + right_arm_pose + right_gripper_pose
    left_arm_pose = [0.5, -0.5, 0.5, 0.0, 1.0, 0.0, 0.0] # position:xyz orientation:wxyz
    left_gripper_pose = [0.78]
    right_arm_pose = [0.5, 0.5, 0.5, 0.0, 1.0, 0.0, 0.0]
    right_gripper_pose = [0.78]
    actions_list = left_arm_pose + left_gripper_pose + right_arm_pose + right_gripper_pose
    
    actions = torch.tensor(actions_list, dtype=torch.float32, device=env.device).repeat(env.num_envs, 1)
    
    print("actions: ", actions)
            
    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 300 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
                
            # step the environment
            obs, rew, terminated, truncated, info = env.step(actions)
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
