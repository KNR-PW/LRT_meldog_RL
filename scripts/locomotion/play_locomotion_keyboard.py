#!/usr/bin/env python3
# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Play trained policy with keyboard control.

Includes real-time sync and infinite episodes.

Usage:
    python scripts/locomotion/play_locomotion_keyboard.py \
        --task Meldog-RL-Locomotion-Rough-Sim-v0 \
        --checkpoint logs/locomotion/LM_rough_sim_.../model_5000.pt

Controls:
    W/S: Forward/Backward
    A/D: Strafe Left/Right
    Q/E: Turn Left/Right
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "source"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Keyboard teleop for Meldog.")
parser.add_argument("--task", type=str, default="Meldog-RL-Locomotion-Rough-Sim-v0", help="Task name.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--real_time", action="store_true", default=True, help="Run in real-time.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ============================================================================

import numpy as np
import torch
import gymnasium as gym
import carb
import omni.appwindow

from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import meldog_rl
from meldog_rl import envs  # This registers the tasks
from meldog_rl import agents


class KeyboardController:
    """Handles keyboard events for velocity commands."""
    
    def __init__(self, max_lin_vel: float = 1.0, max_ang_vel: float = 1.0):
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._sub_keyboard_event
        )
        
        self._command = np.array([0.0, 0.0, 0.0])
        self.max_lin_vel = max_lin_vel
        self.max_ang_vel = max_ang_vel
        
        # Key mappings: [vx, vy, yaw_rate]
        self._key_mapping = {
            carb.input.KeyboardInput.W: np.array([1.0, 0.0, 0.0]),   # Forward
            carb.input.KeyboardInput.S: np.array([-1.0, 0.0, 0.0]),  # Backward
            carb.input.KeyboardInput.A: np.array([0.0, 1.0, 0.0]),   # Strafe left
            carb.input.KeyboardInput.D: np.array([0.0, -1.0, 0.0]),  # Strafe right
            carb.input.KeyboardInput.Q: np.array([0.0, 0.0, 1.0]),   # Turn left
            carb.input.KeyboardInput.E: np.array([0.0, 0.0, -1.0]),  # Turn right
        }

    def _sub_keyboard_event(self, event, *args, **kwargs) -> bool:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input in self._key_mapping:
                self._command += self._key_mapping[event.input]
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input in self._key_mapping:
                self._command -= self._key_mapping[event.input]
        
        # Clip to max velocities
        self._command[:2] = np.clip(self._command[:2], -self.max_lin_vel, self.max_lin_vel)
        self._command[2] = np.clip(self._command[2], -self.max_ang_vel, self.max_ang_vel)
        return True

    def get_command(self) -> np.ndarray:
        """Get current velocity command."""
        return self._command.copy()


def main():
    """Run keyboard-controlled policy."""
    
    # Get configs
    env_cfg = gym.spec(args_cli.task).kwargs["env_cfg_entry_point"]()  # Instantiate!
    agent_cfg = gym.spec(args_cli.task).kwargs["rsl_rl_cfg_entry_point"]()
    
    # Configure for play mode
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device else "cuda:0"
    env_cfg.events = None  # Disable randomization
    env_cfg.episode_length_s = 10000.0  # ~2.7 hours (infinite)
    
    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    
    # Load policy
    print(f"[INFO] Loading checkpoint: {args_cli.checkpoint}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    
    # Initialize controller
    controller = KeyboardController()
    
    # Get simulation timestep for real-time sync
    dt = env.unwrapped.step_dt
    
    print("\n" + "=" * 40)
    print(" KEYBOARD CONTROL ACTIVE")
    print(" W/S: Forward/Backward")
    print(" A/D: Strafe Left/Right")
    print(" Q/E: Turn Left/Right")
    print("=" * 40 + "\n")
    
    # Run loop
    obs = env.get_observations()
    
    while simulation_app.is_running():
        start_time = time.time()
        
        with torch.inference_mode():
            # Update commands from keyboard
            cmd_np = controller.get_command()
            cmd_torch = torch.tensor(cmd_np, device=env.unwrapped.device, dtype=torch.float)
            env.unwrapped._commands[:] = cmd_torch
            
            # Run policy
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
        
        # Real-time sync
        if args_cli.real_time:
            elapsed = time.time() - start_time
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
