"""
Play a trained policy with Keyboard Control.
Includes Real-Time Sync and Infinite Episodes.
"""

import argparse
import time # [!NEW] Required for syncing time
import numpy as np
from isaaclab.app import AppLauncher

# 1. Parse Arguments
parser = argparse.ArgumentParser(description="Keyboard Teleop for Meldog.")
parser.add_argument("--task", type=str, default="Template-Meldog-Simple-Locomotion-Policy-Direct-v0")
parser.add_argument("--num_envs", type=int, default=1) 
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model .pt file")
parser.add_argument("--use_fabric", action="store_true", default=False)
parser.add_argument("--real_time", action="store_true", default=True, help="Run in real-time.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 2. Launch App
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 3. Import Dependencies
import torch
import gymnasium as gym
import carb
import omni.appwindow 

from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

import meldog_simple_locomotion_policy.tasks  

class KeyboardController:
    """Handles keyboard events."""
    def __init__(self):
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._sub_keyboard_event
        )
        self._command = np.array([0.0, 0.0, 0.0])
        self.max_lin_vel = 1.0
        self.max_ang_vel = 1.0
        self._input_keyboard_mapping = {
            carb.input.KeyboardInput.W: np.array([1.0, 0.0, 0.0]),
            carb.input.KeyboardInput.S: np.array([-1.0, 0.0, 0.0]),
            carb.input.KeyboardInput.A: np.array([0.0, 1.0, 0.0]),
            carb.input.KeyboardInput.D: np.array([0.0, -1.0, 0.0]),
            carb.input.KeyboardInput.Q: np.array([0.0, 0.0, 1.0]),
            carb.input.KeyboardInput.E: np.array([0.0, 0.0, -1.0]),
        }

    def _sub_keyboard_event(self, event, *args, **kwargs) -> bool:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input in self._input_keyboard_mapping:
                self._command += self._input_keyboard_mapping[event.input]
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input in self._input_keyboard_mapping:
                self._command -= self._input_keyboard_mapping[event.input]
        
        self._command[:2] = np.clip(self._command[:2], -self.max_lin_vel, self.max_lin_vel)
        self._command[2] = np.clip(self._command[2], -self.max_ang_vel, self.max_ang_vel)
        return True

    def get_command(self):
        return self._command

def main():
    # 4. Load & Setup Environment
    env_cfg = parse_env_cfg(
        args_cli.task, 
        device=args_cli.device, 
        num_envs=args_cli.num_envs
    )
    
    # [!FIX] Settings for Play Mode
    env_cfg.events = None  # Disable Randomization
    env_cfg.episode_length_s = 10000.0 # [!FIX] Infinite Episode (approx 2.7 hours)
    
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

    print(f"[INFO] Creating environment for task: {args_cli.task}")
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # 5. Load Policy
    print(f"[INFO] Loading checkpoint from: {args_cli.checkpoint}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args_cli.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # 6. Initialize Controller
    controller = KeyboardController()

    # 7. Run Loop with Real-Time Sync
    obs = env.get_observations()
    dt = env.unwrapped.step_dt # Get physics step size (e.g. 0.02s)
    
    print("\n" + "="*40)
    print(" KEYBOARD CONTROL ACTIVE")
    print(" W/S: Forward/Back")
    print(" A/D: Strafe Left/Right")
    print(" Q/E: Turn Left/Right")
    print("="*40 + "\n")

    while simulation_app.is_running():
        # [!NEW] Start timer for this frame
        start_time = time.time()
        
        with torch.inference_mode():
            # Control Logic
            cmd_np = controller.get_command()
            cmd_torch = torch.tensor(cmd_np, device=env.unwrapped.device, dtype=torch.float)
            env.unwrapped._commands[:] = cmd_torch

            # Inference & Step
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

        # [!NEW] Real-Time Sync Logic
        if args_cli.real_time:
            # Calculate how long the physics step took
            elapsed = time.time() - start_time
            # Sleep the remainder to match the target DT (0.02s)
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    env.close()

if __name__ == "__main__":
    main()