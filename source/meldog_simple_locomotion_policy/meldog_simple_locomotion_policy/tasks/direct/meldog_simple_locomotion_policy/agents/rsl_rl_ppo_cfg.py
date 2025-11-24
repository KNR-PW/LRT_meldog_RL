# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24  # Increased from 16: Gives PPO more horizon to estimate value
    max_iterations = 1500   # Increased from 500: 500 is rarely enough for a clean walk
    save_interval = 50
    experiment_name = "Meldog_simple_locomotion"
    
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False, # RSL_RL handles norm internally usually, but False is fine if you norm inputs
        critic_obs_normalization=False,
        # A tapered structure captures complex dynamics better than flat [128,128,128]
        actor_hidden_dims=[512, 256, 128], 
        critic_hidden_dims=[512, 256, 128],
        activation="elu", # ELU is smoother than ReLU, better for motors
    )
    
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01, # Slightly higher initial exploration helps prevent "standing still"
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )