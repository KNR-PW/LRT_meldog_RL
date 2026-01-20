# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO agent configuration for Meldog."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class MeldogPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner configuration for Meldog locomotion.
    
    Based on ANYmal rough locomotion settings.
    """
    
    # Runner settings
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 100  # Save every 100 iterations (was 500)
    experiment_name = "meldog_rl_locomotion"
    empirical_normalization = False
    
    # These are required by RSL-RL
    run_name = ""
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"
    
    # Clip actions to [-1, 1] range
    clip_actions = 1.0
    
    policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    
    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
