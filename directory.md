# Meldog RL — Directory & Path Map

Curated map of the paths agents need: the local repo **and** the external Isaac Lab
reference trees (which agents cannot find by searching this repo). Keep this curated,
not exhaustive — update it when structure changes.

## Local repo — root `/home/frydjak/KNR/Meldog/LRT_meldog_RL`

### Env + configs
- `source/meldog_rl/envs/meldog_env.py` — direct RL env (12-DOF quadruped: 4 legs × T/H/K)
- `source/meldog_rl/envs/__init__.py` — gym task registrations (ids below)
- `source/meldog_rl/envs/configs/base_cfg.py` — shared base config
- `source/meldog_rl/envs/configs/simulation/{flat,flat_obs,rough,rough_obs}_cfg.py`
- `source/meldog_rl/envs/configs/sim2real/*` — domain-randomized "Real" variants
- `source/meldog_rl/envs/configs/dataset/*` — perception dataset-collection envs
- `source/meldog_rl/envs/configs/simulation/terrain_utils.py`
- `source/meldog_rl/agents/rsl_rl_ppo_cfg.py` — PPO hyperparameters

### Perception
- `source/meldog_rl/models/perception/heightmap_convgru.py` — V5 (ConvGRU)
- `source/meldog_rl/models/perception/heightmap_autoreg.py` — V6 (autoregressive)
- `source/meldog_rl/models/perception/slam_baseline.py` — SLAM baseline
- `source/meldog_rl/models/perception/{projector,voxel_sparse,common}.py`

### Scripts
- `scripts/locomotion/{train,play,play_keyboard,evaluate}_locomotion.py`
- `scripts/perception/{collect_dataset,train_perception,evaluate_perception}.py`
- `scripts/perception/{evaluate_slam,debug_cameras}.py`
- `scripts/release_model.py` — promote a trained run into `releases/`

### Data / outputs / docs
- `logs/locomotion/`, `logs/perception/` — training runs
- `releases/locomotion/` — promoted checkpoints (`v1.0-rough`, `v1.0-rough-baseline`)
- `datasets/` — collected perception datasets
- `thesis/` — LaTeX thesis (`chapters/`, `images/`, `eiti/`)
- `literature/` — reference papers
- `legacy code structure/` — pre-cleanup layout; do **not** extend
- `AGENTS.md` — agent source of truth · `agentic/prompts.md` — prompts · `agentic/manual.md` — user manual

## Registered task ids
- Locomotion: `Meldog-RL-Locomotion-{Flat,Rough,FlatObs,RoughObs}-{Sim,Real}-v0`
- Dataset: `Meldog-RL-Dataset-{Flat,Rough,FlatObs,RoughObs}-v0`

## External reference — Isaac Lab · root `/home/frydjak/IsaacLab`

Manager-based velocity locomotion, per-robot config base:
`source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/<robot>/`
where `<robot>` ∈ {`anymal_b`, `anymal_c`, `anymal_d`, `go1`, `go2`}. Each has:
- `flat_env_cfg.py` — flat-terrain env config
- `rough_env_cfg.py` — rough-terrain env config (rewards, obs, terrain, curriculum)
- `agents/rsl_rl_ppo_cfg.py` — PPO hyperparameters

Primary references for our tuning:
- `.../config/anymal_c/rough_env_cfg.py` — reward terms incl. `feet_slide`, curriculum
- `.../config/anymal_c/flat_env_cfg.py`
- `.../config/anymal_c/agents/rsl_rl_ppo_cfg.py`
- `.../config/go2/rough_env_cfg.py` — closer scale/dynamics to Meldog
- `.../config/go2/flat_env_cfg.py`
- `.../config/go2/agents/rsl_rl_ppo_cfg.py`

Direct-workflow ANYmal (matches our direct env style, not manager-based):
- `source/isaaclab_tasks/isaaclab_tasks/direct/anymal_c/anymal_c_env_cfg.py`
- `source/isaaclab_tasks/isaaclab_tasks/direct/anymal_c/agents/rsl_rl_ppo_cfg.py`
