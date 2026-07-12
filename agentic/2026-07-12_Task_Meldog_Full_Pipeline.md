# Task: Meldog Full Training & Evaluation Pipeline

**Current Phase:** Phase 3 (Dataset Collection)
**Objective:** End-to-end pipeline from Locomotion PPO training through to Perception SLAM generation and evaluation.

## Phase 0: Test Run & Visual Verification (Sanity Checks)
- `[x]` Run `Flat-Sim-v0` (512 envs, short iterations)
- `[x]` Run `FlatObs-Sim-v0` (512 envs, short iterations)
- `[x]` Run `Rough-Sim-v0` (512 envs, short iterations)
- `[x]` Run `RoughObs-Sim-v0` (512 envs, short iterations)
- `[x]` Run `Rough-Real-v0` (512 envs, short iterations) to verify Real (domain randomization) functionality.

## Phase 1: Sequential Locomotion Batch Training
- `[x]` Train `FlatObs-Real-v0` (4096 envs, 3000 iters, headless) - Completed in ~52m (Stable)
- `[x]` Train `Rough-Sim-v0` (4096 envs, 5000 iters, headless) - Completed in ~1h 42m (Stable)
- `[x]` Train `RoughObs-Sim-v0` (4096 envs, 5000 iters, headless) - Completed in ~1h 41m (Stable)
- `[x]` Train `RoughObs-Real-v0` (4096 envs, 5000 iters, headless) - CRASHED at Iteration 3386 (Physics/Gradient Explosion)

## Phase 2: Locomotion Inspection & Cross-Evaluation
- `[x]` Eval native metrics (`evaluate_locomotion.py` - See Experiment Logs for Survival Rates)
- `[x]` Brief visual inspection (`play_locomotion.py`)

## Phase 3: Sequential Perception Dataset Collection
- `[-]` Collect `Dataset-Flat` (Skipped)
- `[x]` Collect `Dataset-FlatObs`
  - Command: `python scripts/perception/collect_dataset.py --task Meldog-RL-Dataset-FlatObs-v0 --num_envs 32 --max_steps 4000`
- `[x]` Collect `Dataset-Rough`
  - Command: `python scripts/perception/collect_dataset.py --task Meldog-RL-Dataset-Rough-v0 --num_envs 32 --max_steps 4000`
- `[/]` Collect `Dataset-RoughObs`
  - Command: `python scripts/perception/collect_dataset.py --task Meldog-RL-Dataset-RoughObs-v0 --num_envs 32 --max_steps 4000`
- `[ ]` Visual verification of `preview.mp4` for each dataset

## Phase 4: Sequential Perception Training
- `[ ]` Train V5 (ConvGRU) and V6 (Autoregressive) on `FlatObs`
- `[ ]` Train V5 and V6 on `Rough`
- `[ ]` Train V5 and V6 on `RoughObs`

## Phase 5: Complete SLAM & Perception Eval
- `[ ]` Evaluate V5/V6 for all terrains (`evaluate_perception.py`)
- `[ ]` Evaluate SLAM baseline for all terrains (`evaluate_slam.py`)
- `[ ]` Test Perception inference over Locomotion in Sim and Real environments

## Open Maintenance Tasks
- `[ ]` Fix `manual.md` to reflect V5/V6 perception naming scheme
