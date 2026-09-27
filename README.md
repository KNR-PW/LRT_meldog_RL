# Meldog — RL Perception and Locomotion

End-to-end framework for developing locomotion and perception policies for a quadruped robot Meldog using reinforcement learning and supervised learning.

## Demo

### Locomotion Policy
https://github.com/user-attachments/assets/dd9ced4d-5dec-434a-a274-5dd069e8d532

### Terrain Perception
https://github.com/user-attachments/assets/db0f350c-b0d0-448f-b501-6933fc37bb87


## Overview

This project provides a complete pipeline for training  locomotion and perception:

1. **Train locomotion policy** — RL-based controller (PPO) that learns to walk on varied terrain
2. **Collect perception dataset** — use trained locomotion to gather depth camera observations with ground truth height maps
3. **Train perception model** — supervised learning to reconstruct dense terrain from sparse observations

## Perception Models

**Input:** Sparse height map (40×40) from 4 depth cameras + gravity vector from IMU  
**Output:** Reconstructed height map (40×40) 

The terrain perception network comes in three versions, named as in the accompanying
master's thesis:

- **v1 — shallow:** U-Net encoder-decoder with skip connections, 2 encoder stages down to
  10×10. The gravity vector is embedded by an MLP and injected at the bottleneck.
- **v2 — deep:** one more encoder stage, down to 5×5, for a larger receptive field.
- **v3 — temporal:** v2 plus ConvGRU layers at the bottleneck that keep a hidden state
  across frames, a "belief state" that remembers regions the robot can no longer see.

Older code-era models are kept under archived names. The full list, with training
support and parameter counts, is in [docs/manual.md §2](docs/manual.md#2-perception-models).

## Project Structure

Built on Isaac Lab's direct workflow template. Here's where key components live:

### Training Pipelines
- **`scripts/locomotion/`** — RL training, evaluation, and testing for locomotion policies
- **`scripts/perception/`** — Supervised learning pipeline for terrain perception (training, evaluation, dataset collection)

### Robot Description
- **`source/meldog_rl/envs/meldog_env.py`** — Main environment implementation defining robot physics, observations, rewards
- **`source/meldog_rl/envs/configs/`** — Environment configurations organized by use case:
  - `simulation/` — Training configs (flat/rough terrain)
  - `sim2real/` — Real-world deployment configs with domain randomization
  - `dataset/` — Dataset collection configs with camera observations enabled

### Neural Network Models
- **`source/meldog_rl/models/perception/`** — Perception architectures (v1, v2, v3 and archived models); `registry.py` lists them all
- **`source/meldog_rl/agents/`** — Locomotion policy configuration (PPO hyperparameters, network architecture)

### Data
- **`logs/locomotion/`** — RL training checkpoints and TensorBoard logs
- **`logs/perception/`** — Perception model checkpoints and training metrics
- **`datasets/`** — Collected depth camera observations with ground truth height maps

### Supporting Code
- **`source/meldog_rl/datasets/`** — PyTorch dataset classes for perception training
- **`source/meldog_rl/utils/`** — Shared utilities and naming conventions

## Installation

This project is built on the [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) template.
Setup (Isaac Lab, editable install, Git LFS for model weights): see [docs/manual.md](docs/manual.md#0-setup).

## Usage
- [docs/manual.md](docs/manual.md): tasks, training, evaluation, perception pipeline, releases, analyzers
- [docs/evaluation.md](docs/evaluation.md): evaluation metrics and their reference ranges
- [docs/eval_rubric.md](docs/eval_rubric.md): checklist for judging videos

## Acknowledgments

Built using [Isaac Lab](https://github.com/isaac-sim/IsaacLab) simulation framework and project template.
