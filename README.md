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

Currently there are three functioning versions of the terrain perception network:

**Input:** Sparse height map (40×40) from 4 depth cameras + gravity vector from IMU  
**Output:** Reconstructed height map (40×40) 

### V1 — Base Model
Standard U-Net encoder-decoder with skip connections. Gravity vector is embedded via MLP and injected at the bottleneck.

### V2 — Deep + Attention
Deeper encoder (additional level down to 5×5) with self-attention mechanism. Larger receptive field allows the network to capture global terrain structure.

### V3 — Temporal
Adds ConvGRU layers that maintain hidden state across frames. The network builds a "belief state" about terrain — remembering previously observed regions that are now occluded.

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
- **`source/meldog_rl/models/perception/`** — Perception architectures (V1 base, V2 attention, V3 temporal)
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

TBD

## Usage
TBD

## Acknowledgments

Built using [Isaac Lab](https://github.com/isaac-sim/IsaacLab) simulation framework and project template.
