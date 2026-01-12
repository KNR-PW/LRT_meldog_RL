# MelDog — Perceptive Locomotion for Quadruped Robot

End-to-end framework for developing locomotion and perception policies for a quadruped robot using reinforcement learning and supervised learning.

## Demo

### Locomotion Policy
https://github.com/user-attachments/assets/dd9ced4d-5dec-434a-a274-5dd069e8d532

### Terrain Perception
https://github.com/user-attachments/assets/db0f350c-b0d0-448f-b501-6933fc37bb87


## Overview

This project provides a complete pipeline for training perceptive locomotion:

1. **Train locomotion policy** — RL-based controller (PPO) that learns to walk on varied terrain
2. **Collect perception dataset** — use trained locomotion to gather depth camera observations with ground truth height maps
3. **Train perception model** — supervised learning to reconstruct dense terrain from sparse observations

## Perception Models

Currently there are three functioning versions of the terrain perception network:

| Model | Architecture | Key Feature | Parameters |
|-------|-------------|-------------|------------|
| **V1 Base** | 2-level U-Net | Gravity conditioning | ~130K |
| **V2 Deep** | 3-level U-Net | Self-attention in bottleneck | ~590K |
| **V3 Temporal** | 3-level U-Net + ConvGRU | Temporal memory across frames | ~850K |

**Input:** Sparse height map (40×40) from 4 depth cameras + gravity vector from IMU  
**Output:** Dense height map (40×40) with filled occlusions

### V1 — Base Model
Standard U-Net encoder-decoder with skip connections. Gravity vector is embedded via MLP and injected at the bottleneck.

### V2 — Deep + Attention
Deeper encoder (additional level down to 5×5) with self-attention mechanism. Larger receptive field allows the network to capture global terrain structure.

### V3 — Temporal
Adds ConvGRU layers that maintain hidden state across frames. The network builds a "belief state" about terrain — remembering previously observed regions that are now occluded.

## Project Structure
This project uses Isaac Lab direct workflow project template. My work mostly consists of these files:
```
├── scripts/
│   ├── train_locomotion.py
│   ├── play_locomotion.py
│   ├── play_locomotion_keyboard.py
│   └── evaluate_locomotion.py
├── source/
│   └── meldog_simple_locomotion_policy/
│       ├── meldog_simple_locomotion_policy_env.py
│       └── meldog_simple_locomotion_policy_env_cfg.py
└── README.md
```

## Installation

This project is built on the [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) template.

1. Install Isaac Lab following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)

2. Activate the conda environment:
   ```bash
   conda activate isaaclab
   ```

3. Clone this repository and install in editable mode:
   ```bash
   ./isaaclab.sh -p -m pip install -e source/meldog_simple_locomotion_policy
   ```

4. Verify installation:
   ```bash
   ./isaaclab.sh -p scripts/list_envs.py
   ```

## Usage

All scripts are launched via `./isaaclab.sh -p` from the Isaac Lab directory with the `isaaclab` conda environment active.

### Train locomotion policy
```bash
./isaaclab.sh -p scripts/rsl_rl/train.py --task=MelDog-Locomotion-v0
```

### Run trained policy
```bash
./isaaclab.sh -p scripts/play_locomotion.py --checkpoint=path/to/model.pt
```

### Evaluate with keyboard control
```bash
./isaaclab.sh -p scripts/play_locomotion_keyboard.py
```

### Collect perception dataset
```bash
./isaaclab.sh -p scripts/collect_perception_data.py --checkpoint=path/to/locomotion.pt --output=dataset/
```

### Train perception model
```bash
python train_perception.py --data=dataset/ --model=v3_temporal
```

## Acknowledgments

Built using [Isaac Lab](https://github.com/isaac-sim/IsaacLab) simulation framework and project template.
