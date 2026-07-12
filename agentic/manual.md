# Meldog RL: Complete User Manual

This document provides a comprehensive overview of the refactored Meldog framework, detailing the supported environment combinations, perception models, script types, and CLI usage examples.

## 1. Locomotion Environment Combinations

The framework supports 12 distinct Locomotion Environment pipelines based on **4 Terrains** and **3 Target Domains**.

| Terrain Focus | Sim (Fast Training) | Real (Sim2Real Domain Rand) | Dataset (Camera Data Collection) |
| :--- | :--- | :--- | :--- |
| **Flat** | `Meldog-RL-Locomotion-Flat-Sim-v0` | `Meldog-RL-Locomotion-Flat-Real-v0` | `Meldog-RL-Dataset-Flat-v0` |
| **Rough** | `Meldog-RL-Locomotion-Rough-Sim-v0` | `Meldog-RL-Locomotion-Rough-Real-v0` | `Meldog-RL-Dataset-Rough-v0` |
| **Flat + Obs** | `Meldog-RL-Locomotion-FlatObs-Sim-v0` | `Meldog-RL-Locomotion-FlatObs-Real-v0` | `Meldog-RL-Dataset-FlatObs-v0` |
| **Rough + Obs** | `Meldog-RL-Locomotion-RoughObs-Sim-v0` | `Meldog-RL-Locomotion-RoughObs-Real-v0` | `Meldog-RL-Dataset-RoughObs-v0` |

*Note: The RL Algorithm is always PPO (via RSL-RL), but the hyperparameters automatically switch depending on whether the terrain is Flat or Rough.*

## 2. Perception Model Combinations

Perception is decoupled from Locomotion. Once a locomotion policy is trained, you run it in a `Dataset` environment to collect data, which is then used to train one of these **3 Perception Architectures**:

| Model Architecture | Code Implementation | Description |
| :--- | :--- | :--- |
| **V5 Temporal** | `heightmap_v5` | Adds memory states (ConvGRU) to build a "belief state" across multiple frames. |
| **V6 Autoregressive** | `heightmap_v6` | Standard U-Net / Deep Attention network for sequential coordinate reconstruction. |
| **Baseline** | `SLAMBaseline` | Non-learned geometric baseline for comparison. |

---

## 3. Locomotion CLI Usage & Script Types

There are four different ways to run and evaluate your locomotion policies.

### A. Training (`train_locomotion.py`)
Trains the RL policy using PPO. Runs highly parallelized environments.
> [!TIP]
> Use `--headless` when training to significantly increase FPS.
```bash
python scripts/locomotion/train_locomotion.py \
    --task Meldog-RL-Locomotion-Rough-Sim-v0 \
    --num_envs 4096 \
    --max_iterations 5000 \
    --headless
```

### B. Visual Evaluation (`play_locomotion.py`)
Often just called "Eval" or "Play". Runs the policy visually with rendering enabled so you can watch the robot behave.
```bash
python scripts/locomotion/play_locomotion.py \
    --task Meldog-RL-Locomotion-Rough-Sim-v0 \
    --checkpoint logs/locomotion/LM_rough_sim_2026-02-06_14-23-29/model_5000.pt
```

### C. Keyboard Evaluation (`play_locomotion_keyboard.py`)
Similar to visual evaluation, but instead of the robot choosing random velocity commands, it binds your keyboard (WASD) to the velocity targets so you can manually drive the robot around the terrain.
```bash
python scripts/locomotion/play_locomotion_keyboard.py \
    --task Meldog-RL-Locomotion-Flat-Sim-v0 \
    --checkpoint logs/locomotion/LM_flat_sim_.../model_2000.pt
```

### D. Metrics Evaluation (`evaluate_locomotion.py`)
This is the **"Statistics/Metrics Eval"**. Instead of visual rendering, this runs the policy headless across hundreds of episodes to calculate statistical robustness. It prints a report showing the **Survival Rate**, number of crashes, and average episode length.
```bash
python scripts/locomotion/evaluate_locomotion.py \
    --task Meldog-RL-Locomotion-Rough-Sim-v0 \
    --checkpoint logs/locomotion/LM_rough_sim_.../model_5000.pt \
    --num_episodes 100 \
    --headless
```

---

## 4. Perception CLI Usage & Script Types

### A. Dataset Collection (`collect_dataset.py`)
Uses a pre-trained locomotion policy to walk around and record depth camera frames + ground truth heightmaps.
```bash
python scripts/perception/collect_dataset.py \
    --task Meldog-RL-Dataset-Rough-v0 \
    --checkpoint logs/locomotion/LM_rough_sim_.../model_5000.pt \
    --num_samples 10000
```

### B. Perception Training (`train_perception.py`)
Trains the selected Perception model offline using the H5 dataset collected in step A.
```bash
python scripts/perception/train_perception.py \
    --dataset datasets/PD_rough_2026-07-12_13-56-35_6e1677d-dirty/ \
    --model heightmap_v5 \
    --epochs 100 \
    --workers 8 \
    --batch_size 80 \
    --seq_len 32
```
> [!WARNING]
> **Hardware Limits:** PyTorch multiprocessing overhead consumes massive amounts of System RAM. A configuration of `8 workers` and `80 batch size` (with `seq_len=32`) is highly stable and maximizes GPU utilization, but it hovers dangerously close to ~48GB of RAM. Machines with less than 64GB RAM may experience freezing or OOM crashes, particularly during the inter-epoch validation transition. If you experience crashes, drop to `4 workers` and `160 batch size`.

### C. SLAM Baseline Evaluation (`evaluate_slam.py`)
Runs the non-learned geometric SLAM baseline for comparative evaluation.
```bash
python scripts/perception/evaluate_slam.py \
    --locomotion_checkpoint logs/locomotion/LM_rough_sim_.../model_5000.pt \
    --mode sparse_only
```

---

## 5. Releasing Models

Because log directories generate hundreds of checkpoints (`model_50.pt`, `model_100.pt`, etc.), you should "release" your best models into a clean folder before committing them to Git or deploying them.

Our models are tiny (<5MB), so we track them directly in standard Git using the `release_model.py` tool. This script automatically finds the best `.pt` weights, extracts your `env.yaml` and `agent.yaml`, grabs the exact Git status (`git_info.yaml`), and packages them cleanly.

### Usage Example
```bash
python scripts/release_model.py \
    --domain locomotion \
    --log-dir logs/locomotion/LM_rough_sim_2026-02-06_14-23-29 \
    --tag v1.0-rough \
    --message "Initial stable walking policy on rough terrain."
```

**Output Structure:**
```text
releases/
└── locomotion/
    └── v1.0-rough/
        ├── agent.yaml
        ├── env.yaml
        ├── git_info.yaml
        ├── README.md
        └── model.pt
```

After releasing, you can easily evaluate your clean model:
```bash
python scripts/locomotion/play_locomotion.py \
    --task Meldog-RL-Locomotion-Rough-Sim-v0 \
    --checkpoint releases/locomotion/v1.0-rough/model.pt
```

---

## 6. AI & Agentic Workflow
This repository features built-in support for autonomous AI workflows (e.g., using Google Gemini or Claude). 

All AI-driven planning, historical experiment logs, and active tasks are located in the `agentic/` directory.
- `agentic/YYYY-MM-DD_Task_...md` (Active tasks and checklists)
- `agentic/YYYY-MM-DD_Experiment_...md` (Completed experiment results)
- `agentic/archive/` (Past completed tasks and historical experiments)

If you are using an AI agent, it will automatically track its progress in this folder so that any other agent (or human) can pick up the context seamlessly.
