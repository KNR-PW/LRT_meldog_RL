# Meldog RL: User Manual

How to set up the repository, which training and evaluation tasks exist, and how to
train, evaluate, analyze and release locomotion and perception models.

## 0. Setup

- Install [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) and its conda environment,
  then install this package in editable mode from the repository root (`pip install -e .`).
- Model weights, thesis PDFs and images are stored with **Git LFS**. Run
  `git lfs install` once per machine before cloning, or `git lfs pull` after.
- Optional safety hooks: `pip install pre-commit` and `pre-commit install` (blocks
  files over 2 MB outside LFS, private keys, conflict markers, broken YAML/TOML).
- Run every script from the repository root inside the Isaac Lab environment, e.g.
  `conda activate <isaac-env>` and `source <IsaacLab>/_isaac_sim/setup_conda_env.sh`.
- The robot model (USD) is not in the repository: set `MELDOG_USD_PATH` to the Meldog
  USD file, or copy it to `assets/robots/meldog/Meldog-1.4-no-ground-plane.usd`.
- Isaac-free unit tests: `python tests/test_heightmap_transform.py`,
  `python tests/test_elevation_mapper.py`, `python tests/test_robot_specs.py` and
  `python tests/test_benchmark_terrains.py`.

## 1. Locomotion Tasks

The framework registers locomotion pipelines for **4 terrains** and **3 target domains**.

| Terrain | Sim (fast training) | Real (sim-to-real domain randomization) | Dataset (camera data collection) |
| :--- | :--- | :--- | :--- |
| **Flat** | `Meldog-RL-Locomotion-Flat-Sim-v0` | `Meldog-RL-Locomotion-Flat-Real-v0` | `Meldog-RL-Dataset-Flat-v0` |
| **Rough** | `Meldog-RL-Locomotion-Rough-Sim-v0` | `Meldog-RL-Locomotion-Rough-Real-v0` | `Meldog-RL-Dataset-Rough-v0` |
| **Flat + Obstacles** | `Meldog-RL-Locomotion-FlatObs-Sim-v0` | `Meldog-RL-Locomotion-FlatObs-Real-v0` | `Meldog-RL-Dataset-FlatObs-v0` |
| **Rough + Obstacles** | `Meldog-RL-Locomotion-RoughObs-Sim-v0` | `Meldog-RL-Locomotion-RoughObs-Real-v0` | `Meldog-RL-Dataset-RoughObs-v0` |
| **Flat V2** | `Meldog-RL-Locomotion-Flat-Sim-v1` | — | — |
| **Rough V2** | `Meldog-RL-Locomotion-Rough-Sim-v1`, `Meldog-RL-Locomotion-Rough-SimD1-v1` | — | — |

The RL algorithm is always PPO (RSL-RL); hyperparameters switch automatically between
flat and rough terrain.

### V2 tasks (`...-Sim-v1`)
The V2 package targets gait quality:

- **Phase-clock trot:** a per-env gait clock (`gait_clock_freq` 1.7 Hz) is appended to
  the observations as [sin, cos] (235 → 237 observations), and a `contact_schedule`
  reward matches diagonal leg pairs to the two clock halves.
- **Gait shaping:** swing-window, terrain-relative `foot_clearance`; `foot_slip` and
  `joint_deviation_hip` penalties; command shaping (full 3-dim air-time gate, 20 %
  pure-rotation resamples, yaw-rate parity).
- **Robustness:** reset randomization, pushes, observation noise, `velocity_limit` 12.0,
  `action_scale` 0.25, observation normalization.
- **Posture:** `base_height` (trunk height above terrain vs `base_height_target`,
  nominal 0.34 m) and `flat_orientation_terrain` (tilt relative to a plane fitted to the
  height scan, so the robot leans *with* a slope).
- **`Rough-SimD1-v1`:** the Rough V2 task with the posture and clearance weights of
  the best-rated run; used for the `v2.0-rough` release.

Earlier timing-statistics terms (`gait_sync`, `air_time_mode`, `air_time_variance`,
legacy `feet_air_time`) remain implemented but have scale 0 in V2. All V2 scales and
flags default to off in `BaseMeldogEnvCfg`, so every v0 task and checkpoint behaves
exactly as before. Configs: `simulation/flat_v2_cfg.py`, `simulation/rough_v2_cfg.py`;
runners: `MeldogFlatV2PPORunnerCfg`, `MeldogRoughV2PPORunnerCfg`.

## 2. Perception Models

Perception is decoupled from locomotion: a trained locomotion policy walks around a
`Dataset` task to collect data, which then trains one of these models.

| Model | Code | Description |
| :--- | :--- | :--- |
| **V5 Temporal** | `heightmap_v5` | ConvGRU memory builds a "belief state" across frames. |
| **V6 Autoregressive** | `heightmap_v6` | U-Net / deep-attention network for sequential reconstruction. |
| **Baseline (elevation)** | `ElevationMapper` | Non-learned baseline: world-frame elevation mapping (Fankhauser-style, per-cell Kalman fusion, known poses). The default comparator. |
| **Baseline (legacy)** | `SLAMBaseline` | Old shift-and-composite baseline; blurs by construction. Kept as the naive tier. |

> [!NOTE]
> **2026-07-12 transform fix:** `transform_height_map_with_mask` (shared by V6 and the
> legacy baseline) had a translation-axis bug and missing Δz compensation. **V6
> checkpoints trained before this date are stale**; retrain before evaluating.

---

## 3. Locomotion Usage

### A. Training (`train_locomotion.py`)
Trains the policy with PPO in many parallel environments. Use `--headless` for speed.
```bash
python scripts/locomotion/train_locomotion.py \
    --task Meldog-RL-Locomotion-Rough-Sim-v0 \
    --num_envs 4096 \
    --max_iterations 5000 \
    --headless
```
Every run is saved to `logs/locomotion/LM_<task>_<date>_<commit>/`. The folder name ends
with the git commit of the code (`-dirty` if there were uncommitted changes, whose diff
is saved in `params/git_info.yaml`). Commit before training so every run maps to exact code.

### B. Visual evaluation (`play_locomotion.py`)
Runs the policy with rendering so you can watch the robot.
```bash
python scripts/locomotion/play_locomotion.py \
    --task Meldog-RL-Locomotion-Rough-Sim-v0 \
    --checkpoint releases/locomotion/v1.0-rough/model.pt
```

### C. Keyboard control (`play_locomotion_keyboard.py`)
Like visual evaluation, but WASD sets the velocity command so you can drive the robot.
```bash
python scripts/locomotion/play_locomotion_keyboard.py \
    --task Meldog-RL-Locomotion-Flat-Sim-v0 \
    --checkpoint logs/locomotion/LM_flat_sim_.../model_2000.pt
```

### D. Metrics evaluation (`evaluate_locomotion.py`)
Runs the policy headless over many episodes, prints a survival-rate report and, by
default, records the full per-step rollout to `rollout.h5` for the analyzer (§6).
```bash
python scripts/locomotion/evaluate_locomotion.py \
    --task Meldog-RL-Locomotion-Rough-Sim-v0 \
    --checkpoint logs/locomotion/LM_rough_sim_.../model_5000.pt \
    --num_episodes 100 \
    --headless
```

Key flags:
- `--record true|false` (default **true**): write `rollout.h5` into a new
  `logs/locomotion/LE_*` folder. Turn off only for a quick survival-rate print.
- `--benchmark`: **benchmark mode** for comparable numbers across checkpoints. Forces
  `seed=42` and applies a fixed velocity command script every step (per-env episode clock):
  - 0–5 s → `(vx=0.8, vy=0, wz=0)` walk forward
  - 5–10 s → `(0, 0, wz=0.8)` turn in place
  - 10–15 s → `(vx=0.5, vy=0.3, 0)` diagonal, then the 15 s cycle repeats.

  The seed, command script and terrain generation are reproducible. On our machine,
  repeated benchmark runs of the same checkpoint (16 envs, 16 episodes) produced
  identical metrics; other env counts, GPUs or driver versions may show small physics
  solver noise. The reference ranges in [evaluation.md](evaluation.md) assume benchmark mode.

```bash
python scripts/locomotion/evaluate_locomotion.py \
    --task Meldog-RL-Locomotion-Rough-Sim-v0 \
    --checkpoint releases/locomotion/v1.0-rough/model.pt \
    --benchmark --num_envs 16 --num_episodes 16 --headless
```

- `--full_episodes`: count only each env's first episode and start it at step 0. Meldog's
  env randomizes episode length on reset, so otherwise most episodes are shorter than 20 s.
  Every env then runs a full-length episode unless it falls, and `--num_episodes` is set to
  `--num_envs`. **Always on in benchmark mode.** Survival rates from before this change
  (benchmarks recorded before 2026-09-15) are not comparable.
- In benchmark mode every robot is evaluated under the same conditions, whatever its task
  was trained with:
  - `--profile clean` (default): no observation noise, no pushes or mass randomization,
    nominal friction (0.8 / 0.6), reset at the default pose with yaw 0, terrain curriculum off.
  - `--profile real`: `clean` plus uniform observation noise (lin vel ±0.1 m/s, ang vel
    ±0.2 rad/s, gravity ±0.05, joint pos ±0.01 rad, joint vel ±1.5 rad/s, height scan ±0.1 m),
    friction 0.4-1.2 / 0.3-1.0, trunk mass ±10 %, velocity pushes of ±0.5 m/s every 10-15 s,
    motor strength ×0.8-1.2 per robot and one control step of actuation delay.
  - The contact sensor keeps every physics substep (history = decimation). A fall is trunk tilt
    above 1.0 rad, or fatal trunk contact: most environments (Meldog's and every Isaac Lab
    velocity task) terminate the episode themselves above 1 N, and where an environment has no
    such rule the evaluator falls back to 20 % of body weight (`--fall_base_force_bw`).
- `--bench_terrain task|flat|rough|obs|rough_obs` (default `task`): `task` keeps the task's own
  terrain; the others are Meldog's shared terrains (`rough` = the Rough task generator, `obs` /
  `rough_obs` = the FlatObs / RoughObs generators with tall obstacles and walls). Each env gets a
  fixed terrain cell: every sub-terrain kind at difficulty rows 0, 3, 6 and 9. The report then
  shows survival per terrain kind and row.
- `--output_root DIR`: create the evaluation folder inside `DIR` instead of `logs/locomotion`.
- `--num_envs` defaults to **192 in benchmark mode** (8 robots per terrain cell). With fewer envs the
  survival rate on hard terrain varies by about ±0.1 between runs; the other metrics are stable.
- `--bench_commands absolute|froude`: `absolute` (default) gives every robot the same speeds;
  `froude` scales them with the square root of leg length relative to Meldog's 0.50 m, so robots of
  different size are compared at dynamically similar speeds.
- `--video [--video_length 400] [--video_env 0]`: record the run with a third-person camera that
  follows one robot, into `video/` inside the evaluation folder. The robot's own perception cameras
  stay off unless you also pass `--enable_cameras`.

### E. Reference robots (Isaac Lab quadrupeds)
The same evaluator runs the Isaac Lab velocity tasks, so their policies are measured exactly
like Meldog's. Supported robots: ANYmal-B/C/D, Unitree Go1/Go2/A1 and Spot
(`Isaac-Velocity-{Flat,Rough}-<Robot>-v0`, plus the direct `Isaac-Velocity-*-Anymal-C-Direct-v0`).
```bash
python scripts/locomotion/evaluate_locomotion.py \
    --task Isaac-Velocity-Rough-Unitree-Go2-v0 \
    --checkpoint <IsaacLab>/logs/rsl_rl/unitree_go2_rough/<run>/model_1499.pt \
    --benchmark --full_episodes --num_envs 16 --headless
```
- Output folders are named `LE_ref_<robot>_<flat|rough>_*`.
- Foot, knee, hip and shank bodies of each robot are defined in
  `source/meldog_rl/eval/robot_specs.py`; add an entry there to support another robot.
- Every recording stores the robot's size and limits so results can be interpreted per robot:
  mass, per-leg thigh and shank lengths (`leg_length`), per-joint effort and velocity limits,
  default joint pose and knee joint indices.
- Tasks without a height scanner (all flat tasks) record no posture channel.

---

## 4. Perception Usage

### A. Dataset collection (`collect_dataset.py`)
A trained locomotion policy walks around and records depth camera frames plus
ground-truth height maps.
```bash
python scripts/perception/collect_dataset.py \
    --task Meldog-RL-Dataset-Rough-v0 \
    --checkpoint releases/locomotion/v1.0-rough/model.pt \
    --max_steps 10000
```

### B. Perception training (`train_perception.py`)
Trains a perception model offline on a collected `dataset.h5` (`--dataset auto` picks the
latest one).
```bash
python scripts/perception/train_perception.py \
    --dataset datasets/PD_rough_.../dataset.h5 \
    --model heightmap_v5 \
    --epochs 100 \
    --workers 8 \
    --batch_size 80 \
    --seq_len 32
```
> [!WARNING]
> **RAM:** data-loader workers use a lot of system memory. `8 workers`, batch size 80,
> `seq_len` 32 runs stably but peaks near 48 GB RAM. With less than 64 GB, drop to
> `4 workers` and batch size 160.

### C. Perception and baseline evaluation (`evaluate_perception.py`)
Runs a locomotion policy in a camera-enabled `Dataset` task and reconstructs the height
map with `--method model` (V5/V6 checkpoint) or `--method slam`
(`--slam_variant elevation|legacy`, no checkpoint needed). It records `eval.mp4` (map
panels with a fixed colorbar) and `data.h5` for the analyzer. With `--headless` the
script exits after saving. (`evaluate_slam.py` is a deprecated alias for `--method slam`.)

`data.h5` holds one group per env (`env_i/{gt_height, pred_height, sparse_height,
occlusion_mask, robot_pos, robot_quat, dones}`) plus file attributes (`task`,
`locomotion_checkpoint`, `perception_checkpoint` or `slam_variant`, `method`, `date`,
`git_commit`).
```bash
python scripts/perception/evaluate_perception.py \
    --task Meldog-RL-Dataset-Rough-v0 \
    --locomotion_checkpoint releases/locomotion/v1.0-rough/model.pt \
    --perception_checkpoint logs/perception/PM_.../model.pt --model v5

python scripts/perception/evaluate_perception.py \
    --task Meldog-RL-Dataset-Rough-v0 \
    --locomotion_checkpoint releases/locomotion/v1.0-rough/model.pt \
    --method slam --slam_variant elevation --headless
```

### D. Offline baseline replay (`run_slam_offline.py`)
Re-runs a baseline on the sparse maps and poses stored in an existing `data.h5`, without
Isaac Sim, in seconds. This gives a **same-trajectory** model-vs-baseline comparison:
record one model run, replay the baseline on its `data.h5`, analyze both together.
```bash
python scripts/perception/run_slam_offline.py logs/perception/PE_v5_.../data.h5 \
    --variant elevation        # -> new PE_slam-elevation-replay_* folder
python scripts/perception/analyze_perception.py \
    logs/perception/PE_v5_.../data.h5 logs/perception/PE_slam-elevation-replay_.../data.h5 \
    --labels model slam
```

---

## 5. Releasing Models

Training produces hundreds of checkpoints. Promote the one you want to keep into
`releases/` with `release_model.py`: it copies the weights, `env.yaml`, `agent.yaml`
and `git_info.yaml`, and writes a `README.md`. Release weights are stored with Git LFS.

```bash
python scripts/release_model.py \
    --domain locomotion \
    --log-dir logs/locomotion/LM_rough_sim_2026-02-06_14-23-29 \
    --tag v1.0-rough \
    --message "Initial stable walking policy on rough terrain."
```

> [!NOTE]
> `release_model.py` always takes the **highest-iteration** checkpoint. If a different
> checkpoint benchmarks better, copy it by hand in the same layout (see
> `releases/locomotion/v2.0-rough/`).

Current releases: `v1.0-rough-baseline`, `v1.0-rough`, `v2.0-rough` (best V2 policy).

```text
releases/
└── locomotion/
    └── v2.0-rough/
        ├── agent.yaml
        ├── env.yaml
        ├── git_info.yaml
        ├── README.md
        └── model.pt
```

---

## 6. Analyzing Results

Evaluation has a **record** step (in simulation, above) and an **analyze** step. The
analyzers use only numpy, h5py and matplotlib, so they run anywhere in seconds and can
be re-run on old recordings without a GPU.

### A. Locomotion analyzer (`analyze_locomotion.py`)
Reads `rollout.h5`; writes `metrics.json`, `report.md` and `plots/` (gait diagram,
tracking, attitude, actions, swing, posture) next to it.
```bash
python scripts/locomotion/analyze_locomotion.py \
    --rollout logs/locomotion/LE_rough_sim_.../rollout.h5   # [--output DIR]
```

### B. Perception analyzer (`analyze_perception.py`)
Reads one or two `data.h5` files (model and/or baseline); writes `metrics.json`,
`report.md` and `plots/` (ground truth | prediction | sparse input | error panels with
fixed scales, plus error over time). Two inputs give a side-by-side comparison.
```bash
python scripts/perception/analyze_perception.py MODEL.h5 [SLAM.h5] \
    --labels model slam            # [-o DIR] [--env N] [--timesteps 50 200 500 950]
```

### C. Comparing runs (`compare_metrics.py`)
Prints one Markdown table of the key metrics and flags for any number of evaluations, sorted
by robot mass, with robot name and leg length.
```bash
python scripts/locomotion/compare_metrics.py logs/locomotion/LE_a logs/locomotion/LE_b \
    [--output comparison.md] [--csv comparison.csv]
```

### D. What an evaluation run produces

| File | Purpose |
| :--- | :--- |
| `metrics.json` | All metrics: a `meta` block (task, checkpoint, seed, commit, date) plus a `locomotion` or `perception` metric tree and `flags` |
| `report.md` | Readable metric table with a ✅/⚠️/❌ column |
| `plots/*.png` | Figures with fixed color scales and legends |
| `rollout.h5` / `data.h5` | Raw recorded state the analyzer reads |
| `eval.mp4` | Video (perception; optional for locomotion) |

**Flags** rate each metric as good / acceptable / investigate (✅/⚠️/❌) against the
reference ranges in [evaluation.md](evaluation.md). The thresholds are copied into each
analyzer, so change both together. Flags mean "look at this", not pass/fail. Trend-only
metrics (`smooth.*`), metrics without a range, and missing values carry no flag. The
model-vs-baseline comparison flags appear only when the analyzer gets two inputs.

For judging videos, use the checklist in [eval_rubric.md](eval_rubric.md).
