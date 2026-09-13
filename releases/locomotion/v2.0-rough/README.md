# Release: v2.0-rough

**Domain:** locomotion
**Source Log:** LM_rough_simd1_2026-07-26_23-47-04_f168275
**Checkpoint:** model_10000.pt (run went to model_11999, which was never benchmarked)
**Task:** Meldog-RL-Locomotion-Rough-SimD1-v1
**Code commit:** f168275 (clean tree, see `git_info.yaml`; commit hashes before the 2026-09-13 history rewrite, see `agentic/sha-map-2026-09.txt`)

## Release Notes
Best Locomotion V2 policy so far: phase-clock trot on rough terrain, privileged (ground-truth) height scan.
Selected in `agentic/2026-07-19_Task_Locomotion_V2.md`, section "Queue results 2026-07-27" (D1 length curve).
Released by hand because `scripts/release_model.py` always picks the highest iteration.

## Benchmark
`evaluate_locomotion.py --benchmark` (seed 42, scripted commands), 16 envs, 16 episodes, then `analyze_locomotion.py`.
The benchmark is deterministic: the 2026-07-27 run and two runs on 2026-09-13 gave identical `metrics.json`.

| metric | value | v1.0-rough (same benchmark, Rough-Sim-v0) |
|---|---|---|
| survival_rate | 1.00 | 1.00 |
| tracking.lin_err (m/s) | 0.072 | 0.039 |
| tracking.ang_err (rad/s) | 0.220 | 0.224 |
| gait.duty_factor_spread | 0.072 | 0.349 |
| gait.phase_offset (FR, RL, RR vs FL) | 0.51, 0.51, 0.00 | 0.66, 0.36, 0.69 |
| gait.stride_freq (Hz) | 1.72 | 1.08 |
| slip.mean_vel (m/s) | 0.062 | 0.066 |
| posture.base_height (m) | 0.288 | 0.299 |
| posture.pitch_terrain_rel_mean (rad) | 0.079 | 0.069 |
| energy.cost_of_transport | 0.91 | 0.75 |
| actuator.torque_sat_pct (%) | 8.3 | 1.7 |

Known gaps (from the task file): body pitch relative to terrain not corrected (squared posture penalties cause leg-bracing; a deadband is the proposed fix), higher energy use and torque saturation than v1.0-rough, and the impact metric under-reports.

## Usage
Run inside the Isaac wrapper from `AGENTS.md` §1:
```bash
python scripts/locomotion/play_locomotion.py --task Meldog-RL-Locomotion-Rough-SimD1-v1 --checkpoint releases/locomotion/v2.0-rough/model.pt
```
