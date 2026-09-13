# Release: v2.0-rough

**Domain:** locomotion
**Source Log:** LM_rough_simd1_2026-07-26_23-47-04_f168275
**Checkpoint:** model_10000.pt (run went to model_11999, which was never benchmarked)
**Task:** Meldog-RL-Locomotion-Rough-SimD1-v1
**Code commit:** f168275 at training time (clean tree, see `git_info.yaml`); the same code is commit e27fe9f in the current history (commit hashes changed in the 2026-09-13 history cleanup)

## Release Notes
Best Locomotion V2 policy so far: phase-clock trot on rough terrain, privileged (ground-truth) height scan.
Selected from a training-length sweep of this run: checkpoints 2000–10000 were benchmarked, and 10000 gave the best survival (1.00) and turning accuracy.
Released by hand because `scripts/release_model.py` always picks the highest iteration.

## Benchmark
`evaluate_locomotion.py --benchmark` (seed 42, scripted commands), 16 envs, 16 episodes, then `analyze_locomotion.py`.
On our machine the benchmark is deterministic: the 2026-07-27 run and several runs on 2026-09-13 gave identical metrics.

| metric | value | v1.0-rough (same benchmark, Rough-Sim-v0) |
|---|---|---|
| survival_rate | 1.00 | 1.00 |
| tracking.lin_err (m/s) | 0.072 | 0.039 |
| tracking.ang_err (rad/s) | 0.220 | 0.223 |
| gait.duty_factor_spread | 0.072 | 0.349 |
| gait.phase_offset (FR, RL, RR vs FL) | 0.51, 0.51, 0.00 | 0.66, 0.36, 0.68 |
| gait.stride_freq (Hz) | 1.72 | 1.08 |
| slip.mean_vel (m/s) | 0.062 | 0.066 |
| posture.base_height (m) | 0.288 | 0.299 |
| posture.pitch_terrain_rel_mean (rad) | 0.079 | 0.069 |
| energy.cost_of_transport | 0.91 | 0.75 |
| actuator.torque_sat_pct (%) | 8.3 | 1.7 |

Known gaps: body pitch relative to terrain not corrected (squared posture penalties cause leg-bracing; a deadband is the proposed fix), higher energy use and torque saturation than v1.0-rough, and the impact metric under-reports.

## Usage
Run inside your Isaac Lab environment (see `docs/manual.md`):
```bash
python scripts/locomotion/play_locomotion.py --task Meldog-RL-Locomotion-Rough-SimD1-v1 --checkpoint releases/locomotion/v2.0-rough/model.pt
```
