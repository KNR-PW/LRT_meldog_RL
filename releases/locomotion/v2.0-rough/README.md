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

## Benchmark under the updated conditions (2026-09-16)
The benchmark above used randomly truncated episodes (most shorter than 20 s), so its survival
rate overstates robustness. Re-measured with the current benchmark: full 20 s episodes, the same
fall rule for every robot, `--profile clean`, Meldog's shared terrains with fixed terrain cells,
192 envs. The reference range is the minimum-maximum over six published Isaac Lab rough policies
(ANYmal-B/C/D, Go1, Go2, A1; see `docs/evaluation.md`).

| metric (shared rough terrain) | v2.0-rough | v1.0-rough | reference range |
|---|---|---|---|
| survival_rate | 0.74 | 0.995 | 0.92-1.00 |
| tracking.lin_err (m/s) | 0.083 | 0.037 | 0.045-0.075 |
| tracking.ang_err (rad/s) | 0.239 | 0.224 | 0.06-0.18 |
| gait.phase_offset (FR, RL, RR vs FL) | 0.52, 0.51, 0.03 | 0.64, 0.29, 0.62 | ≈0.5, 0.5, 0.0-0.1 |
| gait.stride_freq (Hz) | 1.72 | 1.09 | 1.68-2.36 |
| posture.pitch_terrain_rel_std (rad) | 0.196 | 0.090 | 0.026-0.071 |
| impact.peak_force_bw_p95 (BW) | 0.86 | 0.73 | 0.90-1.77 |
| actuator.torque_sat_pct (%) | 8.1 | 1.3 | 0.00-0.38 |
| energy.cost_of_transport | 0.86 | 0.70 | 0.48-1.16 |

On the shared rough terrain v2.0-rough falls mainly on pyramid stairs: stairs down survive 38 %,
12 % and 0 % of episodes at difficulty rows 3, 6 and 9, stairs up 38-62 %, where v1.0-rough and the
reference policies survive 75-100 % of episodes. On flat terrain both versions survive every episode.

## Usage
Run inside your Isaac Lab environment (see `docs/manual.md`):
```bash
python scripts/locomotion/play_locomotion.py --task Meldog-RL-Locomotion-Rough-SimD1-v1 --checkpoint releases/locomotion/v2.0-rough/model.pt
```
