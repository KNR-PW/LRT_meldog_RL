# Evaluation Metrics and Reference Ranges

Reference ranges used by `analyze_locomotion.py` and `analyze_perception.py` to flag
each metric as ✅ good, ⚠️ acceptable or ❌ investigate. **Cross-robot numbers are
approximations**: ❌ means "look at this", not "failed".

Each range has a confidence level: **H**igh (definitional or measured), **M**edium
(derived from reference configs or standard quadruped values), **L**ow (heuristic, to be
refined with more data). The thresholds are copied into the analyzers; when you change a
range here, change it in the analyzer too.

## Locomotion

The ranges apply to **benchmark-mode** runs (`evaluate_locomotion.py --benchmark`: fixed
seed, terrain cells, command script, full 20 s episodes, `clean` profile unless stated).
Survival counts each env's first episode; a fall is trunk contact above 1 N or trunk tilt
above 1.0 rad. On non-flat terrain the raw attitude is a trend metric and the terrain-relative
posture (`posture.*_terrain_rel_*`) carries the attitude ranges instead.

| Metric | ✅ good | ⚠️ acceptable | ❌ investigate | Conf. | Source / note |
|---|---|---|---|---|---|
| `survival_rate` (flat / rough) | >99 / >95 % | >95 / >85 % | below | M | our own early results as a floor |
| `tracking.lin_err` (m/s) | <0.15 | <0.30 | ≥0.30 | M | exponential tracking reward geometry (Isaac Lab velocity env) |
| `tracking.ang_err` (rad/s) | <0.20 | <0.40 | ≥0.40 | M | same |
| `attitude.roll_mean`, `pitch_mean` (rad) | \|·\|<0.03 | <0.07 | ≥0.07 | M | a nonzero mean is a persistent lean; flat terrain only |
| `attitude.roll_std`, `pitch_std` (rad) | <0.05 | <0.10 | ≥0.10 | M | flat terrain only; trend metric elsewhere |
| `posture.roll_terrain_rel_mean`, `pitch_terrain_rel_mean` (rad) | \|·\|<0.03 | <0.07 | ≥0.07 | M | tilt relative to the local terrain plane; used instead of attitude on non-flat terrain |
| `posture.roll_terrain_rel_std`, `pitch_terrain_rel_std` (rad) | <0.05 | <0.10 | ≥0.10 | M | same, non-flat terrain |
| `gait.duty_factor` (per foot) | 0.50–0.65 | 0.45–0.75 | outside | H | trot at moderate speed; a walk is higher |
| duty-factor spread across feet | <0.05 | <0.10 | ≥0.10 | H | gait symmetry |
| `gait.phase_offset` vs front-left (FR, RL, RR) | 0.5, 0.5, 0.0 ±0.10 | ±0.15 | outside / null | H | trot signature: diagonal pairs in phase. Null = no stable gait cycle (reported, not flagged) |
| `gait.stride_freq` (Hz) | 1.0–2.5 | 0.8–3.0 | outside | M | ANYmal ≈1–1.5 Hz, Go2-scale ≈2–3 Hz; above the range looks rushed |
| `slip.mean_vel` (m/s, feet in contact) | <0.05 | <0.20 | ≥0.20 | L–M | visible sliding starts around 0.2 |
| `impact.peak_force_bw_p95` (95th percentile of touchdown peaks, × body weight) | <2.0 | <3.0 | ≥3.0 | L | nominal trot ≈1–1.5 BW; ≥3 looks like feet smashing down. The mean (`peak_force_bw`) and maximum (`peak_force_bw_max`) are reported as trend metrics; a mean hides occasional smashes |
| `impact.touchdown_vel` (m/s) | <0.3 | <0.5 | ≥0.5 | L | heuristic |
| `smooth.action_rate` | — | — | — | — | trend metric: compare between checkpoints only |
| `smooth.joint_acc` | — | — | — | — | trend metric |
| `actuator.torque_sat_pct` (of 0.9 × each joint's effort limit) | <5 % | <20 % | ≥20 % | M | sustained saturation means an actuator-limited policy, a sim-to-real risk |
| `actuator.vel_sat_pct` (of 0.9 × each joint's velocity limit) | <1 % | <5 % | ≥5 % | M | any saturation suggests wild motion |
| `energy.cost_of_transport` | <1.0 | <2.0 | ≥2.0 | L–M | quadruped robots typically 0.4–1.2 at moderate speed |

Diagnostic metrics without ranges (reported as trends):
- `asymmetry.*`: how unevenly the legs work. `torque_left_right` and `torque_front_rear` are shares
  between −1 and +1 (0 = balanced, +1 = all load on the left / front), `sat_pct_left_right` is the
  difference in torque-limit time in percentage points, and `torque_per_leg` / `sat_pct_per_leg`
  give the four legs in FL, FR, RL, RR order. A policy can track its commands well and still load
  one side far harder than the other; nothing else in the table shows that.
- `actuator.torque_sat_pct_per_joint` and `torque_mean_abs_per_joint`: the same per joint, with the
  joint names, so the report can name the joints that sit at their limit.
- `gait.stride_ref_foot`: which foot the stride cycle was measured from (the one with the clearest
  cycle), so a single misbehaving foot no longer hides the whole gait section.

Notes:
- `slip.mean_vel` and `slip.dist_per_step` exclude the first and last sample of every
  contact phase (touchdown and lift-off artifacts); the unfiltered values are reported as
  `*_raw` without a range.
- `impact.peak_force_bw` uses the maximum over physics substeps when the recording
  provides it; `meta.impact_force_source` in `metrics.json` says which source was used.
  Older snapshot-based recordings can under-read peaks.
- `swing.apex_height`, `swing.knee_excursion` (per foot) and `posture.base_height*` are trend
  metrics without a range yet.

Background reading: *Learning to Walk in Minutes Using Massively Parallel Deep
Reinforcement Learning* (Rudin et al., 2021) for the training and reward setup;
*Learning robust perceptive locomotion for quadrupedal robots in the wild* (Miki et al.,
2022) for the gait quality to aim for.

## Reference robots (measured 2026-09-16)

What a good policy looks like under the benchmark, measured on six published Isaac Lab
rough-terrain policies (NVIDIA pretrained checkpoints for ANYmal-B, ANYmal-C, ANYmal-D,
Unitree Go1, Go2 and A1; 1500 training iterations each): `--benchmark --profile clean`,
192 envs (8 per terrain cell), same command script for every robot. Values are the minimum and
maximum over the six policies. The robots weigh 13-52 kg in simulation with legs of 0.40-0.74 m (Meldog: 22.5 kg,
0.50 m), so size-dependent metrics (stride frequency, trunk height, slip, touchdown velocity) should
be read with the robot's size in mind. ANYmal and Go1 use learned actuator models that smooth their
motion; Go2 and A1 use DC motor models like Meldog.

| Metric | Shared `flat` terrain | Shared `rough` terrain |
|---|---|---|
| `survival_rate` | 0.96-1.00 | 0.92-1.00 |
| `tracking.lin_err` (m/s) | 0.040-0.060 | 0.045-0.075 |
| `tracking.ang_err` (rad/s) | 0.05-0.18 | 0.06-0.18 |
| `gait.duty_factor` (per foot) | 0.34-0.75 | 0.35-0.75 |
| `gait.phase_offset` (FR, RL, RR) | 0.46-0.52, 0.47-0.61, 0.00-0.07 | 0.46-0.53, 0.51-0.60, 0.02-0.10 |
| `gait.stride_freq` (Hz) | 1.67-2.38 | 1.68-2.36 |
| `slip.mean_vel` (m/s) | 0.053-0.148 | 0.071-0.160 |
| `impact.peak_force_bw_p95` (BW) | 0.73-1.50 | 0.90-1.77 |
| `impact.touchdown_vel` (m/s) | 0.043-0.153 | 0.078-0.203 |
| `posture.pitch_terrain_rel_std` (rad) | 0.018-0.046 | 0.026-0.071 |
| `posture.base_height` (m) | 0.25-0.56 | 0.25-0.56 |
| `actuator.torque_sat_pct` (%) | 0.00-0.02 | 0.00-0.38 |
| `actuator.vel_sat_pct` (%) | 0.0-1.6 | 0.0-1.7 |
| `energy.cost_of_transport` | 0.43-1.08 | 0.48-1.16 |

`scripts/locomotion/compare_metrics.py --bands meldog` prints this band for any set of runs and
marks where a robot's values fall outside it.

## Perception

These are **relative requirements** for the thesis, not absolute literature values: the
learned reconstruction should beat the elevation-mapping baseline where the robot cannot
see. Comparisons require the **same trajectory** (identical `data.h5` input, see
`run_slam_offline.py`).

| Check | Requirement | Conf. | Note |
|---|---|---|---|
| `rmse_visible` | <0.05 m | M | should approach the sensor/projection noise floor on a 5 cm grid |
| `rmse_occluded` vs visible | ≤4 × `rmse_visible` | L | a large gap means the model only copies its input |
| model vs baseline, `rmse_all` | model < baseline | H | core comparison |
| model vs baseline, `rmse_occluded` | model < baseline by a clear margin | H | the headline result |
| model vs sparse input, `rmse_occluded` | model ≪ sparse | H | sanity check: reconstruction must beat no inpainting |
| `edge_mae` model vs baseline | model ≤ baseline | M | blurred obstacle edges are the typical failure of L2-trained models |
| `error_vs_time` | decreasing over the first ~5 s, then flat | M | memory accumulates; a rising curve means a state or reset bug |

For absolute numbers, compare with *Reconstructing Occluded Elevation Information in
Terrain Maps With Self-Supervised Learning* before quoting any.

## Updating the ranges

Low-confidence (**L**) rows should be revisited once enough benchmark runs exist: either
raise their confidence or adjust the range, and note the change here with a date.
