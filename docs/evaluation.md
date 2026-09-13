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
seed, terrain and command script). On random rough terrain, attitude and impact ranges
loosen: flag, don't fail.

| Metric | ✅ good | ⚠️ acceptable | ❌ investigate | Conf. | Source / note |
|---|---|---|---|---|---|
| `survival_rate` (flat / rough) | >99 / >95 % | >95 / >85 % | below | M | our own early results as a floor |
| `tracking.lin_err` (m/s) | <0.15 | <0.30 | ≥0.30 | M | exponential tracking reward geometry (Isaac Lab velocity env) |
| `tracking.ang_err` (rad/s) | <0.20 | <0.40 | ≥0.40 | M | same |
| `attitude.roll_mean`, `pitch_mean` (rad) | \|·\|<0.03 | <0.07 | ≥0.07 | M | a nonzero mean is a persistent lean; flat benchmark only |
| `attitude.roll_std`, `pitch_std` (rad) | <0.05 | <0.10 | ≥0.10 | M | flat benchmark; informational on rough |
| `gait.duty_factor` (per foot) | 0.50–0.65 | 0.45–0.75 | outside | H | trot at moderate speed; a walk is higher |
| duty-factor spread across feet | <0.05 | <0.10 | ≥0.10 | H | gait symmetry |
| `gait.phase_offset` vs front-left (FR, RL, RR) | 0.5, 0.5, 0.0 ±0.10 | ±0.15 | outside / null | H | trot signature: diagonal pairs in phase. Null = no stable gait cycle (reported, not flagged) |
| `gait.stride_freq` (Hz) | 1.0–2.5 | 0.8–3.0 | outside | M | ANYmal ≈1–1.5 Hz, Go2-scale ≈2–3 Hz; above the range looks rushed |
| `slip.mean_vel` (m/s, feet in contact) | <0.05 | <0.20 | ≥0.20 | L–M | visible sliding starts around 0.2 |
| `impact.peak_force_bw` (per foot, × body weight) | <2.0 | <3.0 | ≥3.0 | L | nominal trot ≈1–1.5 BW; ≥3 looks like feet smashing down |
| `impact.touchdown_vel` (m/s) | <0.3 | <0.5 | ≥0.5 | L | heuristic |
| `smooth.action_rate` | — | — | — | — | trend metric: compare between checkpoints only |
| `smooth.joint_acc` | — | — | — | — | trend metric |
| `actuator.torque_sat_pct` (of 0.9 × 35 Nm) | <5 % | <20 % | ≥20 % | M | sustained saturation means an actuator-limited policy, a sim-to-real risk |
| `actuator.vel_sat_pct` (of 0.9 × limit) | <1 % | <5 % | ≥5 % | M | any saturation suggests wild motion |
| `energy.cost_of_transport` | <1.0 | <2.0 | ≥2.0 | L–M | quadruped robots typically 0.4–1.2 at moderate speed |

Notes:
- `slip.mean_vel` and `slip.dist_per_step` exclude the first and last sample of every
  contact phase (touchdown and lift-off artifacts); the unfiltered values are reported as
  `*_raw` without a range.
- `impact.peak_force_bw` uses the maximum over physics substeps when the recording
  provides it; `meta.impact_force_source` in `metrics.json` says which source was used.
  Older snapshot-based recordings can under-read peaks.
- `swing.apex_height` and `swing.knee_excursion` (per foot) and the `posture.*` metrics
  are trend metrics without a range yet.

Background reading: *Learning to Walk in Minutes Using Massively Parallel Deep
Reinforcement Learning* (Rudin et al., 2021) for the training and reward setup;
*Learning robust perceptive locomotion for quadrupedal robots in the wild* (Miki et al.,
2022) for the gait quality to aim for.

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
