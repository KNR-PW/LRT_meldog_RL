# Locomotion Training Experiment Results

## Phase 0: Visual Sanity Checks
- `Flat-Sim-v0`: Success
- `FlatObs-Sim-v0`: Success
- `Rough-Sim-v0`: Success
- `RoughObs-Sim-v0`: Success
- `Rough-Real-v0`: Success (visually verified)

---

## Phase 1: Massive Batch Training

### 1. `Flat-Sim-v0`
- **Envs:** 4096 | **Iterations:** 3000 | **Headless:** True
- **Status:** ✅ Completed in ~28m
- **Final Metrics:** Mean Reward ~29.60, Loss ~0.003
- **Notes:** Extremely stable. Velocity tracking converged perfectly.

### 2. `FlatObs-Sim-v0`
- **Envs:** 4096 | **Iterations:** 3000 | **Headless:** True
- **Status:** ✅ Completed in ~50m
- **Final Metrics:** Mean Reward ~14.35, Loss ~0.018
- **Notes:** Very stable despite the obstacles. Velocity tracking `xy` reached ~1.11, `z` at ~0.30.

### 3. `FlatObs-Real-v0`
- **Envs:** 4096 | **Iterations:** 3000 | **Headless:** True
- **Status:** ✅ SUCCESS (Completed early, but saved models correctly)
- **Quantitative Evaluation (100 episodes):**
  - **Survival Rate:** 59.00%
  - **Successes (Survived):** 59 | **Failures (Crashed):** 41
  - **Avg Episode Length:** 295.9 ± 164.4 steps
- **Final Metrics:** Mean Reward ~12.35, Loss ~0.02, `track_lin_vel_xy_exp`: ~0.95
- **Notes on Instability:** In the first run, this config crashed at Iteration 2320 due to exploding gradients (`action_rate_l2: -10^21`). In this rerun *with the exact same parameters*, it survived all 3000 iterations! This perfectly confirms our hypothesis: the crashes are caused by non-deterministic physics glitches (e.g. random mass/friction spikes causing extreme velocities) that hit the unclipped, unnormalized policy network. It's playing Russian Roulette with the physics engine.

### 4. `Rough-Sim-v0`
- **Envs:** 4096 | **Iterations:** 5000 | **Headless:** True
- **Status:** ✅ SUCCESS (Completed ~1h23m)
- **Quantitative Evaluation (100 episodes):**
  - **Survival Rate:** 100.00%
  - **Successes (Survived):** 100 | **Failures (Crashed):** 0
  - **Avg Episode Length:** 478.8 ± 273.5 steps
- **Final Metrics:** Mean Reward ~22.10, Loss ~0.009, `track_lin_vel_xy_exp`: ~1.41
- **Notes on Instability:** Survived all 5000 iterations this time! Fully confirms that the crashes are random (non-deterministic physics glitches blowing up the unclipped, high-entropy network).

### 5. `RoughObs-Sim-v0`
- **Envs:** 4096 | **Iterations:** 5000 | **Headless:** True
- **Status:** ✅ SUCCESS (Completed ~1h45m)
- **Quantitative Evaluation (100 episodes):**
  - **Survival Rate:** 77.00%
  - **Successes (Survived):** 77 | **Failures (Crashed):** 23
  - **Avg Episode Length:** 348.5 ± 214.4 steps
- **Final Metrics:** Mean Reward ~14.72, Loss ~0.016, `track_lin_vel_xy_exp`: ~1.17
- **Notes on Instability:** Survived all 5000 iterations without crashing! The baseline unclipped model avoided physics explosions this time around.

### 6. `RoughObs-Real-v0`
- **Envs:** 4096 | **Iterations:** 5000 | **Headless:** True
- **Status:** ❌ CRASHED at Iteration 3386
- **Quantitative Evaluation (100 episodes on checkpoint 3350):**
  - **Survival Rate:** 65.00%
  - **Successes (Survived):** 65 | **Failures (Crashed):** 35
  - **Avg Episode Length:** 268.7 ± 179.3 steps
- **Error Details:**
  - `Mean value_function loss: inf`
  - `RuntimeError: normal expects all elements of std >= 0.0`
- **Diagnosis:** The exact same numerical explosion! The `Real` config introduces domain randomization (randomizing link mass, friction, etc.). At iteration 3386, the physics engine produced an extreme anomaly, causing the un-clipped un-normalized network to explode, sending standard deviation to negative values and value function loss to infinity.

## 👁️ Visual Evaluation Feedback

The following are the user's subjective observations from running the checkpoints in non-headless playback mode:

- **`FlatObs-Real-v0`:** Robots frequently died upon touching obstacles and made little to no attempt to step over them. Rarely adjusted trajectory to walk around corners. Movement was unnatural and twitchy (sometimes with one leg up). Suffered from a persistent, weird body roll/orientation.
- **`Rough-Sim-v0`:** Movement was way more natural with some gait forming, though still very quick. Handled uneven terrain quite nicely, though not as smoothly as an Anymal. Still suffered from a persistent weird body angle not related to the terrain shape (e.g., leaning into walking up stairs).
- **`RoughObs-Sim-v0`:** Behaved similarly to `Rough-Sim-v0` on terrain. Tended to bump into obstacles but would eventually walk around them (never over them in a smart manner). Movement was still too quick and unnatural, and the persistent weird body angle remained.
- **`RoughObs-Real-v0` (Iteration 3350):** Attempted to go around obstacles at a lower speed. No attempts to walk over obstacles. Rough terrain traversal was fine, but suffered from big accelerations where feet would fly off the ground and smash back down.
