# Meldog RL — Agent Guide (source of truth)

Model-agnostic brain for every agent (Claude/Opus, Fable, Gemini). `CLAUDE.md` /
`GEMINI.md` are thin pointers here — put shared knowledge in this file.

Thesis (KNR robotics): reconstruct a robot-centric terrain map from partial RGBD on
a 12-DOF quadruped (Meldog). Sim-first (Isaac Lab); a real robot exists — keep
sim-to-real in mind.

---

## 1. Running Isaac Lab commands

Do **not** call bare `python` / `conda run`. Every training/eval/collection script
must be wrapped in this exact sequence (exposes `isaaclab` + binds Vulkan/env vars):

```bash
bash -c 'source ~/.bashrc; eval "$(conda shell.bash hook)"; conda activate isaac-sim; cd /home/frydjak/IsaacLab; source _isaac_sim/setup_conda_env.sh; cd /home/frydjak/KNR/Meldog/LRT_meldog_RL; <YOUR_PYTHON_COMMAND>'
```

Example (short sanity training):

```bash
bash -c '...prefix...; python scripts/locomotion/train_locomotion.py --task Meldog-RL-Locomotion-Rough-Sim-v0 --num_envs 2 --max_iterations 10'
```

## 2. Where things are

See **[directory.md](directory.md)** — curated map of local paths, registered task
ids, and the external Isaac Lab reference configs (anymal / unitree, flat + rough).
Read it before searching; agents cannot discover the external `/home/frydjak/IsaacLab`
paths on their own.

## 3. Model tiering — who does what

Tag every task in a plan `[FABLE]` or `[EXEC]`. Heuristic:

> If the hard part is **"what should we do?"** → `[FABLE]`.
> If it is **"do this, precisely specified"** → `[EXEC]` (Opus / Gemini).

| Class | Model(s) | Use for |
|---|---|---|
| `[FABLE]` | Fable 5 (`claude-fable-5`) | Open-ended design, metric/architecture design, research & literature, thesis review, ambiguous judgment where being wrong is expensive. |
| `[EXEC]` (driver) | Opus 4.8 (`claude-opus-4-8`) | Implementing a specified change, git surgery, code review, wiring scripts. |
| `[EXEC]` (pipeline) | Gemini | Long-running training / eval / dataset-collection orchestration. |
| cheap | Haiku 4.5 (`claude-haiku-4-5-20251001`) | Mechanical file moves, boilerplate, log scraping. |

`[FABLE]` tasks produce plans/designs; they do **not** implement or train. `[EXEC]`
tasks execute against a concrete spec (exact file paths + current→proposed values).
There is no separate `fable.md` — Fable is Claude and reads `CLAUDE.md`.

Prompts that put this into practice live in **[agentic/prompts.md](agentic/prompts.md)**.

## 4. Agentic workflow & shared memory

Cross-agent collaboration and rationale must be version-controlled in the repo (not
in any single tool's private memory — Gemini can't see Claude's memory and vice versa).

- **Active tasks / experiments:** `agentic/<YYYY-MM-DD>_Task_*.md` or `_Experiment_*.md`.
- **Archiving:** move finished files to `agentic/archive/`. Do **not** read
  `agentic/archive/` wholesale — it blows out context. Read active `agentic/*.md`
  to assess project state.
- **Reuse, don't re-derive:** existing analysis (e.g. `agentic/*Locomotion*`,
  prior experiment logs) is authoritative — reference it instead of redoing it.

## 5. Manual maintenance

`agentic/manual.md` is the framework's user manual. Update it whenever you add or
change scripts, CLI tools, task ids, or repo structure. Keep docs in sync with code.
