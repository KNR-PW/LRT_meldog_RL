# Meldog RL — Prompt Library

Reusable prompts for driving Fable (design/research) and Opus/Gemini (execution).
See [/AGENTS.md](../AGENTS.md) §3 for the model tiering and the `[FABLE]`/`[EXEC]` tag.

**Strategy:** one goal per chat, not one mega-prompt. Sequence: foundation first
(done), then the feedback harness (unblocks everything), then the loops. Every
task a prompt produces must be tagged `[FABLE]` (design/research) or `[EXEC]`
(precisely specified — hand to Opus/Gemini with exact paths + current→proposed).

---

## Template (copy per chat)

```
Context: master's thesis — robot-centric terrain reconstruction from partial RGBD
on a 12-DOF quadruped (Meldog); Isaac Lab, sim-first but a real robot exists.

This chat's single goal: <one workstream only>.

Read first: /AGENTS.md, /directory.md, then <specific local paths>.
External references: use the Isaac Lab section of /directory.md (anymal / go2, flat+rough).

Deliverable: docs/<name>.md with sections —
  1. Findings
  2. Ranked recommendations (by impact / effort)
  3. Task list — tag EVERY task [FABLE] (open-ended design/research) or [EXEC]
     (precisely specified; hand to Opus/Gemini). For [EXEC]: exact file paths +
     current→proposed values. One-line effort estimate each.

Constraints: don't implement, don't train. Reference existing analysis/experiment
logs in agentic/ instead of re-deriving them. Cite the Isaac Lab file whenever you
copy a pattern. Flag uncertainties rather than guessing.

Checkpoint: show me the outline before writing the full doc.
```

---

## Chat 0 — Agent-agnostic foundation · `[EXEC]` · DONE
AGENTS.md, CLAUDE.md, GEMINI.md, directory.md, prompts.md created. Nothing to run.

---

## Chat 1 — Agent-consumable eval report (feedback harness) · `[FABLE]` design

The unblocker: makes locomotion + perception quality diagnosable without video.

```
Context: master's thesis — robot-centric terrain reconstruction from partial RGBD
on a 12-DOF quadruped (Meldog); Isaac Lab, sim-first but a real robot exists.

This chat's single goal: DESIGN an agent-consumable eval report so an agent with no
video access can diagnose locomotion + perception quality. I stay in the loop on
video; you make everything else quantitative.

Read first: /AGENTS.md, /directory.md, source/meldog_rl/envs/meldog_env.py,
scripts/locomotion/evaluate_locomotion.py, scripts/perception/evaluate_perception.py,
and any agentic/*Locomotion* analysis. Reference: Isaac Lab anymal/go2
rough_env_cfg.py reward + observation terms (see /directory.md) for which signals
are worth logging.

Deliverable: docs/eval-report-design.md —
  1. Locomotion metrics: which scalars + which plots. Expect a gait/contact diagram,
     velocity-tracking error, base height/orientation, feet slip, joint-limit
     saturation, action smoothness, power. Define a PNG panel + a JSON schema,
     identical for every run.
  2. Perception metrics: predicted vs ground-truth heightmap + an error heatmap, plus
     scalars (RMSE, completeness, coverage) as JSON. Specify the exact image so it is
     readable to a multimodal model (top-down, colorbar), not a raw viewport grab.
  3. Task list, every item tagged [FABLE] or [EXEC], with file paths for [EXEC].

Constraints: don't implement, don't train. Cite the Isaac Lab file for any borrowed
signal. Flag uncertainties.

Checkpoint: show me the metric list + JSON schema before writing the full doc.
```

---

## Chat 2 — Repo structure critique + branch cleanup · `[EXEC]` + your decisions

```
Context: <template context>.

This chat's single goal: critique the repo structure and plan the branch
consolidation. Each implementation lived on its own branch; the `cleanup` branch
consolidates them and the pipeline works. Goal: a proper `main`, merge cleanup,
retire the rest, adopt a normal SE workflow.

Read first: /AGENTS.md, /directory.md, README.md, the repo tree, and
`git branch -a` + `git log --oneline --graph --all`. Note "legacy code structure/".

Deliverable: docs/repo-cleanup-plan.md —
  1. Findings: structure smells, local-files-vs-Isaac-logs boundary, dead/legacy dirs.
  2. Ranked recommendations.
  3. Task list [FABLE]/[EXEC]: exact git commands (merge/tag/delete), which branches
     die vs archive, .gitignore fixes, files to move/remove. Flag every DESTRUCTIVE
     step for my explicit approval before it runs.

Constraints: don't execute git yet — plan only. Don't delete anything. Flag risks.

Checkpoint: show me the branch inventory + kill/keep list before the full plan.
```

---

## Chat 3 — Locomotion improvement experiment plan · `[FABLE]` → loop

The diagnosis exists — turn it into ranked experiments, don't re-analyze.

```
Context: <template context>. Locomotion is usable but unnatural (asymmetric gait,
sliding feet, unnatural angles, too-quick motions) and likely fails on hardware.

This chat's single goal: turn the EXISTING locomotion analysis into a prioritized,
runnable experiment plan. Do not re-diagnose from scratch.

Read first: /AGENTS.md, /directory.md, the agentic/*Locomotion* analysis,
source/meldog_rl/envs/configs/base_cfg.py, source/meldog_rl/agents/rsl_rl_ppo_cfg.py.
Reference: Isaac Lab anymal_c + go2 rough_env_cfg.py and agents/rsl_rl_ppo_cfg.py
(feet_slide penalty, action_scale, velocity limits, obs noise, entropy, command
resampling) — cite exact values.

Deliverable: docs/locomotion-experiments.md —
  1. Findings: restate root causes with the reference value we deviate from.
  2. Ranked experiments by expected impact / cost. Each: hypothesis, exact param
     (file:line, current→proposed), the eval-report metric that confirms/refutes it.
  3. Task list [FABLE]/[EXEC]. Param edits + training runs are [EXEC] (Opus/Gemini).

Constraints: don't implement, don't train. One-variable-at-a-time experiments.
This is a loop — expect to revise after each run's eval report.

Checkpoint: show me the ranked experiment list before the full doc.
```

---

## Chat 4 — Thesis review + literature · `[FABLE]` (separate track)

```
Context: <template context>.

This chat's single goal: review the thesis for errors, gaps, and future directions,
grounded in the literature.

Read first: /AGENTS.md, thesis/chapters/*, literature/*. Use the deep-research skill
for external terrain-reconstruction / quadruped-perception papers (ANYmal-lineage).

Deliverable: docs/thesis-review.md —
  1. Findings: technical errors, unsupported claims, structural/clarity gaps.
  2. Ranked recommendations, each tied to a chapter/section + a citation.
  3. Future directions with literature support.
  4. Task list [FABLE]/[EXEC] (writing/experiments to strengthen the thesis).

Constraints: don't rewrite the thesis — critique and cite. Distinguish "must fix"
from "nice to have". Flag anything you cannot verify against a source.

Checkpoint: show me the findings outline before the full review.
```
