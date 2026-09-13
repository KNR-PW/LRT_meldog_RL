# Video Review Rubric

A short checklist for judging a policy or a reconstruction from video. Fill it in
**before** reading `metrics.json` or `report.md`, so your visual judgement stays
independent of the numbers. Anything surprising belongs in the free-text line; it's how
missing metrics get discovered.

---

## Locomotion

```markdown
### Rubric: <task-id> @ <checkpoint> — <date>, by <name>
- Video/source: <path>
- Gait symmetry (1 = chaotic, 5 = clean regular gait): _
- Movement speed looks natural (not twitchy or rushed)? y/n: _
- Persistent body tilt (constant lean or roll)? y/n: _
- Orientation stable (little wobble around that lean)? y/n: _
- Foot slip visible? y/n: _
- Harsh impacts (feet fly up or smash down)? y/n: _
- Obstacles: steps over / walks around / collides / falls: _
- Free text: _
```

## Perception

```markdown
### Rubric: <model or baseline> @ <checkpoint> on <task-id> — <date>, by <name>
- Video/source: <path>
- Reconstructed geometry plausible (shapes match the terrain)? y/n: _
- Obstacles clearly visible in the map? y/n: _
- Memory holds behind and beside the robot (map doesn't fade where unseen)? y/n: _
- Artifacts and noise (1 = unusable, 5 = clean): _
- Free text: _
```

---

**Checking the metrics against your eyes:** fill the rubric from the video only, then
fill a second copy from `metrics.json` and the plots only, and compare. Where they agree,
that question can be answered from the metrics in future. Where they disagree, a metric
is missing or its range in [evaluation.md](evaluation.md) is off.
