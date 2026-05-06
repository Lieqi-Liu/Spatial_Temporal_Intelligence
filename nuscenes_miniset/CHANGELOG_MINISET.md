# Miniset V2 Builder Change Notes

This records what `build_nuscenes_miniset_v2.py` originally did, what was missing, and what we changed.

## 1. Hardness Selection

### Originally implemented

- Used task-level VLM accuracy from `meta.per_task_metrics`.
- Lower-accuracy tasks were selected earlier and received a task-level hardness score.
- Did not use per-example VLM correctness from the `results` list.

### Concern

- All examples inside the same task got the same hardness signal.
- A hard missed example and an easy correct example within one task were treated similarly.

### Improvement

- Parse VLM per-example correctness from `results[*].is_correct` / `predicted_option`.
- Match VLM rows to candidate rows by task, scene, group, object, and question text.
- Add an example-level hardness bonus for examples significantly below their task average.
- Keep the original task-level hardness as a coarse prior.

## 2. Scene-Level Curation

### Originally implemented

- Scene diversity was mostly a soft score bonus/penalty.
- Nearby group reuse was penalized, but not strictly prevented.
- `SC-*` scene-level tasks used the same general selection logic as other tasks.

### Concern

- Scene-level questions can be highly correlated when many come from the same scene or same nearby group.
- Soft penalties do not guarantee enough spread across scenes.

### Improvement

- Enforce hard scene caps during selection.
- Add stricter `SC-*` controls:
  - max questions per scene within one SC task
  - max questions per scene across all SC tasks
  - max questions per exact scene/group pair
- This reduces repeated scene-level descriptions from the same scene or group neighborhood.

## 3. Representativeness Checks

### Originally implemented

- Summary reported total counts, per-task counts, scene count, and MCQ label balance.

### Concern

- It was hard to tell whether the miniset preserved useful coverage from the full set.
- There was no explicit audit for task retention, scene concentration, or group redundancy.

### Improvement

- Add summary diagnostics for:
  - per-task retention from full set to miniset
  - scene coverage fraction
  - scene concentration statistics
  - per-task scene coverage
  - exact group reuse and near-group redundancy

## 4. FRQ Inclusion

### Originally implemented

- Known FRQ tasks had zero count in the miniset summary or were skipped by MCQ-oriented logic.
- The builder only reindexed MCQ choices and did not have FRQ-specific answer diversity controls.

### Concern

- The miniset omitted important free-response tasks: `SP-7`, `SU-7`, `TE-6`, `TM-6`, `TRJ-7`, and `SC-6`.
- FRQs can become repetitive if selected only by scene diversity, because many answers are near-duplicates.

### Improvement

- Added `--frq-task-target` for FRQ quotas.
- Treat `SP-7`, `SU-7`, `TE-6`, `TM-6`, `TRJ-7`, and `SC-6` as FRQ tasks.
- Inject FRQ source rows from VLM `results` when the main input has no rows for that FRQ task.
- Skip MCQ answer reindexing for FRQs.
- Add FRQ answer-bucket caps using normalized ground-truth text.
- Add FRQ hardness from low model-response/ground-truth word overlap.
- Add FRQ diagnostics to the summary representativeness section.

## Notes

- MCQ answer reindexing still balances correct labels (`A/B/C/...`), not semantic answer meanings.
- Example hardness only works well when VLM result rows can be matched to candidate rows.
