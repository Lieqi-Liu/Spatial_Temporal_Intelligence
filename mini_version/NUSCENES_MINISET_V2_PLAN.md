# NuScenes Miniset V2 Design

## Goal

Build a new NuScenes miniset generator focused on:

1. more diverse scenes
2. cover as many distinct scenes as possible
3. fewer redundant nearby groups from the same scene
4. question-level representativeness as the primary objective
5. improved MCQ answer-option balance through choice reindexing
6. targeted emphasis on harder questions using prior VLM evaluation results


## Primary Inputs

### Required

- `/local1/rgao727/nuscenes/questions_with_answers_trainval.json`

This is the main source of:

- task id
- question text
- choices
- ground truth
- scene id
- group id
- source group file

### Optional but strongly preferred

- `/home/rgao727/Spatial_Temporal_Intelligence/vlm_responses_20260331_210718.json`

Intended use:

- identify task types with lower model accuracy
- optionally identify answer distributions that are especially skewed

Note:

- the current local file appears malformed or truncated when parsed as full JSON
- however, its header clearly contains `meta.per_task_metrics`
- the future implementation should therefore:
  - try to parse it normally first
  - if parsing fails, either skip it gracefully or salvage the `meta.per_task_metrics` block if possible


## High-Level Selection Philosophy

This miniset is **question-first**, not group-first.

That means:

- we do not require every selected question to come from the same set of groups
- we do not require all tasks to share the same groups
- we do want the final question counts per task to be controlled directly

At the same time, scene diversity still matters, so the selection process should discourage:

- too many questions from the same scene
- too many nearby groups from the same scene


## Desired Output Size

### Scene-level tasks

For scene-level questions, select:

- `400` questions per task

This mainly applies to:

- `SC-*`

Reason:

- `SC-*` questions are NuScenes-specific and do not need to be matched directly with Waymo
- therefore a larger NuScenes-only scene-level pool is acceptable

### Non-scene tasks

For all other task families, select:

- `200` questions per task

This includes:

- `SP-*`
- `SU-*`
- `TE-*`
- `TM-*`
- `TRJ-*`

Reason:

- these non-`SC` task families are the ones we more likely want to align with Waymo later
- so their size should stay smaller and more comparable

If a task has fewer than the target count available after filtering, keep all available questions and record that shortfall in the summary.


## Core Requirements

### 1. Scene diversity and broad scene coverage

The selected set should cover as many different scenes as reasonably possible.

The desired behavior is not just "diverse scenes in aggregate", but:

- try to touch nearly all scenes if possible
- avoid spending too many slots repeatedly inside a small subset of scenes
- only revisit the same scene when doing so materially improves balance or difficulty coverage

Selection should penalize:

- repeatedly sampling from the same scene
- repeatedly sampling from adjacent or highly overlapping groups inside the same scene

Recommended treatment:

- maintain per-scene selected count
- prefer questions from scenes with lower current coverage
- if two candidate questions come from the same scene and nearby group ids, avoid selecting both unless necessary for balance
- add a strong bonus for scenes not yet covered at all
- after first-pass broad coverage, allow additional questions from a scene only if they improve per-task balance or hardness coverage

Possible heuristic:

- define nearby groups as group indices within a small distance such as `|group_idx_a - group_idx_b| <= 2`
- apply a redundancy penalty when selecting multiple nearby groups from the same scene


### 2. Representative and challenging questions

Questions should remain representative of the benchmark while leaning slightly toward harder cases.

Use prior evaluation information to prioritize:

- task types with lower accuracy
- answer patterns the model often gets wrong

Possible usage of the VLM result file:

- read `meta.per_task_metrics[task].accuracy`
- assign a hardness bonus to tasks with lower accuracy
- optionally prefer examples whose ground-truth answers belong to underperforming or underrepresented answer choices

This should be a soft preference, not a hard rule.


### 3. More even MCQ answer distribution

For MCQ tasks, the final selected set should be closer to balanced across answer options.

There are two parts:

#### 3a. Selection-time balancing

Prefer candidate questions whose current ground-truth choice helps close answer imbalance for that task.

#### 3b. Post-selection choice reindexing

If the semantic content allows it, remap the order of answer choices so the final labeled answers are more evenly distributed across `A/B/C/...`.

This means:

- reorder the `choices`
- update `ground_truth` accordingly
- preserve semantic correctness

Target behavior:

- the final correct-option label distribution should be as close to even as practical
- e.g. avoid having most correct answers concentrated in `A` or `B`
- use permutation of answer order to help flatten the final label histogram

Important:

- reindexing must only permute the option order
- it must not alter the actual content of the correct answer
- it must leave the question meaning unchanged


## Selection Unit

The actual selected record is a **question**, not a group.

However, each candidate question still carries:

- `scene_id`
- `group_id`
- `source_group_file`

These fields should be retained in the output so the corresponding images can be found later.


## Proposed Selection Procedure

### Step 1. Build candidate pools by task

For each task id:

- gather all candidate questions
- keep their metadata:
  - scene id
  - group id
  - source group file
  - question text
  - choices
  - ground truth

### Step 2. Score each candidate question

Each candidate gets a composite score combining:

- representativeness score
- hardness score
- answer-balance gain
- scene-diversity gain
- group-redundancy penalty

Conceptually:

`final_score = hardness_bonus + answer_balance_gain + scene_diversity_gain - redundancy_penalty`

### Step 3. Greedy per-task selection

For each task:

- select until reaching the target count
- always update running statistics:
  - answer distribution for this task
  - scene usage count
  - nearby-group usage count

### Step 4. Choice reindexing pass

After question selection:

- run an answer-choice permutation pass for each MCQ task
- try to reduce residual imbalance across `A/B/C/...`
- preserve semantic equivalence

### Step 5. Build summary

Report:

- per-task selected count
- per-task answer distribution before and after reindexing
- scene coverage count
- per-scene selected question count
- redundancy statistics for same-scene nearby-group reuse
- whether the VLM difficulty prior was used


## Outputs

The new script should eventually produce at least:

### 1. Selected questions JSON

Contains the final selected questions and ground truths.

Required fields per item:

- `id`
- `task`
- `question_format`
- `question`
- `choices`
- `ground_truth`
- `scene_id`
- `group_id`
- `source_group_file`
- original answer label before reindexing if reindexing is used
- new answer label after reindexing if reindexing is used

This selected output is intended to be the final usable NuScenes miniset question file:

- selected questions
- selected ground truths
- reindexed answer choices for better label balance

### 2. Summary JSON

Contains:

- total question count
- per-task selected count
- scene coverage
- scene distribution
- answer distribution by task
- balance improvement statistics
- low-accuracy task usage statistics
- redundancy metrics


## Key Differences from the Previous Miniset Builder

Compared with the earlier metadata-driven group-oriented miniset logic:

- this new version is **question-centric**
- scene diversity is enforced as a soft global constraint
- group uniqueness is encouraged but not absolute
- target counts are controlled per task
- answer balance is improved both by selection and by answer-order permutation
- prior VLM performance is explicitly used as a difficulty prior


## Open Implementation Notes

### VLM file robustness

Because the current `vlm_responses_20260331_210718.json` appears malformed locally, the implementation should not crash if this file cannot be parsed.

Recommended behavior:

- `--vlm-metrics-json` optional
- if unavailable or invalid, continue without hardness prior
- summary should explicitly record whether VLM prior was used

### SP-3a / TRJ numeric tasks

These are not MCQ option-balance tasks.

So:

- include them in per-task count control
- do not apply choice reindexing
- still apply scene diversity and redundancy controls


## Intended Outcome

The final NuScenes miniset should:

- cover more diverse scenes
- avoid overconcentration in one scene or adjacent groups
- preserve representative question content
- emphasize harder questions where useful
- achieve a much more even MCQ answer-label distribution
- keep direct links to the original images through `scene_id`, `group_id`, and `source_group_file`
