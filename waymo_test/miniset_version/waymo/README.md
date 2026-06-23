# Waymo Validation Miniset Release

This folder contains the final Waymo validation miniset used in the reported experiments, together with human review records, evaluation scripts, and model outputs.

The release is intentionally compact: it includes the final selected benchmark and results, but excludes raw Waymo images, full-set intermediate files, logs, caches, and older miniset variants.

## Dataset

The final benchmark file is:

- `dataset/waymo_validation_balanced_diverse_70hard_option_balanced_miniset_from_full_results.json`

It contains 2,400 questions: 100 questions for each of 24 question types.

| Split | Count |
|---|---:|
| Total questions | 2,400 |
| Multiple-choice questions | 1,800 |
| Free-response / numeric questions | 600 |
| Question types | 24 |
| Questions per type | 100 |

Task-level composition:

| Task | Question Types | Count |
|---|---|---:|
| Scenario / corner-case reasoning | `SC-1`, `SC-2`, `SC-3`, `SC-4` | 400 |
| Spatial perception | `SP-2`, `SP-3`, `SP-3a` | 300 |
| Temporal extrapolation | `TE-1`, `TE-2`, `TE-4`, `TE-5` | 400 |
| Temporal memory | `TM-2`, `TM-3`, `TM-5` | 300 |
| Trajectory prediction / reasoning | `TRJ-1`, `TRJ-2`, `TRJ-3`, `TRJ-4`, `TRJ-5`, `TRJ-6`, `TRJ-7`, `TRJ-8`, `TRJ-9`, `TRJ-10` | 1,000 |

Question groups:

- `SC-*`: scenario and corner-case understanding
- `SP-*`: spatial perception
- `TE-*`: temporal extrapolation
- `TM-*`: temporal memory
- `TRJ-*`: trajectory prediction and trajectory reasoning

MCQ question types:

- `SC-1`, `SC-2`, `SC-3`
- `SP-2`, `SP-3`
- `TE-1`, `TE-2`, `TE-4`, `TE-5`
- `TM-2`, `TM-3`, `TM-5`
- `TRJ-1`, `TRJ-2`, `TRJ-3`, `TRJ-4`, `TRJ-8`, `TRJ-10`

FRQ / numeric question types:

- `SC-4`
- `SP-3a`
- `TRJ-5`, `TRJ-6`, `TRJ-7`, `TRJ-9`

The corresponding summary is:

- `dataset/waymo_validation_balanced_diverse_70hard_option_balanced_miniset_from_full_results_summary.json`

More dataset details are in:

- `docs/waymo_miniset_dataset_card.md`

## How The Miniset Was Obtained

The starting point was a full Waymo validation question set generated from front-camera clips. For most non-SC questions, each visual context uses 5 frames sampled over about 2.0 seconds. SC questions preserve the selected scene-level visual context.

The final miniset was selected from the full validation set with these goals:

- 100 examples per question type
- scene diversity first
- scenario / ground-truth diversity
- preference for complex or corner-case scenes
- approximately 70% hard and 30% correct MCQ examples when available, based on full-set model responses
- balanced correct option letters for MCQ questions by permuting answer choices without changing the selected question rows

The option-balanced file is the final version used for model comparison.

The full reproduction workflow is documented in:

- `docs/reproduction_workflow.md`

## Human Review Records

The annotation files are:

- `annotations/sc_scene_summary_review_records.json`
- `annotations/waymo_sc_trj_review_records.json`

`sc_scene_summary_review_records.json` stores manually reviewed scene summaries, including approved or revised scene descriptions, key attention regions, main risk sources, and preferred actions.

`waymo_sc_trj_review_records.json` stores manually reviewed SC/TRJ labels and question-level revisions. Records include `review_status` values such as `approved` and `revised_approved`.

The final miniset JSON also embeds relevant review metadata in each selected question when available.

Manual review coverage in the final miniset:

| Review Source | Covered Question Types | Reviewed Rows In Final Miniset |
|---|---|---:|
| Scene-summary review | `SC-1`, `SC-2`, `SC-3`, `SC-4` | 400 |
| SC/TRJ label review | `SC-1`, `TRJ-8`, `TRJ-9`, `TRJ-10` | 301 |

Breakdown:

| Question Type | Review Source | Approved | Revised Approved | Total Reviewed |
|---|---|---:|---:|---:|
| `SC-1` | scene summary | 30 | 70 | 100 |
| `SC-2` | scene summary | 28 | 72 | 100 |
| `SC-3` | scene summary | 25 | 75 | 100 |
| `SC-4` | scene summary | 27 | 73 | 100 |
| `SC-1` | SC/TRJ label review | 23 | 77 | 100 |
| `TRJ-8` | SC/TRJ label review | 1 | 0 | 1 |
| `TRJ-9` | SC/TRJ label review | 98 | 2 | 100 |
| `TRJ-10` | SC/TRJ label review | 4 | 96 | 100 |

## Results

The aggregate result table is:

- `results/waymo_validation_balanced_diverse_70hard_option_balanced_model_summary.md`

Per-model JSON outputs are stored as:

- `*_responses.json`: raw model responses
- `*_metrics_bleurt.json`: scored metrics, including MCQ accuracy, special numeric metrics, and BLEURT when available
- `*_report_bleurt.md`: human-readable model report

Evaluated models include:

- LLaVA 1.5 13B
- Qwen3-VL 8B
- Qwen3-VL 30B-A3B
- Qwen3-VL 32B

Overall result snapshot:

| Model | MCQ Accuracy | Random Baseline | Invalid Rate | BLEURT Mean |
|---|---:|---:|---:|---:|
| LLaVA 1.5 13B | 32.61% | 20.32% | 0.00% | -0.2929 |
| Qwen3-VL 8B | 35.17% | 20.32% | 0.00% | -0.3476 |
| Qwen3-VL 30B-A3B | 34.00% | 20.32% | 0.00% | -0.3173 |
| Qwen3-VL 32B | 40.33% | 20.32% | 0.00% | -0.3449 |

Per-task MCQ accuracy:

| Task | LLaVA 1.5 13B | Qwen3-VL 8B | Qwen3-VL 30B-A3B | Qwen3-VL 32B |
|---|---:|---:|---:|---:|
| Scenario / corner-case | 38.00% | 38.00% | 34.00% | 37.33% |
| Spatial perception | 35.50% | 42.50% | 26.50% | 59.00% |
| Temporal extrapolation | 24.75% | 27.25% | 29.50% | 29.50% |
| Temporal memory | 25.00% | 26.67% | 29.67% | 31.00% |
| Trajectory prediction / reasoning | 38.00% | 40.83% | 41.67% | 47.50% |

Special FRQ / numeric metrics are also included in the aggregate result file, including `SP-3a` distance errors, `TRJ-5` / `TRJ-6` trajectory ADE/FDE, and BLEURT for `SC-4` and `TRJ-9`.

## Scripts

Important Python and bash scripts are included under `scripts/`.

- `scripts/generation/`: Waymo image/frame export, clip bundle construction, GT generation, and review-record merging.
- `scripts/selection/`: full-set construction, final miniset selection, and MCQ option-letter balancing.
- `scripts/evaluation/`: vLLM/Transformers inference and scoring scripts for full-set and miniset evaluation.

The most important scripts for using the released miniset are:

- `scripts/evaluation/run_waymo_miniset_vllm.py`
- `scripts/evaluation/score_waymo_miniset_mcq.py`
- `scripts/selection/select_miniset_from_full_results.py`
- `scripts/selection/rebalance_mcq_option_letters.py`

## Notes

Image files are not included in this release folder. The question JSON contains local image paths from the original workspace, so another environment must either reproduce the same image layout or remap those paths before running evaluation.

Logs, caches, intermediate miniset versions, and old selection variants are intentionally excluded from this release folder.
