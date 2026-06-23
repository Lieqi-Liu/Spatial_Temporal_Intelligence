# Waymo Validation Miniset Release

This folder contains the final Waymo validation miniset used in the reported experiments, together with human review records and model evaluation outputs.

## Dataset

The final benchmark file is:

- `dataset/waymo_validation_balanced_diverse_70hard_option_balanced_miniset_from_full_results.json`

It contains 2,400 questions: 100 questions for each of 24 question types.

Question groups:

- `SC-*`: scenario and corner-case understanding
- `SP-*`: spatial perception
- `TE-*`: temporal extrapolation
- `TM-*`: temporal memory
- `TRJ-*`: trajectory prediction and trajectory reasoning

The split contains:

- 1,800 multiple-choice questions
- 600 free-response / numeric questions

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
