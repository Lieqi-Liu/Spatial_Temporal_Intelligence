# Waymo Validation Miniset Dataset Card

## Overview

This release contains a compact evaluation subset derived from the Waymo validation split. It is designed for evaluating spatial-temporal visual reasoning in autonomous-driving scenes.

The final benchmark is:

`dataset/waymo_validation_balanced_diverse_70hard_option_balanced_miniset_from_full_results.json`

It contains 2,400 total questions, with 100 examples for each of 24 question types.

## Data Source

The questions were derived from Waymo validation front-camera visual clips and associated metadata. The image files are not included in this release. The JSON keeps the original local image paths used during generation and evaluation.

For most temporal, spatial, and trajectory questions, the visual context uses 5 frames sampled over approximately 2.0 seconds. Scene-level questions use the scene context selected during the scene-summary and review pipeline.

## Question Types

- `SC-1`, `SC-2`, `SC-3`, `SC-4`: scene category, risk, action, and scene-level reasoning
- `SP-2`, `SP-3`, `SP-3a`: spatial perception and distance reasoning
- `TE-1`, `TE-2`, `TE-4`, `TE-5`: temporal extrapolation
- `TM-2`, `TM-3`, `TM-5`: temporal memory
- `TRJ-1`, `TRJ-2`, `TRJ-3`, `TRJ-4`, `TRJ-5`, `TRJ-6`, `TRJ-7`, `TRJ-8`, `TRJ-9`, `TRJ-10`: trajectory prediction, trajectory choice, and trajectory explanation

## Final Selection Procedure

The miniset was selected from the full Waymo validation question set with a scene-diverse and hard-example-aware policy.

Selection goals:

- keep exactly 100 questions per question type
- prioritize diverse scenes
- preserve scenario and ground-truth diversity
- prefer complex driving cases such as turns, merges, intersections, traffic interactions, occlusions, and unusual road situations
- target approximately 70% hard and 30% correct MCQ examples based on a full-set model run when the candidate pool allowed it
- balance correct answer option letters for MCQ questions by permuting choices, without changing the selected rows

The selected rows are therefore fixed; only MCQ option order was changed in the option-balanced version.

## Human Annotation And Review

Two human-review JSON files are included:

- `annotations/sc_scene_summary_review_records.json`
- `annotations/waymo_sc_trj_review_records.json`

`sc_scene_summary_review_records.json` contains scene-level review records. Important fields include:

- `review_status`
- `raw_sc1_label`
- `canonical_sc1_label`
- `original_scene_description`
- `revised_scene_description`
- `original_key_attention`
- `revised_key_attention`
- `original_main_risk_source`
- `revised_main_risk_source`
- `original_preferred_action`
- `revised_preferred_action`

`waymo_sc_trj_review_records.json` contains reviewed SC/TRJ labels and question revisions. Important fields include:

- `task_id`
- `review_status`
- `original_ground_truth`
- `revised_ground_truth`
- `original_ground_truth_text`
- `revised_ground_truth_text`
- `revised_question_text`
- `revised_alternative_maneuver`

The final benchmark JSON also embeds relevant review metadata in `hidden_metadata`.

## Evaluation Outputs

Model outputs and metrics are in `results/`.

Included model families:

- LLaVA 1.5 13B
- Qwen3-VL 8B
- Qwen3-VL 30B-A3B
- Qwen3-VL 32B

Metrics include:

- MCQ accuracy and random baseline
- invalid prediction rate
- BLEURT for text FRQ questions
- `SP-3a` numeric distance metrics
- `TRJ-5` and `TRJ-6` trajectory ADE/FDE metrics
- `TRJ-7` non-empty / error status

The aggregate report is:

`results/waymo_validation_balanced_diverse_70hard_option_balanced_model_summary.md`

## Reuse Notes

To rerun model evaluation, users need access to the corresponding Waymo image frames and must remap the image paths in the benchmark JSON if their local layout differs.

The release intentionally excludes raw logs, caches, generated intermediate files, and earlier miniset variants.
