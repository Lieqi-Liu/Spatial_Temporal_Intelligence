# Reproduction Workflow

This document explains the important scripts included in this release and the high-level workflow used to obtain the final Waymo miniset and evaluation results.

The release contains the final JSON artifacts, so rerunning every upstream generation step is not required unless you want to rebuild the dataset from raw Waymo data.

## 1. From Waymo Data To Visual Clips

Relevant scripts:

- `scripts/generation/export_waymo_front_images.py`
- `scripts/generation/build_waymo_5frame_bundles.py`

Purpose:

- export front-camera images from Waymo validation data
- group frames into visual clips / bundles
- build 5-frame contexts used by the question generation pipeline

The final miniset JSON stores image paths produced by this stage. If your local data path differs, remap the image paths before evaluation.

## 2. Generate And Merge Question Ground Truth

Relevant scripts:

- `scripts/generation/generate_waymo_e2e_5frame_gt.py`
- `scripts/generation/merge_review_records_into_gt.py`
- `scripts/generation/merge_trj9_explanations_into_gt.py`

Purpose:

- generate Waymo question records with ground truth
- merge manually reviewed scene-level and trajectory-level labels
- merge trajectory explanation data for text reasoning questions

Human review records included in this release:

- `annotations/sc_scene_summary_review_records.json`
- `annotations/waymo_sc_trj_review_records.json`

These are the key files documenting the manually approved or revised SC/TRJ annotations.

## 3. Build Full Validation Question Set

Relevant script:

- `scripts/selection/build_waymo_validation_full_set.py`

Purpose:

- convert the reviewed/generated Waymo question records into the full validation question set
- preserve question metadata, image paths, task IDs, ground truth, and review metadata

The full set is not included in this lightweight release because it is large. The final selected miniset is included instead.

## 4. Run Full-Set Model Responses

Relevant script:

- `scripts/evaluation/run_waymo_full_set_vllm.py`

Purpose:

- run a reference VLM on the full validation set
- obtain model responses used to estimate which MCQ examples are hard or easy

The final miniset selection used those full-set responses to target approximately 70% hard and 30% correct MCQ examples when available.

## 5. Select The Final Miniset

Relevant scripts:

- `scripts/selection/select_miniset_from_full_results.py`
- `scripts/selection/rebalance_mcq_option_letters.py`

The selection policy prioritizes:

- 100 examples per question type
- scene diversity
- scenario and ground-truth diversity
- complex and corner-case situations
- approximately 70% hard / 30% correct MCQ examples

After row selection, MCQ answer choices were permuted so that correct option letters are balanced within each question type. The selected question rows themselves were not changed during option balancing.

Final output:

- `dataset/waymo_validation_balanced_diverse_70hard_option_balanced_miniset_from_full_results.json`
- `dataset/waymo_validation_balanced_diverse_70hard_option_balanced_miniset_from_full_results_summary.json`

## 6. Run Miniset Evaluation

Relevant scripts:

- `scripts/evaluation/run_waymo_miniset_vllm.py`
- `scripts/evaluation/score_waymo_miniset_mcq.py`
- `scripts/evaluation/run_qwen3vl_30b_a3b_option_balanced.sh`
- `scripts/evaluation/run_qwen3vl_32b_option_balanced.sh`

Example Qwen command:

```bash
cd /path/to/github_release

python3 scripts/evaluation/run_waymo_miniset_vllm.py \
  --benchmark-json dataset/waymo_validation_balanced_diverse_70hard_option_balanced_miniset_from_full_results.json \
  --output-json results/waymo_validation_balanced_diverse_70hard_option_balanced_qwen3vl_32b_responses.json \
  --model Qwen/Qwen3-VL-32B-Instruct \
  --backend vllm \
  --gpu-ids 4,5 \
  --tensor-parallel-size 2 \
  --max-tokens-frq 256 \
  --score-after-run \
  --enable-bleurt \
  --bleurt-device cuda \
  --hf-home /local1/rgao727/huggingface \
  --score-output-json results/waymo_validation_balanced_diverse_70hard_option_balanced_qwen3vl_32b_metrics_bleurt.json \
  --score-output-md results/waymo_validation_balanced_diverse_70hard_option_balanced_qwen3vl_32b_report_bleurt.md \
  --score-output-csv results/waymo_validation_balanced_diverse_70hard_option_balanced_qwen3vl_32b_predictions_bleurt.csv
```

For LLaVA 1.5 13B, use:

```bash
python3 scripts/evaluation/run_waymo_miniset_vllm.py \
  --benchmark-json dataset/waymo_validation_balanced_diverse_70hard_option_balanced_miniset_from_full_results.json \
  --output-json results/waymo_validation_balanced_diverse_70hard_option_balanced_llava_1_5_13b_responses.json \
  --model llava-hf/llava-1.5-13b-hf \
  --backend transformers_llava \
  --gpu-ids 2,3 \
  --max-tokens-frq 256 \
  --score-after-run \
  --enable-bleurt \
  --bleurt-device cuda \
  --hf-home /local1/rgao727/huggingface \
  --score-output-json results/waymo_validation_balanced_diverse_70hard_option_balanced_llava_1_5_13b_metrics_bleurt.json \
  --score-output-md results/waymo_validation_balanced_diverse_70hard_option_balanced_llava_1_5_13b_report_bleurt.md \
  --score-output-csv results/waymo_validation_balanced_diverse_70hard_option_balanced_llava_1_5_13b_predictions_bleurt.csv
```

## 7. Metrics

`score_waymo_miniset_mcq.py` reports:

- MCQ accuracy and random baseline
- invalid / missing prediction rate
- BLEURT for text FRQ questions
- `SP-3a` numeric distance MAE/RMSE and threshold accuracy
- `TRJ-5` and `TRJ-6` trajectory ADE/FDE
- `TRJ-7` non-empty and error status

Aggregate result:

- `results/waymo_validation_balanced_diverse_70hard_option_balanced_model_summary.md`

## Included Script Groups

`scripts/generation/`

- upstream data extraction, bundle construction, and GT/review merging

`scripts/selection/`

- full-set construction, selected miniset creation, and option balancing

`scripts/evaluation/`

- full-set and miniset inference/scoring scripts

## Practical Notes

- Image data is not included.
- Full-set JSON files are not included because they are large.
- The included final miniset JSON is enough to inspect questions, ground truth, review metadata, and model results.
- To actually rerun inference, update image paths or reproduce the original image directory layout.
