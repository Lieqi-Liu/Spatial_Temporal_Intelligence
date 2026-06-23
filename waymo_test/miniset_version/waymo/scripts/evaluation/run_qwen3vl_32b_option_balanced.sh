#!/usr/bin/env bash
set -euo pipefail

cd /home/rgao727/Spatial_Temporal_Intelligence/waymo_test/miniset_version

python3 run_waymo_miniset_vllm.py \
  --benchmark-json waymo_validation_balanced_diverse_70hard_option_balanced_miniset_from_full_results.json \
  --output-json result/waymo_validation_balanced_diverse_70hard_option_balanced_qwen3vl_32b_responses.json \
  --model Qwen/Qwen3-VL-32B-Instruct \
  --backend vllm \
  --gpu-ids 4,5 \
  --tensor-parallel-size 2 \
  --max-tokens-frq 256 \
  --score-after-run \
  --enable-bleurt \
  --bleurt-device cuda \
  --hf-home /local1/rgao727/huggingface \
  --score-output-json result/waymo_validation_balanced_diverse_70hard_option_balanced_qwen3vl_32b_metrics_bleurt.json \
  --score-output-md result/waymo_validation_balanced_diverse_70hard_option_balanced_qwen3vl_32b_report_bleurt.md \
  --score-output-csv result/waymo_validation_balanced_diverse_70hard_option_balanced_qwen3vl_32b_predictions_bleurt.csv
