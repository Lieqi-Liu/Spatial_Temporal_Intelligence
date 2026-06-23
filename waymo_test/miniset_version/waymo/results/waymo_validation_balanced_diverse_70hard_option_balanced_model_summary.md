# Waymo Option-Balanced Miniset Model Summary

Benchmark: `waymo_validation_balanced_diverse_70hard_option_balanced_miniset_from_full_results.json`

This summary aggregates completed `*_metrics_bleurt.json` files in `result/`.

## Overall

| Model | Status | Responses | MCQ Count | Correct | Accuracy | Random Baseline | Invalid Rate | Missing | BLEURT Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LLaVA 1.5 13B | complete | 2400 | 1800 | 587 | 32.61% | 20.32% | 0.00% | 0 | -0.2929 |
| Qwen3-VL 8B | complete | 2400 | 1800 | 633 | 35.17% | 20.32% | 0.00% | 0 | -0.3476 |
| Qwen3-VL 30B-A3B | complete | 2400 | 1800 | 612 | 34.00% | 20.32% | 0.00% | 0 | -0.3173 |
| Qwen3-VL 32B | complete | 2400 | 1800 | 726 | 40.33% | 20.32% | 0.00% | 0 | -0.3449 |

## Per Task MCQ Accuracy

| Task | LLaVA 1.5 13B | Qwen3-VL 8B | Qwen3-VL 30B-A3B | Qwen3-VL 32B |
|---|---:|---:|---:|---:|
| scenario-corner-case | 38.00% (114/300) | 38.00% (114/300) | 34.00% (102/300) | 37.33% (112/300) |
| space-perception | 35.50% (71/200) | 42.50% (85/200) | 26.50% (53/200) | 59.00% (118/200) |
| time-extrapolation | 24.75% (99/400) | 27.25% (109/400) | 29.50% (118/400) | 29.50% (118/400) |
| time-memory | 25.00% (75/300) | 26.67% (80/300) | 29.67% (89/300) | 31.00% (93/300) |
| trajectory-prediction | 38.00% (228/600) | 40.83% (245/600) | 41.67% (250/600) | 47.50% (285/600) |

## Per Question Type MCQ Accuracy

| Question Type | LLaVA 1.5 13B | Qwen3-VL 8B | Qwen3-VL 30B-A3B | Qwen3-VL 32B |
|---|---:|---:|---:|---:|
| SC-1 | 48.00% (48/100) | 45.00% (45/100) | 44.00% (44/100) | 46.00% (46/100) |
| SC-2 | 36.00% (36/100) | 38.00% (38/100) | 28.00% (28/100) | 31.00% (31/100) |
| SC-3 | 30.00% (30/100) | 31.00% (31/100) | 30.00% (30/100) | 35.00% (35/100) |
| SP-2 | 33.00% (33/100) | 25.00% (25/100) | 22.00% (22/100) | 33.00% (33/100) |
| SP-3 | 38.00% (38/100) | 60.00% (60/100) | 31.00% (31/100) | 85.00% (85/100) |
| TE-1 | 20.00% (20/100) | 22.00% (22/100) | 30.00% (30/100) | 29.00% (29/100) |
| TE-2 | 31.00% (31/100) | 32.00% (32/100) | 26.00% (26/100) | 31.00% (31/100) |
| TE-4 | 18.00% (18/100) | 20.00% (20/100) | 36.00% (36/100) | 28.00% (28/100) |
| TE-5 | 30.00% (30/100) | 35.00% (35/100) | 26.00% (26/100) | 30.00% (30/100) |
| TM-2 | 20.00% (20/100) | 10.00% (10/100) | 21.00% (21/100) | 22.00% (22/100) |
| TM-3 | 24.00% (24/100) | 30.00% (30/100) | 32.00% (32/100) | 36.00% (36/100) |
| TM-5 | 31.00% (31/100) | 40.00% (40/100) | 36.00% (36/100) | 35.00% (35/100) |
| TRJ-1 | 60.00% (60/100) | 55.00% (55/100) | 46.00% (46/100) | 59.00% (59/100) |
| TRJ-2 | 55.00% (55/100) | 55.00% (55/100) | 50.00% (50/100) | 62.00% (62/100) |
| TRJ-3 | 32.00% (32/100) | 44.00% (44/100) | 54.00% (54/100) | 60.00% (60/100) |
| TRJ-4 | 33.00% (33/100) | 31.00% (31/100) | 36.00% (36/100) | 31.00% (31/100) |
| TRJ-8 | 31.00% (31/100) | 36.00% (36/100) | 29.00% (29/100) | 45.00% (45/100) |
| TRJ-10 | 17.00% (17/100) | 24.00% (24/100) | 35.00% (35/100) | 28.00% (28/100) |

## Special Numeric / FRQ Metrics

### SP-3a

| Model | count | parse_count | parse_rate | mae_m | rmse_m | median_abs_error_m | within_0_5m_rate | within_1_0m_rate | within_2_0m_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LLaVA 1.5 13B | 100.0000 | 100.0000 | 100.00% | 45.4410 | 56.2216 | 57.4000 | 25.00% | 26.00% | 26.00% |
| Qwen3-VL 8B | 100.0000 | 100.0000 | 100.00% | 4.0610 | 14.6670 | 0.3000 | 87.00% | 87.00% | 87.00% |
| Qwen3-VL 30B-A3B | 100.0000 | 98.0000 | 98.00% | 21.9673 | 31.4322 | 12.8000 | 11.22% | 12.24% | 12.24% |
| Qwen3-VL 32B | 100.0000 | 96.0000 | 96.00% | 1.5385 | 6.6636 | 0.3000 | 93.75% | 93.75% | 93.75% |

### TRJ-5

| Model | count | parse_count | parse_rate | exact_length_match_rate | mean_ade_m | mean_fde_m | min_ade_m | min_fde_m | median_ade_m | median_fde_m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LLaVA 1.5 13B | 100.0000 | 0.0000 | 0.00% | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Qwen3-VL 8B | 100.0000 | 100.0000 | 100.00% | 100.00% | 47.2627 | 58.9170 | 0.3404 | 0.3500 | 13.2355 | 22.0019 |
| Qwen3-VL 30B-A3B | 100.0000 | 100.0000 | 100.00% | 100.00% | 35.5614 | 46.1339 | 0.3404 | 0.3500 | 12.1708 | 21.5958 |
| Qwen3-VL 32B | 100.0000 | 100.0000 | 100.00% | 100.00% | 16.8079 | 27.7967 | 0.3404 | 0.3500 | 11.7854 | 20.2502 |

### TRJ-6

| Model | count | parse_count | parse_rate | exact_length_match_rate | mean_ade_m | mean_fde_m | min_ade_m | min_fde_m | median_ade_m | median_fde_m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LLaVA 1.5 13B | 100.0000 | 0.0000 | 0.00% | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Qwen3-VL 8B | 100.0000 | 100.0000 | 100.00% | 100.00% | 8.6415 | 14.3777 | 0.0068 | 0.0070 | 5.3744 | 10.1583 |
| Qwen3-VL 30B-A3B | 100.0000 | 97.0000 | 97.00% | 100.00% | 18.4853 | 27.4523 | 0.8983 | 0.9130 | 12.4733 | 19.6138 |
| Qwen3-VL 32B | 100.0000 | 100.0000 | 100.00% | 100.00% | 10.3874 | 17.1158 | 0.0269 | 0.0460 | 6.5868 | 10.9449 |

### TRJ-7

| Model | count | nonempty_count | nonempty_rate | empty_count | error_count |
|---|---:|---:|---:|---:|---:|
| LLaVA 1.5 13B | 100.0000 | 100.0000 | 100.00% | 0.0000 | 0.0000 |
| Qwen3-VL 8B | 100.0000 | 100.0000 | 100.00% | 0.0000 | 0.0000 |
| Qwen3-VL 30B-A3B | 100.0000 | 100.0000 | 100.00% | 0.0000 | 0.0000 |
| Qwen3-VL 32B | 100.0000 | 100.0000 | 100.00% | 0.0000 | 0.0000 |

## BLEURT By Text Question Type

| Question Type | LLaVA 1.5 13B | Qwen3-VL 8B | Qwen3-VL 30B-A3B | Qwen3-VL 32B |
|---|---:|---:|---:|---:|
| SC-4 | -0.3004 (n=100) | -0.3573 (n=100) | -0.3716 (n=100) | -0.3256 (n=100) |
| TRJ-9 | -0.2853 (n=100) | -0.3378 (n=100) | -0.2631 (n=100) | -0.3643 (n=100) |

## Files

- LLaVA 1.5 13B
  - responses: `waymo_validation_balanced_diverse_70hard_option_balanced_llava_1_5_13b_responses.json`
  - metrics: `waymo_validation_balanced_diverse_70hard_option_balanced_llava_1_5_13b_metrics_bleurt.json`
  - report: `waymo_validation_balanced_diverse_70hard_option_balanced_llava_1_5_13b_report_bleurt.md`
- Qwen3-VL 8B
  - responses: `waymo_validation_balanced_diverse_70hard_option_balanced_qwen3vl_8b_responses.json`
  - metrics: `waymo_validation_balanced_diverse_70hard_option_balanced_qwen3vl_8b_metrics_bleurt.json`
  - report: `waymo_validation_balanced_diverse_70hard_option_balanced_qwen3vl_8b_report_bleurt.md`
- Qwen3-VL 30B-A3B
  - responses: `waymo_validation_balanced_diverse_70hard_option_balanced_qwen3vl_30b_a3b_responses.json`
  - metrics: `waymo_validation_balanced_diverse_70hard_option_balanced_qwen3vl_30b_a3b_metrics_bleurt.json`
  - report: `waymo_validation_balanced_diverse_70hard_option_balanced_qwen3vl_30b_a3b_report_bleurt.md`
- Qwen3-VL 32B
  - responses: `waymo_validation_balanced_diverse_70hard_option_balanced_qwen3vl_32b_responses.json`
  - metrics: `waymo_validation_balanced_diverse_70hard_option_balanced_qwen3vl_32b_metrics_bleurt.json`
  - report: `waymo_validation_balanced_diverse_70hard_option_balanced_qwen3vl_32b_report_bleurt.md`