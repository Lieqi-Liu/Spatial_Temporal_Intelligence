# Waymo Miniset Report

- Benchmark: `waymo_validation_balanced_diverse_70hard_option_balanced_miniset_from_full_results.json`
- Responses: `result/waymo_validation_balanced_diverse_70hard_option_balanced_qwen3vl_8b_responses.json`
- Score source: `benchmark`
- MCQ count: `1800`
- Correct: `633`
- Accuracy: **35.17%**
- Random baseline: `20.32%`
- Invalid/unparsed predictions: `0` (0.00%)
- Missing responses: `0`

## Per Question Type

| Question ID | Count | Accuracy | Random Baseline | Invalid |
|---|---:|---:|---:|---:|
| `SC-1` | 100 | 45.00% | 25.00% | 0 |
| `SC-2` | 100 | 38.00% | 20.00% | 0 |
| `SC-3` | 100 | 31.00% | 20.00% | 0 |
| `SP-2` | 100 | 25.00% | 12.50% | 0 |
| `SP-3` | 100 | 60.00% | 20.00% | 0 |
| `TE-1` | 100 | 22.00% | 20.00% | 0 |
| `TE-2` | 100 | 32.00% | 20.00% | 0 |
| `TE-4` | 100 | 20.00% | 16.67% | 0 |
| `TE-5` | 100 | 35.00% | 20.00% | 0 |
| `TM-2` | 100 | 10.00% | 16.67% | 0 |
| `TM-3` | 100 | 30.00% | 16.67% | 0 |
| `TM-5` | 100 | 40.00% | 25.00% | 0 |
| `TRJ-1` | 100 | 55.00% | 16.67% | 0 |
| `TRJ-10` | 100 | 24.00% | 25.00% | 0 |
| `TRJ-2` | 100 | 55.00% | 16.67% | 0 |
| `TRJ-3` | 100 | 44.00% | 16.67% | 0 |
| `TRJ-4` | 100 | 31.00% | 25.00% | 0 |
| `TRJ-8` | 100 | 36.00% | 33.33% | 0 |

## Per Task

| Task | Count | Accuracy | Random Baseline | Invalid |
|---|---:|---:|---:|---:|
| `scenario-corner-case` | 300 | 38.00% | 21.67% | 0 |
| `space-perception` | 200 | 42.50% | 16.25% | 0 |
| `time-extrapolation` | 400 | 27.25% | 19.17% | 0 |
| `time-memory` | 300 | 26.67% | 19.44% | 0 |
| `trajectory-prediction` | 600 | 40.83% | 22.22% | 0 |

## Distance And Trajectory Metrics

### SP-3a Distance FRQ

- Count: `100`
- Parse rate: `100.00%`
- MAE: `4.061 m`
- RMSE: `14.667 m`
- Median AE: `0.300 m`
- Within 1m: `87.00%`

### TRJ-5 Trajectory FRQ

- Count: `100`
- Parse rate: `100.00%`
- Exact length match: `100.00%`
- Mean ADE: `47.263 m`
- Mean FDE: `58.917 m`
- Min FDE: `0.350 m`

### TRJ-6 Trajectory FRQ

- Count: `100`
- Parse rate: `100.00%`
- Exact length match: `100.00%`
- Mean ADE: `8.641 m`
- Mean FDE: `14.378 m`
- Min FDE: `0.007 m`

### TRJ-7 Text FRQ

- Count: `100`
- Non-empty responses: `100` (100.00%)
- Empty responses: `0`
- Error responses: `0`
- Automatic text accuracy: `not scored`


## BLEURT Text FRQ

- Model: `Elron/bleurt-base-512`
- Count: `200`
- Overall mean: `-0.3476`

| Question ID | Count | BLEURT Mean | Min | Max |
|---|---:|---:|---:|---:|
| `SC-4` | 100 | -0.3573 | -0.7784 | -0.0416 |
| `TRJ-9` | 100 | -0.3378 | -0.8995 | -0.0256 |
