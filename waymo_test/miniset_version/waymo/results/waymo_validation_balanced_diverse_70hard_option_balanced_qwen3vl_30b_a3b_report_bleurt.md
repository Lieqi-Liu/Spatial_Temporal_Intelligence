# Waymo Miniset Report

- Benchmark: `waymo_validation_balanced_diverse_70hard_option_balanced_miniset_from_full_results.json`
- Responses: `result/waymo_validation_balanced_diverse_70hard_option_balanced_qwen3vl_30b_a3b_responses.json`
- Score source: `benchmark`
- MCQ count: `1800`
- Correct: `612`
- Accuracy: **34.00%**
- Random baseline: `20.32%`
- Invalid/unparsed predictions: `0` (0.00%)
- Missing responses: `0`

## Per Question Type

| Question ID | Count | Accuracy | Random Baseline | Invalid |
|---|---:|---:|---:|---:|
| `SC-1` | 100 | 44.00% | 25.00% | 0 |
| `SC-2` | 100 | 28.00% | 20.00% | 0 |
| `SC-3` | 100 | 30.00% | 20.00% | 0 |
| `SP-2` | 100 | 22.00% | 12.50% | 0 |
| `SP-3` | 100 | 31.00% | 20.00% | 0 |
| `TE-1` | 100 | 30.00% | 20.00% | 0 |
| `TE-2` | 100 | 26.00% | 20.00% | 0 |
| `TE-4` | 100 | 36.00% | 16.67% | 0 |
| `TE-5` | 100 | 26.00% | 20.00% | 0 |
| `TM-2` | 100 | 21.00% | 16.67% | 0 |
| `TM-3` | 100 | 32.00% | 16.67% | 0 |
| `TM-5` | 100 | 36.00% | 25.00% | 0 |
| `TRJ-1` | 100 | 46.00% | 16.67% | 0 |
| `TRJ-10` | 100 | 35.00% | 25.00% | 0 |
| `TRJ-2` | 100 | 50.00% | 16.67% | 0 |
| `TRJ-3` | 100 | 54.00% | 16.67% | 0 |
| `TRJ-4` | 100 | 36.00% | 25.00% | 0 |
| `TRJ-8` | 100 | 29.00% | 33.33% | 0 |

## Per Task

| Task | Count | Accuracy | Random Baseline | Invalid |
|---|---:|---:|---:|---:|
| `scenario-corner-case` | 300 | 34.00% | 21.67% | 0 |
| `space-perception` | 200 | 26.50% | 16.25% | 0 |
| `time-extrapolation` | 400 | 29.50% | 19.17% | 0 |
| `time-memory` | 300 | 29.67% | 19.44% | 0 |
| `trajectory-prediction` | 600 | 41.67% | 22.22% | 0 |

## Distance And Trajectory Metrics

### SP-3a Distance FRQ

- Count: `100`
- Parse rate: `98.00%`
- MAE: `21.967 m`
- RMSE: `31.432 m`
- Median AE: `12.800 m`
- Within 1m: `12.24%`

### TRJ-5 Trajectory FRQ

- Count: `100`
- Parse rate: `100.00%`
- Exact length match: `100.00%`
- Mean ADE: `35.561 m`
- Mean FDE: `46.134 m`
- Min FDE: `0.350 m`

### TRJ-6 Trajectory FRQ

- Count: `100`
- Parse rate: `97.00%`
- Exact length match: `100.00%`
- Mean ADE: `18.485 m`
- Mean FDE: `27.452 m`
- Min FDE: `0.913 m`

### TRJ-7 Text FRQ

- Count: `100`
- Non-empty responses: `100` (100.00%)
- Empty responses: `0`
- Error responses: `0`
- Automatic text accuracy: `not scored`


## BLEURT Text FRQ

- Model: `Elron/bleurt-base-512`
- Count: `200`
- Overall mean: `-0.3173`

| Question ID | Count | BLEURT Mean | Min | Max |
|---|---:|---:|---:|---:|
| `SC-4` | 100 | -0.3716 | -0.9244 | 0.0009 |
| `TRJ-9` | 100 | -0.2631 | -0.7329 | 0.0741 |
