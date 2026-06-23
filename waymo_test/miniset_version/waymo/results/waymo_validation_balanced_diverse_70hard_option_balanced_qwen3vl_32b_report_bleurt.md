# Waymo Miniset Report

- Benchmark: `waymo_validation_balanced_diverse_70hard_option_balanced_miniset_from_full_results.json`
- Responses: `result/waymo_validation_balanced_diverse_70hard_option_balanced_qwen3vl_32b_responses.json`
- Score source: `benchmark`
- MCQ count: `1800`
- Correct: `726`
- Accuracy: **40.33%**
- Random baseline: `20.32%`
- Invalid/unparsed predictions: `0` (0.00%)
- Missing responses: `0`

## Per Question Type

| Question ID | Count | Accuracy | Random Baseline | Invalid |
|---|---:|---:|---:|---:|
| `SC-1` | 100 | 46.00% | 25.00% | 0 |
| `SC-2` | 100 | 31.00% | 20.00% | 0 |
| `SC-3` | 100 | 35.00% | 20.00% | 0 |
| `SP-2` | 100 | 33.00% | 12.50% | 0 |
| `SP-3` | 100 | 85.00% | 20.00% | 0 |
| `TE-1` | 100 | 29.00% | 20.00% | 0 |
| `TE-2` | 100 | 31.00% | 20.00% | 0 |
| `TE-4` | 100 | 28.00% | 16.67% | 0 |
| `TE-5` | 100 | 30.00% | 20.00% | 0 |
| `TM-2` | 100 | 22.00% | 16.67% | 0 |
| `TM-3` | 100 | 36.00% | 16.67% | 0 |
| `TM-5` | 100 | 35.00% | 25.00% | 0 |
| `TRJ-1` | 100 | 59.00% | 16.67% | 0 |
| `TRJ-10` | 100 | 28.00% | 25.00% | 0 |
| `TRJ-2` | 100 | 62.00% | 16.67% | 0 |
| `TRJ-3` | 100 | 60.00% | 16.67% | 0 |
| `TRJ-4` | 100 | 31.00% | 25.00% | 0 |
| `TRJ-8` | 100 | 45.00% | 33.33% | 0 |

## Per Task

| Task | Count | Accuracy | Random Baseline | Invalid |
|---|---:|---:|---:|---:|
| `scenario-corner-case` | 300 | 37.33% | 21.67% | 0 |
| `space-perception` | 200 | 59.00% | 16.25% | 0 |
| `time-extrapolation` | 400 | 29.50% | 19.17% | 0 |
| `time-memory` | 300 | 31.00% | 19.44% | 0 |
| `trajectory-prediction` | 600 | 47.50% | 22.22% | 0 |

## Distance And Trajectory Metrics

### SP-3a Distance FRQ

- Count: `100`
- Parse rate: `96.00%`
- MAE: `1.539 m`
- RMSE: `6.664 m`
- Median AE: `0.300 m`
- Within 1m: `93.75%`

### TRJ-5 Trajectory FRQ

- Count: `100`
- Parse rate: `100.00%`
- Exact length match: `100.00%`
- Mean ADE: `16.808 m`
- Mean FDE: `27.797 m`
- Min FDE: `0.350 m`

### TRJ-6 Trajectory FRQ

- Count: `100`
- Parse rate: `100.00%`
- Exact length match: `100.00%`
- Mean ADE: `10.387 m`
- Mean FDE: `17.116 m`
- Min FDE: `0.046 m`

### TRJ-7 Text FRQ

- Count: `100`
- Non-empty responses: `100` (100.00%)
- Empty responses: `0`
- Error responses: `0`
- Automatic text accuracy: `not scored`


## BLEURT Text FRQ

- Model: `Elron/bleurt-base-512`
- Count: `200`
- Overall mean: `-0.3449`

| Question ID | Count | BLEURT Mean | Min | Max |
|---|---:|---:|---:|---:|
| `SC-4` | 100 | -0.3256 | -0.7090 | -0.0491 |
| `TRJ-9` | 100 | -0.3643 | -0.7159 | -0.0532 |
