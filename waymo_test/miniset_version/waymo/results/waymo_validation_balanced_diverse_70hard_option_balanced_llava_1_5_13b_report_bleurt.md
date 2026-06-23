# Waymo Miniset Report

- Benchmark: `waymo_validation_balanced_diverse_70hard_option_balanced_miniset_from_full_results.json`
- Responses: `result/waymo_validation_balanced_diverse_70hard_option_balanced_llava_1_5_13b_responses.json`
- Score source: `benchmark`
- MCQ count: `1800`
- Correct: `587`
- Accuracy: **32.61%**
- Random baseline: `20.32%`
- Invalid/unparsed predictions: `0` (0.00%)
- Missing responses: `0`

## Per Question Type

| Question ID | Count | Accuracy | Random Baseline | Invalid |
|---|---:|---:|---:|---:|
| `SC-1` | 100 | 48.00% | 25.00% | 0 |
| `SC-2` | 100 | 36.00% | 20.00% | 0 |
| `SC-3` | 100 | 30.00% | 20.00% | 0 |
| `SP-2` | 100 | 33.00% | 12.50% | 0 |
| `SP-3` | 100 | 38.00% | 20.00% | 0 |
| `TE-1` | 100 | 20.00% | 20.00% | 0 |
| `TE-2` | 100 | 31.00% | 20.00% | 0 |
| `TE-4` | 100 | 18.00% | 16.67% | 0 |
| `TE-5` | 100 | 30.00% | 20.00% | 0 |
| `TM-2` | 100 | 20.00% | 16.67% | 0 |
| `TM-3` | 100 | 24.00% | 16.67% | 0 |
| `TM-5` | 100 | 31.00% | 25.00% | 0 |
| `TRJ-1` | 100 | 60.00% | 16.67% | 0 |
| `TRJ-10` | 100 | 17.00% | 25.00% | 0 |
| `TRJ-2` | 100 | 55.00% | 16.67% | 0 |
| `TRJ-3` | 100 | 32.00% | 16.67% | 0 |
| `TRJ-4` | 100 | 33.00% | 25.00% | 0 |
| `TRJ-8` | 100 | 31.00% | 33.33% | 0 |

## Per Task

| Task | Count | Accuracy | Random Baseline | Invalid |
|---|---:|---:|---:|---:|
| `scenario-corner-case` | 300 | 38.00% | 21.67% | 0 |
| `space-perception` | 200 | 35.50% | 16.25% | 0 |
| `time-extrapolation` | 400 | 24.75% | 19.17% | 0 |
| `time-memory` | 300 | 25.00% | 19.44% | 0 |
| `trajectory-prediction` | 600 | 38.00% | 22.22% | 0 |

## Distance And Trajectory Metrics

### SP-3a Distance FRQ

- Count: `100`
- Parse rate: `100.00%`
- MAE: `45.441 m`
- RMSE: `56.222 m`
- Median AE: `57.400 m`
- Within 1m: `26.00%`

### TRJ-5 Trajectory FRQ

- Count: `100`
- Parse rate: `0.00%`
- Exact length match: `n/a`
- Mean ADE: `n/a`
- Mean FDE: `n/a`
- Min FDE: `n/a`

### TRJ-6 Trajectory FRQ

- Count: `100`
- Parse rate: `0.00%`
- Exact length match: `n/a`
- Mean ADE: `n/a`
- Mean FDE: `n/a`
- Min FDE: `n/a`

### TRJ-7 Text FRQ

- Count: `100`
- Non-empty responses: `100` (100.00%)
- Empty responses: `0`
- Error responses: `0`
- Automatic text accuracy: `not scored`


## BLEURT Text FRQ

- Model: `Elron/bleurt-base-512`
- Count: `200`
- Overall mean: `-0.2929`

| Question ID | Count | BLEURT Mean | Min | Max |
|---|---:|---:|---:|---:|
| `SC-4` | 100 | -0.3004 | -0.7268 | -0.0212 |
| `TRJ-9` | 100 | -0.2853 | -0.5796 | -0.0734 |
