# Spatial_Temporal_Intelligence

This repository builds and evaluates spatial, temporal, scene-level, and ego-trajectory QA tasks on top of the nuScenes dataset.

The current pipeline is centered on [`nuscenes_test`](/home/rgao727/Autonomous%20Driving/Spatial_Temporal_Intelligence/nuscenes_test), and has four stages:

1. Stitch each nuScenes sample into a 6-camera 2x3 grid image.
2. Split each CAM_FRONT scene into 5-frame groups and select one or more target objects per group.
3. Generate task JSON files with automatically derived ground-truth answers.
4. Run VLM evaluation on the generated task set.

## Dataset

The scripts expect a full nuScenes-style data root, not only NuScenes-QA question files.

Minimum required structure:

```text
<NUSC_ROOT>/
  v1.0-mini/
    sample_data.json
    sample.json
    sample_annotation.json
    instance.json
    category.json
    ego_pose.json
    scene.json
    log.json
  samples/
    CAM_FRONT/
    CAM_FRONT_LEFT/
    CAM_FRONT_RIGHT/
    CAM_BACK/
    CAM_BACK_LEFT/
    CAM_BACK_RIGHT/
```

For the setup used in this workspace, the dataset root is typically:

```bash
/home/rgao727/Autonomous Driving/NuScenes-QA/data/nuscenes-v1.0-mini
```

## Benchmark Overview

All tasks in this benchmark use the same visual input format:

- Temporal context: 5 consecutive frames
- Per-frame visual view: a stitched 2x3 multi-camera grid
- Cameras per frame: `CAM_FRONT_LEFT`, `CAM_FRONT`, `CAM_FRONT_RIGHT`, `CAM_BACK_LEFT`, `CAM_BACK`, `CAM_BACK_RIGHT`
- Effective input to the model: 5 stitched 360-degree multi-camera frames

The benchmark currently covers the following task families:

| Family | IDs | Level | Input View | Question Format | What It Covers |
| --- | --- | --- | --- | --- | --- |
| Spatial Perception | `SP-1` to `SP-7` | Object-level | Past 5 stitched 360 frames, query object on frame 5 | MCQ + FRQ | Object trajectory direction, relative position, distance, lane position, occlusion, ground-plane status, spatial summary |
| Spatial Understanding | `SU-1` to `SU-7` | Object/ego-level | Past 5 stitched 360 frames, query object on frame 5 | MCQ + FRQ | Constraints on ego, drivable region, risk, lane-change clearance, feasible maneuver, traffic density, scene understanding summary |
| Time Extrapolation | `TE-1` to `TE-6` | Object/ego-level | Past 5 stitched 360 frames, query object on frame 5 | MCQ + FRQ | Future object motion, collision timing, future lane occupancy, ego response, likely next event, short-horizon scene evolution |
| Time Memory | `TM-1` to `TM-6` | Object/scene-level | Past 5 stitched 360 frames, query object on frame 5 | MCQ + FRQ | Disappeared objects, previous object location, recent motion trend, previous occlusion, earlier lane position, temporal summary |
| Scene Context | `SP-C-1` to `SP-C-6` | Scene-level | Past 5 stitched 360 frames | MCQ + FRQ | Weather, lighting/time of day, road typology, scene density, scene-level risk, holistic environment description |
| Ego Trajectory Prediction | `TRJ-1` to `TRJ-6` | Ego-level | Past 5 stitched 360 frames | MCQ + FRQ | Future maneuver, future endpoint region, future speed trend, 5-step future path, conditional 4-step continuation, natural-language future path description |

Additional notes:

- Object-level tasks may use a bbox-highlighted query image on the 5th frame.
- Scene-level tasks do not use bbox highlighting.
- Trajectory tasks use the past 5 frames as context, but their labels come from the next 5 future frames after the current anchor frame.
- Distance questions are explicitly included in `SP-3` and `SP-3a`.
- Trajectory questions are explicitly included in `TRJ-1` to `TRJ-6`.

## End-To-End Flow

Run everything from:

```bash
cd "/home/rgao727/Autonomous Driving/Spatial_Temporal_Intelligence/nuscenes_test"
```

### 1. Stitch 6-camera frames

```bash
python stitch_scenes.py \
  --root "/home/rgao727/Autonomous Driving/NuScenes-QA/data/nuscenes-v1.0-mini" \
  --output-dir "./formatted_scenes" \
  --overwrite
```

What it does:
- Uses the six nuScenes camera views.
- Builds a 2x3 stitched image for each timestamp.
- Writes images into `formatted_scenes/scene_XXX/`.

### 2. Build 5-frame groups and select target objects

```bash
python bbox_rendering.py \
  --root "/home/rgao727/Autonomous Driving/NuScenes-QA/data/nuscenes-v1.0-mini" \
  --output-dir "./formatted_scenes" \
  --num-selected-objects 1
```

What it does:
- Splits each scene into consecutive 5-frame groups.
- Looks at the 5th frame in each group.
- Collects `vehicle.*` annotations from that frame.
- Randomly samples `--num-selected-objects` vehicles per group.
- Renders one bbox-highlighted query image per selected object.
- Writes `group_*_vehicle_annotations.json`.

Notes:
- If `--num-selected-objects 1`, each 5-frame group produces one object-level sample.
- If `--num-selected-objects > 1`, one base group can produce multiple object-level samples.
- Scene-level tasks still reuse the same 5-frame visual context.
- Ego trajectory tasks are deduplicated per base group and generated once per 5-frame group.

### 3. Generate QA/task JSON with GT answers

```bash
python generate_answers.py \
  --root "/home/rgao727/Autonomous Driving/NuScenes-QA/data/nuscenes-v1.0-mini" \
  --formatted-scenes-dir "./formatted_scenes" \
  --questions-json "./questions.json" \
  --questions-output "./questions_with_answers_all.json" \
  --disable-frq-generation \
  --scene-level-tasks-json "/home/rgao727/Autonomous Driving/scene-level-context-tasks.json" \
  --enable-traj-prediction-tasks
```

What it does:
- Generates object-level spatial, understanding, extrapolation, and memory tasks.
- Injects scene-level tasks from `scene-level-context-tasks.json`.
- Optionally injects ego-trajectory prediction tasks.
- Writes:
  - `formatted_scenes/generated_answers_all.json`
  - `questions_with_answers_all.json`

Trajectory prediction details:
- `TRJ-1`, `TRJ-2`, `TRJ-3` are MCQ tasks.
- `TRJ-4`, `TRJ-5`, `TRJ-6` are FRQ tasks.
- All `TRJ-*` labels use the next 5 future frames after the current anchor frame.
- The trajectory GT is stored in ego-centric local coordinates.
- `TRJ-4` predicts the full next 5 future points.
- `TRJ-5` is a conditional continuation task: the first future point is given, and the model predicts the following 4 points.
- `TRJ-6` is a natural-language future-path description task.
- The `TRJ-6` ground truth is generated automatically from the same future 5-frame trajectory by first deriving:
  - maneuver (`TRJ-1`)
  - endpoint region (`TRJ-2`)
  - speed trend (`TRJ-3`)
  and then rendering those structured labels into a short textual summary.

### 4. Run evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python run_eval.py \
  --tasks "./questions_with_answers_all.json" \
  --formatted-scenes-dir "./formatted_scenes" \
  --tensor-parallel-size 1
```

Quick smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 python run_eval.py \
  --tasks "./questions_with_answers_all.json" \
  --formatted-scenes-dir "./formatted_scenes" \
  --tensor-parallel-size 1 \
  --max-tasks 50
```

## Script Parameters

### `stitch_scenes.py`

Key arguments:
- `--root`: nuScenes dataset root.
- `--sample-data-json`: optional explicit path to `sample_data.json`.
- `--output-dir`: where stitched frames are written.
- `--overwrite`: recreate the output directory.

### `bbox_rendering.py`

Key arguments:
- `--root`: nuScenes dataset root.
- `--output-dir`: `formatted_scenes` directory.
- `--group-size`: number of frames per group. Current pipeline assumes `5`.
- `--seed`: random seed for object sampling.
- `--num-selected-objects`: number of target objects sampled per 5-frame group.

### `generate_answers.py`

Key arguments:
- `--root`: nuScenes dataset root.
- `--version`: usually `v1.0-mini`.
- `--formatted-scenes-dir`: directory containing stitched scenes and group JSON files.
- `--questions-json`: base task template file.
- `--output-json`: path for grouped generated answer dump.
- `--questions-output`: path for merged generated task file used by evaluation.
- `--track-window`: temporal window used by several object-level heuristics.
- `--lane-width-m`: fallback lane width used in some spatial rules.
- `--disable-frq-generation`: skip LLM generation for FRQ targets and only build rule-based GT.
- `--frq-model`: VLM/LLM used for FRQ answer generation.
- `--frq-tensor-parallel-size`: tensor parallel size for FRQ generation.
- `--frq-max-tokens`: max FRQ answer length.
- `--frq-temperature`: FRQ sampling temperature.
- `--scene-level-tasks-json`: external scene-level annotations file.
- `--enable-traj-prediction-tasks`: inject `TRJ-*` ego trajectory tasks.

### `run_eval.py`

Key arguments:
- `--tasks`: generated questions file.
- `--formatted-scenes-dir`: path to `formatted_scenes`.
- `--output`: evaluation results JSON.
- `--model`: VLM name or path. Default is `Qwen/Qwen3-VL-30B-A3B-Instruct`.
- `--tensor-parallel-size`: number of GPUs used by vLLM tensor parallelism.
- `--max-tasks`: optional limit for quick tests.

GPU selection is controlled outside the script, for example:

```bash
CUDA_VISIBLE_DEVICES=0 python run_eval.py --tensor-parallel-size 1
```

## Cache And Model Path

`run_eval.py` currently sets:

```python
os.environ["HF_HOME"] = "/local1/lieqiliu/huggingface"
```

If you want to override it at runtime, you can launch with:

```bash
HF_HOME=/home/rgao727/.cache/huggingface CUDA_VISIBLE_DEVICES=0 python run_eval.py ...
```

## Generated Files

Common generated artifacts:

- `nuscenes_test/formatted_scenes/`
- `nuscenes_test/questions_with_answers_all.json`
- `nuscenes_test/vlm_responses_*.json`

`formatted_scenes/` is intentionally ignored by git because it is large and fully reproducible.

## Sanity Checks

Check scene-level tasks were injected:

```bash
python - <<'PY'
import json
with open("./questions_with_answers_all.json") as f:
    data=json.load(f)
for k, v in sorted(data["generated_answers"].items()):
    if k.startswith("SP-C-"):
        print(k, v["count"])
PY
```

Check trajectory tasks were injected:

```bash
python - <<'PY'
import json
with open("./questions_with_answers_all.json") as f:
    data=json.load(f)
for k in ["TRJ-1", "TRJ-2", "TRJ-3", "TRJ-4", "TRJ-5", "TRJ-6"]:
    print(k, data["generated_answers"].get(k, {}).get("count"))
PY
```
