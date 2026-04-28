# Miniset Metadata Scoring And Selection

## Purpose

This document defines a metadata-driven standard for building `nuscenes` and `waymo` minisets.

The target is not just to downsample. The target is to build a subset that is:

1. complex enough to be meaningful,
2. balanced enough across question ground truth,
3. balanced enough across scene-level attributes such as weather and lighting,
4. reproducible from concrete metadata rather than only from heuristic text labels.

This document is the intended selection policy.

## Important Note

The current [`build_miniset.py`](/home/rgao727/Spatial_Temporal_Intelligence/build_miniset.py) already does:

- group-based selection,
- complexity-aware selection,
- per-task GT balancing,
- scene-level balancing using `SC-1` to `SC-5` labels already present in the input JSON.

But it does **not yet directly read raw external dataset metadata tables**.

This document describes the stricter metadata-based policy we want to follow going forward.

## Selection Unit

Selection should happen at the **group** level, not at the single-question level and not at the object-unit level.

A group is:

- `scene_id`
- `group_id`

When a group is selected:

- all scene-level questions for that group are kept,
- all object-level questions for that group are kept,
- all trajectory questions for that group are kept if available.

Reason:

- the benchmark is built around 5-frame context groups,
- eval is context-based,
- selecting single object-units fragments the visual context and makes scene coverage harder to reason about.

## Metadata Sources

## nuScenes

Primary available metadata under `/local1/rgao727/nuscenes/v1.0-trainval/`:

- `scene.json`
- `log.json`
- `sample.json`
- `sample_data.json`
- `sample_annotation.json`
- `instance.json`
- `attribute.json`
- `ego_pose.json`
- `visibility.json`
- `map.json`

Derived question file:

- `/local1/rgao727/nuscenes/questions_with_answers_trainval.json`

Recommended metadata usage:

- `scene.json`
  - scene boundaries
  - description text
  - scene continuity
- `log.json`
  - location / city
  - route context
- `sample.json`
  - frame timestamps
  - temporal spacing
- `sample_annotation.json`
  - object count
  - class mix
  - relative motion availability
  - box geometry
- `attribute.json`
  - parked / moving / stopped cues
- `ego_pose.json`
  - ego motion
  - turning / acceleration / trajectory change
- `visibility.json`
  - occlusion / visibility quality
- `map.json`
  - map region / road context

## Waymo

Primary available metadata under `/local1/rgao727/waymo_dataset/train/annotations/`:

- `frame_index.json`
- `ego_status/*.json`
- `camera_calibration/*.json`
- `labels/*.json`
- `bbox_distance_estimates.json` after distance estimation

Recommended metadata usage:

- `frame_index.json`
  - frame to scene mapping
  - frame order
  - timestamps
- `ego_status/*.json`
  - ego kinematics
  - future / past states
  - turn / acceleration / stopping dynamics
- `labels/*.json`
  - object counts
  - class diversity
  - spatial distribution
- `bbox_distance_estimates.json`
  - object distances
  - lateral offsets
  - near-field interaction
- `camera_calibration/*.json`
  - geometry support only
  - not directly a complexity signal

## Attribute Standardization

For both datasets, normalize scene attributes into these buckets:

- `weather`
  - `clear`
  - `rain`
  - `wet`
  - `fog`
  - `snow`
  - `overcast`
  - `unknown`
- `time_of_day`
  - `day`
  - `dawn_dusk`
  - `night`
  - `unknown`
- `road_type`
  - `urban_straight`
  - `urban_intersection`
  - `residential`
  - `highway`
  - `parking_lot`
  - `construction`
  - `unknown`
- `traffic_density`
  - `low`
  - `moderate`
  - `dense`
- `risk`
  - `low`
  - `moderate`
  - `high`

If raw metadata cannot determine a field reliably:

1. use derived GT from `SC-1` to `SC-5`,
2. otherwise mark as `unknown`.

## Complexity Score

Each unit gets a `complexity_score`.

Recommended formula:

```text
complexity_score =
  interaction_score
  + density_score
  + motion_score
  + occlusion_score
  + scene_layout_score
  + environment_score
  + diversity_score
```

### 1. Interaction Score

Measures whether objects are close enough and relevant enough to matter.

Recommended signals:

- count of objects within 0 to 5 m
- count of objects within 5 to 15 m
- objects in ego lane or conflict lane
- crossing pedestrians / bicycles
- closing-speed or TTC proxy if available

Recommended scoring:

```text
near_0_5m_count * 2.0
+ near_5_15m_count * 1.0
+ crossing_vru_count * 2.0
+ frontal_conflict_count * 1.5
+ lane_change_conflict_count * 1.2
```

### 2. Density Score

Measures how crowded the scene is.

Recommended signals:

- vehicle count
- pedestrian count
- bicycle / motorcycle count
- static obstacle count

Recommended scoring:

```text
min(vehicle_count, 15) * 0.3
+ min(vru_count, 10) * 0.5
+ min(static_obstacle_count, 10) * 0.2
```

### 3. Motion Score

Measures whether the scene has active dynamics.

Recommended signals:

- number of moving objects
- number of turning objects
- ego yaw-rate proxy
- ego acceleration / braking proxy

Recommended scoring:

```text
moving_object_count * 0.4
+ turning_object_count * 0.6
+ ego_turning_flag * 1.0
+ ego_brake_or_accel_flag * 0.8
```

### 4. Occlusion Score

Measures ambiguity and perception difficulty.

Recommended signals:

- low visibility annotations in nuScenes
- partial boxes / truncated observations
- dense cluster overlap

Recommended scoring:

```text
occluded_object_count * 0.5
+ low_visibility_object_count * 0.7
```

### 5. Scene Layout Score

Measures structural complexity of the road context.

Recommended signals:

- intersection
- merge / split
- roundabout
- construction
- parking lot

Recommended scoring:

```text
intersection_flag * 2.0
+ merge_or_split_flag * 1.2
+ construction_flag * 1.8
+ parking_lot_flag * 0.6
```

### 6. Environment Score

Measures weather and lighting difficulty.

Recommended scoring:

```text
night * 1.5
+ dawn_dusk * 1.0
+ rain * 1.2
+ wet * 0.8
+ fog * 1.5
+ snow * 1.8
```

### 7. Diversity Score

Rewards scenes containing multiple relevant object types.

Recommended signals:

- number of distinct semantic categories among:
  - car
  - truck / bus
  - pedestrian
  - bicycle / motorcycle
  - traffic control / barrier / cone

Recommended scoring:

```text
distinct_category_count * 0.4
```

## Hard Filters Before Selection

Before balancing, filter out low-value units.

Recommended hard filters:

- fewer than 5 valid questions in the unit: drop
- missing `scene_id` or `group_id`: drop
- unknown image linkage: drop
- no usable scene attributes at all: keep only if object interaction score is high
- bottom 50 percent by complexity: drop by default

Default candidate pool:

- keep top 50 percent by `complexity_score`

For very large sets:

- optionally keep top 30 percent

## Tail-Group Filtering

Do not prefer groups near the end of a scene if they lose trajectory supervision.

Recommended rule:

- exclude groups that do not contain `TRJ-*` questions by default

Reason:

- late-scene tail groups often lose enough future context that trajectory tasks disappear,
- this weakens temporal supervision and reduces task diversity.

If a dataset version or experiment must keep them, allow this only as an explicit override.

## Balancing Objectives

After filtering, selection is done by greedy balancing.

We optimize three distributions at once.

### 1. Per-task Ground Truth Balance

For each task, avoid one GT dominating the subset.

Examples:

- `SC-1`: weather GTs should not collapse to mostly `clear`
- `SC-2`: lighting GTs should not collapse to mostly `day`
- `SP-*`, `SU-*`, `TM-*`, `TE-*`, `TRJ-*`: choice labels should not collapse to one dominant answer

Recommended gain:

```text
task_balance_gain(unit) =
sum over rows in unit of 1 / (1 + current_count(task_id, gt))
```

### 2. Scene Attribute Balance

Target balanced counts over:

- weather
- time_of_day
- road_type
- traffic_density
- risk

Recommended gain:

```text
scene_balance_gain(unit) =
sum over unit attrs of 1 / (1 + current_count(attr_name, attr_value))
```

### 3. Complexity Preservation

Do not let balancing drag the subset toward easy scenes.

Recommended gain:

```text
complexity_gain(unit) =
normalized_complexity_score(unit)
```

## Scene Coverage Constraint

Selection should not collapse into a few scenes.

Recommended process:

1. target about 5 percent of all scenes,
2. then select a fixed number of groups from those scenes,
3. bias toward scenes not yet represented,
4. still maintain complexity and GT balance.

For the current real `nuscenes` trainval scale:

- total scenes: about `850`
- 5 percent scene coverage: about `43` scenes

Recommended `nuscenes` target:

- about `43` scenes
- about `335` groups

This gives:

- broad scene diversity,
- controlled total size,
- compatibility with 5 percent style reporting.

## Final Greedy Selection Score

Recommended greedy score:

```text
selection_score(unit) =
  0.35 * complexity_gain
  + 1.00 * task_balance_gain
  + 0.80 * scene_balance_gain
  + scene_diversity_gain
```

Where:

```text
scene_diversity_gain =
  bonus for a previously unseen scene when scene coverage is still below target
  + bonus for scenes that have been selected fewer times so far
```

These are good defaults.

If the subset still looks too easy:

- increase `complexity_gain` weight

If GT distribution is still skewed:

- increase `task_balance_gain` weight

If weather / lighting remain skewed:

- increase `scene_balance_gain` weight

## Recommended Priority Order

When goals conflict, use this priority:

1. keep valid, coherent units
2. preserve high complexity
3. balance weather and time_of_day
4. balance per-task GT
5. balance road_type / density / risk
6. keep group counts across scenes from collapsing into a few heavy scenes

Reason:

- complexity is the point of the benchmark,
- weather and lighting are global biases that strongly affect robustness,
- per-task balancing matters, but should not force the set into low-value easy scenes.

## Dataset-Specific Notes

## nuScenes

Recommended direct metadata features:

- object count from `sample_annotation.json`
- parked / moving cues from `attribute.json`
- visibility / occlusion from `visibility.json`
- ego motion from `ego_pose.json`
- road context from `scene.json`, `log.json`, `map.json`

Recommended fallback:

- if weather / lighting are hard to parse from raw metadata, use `SC-1` and `SC-2` GT values as the canonical labels

## Waymo

Recommended direct metadata features:

- object count and class mix from `labels/*.json`
- distance / lateral proximity from `bbox_distance_estimates.json`
- ego motion from `ego_status/*.json`
- temporal continuity from `frame_index.json`

Recommended fallback:

- if weather is unavailable in raw annotations, keep `SC-1` as the scene-level canonical label once GT is generated

## Required Summary Output

For each built miniset, save a summary JSON with:

- full-set unit count
- miniset group count
- full-set scene count
- miniset scene count
- full-set task count
- miniset task count
- retention fraction
- full-set complexity summary
- miniset complexity summary
- per-task GT distribution for full set
- per-task GT distribution for miniset
- scene-attribute distribution for full set
- scene-attribute distribution for miniset
- metadata source note

This is important because selection quality must be auditable.

## Minimum Deliverables

Each miniset build should produce:

- `questions_with_answers_miniset.json`
- `questions_with_answers_miniset_summary.json`

Optional but recommended:

- a report markdown summarizing:
  - subset size,
  - complexity shift,
  - weather shift,
  - lighting shift,
  - largest GT imbalance that remains.

## Practical Next Step

To fully match this document, `build_miniset.py` should be upgraded in a later pass to:

1. load raw nuScenes metadata tables,
2. load Waymo exported annotation metadata,
3. compute complexity from raw metadata first,
4. use `SC-1` to `SC-5` GT only as fallback labels,
5. select groups rather than object-units,
6. prefer about 5 percent scene coverage,
7. filter out non-trajectory tail groups by default.
