# nuScenes vs Waymo E2ED Metadata Comparison

## Goal

This note summarizes the practical metadata differences between the current
`nuScenes` pipeline and the current `Waymo E2ED` pipeline in this project.

It answers three questions:

1. what metadata each dataset currently provides,
2. what kinds of questions each dataset can support reliably,
3. whether Waymo can support multi-future-path style questions.

## Short Summary

`nuScenes` is currently much richer for scene-level and object-level reasoning.

`Waymo E2ED` is currently strongest for:

- image-based context,
- ego trajectory supervision,
- ego future prediction,
- intent-style future driving questions,
- preference-trajectory style multi-future reasoning.

But `Waymo E2ED` is much weaker than `nuScenes` for:

- direct weather metadata,
- direct lighting metadata,
- direct road-type metadata,
- direct dense object-level metadata from the exported frame annotations.

## nuScenes: Metadata We Can Reliably Use

With the current `nuScenes` setup, we can use or derive:

- scene-level metadata
  - weather
  - lighting / time of day
  - road type
  - traffic density
  - baseline risk
- object-level metadata
  - object categories
  - object boxes
  - object distance
  - object visibility / occlusion
  - motion-related object attributes such as moving / parked
- ego metadata
  - ego pose
  - ego trajectory
- map / structural metadata
  - intersection-like structure
  - construction-like structure
  - highway / straight-road context

In practice, `nuScenes` supports both:

- scene-level question construction,
- object-level question construction,

with relatively complete metadata support.

## Waymo E2ED: Metadata We Can Reliably Use

With the current `Waymo E2ED` pipeline, we can reliably use:

- images
- camera calibration
- frame name / scene id
- ego past trajectory
- ego future trajectory
- ego intent
- preference trajectories for a subset of frames

We also export:

- `frame_index.json`
- `ego_status/*.json`
- `camera_calibration/*.json`
- `labels/*.json`
- `frame_rich_metadata.json`
- `scene_metadata_by_scene.json`

However, the important limitation is:

- the current E2ED data instance does not reliably populate rich `frame.context`
  scene stats beyond the fields explicitly used by the E2ED spec.

## Important Waymo E2ED Limitation

In the official Waymo E2ED proto comments, `frame.context` is documented as
using:

- `name`
- `camera_calibrations`

and indicating that the other `frame.context` fields are unused for this data
format.

This matters because although the broader Waymo schema defines fields like:

- `time_of_day`
- `location`
- `weather`

the current E2ED records we are reading do not actually populate them with
useful values.

So for the current data:

- `weather` is effectively unavailable,
- `time_of_day` is effectively unavailable,
- `location` is effectively unavailable,
- direct segment-level object count stats are effectively unavailable.

This is not mainly an exporter bug. It is a limitation of the actual E2ED data
instance we are reading.

## nuScenes vs Waymo: Main Practical Difference

### nuScenes gives us

- richer direct scene metadata,
- richer object-level annotations,
- easier support for scene-level balancing,
- easier support for metadata-driven miniset selection.

### Waymo E2ED gives us

- strong ego trajectory supervision,
- strong future-path supervision,
- intent supervision,
- candidate future trajectory preference supervision,
- but weaker direct scene metadata.

So the two datasets are not symmetric.

## What Questions nuScenes Can Support Well

`nuScenes` can support the full intended family of questions well:

- `SC-*`
- `SP-*`
- `SU-*`
- `TE-*`
- `TM-*`
- `TRJ-*`

because it has enough scene-level and object-level metadata to support all of
them more naturally.

## What Questions Waymo Can Support Well

With the current `Waymo E2ED` data, the most reliable family is:

- `TRJ-*`

because we have:

- `past_states`
- `future_states`
- `intent`
- and sometimes `preference_trajectories`

Waymo can also support some of:

- `SP-*`
- `SU-*`
- `TE-*`
- `TM-*`

but only after additional downstream processing such as:

- SAM3 bbox extraction,
- distance estimation,
- heuristic or model-based answer generation.

## What Questions Waymo Cannot Support As Naturally As nuScenes

Waymo is currently weaker for:

- `SC-1` weather
- `SC-2` time of day / lighting
- `SC-3` road type

because these are not directly provided in reliable metadata in the current
E2ED export.

That means these question types can still be generated, but they are more
dependent on:

- heuristics,
- downstream model inference,
- or weak defaults,

rather than direct metadata.

## Can Waymo Support Multi-Future-Path Questions?

Yes, this is actually one of Waymo E2ED's strongest advantages.

Waymo E2ED includes:

- `future_states`
- `intent`
- `preference_trajectories`

This means Waymo is well-suited for:

- single-future ego prediction,
- intent-conditioned future prediction,
- multiple candidate future trajectory questions,
- preference-based future path selection questions.

## What `preference_trajectories` Means

`preference_trajectories` does **not** mean:

- one true trajectory and several fake trajectories.

It means:

- several candidate future ego trajectories,
- with human preference scores for at least some frames,
- so we can reason about which future option is more preferred or more
  plausible.

This makes Waymo especially valuable for future-planning question types.

## Practical Interpretation

### nuScenes is better when we need

- broad metadata completeness,
- direct scene balancing by weather / lighting / road type,
- direct object-level reasoning without heavy extra preprocessing.

### Waymo E2ED is better when we need

- ego future prediction,
- future-path reasoning,
- multi-future option selection,
- intent-aware driving trajectory questions.

## Recommended Positioning In This Project

### Use nuScenes for

- metadata-rich balanced benchmark construction,
- scene-level evaluation,
- object-level spatial and situational reasoning,
- balanced miniset design with direct scene metadata.

### Use Waymo E2ED for

- trajectory-centric benchmarks,
- future planning benchmarks,
- multi-future candidate reasoning,
- preference-based path selection,
- intent-aware driving reasoning.

## Final Recommendation

If the question family depends heavily on:

- weather,
- lighting,
- road type,
- dense scene-level metadata,

then `nuScenes` is the stronger source.

If the question family depends heavily on:

- future ego motion,
- future path choices,
- preference among candidate trajectories,

then `Waymo E2ED` is the stronger source.

So Waymo should not be treated as a direct metadata-equivalent replacement for
nuScenes. Instead, it should be treated as a complementary dataset with a
different strength profile:

- weaker scene metadata,
- stronger ego-future trajectory supervision.
