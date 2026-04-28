#!/usr/bin/env python3
"""Export richer Waymo E2E metadata from TFRecords.

This script is intended to complement export_waymo_front_images.py.
It focuses on metadata that can help downstream miniset selection and scene-level
question generation, especially:

- scene-level stats from `frame.context.stats`
  - weather
  - time_of_day
  - location
  - segment-level object counts by type
- frame-level ego trajectory information
  - past path
  - future path
  - simple turn / acceleration summaries
- convenient aggregated outputs:
  - annotations/frame_rich_metadata.json
  - annotations/scene_metadata_by_scene.json

Note:
- explicit road type is not directly exported here because it is not exposed as
  a simple field in the E2E frame context we are reading.
- this script does not overwrite image exports.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2


DEFAULT_TFRECORD_GLOB = (
    "/local1/lieqiliu/waymo_open_dataset_end_to_end_camera_v_1_0_0/"
    "training_*.tfrecord-*-of-*"
)
DEFAULT_ANNOTATIONS_ROOT = "/local1/rgao727/waymo_dataset/train/annotations"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export richer metadata from Waymo E2E TFRecord shards."
    )
    parser.add_argument(
        "--tfrecord-glob",
        type=str,
        default=DEFAULT_TFRECORD_GLOB,
        help="Glob for Waymo E2E TFRecord shards.",
    )
    parser.add_argument(
        "--annotations-root",
        type=Path,
        default=Path(DEFAULT_ANNOTATIONS_ROOT),
        help="Annotations root directory where metadata JSON files will be written.",
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=0,
        help="If > 0, only process the first N shards.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="If > 0, stop after processing this many TFRecord entries.",
    )
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_scene_id(frame_name: str) -> str:
    if "-" not in frame_name:
        return frame_name
    scene_id, suffix = frame_name.rsplit("-", 1)
    return scene_id if suffix.isdigit() else frame_name


def label_type_name(enum_value: int) -> str:
    try:
        return label_pb2.Label.Type.Name(enum_value)
    except Exception:
        return f"UNKNOWN_{enum_value}"


def object_counts_by_type(entries: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        type_name = label_type_name(int(entry.type)).replace("TYPE_", "").lower()
        counts[type_name] = int(entry.count)
    return counts


def has_field(message: Any, field_name: str) -> bool:
    try:
        return bool(message.HasField(field_name))
    except Exception:
        return False


def pair_xy(xs: list[float], ys: list[float]) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in zip(xs, ys)]


def pair_xyz(xs: list[float], ys: list[float], zs: list[float]) -> list[list[float]]:
    return [[float(x), float(y), float(z)] for x, y, z in zip(xs, ys, zs)]


def summarize_ego(e2e: wod_e2ed_pb2.E2EDFrame) -> dict[str, Any]:
    past_x = [float(v) for v in e2e.past_states.pos_x]
    past_y = [float(v) for v in e2e.past_states.pos_y]
    past_vx = [float(v) for v in e2e.past_states.vel_x]
    future_x = [float(v) for v in e2e.future_states.pos_x]
    future_y = [float(v) for v in e2e.future_states.pos_y]
    future_z = [float(v) for v in e2e.future_states.pos_z]

    turn_magnitude = max((abs(v) for v in future_y), default=0.0)
    accel_delta = abs(past_vx[-1] - past_vx[0]) if len(past_vx) >= 2 else 0.0

    return {
        "past_path_xy": pair_xy(past_x, past_y),
        "future_path_xy": pair_xy(future_x, future_y),
        "future_path_xyz": pair_xyz(future_x, future_y, future_z),
        "current_anchor_xy": [0.0, 0.0],
        "turn_magnitude_y_m": float(turn_magnitude),
        "speed_change_mps": float(accel_delta),
    }


def extract_context_stats(e2e: wod_e2ed_pb2.E2EDFrame) -> dict[str, Any]:
    stats = e2e.frame.context.stats
    stats_present = has_field(e2e.frame.context, "stats")
    return {
        "stats_present": stats_present,
        "time_of_day_present": has_field(stats, "time_of_day"),
        "location_present": has_field(stats, "location"),
        "weather_present": has_field(stats, "weather"),
        "time_of_day": str(stats.time_of_day),
        "location": str(stats.location),
        "weather": str(stats.weather),
        "camera_object_counts_by_type": object_counts_by_type(list(stats.camera_object_counts)),
        "laser_object_counts_by_type": object_counts_by_type(list(stats.laser_object_counts)),
    }


def build_frame_metadata(e2e: wod_e2ed_pb2.E2EDFrame, shard_name: str) -> dict[str, Any]:
    frame_name = e2e.frame.context.name
    scene_id = parse_scene_id(frame_name)
    context_stats = extract_context_stats(e2e)
    ego_summary = summarize_ego(e2e)
    return {
        "scene_id": scene_id,
        "frame_name": frame_name,
        "timestamp_micros": int(e2e.frame.timestamp_micros),
        "intent": int(e2e.intent),
        "shard_name": shard_name,
        "e2ed_context_population_note": (
            "Per waymo_open_dataset/protos/end_to_end_driving_data.proto, "
            "E2EDFrame.frame.context populates name and camera_calibrations; "
            "other context fields may be unused in this dataset."
        ),
        "context_stats": context_stats,
        "ego_trajectory": ego_summary,
    }


def merge_scene_entry(existing: dict[str, Any] | None, frame_meta: dict[str, Any]) -> dict[str, Any]:
    context_stats = frame_meta.get("context_stats", {})
    ego_summary = frame_meta.get("ego_trajectory", {})
    if existing is None:
        return {
            "scene_id": frame_meta["scene_id"],
            "weather": context_stats.get("weather", ""),
            "time_of_day": context_stats.get("time_of_day", ""),
            "location": context_stats.get("location", ""),
            "camera_object_counts_by_type": dict(context_stats.get("camera_object_counts_by_type", {})),
            "laser_object_counts_by_type": dict(context_stats.get("laser_object_counts_by_type", {})),
            "frame_names": [frame_meta["frame_name"]],
            "frame_count": 1,
            "intent_histogram": {str(frame_meta.get("intent")): 1},
            "turn_magnitude_max_y_m": float(ego_summary.get("turn_magnitude_y_m", 0.0)),
            "speed_change_max_mps": float(ego_summary.get("speed_change_mps", 0.0)),
        }

    if not existing.get("weather") and context_stats.get("weather"):
        existing["weather"] = context_stats["weather"]
    if not existing.get("time_of_day") and context_stats.get("time_of_day"):
        existing["time_of_day"] = context_stats["time_of_day"]
    if not existing.get("location") and context_stats.get("location"):
        existing["location"] = context_stats["location"]

    for key in ["camera_object_counts_by_type", "laser_object_counts_by_type"]:
        src = context_stats.get(key, {})
        dst = existing.setdefault(key, {})
        for object_type, count in src.items():
            dst[object_type] = max(int(dst.get(object_type, 0)), int(count))

    existing.setdefault("frame_names", []).append(frame_meta["frame_name"])
    existing["frame_count"] = int(existing.get("frame_count", 0)) + 1
    hist = existing.setdefault("intent_histogram", {})
    intent_key = str(frame_meta.get("intent"))
    hist[intent_key] = int(hist.get(intent_key, 0)) + 1
    existing["turn_magnitude_max_y_m"] = max(
        float(existing.get("turn_magnitude_max_y_m", 0.0)),
        float(ego_summary.get("turn_magnitude_y_m", 0.0)),
    )
    existing["speed_change_max_mps"] = max(
        float(existing.get("speed_change_max_mps", 0.0)),
        float(ego_summary.get("speed_change_mps", 0.0)),
    )
    return existing


def main() -> None:
    args = parse_args()
    annotations_root = args.annotations_root.resolve()

    shards = sorted(Path(p) for p in glob.glob(args.tfrecord_glob))
    if not shards:
        raise FileNotFoundError(f"No TFRecord shards matched: {args.tfrecord_glob}")
    if args.max_shards > 0:
        shards = shards[: args.max_shards]

    record_count = 0
    shard_count = 0
    failed_records: list[dict[str, str]] = []
    frame_rows: list[dict[str, Any]] = []
    scene_rows: dict[str, dict[str, Any]] = {}

    for shard in shards:
        shard_count += 1
        print(f"[{shard_count}/{len(shards)}] Processing {shard.name}")
        dataset = tf.data.TFRecordDataset([str(shard)], compression_type="")

        for raw in dataset:
            record_count += 1
            if args.max_records > 0 and record_count > args.max_records:
                break

            try:
                e2e = wod_e2ed_pb2.E2EDFrame()
                e2e.ParseFromString(raw.numpy())
                frame_meta = build_frame_metadata(e2e, shard.name)
                frame_rows.append(frame_meta)
                scene_id = frame_meta["scene_id"]
                scene_rows[scene_id] = merge_scene_entry(scene_rows.get(scene_id), frame_meta)
            except Exception as exc:
                failed_records.append(
                    {
                        "shard": shard.name,
                        "record_index": str(record_count),
                        "error": str(exc),
                    }
                )

        if args.max_records > 0 and record_count >= args.max_records:
            break

    frame_rows.sort(key=lambda x: (x["scene_id"], x["frame_name"]))
    scene_payload = {
        "count": len(scene_rows),
        "scenes": scene_rows,
    }
    frame_payload = {
        "count": len(frame_rows),
        "frames": frame_rows,
    }
    summary = {
        "tfrecord_glob": args.tfrecord_glob,
        "annotations_root": str(annotations_root),
        "shards_processed": shard_count,
        "records_processed": record_count,
        "frames_exported": len(frame_rows),
        "scenes_exported": len(scene_rows),
        "failed_count": len(failed_records),
        "failed_records": failed_records[:100],
        "metadata_availability_note": (
            "This exporter reads context.stats when present, but Waymo E2EDFrame "
            "documentation states that frame.context fields other than name and "
            "camera_calibrations may be unused. Empty weather/time_of_day/location "
            "therefore usually indicates missing source values, not an exporter bug."
        ),
        "outputs": {
            "frame_rich_metadata_json": "frame_rich_metadata.json",
            "scene_metadata_by_scene_json": "scene_metadata_by_scene.json",
        },
    }

    write_json(annotations_root / "frame_rich_metadata.json", frame_payload)
    write_json(annotations_root / "scene_metadata_by_scene.json", scene_payload)
    write_json(annotations_root / "export_summary_rich_metadata.json", summary)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved frame metadata to: {annotations_root / 'frame_rich_metadata.json'}")
    print(f"Saved scene metadata to: {annotations_root / 'scene_metadata_by_scene.json'}")


if __name__ == "__main__":
    main()
