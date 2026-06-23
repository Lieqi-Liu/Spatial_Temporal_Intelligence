#!/usr/bin/env python3
"""Export CAM_FRONT JPEG frames and useful annotations from Waymo E2E TFRecords.

Output layout matches the rest of this repository's Waymo pipeline:

  <output-root>/<scene_id>/CAM_FRONT/<frame_name>_FRONT.jpg

where:
  - frame_name is taken from e2e.frame.context.name
  - scene_id is derived from frame_name by removing the final "-NNN" suffix

This keeps the downstream expectations consistent with:
  - extract_waymo_annotations.py
  - estimate_bbox_distances.py
  - run_eval.py

It also writes the key frame-level artifacts used later by the Waymo test
pipeline:
  - images/<scene_id>/CAM_FRONT/<frame_name>_FRONT.jpg
  - annotations/ego_status/<frame_name>.json
  - annotations/camera_calibration/<frame_name>.json
  - annotations/labels/<frame_name>.json
  - annotations/frame_index.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import tensorflow as tf
from google.protobuf.json_format import MessageToDict
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2


DEFAULT_TFRECORD_GLOB = (
    "/local1/lieqiliu/waymo_open_dataset_end_to_end_camera_v_1_0_0/"
    "test_*.tfrecord-*-of-*"
)
DEFAULT_OUTPUT_BASE = "/local1/rgao727/waymo_dataset"
DEFAULT_CAMERA = "FRONT"
DEFAULT_SPLIT = "test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export JPEG frames from Waymo E2E TFRecord shards."
    )
    parser.add_argument(
        "--tfrecord-glob",
        type=str,
        default=DEFAULT_TFRECORD_GLOB,
        help="Glob for Waymo E2E TFRecord shards.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default=DEFAULT_SPLIT,
        help="Dataset split name used to place exports under <output-base>/<split>/.",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=Path(DEFAULT_OUTPUT_BASE),
        help="Base directory under which split-specific exports are saved.",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=None,
        help="Optional override for the exported image root.",
    )
    parser.add_argument(
        "--annotations-root",
        type=Path,
        default=None,
        help="Optional override for the exported annotation JSON root.",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default=DEFAULT_CAMERA,
        help="Camera name to export, default FRONT.",
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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing image files.",
    )
    return parser.parse_args()


def resolve_output_roots(args: argparse.Namespace) -> tuple[Path, Path]:
    split_root = args.output_base.resolve() / args.split
    images_root = args.images_root.resolve() if args.images_root else split_root / "images"
    annotations_root = (
        args.annotations_root.resolve()
        if args.annotations_root
        else split_root / "annotations"
    )
    return images_root, annotations_root


def camera_enum_to_name(camera_enum_value: int) -> str:
    try:
        return open_dataset.CameraName.Name(camera_enum_value)
    except Exception:
        pass

    try:
        return open_dataset.CameraName.Name.Name(camera_enum_value)
    except Exception:
        pass

    descriptor = open_dataset.CameraName.DESCRIPTOR.values_by_number.get(camera_enum_value)
    if descriptor is not None:
        return descriptor.name
    return f"UNKNOWN_{camera_enum_value}"


def parse_scene_id(frame_name: str) -> str:
    if "-" not in frame_name:
        return frame_name
    scene_id, suffix = frame_name.rsplit("-", 1)
    return scene_id if suffix.isdigit() else frame_name


def write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(payload)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def export_camera_calibration_payload(e2e: wod_e2ed_pb2.E2EDFrame) -> list[dict]:
    calibrations: list[dict] = []
    for c in e2e.frame.context.camera_calibrations:
        calibrations.append(
            {
                "name": camera_enum_to_name(c.name),
                "intrinsic": [float(v) for v in c.intrinsic],
                "extrinsic": [float(v) for v in c.extrinsic.transform],
                "width": int(c.width),
                "height": int(c.height),
                "rolling_shutter_direction": int(c.rolling_shutter_direction),
            }
        )
    return calibrations


def export_ego_status_payload(e2e: wod_e2ed_pb2.E2EDFrame) -> dict:
    return {
        "frame_name": e2e.frame.context.name,
        "intent": int(e2e.intent),
        "past_states": {
            "pos_x": [float(v) for v in e2e.past_states.pos_x],
            "pos_y": [float(v) for v in e2e.past_states.pos_y],
            "vel_x": [float(v) for v in e2e.past_states.vel_x],
            "vel_y": [float(v) for v in e2e.past_states.vel_y],
            "accel_x": [float(v) for v in e2e.past_states.accel_x],
            "accel_y": [float(v) for v in e2e.past_states.accel_y],
        },
        "future_states": {
            "pos_x": [float(v) for v in e2e.future_states.pos_x],
            "pos_y": [float(v) for v in e2e.future_states.pos_y],
            "pos_z": [float(v) for v in e2e.future_states.pos_z],
        },
    }


def export_labels_payload(e2e: wod_e2ed_pb2.E2EDFrame) -> dict:
    preference_trajectories = [
        MessageToDict(
            traj,
            preserving_proto_field_name=True,
            use_integers_for_enums=False,
        )
        for traj in e2e.preference_trajectories
    ]
    return {
        "frame_name": e2e.frame.context.name,
        "preference_trajectories": preference_trajectories,
        "camera_labels_count": len(e2e.frame.camera_labels),
        "laser_labels_count": len(e2e.frame.laser_labels),
        "projected_lidar_labels_count": len(e2e.frame.projected_lidar_labels),
    }


def export_frame_metadata(
    *,
    e2e: wod_e2ed_pb2.E2EDFrame,
    scene_id: str,
    frame_name: str,
    image_rel_path: str,
    camera_name: str,
    shard_name: str,
) -> dict:
    camera_calibrations = export_camera_calibration_payload(e2e)
    selected_camera_calibration = next(
        (c for c in camera_calibrations if str(c.get("name", "")).upper() == camera_name.upper()),
        None,
    )
    return {
        "scene_id": scene_id,
        "frame_name": frame_name,
        "image_path": image_rel_path,
        "camera_name": camera_name,
        "timestamp_micros": int(e2e.frame.timestamp_micros),
        "shard_name": shard_name,
        "intent": int(e2e.intent),
        "available_cameras": [camera_enum_to_name(img.name) for img in e2e.frame.images],
        "selected_camera_calibration": selected_camera_calibration,
        "camera_calibrations": camera_calibrations,
        "vehicle_pose_transform": [float(v) for v in e2e.frame.pose.transform],
        "ego_status": {
            "past_states": {
                "pos_x": [float(v) for v in e2e.past_states.pos_x],
                "pos_y": [float(v) for v in e2e.past_states.pos_y],
                "vel_x": [float(v) for v in e2e.past_states.vel_x],
                "vel_y": [float(v) for v in e2e.past_states.vel_y],
                "accel_x": [float(v) for v in e2e.past_states.accel_x],
                "accel_y": [float(v) for v in e2e.past_states.accel_y],
            },
            "future_states": {
                "pos_x": [float(v) for v in e2e.future_states.pos_x],
                "pos_y": [float(v) for v in e2e.future_states.pos_y],
                "pos_z": [float(v) for v in e2e.future_states.pos_z],
            },
        },
        "label_counts": {
            "camera_labels_count": len(e2e.frame.camera_labels),
            "laser_labels_count": len(e2e.frame.laser_labels),
            "projected_lidar_labels_count": len(e2e.frame.projected_lidar_labels),
            "preference_trajectories_count": len(e2e.preference_trajectories),
        },
    }


def main() -> None:
    args = parse_args()
    images_root, annotations_root = resolve_output_roots(args)
    camera_name = args.camera.upper()

    shards = sorted(Path(p) for p in glob.glob(args.tfrecord_glob))
    if not shards:
        raise FileNotFoundError(f"No TFRecord shards matched: {args.tfrecord_glob}")
    if args.max_shards > 0:
        shards = shards[: args.max_shards]

    images_root.mkdir(parents=True, exist_ok=True)
    annotations_root.mkdir(parents=True, exist_ok=True)
    ego_status_dir = annotations_root / "ego_status"
    calibration_dir = annotations_root / "camera_calibration"
    labels_dir = annotations_root / "labels"

    shard_count = 0
    record_count = 0
    exported_images = 0
    exported_ego_status = 0
    exported_camera_calibration = 0
    exported_labels = 0
    skipped_existing = 0
    missing_camera_frames = 0
    failed_records: list[dict[str, str]] = []
    frame_manifest: list[dict[str, str | int]] = []

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

                frame_name = e2e.frame.context.name
                scene_id = parse_scene_id(frame_name)

                target_image = None
                for image in e2e.frame.images:
                    if camera_enum_to_name(image.name).upper() == camera_name:
                        target_image = image
                        break

                if target_image is None:
                    missing_camera_frames += 1
                    continue

                save_path = images_root / scene_id / f"CAM_{camera_name}" / f"{frame_name}_{camera_name}.jpg"
                image_rel_path = str(save_path.relative_to(images_root))
                ego_status_path = ego_status_dir / f"{frame_name}.json"
                calibration_path = calibration_dir / f"{frame_name}.json"
                labels_path = labels_dir / f"{frame_name}.json"

                if save_path.exists() and not args.overwrite:
                    skipped_existing += 1
                else:
                    write_bytes(save_path, target_image.image)
                    exported_images += 1

                frame_metadata_payload = export_frame_metadata(
                    e2e=e2e,
                    scene_id=scene_id,
                    frame_name=frame_name,
                    image_rel_path=image_rel_path,
                    camera_name=camera_name,
                    shard_name=shard.name,
                )
                write_json(ego_status_path, export_ego_status_payload(e2e))
                write_json(
                    calibration_path,
                    {
                        "frame_name": frame_name,
                        "timestamp_micros": int(e2e.frame.timestamp_micros),
                        "camera_calibrations": frame_metadata_payload["camera_calibrations"],
                        "vehicle_pose_transform": frame_metadata_payload["vehicle_pose_transform"],
                    },
                )
                write_json(labels_path, export_labels_payload(e2e))
                exported_ego_status += 1
                exported_camera_calibration += 1
                exported_labels += 1
                frame_manifest.append(
                    {
                        "scene_id": scene_id,
                        "frame_name": frame_name,
                        "image_path": image_rel_path,
                        "ego_status_path": str(ego_status_path.relative_to(annotations_root)),
                        "camera_calibration_path": str(calibration_path.relative_to(annotations_root)),
                        "labels_path": str(labels_path.relative_to(annotations_root)),
                        "timestamp_micros": frame_metadata_payload["timestamp_micros"],
                        "intent": frame_metadata_payload["intent"],
                        "shard_name": shard.name,
                    }
                )
            except Exception as exc:
                failed_records.append({"shard": shard.name, "record_index": str(record_count), "error": str(exc)})

        if args.max_records > 0 and record_count >= args.max_records:
            break

    summary = {
        "tfrecord_glob": args.tfrecord_glob,
        "images_root": str(images_root),
        "annotations_root": str(annotations_root),
        "camera": camera_name,
        "shards_processed": shard_count,
        "records_processed": record_count,
        "images_exported": exported_images,
        "ego_status_exported": exported_ego_status,
        "camera_calibration_exported": exported_camera_calibration,
        "labels_exported": exported_labels,
        "skipped_existing": skipped_existing,
        "missing_camera_frames": missing_camera_frames,
        "failed_count": len(failed_records),
        "failed_records": failed_records[:100],
        "frame_index_path": str((annotations_root / "frame_index.json").relative_to(annotations_root)),
    }
    manifest_path = annotations_root / "frame_index.json"
    summary_path = annotations_root / f"export_summary_{camera_name.lower()}.json"
    write_json(manifest_path, {"camera": camera_name, "count": len(frame_manifest), "frames": frame_manifest})
    write_json(summary_path, summary)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved manifest to: {manifest_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
