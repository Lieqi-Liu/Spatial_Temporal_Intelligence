#!/usr/bin/env python3
"""
Extract available Waymo E2E annotations for already-exported image frames.

Reads frame names from:
  <images_root>/<sequence_id>/CAM_<CAMERA>/<sequence_id>-<frame_idx>_<CAMERA>.jpg

Then scans train TFRecord shards and exports, per matched frame:
  - ego_status/<frame_name>.json
  - labels/<frame_name>.json
  - camera_calibration/<frame_name>.json
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


DEFAULT_TRAIN_GLOB = (
    "/local1/lieqiliu/waymo_open_dataset_end_to_end_camera_v_1_0_0/"
    "training_*.tfrecord-*-of-00263"
)
DEFAULT_IMAGES_ROOT = "/home/lieqiliu/AutoDriving/waymo_test2/images"
DEFAULT_OUTPUT_ROOT = "/home/lieqiliu/AutoDriving/waymo_test2/annotations"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Waymo E2E annotations for existing image frames."
    )
    parser.add_argument(
        "--images-root",
        type=str,
        default=DEFAULT_IMAGES_ROOT,
        help="Root folder containing exported sequence images.",
    )
    parser.add_argument(
        "--train-glob",
        type=str,
        default=DEFAULT_TRAIN_GLOB,
        help="Glob for Waymo train TFRecord shards.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root for annotations.",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="FRONT",
        help="Camera suffix in image names, e.g. FRONT.",
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=0,
        help="If > 0, only scan first N shards.",
    )
    return parser.parse_args()


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


def collect_target_frame_names(images_root: Path, camera: str) -> set[str]:
    suffix = f"_{camera.upper()}.jpg"
    targets: set[str] = set()
    for jpg_path in images_root.rglob("*.jpg"):
        name = jpg_path.name
        if not name.endswith(suffix):
            continue
        targets.add(name[: -len(suffix)])
    return targets


def list_float(values) -> list[float]:
    return [float(v) for v in values]


def export_ego_status(e2e: wod_e2ed_pb2.E2EDFrame) -> dict:
    return {
        "frame_name": e2e.frame.context.name,
        "intent": int(e2e.intent),
        "past_states": {
            "pos_x": list_float(e2e.past_states.pos_x),
            "pos_y": list_float(e2e.past_states.pos_y),
            "vel_x": list_float(e2e.past_states.vel_x),
            "vel_y": list_float(e2e.past_states.vel_y),
            "accel_x": list_float(e2e.past_states.accel_x),
            "accel_y": list_float(e2e.past_states.accel_y),
        },
        "future_states": {
            "pos_x": list_float(e2e.future_states.pos_x),
            "pos_y": list_float(e2e.future_states.pos_y),
            "pos_z": list_float(e2e.future_states.pos_z),
        },
    }


def export_labels(e2e: wod_e2ed_pb2.E2EDFrame) -> dict:
    pref = [
        MessageToDict(
            traj,
            preserving_proto_field_name=True,
            use_integers_for_enums=False,
        )
        for traj in e2e.preference_trajectories
    ]
    return {
        "frame_name": e2e.frame.context.name,
        "preference_trajectories": pref,
        "camera_labels_count": len(e2e.frame.camera_labels),
        "laser_labels_count": len(e2e.frame.laser_labels),
        "projected_lidar_labels_count": len(e2e.frame.projected_lidar_labels),
    }


def export_camera_calibration(e2e: wod_e2ed_pb2.E2EDFrame) -> dict:
    calibrations = []
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
    return {
        "frame_name": e2e.frame.context.name,
        "timestamp_micros": int(e2e.frame.timestamp_micros),
        "camera_calibrations": calibrations,
        "vehicle_pose_transform": [float(v) for v in e2e.frame.pose.transform],
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    images_root = Path(args.images_root).resolve()
    output_root = Path(args.output_root).resolve()

    if not images_root.exists():
        raise FileNotFoundError(f"Images root does not exist: {images_root}")

    target_frames = collect_target_frame_names(images_root, args.camera)
    if not target_frames:
        raise RuntimeError(
            f"No target frames found in {images_root} with suffix _{args.camera.upper()}.jpg"
        )

    shards = sorted(Path(p) for p in glob.glob(args.train_glob))
    if not shards:
        raise FileNotFoundError(f"No train shards matched: {args.train_glob}")
    if args.max_shards > 0:
        shards = shards[: args.max_shards]

    ego_dir = output_root / "ego_status"
    labels_dir = output_root / "labels"
    calib_dir = output_root / "camera_calibration"
    output_root.mkdir(parents=True, exist_ok=True)

    matched: set[str] = set()
    scanned_records = 0

    for shard in shards:
        dataset = tf.data.TFRecordDataset([str(shard)], compression_type="")
        for raw in dataset:
            scanned_records += 1
            e2e = wod_e2ed_pb2.E2EDFrame()
            e2e.ParseFromString(raw.numpy())
            frame_name = e2e.frame.context.name
            if frame_name not in target_frames:
                continue
            if frame_name in matched:
                continue

            write_json(ego_dir / f"{frame_name}.json", export_ego_status(e2e))
            write_json(labels_dir / f"{frame_name}.json", export_labels(e2e))
            write_json(calib_dir / f"{frame_name}.json", export_camera_calibration(e2e))

            matched.add(frame_name)
            if len(matched) == len(target_frames):
                break
        if len(matched) == len(target_frames):
            break

    missing = sorted(target_frames - matched)
    summary = {
        "images_root": str(images_root),
        "train_glob": args.train_glob,
        "camera": args.camera.upper(),
        "target_frames": len(target_frames),
        "matched_frames": len(matched),
        "missing_frames": len(missing),
        "scanned_records": scanned_records,
        "output_root": str(output_root),
        "subdirs": {
            "ego_status": str(ego_dir),
            "labels": str(labels_dir),
            "camera_calibration": str(calib_dir),
        },
        "missing_frame_names": missing,
    }
    write_json(output_root / "index.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
