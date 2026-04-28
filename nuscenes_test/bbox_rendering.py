#!/usr/bin/env python3
"""Group scene frames and render random visible target annotations.

Workflow:
1) Build CAM_FRONT keyframe scenes using sample_data `prev`/`next`.
2) Split each scene into consecutive groups of 5 frames.
3) For each group's 5th frame, collect all `vehicle.*` and
   `human.pedestrian.*` annotations.
4) Randomly pick visible target annotations and render them with nuScenes SDK.

Outputs are written under each scene folder in `formatted_scenes`, e.g.:
  formatted_scenes/scene_001/group_001_vehicle_annotations.json
  formatted_scenes/scene_001/group_001_selected_vehicle_render.jpg
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, TypeVar

import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility
from PIL import Image

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


T = TypeVar("T")


def progress_iter(items: Iterable[T], *, total: int | None = None, desc: str = "") -> Iterator[T]:
    if tqdm is not None:
        yield from tqdm(items, total=total, desc=desc)
        return

    count = 0
    for item in items:
        count += 1
        if total is not None and (count == 1 or count == total or count % 10 == 0):
            label = f"{desc}: " if desc else ""
            print(f"{label}{count}/{total}")
        yield item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split each CAM_FRONT scene into 5-frame groups and render one random "
            "visible vehicle/pedestrian annotation from each group's 5th frame."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="nuScenes dataset root (default: script directory).",
    )
    parser.add_argument(
        "--sample-data-json",
        type=Path,
        default=None,
        help="Path to sample_data.json (default: <root>/v1.0-mini/sample_data.json).",
    )
    parser.add_argument(
        "--sample-json",
        type=Path,
        default=None,
        help="Path to sample.json (default: <root>/v1.0-mini/sample.json).",
    )
    parser.add_argument(
        "--sample-annotation-json",
        type=Path,
        default=None,
        help=(
            "Path to sample_annotation.json "
            "(default: <root>/v1.0-mini/sample_annotation.json)."
        ),
    )
    parser.add_argument(
        "--instance-json",
        type=Path,
        default=None,
        help="Path to instance.json (default: <root>/v1.0-mini/instance.json).",
    )
    parser.add_argument(
        "--category-json",
        type=Path,
        default=None,
        help="Path to category.json (default: <root>/v1.0-mini/category.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Formatted scenes root (default: <root>/formatted_scenes).",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1.0-mini",
        help="nuScenes version (default: v1.0-mini).",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=5,
        help="Consecutive frames per group (default: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible annotation selection.",
    )
    parser.add_argument(
        "--num-selected-objects",
        type=int,
        default=1,
        help=(
            "How many distinct target annotations to sample per group "
            "(default: 1)."
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_chain_root(token: str, by_token: Dict[str, dict], cache: Dict[str, str]) -> str:
    if token in cache:
        return cache[token]

    seen_path: List[str] = []
    current = token
    while True:
        if current in cache:
            root = cache[current]
            break
        record = by_token[current]
        prev_token = record.get("prev")
        seen_path.append(current)
        if not prev_token or prev_token not in by_token:
            root = current
            break
        current = prev_token

    for item in seen_path:
        cache[item] = root
    return root


def build_scenes_from_front(sample_data_records: List[dict]) -> List[List[dict]]:
    all_front = [
        rec
        for rec in sample_data_records
        if rec.get("filename", "").startswith("samples/CAM_FRONT/")
        or rec.get("filename", "").startswith("sweeps/CAM_FRONT/")
    ]
    front_samples = [
        rec
        for rec in all_front
        if rec.get("filename", "").startswith("samples/CAM_FRONT/")
        and rec.get("filename", "").endswith(".jpg")
    ]
    by_token: Dict[str, dict] = {rec["token"]: rec for rec in all_front}
    root_cache: Dict[str, str] = {}
    grouped: Dict[str, List[dict]] = {}

    for rec in front_samples:
        root = find_chain_root(rec["token"], by_token, root_cache)
        grouped.setdefault(root, []).append(rec)

    scenes = list(grouped.values())
    scenes.sort(key=lambda scene: min(item["timestamp"] for item in scene))
    for scene in scenes:
        scene.sort(key=lambda rec: rec["timestamp"])
    return scenes


def chunk_scene(scene: List[dict], group_size: int) -> List[List[dict]]:
    chunks: List[List[dict]] = []
    for start in range(0, len(scene), group_size):
        group = scene[start : start + group_size]
        if len(group) == group_size:
            chunks.append(group)
    return chunks


def keep_right_panel(image_path: Path) -> None:
    """Keep only the right half of the rendered 1x2 nuScenes annotation figure."""
    with Image.open(image_path) as image:
        width, height = image.size
        cropped = image.crop((width // 2, 0, width, height))
        cropped.save(image_path, format="JPEG", quality=95)


def bbox_volume(size: List[float] | None) -> float:
    """Return bbox volume for weighted sampling; fallback to 1.0."""
    if not size or len(size) != 3:
        return 1.0
    try:
        w, l, h = float(size[0]), float(size[1]), float(size[2])
    except (TypeError, ValueError):
        return 1.0
    volume = w * l * h
    return volume if volume > 0 else 1.0


def is_target_category(category_name: str) -> bool:
    name = category_name.lower()
    return name.startswith("vehicle") or name.startswith("human.pedestrian")


def is_annotation_visible_in_any_camera(nusc: NuScenes, ann_token: str) -> bool:
    ann_record = nusc.get("sample_annotation", ann_token)
    sample_record = nusc.get("sample", ann_record["sample_token"])
    camera_channels = [key for key in sample_record["data"] if key.startswith("CAM")]
    for channel in camera_channels:
        _, boxes, _ = nusc.get_sample_data(
            sample_record["data"][channel],
            box_vis_level=BoxVisibility.ANY,
            selected_anntokens=[ann_token],
        )
        if boxes:
            return True
    return False


def sample_target_records(
    target_records: List[dict],
    num_selected_objects: int,
) -> List[dict]:
    if num_selected_objects <= 0 or not target_records:
        return []
    sample_count = min(num_selected_objects, len(target_records))
    if sample_count == len(target_records):
        return random.sample(target_records, k=sample_count)
    return random.sample(target_records, k=sample_count)


def try_render_annotation(nusc: NuScenes, ann_token: str, output_path: Path) -> bool:
    try:
        nusc.render_annotation(
            ann_token,
            out_path=str(output_path),
            box_vis_level=BoxVisibility.ANY,
        )
    except AssertionError as exc:
        print(f"[WARN] Failed to render annotation {ann_token}: {exc}")
        plt.close("all")
        return False

    plt.close("all")
    keep_right_panel(output_path)
    return True


def compute_distance_to_ego(nusc: NuScenes, sample_token: str, translation: List[float] | None) -> float | None:
    if not translation or len(translation) != 3:
        return None

    sample_record = nusc.get("sample", sample_token)
    lidar_token = sample_record["data"].get("LIDAR_TOP")
    if not lidar_token:
        return None

    sample_data_record = nusc.get("sample_data", lidar_token)
    ego_pose_record = nusc.get("ego_pose", sample_data_record["ego_pose_token"])
    ego_translation = ego_pose_record.get("translation")
    if not ego_translation or len(ego_translation) != 3:
        return None

    dx = float(translation[0]) - float(ego_translation[0])
    dy = float(translation[1]) - float(ego_translation[1])
    dz = float(translation[2]) - float(ego_translation[2])
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def compute_target_priority(record: dict, sample_token: str, nusc: NuScenes) -> float:
    category_name = str(record.get("category_name", "")).lower()
    distance_to_ego = compute_distance_to_ego(nusc, sample_token, record.get("translation"))
    if distance_to_ego is None:
        distance_to_ego = 1e6

    category_weight = 1.0
    if category_name.startswith("human.pedestrian"):
        category_weight = 1.35
    elif "bus" in category_name or "truck" in category_name or "construction" in category_name:
        category_weight = 1.2
    elif "motorcycle" in category_name or "bicycle" in category_name:
        category_weight = 1.15

    lidar_bonus = min(float(record.get("num_lidar_pts") or 0), 50.0) / 100.0
    radar_bonus = min(float(record.get("num_radar_pts") or 0), 20.0) / 100.0
    visibility_bonus = 0.0
    try:
        visibility_bonus = float(record.get("visibility_token") or 0) * 0.05
    except (TypeError, ValueError):
        visibility_bonus = 0.0

    proximity_score = 1.0 / max(distance_to_ego, 1.0)
    priority = category_weight * (proximity_score + lidar_bonus + radar_bonus + visibility_bonus)
    return priority


def main() -> None:
    args = parse_args()
    if args.group_size <= 0:
        raise ValueError("--group-size must be > 0")
    if args.num_selected_objects <= 0:
        raise ValueError("--num-selected-objects must be > 0")

    random.seed(args.seed)
    root = args.root.resolve()
    sample_data_json = (
        args.sample_data_json.resolve()
        if args.sample_data_json
        else root / args.version / "sample_data.json"
    )
    sample_json = (
        args.sample_json.resolve() if args.sample_json else root / args.version / "sample.json"
    )
    sample_annotation_json = (
        args.sample_annotation_json.resolve()
        if args.sample_annotation_json
        else root / args.version / "sample_annotation.json"
    )
    instance_json = (
        args.instance_json.resolve() if args.instance_json else root / args.version / "instance.json"
    )
    category_json = (
        args.category_json.resolve() if args.category_json else root / args.version / "category.json"
    )
    output_dir = args.output_dir.resolve() if args.output_dir else root / "formatted_scenes"

    sample_data = load_json(sample_data_json)
    samples = load_json(sample_json)
    annotations = load_json(sample_annotation_json)
    instances = load_json(instance_json)
    categories = load_json(category_json)

    annotations_by_sample: Dict[str, List[dict]] = {}
    for ann in annotations:
        annotations_by_sample.setdefault(ann["sample_token"], []).append(ann)
    instance_by_token = {record["token"]: record for record in instances}
    category_by_token = {record["token"]: record for record in categories}
    scenes = build_scenes_from_front(sample_data)

    nusc = NuScenes(version=args.version, dataroot=str(root), verbose=False)

    total_groups = 0
    groups_with_visible_target = 0
    renders_written = 0

    for scene_idx, scene in progress_iter(
        enumerate(scenes, start=1),
        total=len(scenes),
        desc="Rendering scenes",
    ):
        scene_dir = output_dir / f"scene_{scene_idx:03d}"
        if not scene_dir.exists():
            print(f"[WARN] Scene directory does not exist, skipping: {scene_dir}")
            continue

        groups = chunk_scene(scene, args.group_size)
        for group_idx, group in progress_iter(
            enumerate(groups, start=1),
            total=len(groups),
            desc=f"scene_{scene_idx:03d} groups",
        ):
            total_groups += 1
            fifth_frame = group[-1]
            sample_token = fifth_frame["sample_token"]

            vehicle_records = []
            pedestrian_records = []
            for ann in annotations_by_sample.get(sample_token, []):
                instance = instance_by_token.get(ann["instance_token"], {})
                category_token = instance.get("category_token")
                category = category_by_token.get(category_token, {}).get("name", "")
                if not is_target_category(category):
                    continue
                record = {
                    "token": ann["token"],
                    "instance_token": ann["instance_token"],
                    "category_name": category,
                    "visibility_token": ann.get("visibility_token"),
                    "translation": ann.get("translation"),
                    "size": ann.get("size"),
                    "rotation": ann.get("rotation"),
                    "num_lidar_pts": ann.get("num_lidar_pts"),
                    "num_radar_pts": ann.get("num_radar_pts"),
                    "prev": ann.get("prev"),
                    "next": ann.get("next"),
                }
                record["distance_to_ego_m"] = compute_distance_to_ego(
                    nusc,
                    sample_token,
                    record.get("translation"),
                )
                record["priority_score"] = compute_target_priority(record, sample_token, nusc)
                if category.startswith("vehicle"):
                    vehicle_records.append(record)
                else:
                    pedestrian_records.append(record)

            visible_target_records = [
                record
                for record in vehicle_records + pedestrian_records
                if is_annotation_visible_in_any_camera(nusc, record["token"])
            ]

            candidate_records: List[dict] = []
            if visible_target_records:
                groups_with_visible_target += 1
                ranked_visible_targets = sorted(
                    visible_target_records,
                    key=lambda record: (
                        float(record.get("priority_score") or 0.0),
                        -(float(record.get("distance_to_ego_m") or 1e6)),
                    ),
                    reverse=True,
                )
                candidate_records = ranked_visible_targets[: args.num_selected_objects]

            if not candidate_records:
                report = {
                    "scene_id": f"scene_{scene_idx:03d}",
                    "group_id": f"group_{group_idx:03d}",
                    "base_group_id": f"group_{group_idx:03d}",
                    "group_size": args.group_size,
                    "frame_indices_1based": list(
                        range((group_idx - 1) * args.group_size + 1, group_idx * args.group_size + 1)
                    ),
                    "fifth_frame_sample_token": sample_token,
                    "fifth_frame_cam_front_filename": fifth_frame["filename"],
                    "vehicle_annotation_count": len(vehicle_records),
                    "vehicle_annotations": vehicle_records,
                    "pedestrian_annotation_count": len(pedestrian_records),
                    "pedestrian_annotations": pedestrian_records,
                    "visible_target_annotation_count": len(visible_target_records),
                    "visible_target_annotations": visible_target_records,
                    "selected_vehicle_annotation_token": None,
                    "selected_vehicle_render_path": None,
                    "selected_vehicle_annotation_tokens": [],
                    "selected_vehicle_render_paths": [],
                }
                report_path = scene_dir / f"group_{group_idx:03d}_vehicle_annotations.json"
                with report_path.open("w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2, ensure_ascii=True)
                continue

            selected_records: List[dict] = []
            selected_render_paths: List[str] = []
            for candidate in candidate_records:
                selected_idx = len(selected_records) + 1
                multi_object = len(candidate_records) > 1
                if multi_object:
                    group_id = f"group_{group_idx:03d}_obj{selected_idx:02d}"
                    render_filename = f"{group_id}_selected_vehicle_render.jpg"
                    report_filename = f"{group_id}_vehicle_annotations.json"
                else:
                    group_id = f"group_{group_idx:03d}"
                    render_filename = f"{group_id}_selected_vehicle_render.jpg"
                    report_filename = f"{group_id}_vehicle_annotations.json"

                selected_render_path = scene_dir / render_filename
                rendered = try_render_annotation(nusc, candidate["token"], selected_render_path)
                if not rendered:
                    continue

                selected_records.append(candidate)
                renders_written += 1
                selected_render_paths.append(str(selected_render_path.relative_to(output_dir)))

                report = {
                    "scene_id": f"scene_{scene_idx:03d}",
                    "group_id": group_id,
                    "base_group_id": f"group_{group_idx:03d}",
                    "group_size": args.group_size,
                    "frame_indices_1based": list(
                        range((group_idx - 1) * args.group_size + 1, group_idx * args.group_size + 1)
                    ),
                    "fifth_frame_sample_token": sample_token,
                    "fifth_frame_cam_front_filename": fifth_frame["filename"],
                    "vehicle_annotation_count": len(vehicle_records),
                    "vehicle_annotations": vehicle_records,
                    "pedestrian_annotation_count": len(pedestrian_records),
                    "pedestrian_annotations": pedestrian_records,
                    "visible_target_annotation_count": len(visible_target_records),
                    "visible_target_annotations": visible_target_records,
                    "selected_vehicle_annotation_token": candidate["token"],
                    "selected_vehicle_render_path": str(selected_render_path.relative_to(output_dir)),
                    "selected_vehicle_annotation_tokens": [
                        record["token"] for record in selected_records
                    ],
                    "selected_vehicle_render_paths": selected_render_paths.copy(),
                    "selected_vehicle_index": selected_idx - 1,
                    "selected_vehicle_count": len(selected_records),
                    "selected_category_name": candidate["category_name"],
                    "selected_priority_score": candidate.get("priority_score"),
                    "selected_distance_to_ego_m": candidate.get("distance_to_ego_m"),
                }
                report_path = scene_dir / report_filename
                with report_path.open("w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2, ensure_ascii=True)

            if not selected_records:
                report = {
                    "scene_id": f"scene_{scene_idx:03d}",
                    "group_id": f"group_{group_idx:03d}",
                    "base_group_id": f"group_{group_idx:03d}",
                    "group_size": args.group_size,
                    "frame_indices_1based": list(
                        range((group_idx - 1) * args.group_size + 1, group_idx * args.group_size + 1)
                    ),
                    "fifth_frame_sample_token": sample_token,
                    "fifth_frame_cam_front_filename": fifth_frame["filename"],
                    "vehicle_annotation_count": len(vehicle_records),
                    "vehicle_annotations": vehicle_records,
                    "pedestrian_annotation_count": len(pedestrian_records),
                    "pedestrian_annotations": pedestrian_records,
                    "visible_target_annotation_count": len(visible_target_records),
                    "visible_target_annotations": visible_target_records,
                    "selected_vehicle_annotation_token": None,
                    "selected_vehicle_render_path": None,
                    "selected_vehicle_annotation_tokens": [],
                    "selected_vehicle_render_paths": [],
                }
                report_path = scene_dir / f"group_{group_idx:03d}_vehicle_annotations.json"
                with report_path.open("w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2, ensure_ascii=True)

    print(f"Total scenes: {len(scenes)}")
    print(f"Total complete groups ({args.group_size} frames): {total_groups}")
    print(f"Groups with >=1 visible target annotation: {groups_with_visible_target}")
    print(f"Rendered selected target annotations: {renders_written}")


if __name__ == "__main__":
    main()
