#!/usr/bin/env python3
"""Group scene frames and render random vehicle annotations.

Workflow:
1) Build CAM_FRONT keyframe scenes using sample_data `prev`/`next`.
2) Split each scene into consecutive groups of 5 frames.
3) For each group's 5th frame, collect all `vehicle.*` annotations.
4) Randomly pick one vehicle annotation and render it with nuScenes SDK.

Outputs are written under each scene folder in `formatted_scenes`, e.g.:
  formatted_scenes/scene_001/group_001_vehicle_annotations.json
  formatted_scenes/scene_001/group_001_selected_vehicle_render.jpg
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split each CAM_FRONT scene into 5-frame groups and render one random "
            "vehicle annotation from each group's 5th frame."
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


def main() -> None:
    args = parse_args()
    if args.group_size <= 0:
        raise ValueError("--group-size must be > 0")

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
    groups_with_vehicle = 0
    renders_written = 0

    for scene_idx, scene in enumerate(scenes, start=1):
        scene_dir = output_dir / f"scene_{scene_idx:03d}"
        if not scene_dir.exists():
            print(f"[WARN] Scene directory does not exist, skipping: {scene_dir}")
            continue

        groups = chunk_scene(scene, args.group_size)
        for group_idx, group in enumerate(groups, start=1):
            total_groups += 1
            fifth_frame = group[-1]
            sample_token = fifth_frame["sample_token"]

            vehicle_records = []
            for ann in annotations_by_sample.get(sample_token, []):
                instance = instance_by_token.get(ann["instance_token"], {})
                category_token = instance.get("category_token")
                category = category_by_token.get(category_token, {}).get("name", "")
                if not category.startswith("vehicle"):
                    continue
                vehicle_records.append(
                    {
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
                )

            selected_token = None
            selected_render_path = None
            if vehicle_records:
                groups_with_vehicle += 1
                weights = [bbox_volume(item.get("size")) for item in vehicle_records]
                selected = random.choices(vehicle_records, weights=weights, k=1)[0]
                selected_token = selected["token"]
                selected_render_path = (
                    scene_dir / f"group_{group_idx:03d}_selected_vehicle_render.jpg"
                )
                nusc.render_annotation(selected_token, out_path=str(selected_render_path))
                plt.close("all")
                keep_right_panel(selected_render_path)
                renders_written += 1

            report = {
                "scene_id": f"scene_{scene_idx:03d}",
                "group_id": f"group_{group_idx:03d}",
                "group_size": args.group_size,
                "frame_indices_1based": list(
                    range((group_idx - 1) * args.group_size + 1, group_idx * args.group_size + 1)
                ),
                "fifth_frame_sample_token": sample_token,
                "fifth_frame_cam_front_filename": fifth_frame["filename"],
                "vehicle_annotation_count": len(vehicle_records),
                "vehicle_annotations": vehicle_records,
                "selected_vehicle_annotation_token": selected_token,
                "selected_vehicle_render_path": (
                    str(selected_render_path.relative_to(output_dir))
                    if selected_render_path is not None
                    else None
                ),
            }
            report_path = scene_dir / f"group_{group_idx:03d}_vehicle_annotations.json"
            with report_path.open("w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=True)

    print(f"Total scenes: {len(scenes)}")
    print(f"Total complete groups ({args.group_size} frames): {total_groups}")
    print(f"Groups with >=1 vehicle annotation: {groups_with_vehicle}")
    print(f"Rendered selected vehicle annotations: {renders_written}")


if __name__ == "__main__":
    main()
