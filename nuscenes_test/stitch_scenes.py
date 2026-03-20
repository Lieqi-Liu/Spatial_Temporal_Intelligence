#!/usr/bin/env python3
"""Group nuScenes keyframes into per-scene stitched multi-view images.

This script reads nuScenes `sample_data.json`, builds temporal chains from
`prev`/`next` links on CAM_FRONT, and renders a 2x3 grid image per frame:

    Row 1: CAM_FRONT_LEFT, CAM_FRONT, CAM_FRONT_RIGHT
    Row 2: CAM_BACK_LEFT, CAM_BACK, CAM_BACK_RIGHT

The results are written into:

    formatted_scenes/
      scene_001/
      scene_002/
      ...
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

from PIL import Image


GRID_CHANNELS = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create per-scene stitched 6-camera images."
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
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <root>/formatted_scenes).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing output directory before writing.",
    )
    return parser.parse_args()


def load_sample_data(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_chain_root(token: str, by_token: Dict[str, dict], cache: Dict[str, str]) -> str:
    if token in cache:
        return cache[token]

    path: List[str] = []
    current = token
    while True:
        if current in cache:
            root = cache[current]
            break
        record = by_token[current]
        prev_token = record.get("prev")
        path.append(current)
        if not prev_token or prev_token not in by_token:
            root = current
            break
        current = prev_token

    for seen in path:
        cache[seen] = root
    return root


def build_scene_chains(all_cam_records: List[dict], sample_cam_records: List[dict]) -> List[List[dict]]:
    by_token: Dict[str, dict] = {record["token"]: record for record in all_cam_records}
    root_cache: Dict[str, str] = {}
    grouped: Dict[str, List[dict]] = {}

    for record in sample_cam_records:
        root = find_chain_root(record["token"], by_token, root_cache)
        grouped.setdefault(root, []).append(record)

    scenes = list(grouped.values())
    scenes.sort(key=lambda scene: min(item["timestamp"] for item in scene))
    for scene in scenes:
        scene.sort(key=lambda record: record["timestamp"])
    return scenes


def build_sample_camera_index(records: List[dict]) -> Dict[str, Dict[str, dict]]:
    sample_to_views: Dict[str, Dict[str, dict]] = {}
    for record in records:
        filename = record.get("filename", "")
        if not filename.startswith("samples/CAM_") or not filename.endswith(".jpg"):
            continue
        parts = filename.split("/")
        if len(parts) < 3:
            continue
        channel = parts[1]
        if channel not in GRID_CHANNELS:
            continue
        sample_token = record.get("sample_token")
        if not sample_token:
            continue
        sample_to_views.setdefault(sample_token, {})[channel] = record
    return sample_to_views


def compose_grid_image(image_paths: List[Path], output_path: Path) -> bool:
    images: List[Image.Image] = []
    try:
        for path in image_paths:
            if not path.exists():
                print(f"[WARN] Missing file, skipping frame: {path}")
                return False
            images.append(Image.open(path).convert("RGB"))

        cell_width = min(image.width for image in images)
        cell_height = min(image.height for image in images)
        resized = [image.resize((cell_width, cell_height), Image.Resampling.BILINEAR) for image in images]

        grid = Image.new("RGB", (3 * cell_width, 2 * cell_height))
        for idx, image in enumerate(resized):
            row = idx // 3
            col = idx % 3
            grid.paste(image, (col * cell_width, row * cell_height))
        grid.save(output_path, format="JPEG", quality=95)
        grid.close()
        for image in resized:
            image.close()
        return True
    finally:
        for image in images:
            image.close()


def render_scenes(
    root: Path,
    scenes: List[List[dict]],
    sample_to_views: Dict[str, Dict[str, dict]],
    output_dir: Path,
) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_rendered = 0
    for idx, scene in enumerate(scenes, start=1):
        scene_dir = output_dir / f"scene_{idx:03d}"
        scene_dir.mkdir(parents=True, exist_ok=True)
        scene.sort(key=lambda record: record["timestamp"])

        for frame_idx, front_record in enumerate(scene, start=1):
            sample_token = front_record.get("sample_token")
            if not sample_token:
                print("[WARN] Missing sample_token, skipping frame.")
                continue

            views = sample_to_views.get(sample_token, {})
            missing_channels = [channel for channel in GRID_CHANNELS if channel not in views]
            if missing_channels:
                print(
                    f"[WARN] Incomplete camera set for sample_token {sample_token}, "
                    f"missing {missing_channels}. Skipping frame."
                )
                continue

            image_paths = [root / views[channel]["filename"] for channel in GRID_CHANNELS]
            front_name = Path(front_record["filename"]).stem
            output_path = scene_dir / f"{frame_idx:03d}_{front_name}_grid.jpg"
            if compose_grid_image(image_paths, output_path):
                total_rendered += 1

    print(f"Created {len(scenes)} scene folders in: {output_dir}")
    print(f"Rendered {total_rendered} stitched multi-view images.")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()

    sample_data_json = (
        args.sample_data_json.resolve()
        if args.sample_data_json
        else (root / "v1.0-mini" / "sample_data.json")
    )
    output_dir = (
        args.output_dir.resolve() if args.output_dir else (root / "formatted_scenes")
    )

    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output directory already exists: {output_dir}\n"
            "Use --overwrite to recreate it."
        )

    records = load_sample_data(sample_data_json)
    all_front_records = [
        record
        for record in records
        if record.get("filename", "").startswith("samples/CAM_FRONT/")
        or record.get("filename", "").startswith("sweeps/CAM_FRONT/")
    ]
    sample_front_records = [
        record
        for record in all_front_records
        if record.get("filename", "").startswith("samples/CAM_FRONT/")
        and record.get("filename", "").endswith(".jpg")
    ]
    all_front_records.sort(key=lambda record: record["timestamp"])
    sample_front_records.sort(key=lambda record: record["timestamp"])

    scenes = build_scene_chains(all_front_records, sample_front_records)
    sample_to_views = build_sample_camera_index(records)
    render_scenes(
        root=root,
        scenes=scenes,
        sample_to_views=sample_to_views,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
