#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = Path("/local1/rgao727/waymo_dataset/val")
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "waymo_5frame_front_bundles"

FRAME_RE = re.compile(r"^(?P<scene>.+)-(?P<idx>\d{3})_FRONT\.jpg$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Waymo front-view 5-frame bundles and aligned metadata bundles. "
            "This is a front-camera temporal bundle, not a 360-degree multi-camera stitch."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"Waymo split root (default: {DEFAULT_DATASET_ROOT}).",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Optional override for images root. Defaults to <dataset-root>/images.",
    )
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=None,
        help="Optional override for annotations root. Defaults to <dataset-root>/annotations.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--bundle-size",
        type=int,
        default=5,
        help="Number of consecutive frames per bundle (default: 5).",
    )
    parser.add_argument(
        "--bundle-stride",
        type=int,
        default=5,
        help=(
            "Stride between consecutive bundle anchors. "
            "Use 5 to create non-overlapping 5-frame groups (default: 5)."
        ),
    )
    parser.add_argument(
        "--limit-scenes",
        type=int,
        default=None,
        help="Optional limit on number of scenes for quick checks.",
    )
    parser.add_argument(
        "--limit-bundles",
        type=int,
        default=None,
        help="Optional limit on total number of bundles for quick checks.",
    )
    parser.add_argument(
        "--no-stitch-image",
        action="store_true",
        help="Only write JSON bundles and skip stitched JPG generation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing bundle JSON/JPG files.",
    )
    return parser.parse_args()


def progress_iter(items, *, total: int | None = None, desc: str = ""):
    if tqdm is not None:
        yield from tqdm(items, total=total, desc=desc)
        return
    count = 0
    for item in items:
        count += 1
        if total is not None and (count == 1 or count == total or count % 25 == 0):
            label = f"{desc}: " if desc else ""
            print(f"{label}{count}/{total}")
        yield item


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def frame_name_from_image(path: Path) -> tuple[str, int] | None:
    m = FRAME_RE.match(path.name)
    if m is None:
        return None
    return m.group("scene"), int(m.group("idx"))


def load_object_maps(annotations_dir: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    bbox_map: dict[str, dict] = {}
    distance_map: dict[str, dict] = {}

    bbox_json = annotations_dir / "sam3_bbox_annotations.json"
    if bbox_json.exists():
        bbox_payload = load_json(bbox_json)
        for item in bbox_payload.get("images", []):
            image_path = str(item.get("image_path", ""))
            frame_name = Path(image_path).stem.replace("_FRONT", "")
            bbox_map[frame_name] = item

    distance_json = annotations_dir / "bbox_distance_estimates.json"
    if distance_json.exists():
        distance_payload = load_json(distance_json)
        for item in distance_payload.get("results", []):
            frame_name = str(item.get("frame_name", "")).strip()
            if frame_name:
                distance_map[frame_name] = item

    return bbox_map, distance_map


def collect_scene_frames(images_dir: Path) -> dict[str, list[dict[str, Any]]]:
    by_scene: dict[str, list[dict[str, Any]]] = {}
    for image_path in images_dir.glob("*/CAM_FRONT/*.jpg"):
        parsed = frame_name_from_image(image_path)
        if parsed is None:
            continue
        scene_id, frame_idx = parsed
        by_scene.setdefault(scene_id, []).append(
            {
                "scene_id": scene_id,
                "frame_idx": frame_idx,
                "frame_name": f"{scene_id}-{frame_idx:03d}",
                "image_path": image_path,
            }
        )
    for scene_frames in by_scene.values():
        scene_frames.sort(key=lambda x: x["frame_idx"])
    return by_scene


def compose_stitched_strip(image_paths: list[Path], output_path: Path) -> bool:
    if Image is None:
        raise RuntimeError("Pillow is required for image stitching but is not installed.")
    images: list[Image.Image] = []
    try:
        for path in image_paths:
            if not path.exists():
                return False
            images.append(Image.open(path).convert("RGB"))
        cell_height = min(img.height for img in images)
        resized = [
            img.resize(
                (int(round(img.width * (cell_height / img.height))), cell_height),
                Image.Resampling.BILINEAR,
            )
            for img in images
        ]
        total_width = sum(img.width for img in resized)
        canvas = Image.new("RGB", (total_width, cell_height))
        x = 0
        for img in resized:
            canvas.paste(img, (x, 0))
            x += img.width
        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path, format="JPEG", quality=95)
        canvas.close()
        for img in resized:
            img.close()
        return True
    finally:
        for img in images:
            img.close()


def main() -> None:
    args = parse_args()
    images_dir = args.images_dir or (args.dataset_root / "images")
    annotations_dir = args.annotations_dir or (args.dataset_root / "annotations")
    output_dir = args.output_dir.resolve()

    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")
    if not annotations_dir.exists():
        raise FileNotFoundError(f"Annotations dir not found: {annotations_dir}")

    scenario_clusters = load_json(annotations_dir / "scenario_clusters.json")
    ego_dir = annotations_dir / "ego_status"
    labels_dir = annotations_dir / "labels"
    calib_dir = annotations_dir / "camera_calibration"
    bbox_map, distance_map = load_object_maps(annotations_dir)
    scenes = collect_scene_frames(images_dir)
    scene_ids = sorted(scenes.keys())
    if args.limit_scenes is not None:
        scene_ids = scene_ids[: args.limit_scenes]

    bundle_size = args.bundle_size
    if bundle_size <= 1:
        raise ValueError("--bundle-size must be >= 2")
    bundle_stride = args.bundle_stride
    if bundle_stride <= 0:
        raise ValueError("--bundle-stride must be >= 1")

    bundles_manifest: list[dict[str, Any]] = []
    total_bundles = 0

    for scene_id in progress_iter(scene_ids, total=len(scene_ids), desc="Building 5-frame bundles"):
        frames = scenes[scene_id]
        if len(frames) < bundle_size:
            continue
        scene_cluster = scenario_clusters.get(scene_id)
        scene_out = output_dir / scene_id
        scene_out.mkdir(parents=True, exist_ok=True)

        for end_pos in range(bundle_size - 1, len(frames), bundle_stride):
            if args.limit_bundles is not None and total_bundles >= args.limit_bundles:
                break

            bundle_frames = frames[end_pos - bundle_size + 1 : end_pos + 1]
            anchor = bundle_frames[-1]
            anchor_frame = anchor["frame_name"]
            group_id = f"group_{anchor['frame_idx']:03d}"
            bundle_id = f"{scene_id}:{group_id}"
            json_path = scene_out / f"{group_id}_bundle.json"
            stitch_path = scene_out / f"{group_id}_stitch.jpg"

            if json_path.exists() and not args.overwrite:
                bundles_manifest.append(
                    {
                        "bundle_id": bundle_id,
                        "scene_id": scene_id,
                        "group_id": group_id,
                        "bundle_json": str(json_path),
                        "stitched_image": str(stitch_path) if stitch_path.exists() else None,
                    }
                )
                total_bundles += 1
                continue

            frame_entries: list[dict[str, Any]] = []
            for rel_idx, frame in enumerate(bundle_frames):
                frame_name = frame["frame_name"]
                ego_status_path = ego_dir / f"{frame_name}.json"
                labels_path = labels_dir / f"{frame_name}.json"
                calib_path = calib_dir / f"{frame_name}.json"
                frame_entries.append(
                    {
                        "relative_index": rel_idx - (bundle_size - 1),
                        "frame_name": frame_name,
                        "frame_index": frame["frame_idx"],
                        "image_path": str(frame["image_path"]),
                        "ego_status_path": str(ego_status_path) if ego_status_path.exists() else None,
                        "labels_path": str(labels_path) if labels_path.exists() else None,
                        "camera_calibration_path": str(calib_path) if calib_path.exists() else None,
                        "bbox_annotations": bbox_map.get(frame_name),
                        "distance_estimates": distance_map.get(frame_name),
                    }
                )

            payload = {
                "dataset": "waymo_e2e_front_5frame",
                "bundle_id": bundle_id,
                "scene_id": scene_id,
                "group_id": group_id,
                "bundle_size": bundle_size,
                "anchor_frame_name": anchor_frame,
                "anchor_frame_index": anchor["frame_idx"],
                "scenario_cluster": scene_cluster,
                "camera_setup": "front_only",
                "frame_entries": frame_entries,
            }
            write_json(json_path, payload)

            stitched_output = None
            if not args.no_stitch_image:
                ok = compose_stitched_strip(
                    [frame["image_path"] for frame in bundle_frames],
                    stitch_path,
                )
                if ok:
                    stitched_output = str(stitch_path)

            bundles_manifest.append(
                {
                    "bundle_id": bundle_id,
                    "scene_id": scene_id,
                    "group_id": group_id,
                    "bundle_json": str(json_path),
                    "stitched_image": stitched_output,
                }
            )
            total_bundles += 1

        if args.limit_bundles is not None and total_bundles >= args.limit_bundles:
            break

    manifest = {
        "dataset_root": str(args.dataset_root),
        "images_dir": str(images_dir),
        "annotations_dir": str(annotations_dir),
        "output_dir": str(output_dir),
        "bundle_size": bundle_size,
        "bundle_stride": bundle_stride,
        "camera_setup": "front_only",
        "bundle_count": len(bundles_manifest),
        "bundles": bundles_manifest,
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(
        {
            "output_dir": str(output_dir),
            "bundle_count": len(bundles_manifest),
            "camera_setup": "front_only",
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
