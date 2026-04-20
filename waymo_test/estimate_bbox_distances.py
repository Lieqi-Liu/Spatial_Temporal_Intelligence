#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


DEFAULT_ANNOTATIONS_JSON = Path(
    "/home/lieqiliu/AutoDriving/waymo_test2/annotations/sam3_bbox_annotations.json"
)
DEFAULT_CALIB_DIR = Path("/home/lieqiliu/AutoDriving/waymo_test2/annotations/camera_calibration")
DEFAULT_OUTPUT_JSON = Path(
    "/home/lieqiliu/AutoDriving/waymo_test2/annotations/bbox_distance_estimates.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate per-bbox distance from Waymo bbox + camera calibration JSON."
    )
    parser.add_argument(
        "--annotations-json",
        type=Path,
        default=DEFAULT_ANNOTATIONS_JSON,
        help="Path to SAM3 bbox annotation JSON.",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=DEFAULT_CALIB_DIR,
        help="Directory containing per-frame camera calibration JSON files.",
    )
    parser.add_argument(
        "--camera-name",
        type=str,
        default="FRONT",
        help="Camera calibration name to use, default FRONT.",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default=None,
        help=(
            "Relative image path in annotations JSON, "
            "e.g. seq/CAM_FRONT/seq-030_FRONT.jpg."
        ),
    )
    parser.add_argument(
        "--frame-name",
        type=str,
        default=None,
        help="Frame name, e.g. f17e...-034.",
    )
    parser.add_argument("--scene-id", type=str, default=None, help="Sequence id.")
    parser.add_argument("--frame-idx", type=int, default=None, help="Frame index, e.g. 34.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Path to save distance estimation results as JSON.",
    )
    return parser.parse_args()


def normalize_target(args: argparse.Namespace) -> tuple[str | None, str] | None:
    if args.image_path:
        image_rel = args.image_path
        frame_name = Path(image_rel).stem
        if frame_name.endswith("_FRONT"):
            frame_name = frame_name[: -len("_FRONT")]
        return image_rel, frame_name

    if args.frame_name:
        return None, args.frame_name

    if args.scene_id and args.frame_idx is not None:
        frame_name = f"{args.scene_id}-{args.frame_idx:03d}"
        image_rel = f"{args.scene_id}/CAM_FRONT/{frame_name}_FRONT.jpg"
        return image_rel, frame_name

    return None


def load_annotations_payload(annotations_json: Path) -> dict:
    if not annotations_json.exists():
        raise FileNotFoundError(f"Annotations JSON not found: {annotations_json}")
    return json.loads(annotations_json.read_text(encoding="utf-8"))


def load_boxes(
    annotations_payload: dict,
    image_rel: str | None,
    frame_name: str,
) -> tuple[str, int, int, list[dict]]:
    images = annotations_payload.get("images", [])

    if image_rel is not None:
        for image in images:
            if image.get("image_path") == image_rel:
                return image_rel, int(image["width"]), int(image["height"]), image.get(
                    "annotations", []
                )
        raise FileNotFoundError(f"No annotation entry found for image_path={image_rel}")

    suffix = f"{frame_name}_FRONT.jpg"
    for image in images:
        image_path = str(image.get("image_path", ""))
        if image_path.endswith(suffix):
            return image_path, int(image["width"]), int(image["height"]), image.get(
                "annotations", []
            )

    raise FileNotFoundError(f"No annotation entry found for frame_name={frame_name}")


def load_camera_calibration(calibration_dir: Path, frame_name: str, camera_name: str) -> dict:
    calib_path = calibration_dir / f"{frame_name}.json"
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration JSON not found: {calib_path}")
    payload = json.loads(calib_path.read_text(encoding="utf-8"))

    calibrations = payload.get("camera_calibrations", [])
    for calib in calibrations:
        if str(calib.get("name", "")).upper() == camera_name.upper():
            return calib
    raise RuntimeError(f"Camera calibration {camera_name} not found in {calib_path}")


def ray_ground_distance(
    u: float,
    v: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    t_cam_to_vehicle: np.ndarray,
) -> tuple[float, dict]:
    # Waymo camera frame: x-forward, y-left, z-up.
    # Pixel mapping: u = cx - fx*(y/x), v = cy - fy*(z/x).
    ray_cam = np.array([1.0, -(u - cx) / fx, -(v - cy) / fy], dtype=np.float64)

    rotation = t_cam_to_vehicle[:3, :3]
    translation = t_cam_to_vehicle[:3, 3]
    ray_vehicle = rotation @ ray_cam
    origin_vehicle = translation

    rz = float(ray_vehicle[2])
    oz = float(origin_vehicle[2])
    if abs(rz) < 1e-8:
        return float("nan"), {"reason": "ray_parallel_to_ground"}

    scale = -oz / rz
    if scale <= 0:
        return float("nan"), {"reason": "intersection_behind_camera"}

    hit = origin_vehicle + scale * ray_vehicle
    forward = float(hit[0])
    lateral = float(hit[1])
    if forward <= 0:
        return float("nan"), {"reason": "point_behind_ego", "forward_m": forward}

    distance = math.sqrt(forward * forward + lateral * lateral)
    return distance, {
        "forward_m": forward,
        "lateral_m": lateral,
        "hit_z_m": float(hit[2]),
    }


def score_transform(
    boxes: list[dict],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    t_cam_to_vehicle: np.ndarray,
) -> tuple[int, list[dict]]:
    estimates: list[dict] = []
    valid = 0
    for ann in boxes:
        x1, y1, x2, y2 = [float(v) for v in ann["bbox_xyxy"]]
        u = 0.5 * (x1 + x2)
        v = y2
        distance, details = ray_ground_distance(u, v, fx, fy, cx, cy, t_cam_to_vehicle)
        out = {
            "label": ann.get("label", ""),
            "score": ann.get("score", None),
            "bbox_xyxy": ann["bbox_xyxy"],
            "sample_pixel_uv": [u, v],
            "distance_m": None if math.isnan(distance) else round(distance, 3),
            "details": details,
        }
        if out["distance_m"] is not None:
            valid += 1
        estimates.append(out)
    return valid, estimates


def estimate_for_target(
    annotations_payload: dict,
    calibration_dir: Path,
    camera_name: str,
    image_rel: str | None,
    frame_name: str,
) -> dict:
    resolved_image_rel, width, height, boxes = load_boxes(
        annotations_payload=annotations_payload,
        image_rel=image_rel,
        frame_name=frame_name,
    )
    if not boxes:
        raise RuntimeError("No boxes found for requested frame/image.")

    calibration = load_camera_calibration(calibration_dir, frame_name, camera_name)
    intrinsic = calibration.get("intrinsic", [])
    extrinsic = calibration.get("extrinsic", [])
    if len(intrinsic) < 4:
        raise RuntimeError("Intrinsic vector is incomplete; need at least fx, fy, cx, cy.")
    if len(extrinsic) != 16:
        raise RuntimeError("Extrinsic must contain 16 values (4x4 transform).")

    fx, fy, cx, cy = [float(intrinsic[i]) for i in range(4)]
    t = np.array(extrinsic, dtype=np.float64).reshape(4, 4)
    t_inv = np.linalg.inv(t)

    # Keep the transform direction that yields more valid ground intersections.
    valid_direct, est_direct = score_transform(boxes, fx, fy, cx, cy, t)
    valid_inv, est_inv = score_transform(boxes, fx, fy, cx, cy, t_inv)
    use_inverse = valid_inv > valid_direct
    chosen = est_inv if use_inverse else est_direct

    return {
        "frame_name": frame_name,
        "image_path": resolved_image_rel,
        "image_size": {"width": width, "height": height},
        "camera_name": camera_name.upper(),
        "intrinsic": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        "transform_choice": {
            "used_inverse_extrinsic": use_inverse,
            "valid_direct": valid_direct,
            "valid_inverse": valid_inv,
        },
        "distance_estimates": chosen,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    annotations_payload = load_annotations_payload(args.annotations_json)
    target = normalize_target(args)

    if target is not None:
        image_rel, frame_name = target
        output = estimate_for_target(
            annotations_payload=annotations_payload,
            calibration_dir=args.calibration_dir,
            camera_name=args.camera_name,
            image_rel=image_rel,
            frame_name=frame_name,
        )
        write_json(args.output_json, output)
        print(f"Saved distance estimates to: {args.output_json}")
        print(json.dumps(output, indent=2))
        return

    results: list[dict] = []
    skipped_no_boxes = 0
    failed_frames: list[dict] = []
    for image in annotations_payload.get("images", []):
        image_path = str(image.get("image_path", ""))
        if not image.get("annotations"):
            skipped_no_boxes += 1
            continue

        frame_name = Path(image_path).stem
        if frame_name.endswith("_FRONT"):
            frame_name = frame_name[: -len("_FRONT")]
        if not frame_name:
            continue

        try:
            results.append(
                estimate_for_target(
                    annotations_payload=annotations_payload,
                    calibration_dir=args.calibration_dir,
                    camera_name=args.camera_name,
                    image_rel=image_path,
                    frame_name=frame_name,
                )
            )
        except Exception as exc:
            failed_frames.append({"frame_name": frame_name, "image_path": image_path, "error": str(exc)})

    output = {
        "mode": "all_images",
        "camera_name": args.camera_name.upper(),
        "total_images": len(results),
        "skipped_no_boxes": skipped_no_boxes,
        "failed_count": len(failed_frames),
        "failed_frames": failed_frames,
        "results": results,
    }
    write_json(args.output_json, output)
    print(f"Saved distance estimates to: {args.output_json}")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
