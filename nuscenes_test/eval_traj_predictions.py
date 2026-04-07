#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


SUPPORTED_TRAJ_IDS = {
    "TRJ-5": 5,
    "TRJ-6": 4,
}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Evaluate trajectory-prediction outputs from run_eval.py results."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=script_dir / "vlm_responses_latest.json",
        help="Path to run_eval.py output JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save trajectory evaluation JSON.",
    )
    return parser.parse_args()


def safe_read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _coerce_point(point: Any) -> list[float] | None:
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        return None
    try:
        x = float(point[0])
        y = float(point[1])
    except (TypeError, ValueError):
        return None
    return [x, y]


def extract_traj_points(raw_text: str, expected_len: int) -> list[list[float]] | None:
    text = raw_text.strip()
    if not text:
        return None

    candidates: list[Any] = []
    try:
        candidates.append(json.loads(text))
    except Exception:
        pass

    if not candidates:
        start = text.find("[[")
        end = text.rfind("]]")
        if start != -1 and end != -1 and end >= start:
            snippet = text[start : end + 2]
            try:
                candidates.append(json.loads(snippet))
            except Exception:
                pass

    for candidate in candidates:
        if isinstance(candidate, dict):
            for key in ("trajectory", "answer", "points", "prediction"):
                if key in candidate:
                    candidate = candidate[key]
                    break

        if not isinstance(candidate, list) or len(candidate) != expected_len:
            continue

        points: list[list[float]] = []
        valid = True
        for item in candidate:
            point = _coerce_point(item)
            if point is None:
                valid = False
                break
            points.append(point)
        if valid:
            return points

    return None


def point_distance(a: list[float], b: list[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def compute_ade(pred: list[list[float]], gt: list[list[float]]) -> float:
    return sum(point_distance(p, g) for p, g in zip(pred, gt)) / float(len(gt))


def compute_fde(pred: list[list[float]], gt: list[list[float]]) -> float:
    return point_distance(pred[-1], gt[-1])


def main() -> None:
    args = parse_args()
    if not args.results.exists():
        raise FileNotFoundError(f"Results file not found: {args.results}")

    payload = safe_read_json(args.results)
    results = payload.get("results", [])
    if not isinstance(results, list):
        raise ValueError("Expected 'results' to be a list.")

    per_task: dict[str, dict[str, Any]] = {
        task_id: {
            "count": 0,
            "valid_count": 0,
            "format_score_sum": 0.0,
            "ade_sum": 0.0,
            "fde_sum": 0.0,
            "items": [],
        }
        for task_id in SUPPORTED_TRAJ_IDS
    }

    for item in results:
        task_id = str(item.get("id", "")).strip()
        if task_id not in SUPPORTED_TRAJ_IDS:
            continue

        expected_len = SUPPORTED_TRAJ_IDS[task_id]
        gt_raw = item.get("ground_truth")
        pred_raw = item.get("model_response", "")
        if not isinstance(gt_raw, str):
            gt_raw = json.dumps(gt_raw)
        gt_points = extract_traj_points(gt_raw, expected_len)
        pred_points = extract_traj_points(str(pred_raw), expected_len)

        bucket = per_task[task_id]
        bucket["count"] += 1

        row: dict[str, Any] = {
            "id": task_id,
            "scene_id": item.get("scene_id"),
            "group_id": item.get("group_id"),
            "question": item.get("question"),
            "ground_truth": gt_raw,
            "model_response": pred_raw,
            "format_score": 0,
            "parsed_prediction": pred_points,
            "parsed_ground_truth": gt_points,
            "ade": None,
            "fde": None,
        }

        if gt_points is not None and pred_points is not None:
            row["format_score"] = 1
            row["ade"] = compute_ade(pred_points, gt_points)
            row["fde"] = compute_fde(pred_points, gt_points)
            bucket["valid_count"] += 1
            bucket["format_score_sum"] += 1.0
            bucket["ade_sum"] += float(row["ade"])
            bucket["fde_sum"] += float(row["fde"])

        bucket["items"].append(row)

    summary: dict[str, Any] = {}
    for task_id, bucket in per_task.items():
        count = bucket["count"]
        valid_count = bucket["valid_count"]
        summary[task_id] = {
            "count": count,
            "valid_count": valid_count,
            "format_score_mean": (bucket["format_score_sum"] / count) if count > 0 else None,
            "ade_mean_valid_only": (bucket["ade_sum"] / valid_count) if valid_count > 0 else None,
            "fde_mean_valid_only": (bucket["fde_sum"] / valid_count) if valid_count > 0 else None,
        }

    output = {
        "meta": {
            "results_file": str(args.results.resolve()),
            "supported_task_ids": sorted(SUPPORTED_TRAJ_IDS.keys()),
            "invalid_prediction_policy": "format_score=0 when parsing fails or point count mismatches",
        },
        "summary": summary,
        "details": per_task,
    }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"Saved trajectory evaluation to: {args.output}")

    for task_id in sorted(summary):
        stats = summary[task_id]
        print(
            f"{task_id}: count={stats['count']} "
            f"valid={stats['valid_count']} "
            f"format_score_mean={stats['format_score_mean']} "
            f"ade_mean_valid_only={stats['ade_mean_valid_only']} "
            f"fde_mean_valid_only={stats['fde_mean_valid_only']}"
        )


if __name__ == "__main__":
    main()
