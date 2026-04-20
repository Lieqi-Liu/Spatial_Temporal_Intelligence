#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_DISTANCE_JSON = Path(
    "/home/lieqiliu/AutoDriving/waymo_test2/annotations/bbox_distance_estimates.json"
)
DEFAULT_QUESTIONS_JSON = Path("/home/lieqiliu/AutoDriving/nuscenes_mini/questions.json")
DEFAULT_OUTPUT_JSON = Path(
    "/home/lieqiliu/AutoDriving/waymo_test2/annotations/generated_answers_waymo.json"
)
DEFAULT_QUESTIONS_OUTPUT = Path(
    "/home/lieqiliu/AutoDriving/waymo_test2/annotations/questions_with_answers_waymo.json"
)
DEFAULT_FRQ_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"

FRAME_DT_S = 0.1
LANE_HALF_WIDTH_M = 1.8
LANE_WIDTH_M = 3.6


@dataclass
class Obj:
    obj_id: str
    label: str
    coarse: str
    score: float
    bbox: list[float]
    u: float
    v: float
    distance_m: float | None
    forward_m: float | None
    lateral_m: float | None
    track_id: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Waymo GT answers for question templates using distance estimates."
    )
    parser.add_argument(
        "--distance-json",
        type=Path,
        default=DEFAULT_DISTANCE_JSON,
        help="Path to bbox_distance_estimates.json.",
    )
    parser.add_argument(
        "--questions-json",
        type=Path,
        default=DEFAULT_QUESTIONS_JSON,
        help="Path to question template JSON (e.g., nuscenes_mini/questions.json).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Output path for generated answers JSON.",
    )
    parser.add_argument(
        "--questions-output",
        type=Path,
        default=DEFAULT_QUESTIONS_OUTPUT,
        help="Path to write questions file with generated answers.",
    )
    parser.add_argument(
        "--disable-frq-generation",
        action="store_true",
        help="Skip FRQ generation with Qwen and only output MCQ GT answers.",
    )
    parser.add_argument(
        "--frq-model",
        type=str,
        default=DEFAULT_FRQ_MODEL,
        help=f"Model name/path used for FRQ generation (default: {DEFAULT_FRQ_MODEL}).",
    )
    parser.add_argument(
        "--frq-tensor-parallel-size",
        type=int,
        default=2,
        help="Tensor parallel size for FRQ vLLM generation (default: 2).",
    )
    parser.add_argument(
        "--frq-max-tokens",
        type=int,
        default=256,
        help="Max generated tokens per FRQ answer (default: 256).",
    )
    parser.add_argument(
        "--frq-temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for FRQ generation (default: 0.2).",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"JSON not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def parse_frame_name(frame_name: str) -> tuple[str, int]:
    if "-" not in frame_name:
        return frame_name, -1
    scene_id, idx_str = frame_name.rsplit("-", 1)
    try:
        return scene_id, int(idx_str)
    except ValueError:
        return scene_id, -1


def coarse_class(label: str) -> str:
    x = label.lower()
    if "pedestrian" in x:
        return "pedestrian"
    if "bicycle" in x or "motorcycle" in x:
        return "bicycle"
    if "bus" in x or "truck" in x or "car" in x:
        return "vehicle"
    if "cone" in x or "barrier" in x or "sign" in x:
        return "static_obstacle"
    return "other"


def iou_xyxy(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def dedupe_objects(raw_objects: list[Obj]) -> list[Obj]:
    kept: list[Obj] = []
    for obj in sorted(raw_objects, key=lambda o: o.score, reverse=True):
        duplicate = False
        for prev in kept:
            if obj.coarse != prev.coarse:
                continue
            if iou_xyxy(obj.bbox, prev.bbox) > 0.9:
                duplicate = True
                break
        if not duplicate:
            kept.append(obj)
    return kept


def bbox_center(b: list[float]) -> tuple[float, float]:
    return 0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])


def assign_track_ids(frames: list[dict]) -> None:
    by_scene: dict[str, list[dict]] = {}
    for f in frames:
        by_scene.setdefault(f["scene_id"], []).append(f)

    for scene_id, scene_frames in by_scene.items():
        scene_frames.sort(key=lambda x: x["frame_idx"])
        next_track_idx = 0
        prev_objects: list[Obj] = []
        for frame in scene_frames:
            cur_objects: list[Obj] = frame["objects"]
            prev_used: set[int] = set()
            # Greedy matching by IoU with class consistency.
            for obj in sorted(cur_objects, key=lambda o: o.score, reverse=True):
                best_idx = -1
                best_score = 0.0
                ocx, ocy = bbox_center(obj.bbox)
                for i, prev in enumerate(prev_objects):
                    if i in prev_used:
                        continue
                    if prev.coarse != obj.coarse:
                        continue
                    iou = iou_xyxy(obj.bbox, prev.bbox)
                    pcx, pcy = bbox_center(prev.bbox)
                    pix_dist = math.hypot(ocx - pcx, ocy - pcy)
                    if iou < 0.2 and pix_dist > 120.0:
                        continue
                    score = iou - 0.001 * pix_dist
                    if score > best_score:
                        best_score = score
                        best_idx = i
                if best_idx >= 0 and prev_objects[best_idx].track_id:
                    obj.track_id = prev_objects[best_idx].track_id
                    prev_used.add(best_idx)
                else:
                    obj.track_id = f"{scene_id}:trk_{next_track_idx:06d}"
                    next_track_idx += 1
            prev_objects = cur_objects


def build_object_ref(obj: Obj) -> str:
    if obj.lateral_m is None or obj.forward_m is None:
        pos = "nearby"
    else:
        if obj.forward_m >= 0:
            if obj.lateral_m > 1.5:
                pos = "front-left"
            elif obj.lateral_m < -1.5:
                pos = "front-right"
            else:
                pos = "front"
        else:
            if obj.lateral_m > 1.5:
                pos = "rear-left"
            elif obj.lateral_m < -1.5:
                pos = "rear-right"
            else:
                pos = "rear"
    dist = "unknown"
    if obj.distance_m is not None:
        dist = f"{int(round(obj.distance_m))}m"
    return f"the {obj.coarse.replace('_', ' ')} {pos} at about {dist}"


def sp2_from_pose(forward_m: float | None, lateral_m: float | None) -> tuple[str, dict]:
    if forward_m is None or lateral_m is None:
        return "A", {"forward_m": None, "lateral_m": None, "angle_deg": None}
    angle = math.degrees(math.atan2(lateral_m, max(1e-6, forward_m)))
    if -22.5 <= angle < 22.5:
        c = "A"
    elif 22.5 <= angle < 67.5:
        c = "B"
    elif 67.5 <= angle < 112.5:
        c = "D"
    elif 112.5 <= angle < 157.5:
        c = "F"
    elif angle >= 157.5 or angle < -157.5:
        c = "H"
    elif -157.5 <= angle < -112.5:
        c = "G"
    elif -112.5 <= angle < -67.5:
        c = "E"
    else:
        c = "C"
    return c, {"forward_m": forward_m, "lateral_m": lateral_m, "angle_deg": angle}


def sp3_from_distance(distance_m: float | None) -> tuple[str, dict]:
    if distance_m is None:
        return "E", {"distance_m": None}
    if distance_m < 5.0:
        return "A", {"distance_m": distance_m}
    if distance_m < 15.0:
        return "B", {"distance_m": distance_m}
    if distance_m < 30.0:
        return "C", {"distance_m": distance_m}
    if distance_m < 50.0:
        return "D", {"distance_m": distance_m}
    return "E", {"distance_m": distance_m}


def sp4_from_lateral(lateral_m: float | None) -> tuple[str, dict]:
    if lateral_m is None:
        return "A", {"lateral_m": None}
    if abs(lateral_m) < LANE_HALF_WIDTH_M:
        return "A", {"lateral_m": lateral_m}
    if 0 < lateral_m < LANE_HALF_WIDTH_M + LANE_WIDTH_M:
        return "B", {"lateral_m": lateral_m}
    if -(LANE_HALF_WIDTH_M + LANE_WIDTH_M) < lateral_m < 0:
        return "C", {"lateral_m": lateral_m}
    if lateral_m >= LANE_HALF_WIDTH_M + LANE_WIDTH_M:
        return "D", {"lateral_m": lateral_m}
    return "E", {"lateral_m": lateral_m}


def sp5_from_bbox(obj: Obj, width: int, height: int) -> tuple[str, dict]:
    x1, y1, x2, y2 = obj.bbox
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    touch = int(x1 <= 1) + int(y1 <= 1) + int(x2 >= width - 1) + int(y2 >= height - 1)
    if touch >= 2 or area < 1400:
        c = "C"
    elif touch == 1:
        c = "B"
    else:
        c = "A"
    return c, {"bbox_area": area, "touch_edges": touch}


def sp6_from_type(obj: Obj) -> tuple[str, dict]:
    if obj.coarse == "pedestrian" or obj.coarse == "bicycle":
        if obj.lateral_m is not None and abs(obj.lateral_m) > 4.0:
            return "B", {"reason": "lateral_offset_large"}
        return "A", {"reason": "vru_near_road"}
    if "sign" in obj.label.lower():
        return "C", {"reason": "sign_above_ground"}
    if "barrier" in obj.label.lower() or "cone" in obj.label.lower():
        return "B", {"reason": "roadside_obstacle"}
    return "A", {"reason": "vehicle_on_road"}


def infer_sp1_choice_waymo(
    selected: Obj,
    prev_match: Obj | None,
    frame_dt_s: float,
) -> tuple[str, dict[str, float]]:
    if prev_match is None:
        return "E", {"speed_rel": 0.0, "range_rate": 0.0, "vx_rel": 0.0, "vy_rel": 0.0, "obj_speed": 0.0, "heading_alignment": 0.0}

    if selected.forward_m is None or selected.lateral_m is None:
        return "E", {"speed_rel": 0.0, "range_rate": 0.0, "vx_rel": 0.0, "vy_rel": 0.0, "obj_speed": 0.0, "heading_alignment": 0.0}
    if prev_match.forward_m is None or prev_match.lateral_m is None:
        return "E", {"speed_rel": 0.0, "range_rate": 0.0, "vx_rel": 0.0, "vy_rel": 0.0, "obj_speed": 0.0, "heading_alignment": 0.0}

    dt = max(frame_dt_s, 1e-6)
    vx_rel = (selected.forward_m - prev_match.forward_m) / dt
    vy_rel = (selected.lateral_m - prev_match.lateral_m) / dt
    speed_rel = math.hypot(vx_rel, vy_rel)

    if selected.distance_m is not None and prev_match.distance_m is not None:
        range_rate = (selected.distance_m - prev_match.distance_m) / dt
    else:
        range_rate = 0.0

    # No true world-frame heading in current Waymo artifacts: keep placeholders.
    obj_speed = speed_rel
    align = 0.0

    static_thr = 0.5
    lateral_thr = 0.7
    range_thr = 0.4

    if speed_rel < static_thr:
        choice = "E"
    elif abs(vy_rel) > abs(vx_rel) * 1.2 and abs(vy_rel) > lateral_thr:
        choice = "C" if vy_rel < 0 else "D"
    elif abs(vy_rel) < 0.6 and abs(vx_rel) < 1.2 and align > 0.8 and obj_speed > 1.0:
        choice = "F"
    elif range_rate < -range_thr:
        choice = "A"
    elif range_rate > range_thr:
        choice = "B"
    else:
        choice = "A" if vx_rel < 0 else "B"

    return choice, {
        "speed_rel": speed_rel,
        "range_rate": range_rate,
        "vx_rel": vx_rel,
        "vy_rel": vy_rel,
        "obj_speed": obj_speed,
        "heading_alignment": align,
    }


def infer_su1_choice_waymo(
    sp1_choice: str,
    sp2_choice: str,
    sp3_choice: str,
    sp4_choice: str,
    sp1_metrics: dict[str, float],
) -> tuple[str, dict[str, Any]]:
    close_range = sp3_choice in {"A", "B"}
    mid_range = sp3_choice == "C"
    same_lane = sp4_choice == "A"
    adjacent_lane = sp4_choice in {"B", "C"}
    crossing_motion = sp1_choice in {"C", "D"}
    toward_motion = sp1_choice == "A"
    dynamic_motion = sp1_choice in {"A", "C", "D"}
    static_motion = sp1_choice == "E"
    front_sector = sp2_choice in {"A", "B", "C"}
    speed_rel = float(sp1_metrics.get("speed_rel", 0.0))

    if dynamic_motion and front_sector and (close_range or mid_range):
        choice, reason = "D", "dynamic_front_intrusion"
    elif same_lane and close_range and (static_motion or speed_rel < 1.0):
        choice, reason = "A", "same_lane_close_block"
    elif (same_lane and mid_range) or (adjacent_lane and close_range) or (crossing_motion and mid_range):
        choice, reason = "B", "partial_interference"
    else:
        choice, reason = "C", "no_significant_constraint"

    return choice, {
        "reason": reason,
        "same_lane": same_lane,
        "adjacent_lane": adjacent_lane,
        "front_sector": front_sector,
        "close_range": close_range,
        "mid_range": mid_range,
        "crossing_motion": crossing_motion,
        "toward_motion": toward_motion,
        "speed_rel": speed_rel,
    }


def infer_su2_choice_waymo(
    sp1_choice: str,
    sp2_choice: str,
    sp3_choice: str,
    sp4_choice: str,
    su1_choice: str,
) -> tuple[str, dict[str, Any]]:
    close_or_mid = sp3_choice in {"A", "B", "C"}
    front_sector = sp2_choice in {"A", "B", "C"}
    left_sector = sp2_choice in {"B", "D", "F"}
    right_sector = sp2_choice in {"C", "E", "G"}
    same_lane = sp4_choice == "A"
    dynamic = sp1_choice in {"A", "C", "D"}

    if su1_choice == "A" and same_lane and close_or_mid and front_sector:
        choice, reason = "F", "front_blocking_same_lane"
    elif su1_choice == "D" and dynamic and front_sector:
        if sp1_choice == "C":
            choice, reason = "C", "dynamic_cross_left_to_right"
        elif sp1_choice == "D":
            choice, reason = "B", "dynamic_cross_right_to_left"
        else:
            choice, reason = "A", "dynamic_toward_front"
    elif su1_choice == "B" and close_or_mid:
        if left_sector and not right_sector:
            choice, reason = "C", "partial_left_obstruction"
        elif right_sector and not left_sector:
            choice, reason = "B", "partial_right_obstruction"
        else:
            choice, reason = "A", "partial_front_obstruction"
    elif su1_choice == "C":
        if close_or_mid and same_lane and front_sector:
            choice, reason = "A", "near_same_lane_still_front_limited"
        elif close_or_mid and (left_sector or right_sector):
            if left_sector and not right_sector:
                choice, reason = "C", "near_left_object_front_right_preferred"
            elif right_sector and not left_sector:
                choice, reason = "B", "near_right_object_front_left_preferred"
            else:
                choice, reason = "A", "near_lateral_object_front_only"
        elif close_or_mid and front_sector:
            choice, reason = "A", "near_front_object_front_only"
        else:
            choice, reason = "G", "far_or_low_impact_all_directions"
    else:
        choice, reason = "A", "default_front_drivable"

    return choice, {
        "reason": reason,
        "same_lane": same_lane,
        "front_sector": front_sector,
        "left_sector": left_sector,
        "right_sector": right_sector,
        "dynamic": dynamic,
        "close_or_mid": close_or_mid,
    }


def infer_su3_choice_waymo(
    sp1_choice: str,
    sp2_choice: str,
    sp3_choice: str,
    su1_choice: str,
    sp1_metrics: dict[str, float],
) -> tuple[str, dict[str, Any]]:
    close_range = sp3_choice in {"A", "B"}
    very_close = sp3_choice == "A"
    front_sector = sp2_choice in {"A", "B", "C"}
    crossing_or_toward = sp1_choice in {"A", "C", "D"}
    static_block = su1_choice == "A"
    dynamic_obstruction = su1_choice == "D"
    partial_obstruction = su1_choice == "B"
    speed_rel = float(sp1_metrics.get("speed_rel", 0.0))

    if very_close and front_sector and (dynamic_obstruction or crossing_or_toward):
        choice, reason = "D", "imminent_dynamic_conflict"
    elif close_range and front_sector and (static_block or dynamic_obstruction):
        choice, reason = "C", "near_front_strong_constraint"
    elif (close_range and partial_obstruction) or (front_sector and speed_rel > 2.0):
        choice, reason = "B", "moderate_interaction_risk"
    else:
        choice, reason = "A", "limited_immediate_risk"

    return choice, {
        "reason": reason,
        "close_range": close_range,
        "very_close": very_close,
        "front_sector": front_sector,
        "crossing_or_toward": crossing_or_toward,
        "static_block": static_block,
        "dynamic_obstruction": dynamic_obstruction,
        "partial_obstruction": partial_obstruction,
        "speed_rel": speed_rel,
    }


def infer_su4_choice_waymo(
    sp1_choice: str,
    sp2_choice: str,
    sp3_choice: str,
    sp4_choice: str,
    su1_choice: str,
    su3_choice: str,
) -> tuple[str, dict[str, Any]]:
    close_or_mid = sp3_choice in {"A", "B", "C"}
    front_sector = sp2_choice in {"A", "B", "C"}
    left_sector = sp2_choice in {"B", "D", "F"}
    right_sector = sp2_choice in {"C", "E", "G"}
    critical_risk = su3_choice == "D"

    left_threat = close_or_mid and (
        sp4_choice in {"B", "D"} or left_sector or (sp1_choice == "D" and front_sector)
    )
    right_threat = close_or_mid and (
        sp4_choice in {"C", "E"} or right_sector or (sp1_choice == "C" and front_sector)
    )
    front_same_lane_threat = close_or_mid and front_sector and sp4_choice == "A" and su1_choice in {"A", "D"}

    if critical_risk and front_same_lane_threat:
        choice, reason = "C", "critical_front_conflict"
    else:
        safe_left = not left_threat and not critical_risk
        safe_right = not right_threat and not critical_risk
        if safe_left and safe_right:
            choice, reason = "D", "both_sides_clear_from_object"
        elif safe_left:
            choice, reason = "A", "left_clear_right_constrained"
        elif safe_right:
            choice, reason = "B", "right_clear_left_constrained"
        else:
            choice, reason = "C", "both_sides_constrained"

    return choice, {
        "reason": reason,
        "close_or_mid": close_or_mid,
        "front_sector": front_sector,
        "left_sector": left_sector,
        "right_sector": right_sector,
        "critical_risk": critical_risk,
        "left_threat": left_threat,
        "right_threat": right_threat,
        "front_same_lane_threat": front_same_lane_threat,
    }


def infer_su5_choice_waymo(
    sp1_choice: str,
    sp2_choice: str,
    sp3_choice: str,
    su1_choice: str,
    su3_choice: str,
    su4_choice: str,
) -> tuple[str, dict[str, Any]]:
    close_front = sp3_choice in {"A", "B"} and sp2_choice in {"A", "B", "C"}
    critical = su3_choice == "D"
    high_risk = su3_choice in {"C", "D"}
    dynamic_obstruction = su1_choice == "D"

    if critical and close_front:
        choice, reason = "F", "critical_close_front"
    elif su4_choice == "A":
        choice, reason = "B", "left_lane_change_available"
    elif su4_choice == "B":
        choice, reason = "C", "right_lane_change_available"
    elif su4_choice == "D" and not high_risk:
        choice, reason = "A", "maintain_lane_safe"
    elif dynamic_obstruction and sp1_choice in {"C", "D"} and sp3_choice in {"B", "C"}:
        choice, reason = "A", "wait_crossing_then_maintain"
    elif high_risk:
        choice, reason = "F", "high_risk_stop"
    else:
        choice, reason = "A", "default_maintain"

    return choice, {
        "reason": reason,
        "critical": critical,
        "high_risk": high_risk,
        "dynamic_obstruction": dynamic_obstruction,
        "close_front": close_front,
        "lane_change_state": su4_choice,
    }


def infer_su6_choice_waymo(vehicle_objects: list[Obj]) -> tuple[str, dict[str, float]]:
    distances = [o.distance_m for o in vehicle_objects if o.distance_m is not None]
    vehicle_count = len(distances)
    near_count = sum(1 for d in distances if d < 15.0)
    mid_count = sum(1 for d in distances if 15.0 <= d < 30.0)

    weighted_density_score = 0.0
    for d in distances:
        if d < 10.0:
            weighted_density_score += 2.5
        elif d < 20.0:
            weighted_density_score += 1.5
        elif d < 35.0:
            weighted_density_score += 0.8
        else:
            weighted_density_score += 0.2

    if weighted_density_score >= 18.0 or near_count >= 8:
        choice = "D"
    elif weighted_density_score >= 11.0 or near_count >= 5:
        choice = "C"
    elif weighted_density_score >= 5.0 or vehicle_count >= 4:
        choice = "B"
    else:
        choice = "A"

    return choice, {
        "vehicle_count": float(vehicle_count),
        "near_count_15m": float(near_count),
        "mid_count_30m": float(mid_count),
        "weighted_density_score": weighted_density_score,
    }


def infer_te1_choice_waymo(
    sp1_metrics: dict[str, float],
) -> tuple[str, dict[str, Any]]:
    speed = float(sp1_metrics.get("speed_rel", 0.0))
    vx_ego = float(sp1_metrics.get("vx_rel", 0.0))
    vy_ego = float(sp1_metrics.get("vy_rel", 0.0))

    if speed < 0.4:
        choice, reason = "D", "low_motion_speed"
    elif abs(vy_ego) > abs(vx_ego) * 1.1 and abs(vy_ego) > 0.8:
        choice, reason = "E", "lateral_crossing_dominant"
    elif abs(vy_ego) <= 0.6:
        choice, reason = "A", "forward_straight_dominant"
    elif vy_ego > 0:
        choice, reason = "B", "leftward_curving_motion"
    else:
        choice, reason = "C", "rightward_curving_motion"

    return choice, {
        "used_future": False,
        "future_steps_used": 0.0,
        "history_steps_used": 1.0,
        "speed": speed,
        "vx_ego": vx_ego,
        "vy_ego": vy_ego,
        "reason": reason,
    }


def infer_te2_choice_waymo(
    distance_now: float | None,
    sp1_metrics: dict[str, float],
) -> tuple[str, dict[str, Any]]:
    if distance_now is None:
        return "A", {"ttc_sec": -1.0, "closing_speed": 0.0, "distance_now": 0.0, "used_fallback": True}
    closing_speed = -float(sp1_metrics.get("range_rate", 0.0))
    if closing_speed <= 0.05:
        return "A", {"ttc_sec": -1.0, "closing_speed": closing_speed, "distance_now": distance_now, "used_fallback": False}

    ttc = distance_now / closing_speed
    if ttc > 10.0:
        choice = "B"
    elif ttc > 5.0:
        choice = "C"
    elif ttc > 2.0:
        choice = "D"
    else:
        choice = "E"
    return choice, {"ttc_sec": ttc, "closing_speed": closing_speed, "distance_now": distance_now, "used_fallback": False}


def infer_te3_choice_waymo(
    sp1_metrics: dict[str, float],
    sp4_choice: str,
    sp6_choice: str,
) -> tuple[str, dict[str, Any]]:
    vy_ego = float(sp1_metrics.get("vy_rel", 0.0))
    if sp6_choice in {"B", "C"}:
        return "E", {"used_future": False, "vy_ego": vy_ego, "reason": "already_off_road"}

    lateral_thr = 0.6
    if abs(vy_ego) <= lateral_thr:
        return "A", {"used_future": False, "vy_ego": vy_ego, "reason": "low_lateral_motion"}

    moving_left = vy_ego > lateral_thr
    moving_right = vy_ego < -lateral_thr

    if sp4_choice == "A":
        choice, reason = ("B", "same_lane_lateral_shift") if moving_left else ("C", "same_lane_lateral_shift")
    elif sp4_choice == "B":
        choice, reason = ("D", "left_adjacent_toward_ego_lane") if moving_right else ("B", "left_adjacent_away_from_ego_lane")
    elif sp4_choice == "C":
        choice, reason = ("D", "right_adjacent_toward_ego_lane") if moving_left else ("C", "right_adjacent_away_from_ego_lane")
    elif sp4_choice == "D":
        choice, reason = ("D", "two_left_transition") if moving_right else ("B", "two_left_transition")
    elif sp4_choice == "E":
        choice, reason = ("D", "two_right_transition") if moving_left else ("C", "two_right_transition")
    else:
        choice, reason = "A", "fallback_remain"

    return choice, {"used_future": False, "vy_ego": vy_ego, "reason": reason}


def infer_te4_choice_waymo(
    te1_choice: str,
    te2_choice: str,
    su3_choice: str,
    su4_choice: str,
    su5_choice: str,
    te2_metrics: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    ttc = float(te2_metrics.get("ttc_sec", -1.0))
    critical = su3_choice == "D" or te2_choice == "E" or (0.0 < ttc < 2.0)
    high = su3_choice == "C" or te2_choice == "D" or (2.0 <= ttc < 5.0)
    moderate = su3_choice == "B" or te2_choice == "C" or (5.0 <= ttc < 10.0)
    crossing = te1_choice == "E"

    if critical:
        if su4_choice in {"A", "B"}:
            choice, reason = "D", "critical_but_lane_change_available"
        else:
            choice, reason = "F", "critical_no_clearance"
    elif high:
        if crossing:
            choice, reason = "E", "high_risk_crossing_yield"
        elif su4_choice in {"A", "B"} and su5_choice in {"B", "C"}:
            choice, reason = "D", "high_risk_lane_change_option"
        else:
            choice, reason = "C", "high_risk_prepare_stop"
    elif moderate:
        if crossing:
            choice, reason = "E", "moderate_crossing_yield"
        else:
            choice, reason = "B", "moderate_risk_slow_down"
    else:
        if su5_choice in {"B", "C"} and su4_choice in {"A", "B"}:
            choice, reason = "D", "proactive_lane_change_feasible"
        else:
            choice, reason = "A", "low_risk_maintain"

    return choice, {
        "reason": reason,
        "ttc_sec": ttc,
        "critical": critical,
        "high": high,
        "moderate": moderate,
        "crossing": crossing,
        "su4_choice": su4_choice,
        "su5_choice": su5_choice,
    }


def infer_te5_choice_waymo(
    te1_choice: str,
    te2_choice: str,
    te3_choice: str,
    su3_choice: str,
    su6_choice: str,
) -> tuple[str, dict[str, Any]]:
    critical_or_high = su3_choice in {"C", "D"} or te2_choice in {"D", "E"}
    lane_transition = te3_choice in {"B", "C", "D"}
    crossing_motion = te1_choice == "E"
    stopping_motion = te1_choice == "D"
    sparse_scene = su6_choice == "A"

    if lane_transition and te3_choice == "D":
        choice, reason = "D", "object_entering_ego_lane_merge"
    elif lane_transition:
        choice, reason = "A", "lane_transition_detected"
    elif stopping_motion and critical_or_high:
        choice, reason = "B", "high_risk_object_stopping"
    elif crossing_motion and critical_or_high:
        choice, reason = "D", "crossing_vehicle_merge_like"
    elif stopping_motion:
        choice, reason = "B", "object_likely_stopping"
    elif sparse_scene and not crossing_motion:
        choice, reason = "E", "low_density_no_major_event"
    else:
        choice, reason = "E", "no_dominant_event"

    return choice, {
        "reason": reason,
        "critical_or_high": critical_or_high,
        "lane_transition": lane_transition,
        "crossing_motion": crossing_motion,
        "stopping_motion": stopping_motion,
        "sparse_scene": sparse_scene,
    }


def tm2_choice_from_sp2(sp2_choice: str) -> str:
    if sp2_choice == "A":
        return "A"
    if sp2_choice == "B":
        return "B"
    if sp2_choice == "C":
        return "C"
    if sp2_choice == "D":
        return "D"
    if sp2_choice == "E":
        return "E"
    return "F"


def choose_selected_object(objects: list[Obj]) -> Obj:
    candidates = [o for o in objects if o.distance_m is not None]
    if candidates:
        return min(candidates, key=lambda o: float(o.distance_m))
    return max(objects, key=lambda o: o.score)


def match_object(current: Obj, previous_objects: list[Obj]) -> Obj | None:
    best: Obj | None = None
    best_iou = 0.0
    for p in previous_objects:
        if p.coarse != current.coarse:
            continue
        score = iou_xyxy(current.bbox, p.bbox)
        if score > best_iou:
            best_iou = score
            best = p
    if best_iou < 0.15:
        return None
    return best


def build_task_template_map(questions: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for task in questions.get("tasks", []):
        tid = str(task.get("id", "")).strip()
        if tid:
            out[tid] = task
    return out


def add_row(
    results: dict[str, list[dict[str, Any]]],
    task_map: dict[str, dict],
    task_id: str,
    scene_id: str,
    frame_idx: int,
    object_id: str,
    object_ref: str,
    ground_truth: str,
    model_response_text: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    task = task_map[task_id]
    question = str(task.get("question", ""))
    if "<obj>" in question:
        question = question.replace("<obj>", object_ref)
    row = {
        "scene_id": scene_id,
        "group_id": f"group_{frame_idx:03d}",
        "source_group_file": f"{scene_id}/group_{frame_idx:03d}_vehicle_annotations.json",
        "question_id": task_id,
        "object_id": object_id,
        "object_reference": object_ref,
        "question": question,
        "choices": task.get("choices", {}),
        "ground_truth": ground_truth,
        "model_response": "",
        "model_response_text": model_response_text,
        "metrics": metrics,
    }
    results[task_id].append(row)
    return row


def answer_text(task_map: dict[str, dict], task_id: str, key: str) -> str:
    choices = task_map[task_id].get("choices", {})
    if isinstance(choices, dict):
        return str(choices.get(key, key))
    return key


def build_frq_prompt(
    frq_question: str,
    object_reference: str,
    mcq_rows: list[dict],
    group_scene_id: str = "",
    vehicle_annotation_count: int | None = None,
) -> str:
    context_lines: list[str] = []
    for row in mcq_rows:
        question = row.get("question", "")
        answer_key = row.get("ground_truth", "")
        answer_text_value = row.get("model_response_text", "")
        context_lines.append(f"- {row.get('question_id', '')}: {question}")
        context_lines.append(f"  GT answer: {answer_key} ({answer_text_value})")
    context_text = "\n".join(context_lines)
    filled_question = frq_question.replace("<obj>", object_reference)

    scene_context_lines: list[str] = []
    if group_scene_id:
        scene_context_lines.append(f"- Group scene_id: {group_scene_id}")
    if vehicle_annotation_count is not None:
        scene_context_lines.append(
            f"- Vehicle annotations in query frame group: {vehicle_annotation_count}"
        )
    scene_context_text = (
        "\n".join(scene_context_lines) if scene_context_lines else "- Scene context: unavailable"
    )
    return (
        "You are an autonomous-driving QA assistant.\n"
        "Use the provided MCQ ground-truth answers as primary evidence.\n"
        "Use scene metadata (if provided) as supplementary context only.\n"
        "Write a concise free-response answer (2-4 sentences) with clear reasoning.\n"
        "Do not mention choice letters and do not invent facts beyond the given context.\n\n"
        f"Target object reference: {object_reference}\n"
        f"Scene context:\n{scene_context_text}\n\n"
        f"FRQ: {filled_question}\n\n"
        "MCQ ground-truth context:\n"
        f"{context_text}\n\n"
        "Answer:"
    )


def generate_frq_response(
    llm: Any,
    sampling_params: Any,
    prompt: str,
) -> str:
    outputs = llm.generate(prompt, sampling_params=sampling_params)
    if not outputs:
        return ""
    text = outputs[0].outputs[0].text
    return text.strip() if text else ""


def validate_mcq_rows_for_frq(mcq_rows: list[dict], label: str) -> None:
    for row in mcq_rows:
        answer_key = str(row.get("ground_truth", "")).strip()
        if not answer_key:
            raise ValueError(
                f"Missing MCQ ground_truth for {label} context: {row.get('question_id', 'unknown')}"
            )
        choices = row.get("choices", {})
        if isinstance(choices, dict) and choices and answer_key not in choices:
            raise ValueError(
                f"Invalid MCQ ground_truth '{answer_key}' for {label} context: "
                f"{row.get('question_id', 'unknown')}"
            )


def main() -> None:
    args = parse_args()
    distance_payload = load_json(args.distance_json)
    questions = load_json(args.questions_json)
    task_map = build_task_template_map(questions)

    required = [
        "SP-1",
        "SP-2",
        "SP-3",
        "SP-4",
        "SP-5",
        "SP-6",
        "SP-7",
        "SU-1",
        "SU-2",
        "SU-3",
        "SU-4",
        "SU-5",
        "SU-6",
        "SU-7",
        "TE-1",
        "TE-2",
        "TE-3",
        "TE-4",
        "TE-5",
        "TE-6",
        "TM-1",
        "TM-2",
        "TM-3",
        "TM-4",
        "TM-5",
        "TM-6",
    ]
    missing = [x for x in required if x not in task_map]
    if missing:
        raise ValueError(f"Missing tasks in questions JSON: {missing}")

    raw_results = distance_payload.get("results", [])
    frames: list[dict] = []
    for rec in raw_results:
        frame_name = str(rec.get("frame_name", ""))
        scene_id, frame_idx = parse_frame_name(frame_name)
        width = int(rec.get("image_size", {}).get("width", 0))
        height = int(rec.get("image_size", {}).get("height", 0))
        objs: list[Obj] = []
        for i, ann in enumerate(rec.get("distance_estimates", [])):
            details = ann.get("details", {}) if isinstance(ann.get("details"), dict) else {}
            uv = ann.get("sample_pixel_uv", [0.0, 0.0])
            bbox = ann.get("bbox_xyxy", [0.0, 0.0, 0.0, 0.0])
            objs.append(
                Obj(
                    obj_id=f"{frame_name}#obj{i:02d}",
                    label=str(ann.get("label", "")),
                    coarse=coarse_class(str(ann.get("label", ""))),
                    score=float(ann.get("score", 0.0) or 0.0),
                    bbox=[float(v) for v in bbox],
                    u=float(uv[0] if len(uv) > 0 else 0.0),
                    v=float(uv[1] if len(uv) > 1 else 0.0),
                    distance_m=float(ann["distance_m"]) if ann.get("distance_m") is not None else None,
                    forward_m=(
                        float(details["forward_m"])
                        if isinstance(details, dict) and details.get("forward_m") is not None
                        else None
                    ),
                    lateral_m=(
                        float(details["lateral_m"])
                        if isinstance(details, dict) and details.get("lateral_m") is not None
                        else None
                    ),
                )
            )
        objs = dedupe_objects(objs)
        frames.append(
            {
                "frame_name": frame_name,
                "scene_id": scene_id,
                "frame_idx": frame_idx,
                "width": width,
                "height": height,
                "objects": objs,
            }
        )

    frames.sort(key=lambda x: (x["scene_id"], x["frame_idx"]))
    assign_track_ids(frames)
    results: dict[str, list[dict[str, Any]]] = {k: [] for k in required}

    # Per scene rolling state for time-memory style questions.
    prev_by_scene: dict[str, dict] = {}
    prev2_by_scene: dict[str, dict] = {}
    frq_enabled = not args.disable_frq_generation
    frq_llm = None
    frq_sampling_params = None
    if frq_enabled:
        from vllm import LLM, SamplingParams

        frq_llm = LLM(
            model=args.frq_model,
            tensor_parallel_size=args.frq_tensor_parallel_size,
            max_model_len=8192,
            gpu_memory_utilization=0.85,
        )
        frq_sampling_params = SamplingParams(
            temperature=args.frq_temperature,
            top_p=0.9,
            top_k=20,
            max_tokens=args.frq_max_tokens,
            repetition_penalty=1.0,
            presence_penalty=0.0,
        )

    for frame in frames:
        scene_id = frame["scene_id"]
        frame_idx = frame["frame_idx"]
        objects: list[Obj] = frame["objects"]
        if not objects:
            continue
        selected = choose_selected_object(objects)
        object_ref = build_object_ref(selected)
        prev = prev_by_scene.get(scene_id)
        prev2 = prev2_by_scene.get(scene_id)

        prev_match = None
        prev2_match = None
        if prev and selected.track_id:
            prev_match = next((o for o in prev["objects"] if o.track_id == selected.track_id), None)
        if prev2 and selected.track_id:
            prev2_match = next((o for o in prev2["objects"] if o.track_id == selected.track_id), None)

        # SP-1: match nuscenes thresholds/ordering.
        sp1, sp1_metrics = infer_sp1_choice_waymo(selected, prev_match, FRAME_DT_S)
        add_row(
            results,
            task_map,
            "SP-1",
            scene_id,
            frame_idx,
            selected.obj_id,
            object_ref,
            sp1,
            answer_text(task_map, "SP-1", sp1),
            {k: round(v, 4) for k, v in sp1_metrics.items()},
        )

        # SP-2/SP-3/SP-4/SP-5/SP-6
        sp2, sp2_metrics = sp2_from_pose(selected.forward_m, selected.lateral_m)
        sp3, sp3_metrics = sp3_from_distance(selected.distance_m)
        sp4, sp4_metrics = sp4_from_lateral(selected.lateral_m)
        sp5, sp5_metrics = sp5_from_bbox(selected, frame["width"], frame["height"])
        sp6, sp6_metrics = sp6_from_type(selected)
        for tid, val, met in [
            ("SP-2", sp2, sp2_metrics),
            ("SP-3", sp3, sp3_metrics),
            ("SP-4", sp4, sp4_metrics),
            ("SP-5", sp5, sp5_metrics),
            ("SP-6", sp6, sp6_metrics),
        ]:
            add_row(
                results,
                task_map,
                tid,
                scene_id,
                frame_idx,
                selected.obj_id,
                object_ref,
                val,
                answer_text(task_map, tid, val),
                met,
            )

        su1, su1_metrics = infer_su1_choice_waymo(
            sp1_choice=sp1,
            sp2_choice=sp2,
            sp3_choice=sp3,
            sp4_choice=sp4,
            sp1_metrics=sp1_metrics,
        )
        add_row(
            results,
            task_map,
            "SU-1",
            scene_id,
            frame_idx,
            selected.obj_id,
            object_ref,
            su1,
            answer_text(task_map, "SU-1", su1),
            su1_metrics,
        )

        su2, su2_metrics = infer_su2_choice_waymo(
            sp1_choice=sp1,
            sp2_choice=sp2,
            sp3_choice=sp3,
            sp4_choice=sp4,
            su1_choice=su1,
        )
        add_row(
            results,
            task_map,
            "SU-2",
            scene_id,
            frame_idx,
            selected.obj_id,
            object_ref,
            su2,
            answer_text(task_map, "SU-2", su2),
            su2_metrics,
        )

        su3, su3_metrics = infer_su3_choice_waymo(
            sp1_choice=sp1,
            sp2_choice=sp2,
            sp3_choice=sp3,
            su1_choice=su1,
            sp1_metrics=sp1_metrics,
        )
        add_row(
            results,
            task_map,
            "SU-3",
            scene_id,
            frame_idx,
            selected.obj_id,
            object_ref,
            su3,
            answer_text(task_map, "SU-3", su3),
            su3_metrics,
        )

        su4, su4_metrics = infer_su4_choice_waymo(
            sp1_choice=sp1,
            sp2_choice=sp2,
            sp3_choice=sp3,
            sp4_choice=sp4,
            su1_choice=su1,
            su3_choice=su3,
        )
        add_row(
            results,
            task_map,
            "SU-4",
            scene_id,
            frame_idx,
            selected.obj_id,
            object_ref,
            su4,
            answer_text(task_map, "SU-4", su4),
            su4_metrics,
        )

        su5, su5_metrics = infer_su5_choice_waymo(
            sp1_choice=sp1,
            sp2_choice=sp2,
            sp3_choice=sp3,
            su1_choice=su1,
            su3_choice=su3,
            su4_choice=su4,
        )
        add_row(
            results,
            task_map,
            "SU-5",
            scene_id,
            frame_idx,
            selected.obj_id,
            object_ref,
            su5,
            answer_text(task_map, "SU-5", su5),
            su5_metrics,
        )

        vehicle_objects = [o for o in objects if o.coarse == "vehicle"]
        su6, su6_metrics = infer_su6_choice_waymo(vehicle_objects)
        add_row(
            results,
            task_map,
            "SU-6",
            scene_id,
            frame_idx,
            selected.obj_id,
            object_ref,
            su6,
            answer_text(task_map, "SU-6", su6),
            su6_metrics,
        )

        te1, te1_metrics = infer_te1_choice_waymo(sp1_metrics)
        add_row(
            results,
            task_map,
            "TE-1",
            scene_id,
            frame_idx,
            selected.obj_id,
            object_ref,
            te1,
            answer_text(task_map, "TE-1", te1),
            te1_metrics,
        )

        te2, te2_metrics = infer_te2_choice_waymo(
            distance_now=selected.distance_m,
            sp1_metrics=sp1_metrics,
        )
        add_row(
            results,
            task_map,
            "TE-2",
            scene_id,
            frame_idx,
            selected.obj_id,
            object_ref,
            te2,
            answer_text(task_map, "TE-2", te2),
            {
                "ttc_sec": (
                    round(float(te2_metrics["ttc_sec"]), 4)
                    if isinstance(te2_metrics.get("ttc_sec"), (int, float))
                    else te2_metrics.get("ttc_sec")
                ),
                "closing_speed": (
                    round(float(te2_metrics["closing_speed"]), 4)
                    if isinstance(te2_metrics.get("closing_speed"), (int, float))
                    else te2_metrics.get("closing_speed")
                ),
                "distance_now": (
                    round(float(te2_metrics["distance_now"]), 4)
                    if isinstance(te2_metrics.get("distance_now"), (int, float))
                    else te2_metrics.get("distance_now")
                ),
                "used_fallback": te2_metrics.get("used_fallback"),
            },
        )

        te3, te3_metrics = infer_te3_choice_waymo(
            sp1_metrics=sp1_metrics,
            sp4_choice=sp4,
            sp6_choice=sp6,
        )
        add_row(
            results,
            task_map,
            "TE-3",
            scene_id,
            frame_idx,
            selected.obj_id,
            object_ref,
            te3,
            answer_text(task_map, "TE-3", te3),
            te3_metrics,
        )

        te4, te4_metrics = infer_te4_choice_waymo(
            te1_choice=te1,
            te2_choice=te2,
            su3_choice=su3,
            su4_choice=su4,
            su5_choice=su5,
            te2_metrics=te2_metrics,
        )
        add_row(
            results,
            task_map,
            "TE-4",
            scene_id,
            frame_idx,
            selected.obj_id,
            object_ref,
            te4,
            answer_text(task_map, "TE-4", te4),
            te4_metrics,
        )

        te5, te5_metrics = infer_te5_choice_waymo(
            te1_choice=te1,
            te2_choice=te2,
            te3_choice=te3,
            su3_choice=su3,
            su6_choice=su6,
        )
        add_row(
            results,
            task_map,
            "TE-5",
            scene_id,
            frame_idx,
            selected.obj_id,
            object_ref,
            te5,
            answer_text(task_map, "TE-5", te5),
            te5_metrics,
        )

        # TM-1 / TM-4: multi-frame memory over last 4 frames.
        history_frames = 4
        history: list[dict] = []
        cur_hist = prev
        while cur_hist is not None and len(history) < history_frames:
            history.append(cur_hist)
            cur_hist = prev2_by_scene.get(scene_id) if cur_hist is prev else None
            if cur_hist is None and len(history) < history_frames:
                # fallback: reconstruct from latest saved pointer chain not available
                break

        # Reconstruct richer history directly from sorted frames by index.
        same_scene_frames = [f for f in frames if f["scene_id"] == scene_id]
        same_scene_frames.sort(key=lambda x: x["frame_idx"])
        pos = next((i for i, f in enumerate(same_scene_frames) if f["frame_idx"] == frame_idx), -1)
        history = []
        if pos >= 0:
            start = max(0, pos - history_frames)
            history = same_scene_frames[start:pos]

        prev_tracks: set[str] = set()
        current_tracks: set[str] = {o.track_id for o in objects if o.track_id}
        track_to_class: dict[str, str] = {}
        for hf in history:
            for o in hf["objects"]:
                if o.track_id:
                    prev_tracks.add(o.track_id)
                    track_to_class[o.track_id] = o.coarse

        disappeared_tracks = prev_tracks - current_tracks
        disappeared_classes = [track_to_class[t] for t in disappeared_tracks if t in track_to_class]
        disappeared_count = len(disappeared_tracks)

        if not disappeared_classes:
            tm1 = "A"
        elif len(set(disappeared_classes)) > 1:
            tm1 = "F"
        else:
            c = disappeared_classes[0]
            tm1 = {"pedestrian": "B", "vehicle": "C", "bicycle": "D"}.get(c, "E")
        add_row(
            results,
            task_map,
            "TM-1",
            scene_id,
            frame_idx,
            selected.obj_id,
            object_ref,
            tm1,
            answer_text(task_map, "TM-1", tm1),
            {
                "history_frames_used": float(len(history)),
                "prev_instance_count": float(len(prev_tracks)),
                "current_instance_count": float(len(current_tracks)),
                "disappeared_instance_count": float(disappeared_count),
                "disappeared_types": ",".join(sorted(set(disappeared_classes))) if disappeared_classes else "none",
            },
        )

        # TM-4 stable instances present through history + now with occlusion proxy.
        window = history + [frame]
        if window:
            stable_tracks = set.intersection(
                *[
                    {o.track_id for o in wf["objects"] if o.track_id}
                    for wf in window
                ]
            ) if all(wf["objects"] for wf in window) else set()
        else:
            stable_tracks = set()

        previously_visible: set[str] = set()
        for trk in stable_tracks:
            for wf in history:
                obj = next((o for o in wf["objects"] if o.track_id == trk), None)
                if obj is None:
                    continue
                sp5_choice, _ = sp5_from_bbox(obj, wf["width"], wf["height"])
                if sp5_choice in {"A", "B"}:
                    previously_visible.add(trk)
                    break
        occluded_now = 0
        for trk in previously_visible:
            obj_now = next((o for o in frame["objects"] if o.track_id == trk), None)
            if obj_now is None:
                continue
            sp5_now, _ = sp5_from_bbox(obj_now, frame["width"], frame["height"])
            if sp5_now in {"C", "D"}:
                occluded_now += 1

        if occluded_now == 0:
            tm4 = "A"
        elif occluded_now == 1:
            tm4 = "B"
        elif occluded_now <= 3:
            tm4 = "C"
        else:
            tm4 = "D"
        add_row(
            results,
            task_map,
            "TM-4",
            scene_id,
            frame_idx,
            selected.obj_id,
            object_ref,
            tm4,
            answer_text(task_map, "TM-4", tm4),
            {
                "history_frames_used": float(len(history)),
                "stable_instance_count": float(len(stable_tracks)),
                "previously_visible_instances": float(len(previously_visible)),
                "occluded_now_instances": float(occluded_now),
            },
        )

        # TM-2
        if prev_match is not None:
            tm2_sp2, tm2_metrics = sp2_from_pose(prev_match.forward_m, prev_match.lateral_m)
            tm2 = tm2_choice_from_sp2(tm2_sp2)
            tm2_metrics = {
                "used_fallback": False,
                "x_ego": prev_match.forward_m,
                "y_ego": prev_match.lateral_m,
                "angle_deg": tm2_metrics.get("angle_deg"),
            }
        else:
            tm2 = "A"
            tm2_metrics = {"used_fallback": True, "x_ego": 0.0, "y_ego": 0.0, "angle_deg": 0.0}
        add_row(
            results,
            task_map,
            "TM-2",
            scene_id,
            frame_idx,
            selected.obj_id,
            object_ref,
            tm2,
            answer_text(task_map, "TM-2", tm2),
            tm2_metrics,
        )

        # TM-3
        prev_range_rate = 0.0
        if prev_match is not None and prev2_match is not None:
            if prev_match.distance_m is not None and prev2_match.distance_m is not None:
                prev_range_rate = (prev_match.distance_m - prev2_match.distance_m) / FRAME_DT_S
        speed1 = float(sp1_metrics.get("speed_rel", 0.0))
        speed0 = max(0.0, speed1 - float(sp1_metrics.get("range_rate", 0.0)) * 0.1)
        vx1 = float(sp1_metrics.get("vx_rel", 0.0))
        vy1 = float(sp1_metrics.get("vy_rel", 0.0))
        if speed1 < 0.35:
            tm3, reason = "F", "stopped_recently"
        elif abs(vy1) > abs(vx1) * 1.1 and abs(vy1) > 0.6:
            tm3, reason = ("D", "lateral_turning_pattern") if vy1 > 0 else ("E", "lateral_turning_pattern")
        elif speed1 - speed0 > 0.5:
            tm3, reason = "C", "accelerating_trend"
        elif speed0 - speed1 > 0.5:
            tm3, reason = "B", "slowing_trend"
        else:
            tm3, reason = "A", "straight_motion_trend"
        add_row(
            results,
            task_map,
            "TM-3",
            scene_id,
            frame_idx,
            selected.obj_id,
            object_ref,
            tm3,
            answer_text(task_map, "TM-3", tm3),
            {
                "reason": reason,
                "speed0": round(speed0, 4),
                "speed1": round(speed1, 4),
                "vx_ego": round(vx1, 4),
                "vy_ego": round(vy1, 4),
                "history_steps_used": 1.0,
                "prev_range_rate_mps": round(prev_range_rate, 4),
            },
        )

        # TM-5
        if prev_match is not None:
            tm5_sp4, tm5_metrics = sp4_from_lateral(prev_match.lateral_m)
            tm5 = {"A": "A", "B": "B", "C": "C", "D": "D", "E": "D"}.get(tm5_sp4, "A")
        else:
            tm5 = "A"
            tm5_metrics = {"reason": "no_prev_match"}
        add_row(
            results,
            task_map,
            "TM-5",
            scene_id,
            frame_idx,
            selected.obj_id,
            object_ref,
            tm5,
            answer_text(task_map, "TM-5", tm5),
            tm5_metrics,
        )

        if frq_enabled and frq_llm is not None and frq_sampling_params is not None:
            sp_mcq_rows = [results["SP-1"][-1], results["SP-2"][-1], results["SP-3"][-1], results["SP-4"][-1], results["SP-5"][-1], results["SP-6"][-1]]
            su_mcq_rows = [results["SU-1"][-1], results["SU-2"][-1], results["SU-3"][-1], results["SU-4"][-1], results["SU-5"][-1], results["SU-6"][-1]]
            te_mcq_rows = [results["TE-1"][-1], results["TE-2"][-1], results["TE-3"][-1], results["TE-4"][-1], results["TE-5"][-1]]
            tm_mcq_rows = [results["TM-1"][-1], results["TM-2"][-1], results["TM-3"][-1], results["TM-4"][-1], results["TM-5"][-1]]

            validate_mcq_rows_for_frq(sp_mcq_rows, "SP-7")
            validate_mcq_rows_for_frq(su_mcq_rows, "SU-7")
            validate_mcq_rows_for_frq(te_mcq_rows, "TE-6")
            validate_mcq_rows_for_frq(tm_mcq_rows, "TM-6")

            vehicle_annotation_count = len(vehicle_objects)

            sp7_prompt = build_frq_prompt(
                task_map["SP-7"]["question"],
                object_ref,
                sp_mcq_rows,
                group_scene_id=scene_id,
                vehicle_annotation_count=vehicle_annotation_count,
            )
            sp7_answer = generate_frq_response(frq_llm, frq_sampling_params, sp7_prompt)
            results["SP-7"].append(
                {
                    "scene_id": scene_id,
                    "group_id": f"group_{frame_idx:03d}",
                    "source_group_file": f"{scene_id}/group_{frame_idx:03d}_vehicle_annotations.json",
                    "question_id": "SP-7",
                    "object_id": selected.obj_id,
                    "object_reference": object_ref,
                    "question": task_map["SP-7"]["question"].replace("<obj>", object_ref),
                    "ground_truth": sp7_answer,
                    "model_response": "",
                }
            )

            su7_prompt = build_frq_prompt(
                task_map["SU-7"]["question"],
                object_ref,
                su_mcq_rows,
                group_scene_id=scene_id,
                vehicle_annotation_count=vehicle_annotation_count,
            )
            su7_answer = generate_frq_response(frq_llm, frq_sampling_params, su7_prompt)
            results["SU-7"].append(
                {
                    "scene_id": scene_id,
                    "group_id": f"group_{frame_idx:03d}",
                    "source_group_file": f"{scene_id}/group_{frame_idx:03d}_vehicle_annotations.json",
                    "question_id": "SU-7",
                    "object_id": selected.obj_id,
                    "object_reference": object_ref,
                    "question": task_map["SU-7"]["question"].replace("<obj>", object_ref),
                    "ground_truth": su7_answer,
                    "model_response": "",
                }
            )

            te6_prompt = build_frq_prompt(
                task_map["TE-6"]["question"],
                object_ref,
                te_mcq_rows,
                group_scene_id=scene_id,
                vehicle_annotation_count=vehicle_annotation_count,
            )
            te6_answer = generate_frq_response(frq_llm, frq_sampling_params, te6_prompt)
            results["TE-6"].append(
                {
                    "scene_id": scene_id,
                    "group_id": f"group_{frame_idx:03d}",
                    "source_group_file": f"{scene_id}/group_{frame_idx:03d}_vehicle_annotations.json",
                    "question_id": "TE-6",
                    "object_id": selected.obj_id,
                    "object_reference": object_ref,
                    "question": task_map["TE-6"]["question"].replace("<obj>", object_ref),
                    "ground_truth": te6_answer,
                    "model_response": "",
                }
            )

            tm6_prompt = build_frq_prompt(
                task_map["TM-6"]["question"],
                object_ref,
                tm_mcq_rows,
                group_scene_id=scene_id,
                vehicle_annotation_count=vehicle_annotation_count,
            )
            tm6_answer = generate_frq_response(frq_llm, frq_sampling_params, tm6_prompt)
            results["TM-6"].append(
                {
                    "scene_id": scene_id,
                    "group_id": f"group_{frame_idx:03d}",
                    "source_group_file": f"{scene_id}/group_{frame_idx:03d}_vehicle_annotations.json",
                    "question_id": "TM-6",
                    "object_id": selected.obj_id,
                    "object_reference": object_ref,
                    "question": task_map["TM-6"]["question"].replace("<obj>", object_ref),
                    "ground_truth": tm6_answer,
                    "model_response": "",
                }
            )

        prev2_by_scene[scene_id] = prev_by_scene.get(scene_id, frame)
        prev_by_scene[scene_id] = frame

    output_tasks: dict[str, dict] = {}
    for task_id in required:
        output_tasks[task_id] = {"count": len(results[task_id]), "results": results[task_id]}

    output = {
        "source": {
            "distance_json": str(args.distance_json),
            "questions_json": str(args.questions_json),
            "dataset": "waymo_test2",
            "method": "heuristic_gt_from_bbox_distance",
        },
        "tasks": output_tasks,
    }
    write_json(args.output_json, output)
    questions_with_answers = copy.deepcopy(questions)
    generated_answers: dict[str, dict[str, Any]] = {}
    for task_id in required:
        template = task_map[task_id]
        generated_tasks: list[dict[str, Any]] = []
        for row in results[task_id]:
            task = copy.deepcopy(template)
            if "object_id" in row:
                task["object_id"] = row["object_id"]
            if "object_reference" in row:
                task["object_reference"] = row["object_reference"]
            task["ground_truth"] = row.get("ground_truth", "")
            task["model_response"] = ""
            task["scene_id"] = row.get("scene_id", "")
            task["group_id"] = row.get("group_id", "")
            task["source_group_file"] = row.get("source_group_file", "")
            generated_tasks.append(task)
        generated_answers[task_id] = {"count": len(generated_tasks), "tasks": generated_tasks}
    questions_with_answers["generated_answers"] = generated_answers
    write_json(args.questions_output, questions_with_answers)
    print(f"Saved Waymo generated answers to: {args.output_json}")
    print(f"Saved questions copy to: {args.questions_output}")


if __name__ == "__main__":
    main()
