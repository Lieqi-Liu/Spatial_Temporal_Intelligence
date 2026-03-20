#!/usr/bin/env python3
"""Generate answers for question templates from nuScenes annotations.

Current implementation covers:
  - SP-1: trajectory direction of selected object relative to ego vehicle.
  - SP-2: spatial relationship of selected object relative to ego vehicle.
  - SP-3: approximate distance between selected object and ego vehicle.
  - SP-4: lane position of selected object relative to ego lane.
  - SP-5: object occlusion state.
  - SP-6: whether object is on road/sidewalk/above ground.
  - SU-1: physical constraints imposed on ego by selected object.
  - SU-2: safely drivable region around ego given selected object.
  - SU-3: risk level posed by selected object.
  - SU-4: lane change clearance around ego.
  - SU-5: geometrically feasible maneuver.
  - SU-6: spatial density around ego from vehicle count and distances.
  - TE-1: likely future motion of selected object.
  - TM-1: objects that disappeared from recent frames.
  - TM-2: previous-frame relative location of selected object.
  - TM-3: recent motion pattern of selected object.
  - TM-4: currently occluded objects that were previously visible.
  - TM-5: lane position of selected object in earlier frame(s).

Input:
  - Group metadata produced in formatted_scenes/scene_*/group_*_vehicle_annotations.json
  - nuScenes metadata tables in v1.0-mini

Output:
  - formatted_scenes/generated_answers_all.json
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

from nuscenes.map_expansion.map_api import NuScenesMap

DEFAULT_FRQ_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"


SP1_CHOICES = {
    "A": "Moving toward ego",
    "B": "Moving away from ego",
    "C": "Moving left-to-right across ego",
    "D": "Moving right-to-left across ego",
    "E": "Static",
    "F": "Same direction as ego",
}

SP2_CHOICES = {
    "A": "In front",
    "B": "Front-left",
    "C": "Front-right",
    "D": "Directly left",
    "E": "Directly right",
    "F": "Rear-left",
    "G": "Rear-right",
    "H": "Behind",
}

SP3_CHOICES = {
    "A": "0-5m",
    "B": "5-15m",
    "C": "15-30m",
    "D": "30-50m",
    "E": ">50m",
}

SP4_CHOICES = {
    "A": "Same lane as ego",
    "B": "Left adjacent lane",
    "C": "Right adjacent lane",
    "D": "Two lanes left",
    "E": "Two lanes right",
}

SP5_CHOICES = {
    "A": "Fully visible",
    "B": "Partially occluded (<50% occluded)",
    "C": "Mostly occluded (>=50% occluded)",
    "D": "Fully occluded but inferred",
}

SP6_CHOICES = {
    "A": "On road",
    "B": "On sidewalk",
    "C": "Above the ground",
}

SU1_CHOICES = {
    "A": "Blocking lane",
    "B": "Partially obstructing path",
    "C": "No constraint",
    "D": "Dynamic obstruction",
}

SU2_CHOICES = {
    "A": "Front only",
    "B": "Front-left",
    "C": "Front-right",
    "D": "Left only",
    "E": "Right only",
    "F": "None",
    "G": "All directions",
}

SU3_CHOICES = {
    "A": "Low risk",
    "B": "Moderate risk",
    "C": "High risk",
    "D": "Critical / imminent collision",
}

SU4_CHOICES = {
    "A": "Safe to change left",
    "B": "Safe to change right",
    "C": "Unsafe both sides",
    "D": "Safe both sides",
}

SU5_CHOICES = {
    "A": "Maintain lane",
    "B": "Lane change left",
    "C": "Lane change right",
    "D": "Turn left",
    "E": "Turn right",
    "F": "Stop only",
}

SU6_CHOICES = {
    "A": "Sparse",
    "B": "Moderate",
    "C": "Dense",
    "D": "Highly congested",
}

TE1_CHOICES = {
    "A": "Continue straight",
    "B": "Turn left",
    "C": "Turn right",
    "D": "Stop",
    "E": "Cross ego path",
}

TE2_CHOICES = {
    "A": "No collision predicted",
    "B": ">10 seconds",
    "C": "5-10 seconds",
    "D": "2-5 seconds",
    "E": "<2 seconds",
}

TE3_CHOICES = {
    "A": "Remain in current lane",
    "B": "Move to left lane",
    "C": "Move to right lane",
    "D": "Enter ego lane",
    "E": "Exit roadway",
}

TE4_CHOICES = {
    "A": "Maintain speed",
    "B": "Slow down",
    "C": "Prepare to stop",
    "D": "Change lane",
    "E": "Yield",
    "F": "Emergency braking",
}

TE5_CHOICES = {
    "A": "Lane change",
    "B": "Vehicle stopping",
    "C": "Pedestrian crossing",
    "D": "Vehicle merging",
    "E": "No significant event",
}

TM1_CHOICES = {
    "A": "No object disappeared",
    "B": "Pedestrian",
    "C": "Vehicle",
    "D": "Bicycle",
    "E": "Static obstacle",
    "F": "Multiple objects",
}

TM2_CHOICES = {
    "A": "In front",
    "B": "Front-left",
    "C": "Front-right",
    "D": "Left",
    "E": "Right",
    "F": "Behind",
}

TM3_CHOICES = {
    "A": "Moving straight",
    "B": "Slowing down",
    "C": "Accelerating",
    "D": "Turning left",
    "E": "Turning right",
    "F": "Stopped previously",
}

TM4_CHOICES = {
    "A": "None",
    "B": "1",
    "C": "2-3",
    "D": "More than 3",
}

TM5_CHOICES = {
    "A": "Same lane",
    "B": "Left adjacent lane",
    "C": "Right adjacent lane",
    "D": "Outside roadway",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate multi-task driving QA answers from annotations."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="nuScenes root directory (default: script directory).",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1.0-mini",
        help="nuScenes version folder name (default: v1.0-mini).",
    )
    parser.add_argument(
        "--formatted-scenes-dir",
        type=Path,
        default=None,
        help="Path to formatted_scenes (default: <root>/formatted_scenes).",
    )
    parser.add_argument(
        "--questions-json",
        type=Path,
        default=None,
        help="Path to question templates JSON (default: <root>/questions.json).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output file path (default: <formatted_scenes>/generated_answers_all.json).",
    )
    parser.add_argument(
        "--questions-output",
        type=Path,
        default=None,
        help=(
            "Write a new questions file with generated answers "
            "(default: <root>/questions_with_answers_all.json)."
        ),
    )
    parser.add_argument(
        "--track-window",
        type=int,
        default=2,
        help="How many prev/next annotations to use around current frame (default: 2).",
    )
    parser.add_argument(
        "--lane-width-m",
        type=float,
        default=3.7,
        help="Approximate lane width used for SP-4 lateral fallback (default: 3.7m).",
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


def load_json(path: Path) -> List[dict] | dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def yaw_from_quaternion_wxyz(q: List[float]) -> float:
    """Extract yaw from quaternion [w, x, y, z]."""
    if len(q) != 4:
        return 0.0
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def global_to_ego_xy(dx: float, dy: float, ego_yaw: float) -> Tuple[float, float]:
    """Rotate a global XY vector into ego frame XY."""
    c = math.cos(ego_yaw)
    s = math.sin(ego_yaw)
    x_ego = c * dx + s * dy
    y_ego = -s * dx + c * dy
    return x_ego, y_ego


def gather_track_tokens(center_token: str, ann_by_token: Dict[str, dict], window: int) -> List[str]:
    prev_tokens: List[str] = []
    cur = center_token
    for _ in range(window):
        prev_token = ann_by_token[cur].get("prev")
        if not prev_token:
            break
        prev_tokens.append(prev_token)
        cur = prev_token
    prev_tokens.reverse()

    next_tokens: List[str] = []
    cur = center_token
    for _ in range(window):
        next_token = ann_by_token[cur].get("next")
        if not next_token:
            break
        next_tokens.append(next_token)
        cur = next_token

    return prev_tokens + [center_token] + next_tokens


def infer_sp1_choice(
    selected_ann_token: str,
    ann_by_token: Dict[str, dict],
    sample_by_token: Dict[str, dict],
    ego_pose_by_token: Dict[str, dict],
    cam_front_sd_by_sample: Dict[str, dict],
    window: int,
) -> Tuple[str, Dict[str, float]]:
    track_tokens = gather_track_tokens(selected_ann_token, ann_by_token, window=window)

    points = []
    for token in track_tokens:
        ann = ann_by_token[token]
        sample_token = ann["sample_token"]
        sd = cam_front_sd_by_sample.get(sample_token)
        if sd is None:
            continue
        ego_pose = ego_pose_by_token.get(sd["ego_pose_token"])
        if ego_pose is None:
            continue
        sample = sample_by_token[sample_token]
        t = float(sample["timestamp"]) * 1e-6
        obj_xyz = ann["translation"]
        ego_xyz = ego_pose["translation"]
        points.append((t, obj_xyz, ego_xyz, ann["token"]))

    if len(points) < 2:
        return "E", {"speed_rel": 0.0, "range_rate": 0.0, "vx_rel": 0.0, "vy_rel": 0.0}

    points.sort(key=lambda x: x[0])
    t0, obj0, ego0, _ = points[0]
    t1, obj1, ego1, _ = points[-1]
    dt = max(t1 - t0, 1e-6)

    center_ann = ann_by_token[selected_ann_token]
    center_sample_token = center_ann["sample_token"]
    center_sd = cam_front_sd_by_sample[center_sample_token]
    center_ego_pose = ego_pose_by_token[center_sd["ego_pose_token"]]
    ego_yaw = yaw_from_quaternion_wxyz(center_ego_pose["rotation"])

    rel0_gx = obj0[0] - ego0[0]
    rel0_gy = obj0[1] - ego0[1]
    rel1_gx = obj1[0] - ego1[0]
    rel1_gy = obj1[1] - ego1[1]
    rel0_ex, rel0_ey = global_to_ego_xy(rel0_gx, rel0_gy, ego_yaw)
    rel1_ex, rel1_ey = global_to_ego_xy(rel1_gx, rel1_gy, ego_yaw)

    vx_rel = (rel1_ex - rel0_ex) / dt
    vy_rel = (rel1_ey - rel0_ey) / dt
    speed_rel = math.hypot(vx_rel, vy_rel)

    range0 = math.hypot(rel0_gx, rel0_gy)
    range1 = math.hypot(rel1_gx, rel1_gy)
    range_rate = (range1 - range0) / dt

    obj_vx = (obj1[0] - obj0[0]) / dt
    obj_vy = (obj1[1] - obj0[1]) / dt
    obj_speed = math.hypot(obj_vx, obj_vy)
    ego_fwd_x = math.cos(ego_yaw)
    ego_fwd_y = math.sin(ego_yaw)
    align = 0.0
    if obj_speed > 1e-6:
        align = (obj_vx * ego_fwd_x + obj_vy * ego_fwd_y) / obj_speed

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

    metrics = {
        "speed_rel": speed_rel,
        "range_rate": range_rate,
        "vx_rel": vx_rel,
        "vy_rel": vy_rel,
        "obj_speed": obj_speed,
        "heading_alignment": align,
    }
    return choice, metrics


def infer_sp2_choice(
    selected_ann_token: str,
    ann_by_token: Dict[str, dict],
    ego_pose_by_token: Dict[str, dict],
    cam_front_sd_by_sample: Dict[str, dict],
) -> Tuple[str, Dict[str, float]]:
    ann = ann_by_token[selected_ann_token]
    sample_token = ann["sample_token"]
    sd = cam_front_sd_by_sample.get(sample_token)
    if sd is None:
        return "A", {"distance_xy": 0.0, "x_ego": 0.0, "y_ego": 0.0, "angle_deg": 0.0}
    ego_pose = ego_pose_by_token.get(sd["ego_pose_token"])
    if ego_pose is None:
        return "A", {"distance_xy": 0.0, "x_ego": 0.0, "y_ego": 0.0, "angle_deg": 0.0}

    ego_yaw = yaw_from_quaternion_wxyz(ego_pose["rotation"])
    dx = ann["translation"][0] - ego_pose["translation"][0]
    dy = ann["translation"][1] - ego_pose["translation"][1]
    x_ego, y_ego = global_to_ego_xy(dx, dy, ego_yaw)
    distance_xy = math.hypot(x_ego, y_ego)
    angle_deg = math.degrees(math.atan2(y_ego, x_ego))

    if -22.5 <= angle_deg < 22.5:
        choice = "A"
    elif 22.5 <= angle_deg < 67.5:
        choice = "B"
    elif 67.5 <= angle_deg < 112.5:
        choice = "D"
    elif 112.5 <= angle_deg < 157.5:
        choice = "F"
    elif angle_deg >= 157.5 or angle_deg < -157.5:
        choice = "H"
    elif -157.5 <= angle_deg < -112.5:
        choice = "G"
    elif -112.5 <= angle_deg < -67.5:
        choice = "E"
    else:
        choice = "C"

    metrics = {
        "distance_xy": distance_xy,
        "x_ego": x_ego,
        "y_ego": y_ego,
        "angle_deg": angle_deg,
    }
    return choice, metrics


def infer_sp3_choice(
    selected_ann_token: str,
    ann_by_token: Dict[str, dict],
    ego_pose_by_token: Dict[str, dict],
    cam_front_sd_by_sample: Dict[str, dict],
) -> Tuple[str, Dict[str, float]]:
    ann = ann_by_token[selected_ann_token]
    sample_token = ann["sample_token"]
    sd = cam_front_sd_by_sample.get(sample_token)
    if sd is None:
        return "E", {"distance_xy": 0.0, "distance_3d": 0.0}
    ego_pose = ego_pose_by_token.get(sd["ego_pose_token"])
    if ego_pose is None:
        return "E", {"distance_xy": 0.0, "distance_3d": 0.0}

    dx = ann["translation"][0] - ego_pose["translation"][0]
    dy = ann["translation"][1] - ego_pose["translation"][1]
    dz = ann["translation"][2] - ego_pose["translation"][2]
    distance_xy = math.hypot(dx, dy)
    distance_3d = math.sqrt(dx * dx + dy * dy + dz * dz)

    # Use horizontal distance for traffic-centric distance bins.
    if distance_xy < 5.0:
        choice = "A"
    elif distance_xy < 15.0:
        choice = "B"
    elif distance_xy < 30.0:
        choice = "C"
    elif distance_xy < 50.0:
        choice = "D"
    else:
        choice = "E"

    return choice, {"distance_xy": distance_xy, "distance_3d": distance_3d}


def infer_sp4_choice(
    selected_ann_token: str,
    ann_by_token: Dict[str, dict],
    ego_pose_by_token: Dict[str, dict],
    cam_front_sd_by_sample: Dict[str, dict],
    sample_by_token: Dict[str, dict],
    scene_by_token: Dict[str, dict],
    log_by_token: Dict[str, dict],
    map_cache: Dict[str, NuScenesMap],
    root: Path,
    lane_width_m: float,
) -> Tuple[str, Dict[str, float | str | None]]:
    ann = ann_by_token[selected_ann_token]
    sample_token = ann["sample_token"]
    sd = cam_front_sd_by_sample.get(sample_token)
    if sd is None:
        return "A", {"method": "fallback_no_sample_data", "lateral_offset_m": 0.0}
    ego_pose = ego_pose_by_token.get(sd["ego_pose_token"])
    if ego_pose is None:
        return "A", {"method": "fallback_no_ego_pose", "lateral_offset_m": 0.0}

    ego_yaw = yaw_from_quaternion_wxyz(ego_pose["rotation"])
    dx = ann["translation"][0] - ego_pose["translation"][0]
    dy = ann["translation"][1] - ego_pose["translation"][1]
    _, y_ego = global_to_ego_xy(dx, dy, ego_yaw)

    sample_row = sample_by_token.get(sample_token, {})
    scene_token = sample_row.get("scene_token")
    scene_row = scene_by_token.get(scene_token, {})
    log_token = scene_row.get("log_token")
    location = log_by_token.get(log_token, {}).get("location")

    ego_lane_token = ""
    obj_lane_token = ""
    if location:
        try:
            if location not in map_cache:
                map_cache[location] = NuScenesMap(dataroot=str(root), map_name=location)
            nusc_map = map_cache[location]
            ego_lane_token = nusc_map.get_closest_lane(
                ego_pose["translation"][0], ego_pose["translation"][1], radius=5
            )
            obj_lane_token = nusc_map.get_closest_lane(
                ann["translation"][0], ann["translation"][1], radius=5
            )
            if ego_lane_token and obj_lane_token and ego_lane_token == obj_lane_token:
                return "A", {
                    "method": "map_same_lane",
                    "location": location,
                    "ego_lane_token": ego_lane_token,
                    "obj_lane_token": obj_lane_token,
                    "lateral_offset_m": y_ego,
                }
        except FileNotFoundError:
            pass

    # Fallback: infer lane difference from ego-frame lateral offset.
    lane_steps = int(round(y_ego / max(lane_width_m, 1e-3)))
    if lane_steps >= 2:
        choice = "D"
    elif lane_steps == 1:
        choice = "B"
    elif lane_steps == 0:
        choice = "A"
    elif lane_steps == -1:
        choice = "C"
    else:
        choice = "E"

    return choice, {
        "method": "lateral_offset_fallback",
        "location": location,
        "ego_lane_token": ego_lane_token or None,
        "obj_lane_token": obj_lane_token or None,
        "lateral_offset_m": y_ego,
        "lane_steps": lane_steps,
        "lane_width_m": lane_width_m,
    }


def infer_sp5_choice(selected_ann_token: str, ann_by_token: Dict[str, dict]) -> Tuple[str, Dict[str, float | str]]:
    ann = ann_by_token[selected_ann_token]
    visibility_token = str(ann.get("visibility_token", ""))
    num_lidar_pts = int(ann.get("num_lidar_pts", 0) or 0)
    num_radar_pts = int(ann.get("num_radar_pts", 0) or 0)
    point_evidence = num_lidar_pts + num_radar_pts

    visibility_ranges = {
        "1": (0.0, 0.4, "v0-40"),
        "2": (0.4, 0.6, "v40-60"),
        "3": (0.6, 0.8, "v60-80"),
        "4": (0.8, 1.0, "v80-100"),
    }
    lo, hi, level = visibility_ranges.get(visibility_token, (0.0, 1.0, "unknown"))
    visible_mid = (lo + hi) / 2.0
    occluded_ratio = 1.0 - visible_mid

    # "Fully occluded but inferred" is approximated by almost no visible support.
    if visibility_token == "1" and point_evidence == 0:
        choice = "D"
    elif occluded_ratio < 0.1:
        choice = "A"
    elif occluded_ratio < 0.5:
        choice = "B"
    else:
        choice = "C"

    return choice, {
        "visibility_token": visibility_token,
        "visibility_level": level,
        "visible_ratio_mid": visible_mid,
        "occluded_ratio_mid": occluded_ratio,
        "num_lidar_pts": float(num_lidar_pts),
        "num_radar_pts": float(num_radar_pts),
        "point_evidence_total": float(point_evidence),
    }


def infer_sp6_choice(
    selected_ann_token: str,
    ann_by_token: Dict[str, dict],
    ego_pose_by_token: Dict[str, dict],
    cam_front_sd_by_sample: Dict[str, dict],
    sample_by_token: Dict[str, dict],
    scene_by_token: Dict[str, dict],
    log_by_token: Dict[str, dict],
    map_cache: Dict[str, NuScenesMap],
    root: Path,
) -> Tuple[str, Dict[str, float | str | None]]:
    ann = ann_by_token[selected_ann_token]
    sample_token = ann["sample_token"]
    sd = cam_front_sd_by_sample.get(sample_token)
    if sd is None:
        return "A", {"method": "fallback_no_sample_data", "z_rel": 0.0}
    ego_pose = ego_pose_by_token.get(sd["ego_pose_token"])
    if ego_pose is None:
        return "A", {"method": "fallback_no_ego_pose", "z_rel": 0.0}

    z_rel = ann["translation"][2] - ego_pose["translation"][2]
    # If the object center is significantly above ego ground plane, classify as above ground.
    if z_rel > 2.5:
        return "C", {"method": "height_threshold", "z_rel": z_rel}

    sample_row = sample_by_token.get(sample_token, {})
    scene_token = sample_row.get("scene_token")
    scene_row = scene_by_token.get(scene_token, {})
    log_token = scene_row.get("log_token")
    location = log_by_token.get(log_token, {}).get("location")

    if location:
        try:
            if location not in map_cache:
                map_cache[location] = NuScenesMap(dataroot=str(root), map_name=location)
            nusc_map = map_cache[location]
            layers = nusc_map.layers_on_point(
                ann["translation"][0],
                ann["translation"][1],
                layer_names=["drivable_area", "road_segment", "lane", "walkway", "sidewalk"],
            )
            on_sidewalk = bool(layers.get("walkway")) or bool(layers.get("sidewalk"))
            on_road = bool(layers.get("drivable_area")) or bool(layers.get("road_segment")) or bool(
                layers.get("lane")
            )
            if on_sidewalk and not on_road:
                return "B", {"method": "map_layer", "location": location, "z_rel": z_rel}
            if on_road:
                return "A", {"method": "map_layer", "location": location, "z_rel": z_rel}
        except FileNotFoundError:
            pass

    # Fallback without map files: vehicles are typically on drivable ground.
    return "A", {"method": "fallback_default_vehicle", "z_rel": z_rel}


def infer_su1_choice(
    sp1_choice: str,
    sp2_choice: str,
    sp3_choice: str,
    sp4_choice: str,
    sp1_metrics: Dict[str, float],
) -> Tuple[str, Dict[str, float | str | bool]]:
    """Infer SU-1 from already computed SP signals."""
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

    # Priority 1: dynamic obstruction risks.
    if dynamic_motion and front_sector and (close_range or mid_range):
        choice = "D"
        reason = "dynamic_front_intrusion"
    # Priority 2: hard block in ego lane.
    elif same_lane and close_range and (static_motion or speed_rel < 1.0):
        choice = "A"
        reason = "same_lane_close_block"
    # Priority 3: partial obstruction / potential interference.
    elif (same_lane and mid_range) or (adjacent_lane and close_range) or (crossing_motion and mid_range):
        choice = "B"
        reason = "partial_interference"
    else:
        choice = "C"
        reason = "no_significant_constraint"

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


def infer_su2_choice(
    sp1_choice: str,
    sp2_choice: str,
    sp3_choice: str,
    sp4_choice: str,
    su1_choice: str,
) -> Tuple[str, Dict[str, str | bool]]:
    """Infer SU-2 from selected object impact on nearby drivable space."""
    close_or_mid = sp3_choice in {"A", "B", "C"}
    front_sector = sp2_choice in {"A", "B", "C"}
    left_sector = sp2_choice in {"B", "D", "F"}
    right_sector = sp2_choice in {"C", "E", "G"}
    same_lane = sp4_choice == "A"
    dynamic = sp1_choice in {"A", "C", "D"}

    if su1_choice == "A" and same_lane and close_or_mid and front_sector:
        choice = "F"  # Lane ahead blocked by selected object.
        reason = "front_blocking_same_lane"
    elif su1_choice == "D" and dynamic and front_sector:
        if sp1_choice == "C":
            choice = "C"  # Object crossing left->right, right side more constrained.
            reason = "dynamic_cross_left_to_right"
        elif sp1_choice == "D":
            choice = "B"  # Object crossing right->left, left side more constrained.
            reason = "dynamic_cross_right_to_left"
        else:
            choice = "A"
            reason = "dynamic_toward_front"
    elif su1_choice == "B" and close_or_mid:
        if left_sector and not right_sector:
            choice = "C"
            reason = "partial_left_obstruction"
        elif right_sector and not left_sector:
            choice = "B"
            reason = "partial_right_obstruction"
        else:
            choice = "A"
            reason = "partial_front_obstruction"
    elif su1_choice == "C":
        # Avoid over-permissive "all directions": nearby objects still constrain space.
        if close_or_mid and same_lane and front_sector:
            choice = "A"
            reason = "near_same_lane_still_front_limited"
        elif close_or_mid and (left_sector or right_sector):
            # If object is nearby on one side, bias safe region to the opposite front side.
            if left_sector and not right_sector:
                choice = "C"
                reason = "near_left_object_front_right_preferred"
            elif right_sector and not left_sector:
                choice = "B"
                reason = "near_right_object_front_left_preferred"
            else:
                choice = "A"
                reason = "near_lateral_object_front_only"
        elif close_or_mid and front_sector:
            choice = "A"
            reason = "near_front_object_front_only"
        else:
            choice = "G"
            reason = "far_or_low_impact_all_directions"
    else:
        choice = "A"
        reason = "default_front_drivable"

    return choice, {
        "reason": reason,
        "same_lane": same_lane,
        "front_sector": front_sector,
        "left_sector": left_sector,
        "right_sector": right_sector,
        "dynamic": dynamic,
        "close_or_mid": close_or_mid,
    }


def infer_su3_choice(
    sp1_choice: str,
    sp2_choice: str,
    sp3_choice: str,
    su1_choice: str,
    sp1_metrics: Dict[str, float],
) -> Tuple[str, Dict[str, float | str | bool]]:
    """Infer SU-3 risk level from motion, position, distance, and constraints."""
    close_range = sp3_choice in {"A", "B"}
    very_close = sp3_choice == "A"
    front_sector = sp2_choice in {"A", "B", "C"}
    crossing_or_toward = sp1_choice in {"A", "C", "D"}
    static_block = su1_choice == "A"
    dynamic_obstruction = su1_choice == "D"
    partial_obstruction = su1_choice == "B"
    speed_rel = float(sp1_metrics.get("speed_rel", 0.0))

    # Most severe first.
    if very_close and front_sector and (dynamic_obstruction or crossing_or_toward):
        choice = "D"
        reason = "imminent_dynamic_conflict"
    elif close_range and front_sector and (static_block or dynamic_obstruction):
        choice = "C"
        reason = "near_front_strong_constraint"
    elif (close_range and partial_obstruction) or (front_sector and speed_rel > 2.0):
        choice = "B"
        reason = "moderate_interaction_risk"
    else:
        choice = "A"
        reason = "limited_immediate_risk"

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


def infer_su4_choice(
    sp1_choice: str,
    sp2_choice: str,
    sp3_choice: str,
    sp4_choice: str,
    su1_choice: str,
    su3_choice: str,
) -> Tuple[str, Dict[str, str | bool]]:
    """Infer whether ego has sufficient lane-change clearance."""
    close_or_mid = sp3_choice in {"A", "B", "C"}
    front_sector = sp2_choice in {"A", "B", "C"}
    left_sector = sp2_choice in {"B", "D", "F"}
    right_sector = sp2_choice in {"C", "E", "G"}
    critical_risk = su3_choice == "D"

    # Threat cues for each side from the selected object.
    left_threat = close_or_mid and (
        sp4_choice in {"B", "D"} or left_sector or (sp1_choice == "D" and front_sector)
    )
    right_threat = close_or_mid and (
        sp4_choice in {"C", "E"} or right_sector or (sp1_choice == "C" and front_sector)
    )
    front_same_lane_threat = close_or_mid and front_sector and sp4_choice == "A" and su1_choice in {"A", "D"}

    if critical_risk and front_same_lane_threat:
        choice = "C"
        reason = "critical_front_conflict"
    else:
        safe_left = not left_threat and not critical_risk
        safe_right = not right_threat and not critical_risk
        if safe_left and safe_right:
            choice = "D"
            reason = "both_sides_clear_from_object"
        elif safe_left:
            choice = "A"
            reason = "left_clear_right_constrained"
        elif safe_right:
            choice = "B"
            reason = "right_clear_left_constrained"
        else:
            choice = "C"
            reason = "both_sides_constrained"

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


def infer_su5_choice(
    sp1_choice: str,
    sp2_choice: str,
    sp3_choice: str,
    su1_choice: str,
    su3_choice: str,
    su4_choice: str,
) -> Tuple[str, Dict[str, str | bool]]:
    """Infer the most feasible immediate maneuver from object-centric constraints."""
    close_front = sp3_choice in {"A", "B"} and sp2_choice in {"A", "B", "C"}
    critical = su3_choice == "D"
    high_risk = su3_choice in {"C", "D"}
    dynamic_obstruction = su1_choice == "D"

    if critical and close_front:
        choice = "F"
        reason = "critical_close_front"
    elif su4_choice == "A":
        choice = "B"
        reason = "left_lane_change_available"
    elif su4_choice == "B":
        choice = "C"
        reason = "right_lane_change_available"
    elif su4_choice == "D" and not high_risk:
        choice = "A"
        reason = "maintain_lane_safe"
    elif dynamic_obstruction and sp1_choice in {"C", "D"} and sp3_choice in {"B", "C"}:
        choice = "A"
        reason = "wait_crossing_then_maintain"
    elif high_risk:
        choice = "F"
        reason = "high_risk_stop"
    else:
        choice = "A"
        reason = "default_maintain"

    return choice, {
        "reason": reason,
        "critical": critical,
        "high_risk": high_risk,
        "dynamic_obstruction": dynamic_obstruction,
        "close_front": close_front,
        "lane_change_state": su4_choice,
    }


def infer_su6_choice(
    vehicle_annotations: List[dict],
    ego_pose_by_token: Dict[str, dict],
    cam_front_sd_by_sample: Dict[str, dict],
    sample_token: str,
) -> Tuple[str, Dict[str, float]]:
    """Infer environment density from vehicle count and ego-relative distance."""
    sd = cam_front_sd_by_sample.get(sample_token)
    if sd is None:
        return "A", {
            "vehicle_count": 0.0,
            "near_count_15m": 0.0,
            "mid_count_30m": 0.0,
            "weighted_density_score": 0.0,
        }
    ego_pose = ego_pose_by_token.get(sd["ego_pose_token"])
    if ego_pose is None:
        return "A", {
            "vehicle_count": float(len(vehicle_annotations)),
            "near_count_15m": 0.0,
            "mid_count_30m": 0.0,
            "weighted_density_score": 0.0,
        }

    ex, ey = ego_pose["translation"][0], ego_pose["translation"][1]
    distances: List[float] = []
    for ann in vehicle_annotations:
        tx, ty = ann["translation"][0], ann["translation"][1]
        distances.append(math.hypot(tx - ex, ty - ey))

    vehicle_count = len(distances)
    near_count = sum(1 for d in distances if d < 15.0)
    mid_count = sum(1 for d in distances if 15.0 <= d < 30.0)

    # Weighted by proximity: closer objects contribute more congestion.
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


def infer_te1_choice(
    selected_ann_token: str,
    ann_by_token: Dict[str, dict],
    sample_by_token: Dict[str, dict],
    ego_pose_by_token: Dict[str, dict],
    cam_front_sd_by_sample: Dict[str, dict],
) -> Tuple[str, Dict[str, float | str | bool]]:
    """Infer future motion; prefer next annotations, fallback to previous trend."""
    ann = ann_by_token[selected_ann_token]

    next_tokens: List[str] = []
    cur = selected_ann_token
    for _ in range(3):
        nxt = ann_by_token[cur].get("next")
        if not nxt:
            break
        next_tokens.append(nxt)
        cur = nxt

    prev_tokens: List[str] = []
    cur = selected_ann_token
    for _ in range(3):
        prv = ann_by_token[cur].get("prev")
        if not prv:
            break
        prev_tokens.append(prv)
        cur = prv
    prev_tokens.reverse()

    use_future = len(next_tokens) > 0
    start_token = selected_ann_token if use_future else (prev_tokens[0] if prev_tokens else selected_ann_token)
    end_token = next_tokens[-1] if use_future else selected_ann_token

    start_ann = ann_by_token[start_token]
    end_ann = ann_by_token[end_token]

    start_sample = sample_by_token[start_ann["sample_token"]]
    end_sample = sample_by_token[end_ann["sample_token"]]
    t0 = float(start_sample["timestamp"]) * 1e-6
    t1 = float(end_sample["timestamp"]) * 1e-6
    dt = max(t1 - t0, 1e-6)

    current_sd = cam_front_sd_by_sample.get(ann["sample_token"])
    if current_sd is None:
        return "A", {"used_future": use_future, "speed": 0.0, "vx_ego": 0.0, "vy_ego": 0.0}
    current_ego = ego_pose_by_token.get(current_sd["ego_pose_token"])
    if current_ego is None:
        return "A", {"used_future": use_future, "speed": 0.0, "vx_ego": 0.0, "vy_ego": 0.0}

    ego_yaw = yaw_from_quaternion_wxyz(current_ego["rotation"])
    vx_world = (end_ann["translation"][0] - start_ann["translation"][0]) / dt
    vy_world = (end_ann["translation"][1] - start_ann["translation"][1]) / dt
    vx_ego, vy_ego = global_to_ego_xy(vx_world, vy_world, ego_yaw)
    speed = math.hypot(vx_ego, vy_ego)

    # If no future is available, use recent trend; still infer "future likely".
    if speed < 0.4:
        choice = "D"
        reason = "low_motion_speed"
    elif abs(vy_ego) > abs(vx_ego) * 1.1 and abs(vy_ego) > 0.8:
        choice = "E"
        reason = "lateral_crossing_dominant"
    elif abs(vy_ego) <= 0.6:
        choice = "A"
        reason = "forward_straight_dominant"
    elif vy_ego > 0:
        choice = "B"
        reason = "leftward_curving_motion"
    else:
        choice = "C"
        reason = "rightward_curving_motion"

    return choice, {
        "used_future": use_future,
        "future_steps_used": float(len(next_tokens)),
        "history_steps_used": float(len(prev_tokens)),
        "speed": speed,
        "vx_ego": vx_ego,
        "vy_ego": vy_ego,
        "reason": reason,
    }


def infer_te2_choice(
    selected_ann_token: str,
    ann_by_token: Dict[str, dict],
    sample_by_token: Dict[str, dict],
    ego_pose_by_token: Dict[str, dict],
    cam_front_sd_by_sample: Dict[str, dict],
) -> Tuple[str, Dict[str, float | str | bool]]:
    """Infer collision time assuming both maintain current motion."""
    ann = ann_by_token[selected_ann_token]
    sample_token = ann["sample_token"]
    current_sd = cam_front_sd_by_sample.get(sample_token)
    if current_sd is None:
        return "A", {"ttc_sec": -1.0, "closing_speed": 0.0, "distance_now": 0.0, "used_fallback": True}
    current_ego = ego_pose_by_token.get(current_sd["ego_pose_token"])
    if current_ego is None:
        return "A", {"ttc_sec": -1.0, "closing_speed": 0.0, "distance_now": 0.0, "used_fallback": True}

    prev_token = ann.get("prev")
    next_token = ann.get("next")
    used_fallback = False

    if prev_token:
        ref_start = ann_by_token[prev_token]
        ref_end = ann
    elif next_token:
        ref_start = ann
        ref_end = ann_by_token[next_token]
        used_fallback = True
    else:
        return "A", {"ttc_sec": -1.0, "closing_speed": 0.0, "distance_now": 0.0, "used_fallback": True}

    s0 = sample_by_token[ref_start["sample_token"]]
    s1 = sample_by_token[ref_end["sample_token"]]
    dt = max((float(s1["timestamp"]) - float(s0["timestamp"])) * 1e-6, 1e-6)

    v_obj_x = (ref_end["translation"][0] - ref_start["translation"][0]) / dt
    v_obj_y = (ref_end["translation"][1] - ref_start["translation"][1]) / dt

    sd0 = cam_front_sd_by_sample.get(ref_start["sample_token"])
    sd1 = cam_front_sd_by_sample.get(ref_end["sample_token"])
    if sd0 is None or sd1 is None:
        return "A", {"ttc_sec": -1.0, "closing_speed": 0.0, "distance_now": 0.0, "used_fallback": True}
    ego0 = ego_pose_by_token.get(sd0["ego_pose_token"])
    ego1 = ego_pose_by_token.get(sd1["ego_pose_token"])
    if ego0 is None or ego1 is None:
        return "A", {"ttc_sec": -1.0, "closing_speed": 0.0, "distance_now": 0.0, "used_fallback": True}

    v_ego_x = (ego1["translation"][0] - ego0["translation"][0]) / dt
    v_ego_y = (ego1["translation"][1] - ego0["translation"][1]) / dt

    rx = ann["translation"][0] - current_ego["translation"][0]
    ry = ann["translation"][1] - current_ego["translation"][1]
    distance_now = math.hypot(rx, ry)

    v_rel_x = v_obj_x - v_ego_x
    v_rel_y = v_obj_y - v_ego_y
    if distance_now < 1e-6:
        return "E", {"ttc_sec": 0.0, "closing_speed": 0.0, "distance_now": 0.0, "used_fallback": used_fallback}

    closing_speed = -((rx * v_rel_x + ry * v_rel_y) / max(distance_now, 1e-6))
    if closing_speed <= 0.05:
        return "A", {
            "ttc_sec": -1.0,
            "closing_speed": closing_speed,
            "distance_now": distance_now,
            "used_fallback": used_fallback,
        }

    ttc = distance_now / closing_speed
    if ttc > 10.0:
        choice = "B"
    elif ttc > 5.0:
        choice = "C"
    elif ttc > 2.0:
        choice = "D"
    else:
        choice = "E"

    return choice, {
        "ttc_sec": ttc,
        "closing_speed": closing_speed,
        "distance_now": distance_now,
        "used_fallback": used_fallback,
    }


def infer_te3_choice(
    selected_ann_token: str,
    ann_by_token: Dict[str, dict],
    sample_by_token: Dict[str, dict],
    ego_pose_by_token: Dict[str, dict],
    cam_front_sd_by_sample: Dict[str, dict],
    sp4_choice: str,
    sp6_choice: str,
) -> Tuple[str, Dict[str, float | str | bool]]:
    """Infer likely lane occupancy in next few seconds."""
    ann = ann_by_token[selected_ann_token]

    next_tokens: List[str] = []
    cur = selected_ann_token
    for _ in range(3):
        nxt = ann_by_token[cur].get("next")
        if not nxt:
            break
        next_tokens.append(nxt)
        cur = nxt

    prev_tokens: List[str] = []
    cur = selected_ann_token
    for _ in range(3):
        prv = ann_by_token[cur].get("prev")
        if not prv:
            break
        prev_tokens.append(prv)
        cur = prv
    prev_tokens.reverse()

    use_future = len(next_tokens) > 0
    start_token = selected_ann_token if use_future else (prev_tokens[0] if prev_tokens else selected_ann_token)
    end_token = next_tokens[-1] if use_future else selected_ann_token
    start_ann = ann_by_token[start_token]
    end_ann = ann_by_token[end_token]

    s0 = sample_by_token[start_ann["sample_token"]]
    s1 = sample_by_token[end_ann["sample_token"]]
    dt = max((float(s1["timestamp"]) - float(s0["timestamp"])) * 1e-6, 1e-6)

    current_sd = cam_front_sd_by_sample.get(ann["sample_token"])
    if current_sd is None:
        return "A", {"used_future": use_future, "vy_ego": 0.0, "reason": "missing_sample_data"}
    current_ego = ego_pose_by_token.get(current_sd["ego_pose_token"])
    if current_ego is None:
        return "A", {"used_future": use_future, "vy_ego": 0.0, "reason": "missing_ego_pose"}

    ego_yaw = yaw_from_quaternion_wxyz(current_ego["rotation"])
    vx_world = (end_ann["translation"][0] - start_ann["translation"][0]) / dt
    vy_world = (end_ann["translation"][1] - start_ann["translation"][1]) / dt
    _, vy_ego = global_to_ego_xy(vx_world, vy_world, ego_yaw)

    # If already not on drivable ground, likely to remain outside roadway.
    if sp6_choice in {"B", "C"}:
        return "E", {"used_future": use_future, "vy_ego": vy_ego, "reason": "already_off_road"}

    lateral_thr = 0.6
    if abs(vy_ego) <= lateral_thr:
        return "A", {"used_future": use_future, "vy_ego": vy_ego, "reason": "low_lateral_motion"}

    moving_left = vy_ego > lateral_thr
    moving_right = vy_ego < -lateral_thr

    # sp4 is object lane relative to ego: A same, B left adj, C right adj, D two left, E two right.
    if sp4_choice == "A":
        choice = "B" if moving_left else "C"
        reason = "same_lane_lateral_shift"
    elif sp4_choice == "B":
        if moving_right:
            choice, reason = "D", "left_adjacent_toward_ego_lane"
        else:
            choice, reason = "B", "left_adjacent_away_from_ego_lane"
    elif sp4_choice == "C":
        if moving_left:
            choice, reason = "D", "right_adjacent_toward_ego_lane"
        else:
            choice, reason = "C", "right_adjacent_away_from_ego_lane"
    elif sp4_choice == "D":
        choice = "D" if moving_right else "B"
        reason = "two_left_transition"
    elif sp4_choice == "E":
        choice = "D" if moving_left else "C"
        reason = "two_right_transition"
    else:
        choice, reason = "A", "fallback_remain"

    return choice, {"used_future": use_future, "vy_ego": vy_ego, "reason": reason}


def infer_te4_choice(
    te1_choice: str,
    te2_choice: str,
    su3_choice: str,
    su4_choice: str,
    su5_choice: str,
    te2_metrics: Dict[str, float | str | bool],
) -> Tuple[str, Dict[str, str | float | bool]]:
    """Infer ego's primary response to selected object."""
    ttc = float(te2_metrics.get("ttc_sec", -1.0))
    critical = su3_choice == "D" or te2_choice == "E" or (0.0 < ttc < 2.0)
    high = su3_choice == "C" or te2_choice == "D" or (2.0 <= ttc < 5.0)
    moderate = su3_choice == "B" or te2_choice == "C" or (5.0 <= ttc < 10.0)
    crossing = te1_choice == "E"

    if critical:
        if su4_choice in {"A", "B"}:
            choice = "D"
            reason = "critical_but_lane_change_available"
        else:
            choice = "F"
            reason = "critical_no_clearance"
    elif high:
        if crossing:
            choice = "E"
            reason = "high_risk_crossing_yield"
        elif su4_choice in {"A", "B"} and su5_choice in {"B", "C"}:
            choice = "D"
            reason = "high_risk_lane_change_option"
        else:
            choice = "C"
            reason = "high_risk_prepare_stop"
    elif moderate:
        if crossing:
            choice = "E"
            reason = "moderate_crossing_yield"
        else:
            choice = "B"
            reason = "moderate_risk_slow_down"
    else:
        if su5_choice in {"B", "C"} and su4_choice in {"A", "B"}:
            choice = "D"
            reason = "proactive_lane_change_feasible"
        else:
            choice = "A"
            reason = "low_risk_maintain"

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


def infer_te5_choice(
    te1_choice: str,
    te2_choice: str,
    te3_choice: str,
    su3_choice: str,
    su6_choice: str,
) -> Tuple[str, Dict[str, str | bool]]:
    """Infer likely next traffic event from object and scene dynamics."""
    critical_or_high = su3_choice in {"C", "D"} or te2_choice in {"D", "E"}
    lane_transition = te3_choice in {"B", "C", "D"}
    crossing_motion = te1_choice == "E"
    stopping_motion = te1_choice == "D"
    sparse_scene = su6_choice == "A"

    if lane_transition and te3_choice == "D":
        choice = "D"
        reason = "object_entering_ego_lane_merge"
    elif lane_transition:
        choice = "A"
        reason = "lane_transition_detected"
    elif stopping_motion and critical_or_high:
        choice = "B"
        reason = "high_risk_object_stopping"
    elif crossing_motion and critical_or_high:
        # Dataset target object is vehicle; crossing often manifests as merge behavior.
        choice = "D"
        reason = "crossing_vehicle_merge_like"
    elif stopping_motion:
        choice = "B"
        reason = "object_likely_stopping"
    elif sparse_scene and not crossing_motion:
        choice = "E"
        reason = "low_density_no_major_event"
    else:
        choice = "E"
        reason = "no_dominant_event"

    return choice, {
        "reason": reason,
        "critical_or_high": critical_or_high,
        "lane_transition": lane_transition,
        "crossing_motion": crossing_motion,
        "stopping_motion": stopping_motion,
        "sparse_scene": sparse_scene,
    }


def map_category_to_tm1_type(category_name: str) -> str | None:
    """Map nuScenes category names to TM-1 answer classes."""
    name = category_name.lower()
    if "bicycle" in name:
        return "Bicycle"
    if name.startswith("human.pedestrian"):
        return "Pedestrian"
    if name.startswith("vehicle"):
        return "Vehicle"
    if name.startswith("movable_object") or name.startswith("static_object"):
        return "Static obstacle"
    return None


def infer_tm1_choice(
    fifth_sample_token: str,
    sample_by_token: Dict[str, dict],
    annotations_by_sample: Dict[str, List[dict]],
    instance_by_token: Dict[str, dict],
    category_by_token: Dict[str, dict],
    history_frames: int = 4,
) -> Tuple[str, Dict[str, float | str]]:
    """Infer disappeared object class from previous frames to current frame."""
    prev_tokens: List[str] = []
    cur = fifth_sample_token
    for _ in range(history_frames):
        prev_token = sample_by_token[cur].get("prev")
        if not prev_token:
            break
        prev_tokens.append(prev_token)
        cur = prev_token

    prev_instances: set[str] = set()
    for token in prev_tokens:
        for ann in annotations_by_sample.get(token, []):
            prev_instances.add(ann["instance_token"])

    current_instances: set[str] = set()
    for ann in annotations_by_sample.get(fifth_sample_token, []):
        current_instances.add(ann["instance_token"])

    disappeared_instances = prev_instances - current_instances
    disappeared_types: set[str] = set()
    for instance_token in disappeared_instances:
        instance = instance_by_token.get(instance_token, {})
        cat_token = instance.get("category_token")
        cat_name = category_by_token.get(cat_token, {}).get("name", "")
        mapped = map_category_to_tm1_type(cat_name)
        if mapped:
            disappeared_types.add(mapped)

    if not disappeared_types:
        choice = "A"
    elif len(disappeared_types) > 1:
        choice = "F"
    elif "Pedestrian" in disappeared_types:
        choice = "B"
    elif "Vehicle" in disappeared_types:
        choice = "C"
    elif "Bicycle" in disappeared_types:
        choice = "D"
    elif "Static obstacle" in disappeared_types:
        choice = "E"
    else:
        choice = "A"

    return choice, {
        "history_frames_used": float(len(prev_tokens)),
        "prev_instance_count": float(len(prev_instances)),
        "current_instance_count": float(len(current_instances)),
        "disappeared_instance_count": float(len(disappeared_instances)),
        "disappeared_types": ",".join(sorted(disappeared_types)) if disappeared_types else "none",
    }


def infer_tm2_choice(
    selected_ann_token: str,
    ann_by_token: Dict[str, dict],
    ego_pose_by_token: Dict[str, dict],
    cam_front_sd_by_sample: Dict[str, dict],
) -> Tuple[str, Dict[str, float | str | bool]]:
    """Infer where object was relative to ego in previous frame(s)."""
    ann = ann_by_token[selected_ann_token]
    prev_token = ann.get("prev")
    if not prev_token:
        prev_token = selected_ann_token
        used_fallback = True
    else:
        used_fallback = False

    prev_ann = ann_by_token[prev_token]
    sample_token = prev_ann["sample_token"]
    sd = cam_front_sd_by_sample.get(sample_token)
    if sd is None:
        return "A", {"used_fallback": True, "x_ego": 0.0, "y_ego": 0.0, "angle_deg": 0.0}
    ego_pose = ego_pose_by_token.get(sd["ego_pose_token"])
    if ego_pose is None:
        return "A", {"used_fallback": True, "x_ego": 0.0, "y_ego": 0.0, "angle_deg": 0.0}

    ego_yaw = yaw_from_quaternion_wxyz(ego_pose["rotation"])
    dx = prev_ann["translation"][0] - ego_pose["translation"][0]
    dy = prev_ann["translation"][1] - ego_pose["translation"][1]
    x_ego, y_ego = global_to_ego_xy(dx, dy, ego_yaw)
    angle_deg = math.degrees(math.atan2(y_ego, x_ego))

    if -22.5 <= angle_deg < 22.5:
        choice = "A"
    elif 22.5 <= angle_deg < 67.5:
        choice = "B"
    elif -67.5 <= angle_deg < -22.5:
        choice = "C"
    elif 67.5 <= angle_deg < 112.5:
        choice = "D"
    elif -112.5 <= angle_deg < -67.5:
        choice = "E"
    else:
        choice = "F"

    return choice, {
        "used_fallback": used_fallback,
        "x_ego": x_ego,
        "y_ego": y_ego,
        "angle_deg": angle_deg,
    }


def infer_tm3_choice(
    selected_ann_token: str,
    ann_by_token: Dict[str, dict],
    sample_by_token: Dict[str, dict],
    ego_pose_by_token: Dict[str, dict],
    cam_front_sd_by_sample: Dict[str, dict],
) -> Tuple[str, Dict[str, float | str | bool]]:
    """Infer recent motion pattern using previous frames only."""
    ann = ann_by_token[selected_ann_token]

    prev_tokens: List[str] = []
    cur = selected_ann_token
    for _ in range(3):
        prev_token = ann_by_token[cur].get("prev")
        if not prev_token:
            break
        prev_tokens.append(prev_token)
        cur = prev_token
    prev_tokens.reverse()

    track_tokens = prev_tokens + [selected_ann_token]
    if len(track_tokens) < 2:
        return "F", {"reason": "insufficient_history", "speed0": 0.0, "speed1": 0.0, "vy_ego": 0.0}

    # Build ego-frame trajectory points from oldest -> newest.
    points: List[Tuple[float, float, float]] = []
    for token in track_tokens:
        a = ann_by_token[token]
        sd = cam_front_sd_by_sample.get(a["sample_token"])
        if sd is None:
            continue
        ego_pose = ego_pose_by_token.get(sd["ego_pose_token"])
        if ego_pose is None:
            continue
        t = float(sample_by_token[a["sample_token"]]["timestamp"]) * 1e-6
        ego_yaw = yaw_from_quaternion_wxyz(ego_pose["rotation"])
        dx = a["translation"][0] - ego_pose["translation"][0]
        dy = a["translation"][1] - ego_pose["translation"][1]
        x_ego, y_ego = global_to_ego_xy(dx, dy, ego_yaw)
        points.append((t, x_ego, y_ego))

    if len(points) < 2:
        return "F", {"reason": "invalid_history_points", "speed0": 0.0, "speed1": 0.0, "vy_ego": 0.0}

    points.sort(key=lambda x: x[0])
    # Last segment speed and prior segment speed for accel/decel trend.
    t0, x0, y0 = points[-2]
    t1, x1, y1 = points[-1]
    dt1 = max(t1 - t0, 1e-6)
    vx1 = (x1 - x0) / dt1
    vy1 = (y1 - y0) / dt1
    speed1 = math.hypot(vx1, vy1)

    if len(points) >= 3:
        ta, xa, ya = points[-3]
        tb, xb, yb = points[-2]
        dt0 = max(tb - ta, 1e-6)
        vx0 = (xb - xa) / dt0
        vy0 = (yb - ya) / dt0
        speed0 = math.hypot(vx0, vy0)
    else:
        speed0 = speed1

    if speed1 < 0.35:
        choice = "F"
        reason = "stopped_recently"
    elif abs(vy1) > abs(vx1) * 1.1 and abs(vy1) > 0.6:
        choice = "D" if vy1 > 0 else "E"
        reason = "lateral_turning_pattern"
    elif speed1 - speed0 > 0.5:
        choice = "C"
        reason = "accelerating_trend"
    elif speed0 - speed1 > 0.5:
        choice = "B"
        reason = "slowing_trend"
    else:
        choice = "A"
        reason = "straight_motion_trend"

    return choice, {
        "reason": reason,
        "speed0": speed0,
        "speed1": speed1,
        "vx_ego": vx1,
        "vy_ego": vy1,
        "history_steps_used": float(len(track_tokens) - 1),
    }


def infer_tm4_choice(
    fifth_sample_token: str,
    sample_by_token: Dict[str, dict],
    annotations_by_sample: Dict[str, List[dict]],
    ann_by_token: Dict[str, dict],
    instance_by_token: Dict[str, dict],
    category_by_token: Dict[str, dict],
    history_frames: int = 4,
) -> Tuple[str, Dict[str, float | str]]:
    """Infer occluded-now objects that were previously visible.

    Only consider stable instances that are present in every frame of the
    5-frame window (previous frames + current), to avoid counting objects that
    simply left the scene.
    """
    # Gather previous sample tokens.
    prev_tokens: List[str] = []
    cur = fifth_sample_token
    for _ in range(history_frames):
        prev_token = sample_by_token[cur].get("prev")
        if not prev_token:
            break
        prev_tokens.append(prev_token)
        cur = prev_token

    window_tokens = list(reversed(prev_tokens)) + [fifth_sample_token]
    if not window_tokens:
        return "A", {
            "history_frames_used": 0.0,
            "stable_instance_count": 0.0,
            "previously_visible_instances": 0.0,
            "occluded_now_instances": 0.0,
        }

    # Build per-frame instance -> annotation lookup.
    frame_maps: list[dict[str, dict]] = []
    for token in window_tokens:
        anns = annotations_by_sample.get(token, [])
        frame_maps.append({ann["instance_token"]: ann for ann in anns})

    # Stable instances: present in every frame of this 5-frame window.
    stable_instances = set(frame_maps[0].keys())
    for mapping in frame_maps[1:]:
        stable_instances &= set(mapping.keys())

    # Previously visible if visible (>=3) in any previous frame in the window.
    previously_visible: set[str] = set()
    for inst in stable_instances:
        for mapping in frame_maps[:-1]:
            vis = int(mapping[inst].get("visibility_token", "0") or 0)
            if vis >= 3:
                previously_visible.add(inst)
                break

    # Occluded now if currently low visibility (<=2) while previously visible.
    current_map = frame_maps[-1]
    occluded_now_instances: set[str] = set()
    for inst in previously_visible:
        vis_now = int(current_map[inst].get("visibility_token", "0") or 0)
        if vis_now <= 2:
            occluded_now_instances.add(inst)

    occluded_count = len(occluded_now_instances)
    if occluded_count == 0:
        choice = "A"
    elif occluded_count == 1:
        choice = "B"
    elif 2 <= occluded_count <= 3:
        choice = "C"
    else:
        choice = "D"

    return choice, {
        "history_frames_used": float(len(prev_tokens)),
        "stable_instance_count": float(len(stable_instances)),
        "previously_visible_instances": float(len(previously_visible)),
        "occluded_now_instances": float(occluded_count),
    }


def infer_tm5_choice(
    selected_ann_token: str,
    ann_by_token: Dict[str, dict],
    ego_pose_by_token: Dict[str, dict],
    cam_front_sd_by_sample: Dict[str, dict],
    sample_by_token: Dict[str, dict],
    scene_by_token: Dict[str, dict],
    log_by_token: Dict[str, dict],
    map_cache: Dict[str, NuScenesMap],
    root: Path,
    lane_width_m: float,
) -> Tuple[str, Dict[str, float | str | bool | None]]:
    """Infer object's lane relation in previous frame(s)."""
    ann = ann_by_token[selected_ann_token]
    prev_token = ann.get("prev")
    if not prev_token:
        prev_token = selected_ann_token
        used_fallback = True
    else:
        used_fallback = False

    prev_ann = ann_by_token[prev_token]
    sample_token = prev_ann["sample_token"]
    sd = cam_front_sd_by_sample.get(sample_token)
    if sd is None:
        return "D", {"used_fallback": True, "method": "missing_sample_data", "lateral_offset_m": 0.0}
    ego_pose = ego_pose_by_token.get(sd["ego_pose_token"])
    if ego_pose is None:
        return "D", {"used_fallback": True, "method": "missing_ego_pose", "lateral_offset_m": 0.0}

    ego_yaw = yaw_from_quaternion_wxyz(ego_pose["rotation"])
    dx = prev_ann["translation"][0] - ego_pose["translation"][0]
    dy = prev_ann["translation"][1] - ego_pose["translation"][1]
    _, y_ego = global_to_ego_xy(dx, dy, ego_yaw)

    sample_row = sample_by_token.get(sample_token, {})
    scene_token = sample_row.get("scene_token")
    scene_row = scene_by_token.get(scene_token, {})
    log_token = scene_row.get("log_token")
    location = log_by_token.get(log_token, {}).get("location")
    ego_lane_token = ""
    obj_lane_token = ""

    if location:
        try:
            if location not in map_cache:
                map_cache[location] = NuScenesMap(dataroot=str(root), map_name=location)
            nusc_map = map_cache[location]
            ego_lane_token = nusc_map.get_closest_lane(
                ego_pose["translation"][0], ego_pose["translation"][1], radius=5
            )
            obj_lane_token = nusc_map.get_closest_lane(
                prev_ann["translation"][0], prev_ann["translation"][1], radius=5
            )
            layers = nusc_map.layers_on_point(
                prev_ann["translation"][0],
                prev_ann["translation"][1],
                layer_names=["drivable_area", "road_segment", "lane", "walkway", "sidewalk"],
            )
            on_road = bool(layers.get("drivable_area")) or bool(layers.get("road_segment")) or bool(
                layers.get("lane")
            )
            if ego_lane_token and obj_lane_token and ego_lane_token == obj_lane_token:
                return "A", {
                    "used_fallback": used_fallback,
                    "method": "map_same_lane",
                    "location": location,
                    "ego_lane_token": ego_lane_token,
                    "obj_lane_token": obj_lane_token,
                    "lateral_offset_m": y_ego,
                }
            if not on_road and not obj_lane_token:
                return "D", {
                    "used_fallback": used_fallback,
                    "method": "map_outside_road",
                    "location": location,
                    "ego_lane_token": ego_lane_token or None,
                    "obj_lane_token": obj_lane_token or None,
                    "lateral_offset_m": y_ego,
                }
        except FileNotFoundError:
            pass

    lane_steps = int(round(y_ego / max(lane_width_m, 1e-3)))
    # Without reliable map geometry, avoid over-labeling "outside roadway".
    # Clamp fallback lane estimate to adjacent/same lane only.
    lane_steps = max(-1, min(1, lane_steps))
    if lane_steps == 1:
        choice = "B"
    elif lane_steps == -1:
        choice = "C"
    else:
        choice = "A"

    return choice, {
        "used_fallback": used_fallback,
        "method": "lateral_offset_fallback",
        "location": location,
        "ego_lane_token": ego_lane_token or None,
        "obj_lane_token": obj_lane_token or None,
        "lateral_offset_m": y_ego,
        "lane_steps": lane_steps,
    }


def build_object_reference(
    selected_ann_token: str,
    ann_by_token: Dict[str, dict],
    ego_pose_by_token: Dict[str, dict],
    cam_front_sd_by_sample: Dict[str, dict],
) -> str:
    """Create a human-friendly object description for question placeholders."""
    ann = ann_by_token[selected_ann_token]
    sample_token = ann["sample_token"]
    sd = cam_front_sd_by_sample.get(sample_token)
    if sd is None:
        return "the selected vehicle"
    ego_pose = ego_pose_by_token.get(sd["ego_pose_token"])
    if ego_pose is None:
        return "the selected vehicle"

    ego_yaw = yaw_from_quaternion_wxyz(ego_pose["rotation"])
    dx = ann["translation"][0] - ego_pose["translation"][0]
    dy = ann["translation"][1] - ego_pose["translation"][1]
    x_ego, y_ego = global_to_ego_xy(dx, dy, ego_yaw)
    distance = math.hypot(dx, dy)

    if x_ego >= 0 and abs(y_ego) < 3.0:
        rel = "in front"
    elif x_ego >= 0 and y_ego >= 3.0:
        rel = "front-left"
    elif x_ego >= 0 and y_ego <= -3.0:
        rel = "front-right"
    elif x_ego < 0 and abs(y_ego) < 3.0:
        rel = "behind"
    elif x_ego < 0 and y_ego >= 3.0:
        rel = "rear-left"
    else:
        rel = "rear-right"

    distance_text = f"{int(round(distance))}m"
    return f"the vehicle {rel} at about {distance_text}"


def build_frq_prompt(
    frq_question: str,
    object_reference: str,
    mcq_rows: List[dict],
    scene_name: str = "",
    scene_description: str = "",
    group_scene_id: str = "",
    vehicle_annotation_count: int | None = None,
) -> str:
    context_lines: List[str] = []
    for row in mcq_rows:
        question = row.get("question", "")
        answer_key = row.get("ground_truth", "")
        answer_text = row.get("model_response_text", "")
        context_lines.append(f"- {row.get('question_id', '')}: {question}")
        context_lines.append(f"  GT answer: {answer_key} ({answer_text})")
    context_text = "\n".join(context_lines)
    filled_question = frq_question.replace("<obj>", object_reference)
    scene_context_lines: List[str] = []
    if group_scene_id:
        scene_context_lines.append(f"- Group scene_id: {group_scene_id}")
    if scene_name:
        scene_context_lines.append(f"- nuScenes scene name: {scene_name}")
    if scene_description:
        scene_context_lines.append(f"- Scene description: {scene_description}")
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


def validate_mcq_rows_for_frq(mcq_rows: List[dict], label: str) -> None:
    """Ensure FRQ context uses parser-generated MCQ ground-truth answers."""
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

    root = args.root.resolve()
    formatted_dir = (
        args.formatted_scenes_dir.resolve()
        if args.formatted_scenes_dir
        else root / "formatted_scenes"
    )
    questions_json = (
        args.questions_json.resolve() if args.questions_json else root / "questions.json"
    )
    output_json = (
        args.output_json.resolve()
        if args.output_json
        else formatted_dir / "generated_answers_all.json"
    )
    questions_output = (
        args.questions_output.resolve()
        if args.questions_output
        else root / "questions_with_answers_all.json"
    )

    version_dir = root / args.version
    sample_data = load_json(version_dir / "sample_data.json")
    sample = load_json(version_dir / "sample.json")
    sample_annotation = load_json(version_dir / "sample_annotation.json")
    instance = load_json(version_dir / "instance.json")
    category = load_json(version_dir / "category.json")
    ego_pose = load_json(version_dir / "ego_pose.json")
    scene = load_json(version_dir / "scene.json")
    log = load_json(version_dir / "log.json")
    questions = load_json(questions_json)

    sample_by_token = {row["token"]: row for row in sample}
    ann_by_token = {row["token"]: row for row in sample_annotation}
    annotations_by_sample: Dict[str, List[dict]] = {}
    for ann in sample_annotation:
        annotations_by_sample.setdefault(ann["sample_token"], []).append(ann)
    instance_by_token = {row["token"]: row for row in instance}
    category_by_token = {row["token"]: row for row in category}
    ego_pose_by_token = {row["token"]: row for row in ego_pose}
    scene_by_token = {row["token"]: row for row in scene}
    log_by_token = {row["token"]: row for row in log}
    cam_front_sd_by_sample: Dict[str, dict] = {}
    for row in sample_data:
        filename = row.get("filename", "")
        if filename.startswith("samples/CAM_FRONT/") and filename.endswith(".jpg"):
            cam_front_sd_by_sample[row["sample_token"]] = row

    sp1_template = None
    sp2_template = None
    sp3_template = None
    sp4_template = None
    sp5_template = None
    sp6_template = None
    su1_template = None
    su2_template = None
    su3_template = None
    su4_template = None
    su5_template = None
    su6_template = None
    te1_template = None
    te2_template = None
    te3_template = None
    te4_template = None
    te5_template = None
    tm1_template = None
    tm2_template = None
    tm3_template = None
    tm4_template = None
    tm5_template = None
    sp7_template = None
    su7_template = None
    te6_template = None
    tm6_template = None
    for task in questions.get("tasks", []):
        if task.get("id") == "SP-1":
            sp1_template = task
        elif task.get("id") == "SP-2":
            sp2_template = task
        elif task.get("id") == "SP-3":
            sp3_template = task
        elif task.get("id") == "SP-4":
            sp4_template = task
        elif task.get("id") == "SP-5":
            sp5_template = task
        elif task.get("id") == "SP-6":
            sp6_template = task
        elif task.get("id") == "SU-1":
            su1_template = task
        elif task.get("id") == "SU-2":
            su2_template = task
        elif task.get("id") == "SU-3":
            su3_template = task
        elif task.get("id") == "SU-4":
            su4_template = task
        elif task.get("id") == "SU-5":
            su5_template = task
        elif task.get("id") == "SU-6":
            su6_template = task
        elif task.get("id") == "TE-1":
            te1_template = task
        elif task.get("id") == "TE-2":
            te2_template = task
        elif task.get("id") == "TE-3":
            te3_template = task
        elif task.get("id") == "TE-4":
            te4_template = task
        elif task.get("id") == "TE-5":
            te5_template = task
        elif task.get("id") == "TM-1":
            tm1_template = task
        elif task.get("id") == "TM-2":
            tm2_template = task
        elif task.get("id") == "TM-3":
            tm3_template = task
        elif task.get("id") == "TM-4":
            tm4_template = task
        elif task.get("id") == "TM-5":
            tm5_template = task
        elif task.get("id") == "SP-7":
            sp7_template = task
        elif task.get("id") == "SU-7":
            su7_template = task
        elif task.get("id") == "TE-6":
            te6_template = task
        elif task.get("id") == "TM-6":
            tm6_template = task
    if sp1_template is None:
        raise ValueError("SP-1 task not found in questions.json")
    if sp2_template is None:
        raise ValueError("SP-2 task not found in questions.json")
    if sp3_template is None:
        raise ValueError("SP-3 task not found in questions.json")
    if sp4_template is None:
        raise ValueError("SP-4 task not found in questions.json")
    if sp5_template is None:
        raise ValueError("SP-5 task not found in questions.json")
    if sp6_template is None:
        raise ValueError("SP-6 task not found in questions.json")
    if su1_template is None:
        raise ValueError("SU-1 task not found in questions.json")
    if su2_template is None:
        raise ValueError("SU-2 task not found in questions.json")
    if su3_template is None:
        raise ValueError("SU-3 task not found in questions.json")
    if su4_template is None:
        raise ValueError("SU-4 task not found in questions.json")
    if su5_template is None:
        raise ValueError("SU-5 task not found in questions.json")
    if su6_template is None:
        raise ValueError("SU-6 task not found in questions.json")
    if te1_template is None:
        raise ValueError("TE-1 task not found in questions.json")
    if te2_template is None:
        raise ValueError("TE-2 task not found in questions.json")
    if te3_template is None:
        raise ValueError("TE-3 task not found in questions.json")
    if te4_template is None:
        raise ValueError("TE-4 task not found in questions.json")
    if te5_template is None:
        raise ValueError("TE-5 task not found in questions.json")
    if tm1_template is None:
        raise ValueError("TM-1 task not found in questions.json")
    if tm2_template is None:
        raise ValueError("TM-2 task not found in questions.json")
    if tm3_template is None:
        raise ValueError("TM-3 task not found in questions.json")
    if tm4_template is None:
        raise ValueError("TM-4 task not found in questions.json")
    if tm5_template is None:
        raise ValueError("TM-5 task not found in questions.json")
    if sp7_template is None:
        raise ValueError("SP-7 task not found in questions.json")
    if su7_template is None:
        raise ValueError("SU-7 task not found in questions.json")
    if te6_template is None:
        raise ValueError("TE-6 task not found in questions.json")
    if tm6_template is None:
        raise ValueError("TM-6 task not found in questions.json")

    group_json_paths = sorted(formatted_dir.glob("scene_*/group_*_vehicle_annotations.json"))
    sp1_results = []
    sp2_results = []
    sp3_results = []
    sp4_results = []
    sp5_results = []
    sp6_results = []
    su1_results = []
    su2_results = []
    su3_results = []
    su4_results = []
    su5_results = []
    su6_results = []
    te1_results = []
    te2_results = []
    te3_results = []
    te4_results = []
    te5_results = []
    tm1_results = []
    tm2_results = []
    tm3_results = []
    tm4_results = []
    tm5_results = []
    sp7_results = []
    su7_results = []
    te6_results = []
    tm6_results = []
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
    map_cache: Dict[str, NuScenesMap] = {}
    for path in group_json_paths:
        payload = load_json(path)
        selected_token = payload.get("selected_vehicle_annotation_token")
        if not selected_token:
            continue
        if selected_token not in ann_by_token:
            continue

        choice, metrics = infer_sp1_choice(
            selected_ann_token=selected_token,
            ann_by_token=ann_by_token,
            sample_by_token=sample_by_token,
            ego_pose_by_token=ego_pose_by_token,
            cam_front_sd_by_sample=cam_front_sd_by_sample,
            window=args.track_window,
        )
        object_ref = build_object_reference(
            selected_ann_token=selected_token,
            ann_by_token=ann_by_token,
            ego_pose_by_token=ego_pose_by_token,
            cam_front_sd_by_sample=cam_front_sd_by_sample,
        )

        sp1_row = {
            "scene_id": payload["scene_id"],
            "group_id": payload["group_id"],
            "question_id": "SP-1",
            "object_id": selected_token,
            "object_reference": object_ref,
            "question": sp1_template["question"].replace("<obj>", object_ref),
            "choices": sp1_template["choices"],
            "ground_truth": choice,
            "model_response": "",
            "model_response_text": SP1_CHOICES[choice],
            "metrics": {k: round(v, 4) for k, v in metrics.items()},
            "source_group_file": str(path.relative_to(formatted_dir)),
        }
        sp1_results.append(sp1_row)

        sp2_choice, sp2_metrics = infer_sp2_choice(
            selected_ann_token=selected_token,
            ann_by_token=ann_by_token,
            ego_pose_by_token=ego_pose_by_token,
            cam_front_sd_by_sample=cam_front_sd_by_sample,
        )
        sp2_row = {
            "scene_id": payload["scene_id"],
            "group_id": payload["group_id"],
            "question_id": "SP-2",
            "object_id": selected_token,
            "object_reference": object_ref,
            "question": sp2_template["question"].replace("<obj>", object_ref),
            "choices": sp2_template["choices"],
            "ground_truth": sp2_choice,
            "model_response": "",
            "model_response_text": SP2_CHOICES[sp2_choice],
            "metrics": {k: round(v, 4) for k, v in sp2_metrics.items()},
            "source_group_file": str(path.relative_to(formatted_dir)),
        }
        sp2_results.append(sp2_row)

        sp3_choice, sp3_metrics = infer_sp3_choice(
            selected_ann_token=selected_token,
            ann_by_token=ann_by_token,
            ego_pose_by_token=ego_pose_by_token,
            cam_front_sd_by_sample=cam_front_sd_by_sample,
        )
        sp3_row = {
            "scene_id": payload["scene_id"],
            "group_id": payload["group_id"],
            "question_id": "SP-3",
            "object_id": selected_token,
            "object_reference": object_ref,
            "question": sp3_template["question"].replace("<obj>", object_ref),
            "choices": sp3_template["choices"],
            "ground_truth": sp3_choice,
            "model_response": "",
            "model_response_text": SP3_CHOICES[sp3_choice],
            "metrics": {k: round(v, 4) for k, v in sp3_metrics.items()},
            "source_group_file": str(path.relative_to(formatted_dir)),
        }
        sp3_results.append(sp3_row)

        sp4_choice, sp4_metrics = infer_sp4_choice(
            selected_ann_token=selected_token,
            ann_by_token=ann_by_token,
            ego_pose_by_token=ego_pose_by_token,
            cam_front_sd_by_sample=cam_front_sd_by_sample,
            sample_by_token=sample_by_token,
            scene_by_token=scene_by_token,
            log_by_token=log_by_token,
            map_cache=map_cache,
            root=root,
            lane_width_m=args.lane_width_m,
        )
        sp4_row = {
            "scene_id": payload["scene_id"],
            "group_id": payload["group_id"],
            "question_id": "SP-4",
            "object_id": selected_token,
            "object_reference": object_ref,
            "question": sp4_template["question"].replace("<obj>", object_ref),
            "choices": sp4_template["choices"],
            "ground_truth": sp4_choice,
            "model_response": "",
            "model_response_text": SP4_CHOICES[sp4_choice],
            "metrics": {
                k: (round(v, 4) if isinstance(v, float) else v) for k, v in sp4_metrics.items()
            },
            "source_group_file": str(path.relative_to(formatted_dir)),
        }
        sp4_results.append(sp4_row)

        sp5_choice, sp5_metrics = infer_sp5_choice(
            selected_ann_token=selected_token,
            ann_by_token=ann_by_token,
        )
        sp5_row = {
            "scene_id": payload["scene_id"],
            "group_id": payload["group_id"],
            "question_id": "SP-5",
            "object_id": selected_token,
            "object_reference": object_ref,
            "question": sp5_template["question"].replace("<obj>", object_ref),
            "choices": sp5_template["choices"],
            "ground_truth": sp5_choice,
            "model_response": "",
            "model_response_text": SP5_CHOICES[sp5_choice],
            "metrics": {
                k: (round(v, 4) if isinstance(v, float) else v) for k, v in sp5_metrics.items()
            },
            "source_group_file": str(path.relative_to(formatted_dir)),
        }
        sp5_results.append(sp5_row)

        sp6_choice, sp6_metrics = infer_sp6_choice(
            selected_ann_token=selected_token,
            ann_by_token=ann_by_token,
            ego_pose_by_token=ego_pose_by_token,
            cam_front_sd_by_sample=cam_front_sd_by_sample,
            sample_by_token=sample_by_token,
            scene_by_token=scene_by_token,
            log_by_token=log_by_token,
            map_cache=map_cache,
            root=root,
        )
        sp6_row = {
            "scene_id": payload["scene_id"],
            "group_id": payload["group_id"],
            "question_id": "SP-6",
            "object_id": selected_token,
            "object_reference": object_ref,
            "question": sp6_template["question"].replace("<obj>", object_ref),
            "choices": sp6_template["choices"],
            "ground_truth": sp6_choice,
            "model_response": "",
            "model_response_text": SP6_CHOICES[sp6_choice],
            "metrics": {
                k: (round(v, 4) if isinstance(v, float) else v) for k, v in sp6_metrics.items()
            },
            "source_group_file": str(path.relative_to(formatted_dir)),
        }
        sp6_results.append(sp6_row)

        su1_choice, su1_metrics = infer_su1_choice(
            sp1_choice=choice,
            sp2_choice=sp2_choice,
            sp3_choice=sp3_choice,
            sp4_choice=sp4_choice,
            sp1_metrics=metrics,
        )
        su1_row = {
            "scene_id": payload["scene_id"],
            "group_id": payload["group_id"],
            "question_id": "SU-1",
            "object_id": selected_token,
            "object_reference": object_ref,
            "question": su1_template["question"].replace("<obj>", object_ref),
            "choices": su1_template["choices"],
            "ground_truth": su1_choice,
            "model_response": "",
            "model_response_text": SU1_CHOICES[su1_choice],
            "metrics": {
                k: (round(v, 4) if isinstance(v, float) else v) for k, v in su1_metrics.items()
            },
            "source_group_file": str(path.relative_to(formatted_dir)),
        }
        su1_results.append(su1_row)

        su2_choice, su2_metrics = infer_su2_choice(
            sp1_choice=choice,
            sp2_choice=sp2_choice,
            sp3_choice=sp3_choice,
            sp4_choice=sp4_choice,
            su1_choice=su1_choice,
        )
        su2_row = {
            "scene_id": payload["scene_id"],
            "group_id": payload["group_id"],
            "question_id": "SU-2",
            "object_id": selected_token,
            "object_reference": object_ref,
            "question": su2_template["question"],
            "choices": su2_template["choices"],
            "ground_truth": su2_choice,
            "model_response": "",
            "model_response_text": SU2_CHOICES[su2_choice],
            "metrics": su2_metrics,
            "source_group_file": str(path.relative_to(formatted_dir)),
        }
        su2_results.append(su2_row)

        su3_choice, su3_metrics = infer_su3_choice(
            sp1_choice=choice,
            sp2_choice=sp2_choice,
            sp3_choice=sp3_choice,
            su1_choice=su1_choice,
            sp1_metrics=metrics,
        )
        su3_row = {
            "scene_id": payload["scene_id"],
            "group_id": payload["group_id"],
            "question_id": "SU-3",
            "object_id": selected_token,
            "object_reference": object_ref,
            "question": su3_template["question"].replace("<obj>", object_ref),
            "choices": su3_template["choices"],
            "ground_truth": su3_choice,
            "model_response": "",
            "model_response_text": SU3_CHOICES[su3_choice],
            "metrics": {
                k: (round(v, 4) if isinstance(v, float) else v) for k, v in su3_metrics.items()
            },
            "source_group_file": str(path.relative_to(formatted_dir)),
        }
        su3_results.append(su3_row)

        su4_choice, su4_metrics = infer_su4_choice(
            sp1_choice=choice,
            sp2_choice=sp2_choice,
            sp3_choice=sp3_choice,
            sp4_choice=sp4_choice,
            su1_choice=su1_choice,
            su3_choice=su3_choice,
        )
        su4_row = {
            "scene_id": payload["scene_id"],
            "group_id": payload["group_id"],
            "question_id": "SU-4",
            "object_id": selected_token,
            "object_reference": object_ref,
            "question": su4_template["question"],
            "choices": su4_template["choices"],
            "ground_truth": su4_choice,
            "model_response": "",
            "model_response_text": SU4_CHOICES[su4_choice],
            "metrics": su4_metrics,
            "source_group_file": str(path.relative_to(formatted_dir)),
        }
        su4_results.append(su4_row)

        su5_choice, su5_metrics = infer_su5_choice(
            sp1_choice=choice,
            sp2_choice=sp2_choice,
            sp3_choice=sp3_choice,
            su1_choice=su1_choice,
            su3_choice=su3_choice,
            su4_choice=su4_choice,
        )
        su5_row = {
            "scene_id": payload["scene_id"],
            "group_id": payload["group_id"],
            "question_id": "SU-5",
            "object_id": selected_token,
            "object_reference": object_ref,
            "question": su5_template["question"],
            "choices": su5_template["choices"],
            "ground_truth": su5_choice,
            "model_response": "",
            "model_response_text": SU5_CHOICES[su5_choice],
            "metrics": su5_metrics,
            "source_group_file": str(path.relative_to(formatted_dir)),
        }
        su5_results.append(su5_row)

        su6_choice, su6_metrics = infer_su6_choice(
            vehicle_annotations=payload.get("vehicle_annotations", []),
            ego_pose_by_token=ego_pose_by_token,
            cam_front_sd_by_sample=cam_front_sd_by_sample,
            sample_token=payload["fifth_frame_sample_token"],
        )
        su6_row = {
            "scene_id": payload["scene_id"],
            "group_id": payload["group_id"],
            "question_id": "SU-6",
            "object_id": selected_token,
            "object_reference": object_ref,
            "question": su6_template["question"],
            "choices": su6_template["choices"],
            "ground_truth": su6_choice,
            "model_response": "",
            "model_response_text": SU6_CHOICES[su6_choice],
            "metrics": {k: round(v, 4) for k, v in su6_metrics.items()},
            "source_group_file": str(path.relative_to(formatted_dir)),
        }
        su6_results.append(su6_row)

        te1_choice, te1_metrics = infer_te1_choice(
            selected_ann_token=selected_token,
            ann_by_token=ann_by_token,
            sample_by_token=sample_by_token,
            ego_pose_by_token=ego_pose_by_token,
            cam_front_sd_by_sample=cam_front_sd_by_sample,
        )
        te1_row = {
            "scene_id": payload["scene_id"],
            "group_id": payload["group_id"],
            "question_id": "TE-1",
            "object_id": selected_token,
            "object_reference": object_ref,
            "question": te1_template["question"].replace("<obj>", object_ref),
            "choices": te1_template["choices"],
            "ground_truth": te1_choice,
            "model_response": "",
            "model_response_text": TE1_CHOICES[te1_choice],
            "metrics": {
                k: (round(v, 4) if isinstance(v, float) else v) for k, v in te1_metrics.items()
            },
            "source_group_file": str(path.relative_to(formatted_dir)),
        }
        te1_results.append(te1_row)

        te2_choice, te2_metrics = infer_te2_choice(
            selected_ann_token=selected_token,
            ann_by_token=ann_by_token,
            sample_by_token=sample_by_token,
            ego_pose_by_token=ego_pose_by_token,
            cam_front_sd_by_sample=cam_front_sd_by_sample,
        )
        te2_row = {
            "scene_id": payload["scene_id"],
            "group_id": payload["group_id"],
            "question_id": "TE-2",
            "object_id": selected_token,
            "object_reference": object_ref,
            "question": te2_template["question"],
            "choices": te2_template["choices"],
            "ground_truth": te2_choice,
            "model_response": "",
            "model_response_text": TE2_CHOICES[te2_choice],
            "metrics": {
                k: (round(v, 4) if isinstance(v, float) else v) for k, v in te2_metrics.items()
            },
            "source_group_file": str(path.relative_to(formatted_dir)),
        }
        te2_results.append(te2_row)

        te3_choice, te3_metrics = infer_te3_choice(
            selected_ann_token=selected_token,
            ann_by_token=ann_by_token,
            sample_by_token=sample_by_token,
            ego_pose_by_token=ego_pose_by_token,
            cam_front_sd_by_sample=cam_front_sd_by_sample,
            sp4_choice=sp4_choice,
            sp6_choice=sp6_choice,
        )
        te3_row = {
            "scene_id": payload["scene_id"],
            "group_id": payload["group_id"],
            "question_id": "TE-3",
            "object_id": selected_token,
            "object_reference": object_ref,
            "question": te3_template["question"].replace("<obj>", object_ref),
            "choices": te3_template["choices"],
            "ground_truth": te3_choice,
            "model_response": "",
            "model_response_text": TE3_CHOICES[te3_choice],
            "metrics": {
                k: (round(v, 4) if isinstance(v, float) else v) for k, v in te3_metrics.items()
            },
            "source_group_file": str(path.relative_to(formatted_dir)),
        }
        te3_results.append(te3_row)

        te4_choice, te4_metrics = infer_te4_choice(
            te1_choice=te1_choice,
            te2_choice=te2_choice,
            su3_choice=su3_choice,
            su4_choice=su4_choice,
            su5_choice=su5_choice,
            te2_metrics=te2_metrics,
        )
        te4_row = {
            "scene_id": payload["scene_id"],
            "group_id": payload["group_id"],
            "question_id": "TE-4",
            "object_id": selected_token,
            "object_reference": object_ref,
            "question": te4_template["question"].replace("<obj>", object_ref),
            "choices": te4_template["choices"],
            "ground_truth": te4_choice,
            "model_response": "",
            "model_response_text": TE4_CHOICES[te4_choice],
            "metrics": {
                k: (round(v, 4) if isinstance(v, float) else v) for k, v in te4_metrics.items()
            },
            "source_group_file": str(path.relative_to(formatted_dir)),
        }
        te4_results.append(te4_row)

        te5_choice, te5_metrics = infer_te5_choice(
            te1_choice=te1_choice,
            te2_choice=te2_choice,
            te3_choice=te3_choice,
            su3_choice=su3_choice,
            su6_choice=su6_choice,
        )
        te5_row = {
            "scene_id": payload["scene_id"],
            "group_id": payload["group_id"],
            "question_id": "TE-5",
            "object_id": selected_token,
            "object_reference": object_ref,
            "question": te5_template["question"],
            "choices": te5_template["choices"],
            "ground_truth": te5_choice,
            "model_response": "",
            "model_response_text": TE5_CHOICES[te5_choice],
            "metrics": te5_metrics,
            "source_group_file": str(path.relative_to(formatted_dir)),
        }
        te5_results.append(te5_row)

        tm1_choice, tm1_metrics = infer_tm1_choice(
            fifth_sample_token=payload["fifth_frame_sample_token"],
            sample_by_token=sample_by_token,
            annotations_by_sample=annotations_by_sample,
            instance_by_token=instance_by_token,
            category_by_token=category_by_token,
            history_frames=4,
        )
        tm1_row = {
            "scene_id": payload["scene_id"],
            "group_id": payload["group_id"],
            "question_id": "TM-1",
            "object_id": selected_token,
            "object_reference": object_ref,
            "question": tm1_template["question"],
            "choices": tm1_template["choices"],
            "ground_truth": tm1_choice,
            "model_response": "",
            "model_response_text": TM1_CHOICES[tm1_choice],
            "metrics": {
                k: (round(v, 4) if isinstance(v, float) else v) for k, v in tm1_metrics.items()
            },
            "source_group_file": str(path.relative_to(formatted_dir)),
        }
        tm1_results.append(tm1_row)

        tm2_choice, tm2_metrics = infer_tm2_choice(
            selected_ann_token=selected_token,
            ann_by_token=ann_by_token,
            ego_pose_by_token=ego_pose_by_token,
            cam_front_sd_by_sample=cam_front_sd_by_sample,
        )
        tm2_row = {
            "scene_id": payload["scene_id"],
            "group_id": payload["group_id"],
            "question_id": "TM-2",
            "object_id": selected_token,
            "object_reference": object_ref,
            "question": tm2_template["question"].replace("<obj>", object_ref),
            "choices": tm2_template["choices"],
            "ground_truth": tm2_choice,
            "model_response": "",
            "model_response_text": TM2_CHOICES[tm2_choice],
            "metrics": {
                k: (round(v, 4) if isinstance(v, float) else v) for k, v in tm2_metrics.items()
            },
            "source_group_file": str(path.relative_to(formatted_dir)),
        }
        tm2_results.append(tm2_row)

        tm3_choice, tm3_metrics = infer_tm3_choice(
            selected_ann_token=selected_token,
            ann_by_token=ann_by_token,
            sample_by_token=sample_by_token,
            ego_pose_by_token=ego_pose_by_token,
            cam_front_sd_by_sample=cam_front_sd_by_sample,
        )
        tm3_row = {
            "scene_id": payload["scene_id"],
            "group_id": payload["group_id"],
            "question_id": "TM-3",
            "object_id": selected_token,
            "object_reference": object_ref,
            "question": tm3_template["question"].replace("<obj>", object_ref),
            "choices": tm3_template["choices"],
            "ground_truth": tm3_choice,
            "model_response": "",
            "model_response_text": TM3_CHOICES[tm3_choice],
            "metrics": {
                k: (round(v, 4) if isinstance(v, float) else v) for k, v in tm3_metrics.items()
            },
            "source_group_file": str(path.relative_to(formatted_dir)),
        }
        tm3_results.append(tm3_row)

        tm4_choice, tm4_metrics = infer_tm4_choice(
            fifth_sample_token=payload["fifth_frame_sample_token"],
            sample_by_token=sample_by_token,
            annotations_by_sample=annotations_by_sample,
            ann_by_token=ann_by_token,
            instance_by_token=instance_by_token,
            category_by_token=category_by_token,
            history_frames=4,
        )
        tm4_row = {
            "scene_id": payload["scene_id"],
            "group_id": payload["group_id"],
            "question_id": "TM-4",
            "object_id": selected_token,
            "object_reference": object_ref,
            "question": tm4_template["question"],
            "choices": tm4_template["choices"],
            "ground_truth": tm4_choice,
            "model_response": "",
            "model_response_text": TM4_CHOICES[tm4_choice],
            "metrics": {
                k: (round(v, 4) if isinstance(v, float) else v) for k, v in tm4_metrics.items()
            },
            "source_group_file": str(path.relative_to(formatted_dir)),
        }
        tm4_results.append(tm4_row)

        tm5_choice, tm5_metrics = infer_tm5_choice(
            selected_ann_token=selected_token,
            ann_by_token=ann_by_token,
            ego_pose_by_token=ego_pose_by_token,
            cam_front_sd_by_sample=cam_front_sd_by_sample,
            sample_by_token=sample_by_token,
            scene_by_token=scene_by_token,
            log_by_token=log_by_token,
            map_cache=map_cache,
            root=root,
            lane_width_m=args.lane_width_m,
        )
        tm5_row = {
            "scene_id": payload["scene_id"],
            "group_id": payload["group_id"],
            "question_id": "TM-5",
            "object_id": selected_token,
            "object_reference": object_ref,
            "question": tm5_template["question"].replace("<obj>", object_ref),
            "choices": tm5_template["choices"],
            "ground_truth": tm5_choice,
            "model_response": "",
            "model_response_text": TM5_CHOICES[tm5_choice],
            "metrics": {
                k: (round(v, 4) if isinstance(v, float) else v) for k, v in tm5_metrics.items()
            },
            "source_group_file": str(path.relative_to(formatted_dir)),
        }
        tm5_results.append(tm5_row)

        if frq_enabled and frq_llm is not None and frq_sampling_params is not None:
            sp_mcq_rows = [sp1_row, sp2_row, sp3_row, sp4_row, sp5_row, sp6_row]
            su_mcq_rows = [su1_row, su2_row, su3_row, su4_row, su5_row, su6_row]
            te_mcq_rows = [te1_row, te2_row, te3_row, te4_row, te5_row]
            tm_mcq_rows = [tm1_row, tm2_row, tm3_row, tm4_row, tm5_row]
            validate_mcq_rows_for_frq(sp_mcq_rows, "SP-7")
            validate_mcq_rows_for_frq(su_mcq_rows, "SU-7")
            validate_mcq_rows_for_frq(te_mcq_rows, "TE-6")
            validate_mcq_rows_for_frq(tm_mcq_rows, "TM-6")
            fifth_sample = sample_by_token.get(payload.get("fifth_frame_sample_token", ""), {})
            scene_token = fifth_sample.get("scene_token", "")
            scene_meta = scene_by_token.get(scene_token, {})
            scene_name = str(scene_meta.get("name", "")).strip()
            scene_description = str(scene_meta.get("description", "")).strip()
            vehicle_annotation_count = payload.get("vehicle_annotation_count")
            if not isinstance(vehicle_annotation_count, int):
                vehicle_annotation_count = None

            sp7_prompt = build_frq_prompt(
                sp7_template["question"],
                object_ref,
                sp_mcq_rows,
                scene_name=scene_name,
                scene_description=scene_description,
                group_scene_id=payload.get("scene_id", ""),
                vehicle_annotation_count=vehicle_annotation_count,
            )
            sp7_answer = generate_frq_response(frq_llm, frq_sampling_params, sp7_prompt)
            sp7_results.append(
                {
                    "scene_id": payload["scene_id"],
                    "group_id": payload["group_id"],
                    "question_id": "SP-7",
                    "object_id": selected_token,
                    "object_reference": object_ref,
                    "question": sp7_template["question"].replace("<obj>", object_ref),
                    "ground_truth": sp7_answer,
                    "model_response": "",
                    "source_group_file": str(path.relative_to(formatted_dir)),
                }
            )

            su7_prompt = build_frq_prompt(
                su7_template["question"],
                object_ref,
                su_mcq_rows,
                scene_name=scene_name,
                scene_description=scene_description,
                group_scene_id=payload.get("scene_id", ""),
                vehicle_annotation_count=vehicle_annotation_count,
            )
            su7_answer = generate_frq_response(frq_llm, frq_sampling_params, su7_prompt)
            su7_results.append(
                {
                    "scene_id": payload["scene_id"],
                    "group_id": payload["group_id"],
                    "question_id": "SU-7",
                    "object_id": selected_token,
                    "object_reference": object_ref,
                    "question": su7_template["question"].replace("<obj>", object_ref),
                    "ground_truth": su7_answer,
                    "model_response": "",
                    "source_group_file": str(path.relative_to(formatted_dir)),
                }
            )

            te6_prompt = build_frq_prompt(
                te6_template["question"],
                object_ref,
                te_mcq_rows,
                scene_name=scene_name,
                scene_description=scene_description,
                group_scene_id=payload.get("scene_id", ""),
                vehicle_annotation_count=vehicle_annotation_count,
            )
            te6_answer = generate_frq_response(frq_llm, frq_sampling_params, te6_prompt)
            te6_results.append(
                {
                    "scene_id": payload["scene_id"],
                    "group_id": payload["group_id"],
                    "question_id": "TE-6",
                    "object_id": selected_token,
                    "object_reference": object_ref,
                    "question": te6_template["question"].replace("<obj>", object_ref),
                    "ground_truth": te6_answer,
                    "model_response": "",
                    "source_group_file": str(path.relative_to(formatted_dir)),
                }
            )

            tm6_prompt = build_frq_prompt(
                tm6_template["question"],
                object_ref,
                tm_mcq_rows,
                scene_name=scene_name,
                scene_description=scene_description,
                group_scene_id=payload.get("scene_id", ""),
                vehicle_annotation_count=vehicle_annotation_count,
            )
            tm6_answer = generate_frq_response(frq_llm, frq_sampling_params, tm6_prompt)
            tm6_results.append(
                {
                    "scene_id": payload["scene_id"],
                    "group_id": payload["group_id"],
                    "question_id": "TM-6",
                    "object_id": selected_token,
                    "object_reference": object_ref,
                    "question": tm6_template["question"].replace("<obj>", object_ref),
                    "ground_truth": tm6_answer,
                    "model_response": "",
                    "source_group_file": str(path.relative_to(formatted_dir)),
                }
            )

    output = {
        "tasks": {
            "SP-1": {"count": len(sp1_results), "results": sp1_results},
            "SP-2": {"count": len(sp2_results), "results": sp2_results},
            "SP-3": {"count": len(sp3_results), "results": sp3_results},
            "SP-4": {"count": len(sp4_results), "results": sp4_results},
            "SP-5": {"count": len(sp5_results), "results": sp5_results},
            "SP-6": {"count": len(sp6_results), "results": sp6_results},
            "SU-1": {"count": len(su1_results), "results": su1_results},
            "SU-2": {"count": len(su2_results), "results": su2_results},
            "SU-3": {"count": len(su3_results), "results": su3_results},
            "SU-4": {"count": len(su4_results), "results": su4_results},
            "SU-5": {"count": len(su5_results), "results": su5_results},
            "SU-6": {"count": len(su6_results), "results": su6_results},
            "TE-1": {"count": len(te1_results), "results": te1_results},
            "TE-2": {"count": len(te2_results), "results": te2_results},
            "TE-3": {"count": len(te3_results), "results": te3_results},
            "TE-4": {"count": len(te4_results), "results": te4_results},
            "TE-5": {"count": len(te5_results), "results": te5_results},
            "TM-1": {"count": len(tm1_results), "results": tm1_results},
            "TM-2": {"count": len(tm2_results), "results": tm2_results},
            "TM-3": {"count": len(tm3_results), "results": tm3_results},
            "TM-4": {"count": len(tm4_results), "results": tm4_results},
            "TM-5": {"count": len(tm5_results), "results": tm5_results},
            "SP-7": {"count": len(sp7_results), "results": sp7_results},
            "SU-7": {"count": len(su7_results), "results": su7_results},
            "TE-6": {"count": len(te6_results), "results": te6_results},
            "TM-6": {"count": len(tm6_results), "results": tm6_results},
        }
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=True)

    # Build a separate questions file with generated answers populated.
    questions_with_answers = copy.deepcopy(questions)
    generated_sp1_tasks = []
    generated_sp2_tasks = []
    generated_sp3_tasks = []
    generated_sp4_tasks = []
    generated_sp5_tasks = []
    generated_sp6_tasks = []
    generated_su1_tasks = []
    generated_su2_tasks = []
    generated_su3_tasks = []
    generated_su4_tasks = []
    generated_su5_tasks = []
    generated_su6_tasks = []
    generated_te1_tasks = []
    generated_te2_tasks = []
    generated_te3_tasks = []
    generated_te4_tasks = []
    generated_te5_tasks = []
    generated_tm1_tasks = []
    generated_tm2_tasks = []
    generated_tm3_tasks = []
    generated_tm4_tasks = []
    generated_tm5_tasks = []
    generated_sp7_tasks = []
    generated_su7_tasks = []
    generated_te6_tasks = []
    generated_tm6_tasks = []
    for row in sp1_results:
        task = copy.deepcopy(sp1_template)
        task["object_id"] = row["object_id"]
        task["object_reference"] = row["object_reference"]
        task["ground_truth"] = row["ground_truth"]
        task["model_response"] = ""
        task["scene_id"] = row["scene_id"]
        task["group_id"] = row["group_id"]
        task["source_group_file"] = row["source_group_file"]
        generated_sp1_tasks.append(task)
    for row in sp2_results:
        task = copy.deepcopy(sp2_template)
        task["object_id"] = row["object_id"]
        task["object_reference"] = row["object_reference"]
        task["ground_truth"] = row["ground_truth"]
        task["model_response"] = ""
        task["scene_id"] = row["scene_id"]
        task["group_id"] = row["group_id"]
        task["source_group_file"] = row["source_group_file"]
        generated_sp2_tasks.append(task)
    for row in sp3_results:
        task = copy.deepcopy(sp3_template)
        task["object_id"] = row["object_id"]
        task["object_reference"] = row["object_reference"]
        task["ground_truth"] = row["ground_truth"]
        task["model_response"] = ""
        task["scene_id"] = row["scene_id"]
        task["group_id"] = row["group_id"]
        task["source_group_file"] = row["source_group_file"]
        generated_sp3_tasks.append(task)
    for row in sp4_results:
        task = copy.deepcopy(sp4_template)
        task["object_id"] = row["object_id"]
        task["object_reference"] = row["object_reference"]
        task["ground_truth"] = row["ground_truth"]
        task["model_response"] = ""
        task["scene_id"] = row["scene_id"]
        task["group_id"] = row["group_id"]
        task["source_group_file"] = row["source_group_file"]
        generated_sp4_tasks.append(task)
    for row in sp5_results:
        task = copy.deepcopy(sp5_template)
        task["object_id"] = row["object_id"]
        task["object_reference"] = row["object_reference"]
        task["ground_truth"] = row["ground_truth"]
        task["model_response"] = ""
        task["scene_id"] = row["scene_id"]
        task["group_id"] = row["group_id"]
        task["source_group_file"] = row["source_group_file"]
        generated_sp5_tasks.append(task)
    for row in sp6_results:
        task = copy.deepcopy(sp6_template)
        task["object_id"] = row["object_id"]
        task["object_reference"] = row["object_reference"]
        task["ground_truth"] = row["ground_truth"]
        task["model_response"] = ""
        task["scene_id"] = row["scene_id"]
        task["group_id"] = row["group_id"]
        task["source_group_file"] = row["source_group_file"]
        generated_sp6_tasks.append(task)
    for row in su1_results:
        task = copy.deepcopy(su1_template)
        task["object_id"] = row["object_id"]
        task["object_reference"] = row["object_reference"]
        task["ground_truth"] = row["ground_truth"]
        task["model_response"] = ""
        task["scene_id"] = row["scene_id"]
        task["group_id"] = row["group_id"]
        task["source_group_file"] = row["source_group_file"]
        generated_su1_tasks.append(task)
    for row in su2_results:
        task = copy.deepcopy(su2_template)
        task["object_id"] = row["object_id"]
        task["object_reference"] = row["object_reference"]
        task["ground_truth"] = row["ground_truth"]
        task["model_response"] = ""
        task["scene_id"] = row["scene_id"]
        task["group_id"] = row["group_id"]
        task["source_group_file"] = row["source_group_file"]
        generated_su2_tasks.append(task)
    for row in su3_results:
        task = copy.deepcopy(su3_template)
        task["object_id"] = row["object_id"]
        task["object_reference"] = row["object_reference"]
        task["ground_truth"] = row["ground_truth"]
        task["model_response"] = ""
        task["scene_id"] = row["scene_id"]
        task["group_id"] = row["group_id"]
        task["source_group_file"] = row["source_group_file"]
        generated_su3_tasks.append(task)
    for row in su4_results:
        task = copy.deepcopy(su4_template)
        task["object_id"] = row["object_id"]
        task["object_reference"] = row["object_reference"]
        task["ground_truth"] = row["ground_truth"]
        task["model_response"] = ""
        task["scene_id"] = row["scene_id"]
        task["group_id"] = row["group_id"]
        task["source_group_file"] = row["source_group_file"]
        generated_su4_tasks.append(task)
    for row in su5_results:
        task = copy.deepcopy(su5_template)
        task["object_id"] = row["object_id"]
        task["object_reference"] = row["object_reference"]
        task["ground_truth"] = row["ground_truth"]
        task["model_response"] = ""
        task["scene_id"] = row["scene_id"]
        task["group_id"] = row["group_id"]
        task["source_group_file"] = row["source_group_file"]
        generated_su5_tasks.append(task)
    for row in su6_results:
        task = copy.deepcopy(su6_template)
        task["object_id"] = row["object_id"]
        task["object_reference"] = row["object_reference"]
        task["ground_truth"] = row["ground_truth"]
        task["model_response"] = ""
        task["scene_id"] = row["scene_id"]
        task["group_id"] = row["group_id"]
        task["source_group_file"] = row["source_group_file"]
        generated_su6_tasks.append(task)
    for row in te1_results:
        task = copy.deepcopy(te1_template)
        task["object_id"] = row["object_id"]
        task["object_reference"] = row["object_reference"]
        task["ground_truth"] = row["ground_truth"]
        task["model_response"] = ""
        task["scene_id"] = row["scene_id"]
        task["group_id"] = row["group_id"]
        task["source_group_file"] = row["source_group_file"]
        generated_te1_tasks.append(task)
    for row in te2_results:
        task = copy.deepcopy(te2_template)
        task["object_id"] = row["object_id"]
        task["object_reference"] = row["object_reference"]
        task["ground_truth"] = row["ground_truth"]
        task["model_response"] = ""
        task["scene_id"] = row["scene_id"]
        task["group_id"] = row["group_id"]
        task["source_group_file"] = row["source_group_file"]
        generated_te2_tasks.append(task)
    for row in te3_results:
        task = copy.deepcopy(te3_template)
        task["object_id"] = row["object_id"]
        task["object_reference"] = row["object_reference"]
        task["ground_truth"] = row["ground_truth"]
        task["model_response"] = ""
        task["scene_id"] = row["scene_id"]
        task["group_id"] = row["group_id"]
        task["source_group_file"] = row["source_group_file"]
        generated_te3_tasks.append(task)
    for row in te4_results:
        task = copy.deepcopy(te4_template)
        task["object_id"] = row["object_id"]
        task["object_reference"] = row["object_reference"]
        task["ground_truth"] = row["ground_truth"]
        task["model_response"] = ""
        task["scene_id"] = row["scene_id"]
        task["group_id"] = row["group_id"]
        task["source_group_file"] = row["source_group_file"]
        generated_te4_tasks.append(task)
    for row in te5_results:
        task = copy.deepcopy(te5_template)
        task["object_id"] = row["object_id"]
        task["object_reference"] = row["object_reference"]
        task["ground_truth"] = row["ground_truth"]
        task["model_response"] = ""
        task["scene_id"] = row["scene_id"]
        task["group_id"] = row["group_id"]
        task["source_group_file"] = row["source_group_file"]
        generated_te5_tasks.append(task)
    for row in tm1_results:
        task = copy.deepcopy(tm1_template)
        task["object_id"] = row["object_id"]
        task["object_reference"] = row["object_reference"]
        task["ground_truth"] = row["ground_truth"]
        task["model_response"] = ""
        task["scene_id"] = row["scene_id"]
        task["group_id"] = row["group_id"]
        task["source_group_file"] = row["source_group_file"]
        generated_tm1_tasks.append(task)
    for row in tm2_results:
        task = copy.deepcopy(tm2_template)
        task["object_id"] = row["object_id"]
        task["object_reference"] = row["object_reference"]
        task["ground_truth"] = row["ground_truth"]
        task["model_response"] = ""
        task["scene_id"] = row["scene_id"]
        task["group_id"] = row["group_id"]
        task["source_group_file"] = row["source_group_file"]
        generated_tm2_tasks.append(task)
    for row in tm3_results:
        task = copy.deepcopy(tm3_template)
        task["object_id"] = row["object_id"]
        task["object_reference"] = row["object_reference"]
        task["ground_truth"] = row["ground_truth"]
        task["model_response"] = ""
        task["scene_id"] = row["scene_id"]
        task["group_id"] = row["group_id"]
        task["source_group_file"] = row["source_group_file"]
        generated_tm3_tasks.append(task)
    for row in tm4_results:
        task = copy.deepcopy(tm4_template)
        task["object_id"] = row["object_id"]
        task["object_reference"] = row["object_reference"]
        task["ground_truth"] = row["ground_truth"]
        task["model_response"] = ""
        task["scene_id"] = row["scene_id"]
        task["group_id"] = row["group_id"]
        task["source_group_file"] = row["source_group_file"]
        generated_tm4_tasks.append(task)
    for row in tm5_results:
        task = copy.deepcopy(tm5_template)
        task["object_id"] = row["object_id"]
        task["object_reference"] = row["object_reference"]
        task["ground_truth"] = row["ground_truth"]
        task["model_response"] = ""
        task["scene_id"] = row["scene_id"]
        task["group_id"] = row["group_id"]
        task["source_group_file"] = row["source_group_file"]
        generated_tm5_tasks.append(task)
    for row in sp7_results:
        task = copy.deepcopy(sp7_template)
        task["object_id"] = row["object_id"]
        task["object_reference"] = row["object_reference"]
        task["ground_truth"] = row["ground_truth"]
        task["model_response"] = ""
        task["scene_id"] = row["scene_id"]
        task["group_id"] = row["group_id"]
        task["source_group_file"] = row["source_group_file"]
        generated_sp7_tasks.append(task)
    for row in su7_results:
        task = copy.deepcopy(su7_template)
        task["object_id"] = row["object_id"]
        task["object_reference"] = row["object_reference"]
        task["ground_truth"] = row["ground_truth"]
        task["model_response"] = ""
        task["scene_id"] = row["scene_id"]
        task["group_id"] = row["group_id"]
        task["source_group_file"] = row["source_group_file"]
        generated_su7_tasks.append(task)
    for row in te6_results:
        task = copy.deepcopy(te6_template)
        task["object_id"] = row["object_id"]
        task["object_reference"] = row["object_reference"]
        task["ground_truth"] = row["ground_truth"]
        task["model_response"] = ""
        task["scene_id"] = row["scene_id"]
        task["group_id"] = row["group_id"]
        task["source_group_file"] = row["source_group_file"]
        generated_te6_tasks.append(task)
    for row in tm6_results:
        task = copy.deepcopy(tm6_template)
        task["object_id"] = row["object_id"]
        task["object_reference"] = row["object_reference"]
        task["ground_truth"] = row["ground_truth"]
        task["model_response"] = ""
        task["scene_id"] = row["scene_id"]
        task["group_id"] = row["group_id"]
        task["source_group_file"] = row["source_group_file"]
        generated_tm6_tasks.append(task)

    questions_with_answers["generated_answers"] = {
        "SP-1": {
            "count": len(generated_sp1_tasks),
            "tasks": generated_sp1_tasks,
        },
        "SP-2": {
            "count": len(generated_sp2_tasks),
            "tasks": generated_sp2_tasks,
        },
        "SP-3": {
            "count": len(generated_sp3_tasks),
            "tasks": generated_sp3_tasks,
        },
        "SP-4": {
            "count": len(generated_sp4_tasks),
            "tasks": generated_sp4_tasks,
        },
        "SP-5": {
            "count": len(generated_sp5_tasks),
            "tasks": generated_sp5_tasks,
        },
        "SP-6": {
            "count": len(generated_sp6_tasks),
            "tasks": generated_sp6_tasks,
        },
        "SP-7": {
            "count": len(generated_sp7_tasks),
            "tasks": generated_sp7_tasks,
        },
        "SU-1": {
            "count": len(generated_su1_tasks),
            "tasks": generated_su1_tasks,
        },
        "SU-2": {
            "count": len(generated_su2_tasks),
            "tasks": generated_su2_tasks,
        },
        "SU-3": {
            "count": len(generated_su3_tasks),
            "tasks": generated_su3_tasks,
        },
        "SU-4": {
            "count": len(generated_su4_tasks),
            "tasks": generated_su4_tasks,
        },
        "SU-5": {
            "count": len(generated_su5_tasks),
            "tasks": generated_su5_tasks,
        },
        "SU-6": {
            "count": len(generated_su6_tasks),
            "tasks": generated_su6_tasks,
        },
        "SU-7": {
            "count": len(generated_su7_tasks),
            "tasks": generated_su7_tasks,
        },
        "TE-1": {
            "count": len(generated_te1_tasks),
            "tasks": generated_te1_tasks,
        },
        "TE-2": {
            "count": len(generated_te2_tasks),
            "tasks": generated_te2_tasks,
        },
        "TE-3": {
            "count": len(generated_te3_tasks),
            "tasks": generated_te3_tasks,
        },
        "TE-4": {
            "count": len(generated_te4_tasks),
            "tasks": generated_te4_tasks,
        },
        "TE-5": {
            "count": len(generated_te5_tasks),
            "tasks": generated_te5_tasks,
        },
        "TE-6": {
            "count": len(generated_te6_tasks),
            "tasks": generated_te6_tasks,
        },
        "TM-1": {
            "count": len(generated_tm1_tasks),
            "tasks": generated_tm1_tasks,
        },
        "TM-2": {
            "count": len(generated_tm2_tasks),
            "tasks": generated_tm2_tasks,
        },
        "TM-3": {
            "count": len(generated_tm3_tasks),
            "tasks": generated_tm3_tasks,
        },
        "TM-4": {
            "count": len(generated_tm4_tasks),
            "tasks": generated_tm4_tasks,
        },
        "TM-5": {
            "count": len(generated_tm5_tasks),
            "tasks": generated_tm5_tasks,
        },
        "TM-6": {
            "count": len(generated_tm6_tasks),
            "tasks": generated_tm6_tasks,
        },
    }
    with questions_output.open("w", encoding="utf-8") as f:
        json.dump(questions_with_answers, f, indent=2, ensure_ascii=True)

    print(f"Processed group files: {len(group_json_paths)}")
    print(f"Generated SP-1 answers: {len(sp1_results)}")
    print(f"Generated SP-2 answers: {len(sp2_results)}")
    print(f"Generated SP-3 answers: {len(sp3_results)}")
    print(f"Generated SP-4 answers: {len(sp4_results)}")
    print(f"Generated SP-5 answers: {len(sp5_results)}")
    print(f"Generated SP-6 answers: {len(sp6_results)}")
    print(f"Generated SP-7 answers: {len(sp7_results)}")
    print(f"Generated SU-1 answers: {len(su1_results)}")
    print(f"Generated SU-2 answers: {len(su2_results)}")
    print(f"Generated SU-3 answers: {len(su3_results)}")
    print(f"Generated SU-4 answers: {len(su4_results)}")
    print(f"Generated SU-5 answers: {len(su5_results)}")
    print(f"Generated SU-6 answers: {len(su6_results)}")
    print(f"Generated SU-7 answers: {len(su7_results)}")
    print(f"Generated TE-1 answers: {len(te1_results)}")
    print(f"Generated TE-2 answers: {len(te2_results)}")
    print(f"Generated TE-3 answers: {len(te3_results)}")
    print(f"Generated TE-4 answers: {len(te4_results)}")
    print(f"Generated TE-5 answers: {len(te5_results)}")
    print(f"Generated TE-6 answers: {len(te6_results)}")
    print(f"Generated TM-1 answers: {len(tm1_results)}")
    print(f"Generated TM-2 answers: {len(tm2_results)}")
    print(f"Generated TM-3 answers: {len(tm3_results)}")
    print(f"Generated TM-4 answers: {len(tm4_results)}")
    print(f"Generated TM-5 answers: {len(tm5_results)}")
    print(f"Generated TM-6 answers: {len(tm6_results)}")
    print(f"Wrote: {output_json}")
    print(f"Wrote questions copy: {questions_output}")


if __name__ == "__main__":
    main()
