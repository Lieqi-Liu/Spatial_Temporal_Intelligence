#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Tuple


TRAJ1_CHOICES = {
    "A": "Go straight",
    "B": "Slightly veer left",
    "C": "Slightly veer right",
    "D": "Turn left",
    "E": "Turn right",
    "F": "Slow to near stop",
}

TRAJ2_CHOICES = {
    "A": "Front-left region",
    "B": "Front-center region",
    "C": "Front-right region",
    "D": "Left-side region",
    "E": "Right-side region",
    "F": "Near the current position",
}

TRAJ3_CHOICES = {
    "A": "Accelerating",
    "B": "Maintaining roughly constant speed",
    "C": "Decelerating",
    "D": "Stopping / nearly stopped",
}


def get_traj_prediction_task_templates() -> List[dict]:
    return [
        {
            "id": "TRJ-1",
            "task": "trajectory-prediction",
            "question_format": "MCQ",
            "question": (
                "Based on the past 5 consecutive 360-degree multi-camera frames, what is the most likely "
                "ego-vehicle maneuver over the next 5 future frames (approximately 2.5 seconds)?"
            ),
            "choices": TRAJ1_CHOICES,
            "ground_truth": "",
            "model_response": "",
        },
        {
            "id": "TRJ-2",
            "task": "trajectory-prediction",
            "question_format": "MCQ",
            "question": (
                "Based on the past 5 consecutive 360-degree multi-camera frames, at the 5th future frame "
                "(approximately 2.5 seconds ahead), which relative endpoint region best matches the ego "
                "vehicle trajectory in the current ego-centric coordinate system?"
            ),
            "choices": TRAJ2_CHOICES,
            "ground_truth": "",
            "model_response": "",
        },
        {
            "id": "TRJ-3",
            "task": "trajectory-prediction",
            "question_format": "MCQ",
            "question": (
                "Based on the past 5 consecutive 360-degree multi-camera frames, how does the ego vehicle "
                "speed trend evolve over the next 5 future frames (approximately 2.5 seconds)?"
            ),
            "choices": TRAJ3_CHOICES,
            "ground_truth": "",
            "model_response": "",
        },
        {
            "id": "TRJ-4",
            "task": "trajectory-prediction",
            "question_format": "FRQ",
            "question": (
                "Based on the past 5 consecutive 360-degree multi-camera frames, predict the ego vehicle "
                "trajectory for the next 5 future frames (approximately 2.5 seconds) in the current ego-centric "
                "coordinate system. Output only a JSON array of exactly 5 points formatted as "
                "[[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5]]."
            ),
            "ground_truth": "",
            "model_response": "",
        },
        {
            "id": "TRJ-5",
            "task": "trajectory-prediction",
            "question_format": "FRQ",
            "question": (
                "Based on the past 5 consecutive 360-degree multi-camera frames, the ego vehicle position at the "
                "next future frame is already known in the current ego-centric coordinate system as [x1, y1]. "
                "Given that first future point, predict the trajectory for the following 4 future frames. "
                "Output only a JSON array of exactly 4 points formatted as "
                "[[x2, y2], [x3, y3], [x4, y4], [x5, y5]]."
            ),
            "ground_truth": "",
            "model_response": "",
        },
        {
            "id": "TRJ-6",
            "task": "trajectory-prediction",
            "question_format": "FRQ",
            "question": (
                "Based on the past 5 consecutive 360-degree multi-camera frames, briefly describe the ego "
                "vehicle's likely future path over the next 5 future frames (approximately 2.5 seconds). "
                "Mention the overall maneuver, endpoint region, and speed trend in 1-2 sentences."
            ),
            "ground_truth": "",
            "model_response": "",
        },
    ]


def yaw_from_quaternion_wxyz(q: List[float]) -> float:
    if len(q) != 4:
        return 0.0
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def global_to_local_xy(dx: float, dy: float, yaw: float) -> Tuple[float, float]:
    c = math.cos(yaw)
    s = math.sin(yaw)
    x_local = c * dx + s * dy
    y_local = -s * dx + c * dy
    return x_local, y_local


def _ego_pose_for_sample(
    sample_token: str,
    ego_pose_by_token: Dict[str, dict],
    cam_front_sd_by_sample: Dict[str, dict],
) -> dict | None:
    sd = cam_front_sd_by_sample.get(sample_token)
    if sd is None:
        return None
    return ego_pose_by_token.get(sd["ego_pose_token"])


def _collect_future_sample_tokens(
    current_sample_token: str,
    sample_by_token: Dict[str, dict],
    future_steps: int,
) -> List[str]:
    tokens: List[str] = []
    cur = current_sample_token
    for _ in range(future_steps):
        nxt = sample_by_token.get(cur, {}).get("next")
        if not nxt:
            break
        tokens.append(nxt)
        cur = nxt
    return tokens


def _compute_speed_mps(
    sample_a: dict,
    sample_b: dict,
    ego_pose_a: dict,
    ego_pose_b: dict,
) -> float:
    dt = max((float(sample_b["timestamp"]) - float(sample_a["timestamp"])) * 1e-6, 1e-6)
    dx = float(ego_pose_b["translation"][0]) - float(ego_pose_a["translation"][0])
    dy = float(ego_pose_b["translation"][1]) - float(ego_pose_a["translation"][1])
    return math.hypot(dx, dy) / dt


def _infer_trj1_choice(endpoint_x: float, endpoint_y: float, endpoint_dist: float) -> Tuple[str, str]:
    if endpoint_dist < 1.0 or endpoint_x < 1.0:
        return "F", "near_stop"
    angle_deg = math.degrees(math.atan2(endpoint_y, endpoint_x))
    if angle_deg >= 35.0:
        return "D", "strong_left_turn"
    if angle_deg >= 10.0:
        return "B", "slight_left"
    if angle_deg <= -35.0:
        return "E", "strong_right_turn"
    if angle_deg <= -10.0:
        return "C", "slight_right"
    return "A", "straight"


def _infer_trj2_choice(endpoint_x: float, endpoint_y: float, endpoint_dist: float) -> Tuple[str, str]:
    if endpoint_dist < 1.0:
        return "F", "near_current_position"
    angle_deg = math.degrees(math.atan2(endpoint_y, endpoint_x))
    if 20.0 <= angle_deg <= 70.0:
        return "A", "front_left"
    if -20.0 < angle_deg < 20.0:
        return "B", "front_center"
    if -70.0 <= angle_deg <= -20.0:
        return "C", "front_right"
    if angle_deg > 70.0:
        return "D", "left_side"
    return "E", "right_side"


def _infer_trj3_choice(current_speed: float, future_speed: float, endpoint_dist: float) -> Tuple[str, str]:
    if endpoint_dist < 1.0 or future_speed < 0.5:
        return "D", "near_stop"
    delta = future_speed - current_speed
    if delta > 0.75:
        return "A", "speed_increasing"
    if delta < -0.75:
        return "C", "speed_decreasing"
    return "B", "speed_stable"


def generate_traj_prediction_rows(
    *,
    payload: dict,
    sample_by_token: Dict[str, dict],
    ego_pose_by_token: Dict[str, dict],
    cam_front_sd_by_sample: Dict[str, dict],
    future_steps: int = 5,
) -> Dict[str, dict] | None:
    current_sample_token = str(payload.get("fifth_frame_sample_token", "")).strip()
    if not current_sample_token or current_sample_token not in sample_by_token:
        return None

    future_sample_tokens = _collect_future_sample_tokens(
        current_sample_token=current_sample_token,
        sample_by_token=sample_by_token,
        future_steps=future_steps,
    )
    if len(future_sample_tokens) < future_steps:
        return None

    current_sample = sample_by_token[current_sample_token]
    current_ego = _ego_pose_for_sample(
        sample_token=current_sample_token,
        ego_pose_by_token=ego_pose_by_token,
        cam_front_sd_by_sample=cam_front_sd_by_sample,
    )
    if current_ego is None:
        return None

    current_yaw = yaw_from_quaternion_wxyz(current_ego["rotation"])
    current_xy = current_ego["translation"]

    future_local_points: List[List[float]] = []
    future_global_points: List[List[float]] = []
    future_samples: List[dict] = []
    future_egos: List[dict] = []
    for token in future_sample_tokens:
        sample_row = sample_by_token[token]
        ego_row = _ego_pose_for_sample(
            sample_token=token,
            ego_pose_by_token=ego_pose_by_token,
            cam_front_sd_by_sample=cam_front_sd_by_sample,
        )
        if ego_row is None:
            return None
        dx = float(ego_row["translation"][0]) - float(current_xy[0])
        dy = float(ego_row["translation"][1]) - float(current_xy[1])
        x_local, y_local = global_to_local_xy(dx, dy, current_yaw)
        future_local_points.append([round(x_local, 3), round(y_local, 3)])
        future_global_points.append(
            [round(float(ego_row["translation"][0]), 3), round(float(ego_row["translation"][1]), 3)]
        )
        future_samples.append(sample_row)
        future_egos.append(ego_row)

    endpoint_x, endpoint_y = future_local_points[-1]
    endpoint_dist = math.hypot(endpoint_x, endpoint_y)

    prev_token = current_sample.get("prev")
    if prev_token and prev_token in sample_by_token:
        prev_sample = sample_by_token[prev_token]
        prev_ego = _ego_pose_for_sample(
            sample_token=prev_token,
            ego_pose_by_token=ego_pose_by_token,
            cam_front_sd_by_sample=cam_front_sd_by_sample,
        )
        if prev_ego is not None:
            current_speed = _compute_speed_mps(prev_sample, current_sample, prev_ego, current_ego)
        else:
            current_speed = _compute_speed_mps(current_sample, future_samples[0], current_ego, future_egos[0])
    else:
        current_speed = _compute_speed_mps(current_sample, future_samples[0], current_ego, future_egos[0])

    future_speed = _compute_speed_mps(
        future_samples[-2],
        future_samples[-1],
        future_egos[-2],
        future_egos[-1],
    )

    trj1_choice, trj1_reason = _infer_trj1_choice(endpoint_x, endpoint_y, endpoint_dist)
    trj2_choice, trj2_reason = _infer_trj2_choice(endpoint_x, endpoint_y, endpoint_dist)
    trj3_choice, trj3_reason = _infer_trj3_choice(current_speed, future_speed, endpoint_dist)
    traj_summary = (
        f"The ego vehicle is most likely to {TRAJ1_CHOICES[trj1_choice].lower()} over the next 5 future "
        f"frames, ending in the {TRAJ2_CHOICES[trj2_choice].lower()} with a "
        f"{TRAJ3_CHOICES[trj3_choice].lower()}."
    )

    common = {
        "scene_id": payload["scene_id"],
        "group_id": payload["group_id"],
        "source_group_file": payload["source_group_file"],
        "future_steps": future_steps,
        "future_sample_tokens": future_sample_tokens,
        "future_traj_local": future_local_points,
        "future_traj_global": future_global_points,
        "metrics": {
            "endpoint_x": round(endpoint_x, 3),
            "endpoint_y": round(endpoint_y, 3),
            "endpoint_distance": round(endpoint_dist, 3),
            "current_speed_mps": round(current_speed, 3),
            "future_speed_mps": round(future_speed, 3),
        },
    }

    return {
        "TRJ-1": {
            **common,
            "question_id": "TRJ-1",
            "question": (
                "Based on the past 5 consecutive 360-degree multi-camera frames, what is the most likely "
                "ego-vehicle maneuver over the next 5 future frames (approximately 2.5 seconds)?"
            ),
            "choices": TRAJ1_CHOICES,
            "ground_truth": trj1_choice,
            "model_response": "",
            "model_response_text": TRAJ1_CHOICES[trj1_choice],
            "metrics": {**common["metrics"], "reason": trj1_reason},
        },
        "TRJ-2": {
            **common,
            "question_id": "TRJ-2",
            "question": (
                "Based on the past 5 consecutive 360-degree multi-camera frames, at the 5th future frame "
                "(approximately 2.5 seconds ahead), which relative endpoint region best matches the ego "
                "vehicle trajectory in the current ego-centric coordinate system?"
            ),
            "choices": TRAJ2_CHOICES,
            "ground_truth": trj2_choice,
            "model_response": "",
            "model_response_text": TRAJ2_CHOICES[trj2_choice],
            "metrics": {**common["metrics"], "reason": trj2_reason},
        },
        "TRJ-3": {
            **common,
            "question_id": "TRJ-3",
            "question": (
                "Based on the past 5 consecutive 360-degree multi-camera frames, how does the ego vehicle "
                "speed trend evolve over the next 5 future frames (approximately 2.5 seconds)?"
            ),
            "choices": TRAJ3_CHOICES,
            "ground_truth": trj3_choice,
            "model_response": "",
            "model_response_text": TRAJ3_CHOICES[trj3_choice],
            "metrics": {**common["metrics"], "reason": trj3_reason},
        },
        "TRJ-4": {
            **common,
            "question_id": "TRJ-4",
            "question": (
                "Based on the past 5 consecutive 360-degree multi-camera frames, predict the ego vehicle "
                "trajectory for the next 5 future frames (approximately 2.5 seconds) in the current ego-centric "
                "coordinate system. Output only a JSON array of exactly 5 points formatted as "
                "[[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5]]."
            ),
            "ground_truth": json.dumps(future_local_points, ensure_ascii=True),
            "model_response": "",
        },
        "TRJ-5": {
            **common,
            "question_id": "TRJ-5",
            "question": (
                "Based on the past 5 consecutive 360-degree multi-camera frames, the ego vehicle position at the "
                "next future frame is already known in the current ego-centric coordinate system as "
                f"{json.dumps(future_local_points[0], ensure_ascii=True)}. "
                "Given that first future point, predict the trajectory for the following 4 future frames. "
                "Output only a JSON array of exactly 4 points formatted as "
                "[[x2, y2], [x3, y3], [x4, y4], [x5, y5]]."
            ),
            "known_future_point_local": future_local_points[0],
            "ground_truth": json.dumps(future_local_points[1:], ensure_ascii=True),
            "model_response": "",
        },
        "TRJ-6": {
            **common,
            "question_id": "TRJ-6",
            "question": (
                "Based on the past 5 consecutive 360-degree multi-camera frames, briefly describe the ego "
                "vehicle's likely future path over the next 5 future frames (approximately 2.5 seconds). "
                "Mention the overall maneuver, endpoint region, and speed trend in 1-2 sentences."
            ),
            "ground_truth": traj_summary,
            "model_response": "",
        },
    }
