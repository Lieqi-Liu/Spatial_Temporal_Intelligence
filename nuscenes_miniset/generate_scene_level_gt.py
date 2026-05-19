#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


WEATHER_CHOICES = {
    "A": "Clear / Sunny",
    "B": "Rain / Wet road surface",
    "C": "Overcast / Cloudy",
    "D": "Snow / Ice",
    "E": "Fog / Heavy haze",
}

TIME_CHOICES = {
    "A": "Daytime (adequate natural lighting)",
    "B": "Nighttime (relying on streetlights/headlights)",
    "C": "Dawn / Dusk (transition lighting, potential glare)",
}

ROAD_CHOICES = {
    "A": "Urban intersection / Crossroad",
    "B": "Straight urban/suburban road",
    "C": "Highway / Expressway",
    "D": "Construction zone",
    "E": "Parking lot / Off-road area",
}

TRAFFIC_CHOICES = {
    "A": "Sparse",
    "B": "Moderate",
    "C": "Dense",
}

RISK_CHOICES = {
    "A": "Low risk",
    "B": "Moderate risk",
    "C": "High risk",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate scene-level GT tasks for nuScenes mini."
    )
    parser.add_argument(
        "--version-dir",
        type=Path,
        default=Path(
            "/home/rgao727/Spatial_Temporal_Intelligence/data/nuscenes-v1.0-mini/v1.0-mini"
        ),
        help="Path to the nuScenes version directory containing scene.json, sample.json, etc.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/home/rgao727/Spatial_Temporal_Intelligence/nuscenes_test/scene_level_tasks.json"
        ),
        help="Destination JSON file for generated scene-level tasks.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def classify_weather(description: str) -> str:
    desc = description.lower()
    if contains_any(desc, ("snow", "ice", "icy")):
        return "D"
    if contains_any(desc, ("fog", "foggy", "haze", "hazy", "mist")):
        return "E"
    if contains_any(desc, ("rain", "rainy", "wet", "after rain")):
        return "B"
    if contains_any(desc, ("overcast", "cloudy", "cloud")):
        return "C"
    return "A"


def classify_time_of_day(description: str) -> str:
    desc = description.lower()
    if contains_any(desc, ("night", "dark", "difficult lighting")):
        return "B"
    if contains_any(desc, ("dawn", "dusk", "sunrise", "sunset", "twilight")):
        return "C"
    return "A"


def classify_road_type(description: str) -> str:
    desc = description.lower()
    if "parking lot" in desc:
        return "E"
    if contains_any(desc, ("construction", "traffic cone", "barrier", "lane shift")):
        return "D"
    if contains_any(desc, ("intersection", "cross intersection", "crosswalk", "turn left", "turn right")):
        return "A"
    if contains_any(desc, ("high speed", "highway", "expressway", "merge")):
        return "C"
    return "B"


def classify_density(avg_road_user_count: float) -> str:
    if avg_road_user_count <= 5.0:
        return "A"
    if avg_road_user_count <= 12.0:
        return "B"
    return "C"


def classify_risk(
    *,
    weather_code: str,
    time_code: str,
    road_code: str,
    density_code: str,
    pedestrian_avg: float,
    cyclist_avg: float,
) -> str:
    score = 0
    if time_code == "B":
        score += 1
    elif time_code == "C":
        score += 1

    if weather_code == "B":
        score += 1
    elif weather_code in {"D", "E"}:
        score += 2

    if road_code == "A":
        score += 1
    elif road_code == "D":
        score += 2
    elif road_code == "E":
        score += 1

    if density_code == "B":
        score += 1
    elif density_code == "C":
        score += 2

    if pedestrian_avg >= 2.5:
        score += 1
    if cyclist_avg >= 1.0:
        score += 1

    if score <= 1:
        return "A"
    if score <= 4:
        return "B"
    return "C"


def scene_sample_tokens(scene_row: dict[str, Any], sample_by_token: dict[str, dict[str, Any]]) -> list[str]:
    tokens: list[str] = []
    token = scene_row["first_sample_token"]
    while token:
        tokens.append(token)
        token = sample_by_token[token]["next"]
    return tokens


def pick_representative_tokens(tokens: list[str]) -> list[str]:
    if not tokens:
        return []
    if len(tokens) <= 3:
        return tokens

    picks = {
        0,
        len(tokens) // 2,
        len(tokens) - 1,
    }
    return [tokens[idx] for idx in sorted(picks)]


def mean(values: list[int]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def compute_scene_stats(
    scene_row: dict[str, Any],
    *,
    sample_by_token: dict[str, dict[str, Any]],
    annotations_by_sample: dict[str, list[dict[str, Any]]],
    category_name_by_instance: dict[str, str],
    ego_xy_by_sample: dict[str, tuple[float, float]],
) -> dict[str, float]:
    tokens = scene_sample_tokens(scene_row, sample_by_token)
    representative = pick_representative_tokens(tokens)

    road_user_counts: list[int] = []
    pedestrian_counts: list[int] = []
    cyclist_counts: list[int] = []
    vehicle_counts: list[int] = []

    for sample_token in representative:
        road_users = 0
        pedestrians = 0
        cyclists = 0
        vehicles = 0

        for ann in annotations_by_sample.get(sample_token, []):
            if (ann.get("num_lidar_pts", 0) + ann.get("num_radar_pts", 0)) <= 0:
                continue

            ego_xy = ego_xy_by_sample.get(sample_token)
            if ego_xy is not None:
                dx = float(ann["translation"][0]) - ego_xy[0]
                dy = float(ann["translation"][1]) - ego_xy[1]
                if math.hypot(dx, dy) > 40.0:
                    continue

            category_name = category_name_by_instance.get(ann["instance_token"], "")
            if category_name.startswith("human.pedestrian"):
                pedestrians += 1
                road_users += 1
            elif category_name == "vehicle.bicycle":
                cyclists += 1
                road_users += 1
            elif category_name.startswith("vehicle.") and not category_name.startswith("vehicle.emergency"):
                vehicles += 1
                road_users += 1
            elif category_name.startswith("vehicle.emergency"):
                vehicles += 1
                road_users += 1

        road_user_counts.append(road_users)
        pedestrian_counts.append(pedestrians)
        cyclist_counts.append(cyclists)
        vehicle_counts.append(vehicles)

    return {
        "avg_road_users": mean(road_user_counts),
        "avg_pedestrians": mean(pedestrian_counts),
        "avg_cyclists": mean(cyclist_counts),
        "avg_vehicles": mean(vehicle_counts),
        "representative_frame_count": float(len(representative)),
    }


def build_notable_activity(
    *,
    description: str,
    road_code: str,
    density_code: str,
    avg_pedestrians: float,
    avg_cyclists: float,
    avg_vehicles: float,
) -> str:
    snippets: list[str] = []
    desc = description.strip()
    if desc:
        snippets.append(desc[0].lower() + desc[1:])

    if road_code == "A":
        snippets.append("intersection-related interactions")
    elif road_code == "D":
        snippets.append("construction-related roadside activity")
    elif road_code == "E":
        snippets.append("parking-lot traffic and parked vehicles")

    if avg_pedestrians >= 2.5:
        snippets.append("multiple pedestrians near the roadway")
    if avg_cyclists >= 1.0:
        snippets.append("bicycles or scooters sharing space with traffic")
    if avg_vehicles >= 6.0:
        snippets.append("steady multi-vehicle traffic flow")

    if density_code == "A" and avg_vehicles < 3.0 and avg_pedestrians < 1.5:
        snippets.append("relatively light surrounding activity")

    if not snippets:
        snippets.append("typical urban driving activity")

    seen: set[str] = set()
    deduped: list[str] = []
    for snippet in snippets:
        if snippet not in seen:
            deduped.append(snippet)
            seen.add(snippet)
    return "; ".join(deduped)


def build_open_ended_answer(
    *,
    weather_code: str,
    time_code: str,
    road_code: str,
    density_code: str,
    risk_code: str,
    notable_activity: str,
) -> str:
    return (
        f"The scene takes place during {TIME_CHOICES[time_code].lower()} under "
        f"{WEATHER_CHOICES[weather_code].lower()} conditions. It is set in a "
        f"{ROAD_CHOICES[road_code].lower()} environment with {TRAFFIC_CHOICES[density_code].lower()} traffic. "
        f"The baseline scene risk is {RISK_CHOICES[risk_code].lower()}, and notable surrounding activity includes "
        f"{notable_activity}."
    )


def make_mcq_task(
    *,
    scene_name: str,
    task_id: str,
    question: str,
    choices: dict[str, str],
    ground_truth: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "scene_id": scene_name,
        "id": task_id,
        "task": "scene-context",
        "question_format": "MCQ",
        "question": question,
        "choices": choices,
        "ground_truth": ground_truth,
        "model_response": "",
    }
    row.update(metadata)
    return row


def make_frq_task(
    *,
    scene_name: str,
    task_id: str,
    question: str,
    ground_truth: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "scene_id": scene_name,
        "id": task_id,
        "task": "scene-context",
        "question_format": "FRQ",
        "question": question,
        "ground_truth": ground_truth,
        "model_response": "",
    }
    row.update(metadata)
    return row


def main() -> None:
    args = parse_args()
    version_dir = args.version_dir

    scene_rows = load_json(version_dir / "scene.json")
    sample_rows = load_json(version_dir / "sample.json")
    sample_data_rows = load_json(version_dir / "sample_data.json")
    annotation_rows = load_json(version_dir / "sample_annotation.json")
    instance_rows = load_json(version_dir / "instance.json")
    category_rows = load_json(version_dir / "category.json")
    ego_pose_rows = load_json(version_dir / "ego_pose.json")
    calibrated_sensor_rows = load_json(version_dir / "calibrated_sensor.json")
    sensor_rows = load_json(version_dir / "sensor.json")
    log_rows = load_json(version_dir / "log.json")

    sample_by_token = {row["token"]: row for row in sample_rows}
    annotations_by_sample: dict[str, list[dict[str, Any]]] = {}
    for row in annotation_rows:
        annotations_by_sample.setdefault(row["sample_token"], []).append(row)

    category_name_by_token = {row["token"]: row["name"] for row in category_rows}
    category_name_by_instance = {
        row["token"]: category_name_by_token.get(row["category_token"], "")
        for row in instance_rows
    }
    ego_pose_by_token = {row["token"]: row for row in ego_pose_rows}
    sensor_by_token = {row["token"]: row for row in sensor_rows}
    calibrated_sensor_channel = {
        row["token"]: sensor_by_token.get(row["sensor_token"], {}).get("channel", "")
        for row in calibrated_sensor_rows
    }
    ego_xy_by_sample: dict[str, tuple[float, float]] = {}
    for row in sample_data_rows:
        if not row.get("is_key_frame"):
            continue
        if calibrated_sensor_channel.get(row["calibrated_sensor_token"]) != "LIDAR_TOP":
            continue
        ego_pose = ego_pose_by_token.get(row["ego_pose_token"])
        if not ego_pose:
            continue
        ego_xy_by_sample[row["sample_token"]] = (
            float(ego_pose["translation"][0]),
            float(ego_pose["translation"][1]),
        )
    log_by_token = {row["token"]: row for row in log_rows}

    rows: list[dict[str, Any]] = []
    for scene_row in scene_rows:
        description = str(scene_row.get("description", "")).strip()
        scene_name = str(scene_row["name"])
        log_row = log_by_token.get(scene_row["log_token"], {})
        stats = compute_scene_stats(
            scene_row,
            sample_by_token=sample_by_token,
            annotations_by_sample=annotations_by_sample,
            category_name_by_instance=category_name_by_instance,
            ego_xy_by_sample=ego_xy_by_sample,
        )

        weather_code = classify_weather(description)
        time_code = classify_time_of_day(description)
        road_code = classify_road_type(description)
        density_code = classify_density(stats["avg_road_users"])
        risk_code = classify_risk(
            weather_code=weather_code,
            time_code=time_code,
            road_code=road_code,
            density_code=density_code,
            pedestrian_avg=stats["avg_pedestrians"],
            cyclist_avg=stats["avg_cyclists"],
        )

        notable_activity = build_notable_activity(
            description=description,
            road_code=road_code,
            density_code=density_code,
            avg_pedestrians=stats["avg_pedestrians"],
            avg_cyclists=stats["avg_cyclists"],
            avg_vehicles=stats["avg_vehicles"],
        )
        open_answer = build_open_ended_answer(
            weather_code=weather_code,
            time_code=time_code,
            road_code=road_code,
            density_code=density_code,
            risk_code=risk_code,
            notable_activity=notable_activity,
        )

        metadata = {
            "nuscenes_log_location": log_row.get("location", ""),
            "scene_description": description,
            "auto_generation_metadata": {
                "avg_road_users": round(stats["avg_road_users"], 2),
                "avg_pedestrians": round(stats["avg_pedestrians"], 2),
                "avg_cyclists": round(stats["avg_cyclists"], 2),
                "avg_vehicles": round(stats["avg_vehicles"], 2),
                "representative_frame_count": int(stats["representative_frame_count"]),
                "generation_source": "rule-based scene-level GT bootstrap",
            },
        }

        rows.extend(
            [
                make_mcq_task(
                    scene_name=scene_name,
                    task_id="SC-1",
                    question="What is the primary weather and environmental condition in the current scene?",
                    choices=WEATHER_CHOICES,
                    ground_truth=weather_code,
                    metadata=metadata,
                ),
                make_mcq_task(
                    scene_name=scene_name,
                    task_id="SC-2",
                    question="Based on the lighting conditions and sky visibility, what is the estimated time of day?",
                    choices=TIME_CHOICES,
                    ground_truth=time_code,
                    metadata=metadata,
                ),
                make_mcq_task(
                    scene_name=scene_name,
                    task_id="SC-3",
                    question="Which of the following best describes the overall road typology and driving environment?",
                    choices=ROAD_CHOICES,
                    ground_truth=road_code,
                    metadata=metadata,
                ),
                make_mcq_task(
                    scene_name=scene_name,
                    task_id="SC-4",
                    question="How spatially dense or crowded is the current traffic environment considering all visible road users?",
                    choices=TRAFFIC_CHOICES,
                    ground_truth=density_code,
                    metadata=metadata,
                ),
                make_mcq_task(
                    scene_name=scene_name,
                    task_id="SC-5",
                    question="What is the baseline safety risk level of the overall scene, independent of any single specific object?",
                    choices=RISK_CHOICES,
                    ground_truth=risk_code,
                    metadata=metadata,
                ),
                make_frq_task(
                    scene_name=scene_name,
                    task_id="SC-6",
                    question="Describe the overall environment of the scene, including weather, lighting conditions, road type, and notable surrounding activity.",
                    ground_truth=open_answer,
                    metadata=metadata,
                ),
            ]
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=True)

    print(f"Generated {len(rows)} scene-level tasks across {len(scene_rows)} scenes.")
    print(f"Saved scene-level GT to {args.output}")


if __name__ == "__main__":
    main()
