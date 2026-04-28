#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SCENE_TASK_IDS = {"SC-1", "SC-2", "SC-3", "SC-4", "SC-5"}


@dataclass
class TaskRow:
    task_id: str
    row: dict[str, Any]


@dataclass
class SampleUnit:
    unit_id: str
    scene_id: str
    group_id: str
    object_id: str
    source_group_file: str
    dataset: str
    tasks: list[TaskRow]
    scene_attrs: dict[str, str]
    image_locator: dict[str, Any]
    metadata_features: dict[str, Any]
    complexity_breakdown: dict[str, float]
    complexity_score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a metadata-driven miniset from a questions_with_answers JSON. "
            "Selection is group-based, metadata-driven, complexity-aware, and balances "
            "question GT distribution plus scene-level attribute distribution."
        )
    )
    parser.add_argument("--input-json", type=Path, required=True, help="Input questions_with_answers JSON.")
    parser.add_argument("--output-json", type=Path, required=True, help="Output selected miniset JSON.")
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional output summary JSON. Defaults to <output-json stem>_summary.json.",
    )
    parser.add_argument(
        "--target-groups",
        type=int,
        default=335,
        help="Number of groups to select.",
    )
    parser.add_argument(
        "--target-scenes",
        type=int,
        default=None,
        help="Target number of distinct scenes to cover. If omitted, defaults to about 5%% of total scenes.",
    )
    parser.add_argument(
        "--min-complexity-quantile",
        type=float,
        default=0.5,
        help="Keep only candidates in the top complexity quantile before balancing.",
    )
    parser.add_argument(
        "--complexity-weight",
        type=float,
        default=0.35,
        help="Weight for normalized complexity score in greedy selection.",
    )
    parser.add_argument(
        "--task-balance-weight",
        type=float,
        default=1.0,
        help="Weight for per-task ground-truth balancing in greedy selection.",
    )
    parser.add_argument(
        "--scene-balance-weight",
        type=float,
        default=0.8,
        help="Weight for scene-attribute balancing in greedy selection.",
    )
    parser.add_argument(
        "--scene-diversity-weight",
        type=float,
        default=1.2,
        help="Weight for preferring groups from new or underrepresented scenes.",
    )
    parser.add_argument(
        "--allow-groups-without-traj",
        action="store_true",
        help="If set, allow selecting groups that do not contain trajectory tasks. Default behavior filters them out.",
    )
    parser.add_argument(
        "--nuscenes-root",
        type=Path,
        default=Path("/local1/rgao727/nuscenes"),
        help="nuScenes root used for formatted scene metadata and raw tables.",
    )
    parser.add_argument(
        "--waymo-annotations-root",
        type=Path,
        default=Path("/local1/rgao727/waymo_dataset/train/annotations"),
        help="Waymo annotations root used for exported metadata.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def infer_default_path(base: Path, suffix: str) -> Path:
    return base.with_name(f"{base.stem}{suffix}{base.suffix}")


def flatten_rows(payload: dict[str, Any]) -> list[TaskRow]:
    rows: list[TaskRow] = []
    generated = payload.get("generated_answers", {})
    if generated:
        for task_id, bucket in generated.items():
            for row in bucket.get("tasks", []):
                rows.append(TaskRow(task_id=task_id, row=row))
        return rows

    for row in payload.get("tasks", []):
        task_id = str(row.get("id", "")).strip()
        if task_id:
            rows.append(TaskRow(task_id=task_id, row=row))
    return rows


def resolve_gt_text(row: dict[str, Any]) -> str:
    gt = str(row.get("ground_truth", "")).strip()
    choices = row.get("choices", {})
    if isinstance(choices, dict) and gt in choices and isinstance(choices[gt], str):
        return choices[gt].strip()
    return gt


def is_balancable_task(row: dict[str, Any]) -> bool:
    gt = str(row.get("ground_truth", "")).strip()
    choices = row.get("choices", {})
    return isinstance(choices, dict) and bool(choices) and gt in choices


def make_group_key(row: dict[str, Any]) -> tuple[str, str, str]:
    scene_id = str(row.get("scene_id", ""))
    group_id = str(row.get("group_id", ""))
    source_group_file = str(row.get("source_group_file", ""))
    return scene_id, group_id, source_group_file


def detect_dataset(input_json: Path, rows: list[TaskRow]) -> str:
    path_str = str(input_json).lower()
    if "waymo" in path_str:
        return "waymo"
    if "nuscenes" in path_str:
        return "nuscenes"
    for item in rows[:50]:
        source = str(item.row.get("source_group_file", "")).lower()
        scene_id = str(item.row.get("scene_id", "")).lower()
        if "vehicle_annotations" in source or scene_id.startswith("scene_"):
            return "nuscenes"
        if scene_id and len(scene_id) > 20 and "-" not in scene_id:
            return "waymo"
    return "unknown"


def extract_scene_attrs(rows: list[TaskRow]) -> dict[str, str]:
    attrs = {
        "weather": "",
        "time_of_day": "",
        "road_type": "",
        "traffic_density": "",
        "risk": "",
    }
    mapping = {
        "SC-1": "weather",
        "SC-2": "time_of_day",
        "SC-3": "road_type",
        "SC-4": "traffic_density",
        "SC-5": "risk",
    }
    for item in rows:
        key = mapping.get(item.task_id)
        if key:
            attrs[key] = resolve_gt_text(item.row)
    return attrs


def merge_scene_attrs(base: dict[str, str], fallback: dict[str, str]) -> dict[str, str]:
    merged = dict(base)
    for key, value in fallback.items():
        if not merged.get(key) and value:
            merged[key] = value
    return merged


def keyword_bonus(text: str) -> float:
    x = text.lower()
    score = 0.0
    weights = {
        "dense": 2.0,
        "moderate": 0.8,
        "high risk": 2.5,
        "night": 1.5,
        "dawn": 1.0,
        "dusk": 1.0,
        "rain": 1.2,
        "wet": 0.8,
        "snow": 1.8,
        "fog": 1.5,
        "haze": 1.0,
        "intersection": 1.8,
        "crossroad": 1.8,
        "construction": 1.6,
        "highway": 0.8,
        "close": 1.2,
        "occluded": 0.8,
        "collision": 1.2,
    }
    for phrase, weight in weights.items():
        if phrase in x:
            score += weight
    return score


def safe_mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def category_bucket(category_name: str) -> str:
    x = category_name.lower()
    if "pedestrian" in x:
        return "pedestrian"
    if "bicycle" in x or "motorcycle" in x or "cycle" in x:
        return "two_wheeler"
    if "bus" in x or "truck" in x or "trailer" in x:
        return "large_vehicle"
    if "barrier" in x or "cone" in x or "trafficcone" in x:
        return "traffic_control"
    if "vehicle" in x or "car" in x:
        return "car"
    return "other"


@dataclass
class NuscenesContext:
    root: Path
    formatted_root: Path
    attr_name_by_token: dict[str, str]
    visibility_level_by_token: dict[str, int]
    annotation_attrs_by_token: dict[str, list[str]]
    group_cache: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, root: Path) -> "NuscenesContext":
        table_root = root / "v1.0-trainval"
        attr_name_by_token = {
            row["token"]: row["name"]
            for row in read_json(table_root / "attribute.json")
        }
        visibility_level_by_token = {
            row["token"]: int(row["token"])
            for row in read_json(table_root / "visibility.json")
        }
        annotation_attrs_by_token: dict[str, list[str]] = {}
        for row in read_json(table_root / "sample_annotation.json"):
            annotation_attrs_by_token[row["token"]] = [
                attr_name_by_token.get(token, "")
                for token in row.get("attribute_tokens", [])
                if attr_name_by_token.get(token, "")
            ]
        return cls(
            root=root,
            formatted_root=root / "formatted_scenes_trainval",
            attr_name_by_token=attr_name_by_token,
            visibility_level_by_token=visibility_level_by_token,
            annotation_attrs_by_token=annotation_attrs_by_token,
        )

    def load_group_metadata(self, source_group_file: str) -> dict[str, Any]:
        if source_group_file in self.group_cache:
            return self.group_cache[source_group_file]
        path = self.formatted_root / source_group_file
        if path.exists():
            self.group_cache[source_group_file] = read_json(path)
        else:
            self.group_cache[source_group_file] = {}
        return self.group_cache[source_group_file]

    def build_unit_metadata(
        self,
        scene_id: str,
        group_id: str,
        object_id: str,
        source_group_file: str,
        scene_attrs: dict[str, str],
    ) -> tuple[dict[str, Any], dict[str, float], dict[str, Any]]:
        group_meta = self.load_group_metadata(source_group_file)
        vehicles = group_meta.get("vehicle_annotations", [])
        pedestrians = group_meta.get("pedestrian_annotations", [])
        all_annotations = vehicles + pedestrians

        near_0_5 = 0
        near_5_15 = 0
        low_visibility = 0
        moving_count = 0
        distinct_categories: set[str] = set()
        for ann in all_annotations:
            dist = ann.get("distance_to_ego_m")
            if isinstance(dist, (int, float)):
                if dist <= 5.0:
                    near_0_5 += 1
                elif dist <= 15.0:
                    near_5_15 += 1
            vis_token = str(ann.get("visibility_token", ""))
            vis_level = self.visibility_level_by_token.get(vis_token)
            if vis_level is not None and vis_level <= 2:
                low_visibility += 1
            for attr_name in self.annotation_attrs_by_token.get(str(ann.get("token", "")), []):
                if "moving" in attr_name or "with_rider" in attr_name:
                    moving_count += 1
                    break
            distinct_categories.add(category_bucket(str(ann.get("category_name", ""))))

        selected_distance = group_meta.get("selected_distance_to_ego_m")
        selected_priority = group_meta.get("selected_priority_score", 0.0)
        vehicle_count = int(group_meta.get("vehicle_annotation_count", len(vehicles)) or 0)
        pedestrian_count = int(group_meta.get("pedestrian_annotation_count", len(pedestrians)) or 0)
        visible_count = int(group_meta.get("visible_target_annotation_count", len(all_annotations)) or 0)
        group_size = int(group_meta.get("group_size", 0) or 0)

        interaction_score = near_0_5 * 2.0 + near_5_15 * 1.0
        if isinstance(selected_distance, (int, float)):
            if selected_distance <= 5.0:
                interaction_score += 2.0
            elif selected_distance <= 15.0:
                interaction_score += 1.0

        density_score = min(vehicle_count, 15) * 0.3 + min(pedestrian_count, 10) * 0.5
        density_score += min(max(0, visible_count - vehicle_count - pedestrian_count), 10) * 0.2
        motion_score = moving_count * 0.4
        occlusion_score = low_visibility * 0.5
        layout_score = keyword_bonus(scene_attrs.get("road_type", ""))
        environment_score = keyword_bonus(scene_attrs.get("weather", "")) + keyword_bonus(scene_attrs.get("time_of_day", ""))
        diversity_score = len({x for x in distinct_categories if x and x != "other"}) * 0.4
        priority_score = float(selected_priority or 0.0) * 0.5
        temporal_score = min(group_size, 5) * 0.1

        breakdown = {
            "interaction": interaction_score,
            "density": density_score,
            "motion": motion_score,
            "occlusion": occlusion_score,
            "scene_layout": layout_score,
            "environment": environment_score,
            "diversity": diversity_score,
            "priority": priority_score,
            "temporal": temporal_score,
        }
        total = sum(breakdown.values())

        image_locator = {
            "dataset": "nuscenes",
            "scene_id": scene_id,
            "group_id": group_id,
            "object_id": object_id,
            "source_group_file": source_group_file,
            "group_annotation_path": str(self.formatted_root / source_group_file) if source_group_file else "",
            "query_image_filename": group_meta.get("fifth_frame_cam_front_filename", ""),
            "query_image_path": (
                str((self.root / group_meta.get("fifth_frame_cam_front_filename", "")).resolve())
                if group_meta.get("fifth_frame_cam_front_filename")
                else ""
            ),
            "selected_vehicle_render_path": group_meta.get("selected_vehicle_render_path", ""),
        }
        metadata_features = {
            "group_size": group_size,
            "frame_indices_1based": group_meta.get("frame_indices_1based", []),
            "vehicle_count": vehicle_count,
            "pedestrian_count": pedestrian_count,
            "visible_target_count": visible_count,
            "near_0_5m_count": near_0_5,
            "near_5_15m_count": near_5_15,
            "low_visibility_count": low_visibility,
            "moving_object_count": moving_count,
            "distinct_category_count": len(distinct_categories),
            "selected_distance_to_ego_m": selected_distance,
            "selected_priority_score": selected_priority,
            "selected_category_name": group_meta.get("selected_category_name", ""),
        }
        return metadata_features, {**breakdown, "total": total}, image_locator


@dataclass
class WaymoContext:
    annotations_root: Path
    frame_index: dict[str, Any]
    labels_cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    ego_cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    distance_map: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, annotations_root: Path) -> "WaymoContext":
        frame_index_path = annotations_root / "frame_index.json"
        frame_index = read_json(frame_index_path) if frame_index_path.exists() else {}
        distance_map: dict[str, dict[str, Any]] = {}
        distance_path = annotations_root / "bbox_distance_estimates.json"
        if distance_path.exists():
            payload = read_json(distance_path)
            for row in payload.get("results", []):
                distance_map[row.get("frame_name", "")] = row
        alt20 = annotations_root / "bbox_distance_estimates_20.json"
        if not distance_map and alt20.exists():
            payload = read_json(alt20)
            for row in payload.get("results", []):
                distance_map[row.get("frame_name", "")] = row
        return cls(annotations_root=annotations_root, frame_index=frame_index, distance_map=distance_map)

    def load_labels(self, frame_name: str) -> dict[str, Any]:
        if frame_name in self.labels_cache:
            return self.labels_cache[frame_name]
        path = self.annotations_root / "labels" / f"{frame_name}.json"
        self.labels_cache[frame_name] = read_json(path) if path.exists() else {}
        return self.labels_cache[frame_name]

    def load_ego(self, frame_name: str) -> dict[str, Any]:
        if frame_name in self.ego_cache:
            return self.ego_cache[frame_name]
        path = self.annotations_root / "ego_status" / f"{frame_name}.json"
        self.ego_cache[frame_name] = read_json(path) if path.exists() else {}
        return self.ego_cache[frame_name]

    def infer_query_frame_name(self, scene_id: str, group_id: str) -> str:
        suffix = group_id.split("_")[-1] if group_id else ""
        if suffix.isdigit():
            return f"{scene_id}-{int(suffix):03d}"
        return ""

    def build_unit_metadata(
        self,
        scene_id: str,
        group_id: str,
        object_id: str,
        source_group_file: str,
        scene_attrs: dict[str, str],
    ) -> tuple[dict[str, Any], dict[str, float], dict[str, Any]]:
        frame_name = self.infer_query_frame_name(scene_id, group_id)
        label_meta = self.load_labels(frame_name)
        ego_meta = self.load_ego(frame_name)
        dist_meta = self.distance_map.get(frame_name, {})

        distance_estimates = dist_meta.get("distance_estimates", [])
        near_0_5 = 0
        near_5_15 = 0
        distinct_labels: set[str] = set()
        for item in distance_estimates:
            dist = item.get("distance_m")
            if isinstance(dist, (int, float)):
                if dist <= 5.0:
                    near_0_5 += 1
                elif dist <= 15.0:
                    near_5_15 += 1
            label = str(item.get("label", "")).strip().lower()
            if label:
                distinct_labels.add(label)

        camera_labels_count = int(label_meta.get("camera_labels_count", 0) or 0)
        laser_labels_count = int(label_meta.get("laser_labels_count", 0) or 0)
        projected_lidar_labels_count = int(label_meta.get("projected_lidar_labels_count", 0) or 0)
        future_states = ego_meta.get("future_states", {}) if isinstance(ego_meta.get("future_states", {}), dict) else {}
        past_states = ego_meta.get("past_states", {}) if isinstance(ego_meta.get("past_states", {}), dict) else {}

        future_pos_y = future_states.get("pos_y", []) if isinstance(future_states.get("pos_y", []), list) else []
        future_pos_x = future_states.get("pos_x", []) if isinstance(future_states.get("pos_x", []), list) else []
        past_vel_x = past_states.get("vel_x", []) if isinstance(past_states.get("vel_x", []), list) else []
        ego_turn_flag = 1.0 if future_pos_y and max(abs(float(v)) for v in future_pos_y) > 1.0 else 0.0
        ego_accel_flag = 0.0
        if len(past_vel_x) >= 2:
            ego_accel_flag = 1.0 if abs(float(past_vel_x[-1]) - float(past_vel_x[0])) > 1.0 else 0.0

        interaction_score = near_0_5 * 2.0 + near_5_15 * 1.0
        density_score = min(camera_labels_count, 15) * 0.2 + min(laser_labels_count, 20) * 0.2
        density_score += min(projected_lidar_labels_count, 20) * 0.1
        motion_score = ego_turn_flag * 1.0 + ego_accel_flag * 0.8
        occlusion_score = 0.0
        layout_score = keyword_bonus(scene_attrs.get("road_type", ""))
        environment_score = keyword_bonus(scene_attrs.get("weather", "")) + keyword_bonus(scene_attrs.get("time_of_day", ""))
        diversity_score = len(distinct_labels) * 0.4
        temporal_score = 0.5 if future_pos_x else 0.0

        breakdown = {
            "interaction": interaction_score,
            "density": density_score,
            "motion": motion_score,
            "occlusion": occlusion_score,
            "scene_layout": layout_score,
            "environment": environment_score,
            "diversity": diversity_score,
            "priority": 0.0,
            "temporal": temporal_score,
        }
        total = sum(breakdown.values())

        image_path = dist_meta.get("image_path", "")
        image_locator = {
            "dataset": "waymo",
            "scene_id": scene_id,
            "group_id": group_id,
            "object_id": object_id,
            "source_group_file": source_group_file,
            "query_frame_name": frame_name,
            "image_path": image_path,
        }
        metadata_features = {
            "query_frame_name": frame_name,
            "camera_labels_count": camera_labels_count,
            "laser_labels_count": laser_labels_count,
            "projected_lidar_labels_count": projected_lidar_labels_count,
            "near_0_5m_count": near_0_5,
            "near_5_15m_count": near_5_15,
            "distance_estimate_count": len(distance_estimates),
            "distinct_label_count": len(distinct_labels),
            "ego_turn_flag": ego_turn_flag,
            "ego_accel_flag": ego_accel_flag,
            "intent": ego_meta.get("intent"),
        }
        return metadata_features, {**breakdown, "total": total}, image_locator


def build_units(
    flat_rows: list[TaskRow],
    dataset: str,
    nuscenes_ctx: NuscenesContext | None,
    waymo_ctx: WaymoContext | None,
) -> list[SampleUnit]:
    grouped: dict[tuple[str, str, str], list[TaskRow]] = defaultdict(list)
    scene_attrs_by_group: dict[tuple[str, str], dict[str, str]] = {}
    for item in flat_rows:
        grouped[make_group_key(item.row)].append(item)
        if item.task_id in SCENE_TASK_IDS:
            scene_id = str(item.row.get("scene_id", ""))
            group_id = str(item.row.get("group_id", ""))
            key = (scene_id, group_id)
            current = scene_attrs_by_group.get(key, {
                "weather": "",
                "time_of_day": "",
                "road_type": "",
                "traffic_density": "",
                "risk": "",
            })
            scene_attrs_by_group[key] = merge_scene_attrs(
                current,
                extract_scene_attrs([item]),
            )

    units: list[SampleUnit] = []
    for (scene_id, group_id, source_group_file), rows in grouped.items():
        scene_attrs = merge_scene_attrs(
            extract_scene_attrs(rows),
            scene_attrs_by_group.get((scene_id, group_id), {}),
        )
        task_ids = {item.task_id for item in rows}
        object_ids = sorted({str(item.row.get("object_id", "")).strip() for item in rows if str(item.row.get("object_id", "")).strip()})
        representative_object_id = object_ids[0] if len(object_ids) == 1 else ""
        if dataset == "nuscenes" and nuscenes_ctx is not None:
            metadata_features, breakdown, image_locator = nuscenes_ctx.build_unit_metadata(
                scene_id, group_id, representative_object_id, source_group_file, scene_attrs
            )
        elif dataset == "waymo" and waymo_ctx is not None:
            metadata_features, breakdown, image_locator = waymo_ctx.build_unit_metadata(
                scene_id, group_id, representative_object_id, source_group_file, scene_attrs
            )
        else:
            metadata_features = {}
            image_locator = {
                "dataset": dataset,
                "scene_id": scene_id,
                "group_id": group_id,
                "object_id": representative_object_id,
                "source_group_file": source_group_file,
            }
            breakdown = {
                "interaction": 0.0,
                "density": 0.0,
                "motion": 0.0,
                "occlusion": 0.0,
                "scene_layout": keyword_bonus(scene_attrs.get("road_type", "")),
                "environment": keyword_bonus(scene_attrs.get("weather", "")) + keyword_bonus(scene_attrs.get("time_of_day", "")),
                "diversity": 0.0,
                "priority": 0.0,
                "temporal": 0.0,
            }
            breakdown["total"] = sum(breakdown.values())

        has_traj_questions = any(task_id.startswith("TRJ-") for task_id in task_ids)
        metadata_features["group_object_count"] = len(object_ids)
        metadata_features["object_ids"] = object_ids[:50]
        metadata_features["has_traj_questions"] = has_traj_questions
        unit_id = "|".join([scene_id, group_id])
        units.append(
            SampleUnit(
                unit_id=unit_id,
                scene_id=scene_id,
                group_id=group_id,
                object_id=representative_object_id,
                source_group_file=source_group_file,
                dataset=dataset,
                tasks=rows,
                scene_attrs=scene_attrs,
                image_locator=image_locator,
                metadata_features=metadata_features,
                complexity_breakdown=breakdown,
                complexity_score=float(breakdown.get("total", 0.0)),
            )
        )
    return units


def quantile_threshold(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0.0:
        return min(values)
    if q >= 1.0:
        return max(values)
    sorted_values = sorted(values)
    idx = min(len(sorted_values) - 1, max(0, math.floor(q * (len(sorted_values) - 1))))
    return sorted_values[idx]


def complexity_norm(score: float, min_score: float, max_score: float) -> float:
    if max_score <= min_score:
        return 0.0
    return (score - min_score) / (max_score - min_score)


def compute_selection_score(
    unit: SampleUnit,
    task_counts: dict[str, Counter],
    scene_attr_counts: dict[str, Counter],
    scene_counts: Counter,
    selected_scene_count: int,
    target_scenes: int,
    min_score: float,
    max_score: float,
    complexity_weight: float,
    task_balance_weight: float,
    scene_balance_weight: float,
    scene_diversity_weight: float,
) -> float:
    score = complexity_weight * complexity_norm(unit.complexity_score, min_score, max_score)

    for item in unit.tasks:
        if is_balancable_task(item.row):
            gt = str(item.row.get("ground_truth", ""))
            score += task_balance_weight * (1.0 / (1.0 + task_counts[item.task_id][gt]))

    for attr_name, attr_value in unit.scene_attrs.items():
        if attr_value:
            score += scene_balance_weight * (1.0 / (1.0 + scene_attr_counts[attr_name][attr_value]))

    if unit.scene_id:
        score += 0.5 * scene_diversity_weight * (1.0 / (1.0 + scene_counts[unit.scene_id]))
        if selected_scene_count < target_scenes and scene_counts[unit.scene_id] == 0:
            score += 1.0 * scene_diversity_weight

    if unit.metadata_features.get("has_traj_questions"):
        score += 0.4

    return score


def select_units(units: list[SampleUnit], args: argparse.Namespace) -> list[SampleUnit]:
    if not units:
        return []

    values = [u.complexity_score for u in units]
    threshold = quantile_threshold(values, args.min_complexity_quantile)
    candidates = [u for u in units if u.complexity_score >= threshold]
    if not args.allow_groups_without_traj:
        traj_candidates = [u for u in candidates if u.metadata_features.get("has_traj_questions")]
        if traj_candidates:
            candidates = traj_candidates
    if not candidates:
        candidates = units[:]

    min_score = min(u.complexity_score for u in candidates)
    max_score = max(u.complexity_score for u in candidates)

    selected: list[SampleUnit] = []
    task_counts: dict[str, Counter] = defaultdict(Counter)
    scene_attr_counts: dict[str, Counter] = defaultdict(Counter)
    scene_counts: Counter = Counter()
    remaining = candidates[:]
    target = min(args.target_groups, len(remaining))
    total_scenes = len({u.scene_id for u in units if u.scene_id})
    target_scenes = args.target_scenes or max(1, round(total_scenes * 0.05))

    while remaining and len(selected) < target:
        best = max(
            remaining,
            key=lambda u: compute_selection_score(
                u,
                task_counts=task_counts,
                scene_attr_counts=scene_attr_counts,
                scene_counts=scene_counts,
                selected_scene_count=len(scene_counts),
                target_scenes=target_scenes,
                min_score=min_score,
                max_score=max_score,
                complexity_weight=args.complexity_weight,
                task_balance_weight=args.task_balance_weight,
                scene_balance_weight=args.scene_balance_weight,
                scene_diversity_weight=args.scene_diversity_weight,
            ),
        )
        selected.append(best)
        remaining.remove(best)
        scene_counts[best.scene_id] += 1

        for item in best.tasks:
            if is_balancable_task(item.row):
                task_counts[item.task_id][str(item.row.get("ground_truth", ""))] += 1
        for attr_name, attr_value in best.scene_attrs.items():
            if attr_value:
                scene_attr_counts[attr_name][attr_value] += 1

    return selected


def compute_task_count_map(rows: list[TaskRow]) -> dict[str, int]:
    counter = Counter(item.task_id for item in rows)
    return dict(sorted(counter.items()))


def compute_gt_distribution(rows: list[TaskRow]) -> dict[str, dict[str, int]]:
    per_task: dict[str, Counter] = defaultdict(Counter)
    for item in rows:
        if is_balancable_task(item.row):
            per_task[item.task_id][str(item.row.get("ground_truth", ""))] += 1
    return {task_id: dict(counter) for task_id, counter in sorted(per_task.items())}


def compute_mcq_balance_metrics(gt_distribution: dict[str, dict[str, int]]) -> dict[str, Any]:
    per_task: dict[str, dict[str, Any]] = {}
    max_option_fractions: list[float] = []
    imbalance_ratios: list[float] = []

    for task_id, counts in sorted(gt_distribution.items()):
        total = sum(counts.values())
        if total <= 0:
            continue
        max_count = max(counts.values())
        min_count = min(counts.values())
        option_count = len(counts)
        ideal_fraction = (1.0 / option_count) if option_count > 0 else None
        max_option_fraction = max_count / total
        min_option_fraction = min_count / total
        imbalance_ratio = (max_count / min_count) if min_count > 0 else None

        max_option_fractions.append(max_option_fraction)
        if imbalance_ratio is not None:
            imbalance_ratios.append(imbalance_ratio)

        per_task[task_id] = {
            "option_count": option_count,
            "total_questions": total,
            "ideal_fraction_per_option": ideal_fraction,
            "max_option_fraction": max_option_fraction,
            "min_option_fraction": min_option_fraction,
            "imbalance_ratio_max_over_min": imbalance_ratio,
        }

    return {
        "per_task": per_task,
        "aggregate": {
            "task_count": len(per_task),
            "mean_max_option_fraction": statistics.mean(max_option_fractions) if max_option_fractions else None,
            "median_max_option_fraction": statistics.median(max_option_fractions) if max_option_fractions else None,
            "mean_imbalance_ratio_max_over_min": statistics.mean(imbalance_ratios) if imbalance_ratios else None,
            "median_imbalance_ratio_max_over_min": statistics.median(imbalance_ratios) if imbalance_ratios else None,
        },
    }


def compute_scene_attr_distribution(units: list[SampleUnit]) -> dict[str, dict[str, int]]:
    counters: dict[str, Counter] = defaultdict(Counter)
    for unit in units:
        for attr_name, attr_value in unit.scene_attrs.items():
            if attr_value:
                counters[attr_name][attr_value] += 1
    return {attr_name: dict(counter) for attr_name, counter in sorted(counters.items())}


def complexity_stats(units: list[SampleUnit]) -> dict[str, Any]:
    scores = [u.complexity_score for u in units]
    if not scores:
        return {"min": None, "max": None, "mean": None, "median": None}
    return {
        "min": min(scores),
        "max": max(scores),
        "mean": statistics.mean(scores),
        "median": statistics.median(scores),
    }


def aggregate_breakdown(units: list[SampleUnit]) -> dict[str, float]:
    if not units:
        return {}
    keys = sorted({k for unit in units for k in unit.complexity_breakdown.keys() if k != "total"})
    return {k: safe_mean([unit.complexity_breakdown.get(k, 0.0) for unit in units]) for k in keys}


def metadata_source_info(
    dataset: str,
    nuscenes_ctx: NuscenesContext | None,
    waymo_ctx: WaymoContext | None,
) -> dict[str, Any]:
    if dataset == "nuscenes":
        return {
            "dataset": dataset,
            "uses_external_metadata": True,
            "metadata_roots": {
                "nuscenes_root": str(nuscenes_ctx.root) if nuscenes_ctx else "",
                "formatted_scenes_root": str(nuscenes_ctx.formatted_root) if nuscenes_ctx else "",
            },
            "metadata_used": [
                "formatted_scenes_trainval/group_*_vehicle_annotations.json",
                "v1.0-trainval/attribute.json",
                "v1.0-trainval/visibility.json",
                "v1.0-trainval/sample_annotation.json",
            ],
            "scene_attribute_source": "SC-1 to SC-5 ground truth labels in the input JSON.",
        }
    if dataset == "waymo":
        return {
            "dataset": dataset,
            "uses_external_metadata": True,
            "metadata_roots": {
                "waymo_annotations_root": str(waymo_ctx.annotations_root) if waymo_ctx else "",
            },
            "metadata_used": [
                "frame_index.json",
                "labels/*.json",
                "ego_status/*.json",
                "bbox_distance_estimates*.json when available",
            ],
            "scene_attribute_source": "SC-1 to SC-5 ground truth labels in the input JSON.",
        }
    return {
        "dataset": dataset,
        "uses_external_metadata": False,
        "metadata_roots": {},
        "metadata_used": [],
        "scene_attribute_source": "SC-1 to SC-5 ground truth labels in the input JSON.",
    }


def build_selected_output(
    input_json: Path,
    dataset: str,
    selected_units: list[SampleUnit],
    args: argparse.Namespace,
    metadata_info: dict[str, Any],
) -> dict[str, Any]:
    selected_questions: list[dict[str, Any]] = []
    selected_units_payload: list[dict[str, Any]] = []
    global_question_number = 1

    for rank, unit in enumerate(selected_units, start=1):
        unit_questions: list[dict[str, Any]] = []
        for local_idx, item in enumerate(sorted(unit.tasks, key=lambda x: (x.task_id, str(x.row.get("question", "")))), start=1):
            question_payload = {
                "question_number": global_question_number,
                "question_number_within_unit": local_idx,
                "task_id": item.task_id,
                "question": item.row.get("question", ""),
                "choices": item.row.get("choices", {}),
                "ground_truth": item.row.get("ground_truth", ""),
                "question_format": item.row.get("question_format", ""),
                "object_reference": item.row.get("object_reference", ""),
                "scene_id": unit.scene_id,
                "group_id": unit.group_id,
                "object_id": unit.object_id,
                "source_group_file": unit.source_group_file,
                "unit_id": unit.unit_id,
            }
            unit_questions.append(question_payload)
            selected_questions.append(question_payload)
            global_question_number += 1

        selected_units_payload.append(
            {
                "selection_rank": rank,
                "unit_id": unit.unit_id,
                "scene_id": unit.scene_id,
                "group_id": unit.group_id,
                "object_id": unit.object_id,
                "source_group_file": unit.source_group_file,
                "image_locator": unit.image_locator,
                "scene_attrs": unit.scene_attrs,
                "metadata_features": unit.metadata_features,
                "complexity_breakdown": unit.complexity_breakdown,
                "questions": unit_questions,
            }
        )

    return {
        "source_input_json": str(input_json),
        "dataset": dataset,
        "selection_config": {
            "target_groups": args.target_groups,
            "target_scenes": args.target_scenes,
            "min_complexity_quantile": args.min_complexity_quantile,
            "complexity_weight": args.complexity_weight,
            "task_balance_weight": args.task_balance_weight,
            "scene_balance_weight": args.scene_balance_weight,
            "scene_diversity_weight": args.scene_diversity_weight,
            "allow_groups_without_traj": args.allow_groups_without_traj,
        },
        "metadata_source": metadata_info,
        "selected_group_count": len(selected_units),
        "selected_scene_count": len({u.scene_id for u in selected_units if u.scene_id}),
        "selected_question_count": len(selected_questions),
        "selected_groups": selected_units_payload,
        "selected_questions": selected_questions,
    }


def build_summary(
    dataset: str,
    args: argparse.Namespace,
    flat_rows: list[TaskRow],
    all_units: list[SampleUnit],
    selected_units: list[SampleUnit],
    metadata_info: dict[str, Any],
) -> dict[str, Any]:
    selected_rows = [item for unit in selected_units for item in unit.tasks]
    full_gt_distribution = compute_gt_distribution(flat_rows)
    miniset_gt_distribution = compute_gt_distribution(selected_rows)
    full_mcq_balance = compute_mcq_balance_metrics(full_gt_distribution)
    miniset_mcq_balance = compute_mcq_balance_metrics(miniset_gt_distribution)
    return {
        "dataset": dataset,
        "selection_config": {
            "target_groups": args.target_groups,
            "target_scenes": args.target_scenes,
            "min_complexity_quantile": args.min_complexity_quantile,
            "complexity_weight": args.complexity_weight,
            "task_balance_weight": args.task_balance_weight,
            "scene_balance_weight": args.scene_balance_weight,
            "scene_diversity_weight": args.scene_diversity_weight,
            "allow_groups_without_traj": args.allow_groups_without_traj,
        },
        "metadata_source": metadata_info,
        "full_set": {
            "group_count": len(all_units),
            "scene_count": len({u.scene_id for u in all_units if u.scene_id}),
            "question_count": len(flat_rows),
            "task_type_count": len({item.task_id for item in flat_rows}),
            "complexity": complexity_stats(all_units),
            "average_complexity_breakdown": aggregate_breakdown(all_units),
            "per_task_counts": compute_task_count_map(flat_rows),
            "per_task_ground_truth_distribution": full_gt_distribution,
            "mcq_balance": full_mcq_balance,
            "scene_attribute_distribution": compute_scene_attr_distribution(all_units),
        },
        "miniset": {
            "group_count": len(selected_units),
            "scene_count": len({u.scene_id for u in selected_units if u.scene_id}),
            "question_count": len(selected_rows),
            "task_type_count": len({item.task_id for item in selected_rows}),
            "complexity": complexity_stats(selected_units),
            "average_complexity_breakdown": aggregate_breakdown(selected_units),
            "per_task_counts": compute_task_count_map(selected_rows),
            "per_task_ground_truth_distribution": miniset_gt_distribution,
            "mcq_balance": miniset_mcq_balance,
            "scene_attribute_distribution": compute_scene_attr_distribution(selected_units),
        },
        "retention": {
            "group_fraction": (len(selected_units) / len(all_units)) if all_units else 0.0,
            "scene_fraction": (
                len({u.scene_id for u in selected_units if u.scene_id}) /
                len({u.scene_id for u in all_units if u.scene_id})
            ) if all_units else 0.0,
            "question_fraction": (len(selected_rows) / len(flat_rows)) if flat_rows else 0.0,
        },
        "comparison": {
            "mcq_balance": {
                "full_set_mean_max_option_fraction": full_mcq_balance["aggregate"]["mean_max_option_fraction"],
                "miniset_mean_max_option_fraction": miniset_mcq_balance["aggregate"]["mean_max_option_fraction"],
                "full_set_mean_imbalance_ratio": full_mcq_balance["aggregate"]["mean_imbalance_ratio_max_over_min"],
                "miniset_mean_imbalance_ratio": miniset_mcq_balance["aggregate"]["mean_imbalance_ratio_max_over_min"],
            }
        },
    }


def main() -> None:
    args = parse_args()
    summary_json = args.summary_json or infer_default_path(args.output_json, "_summary")

    payload = read_json(args.input_json)
    flat_rows = flatten_rows(payload)
    dataset = detect_dataset(args.input_json, flat_rows)

    nuscenes_ctx = None
    waymo_ctx = None
    if dataset == "nuscenes" and args.nuscenes_root.exists():
        nuscenes_ctx = NuscenesContext.load(args.nuscenes_root)
    if dataset == "waymo" and args.waymo_annotations_root.exists():
        waymo_ctx = WaymoContext.load(args.waymo_annotations_root)

    all_units = build_units(flat_rows, dataset, nuscenes_ctx, waymo_ctx)
    selected_units = select_units(all_units, args)
    metadata_info = metadata_source_info(dataset, nuscenes_ctx, waymo_ctx)

    selected_output = build_selected_output(args.input_json, dataset, selected_units, args, metadata_info)
    summary_output = build_summary(dataset, args, flat_rows, all_units, selected_units, metadata_info)

    write_json(args.output_json, selected_output)
    write_json(summary_json, summary_output)

    print(f"Dataset: {dataset}")
    print(f"Loaded groups: {len(all_units)}")
    print(f"Selected groups: {len(selected_units)}")
    print(f"Saved selected miniset to: {args.output_json}")
    print(f"Saved summary to: {summary_json}")


if __name__ == "__main__":
    main()
