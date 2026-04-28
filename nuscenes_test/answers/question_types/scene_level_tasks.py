from __future__ import annotations

import copy
from typing import Any


def _index_scene_level_tasks(payload: Any) -> dict[str, list[dict]]:
    if not isinstance(payload, list):
        raise ValueError("Expected a list of scene-level tasks.")

    tasks_by_scene: dict[str, list[dict]] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        scene_name = str(row.get("scene_id", "")).strip()
        if not scene_name:
            continue
        tasks_by_scene.setdefault(scene_name, []).append(row)
    return tasks_by_scene


def merge_scene_level_tasks(
    *,
    embedded_payload: Any = None,
    external_payload: Any = None,
) -> dict[str, list[dict]]:
    merged: dict[str, list[dict]] = {}

    if isinstance(embedded_payload, list):
        for scene_name, rows in _index_scene_level_tasks(embedded_payload).items():
            merged.setdefault(scene_name, []).extend(rows)

    if isinstance(external_payload, list):
        for scene_name, rows in _index_scene_level_tasks(external_payload).items():
            merged.setdefault(scene_name, []).extend(rows)

    return merged


def clone_scene_level_task_for_group(
    template: dict,
    *,
    group_scene_id: str,
    group_id: str,
    source_group_file: str,
    nuscenes_scene_name: str,
) -> dict:
    task = copy.deepcopy(template)
    task["scene_id"] = group_scene_id
    task["group_id"] = group_id
    task["source_group_file"] = source_group_file
    task["nuscenes_scene_name"] = nuscenes_scene_name
    task["scene_level_source_scene_id"] = str(template.get("scene_id", "")).strip()
    task["model_response"] = ""
    return task
