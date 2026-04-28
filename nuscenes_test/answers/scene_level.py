"""Compatibility wrapper for older imports.

Prefer importing from ``answers.question_types.scene_level_tasks``.
"""

from .question_types.scene_level_tasks import (
    clone_scene_level_task_for_group,
    merge_scene_level_tasks,
)

__all__ = ["clone_scene_level_task_for_group", "merge_scene_level_tasks"]
