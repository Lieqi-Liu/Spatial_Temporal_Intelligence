from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..common.generation_config import AnswerGenerationConfig
from ..common.json_store import load_json
from ..question_types.scene_level_tasks import merge_scene_level_tasks


@dataclass
class PipelineState:
    questions: dict[str, Any]
    scene_level_tasks_by_scene: dict[str, list[dict]]
    sample_data: list[dict[str, Any]]
    sample: list[dict[str, Any]]
    sample_annotation: list[dict[str, Any]]
    instance: list[dict[str, Any]]
    category: list[dict[str, Any]]
    ego_pose: list[dict[str, Any]]
    scene: list[dict[str, Any]]
    log: list[dict[str, Any]]


class AnswerGenerationPipeline:
    """Shared state assembly for the new modular answer-generation flow."""

    def __init__(self, config: AnswerGenerationConfig):
        self.config = config

    def load_state(self) -> PipelineState:
        version_dir = self.config.version_dir()
        questions = load_json(self.config.resolved_questions_json())
        if not isinstance(questions, dict):
            raise ValueError("questions JSON must be a dict.")

        external_scene_level = None
        if self.config.scene_level_tasks_json is not None and self.config.scene_level_tasks_json.exists():
            external_scene_level = load_json(self.config.scene_level_tasks_json)

        scene_level_tasks_by_scene = merge_scene_level_tasks(
            embedded_payload=questions.get("scene_level_tasks"),
            external_payload=external_scene_level,
        )

        sample_data = load_json(version_dir / "sample_data.json")
        sample = load_json(version_dir / "sample.json")
        sample_annotation = load_json(version_dir / "sample_annotation.json")
        instance = load_json(version_dir / "instance.json")
        category = load_json(version_dir / "category.json")
        ego_pose = load_json(version_dir / "ego_pose.json")
        scene = load_json(version_dir / "scene.json")
        log = load_json(version_dir / "log.json")

        if not isinstance(sample_data, list):
            raise ValueError("sample_data.json must be a list.")
        if not isinstance(sample, list):
            raise ValueError("sample.json must be a list.")
        if not isinstance(sample_annotation, list):
            raise ValueError("sample_annotation.json must be a list.")
        if not isinstance(instance, list):
            raise ValueError("instance.json must be a list.")
        if not isinstance(category, list):
            raise ValueError("category.json must be a list.")
        if not isinstance(ego_pose, list):
            raise ValueError("ego_pose.json must be a list.")
        if not isinstance(scene, list):
            raise ValueError("scene.json must be a list.")
        if not isinstance(log, list):
            raise ValueError("log.json must be a list.")

        return PipelineState(
            questions=questions,
            scene_level_tasks_by_scene=scene_level_tasks_by_scene,
            sample_data=sample_data,
            sample=sample,
            sample_annotation=sample_annotation,
            instance=instance,
            category=category,
            ego_pose=ego_pose,
            scene=scene,
            log=log,
        )

    def summarize(self) -> dict[str, Any]:
        state = self.load_state()
        return {
            "questions_json": str(self.config.resolved_questions_json()),
            "formatted_scenes_dir": str(self.config.resolved_formatted_scenes_dir()),
            "output_json": str(self.config.resolved_output_json()),
            "questions_output": str(self.config.resolved_questions_output()),
            "question_template_count": len(state.questions.get("tasks", [])),
            "scene_level_scene_count": len(state.scene_level_tasks_by_scene),
            "scene_level_task_count": sum(len(rows) for rows in state.scene_level_tasks_by_scene.values()),
            "nuScenes_tables": {
                "sample_data": len(state.sample_data),
                "sample": len(state.sample),
                "sample_annotation": len(state.sample_annotation),
                "instance": len(state.instance),
                "category": len(state.category),
                "ego_pose": len(state.ego_pose),
                "scene": len(state.scene),
                "log": len(state.log),
            },
        }
