from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AnswerGenerationConfig:
    root: Path
    version: str = "v1.0-mini"
    formatted_scenes_dir: Path | None = None
    questions_json: Path | None = None
    output_json: Path | None = None
    questions_output: Path | None = None
    scene_level_tasks_json: Path | None = None
    enable_traj_prediction_tasks: bool = False
    disable_frq_generation: bool = False

    def version_dir(self) -> Path:
        return self.root / self.version

    def resolved_formatted_scenes_dir(self) -> Path:
        if self.formatted_scenes_dir is not None:
            return self.formatted_scenes_dir.resolve()
        return (self.root / "formatted_scenes").resolve()

    def resolved_questions_json(self) -> Path:
        if self.questions_json is not None:
            return self.questions_json.resolve()
        return (self.root / "questions.json").resolve()

    def resolved_output_json(self) -> Path:
        if self.output_json is not None:
            return self.output_json.resolve()
        return (self.resolved_formatted_scenes_dir() / "generated_answers_all.json").resolve()

    def resolved_questions_output(self) -> Path:
        if self.questions_output is not None:
            return self.questions_output.resolve()
        return (self.root / "questions_with_answers_all.json").resolve()
