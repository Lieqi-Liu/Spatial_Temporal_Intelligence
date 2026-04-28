from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..common.generation_config import AnswerGenerationConfig
from ..workflows.generation_pipeline import AnswerGenerationPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize inputs for the new modular answer-generation flow."
    )
    parser.add_argument("--root", type=Path, required=True, help="Path to the nuScenes data root.")
    parser.add_argument("--version", type=str, default="v1.0-mini")
    parser.add_argument("--formatted-scenes-dir", type=Path, default=None)
    parser.add_argument("--questions-json", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--questions-output", type=Path, default=None)
    parser.add_argument("--scene-level-tasks-json", type=Path, default=None)
    parser.add_argument("--enable-traj-prediction-tasks", action="store_true")
    parser.add_argument("--disable-frq-generation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AnswerGenerationConfig(
        root=args.root.resolve(),
        version=args.version,
        formatted_scenes_dir=args.formatted_scenes_dir,
        questions_json=args.questions_json,
        output_json=args.output_json,
        questions_output=args.questions_output,
        scene_level_tasks_json=args.scene_level_tasks_json,
        enable_traj_prediction_tasks=args.enable_traj_prediction_tasks,
        disable_frq_generation=args.disable_frq_generation,
    )
    pipeline = AnswerGenerationPipeline(config)
    print(json.dumps(pipeline.summarize(), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
