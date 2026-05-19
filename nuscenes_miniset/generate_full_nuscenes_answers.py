#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
QUESTION_TEMPLATES = THIS_DIR / "questions.json"

NUSCENES_ROOT = Path("/local1/lieqiliu/nuscenes/fullset")
FORMATTED_SCENES_DIR = NUSCENES_ROOT / "formatted_scenes"
GT_GENERATOR = Path("/home/lieqiliu/AutoDriving/nuscenes/generate_answers.py")
VLM_ANNOTATOR = REPO_ROOT / "lieqiliu" / "run_fake_full_vlm_batch.py"

GT_RAW_JSON = FORMATTED_SCENES_DIR / "generated_answers_all.json"
GT_QUESTIONS_JSON = NUSCENES_ROOT / "questions_with_answers_all.json"
TRJ_GT_JSON = NUSCENES_ROOT / "trajectory_tasks.json"
SC_GT_JSON = NUSCENES_ROOT / "scene_level_tasks.json"
GT_FLAT_JSON = NUSCENES_ROOT / "questions_with_answers_all_flat.json"
QWEN_OUTPUT_JSON = NUSCENES_ROOT / "questions_with_answers_all_qwen3vl30b.json"

QWEN30B = "Qwen/Qwen3-VL-30B-A3B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate full NuScenes ground-truth answers from formatted_scenes, then annotate all "
            "questions with Qwen3-VL-30B model responses."
        )
    )
    parser.add_argument("--questions-json", type=Path, default=QUESTION_TEMPLATES)
    parser.add_argument("--nuscenes-root", type=Path, default=NUSCENES_ROOT)
    parser.add_argument("--formatted-scenes-dir", type=Path, default=FORMATTED_SCENES_DIR)
    parser.add_argument("--gt-generator", type=Path, default=GT_GENERATOR)
    parser.add_argument("--vlm-annotator", type=Path, default=VLM_ANNOTATOR)
    parser.add_argument("--gt-raw-json", type=Path, default=GT_RAW_JSON)
    parser.add_argument("--gt-questions-json", type=Path, default=GT_QUESTIONS_JSON)
    parser.add_argument("--trj-gt-json", type=Path, default=TRJ_GT_JSON)
    parser.add_argument("--scene-gt-json", type=Path, default=SC_GT_JSON)
    parser.add_argument("--gt-flat-json", type=Path, default=GT_FLAT_JSON)
    parser.add_argument("--output-json", type=Path, default=QWEN_OUTPUT_JSON)
    parser.add_argument("--model", default=QWEN30B)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--hf-home", type=Path, default=Path("/local1/lieqiliu/huggingface"))
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--save-every", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--mcq-max-tokens", type=int, default=8)
    parser.add_argument("--frq-max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tasks", type=int, default=0, help="If >0, annotate only first N rows.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rerun-errors", action="store_true")
    parser.add_argument("--force-ground-truth", action="store_true")
    parser.add_argument("--skip-ground-truth", action="store_true")
    parser.add_argument("--skip-model-response", action="store_true")
    parser.add_argument(
        "--allow-downloads",
        action="store_true",
        help="Allow HF downloads. By default the model must already be cached locally.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp.replace(path)


def normalize_questions_for_gt_generator(questions_json: Path, output_path: Path) -> None:
    payload = read_json(questions_json)
    tasks = payload.get("tasks") or payload.get("questions")
    if not isinstance(tasks, list) or not tasks:
        raise RuntimeError(f"{questions_json} must contain a non-empty 'questions' or 'tasks' list.")

    # The current GT generator supports SP/SU/TE/TM object-centric tasks.
    # TRJ/SC templates remain in questions.json for the full benchmark definition,
    # but need their own GT generator before they can be included here.
    supported_prefixes = ("SP-", "SU-", "TE-", "TM-")
    supported = [dict(task) for task in tasks if str(task.get("id", "")).startswith(supported_prefixes)]
    for task in supported:
        if "choices" not in task:
            task["choices"] = None

    out = {
        "meta": {
            "source_questions_json": str(questions_json),
            "note": "Compatibility file for AutoDriving/nuscenes/generate_answers.py.",
            "unsupported_families_not_generated_here": ["TRJ", "SC"],
            "supported_task_count": len(supported),
        },
        "tasks": supported,
    }
    write_json(output_path, out)


def relative_group_file(group_file: Path, formatted_scenes_dir: Path) -> str:
    return str(group_file.relative_to(formatted_scenes_dir))


def infer_question_format(row: dict[str, Any]) -> str:
    if row.get("question_format"):
        return str(row["question_format"])
    choices = row.get("choices")
    return "MCQ" if isinstance(choices, dict) and choices else "FRQ"


def normalize_extra_task(row: dict[str, Any], *, task_id: str | None = None) -> dict[str, Any]:
    out = dict(row)
    if task_id is None:
        task_id = str(out.get("id") or out.get("question_id") or "")
    out["id"] = task_id
    out["question_format"] = infer_question_format(out)
    out.setdefault("task", "trajectory-prediction" if task_id.startswith("TRJ-") else "scene-context")
    out.setdefault("model_response", "")
    out.pop("question_id", None)
    return out


def materialize_question(row: dict[str, Any]) -> dict[str, Any]:
    """Replace template placeholders with row-specific text before VLM annotation."""
    out = dict(row)
    question = str(out.get("question", ""))
    # Do not leak semantic object descriptions like "vehicle front-left at 41m" into the prompt.
    # The current formatted scene metadata provides red-box renders, not explicit 2D bbox coordinates.
    question = question.replace("<obj>", "the object in the red bounding box")
    out["question"] = question
    return out


def scene_group_lookup(formatted_scenes_dir: Path) -> dict[str, tuple[str, str]]:
    lookup: dict[str, tuple[str, str]] = {}
    for group_file in sorted(formatted_scenes_dir.glob("scene_*/group_*_vehicle_annotations.json")):
        scene_id = group_file.parent.name
        if scene_id in lookup:
            continue
        try:
            payload = read_json(group_file)
            group_id = str(payload.get("group_id") or group_file.stem.replace("_vehicle_annotations", ""))
        except Exception:
            group_id = group_file.stem.replace("_vehicle_annotations", "")
        lookup[scene_id] = (group_id, relative_group_file(group_file, formatted_scenes_dir))
    return lookup


def generate_trajectory_gt(
    *,
    nuscenes_root: Path,
    version: str,
    formatted_scenes_dir: Path,
    output_json: Path,
) -> None:
    sys.path.insert(0, str(THIS_DIR))
    from traj_prediction_tasks import generate_traj_prediction_rows

    version_dir = nuscenes_root / version
    sample_rows = read_json(version_dir / "sample.json")
    sample_data_rows = read_json(version_dir / "sample_data.json")
    ego_pose_rows = read_json(version_dir / "ego_pose.json")

    sample_by_token = {row["token"]: row for row in sample_rows}
    ego_pose_by_token = {row["token"]: row for row in ego_pose_rows}
    cam_front_sd_by_sample = {
        row["sample_token"]: row
        for row in sample_data_rows
        if str(row.get("filename", "")).startswith("samples/CAM_FRONT/")
        and str(row.get("filename", "")).endswith(".jpg")
    }

    rows: list[dict[str, Any]] = []
    skipped = Counter()
    for group_file in sorted(formatted_scenes_dir.glob("scene_*/group_*_vehicle_annotations.json")):
        payload = read_json(group_file)
        payload.setdefault("source_group_file", relative_group_file(group_file, formatted_scenes_dir))
        generated = generate_traj_prediction_rows(
            payload=payload,
            sample_by_token=sample_by_token,
            ego_pose_by_token=ego_pose_by_token,
            cam_front_sd_by_sample=cam_front_sd_by_sample,
        )
        if generated is None:
            skipped["missing_past_or_future_context"] += 1
            continue
        for task_id in sorted(generated):
            rows.append(normalize_extra_task(generated[task_id], task_id=task_id))

    write_json(
        output_json,
        {
            "meta": {
                "source": "traj_prediction_tasks.py",
                "formatted_scenes_dir": str(formatted_scenes_dir),
                "task_count": len(rows),
                "skipped_groups": dict(skipped),
            },
            "tasks": rows,
        },
    )
    print(f"Wrote trajectory GT tasks: {output_json} ({len(rows)} rows)")


def generate_scene_gt(
    *,
    nuscenes_root: Path,
    version: str,
    formatted_scenes_dir: Path,
    output_json: Path,
) -> None:
    run_command(
        [
            sys.executable,
            str(THIS_DIR / "generate_scene_level_gt.py"),
            "--version-dir",
            str(nuscenes_root / version),
            "--output",
            str(output_json),
        ]
    )
    rows = read_json(output_json)
    if not isinstance(rows, list):
        raise RuntimeError(f"{output_json} should contain a list of scene-level tasks.")

    group_lookup = scene_group_lookup(formatted_scenes_dir)
    augmented = []
    missing_visual_group = 0
    for row in rows:
        task = normalize_extra_task(row)
        scene_id = str(task.get("scene_id", ""))
        group_info = group_lookup.get(scene_id)
        if group_info is not None:
            task.setdefault("group_id", group_info[0])
            task.setdefault("source_group_file", group_info[1])
        else:
            missing_visual_group += 1
        augmented.append(task)

    write_json(
        output_json,
        {
            "meta": {
                "source": "generate_scene_level_gt.py",
                "task_count": len(augmented),
                "missing_visual_group": missing_visual_group,
            },
            "tasks": augmented,
        },
    )
    print(f"Wrote scene-level GT tasks: {output_json} ({len(augmented)} rows)")


def flatten_generated_answers(
    gt_questions_json: Path,
    flat_output_json: Path,
    *,
    trj_json: Path | None = None,
    scene_json: Path | None = None,
) -> None:
    payload = read_json(gt_questions_json)
    generated = payload.get("generated_answers", {})
    if not isinstance(generated, dict) or not generated:
        raise RuntimeError(f"{gt_questions_json} does not contain generated_answers.")

    tasks: list[dict[str, Any]] = []
    for task_id in sorted(generated):
        bucket = generated[task_id]
        for row in bucket.get("tasks", []):
            tasks.append(materialize_question(row))

    extra_sources = {}
    for label, path in (("trajectory", trj_json), ("scene", scene_json)):
        if path is None or not path.exists():
            continue
        extra_payload = read_json(path)
        extra_rows = extra_payload.get("tasks", extra_payload if isinstance(extra_payload, list) else [])
        if not isinstance(extra_rows, list):
            continue
        for row in extra_rows:
            tasks.append(materialize_question(normalize_extra_task(row)))
        extra_sources[label] = {"path": str(path), "count": len(extra_rows)}

    summary = {
        "total_tasks": len(tasks),
        "by_question_id": dict(sorted(Counter(row.get("id") for row in tasks).items())),
        "by_question_format": dict(sorted(Counter(row.get("question_format") for row in tasks).items())),
    }
    flat_payload = {
        "meta": {
            "source_gt_questions_json": str(gt_questions_json),
            "extra_sources": extra_sources,
            "summary": summary,
        },
        "tasks": tasks,
    }
    write_json(flat_output_json, flat_payload)


def run_command(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()

    gt_questions_for_generator = THIS_DIR / "questions_for_generate_answers.json"

    if not args.skip_ground_truth:
        normalize_questions_for_gt_generator(args.questions_json, gt_questions_for_generator)
        if args.force_ground_truth or not args.gt_questions_json.exists():
            run_command(
                [
                    sys.executable,
                    str(args.gt_generator),
                    "--root",
                    str(args.nuscenes_root),
                    "--version",
                    args.version,
                    "--formatted-scenes-dir",
                    str(args.formatted_scenes_dir),
                    "--questions-json",
                    str(gt_questions_for_generator),
                    "--output-json",
                    str(args.gt_raw_json),
                    "--questions-output",
                    str(args.gt_questions_json),
                    "--frq-model",
                    QWEN30B,
                    "--frq-tensor-parallel-size",
                    str(args.tensor_parallel_size),
                    "--frq-max-tokens",
                    str(args.frq_max_tokens),
                ]
            )
        else:
            print(f"Ground-truth file exists; reusing {args.gt_questions_json}")

        if args.force_ground_truth or not args.trj_gt_json.exists():
            generate_trajectory_gt(
                nuscenes_root=args.nuscenes_root,
                version=args.version,
                formatted_scenes_dir=args.formatted_scenes_dir,
                output_json=args.trj_gt_json,
            )
        else:
            print(f"Trajectory GT file exists; reusing {args.trj_gt_json}")

        if args.force_ground_truth or not args.scene_gt_json.exists():
            generate_scene_gt(
                nuscenes_root=args.nuscenes_root,
                version=args.version,
                formatted_scenes_dir=args.formatted_scenes_dir,
                output_json=args.scene_gt_json,
            )
        else:
            print(f"Scene-level GT file exists; reusing {args.scene_gt_json}")

    flatten_generated_answers(
        args.gt_questions_json,
        args.gt_flat_json,
        trj_json=args.trj_gt_json,
        scene_json=args.scene_gt_json,
    )
    flat_payload = read_json(args.gt_flat_json)
    flat_summary = flat_payload.get("meta", {}).get("summary", {})
    print(
        f"Wrote flat GT task file: {args.gt_flat_json}\n"
        f"Annotation workload: {flat_summary.get('total_tasks', 0)} total tasks; "
        f"{flat_summary.get('by_question_format', {})}"
    )

    if args.skip_model_response:
        return

    command = [
        sys.executable,
        str(args.vlm_annotator),
        "--tasks",
        str(args.gt_flat_json),
        "--output",
        str(args.output_json),
        "--formatted-scenes-dir",
        str(args.formatted_scenes_dir),
        "--nuscenes-root",
        str(args.nuscenes_root),
        "--version",
        args.version,
        "--model",
        args.model,
        "--hf-home",
        str(args.hf_home),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--batch-size",
        str(args.batch_size),
        "--save-every",
        str(args.save_every),
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--mcq-max-tokens",
        str(args.mcq_max_tokens),
        "--frq-max-tokens",
        str(args.frq_max_tokens),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--resume",
    ]
    if args.rerun_errors:
        command.append("--rerun-errors")
    if args.max_tasks > 0:
        command.extend(["--max-tasks", str(args.max_tasks)])
    if args.start_index > 0:
        command.extend(["--start-index", str(args.start_index)])
    if args.allow_downloads:
        command.append("--no-local-files-only")

    run_command(command)


if __name__ == "__main__":
    main()
