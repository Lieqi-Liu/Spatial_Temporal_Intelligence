#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

os.environ["HF_HOME"] = "/local1/lieqiliu/huggingface"

MAX_IMAGE_PIXELS = 640 * 640
MAX_MODEL_LEN = 8192
DEFAULT_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run nuScenes QA with a VLM using 5-frame context (ask on frame 5)."
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=script_dir / "questions_with_answers_all.json",
        help="Path to generated questions/tasks JSON.",
    )
    parser.add_argument(
        "--formatted-scenes-dir",
        type=Path,
        default=script_dir / "formatted_scenes",
        help="Path to formatted_scenes directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / f"vlm_responses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        help="Output JSON path for model responses.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Vision-language model name/path.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=2,
        help="Tensor parallel size for vLLM.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=0,
        help="If >0, only run first N tasks (quick test mode).",
    )
    return parser.parse_args()


def resize_for_vlm(image: Image.Image, max_pixels: int = MAX_IMAGE_PIXELS) -> Image.Image:
    width, height = image.size
    current_pixels = width * height
    if current_pixels <= max_pixels:
        return image
    scale = (max_pixels / float(current_pixels)) ** 0.5
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def build_prompt(processor: AutoProcessor, messages: list[dict[str, Any]]) -> str:
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def run_generate(
    llm: LLM,
    sampling_params: SamplingParams,
    prompt: str,
    images: list[Image.Image],
) -> str:
    request: dict[str, Any] = {"prompt": prompt, "multi_modal_data": {"image": images}}
    outputs = llm.generate(request, sampling_params=sampling_params)
    return outputs[0].outputs[0].text.strip() if outputs else ""


def safe_read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def flatten_tasks(tasks_json: dict[str, Any]) -> list[dict[str, Any]]:
    generated = tasks_json.get("generated_answers", {})
    if isinstance(generated, dict) and generated:
        out: list[dict[str, Any]] = []
        for _, payload in generated.items():
            out.extend(payload.get("tasks", []))
        return out
    return tasks_json.get("tasks", [])


def resolve_context_images(
    task: dict[str, Any],
    formatted_scenes_dir: Path,
    group_cache: dict[str, dict[str, Any]],
) -> list[Path]:
    source_group_file = task.get("source_group_file")
    if not source_group_file:
        raise ValueError("Task missing source_group_file.")

    if source_group_file not in group_cache:
        group_path = formatted_scenes_dir / source_group_file
        group_cache[source_group_file] = safe_read_json(group_path)
    group_payload = group_cache[source_group_file]

    scene_id = group_payload["scene_id"]
    frame_indices = group_payload.get("frame_indices_1based", [])
    if len(frame_indices) != 5:
        raise ValueError(f"Expected 5 frame indices, got {len(frame_indices)} for {source_group_file}")

    scene_dir = formatted_scenes_dir / scene_id
    image_paths: list[Path] = []
    for idx in frame_indices:
        matches = sorted(scene_dir.glob(f"{idx:03d}_*_grid.jpg"))
        if not matches:
            raise FileNotFoundError(f"Missing frame image for index {idx} in {scene_dir}")
        image_paths.append(matches[0])
    # For object-level tasks, prefer the pre-rendered annotated query frame that
    # highlights the selected object. Scene-level tasks should use the plain grid.
    group_id = group_payload.get("group_id")
    is_object_level = bool(task.get("object_id")) or task.get("task") not in {
        "scene-context",
        "trajectory-prediction",
    }
    if is_object_level and scene_id and group_id:
        annotated_name = f"{group_id}_selected_vehicle_render.jpg"
        annotated_path = scene_dir / annotated_name
        if annotated_path.exists():
            image_paths[4] = annotated_path

    return image_paths


def load_images(image_paths: list[Path]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        images.append(resize_for_vlm(img))
    return images


def build_user_message(task: dict[str, Any]) -> dict[str, Any]:
    image_content = [{"type": "image"} for _ in range(5)]
    object_ref = task.get("object_reference", "the object")
    question_text = task.get("question", "").replace("<obj>", object_ref)
    choices = task.get("choices", {})
    question_format = task.get("question_format", "MCQ")
    is_scene_level = task.get("task") == "scene-context"
    is_traj_prediction = task.get("task") == "trajectory-prediction"

    if is_traj_prediction and question_format != "MCQ":
        if task.get("id") == "TRJ-6":
            instruction = (
                "You are given 5 consecutive driving frames ending at the current anchor frame. "
                "The first future ego-trajectory point is already provided in the question. "
                "Predict the following 4 future frames.\n"
                "Respond with only a JSON array of exactly 4 points formatted as "
                "[[x2, y2], [x3, y3], [x4, y4], [x5, y5]]."
            )
        elif task.get("id") == "TRJ-7":
            instruction = (
                "You are given 5 consecutive driving frames ending at the current anchor frame. "
                "Briefly describe the likely future ego path over the next 5 future frames. "
                "Mention the overall maneuver, endpoint region, and speed trend in 1-2 sentences."
            )
        else:
            instruction = (
                "You are given 5 consecutive driving frames ending at the current anchor frame. "
                "Predict the ego vehicle trajectory for the next 5 future frames.\n"
                "Respond with only a JSON array of exactly 5 points formatted as "
                "[[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5]]."
            )
    elif is_scene_level and question_format == "MCQ" and choices:
        instruction = (
            "You are given 5 consecutive driving frames from the same scene. "
            "Use the full scene context across the frames to answer the question.\n"
            "Select exactly one option from the provided choices. "
            "Respond with the option key (for example: A)."
        )
    elif is_scene_level:
        instruction = (
            "You are given 5 consecutive driving frames from the same scene. "
            "Use the full scene context across the frames to answer the question.\n"
            "Provide a concise answer in plain text."
        )
    elif question_format == "MCQ" and choices:
        instruction = (
            "You are given 5 consecutive driving frames (first 4 are context, "
            "5th is the query frame for answering and contains a bounding-box "
            "highlight of the target object.\n"
            "Select exactly one option from the provided choices. "
            "Respond with the option key (for example: A)."
        )
    else:
        instruction = (
            "You are given 5 consecutive driving frames (first 4 are context, "
            "5th is the query frame for answering and contains a bounding-box "
            "highlight of the target object.\n"
            "Provide a concise answer in plain text."
        )

    prompt_text = (
        f"Question ID: {task.get('id', '')}\n"
        f"Question: {question_text}\n"
        f"Choices: {json.dumps(choices, ensure_ascii=False)}\n\n"
        f"{instruction}"
    )
    return {"role": "user", "content": image_content + [{"type": "text", "text": prompt_text}]}


def extract_answer_key(raw_text: str, choices: dict[str, Any] | None = None) -> str | None:
    """Extract MCQ option key from model output text."""
    text = raw_text.strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            ans = payload.get("answer")
            if isinstance(ans, str) and ans.strip():
                return ans.strip().upper()
    except Exception:
        pass

    # Fallback 1: raw output might already be a key like "A".
    key = text.strip().upper()
    if len(key) == 1 and key.isalpha():
        return key

    # Fallback 2: parse patterns like "Answer: B", "(C)", "option D".
    match = re.search(r"\b(?:answer|option)?\s*[:\-]?\s*\(?([A-Z])\)?\b", key)
    if match:
        return match.group(1)

    # Fallback 3: match the choice text itself.
    if choices:
        lowered = text.lower()
        for option_key, option_text in choices.items():
            if isinstance(option_text, str) and option_text.strip():
                if option_text.lower() in lowered:
                    return str(option_key).upper()
    return None


def extract_numeric_meters(raw_text: str) -> float | None:
    """Extract the first numeric distance value in meters from model output."""
    text = raw_text.strip()
    if not text:
        return None

    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            for key in ("answer", "distance", "value"):
                val = payload.get(key)
                if isinstance(val, (int, float)):
                    return float(val)
                if isinstance(val, str):
                    text = val.strip()
                    break
    except Exception:
        pass

    match = re.search(r"[-+]?\d*\.?\d+", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def main() -> None:
    args = parse_args()
    if not args.tasks.exists():
        raise FileNotFoundError(f"Tasks file not found: {args.tasks}")
    if not args.formatted_scenes_dir.exists():
        raise FileNotFoundError(f"formatted_scenes directory not found: {args.formatted_scenes_dir}")

    tasks_json = safe_read_json(args.tasks)
    tasks = flatten_tasks(tasks_json)
    if args.max_tasks > 0:
        tasks = tasks[: args.max_tasks]
    if not tasks:
        raise RuntimeError("No tasks found in task JSON.")

    processor = AutoProcessor.from_pretrained(args.model)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.85,
    )
    sampling_params = SamplingParams(
        temperature=0.2,
        top_p=0.9,
        top_k=20,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        max_tokens=2048,
    )

    group_cache: dict[str, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    total_mcq = 0
    correct_mcq = 0
    baseline_sum = 0.0
    per_task_stats: dict[str, dict[str, Any]] = {}
    numeric_task_stats: dict[str, dict[str, Any]] = {}
    for idx, task in enumerate(tasks, start=1):
        try:
            image_paths = resolve_context_images(task, args.formatted_scenes_dir, group_cache)
            images = load_images(image_paths)
            user_message = build_user_message(task)
            prompt = build_prompt(processor, [user_message])
            raw_text = run_generate(llm=llm, sampling_params=sampling_params, prompt=prompt, images=images)
            for im in images:
                im.close()

            question_id = str(task.get("id", ""))
            question_format = str(task.get("question_format", ""))
            choices = task.get("choices", {})
            ground_truth = str(task.get("ground_truth", "")).strip().upper()
            predicted_key = extract_answer_key(raw_text, choices=choices)
            predicted_value = None
            ground_truth_value = None
            absolute_error = None
            squared_error = None
            is_correct = None
            random_baseline = None

            if question_format == "MCQ" and isinstance(choices, dict) and choices:
                num_options = len(choices)
                random_baseline = 1.0 / float(num_options)
                if ground_truth:
                    is_correct = predicted_key == ground_truth
                    total_mcq += 1
                    baseline_sum += random_baseline
                    if is_correct:
                        correct_mcq += 1
                    bucket = per_task_stats.setdefault(
                        question_id,
                        {
                            "total": 0.0,
                            "correct": 0.0,
                            "baseline_sum": 0.0,
                            "ground_truth_distribution": {},
                            "model_answer_distribution": {},
                        },
                    )
                    bucket["total"] += 1.0
                    bucket["baseline_sum"] += random_baseline
                    if is_correct:
                        bucket["correct"] += 1.0
                    gt_key = ground_truth if ground_truth else "__missing__"
                    pred_key = predicted_key if predicted_key else "__invalid__"
                    bucket["ground_truth_distribution"][gt_key] = (
                        bucket["ground_truth_distribution"].get(gt_key, 0) + 1
                    )
                    bucket["model_answer_distribution"][pred_key] = (
                        bucket["model_answer_distribution"].get(pred_key, 0) + 1
                    )
            elif question_id == "SP-3a":
                predicted_value = extract_numeric_meters(raw_text)
                ground_truth_value = extract_numeric_meters(str(task.get("ground_truth", "")))
                if predicted_value is not None and ground_truth_value is not None:
                    absolute_error = abs(predicted_value - ground_truth_value)
                    squared_error = (predicted_value - ground_truth_value) ** 2
                    bucket = numeric_task_stats.setdefault(
                        question_id,
                        {
                            "count": 0.0,
                            "sum_abs_error": 0.0,
                            "sum_sq_error": 0.0,
                        },
                    )
                    bucket["count"] += 1.0
                    bucket["sum_abs_error"] += absolute_error
                    bucket["sum_sq_error"] += squared_error

            result = {
                "id": question_id,
                "scene_id": task.get("scene_id"),
                "group_id": task.get("group_id"),
                "question": task.get("question", "").replace("<obj>", task.get("object_reference", "the object")),
                "choices": choices,
                "object_id": task.get("object_id"),
                "object_reference": task.get("object_reference"),
                "model_response": raw_text,
                "predicted_option": predicted_key,
                "predicted_value": predicted_value,
                "ground_truth": ground_truth if ground_truth else None,
                "ground_truth_value": ground_truth_value,
                "is_correct": is_correct,
                "absolute_error": absolute_error,
                "squared_error": squared_error,
                "random_baseline": random_baseline,
            }
            results.append(result)
            status = "ok"
        except Exception as exc:
            results.append(
                {
                    "id": task.get("id"),
                    "scene_id": task.get("scene_id"),
                    "group_id": task.get("group_id"),
                    "question": task.get("question", ""),
                    "model_response": f"[ERROR] {exc}",
                }
            )
            status = "failed"

        print(f"[{idx}/{len(tasks)}] {task.get('id')} {task.get('scene_id')}:{task.get('group_id')} -> {status}")

    per_task_summary: dict[str, dict[str, Any]] = {}
    for task_id, stats in per_task_stats.items():
        total = max(stats["total"], 1.0)
        per_task_summary[task_id] = {
            "count": stats["total"],
            "accuracy": stats["correct"] / total,
            "random_baseline": stats["baseline_sum"] / total,
            "ground_truth_distribution": stats["ground_truth_distribution"],
            "model_answer_distribution": stats["model_answer_distribution"],
        }

    numeric_task_summary: dict[str, dict[str, Any]] = {}
    for task_id, stats in numeric_task_stats.items():
        count = max(stats["count"], 1.0)
        mse = stats["sum_sq_error"] / count
        numeric_task_summary[task_id] = {
            "count": stats["count"],
            "mae": stats["sum_abs_error"] / count,
            "rmse": math.sqrt(mse),
            "mse": mse,
        }

    overall_accuracy = (float(correct_mcq) / float(total_mcq)) if total_mcq > 0 else None
    overall_random_baseline = (baseline_sum / float(total_mcq)) if total_mcq > 0 else None

    output = {
        "meta": {
            "created_at": datetime.now().isoformat(),
            "tasks_file": str(args.tasks.resolve()),
            "formatted_scenes_dir": str(args.formatted_scenes_dir.resolve()),
            "model": args.model,
            "num_tasks": len(tasks),
            "mcq_evaluated_count": total_mcq,
            "mcq_correct_count": correct_mcq,
            "accuracy": overall_accuracy,
            "random_guess_baseline": overall_random_baseline,
            "per_task_metrics": per_task_summary,
            "numeric_metrics": numeric_task_summary,
        },
        "results": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Saved responses to: {args.output}")


if __name__ == "__main__":
    main()
