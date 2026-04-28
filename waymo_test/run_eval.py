#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

os.environ["HF_HOME"] = "/local1/lieqiliu/huggingface"

MAX_IMAGE_PIXELS = 640 * 640
MAX_MODEL_LEN = 8192
DEFAULT_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run Waymo QA with Qwen VLM using 5-frame temporal context."
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=script_dir / "annotations/questions_with_answers_waymo.json",
        help="Path to generated questions/tasks JSON.",
    )
    parser.add_argument(
        "--distance-json",
        type=Path,
        default=script_dir / "annotations/bbox_distance_estimates.json",
        help="Distance estimates JSON used to resolve frame images and object bbox.",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=script_dir / "images",
        help="Root folder for Waymo images.",
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate pipeline and emit placeholder responses without loading model.",
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


def parse_frame_name(frame_name: str) -> tuple[str, int]:
    if "-" not in frame_name:
        return frame_name, -1
    scene_id, idx_str = frame_name.rsplit("-", 1)
    try:
        return scene_id, int(idx_str)
    except ValueError:
        return scene_id, -1


def parse_object_id(object_id: str) -> tuple[str, int] | None:
    # format: <frame_name>#objXX
    if "#obj" not in object_id:
        return None
    frame_name, obj_idx = object_id.split("#obj", 1)
    try:
        return frame_name, int(obj_idx)
    except ValueError:
        return None


def build_distance_index(
    distance_payload: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[int]]]:
    frame_index: dict[str, dict[str, Any]] = {}
    scene_to_indices: dict[str, list[int]] = {}
    for rec in distance_payload.get("results", []):
        frame_name = str(rec.get("frame_name", ""))
        scene_id, frame_idx = parse_frame_name(frame_name)
        if not frame_name or frame_idx < 0:
            continue
        frame_index[frame_name] = rec
        scene_to_indices.setdefault(scene_id, []).append(frame_idx)
    for scene_id in scene_to_indices:
        scene_to_indices[scene_id] = sorted(set(scene_to_indices[scene_id]))
    return frame_index, scene_to_indices


def select_context_indices(sorted_indices: list[int], query_idx: int, context_len: int = 5) -> list[int]:
    # pick up to context_len-1 previous frames + query frame
    prev = [i for i in sorted_indices if i < query_idx]
    chosen_prev = prev[-(context_len - 1) :]
    chosen = chosen_prev + [query_idx]
    if not chosen:
        return []
    while len(chosen) < context_len:
        chosen.insert(0, chosen[0])
    return chosen


def resolve_context_images(
    task: dict[str, Any],
    frame_index: dict[str, dict[str, Any]],
    scene_to_indices: dict[str, list[int]],
    images_root: Path,
) -> tuple[list[Path], tuple[float, float, float, float] | None]:
    scene_id = str(task.get("scene_id", ""))
    group_id = str(task.get("group_id", "group_-001"))
    if not scene_id:
        raise ValueError("Task missing scene_id.")
    if not group_id.startswith("group_"):
        raise ValueError(f"Invalid group_id format: {group_id}")
    query_idx = int(group_id.split("_")[1])

    indices = scene_to_indices.get(scene_id, [])
    if not indices:
        raise ValueError(f"No frames found for scene_id={scene_id} in distance JSON.")
    if query_idx not in indices:
        # fallback to nearest
        query_idx = min(indices, key=lambda x: abs(x - query_idx))

    chosen = select_context_indices(indices, query_idx=query_idx, context_len=5)
    image_paths: list[Path] = []
    for idx in chosen:
        frame_name = f"{scene_id}-{idx:03d}"
        rec = frame_index.get(frame_name)
        if rec is None:
            raise FileNotFoundError(f"Missing distance record for frame: {frame_name}")
        image_rel = rec.get("image_path")
        if not image_rel:
            raise FileNotFoundError(f"Missing image_path in distance record: {frame_name}")
        p = images_root / str(image_rel)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")
        image_paths.append(p)

    # Resolve selected object bbox for query frame
    bbox = None
    object_id = str(task.get("object_id", ""))
    parsed = parse_object_id(object_id)
    if parsed is not None:
        obj_frame_name, obj_idx = parsed
        rec = frame_index.get(obj_frame_name)
        if rec is not None:
            estimates = rec.get("distance_estimates", [])
            if 0 <= obj_idx < len(estimates):
                b = estimates[obj_idx].get("bbox_xyxy")
                if isinstance(b, list) and len(b) == 4:
                    bbox = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
    return image_paths, bbox


def load_images(image_paths: list[Path], query_bbox: tuple[float, float, float, float] | None) -> list[Image.Image]:
    images: list[Image.Image] = []
    for i, p in enumerate(image_paths):
        img = Image.open(p).convert("RGB")
        if i == len(image_paths) - 1 and query_bbox is not None:
            # Draw selected bbox on the query frame, similar intent to nuScenes selected render.
            draw = ImageDraw.Draw(img)
            x1, y1, x2, y2 = query_bbox
            draw.rectangle([x1, y1, x2, y2], outline=(255, 32, 32), width=4)
            draw.text((x1, max(0.0, y1 - 16.0)), "target", fill=(255, 32, 32))
        images.append(resize_for_vlm(img))
    return images


def build_user_message(task: dict[str, Any]) -> dict[str, Any]:
    image_content = [{"type": "image"} for _ in range(5)]
    object_ref = task.get("object_reference", "the object")
    question_text = task.get("question", "").replace("<obj>", object_ref)
    choices = task.get("choices", {})
    question_format = task.get("question_format", "MCQ")

    if question_format == "MCQ" and choices:
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

    key = text.strip().upper()
    if len(key) == 1 and key.isalpha():
        return key

    match = re.search(r"\b(?:answer|option)?\s*[:\-]?\s*\(?([A-Z])\)?\b", key)
    if match:
        return match.group(1)

    if choices:
        lowered = text.lower()
        for option_key, option_text in choices.items():
            if isinstance(option_text, str) and option_text.strip():
                if option_text.lower() in lowered:
                    return str(option_key).upper()
    return None


def main() -> None:
    args = parse_args()
    if not args.tasks.exists():
        raise FileNotFoundError(f"Tasks file not found: {args.tasks}")
    if not args.distance_json.exists():
        raise FileNotFoundError(f"Distance JSON not found: {args.distance_json}")
    if not args.images_root.exists():
        raise FileNotFoundError(f"Images root not found: {args.images_root}")

    tasks_json = safe_read_json(args.tasks)
    tasks = flatten_tasks(tasks_json)
    if args.max_tasks > 0:
        tasks = tasks[: args.max_tasks]
    if not tasks:
        raise RuntimeError("No tasks found in task JSON.")

    distance_payload = safe_read_json(args.distance_json)
    frame_index, scene_to_indices = build_distance_index(distance_payload)

    processor = None
    llm = None
    sampling_params = None
    if not args.dry_run:
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

    results: list[dict[str, Any]] = []
    total_mcq = 0
    correct_mcq = 0
    baseline_sum = 0.0
    per_task_stats: dict[str, dict[str, Any]] = {}

    for idx, task in enumerate(tasks, start=1):
        try:
            image_paths, query_bbox = resolve_context_images(
                task=task,
                frame_index=frame_index,
                scene_to_indices=scene_to_indices,
                images_root=args.images_root,
            )
            images = load_images(image_paths=image_paths, query_bbox=query_bbox)
            if args.dry_run:
                raw_text = "__dry_run__"
            else:
                user_message = build_user_message(task)
                assert processor is not None and llm is not None and sampling_params is not None
                prompt = build_prompt(processor, [user_message])
                raw_text = run_generate(
                    llm=llm,
                    sampling_params=sampling_params,
                    prompt=prompt,
                    images=images,
                )
            for im in images:
                im.close()

            question_id = str(task.get("id", ""))
            question_format = str(task.get("question_format", ""))
            choices = task.get("choices", {})
            ground_truth = str(task.get("ground_truth", "")).strip().upper()
            predicted_key = None if args.dry_run else extract_answer_key(raw_text, choices=choices)
            is_correct = None
            random_baseline = None

            if question_format == "MCQ" and isinstance(choices, dict) and choices:
                num_options = len(choices)
                random_baseline = 1.0 / float(num_options)
                if ground_truth and not args.dry_run:
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
                "ground_truth": ground_truth if ground_truth else None,
                "is_correct": is_correct,
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

    overall_accuracy = (float(correct_mcq) / float(total_mcq)) if total_mcq > 0 else None
    overall_random_baseline = (baseline_sum / float(total_mcq)) if total_mcq > 0 else None

    output = {
        "meta": {
            "created_at": datetime.now().isoformat(),
            "tasks_file": str(args.tasks.resolve()),
            "distance_json": str(args.distance_json.resolve()),
            "images_root": str(args.images_root.resolve()),
            "model": args.model,
            "num_tasks": len(tasks),
            "dry_run": args.dry_run,
            "mcq_evaluated_count": total_mcq,
            "mcq_correct_count": correct_mcq,
            "accuracy": overall_accuracy,
            "random_guess_baseline": overall_random_baseline,
            "per_task_metrics": per_task_summary,
        },
        "results": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Saved responses to: {args.output}")


if __name__ == "__main__":
    main()
