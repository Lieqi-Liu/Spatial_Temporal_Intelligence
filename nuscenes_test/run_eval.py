#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, TypeVar

from PIL import Image
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

os.environ["HF_HOME"] = "/data2/rgao727/hf_cache_store"

MAX_IMAGE_PIXELS = 640 * 640
MAX_MODEL_LEN = 8192
DEFAULT_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"
TRAJ_POINT_TASKS = {
    "TRJ-5": 5,
    "TRJ-6": 4,
}

T = TypeVar("T")


def progress_iter(items: Iterable[T], *, total: int | None = None, desc: str = "") -> Iterator[T]:
    if tqdm is not None:
        yield from tqdm(items, total=total, desc=desc)
        return

    count = 0
    for item in items:
        count += 1
        if total is not None and (count == 1 or count == total or count % 25 == 0):
            label = f"{desc}: " if desc else ""
            print(f"{label}{count}/{total}")
        yield item


def log_progress(message: str) -> None:
    if tqdm is not None:
        tqdm.write(message)
    else:
        print(message)


def batched(items: list[T], batch_size: int) -> Iterator[list[T]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be >= 1")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


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
        "--backend",
        choices=("transformers", "vllm"),
        default="transformers",
        help="Inference backend to use.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=2,
        help="Tensor parallel size for vLLM.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=str,
        default=os.environ.get("HF_HOME", "/data2/rgao727/hf_cache_store"),
        help="Hugging Face cache directory for model downloads.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=0,
        help="If >0, only run first N tasks (quick test mode).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of tasks to process together per generation call.",
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


def get_model_input_device(model: AutoModelForImageTextToText) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_generate_vllm(
    llm: LLM,
    sampling_params: SamplingParams,
    prompt: str,
    images: list[Image.Image],
) -> str:
    request: dict[str, Any] = {"prompt": prompt, "multi_modal_data": {"image": images}}
    outputs = llm.generate(request, sampling_params=sampling_params)
    return outputs[0].outputs[0].text.strip() if outputs else ""


def run_generate_vllm_batch(
    *,
    llm: LLM,
    sampling_params: SamplingParams,
    prompts: list[str],
    image_batches: list[list[Image.Image]],
) -> list[str]:
    requests = [
        {"prompt": prompt, "multi_modal_data": {"image": images}}
        for prompt, images in zip(prompts, image_batches)
    ]
    outputs = llm.generate(requests, sampling_params=sampling_params)
    return [output.outputs[0].text.strip() if output.outputs else "" for output in outputs]


def run_generate_transformers(
    *,
    model: AutoModelForImageTextToText,
    processor: AutoProcessor,
    prompt: str,
    images: list[Image.Image],
) -> str:
    inputs = processor(
        text=[prompt],
        images=images,
        return_tensors="pt",
        padding=True,
    )
    device = get_model_input_device(model)
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    generated = model.generate(
        **inputs,
        max_new_tokens=2048,
        do_sample=False,
    )
    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = generated[:, prompt_len:]
    text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return text[0].strip() if text else ""


def run_generate_transformers_batch(
    *,
    model: AutoModelForImageTextToText,
    processor: AutoProcessor,
    prompts: list[str],
    image_batches: list[list[Image.Image]],
) -> list[str]:
    inputs = processor(
        text=prompts,
        images=image_batches,
        return_tensors="pt",
        padding=True,
    )
    device = get_model_input_device(model)
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    generated = model.generate(
        **inputs,
        max_new_tokens=2048,
        do_sample=False,
    )
    prompt_lens = inputs["attention_mask"].sum(dim=1).tolist()
    generated_ids = [generated[i, int(prompt_lens[i]) :] for i in range(generated.shape[0])]
    text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return [item.strip() for item in text]


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


def _coerce_point(point: Any) -> list[float] | None:
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        return None
    try:
        x = float(point[0])
        y = float(point[1])
    except (TypeError, ValueError):
        return None
    return [x, y]


def extract_traj_points(raw_text: str, expected_len: int) -> list[list[float]] | None:
    text = raw_text.strip()
    if not text:
        return None

    candidates: list[Any] = []
    try:
        candidates.append(json.loads(text))
    except Exception:
        pass

    if not candidates:
        start = text.find("[[")
        end = text.rfind("]]")
        if start != -1 and end != -1 and end >= start:
            snippet = text[start : end + 2]
            try:
                candidates.append(json.loads(snippet))
            except Exception:
                pass

    for candidate in candidates:
        if isinstance(candidate, dict):
            for key in ("trajectory", "answer", "points", "prediction"):
                if key in candidate:
                    candidate = candidate[key]
                    break

        if not isinstance(candidate, list) or len(candidate) != expected_len:
            continue

        points: list[list[float]] = []
        valid = True
        for item in candidate:
            point = _coerce_point(item)
            if point is None:
                valid = False
                break
            points.append(point)
        if valid:
            return points
    return None


def point_distance(a: list[float], b: list[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def compute_ade(pred: list[list[float]], gt: list[list[float]]) -> float:
    return sum(point_distance(p, g) for p, g in zip(pred, gt)) / float(len(gt))


def compute_fde(pred: list[list[float]], gt: list[list[float]]) -> float:
    return point_distance(pred[-1], gt[-1])


def append_result_and_update_metrics(
    *,
    task: dict[str, Any],
    raw_text: str,
    results: list[dict[str, Any]],
    per_task_stats: dict[str, dict[str, Any]],
    numeric_task_stats: dict[str, dict[str, Any]],
    traj_task_stats: dict[str, dict[str, Any]],
) -> tuple[int, int, float]:
    total_mcq_delta = 0
    correct_mcq_delta = 0
    baseline_delta = 0.0

    question_id = str(task.get("id", ""))
    question_format = str(task.get("question_format", ""))
    choices = task.get("choices", {})
    ground_truth = str(task.get("ground_truth", "")).strip().upper()
    predicted_key = extract_answer_key(raw_text, choices=choices)
    predicted_value = None
    ground_truth_value = None
    absolute_error = None
    squared_error = None
    predicted_traj = None
    ground_truth_traj = None
    traj_ade = None
    traj_fde = None
    traj_format_score = None
    is_correct = None
    random_baseline = None

    if question_format == "MCQ" and isinstance(choices, dict) and choices:
        num_options = len(choices)
        random_baseline = 1.0 / float(num_options)
        if ground_truth:
            is_correct = predicted_key == ground_truth
            total_mcq_delta = 1
            baseline_delta = random_baseline
            if is_correct:
                correct_mcq_delta = 1
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
    elif question_id in TRAJ_POINT_TASKS:
        expected_len = TRAJ_POINT_TASKS[question_id]
        predicted_traj = extract_traj_points(raw_text, expected_len=expected_len)
        gt_raw = task.get("ground_truth", "")
        gt_text = gt_raw if isinstance(gt_raw, str) else json.dumps(gt_raw)
        ground_truth_traj = extract_traj_points(gt_text, expected_len=expected_len)
        traj_format_score = 1.0 if (
            predicted_traj is not None and ground_truth_traj is not None
        ) else 0.0
        bucket = traj_task_stats.setdefault(
            question_id,
            {
                "count": 0.0,
                "valid_count": 0.0,
                "format_score_sum": 0.0,
                "ade_sum": 0.0,
                "fde_sum": 0.0,
            },
        )
        bucket["count"] += 1.0
        bucket["format_score_sum"] += traj_format_score
        if predicted_traj is not None and ground_truth_traj is not None:
            traj_ade = compute_ade(predicted_traj, ground_truth_traj)
            traj_fde = compute_fde(predicted_traj, ground_truth_traj)
            bucket["valid_count"] += 1.0
            bucket["ade_sum"] += traj_ade
            bucket["fde_sum"] += traj_fde

    results.append(
        {
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
            "predicted_traj": predicted_traj,
            "ground_truth": ground_truth if ground_truth else None,
            "ground_truth_value": ground_truth_value,
            "ground_truth_traj": ground_truth_traj,
            "is_correct": is_correct,
            "absolute_error": absolute_error,
            "squared_error": squared_error,
            "traj_ade": traj_ade,
            "traj_fde": traj_fde,
            "traj_format_score": traj_format_score,
            "random_baseline": random_baseline,
        }
    )
    return total_mcq_delta, correct_mcq_delta, baseline_delta


def main() -> None:
    args = parse_args()
    os.environ["HF_HOME"] = args.hf_cache_dir
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

    min_pixels = 256 * 28 * 28
    max_pixels = 768 * 28 * 28
    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=True,
        cache_dir=args.hf_cache_dir,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "left"
    llm = None
    sampling_params = None
    model = None
    if args.backend == "vllm":
        if LLM is None or SamplingParams is None:
            raise ImportError("vLLM backend requested, but vllm is not installed in this environment.")
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
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            cache_dir=args.hf_cache_dir,
        )
        model.eval()

    group_cache: dict[str, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    total_mcq = 0
    correct_mcq = 0
    baseline_sum = 0.0
    per_task_stats: dict[str, dict[str, Any]] = {}
    numeric_task_stats: dict[str, dict[str, Any]] = {}
    traj_task_stats: dict[str, dict[str, Any]] = {}
    for task_batch in progress_iter(
        list(batched(tasks, args.batch_size)),
        total=math.ceil(len(tasks) / float(args.batch_size)),
        desc="Running eval",
    ):
        prepared: list[dict[str, Any]] = []
        for task in task_batch:
            try:
                image_paths = resolve_context_images(task, args.formatted_scenes_dir, group_cache)
                images = load_images(image_paths)
                user_message = build_user_message(task)
                prompt = build_prompt(processor, [user_message])
                prepared.append({"task": task, "images": images, "prompt": prompt})
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

        if not prepared:
            continue

        try:
            prompts = [item["prompt"] for item in prepared]
            image_batches = [item["images"] for item in prepared]
            if args.backend == "vllm":
                raw_texts = run_generate_vllm_batch(
                    llm=llm,
                    sampling_params=sampling_params,
                    prompts=prompts,
                    image_batches=image_batches,
                )
            else:
                raw_texts = run_generate_transformers_batch(
                    model=model,
                    processor=processor,
                    prompts=prompts,
                    image_batches=image_batches,
                )
        except Exception as batch_exc:
            raw_texts = []
            log_progress(f"Batch inference failed, retrying individually: {batch_exc}")
            for item in prepared:
                try:
                    if args.backend == "vllm":
                        raw_text = run_generate_vllm(
                            llm=llm,
                            sampling_params=sampling_params,
                            prompt=item["prompt"],
                            images=item["images"],
                        )
                    else:
                        raw_text = run_generate_transformers(
                            model=model,
                            processor=processor,
                            prompt=item["prompt"],
                            images=item["images"],
                        )
                    raw_texts.append(raw_text)
                except Exception as exc:
                    results.append(
                        {
                            "id": item["task"].get("id"),
                            "scene_id": item["task"].get("scene_id"),
                            "group_id": item["task"].get("group_id"),
                            "question": item["task"].get("question", ""),
                            "model_response": f"[ERROR] {exc}",
                        }
                    )
                    raw_texts.append("")

        for item, raw_text in zip(prepared, raw_texts):
            if raw_text.startswith("[ERROR]"):
                continue
            if raw_text == "":
                # Empty string is a valid model output in principle, but if this item already
                # recorded an exception above, skip duplicate result creation.
                existing_error = (
                    results
                    and results[-1].get("id") == item["task"].get("id")
                    and str(results[-1].get("model_response", "")).startswith("[ERROR]")
                )
                if existing_error:
                    continue
            total_delta, correct_delta, baseline_delta = append_result_and_update_metrics(
                task=item["task"],
                raw_text=raw_text,
                results=results,
                per_task_stats=per_task_stats,
                numeric_task_stats=numeric_task_stats,
                traj_task_stats=traj_task_stats,
            )
            total_mcq += total_delta
            correct_mcq += correct_delta
            baseline_sum += baseline_delta

        for item in prepared:
            for im in item["images"]:
                im.close()
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

    traj_task_summary: dict[str, dict[str, Any]] = {}
    for task_id, stats in traj_task_stats.items():
        count = max(stats["count"], 1.0)
        valid_count = max(stats["valid_count"], 1.0)
        traj_task_summary[task_id] = {
            "count": stats["count"],
            "valid_count": stats["valid_count"],
            "format_score_mean": stats["format_score_sum"] / count,
            "ade_mean_valid_only": (
                stats["ade_sum"] / valid_count if stats["valid_count"] > 0 else None
            ),
            "fde_mean_valid_only": (
                stats["fde_sum"] / valid_count if stats["valid_count"] > 0 else None
            ),
        }

    overall_accuracy = (float(correct_mcq) / float(total_mcq)) if total_mcq > 0 else None
    overall_random_baseline = (baseline_sum / float(total_mcq)) if total_mcq > 0 else None

    output = {
        "meta": {
            "created_at": datetime.now().isoformat(),
            "tasks_file": str(args.tasks.resolve()),
            "formatted_scenes_dir": str(args.formatted_scenes_dir.resolve()),
            "model": args.model,
            "backend": args.backend,
            "hf_cache_dir": args.hf_cache_dir,
            "num_tasks": len(tasks),
            "mcq_evaluated_count": total_mcq,
            "mcq_correct_count": correct_mcq,
            "accuracy": overall_accuracy,
            "random_guess_baseline": overall_random_baseline,
            "per_task_metrics": per_task_summary,
            "numeric_metrics": numeric_task_summary,
            "trajectory_metrics": traj_task_summary,
        },
        "results": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    log_progress(f"Saved responses to: {args.output}")


if __name__ == "__main__":
    main()
