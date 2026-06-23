#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BENCHMARK_JSON = SCRIPT_DIR / "waymo_validation_miniset_100_per_question_type_with_gt.json"
DEFAULT_OUTPUT_JSON = SCRIPT_DIR / "waymo_validation_miniset_vlm_responses.json"
DEFAULT_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"
DEFAULT_HF_HOME = Path("/local1/rgao727/huggingface")
MAX_IMAGE_PIXELS = 640 * 640
MAX_MODEL_LEN = 8192


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VLM inference on the Waymo validation miniset.")
    parser.add_argument("--benchmark-json", type=Path, default=DEFAULT_BENCHMARK_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--backend", choices=["vllm", "transformers_llava"], default="vllm")
    parser.add_argument("--hf-home", type=Path, default=DEFAULT_HF_HOME)
    parser.add_argument("--gpu-ids", type=str, default="4,5")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens-mcq", type=int, default=16)
    parser.add_argument("--max-tokens-frq", type=int, default=256)
    parser.add_argument("--torch-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 means run all remaining questions.")
    parser.add_argument("--mcq-only", action="store_true", help="Only run MCQ questions.")
    parser.add_argument("--task-ids", type=str, default="", help="Comma-separated ids, e.g. SC-1,TRJ-8.")
    parser.add_argument("--flush-every", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--score-after-run", action="store_true", help="Run scoring after writing responses.")
    parser.add_argument("--score-output-json", type=Path, default=None)
    parser.add_argument("--score-output-md", type=Path, default=None)
    parser.add_argument("--score-output-csv", type=Path, default=None)
    parser.add_argument("--score-source", choices=["auto", "benchmark", "responses"], default="benchmark")
    parser.add_argument("--enable-bleurt", action="store_true", help="Also score text FRQ with BLEURT.")
    parser.add_argument("--bleurt-model", type=str, default="Elron/bleurt-base-512")
    parser.add_argument("--bleurt-batch-size", type=int, default=16)
    parser.add_argument("--bleurt-device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--bleurt-local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use only locally cached BLEURT files unless set to --no-bleurt-local-files-only.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def resize_for_vlm(image: Image.Image, max_pixels: int = MAX_IMAGE_PIXELS) -> Image.Image:
    width, height = image.size
    if width * height <= max_pixels:
        return image
    scale = (max_pixels / float(width * height)) ** 0.5
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def load_images(image_paths: list[str]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for image_path in image_paths:
        path = Path(image_path)
        if path.exists():
            images.append(resize_for_vlm(Image.open(path).convert("RGB")))
    return images


def make_llava_contact_sheet(images: list[Image.Image], tile_height: int = 336, gap: int = 8) -> Image.Image | None:
    if not images:
        return None
    resized: list[Image.Image] = []
    for image in images:
        width, height = image.size
        if height <= 0:
            continue
        scale = tile_height / float(height)
        tile_width = max(1, int(width * scale))
        resized.append(image.resize((tile_width, tile_height), Image.Resampling.LANCZOS))
    if not resized:
        return None
    canvas_width = sum(image.width for image in resized) + gap * (len(resized) - 1)
    canvas = Image.new("RGB", (canvas_width, tile_height), "black")
    x = 0
    for image in resized:
        canvas.paste(image, (x, 0))
        x += image.width + gap
    return canvas


def resolve_hf_cache_root(model_id: str, hf_home: Path) -> Path:
    model_dir = f"models--{model_id.replace('/', '--')}"
    candidates = [
        hf_home / "hub",
        hf_home,
        Path.home() / ".cache" / "huggingface" / "hub",
        Path.home() / ".cache" / "huggingface",
    ]
    for candidate in candidates:
        if (candidate / model_dir).exists():
            return candidate
    return hf_home / "hub"


def choices_text(choices: dict[str, str]) -> str:
    return "\n".join(f"{key}. {value}" for key, value in choices.items())


def visual_context_description(question: dict[str, Any]) -> str:
    hidden = question.get("hidden_metadata") if isinstance(question.get("hidden_metadata"), dict) else {}
    frame_count = len(question.get("history_image_paths") or question.get("image_paths") or []) or 5
    seconds = hidden.get("approx_context_window_seconds")
    stride = hidden.get("context_frame_stride")
    if isinstance(seconds, (int, float)) and seconds > 0:
        return f"You are given {frame_count} front-view frames sampled over approximately {seconds:.1f} seconds from a Waymo validation driving clip.\n"
    if isinstance(stride, int) and stride > 1:
        return f"You are given {frame_count} front-view frames sampled with a stride of {stride} raw frames from a Waymo validation driving clip.\n"
    return f"You are given {frame_count} front-view frames from a Waymo validation driving clip.\n"


def build_prompt_text(question: dict[str, Any]) -> str:
    q = str(question.get("question", "")).strip()
    qid = str(question.get("id", "")).strip()
    qf = str(question.get("question_format", "")).upper()
    choices = question.get("choices") if isinstance(question.get("choices"), dict) else {}
    context_line = visual_context_description(question)

    if qf == "MCQ" and choices:
        return (
            context_line +
            "Answer the multiple-choice question using only the visible frames and the question text.\n"
            "Respond with exactly one option key, such as A, B, C, or D. Do not include explanation.\n\n"
            f"Question ID: {qid}\n"
            f"Question: {q}\n\n"
            f"Choices:\n{choices_text(choices)}\n\n"
            "Final answer:"
        )

    return (
        context_line +
        "Answer the question concisely using only the visible frames and the question text.\n\n"
        f"Question ID: {qid}\n"
        f"Question: {q}\n\n"
        "Final answer:"
    )


def build_llava_prompt(prompt_text: str) -> str:
    contact_sheet_note = (
        "The image is a left-to-right contact sheet of the visible driving frames in temporal order; "
        "the leftmost frame is earliest and the rightmost frame is latest.\n"
    )
    return f"USER: <image>\n{contact_sheet_note}{prompt_text}\nASSISTANT:"


def torch_dtype_from_arg(value: str) -> Any:
    import torch

    if value == "float16":
        return torch.float16
    if value == "bfloat16":
        return torch.bfloat16
    if value == "float32":
        return torch.float32
    return "auto"


def first_model_device(model: Any) -> Any:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return getattr(model, "device", "cuda")


def extract_answer_key(text: str, choices: dict[str, str]) -> str | None:
    valid = {str(k).strip().upper() for k in choices}
    raw = str(text or "").strip()
    upper = raw.upper()
    if upper in valid:
        return upper
    for pattern in [
        r"(?:ANSWER|OPTION|CHOICE|FINAL ANSWER)\s*(?:IS|:)?\s*[\(\[]?([A-Z])[\)\].:]?",
        r"^\s*[\(\[]?([A-Z])[\)\].:]?",
    ]:
        match = re.search(pattern, upper)
        if match and match.group(1) in valid:
            return match.group(1)
    for key, value in sorted(choices.items(), key=lambda kv: len(str(kv[1])), reverse=True):
        if str(value).strip().lower() in raw.lower():
            return str(key).strip().upper()
    return None


def main() -> None:
    args = parse_args()
    payload = read_json(args.benchmark_json)
    questions = list(payload.get("questions", []))

    allowed_ids = {x.strip() for x in args.task_ids.split(",") if x.strip()}
    if args.mcq_only:
        questions = [q for q in questions if str(q.get("question_format", "")).upper() == "MCQ"]
    if allowed_ids:
        questions = [q for q in questions if str(q.get("id", "")) in allowed_ids]
    if args.limit > 0:
        questions = questions[args.offset : args.offset + args.limit]
    else:
        questions = questions[args.offset :]

    if not questions:
        raise RuntimeError("No questions selected.")

    processor = None
    llm = None
    model = None
    if not args.dry_run:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        cache_root = resolve_hf_cache_root(args.model, args.hf_home)
        if args.backend == "vllm":
            from transformers import AutoProcessor
            from vllm import LLM, SamplingParams

            processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, cache_dir=str(cache_root))
            llm = LLM(
                model=args.model,
                trust_remote_code=True,
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
                download_dir=str(cache_root),
            )
            mcq_sampling = SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens_mcq,
            )
            frq_sampling = SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens_frq,
            )
        elif args.backend == "transformers_llava":
            from transformers import AutoProcessor, LlavaForConditionalGeneration

            processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, cache_dir=str(cache_root))
            model = LlavaForConditionalGeneration.from_pretrained(
                args.model,
                torch_dtype=torch_dtype_from_arg(args.torch_dtype),
                device_map="auto",
                trust_remote_code=True,
                cache_dir=str(cache_root),
            )
            model.eval()
            mcq_sampling = frq_sampling = None
        else:
            raise ValueError(f"Unsupported backend: {args.backend}")
    else:
        mcq_sampling = frq_sampling = None

    responses: list[dict[str, Any]] = []
    for idx, question in enumerate(questions, start=1):
        qf = str(question.get("question_format", "")).upper()
        choices = question.get("choices") if isinstance(question.get("choices"), dict) else {}
        image_paths = list(question.get("history_image_paths") or question.get("image_paths") or [])
        prompt_text = build_prompt_text(question)

        if args.dry_run:
            raw_text = "__dry_run__"
        else:
            assert processor is not None
            images = load_images(image_paths)
            if args.backend == "vllm":
                assert llm is not None
                messages = [
                    {
                        "role": "user",
                        "content": (
                            [{"type": "image", "image": image} for image in images]
                            + [{"type": "text", "text": prompt_text}]
                        ),
                    }
                ]
                prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                sampling = mcq_sampling if qf == "MCQ" else frq_sampling
                outputs = llm.generate({"prompt": prompt, "multi_modal_data": {"image": images}}, sampling_params=sampling)
                raw_text = outputs[0].outputs[0].text.strip() if outputs else ""
            elif args.backend == "transformers_llava":
                assert model is not None
                import torch

                contact_sheet = make_llava_contact_sheet(images)
                if contact_sheet is None:
                    raw_text = "[ERROR] Missing image input"
                else:
                    prompt = build_llava_prompt(prompt_text)
                    inputs = processor(text=prompt, images=contact_sheet, return_tensors="pt")
                    device = first_model_device(model)
                    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
                    max_new_tokens = args.max_tokens_mcq if qf == "MCQ" else args.max_tokens_frq
                    with torch.inference_mode():
                        generate_kwargs = {
                            "max_new_tokens": max_new_tokens,
                            "do_sample": args.temperature > 0,
                        }
                        if args.temperature > 0:
                            generate_kwargs["temperature"] = args.temperature
                            generate_kwargs["top_p"] = args.top_p
                        generated = model.generate(**inputs, **generate_kwargs)
                    input_len = int(inputs["input_ids"].shape[-1])
                    raw_text = processor.decode(generated[0][input_len:], skip_special_tokens=True).strip()
                    contact_sheet.close()
            else:
                raise ValueError(f"Unsupported backend: {args.backend}")
            for image in images:
                image.close()

        predicted_option = extract_answer_key(raw_text, choices) if qf == "MCQ" and choices else None
        gt = str(question.get("ground_truth", "")).strip().upper()
        responses.append(
            {
                "question_id": question.get("question_id"),
                "id": question.get("id"),
                "task": question.get("task"),
                "question_format": question.get("question_format"),
                "scene_id": question.get("scene_id"),
                "bundle_id": question.get("bundle_id"),
                "question": question.get("question"),
                "choices": choices,
                "ground_truth": question.get("ground_truth"),
                "ground_truth_text": question.get("ground_truth_text"),
                "model_response": raw_text,
                "predicted_option": predicted_option,
                "is_correct": (predicted_option == gt) if predicted_option and gt else None,
            }
        )

        print(f"[{idx}/{len(questions)}] {question.get('id')} {question.get('question_id')} -> {predicted_option or 'FRQ/invalid'}")
        if args.flush_every > 0 and idx % args.flush_every == 0:
            write_json(args.output_json, {"metadata": output_metadata(args, len(questions)), "responses": responses})

    write_json(args.output_json, {"metadata": output_metadata(args, len(questions)), "responses": responses})
    print(json.dumps({"output_json": str(args.output_json), "responses": len(responses)}, indent=2))
    if args.score_after_run or args.enable_bleurt:
        processor = None
        llm = None
        model = None
        run_scoring(args)


def output_metadata(args: argparse.Namespace, count: int) -> dict[str, Any]:
    return {
        "created_at": datetime.now().isoformat(),
        "benchmark_json": str(args.benchmark_json),
        "model": args.model,
        "backend": args.backend,
        "question_count": count,
        "mcq_only": args.mcq_only,
        "task_ids": args.task_ids,
        "offset": args.offset,
        "limit": args.limit,
    }


def default_score_outputs(output_json: Path) -> tuple[Path, Path, Path]:
    stem = output_json.with_suffix("")
    return (
        stem.with_name(f"{stem.name}_metrics.json"),
        stem.with_name(f"{stem.name}_report.md"),
        stem.with_name(f"{stem.name}_predictions.csv"),
    )


def run_scoring(args: argparse.Namespace) -> None:
    default_json, default_md, default_csv = default_score_outputs(args.output_json)
    output_json = args.score_output_json or default_json
    output_md = args.score_output_md or default_md
    output_csv = args.score_output_csv or default_csv

    # vLLM can hold GPU memory until the objects are collected. Release what we can
    # before launching BLEURT/scoring as a child process.
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "score_waymo_miniset_mcq.py"),
        "--benchmark-json",
        str(args.benchmark_json),
        "--responses-json",
        str(args.output_json),
        "--output-json",
        str(output_json),
        "--output-md",
        str(output_md),
        "--output-csv",
        str(output_csv),
        "--score-source",
        args.score_source,
    ]
    if args.enable_bleurt:
        cmd.extend(
            [
                "--enable-bleurt",
                "--bleurt-model",
                args.bleurt_model,
                "--bleurt-batch-size",
                str(args.bleurt_batch_size),
                "--bleurt-device",
                args.bleurt_device,
                "--hf-home",
                str(args.hf_home),
            ]
        )
        if not args.bleurt_local_files_only:
            cmd.append("--no-bleurt-local-files-only")

    print("Running scoring:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
