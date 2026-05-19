#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import re
import sys
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
ANNOTATOR_PATH = REPO_ROOT / "lieqiliu" / "run_fake_full_vlm_batch.py"

DEFAULT_INPUT_JSON = SCRIPT_DIR / "questions_with_answers_full_nuscenes_miniset_150_per_id.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "miniset_model_benchmark_outputs"
DEFAULT_FORMATTED_SCENES_DIR = Path("/local1/lieqiliu/nuscenes/fullset/formatted_scenes")
DEFAULT_NUSCENES_ROOT = Path("/local1/lieqiliu/nuscenes/fullset")
DEFAULT_HF_HOME = Path("/local1/lieqiliu/huggingface")


# Edit this list when adding/removing benchmark models.
MODEL_SPECS = [
    {
        "name": "qwen3_vl_8b",
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "backend": "vllm",
        "enabled": True,
    },
    {
        "name": "qwen3_vl_30b_a3b",
        "model_id": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "backend": "vllm",
        "enabled": True,
    },
    {
        "name": "qwen3_5_27b",
        "model_id": "Qwen/Qwen3.5-27B",
        "backend": "vllm",
        "enabled": True,
    },
]


@dataclass
class LoadedModel:
    processor: Any
    llm: Any
    mcq_sampling: Any
    frq_sampling: Any


def load_annotator_module() -> Any:
    spec = importlib.util.spec_from_file_location("full_nuscenes_vlm_annotator", ANNOTATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load annotator module from {ANNOTATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ANNOTATOR = load_annotator_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiple VLMs on the selected full-NuScenes miniset and summarize outcomes."
    )
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--formatted-scenes-dir", type=Path, default=DEFAULT_FORMATTED_SCENES_DIR)
    parser.add_argument("--nuscenes-root", type=Path, default=DEFAULT_NUSCENES_ROOT)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--hf-home", type=Path, default=DEFAULT_HF_HOME)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--mcq-max-tokens", type=int, default=8)
    parser.add_argument("--frq-max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=256)
    parser.add_argument(
        "--only-model",
        action="append",
        default=[],
        help="Run only matching MODEL_SPECS name(s). Can be repeated.",
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use only locally cached HF files by default. Use --no-local-files-only to download.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp.replace(path)


def safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_").lower()


def is_mcq(row: dict[str, Any]) -> bool:
    return str(row.get("question_format", "")).upper() == "MCQ" and isinstance(row.get("choices"), dict)


def is_frq(row: dict[str, Any]) -> bool:
    return str(row.get("question_format", "")).upper() == "FRQ"


def is_error_response(row: dict[str, Any]) -> bool:
    return str(row.get("model_response", "")).strip().startswith("[ERROR]")


def row_correctness(row: dict[str, Any]) -> bool | None:
    value = row.get("model_is_correct")
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    return None


def clear_prior_outputs(task: dict[str, Any]) -> None:
    ANNOTATOR.clear_model_response(task)
    for key in ("bleurt_model_gt", "bleurt_model", "bleurt_scored_at"):
        task.pop(key, None)


def prepare_payload(input_payload: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    payload = ANNOTATOR.sanitize_payload(input_payload)
    tasks = payload.get("tasks", [])
    if not isinstance(tasks, list):
        raise ValueError("Input JSON must contain a top-level tasks list.")

    if args.start_index:
        tasks = tasks[args.start_index :]
    if args.max_tasks > 0:
        tasks = tasks[: args.max_tasks]
    tasks = [dict(task) for task in tasks]
    for task in tasks:
        clear_prior_outputs(task)
    payload["tasks"] = tasks
    payload.setdefault("meta", {})
    return payload


def model_cache_dir_name(model_id: str) -> str:
    return f"models--{model_id.replace('/', '--')}"


def resolve_hf_home(model_id: str, hf_home: Path, local_files_only: bool) -> Path:
    hf_home = hf_home.expanduser()
    if (hf_home / "hub" / model_cache_dir_name(model_id)).exists():
        return hf_home

    default_hf_home = Path.home() / ".cache" / "huggingface"
    if (default_hf_home / "hub" / model_cache_dir_name(model_id)).exists():
        return default_hf_home

    if local_files_only:
        raise FileNotFoundError(
            f"Local cache not found for {model_id}. Checked {hf_home} and {default_hf_home}."
        )
    return hf_home


def load_vllm_model(model_id: str, args: argparse.Namespace) -> LoadedModel:
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    hf_home = resolve_hf_home(model_id, args.hf_home, args.local_files_only)
    processor = AutoProcessor.from_pretrained(
        model_id,
        cache_dir=str(hf_home / "hub"),
        local_files_only=args.local_files_only,
        trust_remote_code=True,
    )
    llm = LLM(
        model=model_id,
        download_dir=str(hf_home / "hub"),
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )
    mcq_sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.mcq_max_tokens,
        repetition_penalty=1.0,
    )
    frq_sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.frq_max_tokens,
        repetition_penalty=1.0,
    )
    return LoadedModel(
        processor=processor,
        llm=llm,
        mcq_sampling=mcq_sampling,
        frq_sampling=frq_sampling,
    )


def load_model(model_spec: dict[str, Any], args: argparse.Namespace) -> LoadedModel:
    backend = str(model_spec.get("backend", ""))
    if backend != "vllm":
        raise ValueError(f"Unsupported backend: {backend}")
    return load_vllm_model(str(model_spec["model_id"]), args)


def iter_batches(indices: list[int], batch_size: int, desc: str):
    iterator = range(0, len(indices), batch_size)
    if tqdm is not None:
        iterator = tqdm(iterator, total=(len(indices) + batch_size - 1) // batch_size, desc=desc)
    for offset in iterator:
        yield indices[offset : offset + batch_size]


def run_indices(
    tasks: list[dict[str, Any]],
    indices: list[int],
    loaded: LoadedModel,
    sampling_params: Any,
    resolver: Any,
    model_id: str,
    output_path: Path,
    payload: dict[str, Any],
    args: argparse.Namespace,
    desc: str,
) -> int:
    completed_since_save = 0
    completed = 0
    for batch_indices in iter_batches(indices, args.batch_size, desc):
        requests: list[dict[str, Any]] = []
        runnable_indices: list[int] = []
        for idx in batch_indices:
            try:
                prompt = ANNOTATOR.build_prompt(loaded.processor, tasks[idx])
                images = resolver.load_images(tasks[idx])
                requests.append({"prompt": prompt, "multi_modal_data": {"image": images}})
                runnable_indices.append(idx)
            except Exception as exc:
                ANNOTATOR.apply_result(tasks[idx], f"[ERROR] {type(exc).__name__}: {exc}", model_id)
                completed += 1
                completed_since_save += 1

        if requests:
            outputs = loaded.llm.generate(requests, sampling_params=sampling_params)
            for idx, output in zip(runnable_indices, outputs):
                text = output.outputs[0].text.strip() if output.outputs else ""
                ANNOTATOR.apply_result(tasks[idx], text, model_id)
                completed += 1
                completed_since_save += 1

        if args.save_every > 0 and completed_since_save >= args.save_every:
            payload["meta"]["partial_summary"] = summarize_model(tasks)
            write_json(output_path, payload)
            completed_since_save = 0
    return completed


def summarize_model(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    by_id: dict[str, Counter[str]] = defaultdict(Counter)
    mcq_rows = [row for row in tasks if is_mcq(row)]
    frq_rows = [row for row in tasks if is_frq(row)]
    scored_mcq = [row for row in mcq_rows if row_correctness(row) is not None]
    correct_mcq = [row for row in scored_mcq if row_correctness(row) is True]
    random_baselines = [
        float(row["model_random_baseline"])
        for row in scored_mcq
        if isinstance(row.get("model_random_baseline"), (int, float))
    ]

    for row in tasks:
        qid = str(row.get("id", ""))
        counter = by_id[qid]
        counter["total"] += 1
        if row.get("model_response"):
            counter["answered"] += 1
        if is_error_response(row):
            counter["errors"] += 1
        correct = row_correctness(row)
        if correct is not None:
            counter["scored"] += 1
            counter["correct"] += int(correct)

    per_question_id = {}
    for qid, counter in sorted(by_id.items()):
        per_question_id[qid] = {
            "total": counter["total"],
            "answered": counter["answered"],
            "error_responses": counter["errors"],
            "scored_accuracy_rows": counter["scored"],
            "correct": counter["correct"],
            "accuracy": counter["correct"] / counter["scored"] if counter["scored"] else None,
        }

    mcq_accuracy = len(correct_mcq) / len(scored_mcq) if scored_mcq else None
    random_baseline = sum(random_baselines) / len(random_baselines) if random_baselines else None
    return {
        "total": len(tasks),
        "by_question_format": dict(sorted(Counter(row.get("question_format") for row in tasks).items())),
        "answered": sum(1 for row in tasks if row.get("model_response")),
        "error_responses": sum(1 for row in tasks if is_error_response(row)),
        "mcq": {
            "total": len(mcq_rows),
            "scored": len(scored_mcq),
            "correct": len(correct_mcq),
            "accuracy": mcq_accuracy,
            "random_guess_baseline": random_baseline,
            "accuracy_minus_random_guess": mcq_accuracy - random_baseline
            if mcq_accuracy is not None and random_baseline is not None
            else None,
        },
        "frq": {
            "total": len(frq_rows),
            "responses": sum(1 for row in frq_rows if row.get("model_response") and not is_error_response(row)),
            "bleurt_count": 0,
            "note": "Run FRQ BLEURT scoring separately after benchmark responses are written.",
        },
        "per_question_id": per_question_id,
    }


def cleanup_loaded_model(loaded: LoadedModel | None) -> None:
    if loaded is None:
        return
    del loaded
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run_model_spec(
    model_spec: dict[str, Any],
    input_payload: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    name = str(model_spec["name"])
    model_id = str(model_spec["model_id"])
    output_path = args.output_dir / f"{safe_slug(name)}_responses.json"
    started_at = datetime.now().isoformat(timespec="seconds")
    payload = prepare_payload(input_payload, args)
    tasks = payload["tasks"]
    payload["meta"]["benchmark_model"] = model_spec
    payload["meta"]["benchmark_started_at"] = started_at

    loaded: LoadedModel | None = None
    try:
        loaded = load_model(model_spec, args)
    except Exception as exc:
        summary = {
            "name": name,
            "model_id": model_id,
            "backend": model_spec.get("backend"),
            "status": "skipped_load_failed",
            "started_at": started_at,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "load_error": f"{type(exc).__name__}: {exc}",
            "load_traceback": traceback.format_exc(limit=8),
            "output_json": None,
        }
        return summary

    try:
        resolver = ANNOTATOR.VisualResolver(
            formatted_scenes_dir=args.formatted_scenes_dir,
            nuscenes_root=args.nuscenes_root,
            version=args.version,
            image_cache_size=64,
            render_missing_boxes=True,
        )
        try:
            mcq_indices = [idx for idx, task in enumerate(tasks) if is_mcq(task)]
            frq_indices = [idx for idx, task in enumerate(tasks) if is_frq(task)]
            run_indices(
                tasks=tasks,
                indices=mcq_indices,
                loaded=loaded,
                sampling_params=loaded.mcq_sampling,
                resolver=resolver,
                model_id=model_id,
                output_path=output_path,
                payload=payload,
                args=args,
                desc=f"{name} MCQ",
            )
            run_indices(
                tasks=tasks,
                indices=frq_indices,
                loaded=loaded,
                sampling_params=loaded.frq_sampling,
                resolver=resolver,
                model_id=model_id,
                output_path=output_path,
                payload=payload,
                args=args,
                desc=f"{name} FRQ",
            )
        finally:
            resolver.close()

        model_summary = summarize_model(tasks)
        payload["meta"]["benchmark_finished_at"] = datetime.now().isoformat(timespec="seconds")
        payload["meta"]["benchmark_summary"] = model_summary
        write_json(output_path, payload)
        return {
            "name": name,
            "model_id": model_id,
            "backend": model_spec.get("backend"),
            "status": "completed",
            "started_at": started_at,
            "finished_at": payload["meta"]["benchmark_finished_at"],
            "output_json": str(output_path),
            "summary": model_summary,
        }
    except Exception as exc:
        write_json(output_path, payload)
        return {
            "name": name,
            "model_id": model_id,
            "backend": model_spec.get("backend"),
            "status": "failed_during_inference",
            "started_at": started_at,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "inference_error": f"{type(exc).__name__}: {exc}",
            "inference_traceback": traceback.format_exc(limit=8),
            "output_json": str(output_path),
            "partial_summary": summarize_model(tasks),
        }
    finally:
        cleanup_loaded_model(loaded)


def selected_model_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    requested = set(args.only_model)
    specs = []
    for spec in MODEL_SPECS:
        if not spec.get("enabled", True):
            continue
        if requested and spec["name"] not in requested:
            continue
        specs.append(spec)
    return specs


def merge_existing_results(
    summary_path: Path,
    new_results: list[dict[str, Any]],
    enabled_model_names: set[str],
) -> list[dict[str, Any]]:
    merged_by_name: dict[str, dict[str, Any]] = {}
    if summary_path.exists():
        try:
            existing = read_json(summary_path)
            for result in existing.get("results", []):
                name = str(result.get("name", ""))
                if name in enabled_model_names:
                    merged_by_name[name] = result
        except Exception as exc:
            print(f"[warn] Could not merge existing benchmark summary: {type(exc).__name__}: {exc}")

    for result in new_results:
        name = str(result.get("name", ""))
        if name in enabled_model_names:
            merged_by_name[name] = result

    return [merged_by_name[name] for name in sorted(merged_by_name)]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.hf_home = args.hf_home.expanduser()
    os.environ["HF_HOME"] = str(args.hf_home)
    os.environ.setdefault("HF_HUB_CACHE", str(args.hf_home / "hub"))
    if args.local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    input_payload = read_json(args.input_json)
    specs = selected_model_specs(args)
    results = []
    for spec in specs:
        print(f"\n=== Benchmarking {spec['name']} ({spec['model_id']}) ===", flush=True)
        result = run_model_spec(spec, input_payload, args)
        results.append(result)
        print(json.dumps({k: v for k, v in result.items() if k not in {"load_traceback", "inference_traceback"}}, indent=2), flush=True)

    summary_path = args.output_dir / "benchmark_summary.json"
    enabled_specs = [spec for spec in MODEL_SPECS if spec.get("enabled", True)]
    enabled_model_names = {str(spec["name"]) for spec in enabled_specs}
    merged_results = merge_existing_results(summary_path, results, enabled_model_names)
    benchmark_summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_json": str(args.input_json),
        "output_dir": str(args.output_dir),
        "model_specs": enabled_specs,
        "last_run_model_specs": specs,
        "results": merged_results,
    }
    write_json(summary_path, benchmark_summary)
    print(f"\nWrote benchmark summary: {summary_path}")


if __name__ == "__main__":
    main()
