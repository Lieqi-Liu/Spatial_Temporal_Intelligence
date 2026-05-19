#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_JSON = Path("/local1/lieqiliu/nuscenes/fullset/questions_with_answers_all_qwen3vl30b.json")
DEFAULT_OUTPUT_JSON = SCRIPT_DIR / "questions_with_answers_full_nuscenes_miniset_150_per_id.json"
DEFAULT_SUMMARY_JSON = SCRIPT_DIR / "questions_with_answers_full_nuscenes_miniset_150_per_id_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a question-centric full NuScenes miniset from the fully annotated Qwen3-VL-30B file. "
            "The default target is 150 examples for each question id, i.e. 39 * 150 = 5,850 rows."
        )
    )
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--target-per-question-id", type=int, default=150)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--exclude-errors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude rows whose model_response starts with [ERROR].",
    )
    parser.add_argument(
        "--max-per-scene-global",
        type=int,
        default=12,
        help="Maximum selected rows from one scene across the whole miniset. 0 disables.",
    )
    parser.add_argument(
        "--max-per-scene-per-question-id",
        type=int,
        default=1,
        help="Maximum selected rows from one scene within one question id. 0 disables.",
    )
    parser.add_argument("--near-group-window", type=int, default=2)
    parser.add_argument("--scene-coverage-weight", type=float, default=4.0)
    parser.add_argument("--task-scene-diversity-weight", type=float, default=2.0)
    parser.add_argument("--hardness-weight", type=float, default=1.5)
    parser.add_argument("--frq-hardness-weight", type=float, default=1.5)
    parser.add_argument("--exact-group-penalty", type=float, default=1.0)
    parser.add_argument("--near-group-penalty", type=float, default=0.75)
    parser.add_argument(
        "--frq-answer-bucket-fraction",
        type=float,
        default=0.05,
        help="Soft cap fraction for repeated normalized FRQ ground-truth answers within one question id.",
    )
    parser.add_argument("--frq-answer-bucket-min", type=int, default=5)
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

    ground_truth = str(row.get("ground_truth", "")).strip().upper()
    predicted = str(row.get("model_predicted_option", "")).strip().upper()
    if ground_truth and predicted:
        return predicted == ground_truth
    return None


def normalize_text(value: Any) -> str:
    chars = [ch.lower() if ch.isalnum() else " " for ch in str(value or "")]
    return " ".join("".join(chars).split())


def answer_bucket(row: dict[str, Any]) -> str:
    text = normalize_text(row.get("ground_truth"))
    return text[:160] if text else "__empty__"


def token_overlap(left: Any, right: Any) -> float:
    left_tokens = set(normalize_text(left).split())
    right_tokens = set(normalize_text(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def parse_numeric_suffix(value: Any) -> int:
    text = str(value or "")
    if not text:
        return -1
    try:
        return int(text.split("_")[-1])
    except Exception:
        digits = "".join(ch for ch in text if ch.isdigit())
        return int(digits) if digits else -1


def stable_jitter(*parts: Any) -> float:
    text = "|".join(str(part) for part in parts)
    return (sum(ord(ch) for ch in text) % 1000) / 1_000_000.0


def bucket(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "min": ordered[0],
        "max": ordered[-1],
    }


def task_accuracy(rows: list[dict[str, Any]]) -> float | None:
    scored = [row_correctness(row) for row in rows if row_correctness(row) is not None]
    if not scored:
        return None
    return sum(1 for value in scored if value) / len(scored)


def frq_hardness(row: dict[str, Any], task_bleurt_mean: float | None) -> float:
    response = str(row.get("model_response", "")).strip()
    ground_truth = str(row.get("ground_truth", "")).strip()
    overlap_hardness = 1.0 - token_overlap(response, ground_truth) if response and ground_truth else 0.0

    bleurt = row.get("bleurt_model_gt")
    if isinstance(bleurt, (int, float)) and task_bleurt_mean is not None:
        # Lower-than-average BLEURT means the model response is relatively worse, hence harder.
        bleurt_hardness = max(0.0, task_bleurt_mean - float(bleurt))
    else:
        bleurt_hardness = 0.0
    return overlap_hardness + bleurt_hardness


def candidate_score(
    row: dict[str, Any],
    qid_accuracy: float | None,
    qid_bleurt_mean: float | None,
    global_scene_counts: Counter[str],
    task_scene_counts: Counter[str],
    selected_groups_by_scene: dict[str, list[int]],
    selected_exact_groups: set[tuple[str, str]],
    args: argparse.Namespace,
) -> float:
    scene_id = str(row.get("scene_id", ""))
    group_id = str(row.get("group_id", ""))
    group_idx = parse_numeric_suffix(group_id)

    global_scene_count = global_scene_counts[scene_id]
    task_scene_count = task_scene_counts[scene_id]
    scene_coverage_bonus = 2.0 if global_scene_count == 0 else 1.0 / math.sqrt(global_scene_count + 1)
    task_scene_bonus = 1.0 if task_scene_count == 0 else 1.0 / math.sqrt(task_scene_count + 1)

    hardness = 0.0
    if is_mcq(row):
        correct = row.get("_selection_correctness")
        if correct is None:
            correct = row_correctness(row)
        if correct is False:
            hardness = 1.0
        elif correct is True and qid_accuracy is not None:
            hardness = max(0.0, 1.0 - qid_accuracy) * 0.25
    elif is_frq(row):
        hardness = float(row.get("_selection_frq_hardness", 0.0))

    exact_penalty = args.exact_group_penalty if (scene_id, group_id) in selected_exact_groups else 0.0
    near_penalty = 0.0
    for selected_idx in selected_groups_by_scene.get(scene_id, []):
        if group_idx < 0 or selected_idx < 0:
            continue
        distance = abs(group_idx - selected_idx)
        if 0 < distance <= args.near_group_window:
            near_penalty = max(
                near_penalty,
                args.near_group_penalty * (args.near_group_window + 1 - distance) / args.near_group_window,
            )

    saturation_penalty = 0.04 * global_scene_count
    hardness_weight = args.frq_hardness_weight if is_frq(row) else args.hardness_weight
    return (
        args.scene_coverage_weight * scene_coverage_bonus
        + args.task_scene_diversity_weight * task_scene_bonus
        + hardness_weight * hardness
        - exact_penalty
        - near_penalty
        - saturation_penalty
        + stable_jitter(row.get("id"), scene_id, group_id, row.get("object_id"))
    )


def passes_scene_caps(
    row: dict[str, Any],
    global_scene_counts: Counter[str],
    task_scene_counts: Counter[str],
    args: argparse.Namespace,
) -> bool:
    scene_id = str(row.get("scene_id", ""))
    if not scene_id:
        return True
    if args.max_per_scene_global > 0 and global_scene_counts[scene_id] >= args.max_per_scene_global:
        return False
    if (
        args.max_per_scene_per_question_id > 0
        and task_scene_counts[scene_id] >= args.max_per_scene_per_question_id
    ):
        return False
    return True


def select_for_question_id(
    qid: str,
    rows: list[dict[str, Any]],
    target: int,
    global_scene_counts: Counter[str],
    selected_groups_by_scene: dict[str, list[int]],
    selected_exact_groups: set[tuple[str, str]],
    args: argparse.Namespace,
    rng: random.Random,
) -> list[dict[str, Any]]:
    remaining = rows[:]
    rng.shuffle(remaining)
    selected: list[dict[str, Any]] = []
    task_scene_counts: Counter[str] = Counter()
    answer_bucket_counts: Counter[str] = Counter()
    qid_accuracy = task_accuracy(rows)
    bleurt_values = [float(row["bleurt_model_gt"]) for row in rows if isinstance(row.get("bleurt_model_gt"), (int, float))]
    qid_bleurt_mean = statistics.mean(bleurt_values) if bleurt_values else None
    for row in remaining:
        row["_selection_correctness"] = row_correctness(row)
        if is_frq(row):
            row["_selection_frq_hardness"] = frq_hardness(row, qid_bleurt_mean)
    frq_bucket_cap = max(args.frq_answer_bucket_min, math.ceil(target * args.frq_answer_bucket_fraction))

    while remaining and len(selected) < target:
        best_idx = None
        best_score = None
        enforce_answer_caps = [True, False] if any(is_frq(row) for row in remaining) else [False]
        for enforce_answer_cap in enforce_answer_caps:
            for idx, row in enumerate(remaining):
                if not passes_scene_caps(row, global_scene_counts, task_scene_counts, args):
                    continue
                if enforce_answer_cap and is_frq(row) and answer_bucket_counts[answer_bucket(row)] >= frq_bucket_cap:
                    continue
                score = candidate_score(
                    row=row,
                    qid_accuracy=qid_accuracy,
                    qid_bleurt_mean=qid_bleurt_mean,
                    global_scene_counts=global_scene_counts,
                    task_scene_counts=task_scene_counts,
                    selected_groups_by_scene=selected_groups_by_scene,
                    selected_exact_groups=selected_exact_groups,
                    args=args,
                )
                if best_score is None or score > best_score:
                    best_idx = idx
                    best_score = score
            if best_idx is not None:
                break

        if best_idx is None:
            break

        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        scene_id = str(chosen.get("scene_id", ""))
        group_id = str(chosen.get("group_id", ""))
        global_scene_counts[scene_id] += 1
        task_scene_counts[scene_id] += 1
        selected_groups_by_scene[scene_id].append(parse_numeric_suffix(group_id))
        selected_exact_groups.add((scene_id, group_id))
        if is_frq(chosen):
            answer_bucket_counts[answer_bucket(chosen)] += 1

    if len(selected) < target:
        print(f"[warn] {qid}: selected {len(selected)} / {target}; scene caps may be too strict.")
    return selected


def normalize_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("tasks") or payload.get("results")
    if not isinstance(rows, list):
        raise ValueError("Input JSON must contain a top-level tasks or results list.")
    normalized = []
    for row in rows:
        out = dict(row)
        out["id"] = str(out.get("id", ""))
        out["question_format"] = str(out.get("question_format", "")).upper()
        normalized.append(out)
    return normalized


def strip_selection_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_selection_")}


def distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "by_question_id": dict(sorted(Counter(str(row.get("id", "")) for row in rows).items())),
        "by_question_format": dict(sorted(Counter(str(row.get("question_format", "")) for row in rows).items())),
        "by_task": dict(sorted(Counter(str(row.get("task", "")) for row in rows).items())),
        "scene_count": len({str(row.get("scene_id", "")) for row in rows if row.get("scene_id")}),
        "top_scenes": dict(Counter(str(row.get("scene_id", "")) for row in rows if row.get("scene_id")).most_common(25)),
    }


def accuracy_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    mcq_rows = [row for row in rows if is_mcq(row)]
    scored = [row for row in mcq_rows if row_correctness(row) is not None]
    correct = [row for row in scored if row_correctness(row) is True]
    random_baselines = [
        float(row["model_random_baseline"])
        for row in scored
        if isinstance(row.get("model_random_baseline"), (int, float))
    ]
    bleurt_values = [
        float(row["bleurt_model_gt"])
        for row in rows
        if is_frq(row) and isinstance(row.get("bleurt_model_gt"), (int, float))
    ]
    return {
        "mcq_count": len(mcq_rows),
        "mcq_scored_count": len(scored),
        "mcq_correct_count": len(correct),
        "mcq_accuracy": len(correct) / len(scored) if scored else None,
        "mcq_random_guess_baseline_mean": statistics.mean(random_baselines) if random_baselines else None,
        "mcq_accuracy_minus_random_guess": (
            (len(correct) / len(scored)) - statistics.mean(random_baselines)
            if scored and random_baselines
            else None
        ),
        "frq_count": sum(1 for row in rows if is_frq(row)),
        "frq_bleurt": bucket(bleurt_values),
    }


def per_question_summary(full_rows: list[dict[str, Any]], selected_rows: list[dict[str, Any]]) -> dict[str, Any]:
    full_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    selected_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in full_rows:
        full_by_id[str(row.get("id", ""))].append(row)
    for row in selected_rows:
        selected_by_id[str(row.get("id", ""))].append(row)

    result: dict[str, Any] = {}
    for qid in sorted(full_by_id):
        selected = selected_by_id.get(qid, [])
        full = full_by_id[qid]
        result[qid] = {
            "full_count": len(full),
            "selected_count": len(selected),
            "retention_fraction": len(selected) / len(full) if full else 0.0,
            "question_format": str(full[0].get("question_format", "")) if full else "",
            "full_accuracy": accuracy_summary(full),
            "miniset_accuracy": accuracy_summary(selected),
            "selected_scene_count": len({str(row.get("scene_id", "")) for row in selected if row.get("scene_id")}),
            "selected_error_responses": sum(1 for row in selected if is_error_response(row)),
        }
    return result


def main() -> None:
    args = parse_args()
    if args.target_per_question_id <= 0:
        raise ValueError("--target-per-question-id must be > 0")

    rng = random.Random(args.seed)
    payload = read_json(args.input_json)
    all_rows = normalize_rows(payload)
    source_counts = Counter(str(row.get("id", "")) for row in all_rows)

    selectable_rows = [
        row
        for row in all_rows
        if row.get("id") and (not args.exclude_errors or not is_error_response(row))
    ]
    rows_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selectable_rows:
        rows_by_id[str(row.get("id", ""))].append(row)

    question_ids = sorted(source_counts)
    global_scene_counts: Counter[str] = Counter()
    selected_groups_by_scene: dict[str, list[int]] = defaultdict(list)
    selected_exact_groups: set[tuple[str, str]] = set()
    selected_by_id: dict[str, list[dict[str, Any]]] = {}

    for qid in question_ids:
        print(f"Selecting {qid} ({len(rows_by_id.get(qid, []))} candidates)...", flush=True)
        selected_by_id[qid] = select_for_question_id(
            qid=qid,
            rows=rows_by_id.get(qid, []),
            target=min(args.target_per_question_id, len(rows_by_id.get(qid, []))),
            global_scene_counts=global_scene_counts,
            selected_groups_by_scene=selected_groups_by_scene,
            selected_exact_groups=selected_exact_groups,
            args=args,
            rng=rng,
        )

    selected_rows = [
        strip_selection_fields(row)
        for qid in question_ids
        for row in sorted(
            selected_by_id[qid],
            key=lambda r: (
                str(r.get("scene_id", "")),
                parse_numeric_suffix(r.get("group_id")),
                str(r.get("object_id", "")),
                str(r.get("id", "")),
            ),
        )
    ]
    generated_answers = {
        qid: {"count": len(selected_by_id[qid]), "tasks": [strip_selection_fields(row) for row in selected_by_id[qid]]}
        for qid in question_ids
    }

    output_payload = {
        "meta": {
            "source_input_json": str(args.input_json),
            "selection_method": "full_nuscenes_question_centric_150_per_id",
            "target_per_question_id": args.target_per_question_id,
            "target_total_if_all_ids_available": len(question_ids) * args.target_per_question_id,
            "exclude_errors": args.exclude_errors,
            "seed": args.seed,
            "max_per_scene_global": args.max_per_scene_global,
            "max_per_scene_per_question_id": args.max_per_scene_per_question_id,
        },
        "tasks": selected_rows,
        "generated_answers": generated_answers,
    }

    summary = {
        "headline_summary": {
            "full_question_count": len(all_rows),
            "question_id_count": len(question_ids),
            "target_per_question_id": args.target_per_question_id,
            "selected_question_count": len(selected_rows),
            "retention_fraction": len(selected_rows) / len(all_rows) if all_rows else 0.0,
            "selected_scene_count": len({str(row.get("scene_id", "")) for row in selected_rows if row.get("scene_id")}),
            "selected_error_response_count": sum(1 for row in selected_rows if is_error_response(row)),
        },
        "meta": output_payload["meta"],
        "full_set": {
            "distributions": distribution(all_rows),
            "accuracy": accuracy_summary(all_rows),
            "error_response_count": sum(1 for row in all_rows if is_error_response(row)),
        },
        "miniset": {
            "distributions": distribution(selected_rows),
            "accuracy": accuracy_summary(selected_rows),
            "per_question_id": per_question_summary(all_rows, selected_rows),
        },
    }

    write_json(args.output_json, output_payload)
    write_json(args.summary_json, summary)
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "summary_json": str(args.summary_json),
                "selected_question_count": len(selected_rows),
                "question_id_count": len(question_ids),
                "target_per_question_id": args.target_per_question_id,
                "selected_error_response_count": summary["headline_summary"]["selected_error_response_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
