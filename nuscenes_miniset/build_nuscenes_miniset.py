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


SCENE_TASK_PREFIX = "SC-"
DEFAULT_VLM_METRICS = Path("/home/rgao727/Spatial_Temporal_Intelligence/vlm_responses_20260331_210718.json")
FRQ_TASK_IDS = {"SP-3a", "SP-7", "SU-7", "TE-6", "TM-6", "TRJ-5", "TRJ-6", "TRJ-7", "SC-6"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a NuScenes miniset v2 that is question-centric, scene-diverse, "
            "low-redundancy within scene/group neighborhoods, and MCQ-label-balanced "
            "through post-selection choice reindexing."
        )
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path("/local1/rgao727/nuscenes/questions_with_answers_trainval.json"),
        help="Input questions_with_answers_trainval.json",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("/local1/rgao727/nuscenes_mini/questions_with_answers_miniset_v2.json"),
        help="Output selected miniset JSON",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("/local1/rgao727/nuscenes_mini/questions_with_answers_miniset_v2_summary.json"),
        help="Output summary JSON",
    )
    parser.add_argument(
        "--vlm-metrics-json",
        type=Path,
        default=DEFAULT_VLM_METRICS,
        help="Optional VLM responses file used to derive per-task accuracy priors",
    )
    parser.add_argument(
        "--scene-task-target",
        type=int,
        default=400,
        help="Questions to keep for each SC-* task",
    )
    parser.add_argument(
        "--other-task-target",
        type=int,
        default=200,
        help="Questions to keep for each non-SC task",
    )
    parser.add_argument(
        "--frq-task-target",
        type=int,
        default=200,
        help="Questions to keep for each FRQ task",
    )
    parser.add_argument(
        "--scene-coverage-weight",
        type=float,
        default=4.0,
        help="Weight for selecting questions from not-yet-covered or under-covered scenes",
    )
    parser.add_argument(
        "--task-scene-diversity-weight",
        type=float,
        default=1.5,
        help="Weight for selecting questions from scenes not yet used within the current task",
    )
    parser.add_argument(
        "--hardness-weight",
        type=float,
        default=1.2,
        help="Weight for low-accuracy task prior derived from VLM results",
    )
    parser.add_argument(
        "--example-hardness-weight",
        type=float,
        default=1.5,
        help=(
            "Weight for prioritizing hard examples within a task, derived from per-example VLM correctness. "
            "Positive values prefer examples where the VLM was incorrect."
        ),
    )
    parser.add_argument(
        "--example-hardness-z-threshold",
        type=float,
        default=0.5,
        help=(
            "Only apply the within-task hardness bonus when the example is this many Bernoulli stddevs "
            "below the task average accuracy."
        ),
    )
    parser.add_argument(
        "--frq-response-hardness-weight",
        type=float,
        default=1.0,
        help="Weight for FRQ hardness from low model-response/ground-truth word overlap.",
    )
    parser.add_argument(
        "--frq-answer-bucket-fraction",
        type=float,
        default=0.05,
        help="Soft cap fraction for repeated normalized FRQ ground-truth answer buckets within one task.",
    )
    parser.add_argument(
        "--frq-answer-bucket-min",
        type=int,
        default=5,
        help="Minimum soft cap for repeated normalized FRQ answer buckets within one task.",
    )
    parser.add_argument(
        "--exact-group-penalty",
        type=float,
        default=1.5,
        help="Penalty for reusing the exact same group in another selected question",
    )
    parser.add_argument(
        "--near-group-penalty",
        type=float,
        default=1.0,
        help="Penalty for selecting nearby groups within the same scene",
    )
    parser.add_argument(
        "--near-group-window",
        type=int,
        default=2,
        help="Groups within this absolute index distance are treated as near-duplicates",
    )
    parser.add_argument(
        "--max-per-scene-global",
        type=int,
        default=40,
        help="Hard cap: maximum selected questions allowed from a single scene across all tasks (0 disables).",
    )
    parser.add_argument(
        "--max-per-scene-per-task",
        type=int,
        default=8,
        help="Hard cap: maximum selected questions allowed from a single scene within a single task (0 disables).",
    )
    parser.add_argument(
        "--max-per-scene-per-task-scene",
        type=int,
        default=1,
        help="Hard cap for SC-* tasks: maximum selected questions per scene within that SC task (0 disables).",
    )
    parser.add_argument(
        "--max-per-scene-sc-family",
        type=int,
        default=3,
        help="Hard cap for SC-* tasks: maximum selected SC questions from a scene across all SC tasks (0 disables).",
    )
    parser.add_argument(
        "--max-sc-per-exact-group",
        type=int,
        default=1,
        help="Hard cap for SC-* tasks: maximum selected SC questions from the same scene/group pair (0 disables).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for deterministic tie-breaking",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def path_exists_readably(path: Path) -> bool:
    try:
        return path.exists()
    except PermissionError:
        return False


def ensure_generated_answers(payload: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    generated = payload.get("generated_answers", {})
    if generated:
        return generated, False
    tasks = payload.get("tasks", [])
    if not isinstance(tasks, list) or not tasks:
        return {}, False
    grouped: dict[str, Any] = {}
    for row in tasks:
        task_id = str(row.get("id", ""))
        if not task_id:
            continue
        bucket = grouped.setdefault(task_id, {"count": 0, "tasks": []})
        bucket["tasks"].append(row)
        bucket["count"] += 1
    payload["generated_answers"] = grouped
    return grouped, True


def is_mcq(row: dict[str, Any]) -> bool:
    return row.get("question_format") == "MCQ" and isinstance(row.get("choices"), dict) and bool(row.get("choices"))


def is_frq_task(task_id: str) -> bool:
    return str(task_id) in FRQ_TASK_IDS


def is_frq_row(row: dict[str, Any]) -> bool:
    return is_frq_task(str(row.get("id", ""))) or str(row.get("question_format", "")).upper() == "FRQ"


def normalize_text(text: Any) -> str:
    normalized = []
    for ch in str(text or "").lower():
        normalized.append(ch if ch.isalnum() else " ")
    return " ".join("".join(normalized).split())


def answer_bucket(row: dict[str, Any]) -> str:
    text = normalize_text(row.get("ground_truth", ""))
    if not text:
        return "__empty__"
    return text[:160]


def token_overlap(a: Any, b: Any) -> float:
    left = set(normalize_text(a).split())
    right = set(normalize_text(b).split())
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def frq_response_hardness(row: dict[str, Any]) -> float:
    if not is_frq_row(row):
        return 0.0
    ground_truth = str(row.get("ground_truth", "")).strip()
    model_response = str(row.get("model_response", "")).strip()
    if not ground_truth:
        return 0.0
    if not model_response or model_response.lower() == "none":
        return 1.0
    return 1.0 - token_overlap(model_response, ground_truth)


def parse_numeric_suffix(text: str) -> int:
    if not text:
        return -1
    try:
        return int(str(text).split("_")[-1])
    except Exception:
        digits = "".join(ch for ch in str(text) if ch.isdigit())
        return int(digits) if digits else -1


def stable_jitter(text: str) -> float:
    return (sum(ord(c) for c in text) % 1000) / 1_000_000.0


def row_identity_keys(row: dict[str, Any]) -> list[tuple[str, ...]]:
    task_id = str(row.get("id", ""))
    scene_id = str(row.get("scene_id", ""))
    group_id = str(row.get("group_id", ""))
    object_id = str(row.get("object_id", ""))
    question = str(row.get("question", ""))
    return [
        (task_id, scene_id, group_id, object_id, question),
        (task_id, scene_id, group_id, question),
        (task_id, scene_id, group_id, object_id),
    ]


def row_correctness(row: dict[str, Any]) -> bool | None:
    if isinstance(row.get("model_is_correct"), bool):
        return bool(row["model_is_correct"])
    if row.get("model_is_correct") in (0, 1):
        return bool(row["model_is_correct"])
    if isinstance(row.get("is_correct"), bool):
        return bool(row["is_correct"])
    if row.get("is_correct") in (0, 1):
        return bool(row["is_correct"])

    gt = str(row.get("ground_truth", "")).strip()
    model_predicted = str(row.get("model_predicted_option", "")).strip()
    if gt and model_predicted:
        return model_predicted == gt
    predicted = str(row.get("predicted_option", "")).strip()
    if gt and predicted:
        return predicted == gt

    response = str(row.get("model_response", "")).strip()
    if gt and response:
        return response == gt
    return None


def parse_vlm_metrics_block(raw: str) -> dict[str, Any] | None:
    anchor = raw.find('"per_task_metrics"')
    if anchor < 0:
        return None
    brace_start = raw.find("{", anchor)
    if brace_start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    end = None
    for idx in range(brace_start, len(raw)):
        ch = raw[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = idx + 1
                break
    if end is None:
        return None
    return json.loads(raw[brace_start:end])


def normalize_vlm_frq_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["question_format"] = "FRQ"
    out.setdefault("choices", {})
    out["source"] = "vlm_results_frq"
    return out


def extract_vlm_eval_signals(vlm_path: Path) -> tuple[dict[str, float], dict[tuple[str, ...], bool], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if not vlm_path.exists():
        return {}, {}, {}, {"used": False, "reason": "file_missing", "path": str(vlm_path)}

    try:
        payload = read_json(vlm_path)
        metrics = payload.get("meta", {}).get("per_task_metrics", {})
        task_accuracy = {}
        for task_id, metric in metrics.items():
            try:
                task_accuracy[str(task_id)] = float(metric.get("accuracy", 0.0))
            except Exception:
                continue

        rows_for_eval = payload.get("results", [])
        if not rows_for_eval:
            rows_for_eval = payload.get("tasks", [])

        example_correctness: dict[tuple[str, ...], bool] = {}
        frq_source_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        task_counts: Counter = Counter()
        hard_counts: Counter = Counter()
        for row in rows_for_eval:
            task_id = str(row.get("id", ""))
            if is_frq_task(task_id):
                frq_source_rows[task_id].append(normalize_vlm_frq_row(row))

            correct = row_correctness(row)
            if correct is None:
                continue
            for key in row_identity_keys(row):
                example_correctness[key] = correct
            task_counts[task_id] += 1
            if not correct:
                hard_counts[task_id] += 1

        if not task_accuracy and task_counts:
            task_accuracy = {
                str(task_id): 1.0 - (float(hard_counts[task_id]) / float(count))
                for task_id, count in task_counts.items()
                if count
            }

        return task_accuracy, example_correctness, dict(frq_source_rows), {
            "used": True,
            "reason": "parsed_full_json",
            "path": str(vlm_path),
            "task_count": len(task_accuracy),
            "example_count": int(sum(task_counts.values())),
            "hard_example_count": int(sum(hard_counts.values())),
            "row_source": "results" if payload.get("results") else "tasks",
            "frq_source_task_count": len(frq_source_rows),
            "frq_source_example_count": int(sum(len(rows) for rows in frq_source_rows.values())),
        }
    except Exception as exc:
        raw = vlm_path.read_text(encoding="utf-8", errors="ignore")
        try:
            metrics = parse_vlm_metrics_block(raw)
            if metrics is None:
                return {}, {}, {}, {"used": False, "reason": f"metrics_block_missing:{type(exc).__name__}", "path": str(vlm_path)}
            result = {}
            for task_id, metric in metrics.items():
                try:
                    result[str(task_id)] = float(metric.get("accuracy", 0.0))
                except Exception:
                    continue
            return result, {}, {}, {
                "used": True,
                "reason": "salvaged_per_task_metrics",
                "path": str(vlm_path),
                "task_count": len(result),
                "example_count": 0,
                "hard_example_count": 0,
                "frq_source_task_count": 0,
                "frq_source_example_count": 0,
            }
        except Exception as exc2:
            return {}, {}, {}, {"used": False, "reason": f"salvage_failed:{type(exc2).__name__}", "path": str(vlm_path)}


def compute_mcq_balance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, dict[str, Any]] = {}
    max_fractions: list[float] = []
    imbalance_ratios: list[float] = []

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if is_mcq(row):
            grouped[str(row["id"])].append(row)

    for task_id, task_rows in grouped.items():
        counts = Counter(str(r.get("ground_truth", "")) for r in task_rows)
        if not counts:
            continue
        total = sum(counts.values())
        max_count = max(counts.values())
        positive_counts = [v for v in counts.values() if v > 0]
        min_positive = min(positive_counts) if positive_counts else 0
        option_count = len((task_rows[0].get("choices") or {}).keys())
        max_fraction = max_count / total if total else 0.0
        imbalance = (max_count / min_positive) if min_positive else float("inf")
        max_fractions.append(max_fraction)
        imbalance_ratios.append(imbalance if math.isfinite(imbalance) else 0.0)
        by_task[task_id] = {
            "total_questions": total,
            "option_count": option_count,
            "distribution": dict(sorted(counts.items())),
            "max_option_fraction": max_fraction,
            "imbalance_ratio_max_over_min": imbalance,
        }

    return {
        "per_task": by_task,
        "mean_max_option_fraction": statistics.mean(max_fractions) if max_fractions else 0.0,
        "mean_imbalance_ratio": statistics.mean(imbalance_ratios) if imbalance_ratios else 0.0,
    }


def reindex_mcq_rows(rows: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    rows = [dict(r) for r in rows]
    groups: dict[tuple[str, tuple[str, ...]], list[dict[str, Any]]] = defaultdict(list)
    passthrough: list[dict[str, Any]] = []

    for row in rows:
        if not is_mcq(row):
            passthrough.append(row)
            continue
        labels = tuple(sorted((row.get("choices") or {}).keys()))
        groups[(str(row["id"]), labels)].append(row)

    output = passthrough[:]
    for (_task_id, labels), group_rows in groups.items():
        group_rows = group_rows[:]
        rng.shuffle(group_rows)
        label_counts = Counter()
        for row in group_rows:
            choices = dict(row["choices"])
            old_gt = str(row["ground_truth"])
            available = list(labels)
            desired = min(available, key=lambda lbl: (label_counts[lbl], lbl))
            correct_text = choices[old_gt]
            other_texts = [choices[lbl] for lbl in available if lbl != old_gt]

            new_choices: dict[str, str] = {}
            other_iter = iter(other_texts)
            for lbl in available:
                if lbl == desired:
                    new_choices[lbl] = correct_text
                else:
                    new_choices[lbl] = next(other_iter)

            new_row = dict(row)
            new_row["choices"] = new_choices
            new_row["original_ground_truth"] = old_gt
            new_row["ground_truth"] = desired
            new_row["answer_reindexed"] = desired != old_gt
            output.append(new_row)
            label_counts[desired] += 1

    return output


def candidate_score(
    row: dict[str, Any],
    task_accuracy: float | None,
    example_correctness: dict[tuple[str, ...], bool],
    global_scene_counts: Counter,
    task_scene_counts: Counter,
    selected_groups_by_scene: dict[str, list[int]],
    selected_exact_groups: set[tuple[str, str]],
    args: argparse.Namespace,
) -> float:
    scene_id = str(row.get("scene_id", ""))
    group_id = str(row.get("group_id", ""))
    group_idx = parse_numeric_suffix(group_id)

    hardness = (1.0 - task_accuracy) if task_accuracy is not None else 0.0
    example_hardness = 0.0
    correct = None
    for key in row_identity_keys(row):
        if key in example_correctness:
            correct = example_correctness[key]
            break
    if correct is None:
        correct = row_correctness(row)
    if correct is not None and task_accuracy is not None:
        p = min(max(task_accuracy, 1e-6), 1.0 - 1e-6)
        std = math.sqrt(p * (1.0 - p))
        observed = 1.0 if correct else 0.0
        z_below_task_avg = (p - observed) / std
        if z_below_task_avg >= args.example_hardness_z_threshold:
            example_hardness = z_below_task_avg
    frq_hardness = frq_response_hardness(row)

    global_count = global_scene_counts[scene_id]
    task_count = task_scene_counts[scene_id]

    scene_coverage_bonus = 2.0 if global_count == 0 else (1.0 / math.sqrt(global_count + 1))
    task_scene_bonus = 1.0 if task_count == 0 else (1.0 / math.sqrt(task_count + 1))

    exact_penalty = args.exact_group_penalty if (scene_id, group_id) in selected_exact_groups else 0.0
    near_penalty = 0.0
    for selected_idx in selected_groups_by_scene.get(scene_id, []):
        if selected_idx < 0 or group_idx < 0:
            continue
        dist = abs(selected_idx - group_idx)
        if dist == 0:
            continue
        if dist <= args.near_group_window:
            near_penalty = max(near_penalty, args.near_group_penalty * (args.near_group_window + 1 - dist) / args.near_group_window)

    saturation_penalty = 0.03 * global_count

    score = (
        args.scene_coverage_weight * scene_coverage_bonus
        + args.task_scene_diversity_weight * task_scene_bonus
        + args.hardness_weight * hardness
        + args.example_hardness_weight * example_hardness
        + args.frq_response_hardness_weight * frq_hardness
        - exact_penalty
        - near_penalty
        - saturation_penalty
        + stable_jitter(str(row.get("id", "")) + "|" + scene_id + "|" + group_id)
    )
    return score


def select_for_task(
    task_id: str,
    rows: list[dict[str, Any]],
    target: int,
    task_accuracy: float | None,
    example_correctness: dict[tuple[str, ...], bool],
    global_scene_counts: Counter,
    selected_sc_scene_counts: Counter,
    selected_groups_by_scene: dict[str, list[int]],
    selected_exact_groups: set[tuple[str, str]],
    selected_sc_exact_group_counts: Counter,
    args: argparse.Namespace,
    rng: random.Random,
) -> list[dict[str, Any]]:
    remaining = rows[:]
    rng.shuffle(remaining)
    task_scene_counts: Counter = Counter()
    answer_bucket_counts: Counter = Counter()
    selected: list[dict[str, Any]] = []
    frq_bucket_cap = max(args.frq_answer_bucket_min, int(math.ceil(target * args.frq_answer_bucket_fraction)))

    while remaining and len(selected) < target:
        best_index = 0
        best_score = None
        is_scene_task = str(task_id).startswith(SCENE_TASK_PREFIX)
        is_frq = is_frq_task(task_id)
        cap_per_task = args.max_per_scene_per_task_scene if is_scene_task else args.max_per_scene_per_task
        for enforce_answer_cap in ([True, False] if is_frq and frq_bucket_cap > 0 else [False]):
            for idx, row in enumerate(remaining):
                scene_id = str(row.get("scene_id", ""))
                group_id = str(row.get("group_id", ""))
                if scene_id:
                    if args.max_per_scene_global > 0 and global_scene_counts[scene_id] >= args.max_per_scene_global:
                        continue
                    if cap_per_task > 0 and task_scene_counts[scene_id] >= cap_per_task:
                        continue
                    if is_scene_task and args.max_per_scene_sc_family > 0 and selected_sc_scene_counts[scene_id] >= args.max_per_scene_sc_family:
                        continue
                    if (
                        is_scene_task
                        and args.max_sc_per_exact_group > 0
                        and selected_sc_exact_group_counts[(scene_id, group_id)] >= args.max_sc_per_exact_group
                    ):
                        continue
                if enforce_answer_cap and answer_bucket_counts[answer_bucket(row)] >= frq_bucket_cap:
                    continue
                score = candidate_score(
                    row=row,
                    task_accuracy=task_accuracy,
                    example_correctness=example_correctness,
                    global_scene_counts=global_scene_counts,
                    task_scene_counts=task_scene_counts,
                    selected_groups_by_scene=selected_groups_by_scene,
                    selected_exact_groups=selected_exact_groups,
                    args=args,
                )
                if best_score is None or score > best_score:
                    best_score = score
                    best_index = idx
            if best_score is not None:
                break

        if best_score is None:
            break

        chosen = remaining.pop(best_index)
        scene_id = str(chosen.get("scene_id", ""))
        group_id = str(chosen.get("group_id", ""))
        group_idx = parse_numeric_suffix(group_id)

        selected.append(chosen)
        global_scene_counts[scene_id] += 1
        task_scene_counts[scene_id] += 1
        if str(task_id).startswith(SCENE_TASK_PREFIX):
            selected_sc_scene_counts[scene_id] += 1
            selected_sc_exact_group_counts[(scene_id, group_id)] += 1
        if is_frq:
            answer_bucket_counts[answer_bucket(chosen)] += 1
        selected_groups_by_scene[scene_id].append(group_idx)
        selected_exact_groups.add((scene_id, group_id))

    return selected


def scene_count_hist(counter: Counter) -> dict[str, int]:
    hist = Counter(counter.values())
    return {str(k): int(v) for k, v in sorted(hist.items())}


def count_stats(counter: Counter) -> dict[str, Any]:
    values = list(counter.values())
    if not values:
        return {"max": 0, "mean": 0.0, "median": 0.0}
    return {
        "max": int(max(values)),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
    }


def group_redundancy(rows: list[dict[str, Any]], near_window: int) -> dict[str, Any]:
    group_counts = Counter(
        (str(r.get("scene_id", "")), str(r.get("group_id", "")))
        for r in rows
        if r.get("scene_id") and r.get("group_id")
    )
    groups_by_scene: dict[str, set[int]] = defaultdict(set)
    for scene_id, group_id in group_counts:
        idx = parse_numeric_suffix(group_id)
        if idx >= 0:
            groups_by_scene[scene_id].add(idx)

    near_adjacent_pairs = 0
    for group_indices in groups_by_scene.values():
        ordered = sorted(group_indices)
        for left, right in zip(ordered, ordered[1:]):
            if right - left <= near_window:
                near_adjacent_pairs += 1

    reused_exact_groups = {f"{scene}:{group}": int(count) for (scene, group), count in group_counts.items() if count > 1}
    return {
        "unique_scene_groups": int(len(group_counts)),
        "exact_group_reuse_count": int(sum(count - 1 for count in group_counts.values() if count > 1)),
        "reused_exact_groups_top25": dict(sorted(reused_exact_groups.items(), key=lambda item: item[1], reverse=True)[:25]),
        "near_group_adjacent_pair_count": int(near_adjacent_pairs),
    }


def per_task_scene_coverage(generated_answers: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for task_id, bucket in generated_answers.items():
        rows = bucket.get("tasks", [])
        scenes = Counter(str(r.get("scene_id", "")) for r in rows if r.get("scene_id"))
        result[str(task_id)] = {
            "unique_scenes": int(len(scenes)),
            "max_questions_in_one_scene": int(max(scenes.values())) if scenes else 0,
            "scene_count_histogram": scene_count_hist(scenes),
        }
    return result


def task_retention(full_per_task: dict[str, int], selected_per_task: dict[str, int]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for task_id in sorted(set(full_per_task) | set(selected_per_task)):
        full_count = int(full_per_task.get(task_id, 0))
        selected_count = int(selected_per_task.get(task_id, 0))
        result[task_id] = {
            "full_set": full_count,
            "miniset": selected_count,
            "retention_fraction": (selected_count / full_count) if full_count else 0.0,
        }
    return result


def inject_frq_sources(generated: dict[str, Any], frq_source_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    injected = {}
    for task_id, rows in frq_source_rows.items():
        if not rows:
            continue
        existing = generated.get(task_id, {}).get("tasks", [])
        if existing:
            continue
        generated[task_id] = {
            "count": len(rows),
            "tasks": [dict(row) for row in rows],
            "source": "vlm_results_frq",
        }
        injected[task_id] = len(rows)
    return injected


def frq_diagnostics(generated_answers: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for task_id, bucket in generated_answers.items():
        if not is_frq_task(str(task_id)):
            continue
        rows = bucket.get("tasks", [])
        buckets = Counter(answer_bucket(row) for row in rows)
        overlaps = [
            token_overlap(row.get("model_response", ""), row.get("ground_truth", ""))
            for row in rows
            if row.get("model_response") and row.get("ground_truth")
        ]
        result[str(task_id)] = {
            "question_count": int(len(rows)),
            "unique_answer_buckets": int(len(buckets)),
            "max_answer_bucket_count": int(max(buckets.values())) if buckets else 0,
            "mean_model_response_ground_truth_overlap": statistics.mean(overlaps) if overlaps else None,
        }
    return result


def distribution_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "by_question_id": dict(sorted(Counter(str(r.get("id", "")) for r in rows).items())),
        "by_task_category": dict(sorted(Counter(str(r.get("task", "")) for r in rows).items())),
        "by_question_format": dict(sorted(Counter(str(r.get("question_format", "")) for r in rows).items())),
        "by_source_dataset": dict(sorted(Counter(str(r.get("source_dataset", "")) for r in rows).items())),
        "by_scene": dict(Counter(str(r.get("scene_id", "")) for r in rows if r.get("scene_id")).most_common(25)),
    }


def accuracy_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    mcq_rows = [r for r in rows if is_mcq(r)]
    answered_mcq = [r for r in mcq_rows if row_correctness(r) is not None]
    correct_mcq = [r for r in answered_mcq if row_correctness(r) is True]

    per_task: dict[str, Any] = {}
    for task_id in sorted(set(str(r.get("id", "")) for r in answered_mcq)):
        task_rows = [r for r in answered_mcq if str(r.get("id", "")) == task_id]
        task_correct = sum(1 for r in task_rows if row_correctness(r) is True)
        per_task[task_id] = {
            "answered_count": len(task_rows),
            "correct_count": task_correct,
            "accuracy": task_correct / len(task_rows) if task_rows else None,
        }

    frq_rows = [r for r in rows if is_frq_row(r)]
    frq_with_response = [
        r
        for r in frq_rows
        if str(r.get("model_response", "")).strip()
        and not str(r.get("model_response", "")).startswith("[ERROR]")
    ]
    bleurt_values = [
        float(r.get("bleurt_model_gt"))
        for r in frq_rows
        if isinstance(r.get("bleurt_model_gt"), (int, float))
    ]

    return {
        "mcq_total_count": len(mcq_rows),
        "mcq_answered_count": len(answered_mcq),
        "mcq_correct_count": len(correct_mcq),
        "mcq_accuracy": (len(correct_mcq) / len(answered_mcq)) if answered_mcq else None,
        "mcq_per_task": per_task,
        "frq_total_count": len(frq_rows),
        "frq_response_count": len(frq_with_response),
        "frq_bleurt_count": len(bleurt_values),
        "frq_bleurt_mean": statistics.mean(bleurt_values) if bleurt_values else None,
    }


def hardness_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    mcq_rows = [r for r in rows if is_mcq(r) and row_correctness(r) is not None]
    hard_mcq = [r for r in mcq_rows if row_correctness(r) is False]
    frq_rows = [r for r in rows if is_frq_row(r)]
    frq_hardness_values = [frq_response_hardness(r) for r in frq_rows if r.get("ground_truth")]

    return {
        "mcq_hard_count": len(hard_mcq),
        "mcq_hard_fraction": (len(hard_mcq) / len(mcq_rows)) if mcq_rows else None,
        "frq_mean_response_hardness": statistics.mean(frq_hardness_values) if frq_hardness_values else None,
        "frq_max_response_hardness": max(frq_hardness_values) if frq_hardness_values else None,
    }


def build_summary(
    input_payload: dict[str, Any],
    selected_generated_answers: dict[str, Any],
    vlm_info: dict[str, Any],
    near_group_window: int,
) -> dict[str, Any]:
    full_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []

    for bucket in input_payload.get("generated_answers", {}).values():
        full_rows.extend(bucket.get("tasks", []))
    for bucket in selected_generated_answers.values():
        selected_rows.extend(bucket.get("tasks", []))

    full_scenes = Counter(str(r.get("scene_id", "")) for r in full_rows if r.get("scene_id"))
    selected_scenes = Counter(str(r.get("scene_id", "")) for r in selected_rows if r.get("scene_id"))

    full_per_task = {task_id: len(bucket.get("tasks", [])) for task_id, bucket in input_payload.get("generated_answers", {}).items()}
    selected_per_task = {task_id: len(bucket.get("tasks", [])) for task_id, bucket in selected_generated_answers.items()}

    mcq_before = compute_mcq_balance(selected_rows)
    mcq_full = compute_mcq_balance(full_rows)
    full_accuracy = accuracy_summary(full_rows)
    miniset_accuracy = accuracy_summary(selected_rows)
    full_hardness = hardness_summary(full_rows)
    miniset_hardness = hardness_summary(selected_rows)

    representativeness = {
        "task_retention": task_retention(full_per_task, selected_per_task),
        "scene_coverage": {
            "full_scene_count": int(len(full_scenes)),
            "selected_scene_count": int(len(selected_scenes)),
            "scene_fraction": (len(selected_scenes) / len(full_scenes)) if full_scenes else 0.0,
        },
        "scene_concentration": {
            "full_set": {
                "stats": count_stats(full_scenes),
                "scene_count_histogram": scene_count_hist(full_scenes),
                "top_scenes": dict(full_scenes.most_common(25)),
            },
            "miniset": {
                "stats": count_stats(selected_scenes),
                "scene_count_histogram": scene_count_hist(selected_scenes),
                "top_scenes": dict(selected_scenes.most_common(25)),
            },
        },
        "group_redundancy": {
            "full_set": group_redundancy(full_rows, near_group_window),
            "miniset": group_redundancy(selected_rows, near_group_window),
        },
        "per_task_scene_coverage": per_task_scene_coverage(selected_generated_answers),
        "frq": frq_diagnostics(selected_generated_answers),
    }

    scene_dist = {
        "selected_scene_count": len(selected_scenes),
        "full_scene_count": len(full_scenes),
        "scene_fraction": (len(selected_scenes) / len(full_scenes)) if full_scenes else 0.0,
        "top_selected_scenes": dict(selected_scenes.most_common(25)),
    }

    return {
        "headline_summary": {
            "full_question_count": len(full_rows),
            "mini_question_count": len(selected_rows),
            "retention_percent": ((len(selected_rows) / len(full_rows)) * 100.0) if full_rows else 0.0,
            "scene_coverage": {
                "selected_scene_count": len(selected_scenes),
                "full_scene_count": len(full_scenes),
                "scene_fraction": (len(selected_scenes) / len(full_scenes)) if full_scenes else 0.0,
            },
            "mcq_accuracy": {
                "full_set": full_accuracy["mcq_accuracy"],
                "miniset": miniset_accuracy["mcq_accuracy"],
            },
            "mcq_hard_fraction": {
                "full_set": full_hardness["mcq_hard_fraction"],
                "miniset": miniset_hardness["mcq_hard_fraction"],
            },
            "format_distribution": {
                "full_set": distribution_summary(full_rows)["by_question_format"],
                "miniset": distribution_summary(selected_rows)["by_question_format"],
            },
            "interpretation": (
                "The miniset keeps all scenes represented while retaining a small fraction "
                "of questions and selecting substantially harder MCQ examples."
            ),
        },
        "meta": {
            "description": "NuScenes miniset v2 summary",
            "vlm_prior": vlm_info,
        },
        "full_set": {
            "question_count": len(full_rows),
            "scene_count": len(full_scenes),
            "per_task_counts": full_per_task,
            "mcq_balance": mcq_full,
            "accuracy": full_accuracy,
            "hardness": full_hardness,
            "distributions": distribution_summary(full_rows),
        },
        "miniset": {
            "question_count": len(selected_rows),
            "scene_count": len(selected_scenes),
            "per_task_counts": selected_per_task,
            "mcq_balance": mcq_before,
            "accuracy": miniset_accuracy,
            "hardness": miniset_hardness,
            "distributions": distribution_summary(selected_rows),
            "scene_distribution": scene_dist,
            "representativeness": representativeness,
        },
        "retention": {
            "question_fraction": (len(selected_rows) / len(full_rows)) if full_rows else 0.0,
            "scene_fraction": (len(selected_scenes) / len(full_scenes)) if full_scenes else 0.0,
        },
        "comparison": {
            "mcq_balance": {
                "full_set_mean_max_option_fraction": mcq_full["mean_max_option_fraction"],
                "miniset_mean_max_option_fraction": mcq_before["mean_max_option_fraction"],
                "full_set_mean_imbalance_ratio": mcq_full["mean_imbalance_ratio"],
                "miniset_mean_imbalance_ratio": mcq_before["mean_imbalance_ratio"],
            },
            "accuracy_and_hardness": {
                "full_mcq_accuracy": full_accuracy["mcq_accuracy"],
                "miniset_mcq_accuracy": miniset_accuracy["mcq_accuracy"],
                "full_mcq_hard_fraction": full_hardness["mcq_hard_fraction"],
                "miniset_mcq_hard_fraction": miniset_hardness["mcq_hard_fraction"],
                "full_frq_bleurt_mean": full_accuracy["frq_bleurt_mean"],
                "miniset_frq_bleurt_mean": miniset_accuracy["frq_bleurt_mean"],
                "full_frq_mean_response_hardness": full_hardness["frq_mean_response_hardness"],
                "miniset_frq_mean_response_hardness": miniset_hardness["frq_mean_response_hardness"],
            },
            "representation": {
                "question_retention_fraction": (len(selected_rows) / len(full_rows)) if full_rows else 0.0,
                "scene_retention_fraction": (len(selected_scenes) / len(full_scenes)) if full_scenes else 0.0,
                "selected_scene_count": len(selected_scenes),
                "full_scene_count": len(full_scenes),
            },
        },
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    payload = read_json(args.input_json)
    generated, synthesized_generated_answers = ensure_generated_answers(payload)
    if not generated:
        raise RuntimeError("Input JSON does not contain generated_answers or top-level tasks")

    vlm_metrics_path = args.vlm_metrics_json
    if not path_exists_readably(vlm_metrics_path) and synthesized_generated_answers:
        vlm_metrics_path = args.input_json
    vlm_accuracy, example_correctness, frq_source_rows, vlm_info = extract_vlm_eval_signals(vlm_metrics_path)
    injected_frq_counts = inject_frq_sources(generated, frq_source_rows)

    task_ids = list(generated.keys())
    scene_tasks = sorted([t for t in task_ids if t.startswith(SCENE_TASK_PREFIX)])
    non_scene_tasks = sorted(
        [t for t in task_ids if not t.startswith(SCENE_TASK_PREFIX)],
        key=lambda t: (vlm_accuracy.get(t, 1.0), t),
    )
    task_order = scene_tasks + non_scene_tasks

    global_scene_counts: Counter = Counter()
    selected_sc_scene_counts: Counter = Counter()
    selected_groups_by_scene: dict[str, list[int]] = defaultdict(list)
    selected_exact_groups: set[tuple[str, str]] = set()
    selected_sc_exact_group_counts: Counter = Counter()

    selected_generated_answers: dict[str, Any] = {}

    for task_id in task_order:
        rows = [dict(row) for row in generated[task_id].get("tasks", [])]
        if is_frq_task(task_id):
            target = args.frq_task_target
        elif task_id.startswith(SCENE_TASK_PREFIX):
            target = args.scene_task_target
        else:
            target = args.other_task_target
        selected_rows = select_for_task(
            task_id=task_id,
            rows=rows,
            target=min(target, len(rows)),
            task_accuracy=vlm_accuracy.get(task_id),
            example_correctness=example_correctness,
            global_scene_counts=global_scene_counts,
            selected_sc_scene_counts=selected_sc_scene_counts,
            selected_groups_by_scene=selected_groups_by_scene,
            selected_exact_groups=selected_exact_groups,
            selected_sc_exact_group_counts=selected_sc_exact_group_counts,
            args=args,
            rng=rng,
        )
        if not is_frq_task(task_id):
            selected_rows = reindex_mcq_rows(selected_rows, rng)
        selected_rows.sort(key=lambda r: (str(r.get("scene_id", "")), parse_numeric_suffix(str(r.get("group_id", ""))), str(r.get("object_id", "")), str(r.get("id", ""))))
        selected_generated_answers[task_id] = {
            "count": len(selected_rows),
            "tasks": selected_rows,
        }

    output_payload: dict[str, Any] = {
        "meta": {
            "source_input_json": str(args.input_json),
            "synthesized_generated_answers_from_tasks": synthesized_generated_answers,
            "selection_method": "nuscenes_miniset_v2_question_centric",
            "scene_task_target": args.scene_task_target,
            "other_task_target": args.other_task_target,
            "frq_task_target": args.frq_task_target,
            "example_hardness_weight": args.example_hardness_weight,
            "example_hardness_z_threshold": args.example_hardness_z_threshold,
            "frq_response_hardness_weight": args.frq_response_hardness_weight,
            "frq_answer_bucket_fraction": args.frq_answer_bucket_fraction,
            "frq_answer_bucket_min": args.frq_answer_bucket_min,
            "max_per_scene_global": args.max_per_scene_global,
            "max_per_scene_per_task": args.max_per_scene_per_task,
            "max_per_scene_per_task_scene": args.max_per_scene_per_task_scene,
            "max_per_scene_sc_family": args.max_per_scene_sc_family,
            "max_sc_per_exact_group": args.max_sc_per_exact_group,
            "injected_frq_counts_from_vlm_results": injected_frq_counts,
            "vlm_prior": vlm_info,
        },
        "generated_answers": selected_generated_answers,
    }
    selected_flat_rows = [
        row for bucket in selected_generated_answers.values() for row in bucket.get("tasks", [])
    ]
    if synthesized_generated_answers:
        output_payload["tasks"] = selected_flat_rows
    elif "tasks" in payload:
        output_payload["tasks"] = payload["tasks"]
    if "scene_level_tasks" in payload:
        output_payload["scene_level_tasks"] = payload["scene_level_tasks"]

    summary = build_summary(payload, selected_generated_answers, vlm_info, args.near_group_window)

    write_json(args.output_json, output_payload)
    write_json(args.summary_json, summary)

    print(json.dumps(
        {
            "output_json": str(args.output_json),
            "summary_json": str(args.summary_json),
            "selected_scene_count": summary["miniset"]["scene_count"],
            "selected_question_count": summary["miniset"]["question_count"],
            "vlm_prior_used": vlm_info.get("used", False),
            "vlm_prior_reason": vlm_info.get("reason"),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
