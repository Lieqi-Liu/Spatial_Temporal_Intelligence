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


def is_mcq(row: dict[str, Any]) -> bool:
    return row.get("question_format") == "MCQ" and isinstance(row.get("choices"), dict) and bool(row.get("choices"))


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


def extract_vlm_task_accuracy(vlm_path: Path) -> tuple[dict[str, float], dict[str, Any]]:
    if not vlm_path.exists():
        return {}, {"used": False, "reason": "file_missing", "path": str(vlm_path)}

    try:
        payload = read_json(vlm_path)
        metrics = payload.get("meta", {}).get("per_task_metrics", {})
        result = {}
        for task_id, metric in metrics.items():
            try:
                result[str(task_id)] = float(metric.get("accuracy", 0.0))
            except Exception:
                continue
        return result, {"used": True, "reason": "parsed_full_json", "path": str(vlm_path), "task_count": len(result)}
    except Exception as exc:
        raw = vlm_path.read_text(encoding="utf-8", errors="ignore")
        anchor = raw.find('"per_task_metrics"')
        if anchor < 0:
            return {}, {"used": False, "reason": f"json_parse_failed:{type(exc).__name__}", "path": str(vlm_path)}
        brace_start = raw.find("{", anchor)
        if brace_start < 0:
            return {}, {"used": False, "reason": f"metrics_block_missing:{type(exc).__name__}", "path": str(vlm_path)}
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
            return {}, {"used": False, "reason": "metrics_block_truncated", "path": str(vlm_path)}
        try:
            metrics = json.loads(raw[brace_start:end])
            result = {}
            for task_id, metric in metrics.items():
                try:
                    result[str(task_id)] = float(metric.get("accuracy", 0.0))
                except Exception:
                    continue
            return result, {"used": True, "reason": "salvaged_per_task_metrics", "path": str(vlm_path), "task_count": len(result)}
        except Exception as exc2:
            return {}, {"used": False, "reason": f"salvage_failed:{type(exc2).__name__}", "path": str(vlm_path)}


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
    global_scene_counts: Counter,
    selected_groups_by_scene: dict[str, list[int]],
    selected_exact_groups: set[tuple[str, str]],
    args: argparse.Namespace,
    rng: random.Random,
) -> list[dict[str, Any]]:
    remaining = rows[:]
    rng.shuffle(remaining)
    task_scene_counts: Counter = Counter()
    selected: list[dict[str, Any]] = []

    while remaining and len(selected) < target:
        best_index = 0
        best_score = None
        for idx, row in enumerate(remaining):
            score = candidate_score(
                row=row,
                task_accuracy=task_accuracy,
                global_scene_counts=global_scene_counts,
                task_scene_counts=task_scene_counts,
                selected_groups_by_scene=selected_groups_by_scene,
                selected_exact_groups=selected_exact_groups,
                args=args,
            )
            if best_score is None or score > best_score:
                best_score = score
                best_index = idx

        chosen = remaining.pop(best_index)
        scene_id = str(chosen.get("scene_id", ""))
        group_id = str(chosen.get("group_id", ""))
        group_idx = parse_numeric_suffix(group_id)

        selected.append(chosen)
        global_scene_counts[scene_id] += 1
        task_scene_counts[scene_id] += 1
        selected_groups_by_scene[scene_id].append(group_idx)
        selected_exact_groups.add((scene_id, group_id))

    return selected


def build_summary(
    input_payload: dict[str, Any],
    selected_generated_answers: dict[str, Any],
    vlm_info: dict[str, Any],
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

    scene_dist = {
        "selected_scene_count": len(selected_scenes),
        "full_scene_count": len(full_scenes),
        "scene_fraction": (len(selected_scenes) / len(full_scenes)) if full_scenes else 0.0,
        "top_selected_scenes": dict(selected_scenes.most_common(25)),
    }

    return {
        "meta": {
            "description": "NuScenes miniset v2 summary",
            "vlm_prior": vlm_info,
        },
        "full_set": {
            "question_count": len(full_rows),
            "scene_count": len(full_scenes),
            "per_task_counts": full_per_task,
            "mcq_balance": mcq_full,
        },
        "miniset": {
            "question_count": len(selected_rows),
            "scene_count": len(selected_scenes),
            "per_task_counts": selected_per_task,
            "mcq_balance": mcq_before,
            "scene_distribution": scene_dist,
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
            }
        },
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    payload = read_json(args.input_json)
    generated = payload.get("generated_answers", {})
    if not generated:
        raise RuntimeError("Input JSON does not contain generated_answers")

    vlm_accuracy, vlm_info = extract_vlm_task_accuracy(args.vlm_metrics_json)

    task_ids = list(generated.keys())
    scene_tasks = sorted([t for t in task_ids if t.startswith(SCENE_TASK_PREFIX)])
    non_scene_tasks = sorted(
        [t for t in task_ids if not t.startswith(SCENE_TASK_PREFIX)],
        key=lambda t: (vlm_accuracy.get(t, 1.0), t),
    )
    task_order = scene_tasks + non_scene_tasks

    global_scene_counts: Counter = Counter()
    selected_groups_by_scene: dict[str, list[int]] = defaultdict(list)
    selected_exact_groups: set[tuple[str, str]] = set()

    selected_generated_answers: dict[str, Any] = {}

    for task_id in task_order:
        rows = [dict(row) for row in generated[task_id].get("tasks", [])]
        target = args.scene_task_target if task_id.startswith(SCENE_TASK_PREFIX) else args.other_task_target
        selected_rows = select_for_task(
            task_id=task_id,
            rows=rows,
            target=min(target, len(rows)),
            task_accuracy=vlm_accuracy.get(task_id),
            global_scene_counts=global_scene_counts,
            selected_groups_by_scene=selected_groups_by_scene,
            selected_exact_groups=selected_exact_groups,
            args=args,
            rng=rng,
        )
        selected_rows = reindex_mcq_rows(selected_rows, rng)
        selected_rows.sort(key=lambda r: (str(r.get("scene_id", "")), parse_numeric_suffix(str(r.get("group_id", ""))), str(r.get("object_id", "")), str(r.get("id", ""))))
        selected_generated_answers[task_id] = {
            "count": len(selected_rows),
            "tasks": selected_rows,
        }

    output_payload: dict[str, Any] = {
        "meta": {
            "source_input_json": str(args.input_json),
            "selection_method": "nuscenes_miniset_v2_question_centric",
            "scene_task_target": args.scene_task_target,
            "other_task_target": args.other_task_target,
            "vlm_prior": vlm_info,
        },
        "generated_answers": selected_generated_answers,
    }
    if "tasks" in payload:
        output_payload["tasks"] = payload["tasks"]
    if "scene_level_tasks" in payload:
        output_payload["scene_level_tasks"] = payload["scene_level_tasks"]

    summary = build_summary(payload, selected_generated_answers, vlm_info)

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
