#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("/local1/lieqiliu/nuscenes/fullset/questions_with_answers_all_qwen3vl30b.json")
DEFAULT_OUTPUT_JSON = Path("/local1/lieqiliu/nuscenes/fullset/qwen3vl30b_question_type_summary.json")
DEFAULT_OUTPUT_CSV = Path("/local1/lieqiliu/nuscenes/fullset/qwen3vl30b_question_type_summary.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize full NuScenes Qwen annotations by question id."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
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


def is_scored_bool(value: Any) -> bool:
    return isinstance(value, bool)


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


def summarize(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    by_id: dict[str, Counter[str]] = defaultdict(Counter)
    bleurt_by_id: dict[str, list[float]] = defaultdict(list)
    random_baseline_by_id: dict[str, list[float]] = defaultdict(list)
    overall_bleurt: list[float] = []
    overall_random_baseline: list[float] = []

    for row in tasks:
        qid = str(row.get("id", ""))
        fmt = str(row.get("question_format", ""))
        counters = by_id[qid]
        counters["total"] += 1
        counters[fmt or "UNKNOWN"] += 1
        response = str(row.get("model_response", ""))
        if response:
            counters["answered"] += 1
        if response.startswith("[ERROR]"):
            counters["error_responses"] += 1
        correctness = row.get("model_is_correct")
        if is_scored_bool(correctness):
            counters["scored_accuracy_rows"] += 1
            counters["correct"] += int(correctness)
        else:
            counters["missing_accuracy_rows"] += 1
        bleurt = row.get("bleurt_model_gt")
        if isinstance(bleurt, (int, float)):
            value = float(bleurt)
            bleurt_by_id[qid].append(value)
            overall_bleurt.append(value)
        random_baseline = row.get("model_random_baseline")
        if isinstance(random_baseline, (int, float)):
            value = float(random_baseline)
            random_baseline_by_id[qid].append(value)
            overall_random_baseline.append(value)

    per_question_id: dict[str, dict[str, Any]] = {}
    for qid, counters in sorted(by_id.items()):
        scored = counters["scored_accuracy_rows"]
        correct = counters["correct"]
        accuracy = correct / scored if scored else None
        random_guess_baseline = (
            statistics.mean(random_baseline_by_id[qid]) if random_baseline_by_id[qid] else None
        )
        per_question_id[qid] = {
            "total": counters["total"],
            "question_format": "MCQ" if counters["MCQ"] else "FRQ" if counters["FRQ"] else "UNKNOWN",
            "answered": counters["answered"],
            "error_responses": counters["error_responses"],
            "accuracy_scored_rows": scored,
            "correct": correct,
            "accuracy": accuracy,
            "random_guess_baseline": random_guess_baseline,
            "random_guess_expected_correct": random_guess_baseline * scored
            if random_guess_baseline is not None
            else None,
            "accuracy_minus_random_guess": accuracy - random_guess_baseline
            if accuracy is not None and random_guess_baseline is not None
            else None,
            "missing_accuracy_rows": counters["missing_accuracy_rows"],
            "bleurt_model_gt": bucket(bleurt_by_id[qid]),
        }

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "total_tasks": len(tasks),
        "by_question_format": dict(sorted(Counter(row.get("question_format") for row in tasks).items())),
        "overall_random_guess_baseline": bucket(overall_random_baseline),
        "overall_bleurt_model_gt": bucket(overall_bleurt),
        "per_question_id": per_question_id,
    }


def write_csv(path: Path, per_question_id: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "question_format",
                "total",
                "answered",
                "error_responses",
                "accuracy_scored_rows",
                "correct",
                "accuracy",
                "random_guess_baseline",
                "random_guess_expected_correct",
                "accuracy_minus_random_guess",
                "missing_accuracy_rows",
                "bleurt_count",
                "bleurt_mean",
                "bleurt_min",
                "bleurt_max",
            ],
        )
        writer.writeheader()
        for qid, row in per_question_id.items():
            bleurt = row["bleurt_model_gt"]
            writer.writerow(
                {
                    "id": qid,
                    "question_format": row["question_format"],
                    "total": row["total"],
                    "answered": row["answered"],
                    "error_responses": row["error_responses"],
                    "accuracy_scored_rows": row["accuracy_scored_rows"],
                    "correct": row["correct"],
                    "accuracy": row["accuracy"],
                    "random_guess_baseline": row["random_guess_baseline"],
                    "random_guess_expected_correct": row["random_guess_expected_correct"],
                    "accuracy_minus_random_guess": row["accuracy_minus_random_guess"],
                    "missing_accuracy_rows": row["missing_accuracy_rows"],
                    "bleurt_count": bleurt["count"],
                    "bleurt_mean": bleurt["mean"],
                    "bleurt_min": bleurt["min"],
                    "bleurt_max": bleurt["max"],
                }
            )


def main() -> None:
    args = parse_args()
    payload = read_json(args.input)
    tasks = payload.get("tasks", [])
    if not isinstance(tasks, list):
        raise ValueError("Input JSON must contain a top-level tasks list.")

    summary = summarize(tasks)
    summary["input"] = str(args.input.resolve())
    write_json(args.output_json, summary)
    write_csv(args.output_csv, summary["per_question_id"])

    print(f"Total tasks: {summary['total_tasks']}")
    print(f"Question formats: {summary['by_question_format']}")
    print(f"BLEURT rows: {summary['overall_bleurt_model_gt']['count']}")
    print(f"Wrote: {args.output_json}")
    print(f"Wrote: {args.output_csv}")


if __name__ == "__main__":
    main()
