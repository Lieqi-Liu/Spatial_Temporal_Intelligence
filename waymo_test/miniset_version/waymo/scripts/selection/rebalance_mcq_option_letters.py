#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_JSON = SCRIPT_DIR / "waymo_validation_balanced_diverse_70hard_miniset_from_full_results.json"
DEFAULT_OUTPUT_JSON = SCRIPT_DIR / "waymo_validation_balanced_diverse_70hard_option_balanced_miniset_from_full_results.json"
DEFAULT_SUMMARY_JSON = SCRIPT_DIR / "waymo_validation_balanced_diverse_70hard_option_balanced_miniset_from_full_results_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebalance MCQ correct option letters by permuting choices without changing question rows."
    )
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--seed", type=int, default=727)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def token_set(value: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", norm(value)))


def token_similarity(a: Any, b: Any) -> float:
    a_norm = norm(a)
    b_norm = norm(b)
    if not a_norm or not b_norm:
        return 0.0
    if a_norm == b_norm:
        return 1.0
    if a_norm in b_norm or b_norm in a_norm:
        return 0.9
    toks_a = token_set(a_norm)
    toks_b = token_set(b_norm)
    if not toks_a or not toks_b:
        return 0.0
    return len(toks_a & toks_b) / len(toks_a | toks_b)


def canonical_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or "").strip()


def resolve_correct_key(row: dict[str, Any]) -> str | None:
    choices = row.get("choices") if isinstance(row.get("choices"), dict) else {}
    if not choices:
        return None
    gt = str(row.get("ground_truth", "")).strip()
    gt_upper = gt.upper()
    if gt_upper in choices:
        return gt_upper
    candidates = [row.get("ground_truth_text"), row.get("ground_truth")]
    for candidate in candidates:
        candidate_norm = norm(candidate)
        if not candidate_norm:
            continue
        for key, text in choices.items():
            if norm(text) == candidate_norm:
                return str(key).strip().upper()

    best_key = None
    best_score = -1.0
    for candidate in candidates:
        for key, text in choices.items():
            score = token_similarity(candidate, text)
            if score > best_score:
                best_key = str(key).strip().upper()
                best_score = score
    return best_key


def rebalance_rows(rows: list[dict[str, Any]], rng: random.Random) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_id[canonical_id(row)].append(row)

    new_rows: list[dict[str, Any]] = []
    before: dict[str, Counter[str]] = defaultdict(Counter)
    after: dict[str, Counter[str]] = defaultdict(Counter)
    changed = 0
    unresolved = 0

    for qid in sorted(by_id):
        q_rows = sorted(by_id[qid], key=lambda r: str(r.get("question_id", "")))
        rng.shuffle(q_rows)
        counts: Counter[str] = Counter()
        processed: list[dict[str, Any]] = []

        for row in q_rows:
            row = copy.deepcopy(row)
            choices = row.get("choices") if isinstance(row.get("choices"), dict) else {}
            if not choices:
                processed.append(row)
                continue

            keys = [str(k).strip().upper() for k in choices.keys()]
            original_correct_key = resolve_correct_key(row)
            if original_correct_key is None or original_correct_key not in choices:
                unresolved += 1
                processed.append(row)
                after[qid][str(row.get("ground_truth", "")).strip().upper()] += 1
                continue

            before[qid][original_correct_key] += 1
            min_count = min(counts[key] for key in keys)
            candidate_keys = [key for key in keys if counts[key] == min_count]
            desired_key = sorted(candidate_keys)[0]

            if desired_key != original_correct_key:
                new_choices = dict(choices)
                new_choices[desired_key], new_choices[original_correct_key] = (
                    new_choices[original_correct_key],
                    new_choices[desired_key],
                )
                row["choices"] = new_choices
                changed += 1

            row["ground_truth"] = desired_key
            hidden = row.setdefault("hidden_metadata", {})
            hidden["option_rebalanced"] = {
                "original_ground_truth": original_correct_key,
                "new_ground_truth": desired_key,
                "method": "choice_text_swap_within_same_question",
            }
            counts[desired_key] += 1
            after[qid][desired_key] += 1
            processed.append(row)

        new_rows.extend(sorted(processed, key=lambda r: (canonical_id(r), str(r.get("question_id", "")))))

    summary = {
        "changed_rows": changed,
        "unresolved_mcq_rows": unresolved,
        "before_correct_option_distribution_by_question_type": {
            key: dict(sorted(value.items())) for key, value in sorted(before.items())
        },
        "after_correct_option_distribution_by_question_type": {
            key: dict(sorted(value.items())) for key, value in sorted(after.items())
        },
    }
    return new_rows, summary


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    payload = read_json(args.input_json)
    questions = payload.get("questions") if isinstance(payload, dict) else None
    if not isinstance(questions, list):
        raise ValueError(f"Input JSON must contain questions list: {args.input_json}")

    new_questions, rebalance_summary = rebalance_rows(questions, rng)
    out = copy.deepcopy(payload)
    out["questions"] = new_questions
    metadata = out.setdefault("metadata", {})
    metadata["option_rebalanced_from"] = str(args.input_json)
    metadata["option_rebalanced_at_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["option_rebalance_method"] = "permute MCQ choices so correct option letters are near-even within each question type"

    per_id = Counter(canonical_id(row) for row in new_questions)
    summary = {
        "metadata": metadata,
        "total_questions": len(new_questions),
        "per_question_type_counts": dict(sorted(per_id.items())),
        **rebalance_summary,
    }
    write_json(args.output_json, out)
    write_json(args.summary_json, summary)
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "summary_json": str(args.summary_json),
                "total_questions": len(new_questions),
                "changed_rows": rebalance_summary["changed_rows"],
                "unresolved_mcq_rows": rebalance_summary["unresolved_mcq_rows"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
