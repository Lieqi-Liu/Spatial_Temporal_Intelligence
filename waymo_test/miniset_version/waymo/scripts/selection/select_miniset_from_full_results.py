#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_QUESTIONS_JSON = SCRIPT_DIR / "waymo_validation_full_set_questions.json"
DEFAULT_RESPONSES_JSON = SCRIPT_DIR / "waymo_validation_full_set_vlm_responses.json"
DEFAULT_OUTPUT_JSON = SCRIPT_DIR / "waymo_validation_selected_miniset_from_full_results.json"
DEFAULT_SUMMARY_JSON = SCRIPT_DIR / "waymo_validation_selected_miniset_from_full_results_summary.json"

COMPLEX_KEYWORDS = {
    "left turn": 2.5,
    "turn left": 2.5,
    "right turn": 1.8,
    "turn right": 1.8,
    "merge": 2.4,
    "highway merge": 3.0,
    "cut-in": 2.4,
    "cut in": 2.4,
    "intersection": 2.2,
    "traffic intersection": 2.2,
    "crosswalk": 1.8,
    "pedestrian": 1.8,
    "cyclist": 1.8,
    "construction": 2.0,
    "debris": 1.6,
    "foreign object": 1.6,
    "special vehicle": 1.5,
    "emergency": 1.8,
    "lane change": 1.8,
    "change lane": 1.8,
    "adjacent lane": 1.2,
    "dense": 1.5,
    "night": 1.2,
    "occluded": 1.2,
    "barrier": 1.0,
    "vulnerable road user": 1.5,
}

SIMPLE_KEYWORDS = {
    "residential street": 2.5,
    "single-lane maneuvers": 1.5,
    "others": 1.2,
    "moving straight": 1.2,
    "continue straight": 1.2,
    "go straight": 1.0,
    "same lane": 1.0,
    "in front": 0.8,
    "no significant event": 1.5,
    "no collision predicted": 1.5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a 100-per-question-type Waymo miniset from full-set model results. "
            "The default policy is scene-first balanced diversity with a moderate hard-example bias."
        )
    )
    parser.add_argument("--questions-json", type=Path, default=DEFAULT_QUESTIONS_JSON)
    parser.add_argument("--responses-json", type=Path, default=DEFAULT_RESPONSES_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--per-question-type", type=int, default=100)
    parser.add_argument("--seed", type=int, default=727)
    parser.add_argument(
        "--policy",
        choices=["balanced-diverse-hard", "diverse-hard", "random", "balanced-mcq", "incorrect-first", "correct-first"],
        default="balanced-diverse-hard",
    )
    parser.add_argument("--difficulty-weight", type=float, default=0.6)
    parser.add_argument("--complexity-weight", type=float, default=1.2)
    parser.add_argument("--diversity-weight", type=float, default=2.0)
    parser.add_argument("--max-simple-ratio", type=float, default=0.25)
    parser.add_argument("--mcq-hard-ratio", type=float, default=0.4)
    parser.add_argument("--mcq-correct-ratio", type=float, default=0.4)
    parser.add_argument("--scenario-soft-cap-ratio", type=float, default=0.25)
    parser.add_argument(
        "--allow-missing-responses",
        action="store_true",
        help="Allow selecting questions that have no full-set response row.",
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


def rows_from_payload(payload: Any, key: str) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get(key), list):
        return [x for x in payload[key] if isinstance(x, dict)]
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    return []


def norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def canonical_question_type(row: dict[str, Any]) -> str:
    return str(row.get("id") or "").strip()


def normalize_choice_answer(value: Any, choices: dict[str, Any], ground_truth_text: Any = "") -> str:
    valid = {str(k).strip().upper(): str(v).strip() for k, v in choices.items()}
    raw = str(value or "").strip()
    upper = raw.upper()
    if upper in valid:
        return upper
    raw_norm = norm_text(raw)
    text_norm = norm_text(ground_truth_text)
    for key, choice_text in valid.items():
        choice_norm = norm_text(choice_text)
        if choice_norm and choice_norm in {raw_norm, text_norm}:
            return key
    return upper


def extract_first_float(text: Any) -> float | None:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(text or ""))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def parse_point_sequence(text: Any) -> list[list[float]] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    candidates = [raw]
    json_match = re.search(r"\[\s*\[.*\]\s*\]", raw, flags=re.S)
    if json_match:
        candidates.append(json_match.group(0))
    code_match = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.S)
    if code_match:
        candidates.append(code_match.group(1).strip())
    for candidate in candidates:
        for parser in (json.loads, ast.literal_eval):
            try:
                payload = parser(candidate)
            except Exception:
                continue
            if not isinstance(payload, list):
                continue
            points: list[list[float]] = []
            for item in payload:
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    points = []
                    break
                try:
                    points.append([float(item[0]), float(item[1])])
                except Exception:
                    points = []
                    break
            if points:
                return points
    return None


def euclidean(a: list[float], b: list[float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def token_jaccard(a: Any, b: Any) -> float:
    toks_a = set(re.findall(r"[a-z0-9]+", norm_text(a)))
    toks_b = set(re.findall(r"[a-z0-9]+", norm_text(b)))
    if not toks_a or not toks_b:
        return 0.0
    return len(toks_a & toks_b) / len(toks_a | toks_b)


def keyword_score(text: str, weights: dict[str, float]) -> float:
    lowered = norm_text(text)
    return sum(weight for phrase, weight in weights.items() if phrase in lowered)


def collect_scene_text(row: dict[str, Any]) -> str:
    hidden = row.get("hidden_metadata") if isinstance(row.get("hidden_metadata"), dict) else {}
    structured = hidden.get("structured_scene_gt") if isinstance(hidden.get("structured_scene_gt"), dict) else {}
    pieces: list[str] = [
        row.get("scenario_cluster", ""),
        row.get("ground_truth_text", ""),
        row.get("question", ""),
        hidden.get("rule_reason", ""),
        hidden.get("reason", ""),
        hidden.get("waymo_full_scene_annotation", ""),
        hidden.get("canonical_sc1_label", ""),
        hidden.get("full_scene_best_clip_reason", ""),
    ]
    pieces.extend(str(x) for x in structured.values())
    choices = row.get("choices") if isinstance(row.get("choices"), dict) else {}
    pieces.extend(str(x) for x in choices.values())
    return " ".join(str(x) for x in pieces if x not in (None, ""))


def scenario_key(row: dict[str, Any]) -> str:
    return str(row.get("scenario_cluster") or "unknown").strip() or "unknown"


def gt_bucket(row: dict[str, Any]) -> str:
    gt_text = str(row.get("ground_truth_text") or "").strip()
    if gt_text:
        return gt_text
    return str(row.get("ground_truth") or "").strip()


def attach_response(question: dict[str, Any], response: dict[str, Any] | None) -> dict[str, Any]:
    row = json.loads(json.dumps(question, ensure_ascii=False))
    hidden = row.setdefault("hidden_metadata", {})
    choices = row.get("choices") if isinstance(row.get("choices"), dict) else {}
    predicted = str(response.get("predicted_option") or "").strip().upper() if response else ""
    normalized_gt = normalize_choice_answer(row.get("ground_truth"), choices, row.get("ground_truth_text")) if choices else str(row.get("ground_truth") or "").strip().upper()
    is_correct = None
    if response is not None:
        if choices and predicted:
            is_correct = predicted == normalized_gt
        else:
            is_correct = response.get("is_correct")
    full_eval = {
        "has_response": response is not None,
        "model_response": response.get("model_response") if response else "",
        "predicted_option": predicted or None,
        "normalized_ground_truth": normalized_gt,
        "original_response_is_correct": response.get("is_correct") if response else None,
        "is_correct": is_correct,
    }
    hidden["full_set_eval"] = full_eval
    return row


def difficulty_score(row: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    hidden = row.get("hidden_metadata") if isinstance(row.get("hidden_metadata"), dict) else {}
    full_eval = hidden.get("full_set_eval") if isinstance(hidden.get("full_set_eval"), dict) else {}
    qid = canonical_question_type(row)
    model_response = full_eval.get("model_response", "")
    if not full_eval.get("has_response"):
        return 0.8, {"difficulty_reason": "missing_response"}

    choices = row.get("choices") if isinstance(row.get("choices"), dict) else {}
    if choices:
        if not full_eval.get("predicted_option"):
            return 1.0, {"difficulty_reason": "invalid_mcq"}
        if full_eval.get("is_correct") is False:
            return 1.0, {"difficulty_reason": "incorrect_mcq"}
        return 0.15, {"difficulty_reason": "correct_mcq"}

    if qid == "SP-3a":
        pred = extract_first_float(model_response)
        gt = extract_first_float(row.get("ground_truth"))
        if pred is None or gt is None:
            return 1.0, {"difficulty_reason": "distance_parse_fail"}
        err = abs(pred - gt)
        return min(1.0, err / 20.0), {"difficulty_reason": "distance_error", "abs_error_m": err}

    if qid in {"TRJ-5", "TRJ-6"}:
        pred = parse_point_sequence(model_response)
        gt = parse_point_sequence(row.get("ground_truth"))
        if pred is None or gt is None or not gt:
            return 1.0, {"difficulty_reason": "trajectory_parse_fail"}
        compare_len = min(len(pred), len(gt))
        errors = [euclidean(pred[i], gt[i]) for i in range(compare_len)]
        ade = sum(errors) / compare_len
        fde = errors[-1]
        score = min(1.0, 0.5 * (ade / 20.0) + 0.5 * (fde / 40.0))
        return score, {"difficulty_reason": "trajectory_error", "ade_m": ade, "fde_m": fde}

    if qid in {"TRJ-7", "TRJ-9"}:
        if not str(model_response or "").strip() or str(model_response).startswith("[ERROR]"):
            return 1.0, {"difficulty_reason": "text_empty_or_error"}
        overlap = token_jaccard(model_response, row.get("ground_truth"))
        return max(0.1, 1.0 - overlap), {"difficulty_reason": "text_low_overlap", "token_jaccard": overlap}

    return 0.5, {"difficulty_reason": "unscored_frq"}


def enrich_row_for_selection(row: dict[str, Any]) -> dict[str, Any]:
    text = collect_scene_text(row)
    complex_raw = keyword_score(text, COMPLEX_KEYWORDS)
    simple_raw = keyword_score(text, SIMPLE_KEYWORDS)
    scenario = scenario_key(row)
    if scenario in {"Intersection", "Cut-ins", "Construction", "Cyclist", "Pedestrian", "Multi-Lane Maneuvers"}:
        complex_raw += 1.0
    if scenario in {"Others", "Single-Lane Maneuvers"}:
        simple_raw += 1.0
    complexity = max(0.0, min(1.0, complex_raw / 6.0))
    simplicity = max(0.0, min(1.0, simple_raw / 4.0))
    difficulty, details = difficulty_score(row)
    hidden = row.setdefault("hidden_metadata", {})
    hidden["selection_from_full_set"] = {
        "difficulty_score": difficulty,
        "complexity_score": complexity,
        "simplicity_score": simplicity,
        "scenario_key": scenario,
        "gt_bucket": gt_bucket(row),
        **details,
    }
    return row


def enforce_trj10_option_prompt(row: dict[str, Any]) -> None:
    if canonical_question_type(row) != "TRJ-10":
        return
    instruction = "Return only one option letter (A, B, C, or D)."
    question = str(row.get("question", "")).strip()
    if instruction.lower() not in question.lower():
        row["question"] = f"{question} {instruction}".strip()


def simple_row(row: dict[str, Any]) -> bool:
    info = ((row.get("hidden_metadata") or {}).get("selection_from_full_set") or {})
    return float(info.get("simplicity_score") or 0.0) >= 0.5 and float(info.get("complexity_score") or 0.0) < 0.5


def base_rank(row: dict[str, Any], args: argparse.Namespace) -> float:
    info = ((row.get("hidden_metadata") or {}).get("selection_from_full_set") or {})
    return (
        args.difficulty_weight * float(info.get("difficulty_score") or 0.0)
        + args.complexity_weight * float(info.get("complexity_score") or 0.0)
        - 0.35 * float(info.get("simplicity_score") or 0.0)
    )


def row_difficulty_bucket(row: dict[str, Any]) -> str:
    choices = row.get("choices") if isinstance(row.get("choices"), dict) else {}
    full_eval = ((row.get("hidden_metadata") or {}).get("full_set_eval") or {})
    info = ((row.get("hidden_metadata") or {}).get("selection_from_full_set") or {})
    if choices:
        if full_eval.get("is_correct") is False or not full_eval.get("predicted_option"):
            return "hard"
        if full_eval.get("is_correct") is True:
            return "correct"
        return "unknown"
    difficulty = float(info.get("difficulty_score") or 0.0)
    if difficulty >= 0.67:
        return "hard"
    if difficulty <= 0.34:
        return "easy"
    return "medium"


def balanced_targets(rows: list[dict[str, Any]], n: int, args: argparse.Namespace) -> dict[str, int]:
    has_mcq = any(isinstance(r.get("choices"), dict) and r.get("choices") for r in rows)
    if has_mcq:
        hard = int(round(n * args.mcq_hard_ratio))
        correct = int(round(n * args.mcq_correct_ratio))
        return {"hard": hard, "correct": correct}
    hard = int(round(n * 0.4))
    medium = int(round(n * 0.4))
    return {"hard": hard, "medium": medium}


def select_balanced_diverse_hard(
    rows: list[dict[str, Any]],
    n: int,
    rng: random.Random,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows = sorted((enrich_row_for_selection(r) for r in rows), key=lambda r: str(r.get("question_id", "")))
    if len(rows) <= n:
        return rows

    global_scenarios = Counter(scenario_key(r) for r in rows)
    global_gt = Counter(gt_bucket(r) for r in rows)
    target_buckets = balanced_targets(rows, n, args)
    scenario_soft_cap = max(1, int(round(n * args.scenario_soft_cap_ratio)))
    simple_limit = max(1, int(round(n * args.max_simple_ratio)))

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    scene_counts: Counter[str] = Counter()
    bundle_counts: Counter[str] = Counter()
    scenario_counts: Counter[str] = Counter()
    gt_counts: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()

    while len(selected) < n:
        best: tuple[float, float, str, dict[str, Any]] | None = None
        remaining_slots = n - len(selected)
        remaining = [r for r in rows if str(r.get("question_id", "")) not in selected_ids]
        if not remaining:
            break
        non_simple_remaining = sum(1 for r in remaining if not simple_row(r))
        simple_count = sum(1 for r in selected if simple_row(r))

        for row in remaining:
            q_uid = str(row.get("question_id", ""))
            scene = str(row.get("scene_id", ""))
            bundle = str(row.get("bundle_id", ""))
            scen = scenario_key(row)
            gt = gt_bucket(row)
            bucket = row_difficulty_bucket(row)
            info = ((row.get("hidden_metadata") or {}).get("selection_from_full_set") or {})
            is_simple = simple_row(row)
            if is_simple and simple_count >= simple_limit and non_simple_remaining >= remaining_slots:
                continue

            scene_novelty = 1.0 if scene_counts[scene] == 0 else -1.35 * scene_counts[scene]
            bundle_novelty = 0.45 if bundle_counts[bundle] == 0 else -0.6 * bundle_counts[bundle]
            scenario_balance = 1.0 / math.sqrt(1 + scenario_counts[scen])
            gt_balance = 1.0 / math.sqrt(1 + gt_counts[gt])
            rare_scenario = 1.0 / math.sqrt(global_scenarios[scen])
            rare_gt = 1.0 / math.sqrt(global_gt[gt])

            score = (
                args.diversity_weight * (1.25 * scene_novelty + 0.55 * scenario_balance + 0.35 * gt_balance)
                + 0.45 * bundle_novelty
                + args.complexity_weight * float(info.get("complexity_score") or 0.0)
                + args.difficulty_weight * float(info.get("difficulty_score") or 0.0)
                - 0.6 * float(info.get("simplicity_score") or 0.0)
                + 0.35 * rare_scenario
                + 0.2 * rare_gt
            )

            target = target_buckets.get(bucket)
            if target is not None and bucket_counts[bucket] < target:
                score += 1.25
            elif bucket in target_buckets and bucket_counts[bucket] >= target:
                score -= 0.8

            if scenario_counts[scen] >= scenario_soft_cap:
                score -= 1.4 + 0.3 * (scenario_counts[scen] - scenario_soft_cap)

            score += rng.random() * 0.001
            key = (score, base_rank(row, args), q_uid, row)
            if best is None or key[:3] > best[:3]:
                best = key

        if best is None:
            args.max_simple_ratio = 1.0
            args.scenario_soft_cap_ratio = 1.0
            continue

        row = best[3]
        selected.append(row)
        selected_ids.add(str(row.get("question_id", "")))
        scene_counts[str(row.get("scene_id", ""))] += 1
        bundle_counts[str(row.get("bundle_id", ""))] += 1
        scenario_counts[scenario_key(row)] += 1
        gt_counts[gt_bucket(row)] += 1
        bucket_counts[row_difficulty_bucket(row)] += 1

    return sorted(selected, key=lambda r: str(r.get("question_id", "")))


def select_diverse_hard(rows: list[dict[str, Any]], n: int, rng: random.Random, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = sorted((enrich_row_for_selection(r) for r in rows), key=lambda r: str(r.get("question_id", "")))
    if len(rows) <= n:
        return rows

    global_scenarios = Counter(scenario_key(r) for r in rows)
    global_gt = Counter(gt_bucket(r) for r in rows)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    scene_counts: Counter[str] = Counter()
    bundle_counts: Counter[str] = Counter()
    scenario_counts: Counter[str] = Counter()
    gt_counts: Counter[str] = Counter()
    simple_limit = max(1, int(round(n * args.max_simple_ratio)))

    while len(selected) < n:
        best: tuple[float, float, str, dict[str, Any]] | None = None
        remaining_slots = n - len(selected)
        remaining = [r for r in rows if str(r.get("question_id", "")) not in selected_ids]
        if not remaining:
            break
        non_simple_remaining = sum(1 for r in remaining if not simple_row(r))
        simple_count = sum(1 for r in selected if simple_row(r))

        for row in remaining:
            q_uid = str(row.get("question_id", ""))
            scene = str(row.get("scene_id", ""))
            bundle = str(row.get("bundle_id", ""))
            scen = scenario_key(row)
            gt = gt_bucket(row)
            is_simple = simple_row(row)
            if is_simple and simple_count >= simple_limit and non_simple_remaining >= remaining_slots:
                continue

            score = base_rank(row, args)
            score += args.diversity_weight * (0.75 if scene_counts[scene] == 0 else -0.35 * scene_counts[scene])
            score += args.diversity_weight * (0.35 if bundle_counts[bundle] == 0 else -0.4 * bundle_counts[bundle])
            score += args.diversity_weight * (0.35 / math.sqrt(1 + scenario_counts[scen]))
            score += args.diversity_weight * (0.25 / math.sqrt(1 + gt_counts[gt]))
            score += 0.25 / math.sqrt(global_scenarios[scen])
            score += 0.15 / math.sqrt(global_gt[gt])
            score += rng.random() * 0.001
            key = (score, base_rank(row, args), q_uid, row)
            if best is None or key[:3] > best[:3]:
                best = key

        if best is None:
            # Relax the simple-scene cap if the remaining pool is too constrained.
            args.max_simple_ratio = 1.0
            continue
        row = best[3]
        selected.append(row)
        selected_ids.add(str(row.get("question_id", "")))
        scene_counts[str(row.get("scene_id", ""))] += 1
        bundle_counts[str(row.get("bundle_id", ""))] += 1
        scenario_counts[scenario_key(row)] += 1
        gt_counts[gt_bucket(row)] += 1

    return sorted(selected, key=lambda r: str(r.get("question_id", "")))


def select_legacy(rows: list[dict[str, Any]], n: int, policy: str, rng: random.Random) -> list[dict[str, Any]]:
    rows = sorted((enrich_row_for_selection(r) for r in rows), key=lambda r: str(r.get("question_id", "")))
    if len(rows) <= n:
        return rows
    if policy == "random":
        return sorted(rng.sample(rows, n), key=lambda r: str(r.get("question_id", "")))

    def is_correct(row: dict[str, Any]) -> bool | None:
        return ((row.get("hidden_metadata") or {}).get("full_set_eval") or {}).get("is_correct")

    correct = [r for r in rows if is_correct(r) is True]
    incorrect = [r for r in rows if is_correct(r) is False]
    unknown = [r for r in rows if is_correct(r) is None]
    if policy == "incorrect-first":
        return (incorrect + correct + unknown)[:n]
    if policy == "correct-first":
        return (correct + incorrect + unknown)[:n]
    half = n // 2
    selected = []
    selected.extend(rng.sample(incorrect, min(len(incorrect), half)))
    selected.extend(rng.sample(correct, min(len(correct), n - len(selected))))
    remaining = [r for r in rows if r not in selected]
    if len(selected) < n:
        selected.extend(rng.sample(remaining, n - len(selected)))
    return sorted(selected, key=lambda r: str(r.get("question_id", "")))


def summarize_selected(selected: list[dict[str, Any]], missing_response: int, metadata: dict[str, Any], shortages: dict[str, int]) -> dict[str, Any]:
    per_id = Counter(canonical_question_type(row) for row in selected)
    per_task = Counter(str(row.get("task", "")) for row in selected)
    per_scenario_by_id: dict[str, Counter[str]] = defaultdict(Counter)
    per_gt_by_id: dict[str, Counter[str]] = defaultdict(Counter)
    correctness_by_id: dict[str, Counter[str]] = defaultdict(Counter)
    difficulty_bucket_by_id: dict[str, Counter[str]] = defaultdict(Counter)
    score_sums: dict[str, Counter[str]] = defaultdict(Counter)
    scenes_by_id: dict[str, set[str]] = defaultdict(set)
    for row in selected:
        qid = canonical_question_type(row)
        info = ((row.get("hidden_metadata") or {}).get("selection_from_full_set") or {})
        full_eval = ((row.get("hidden_metadata") or {}).get("full_set_eval") or {})
        per_scenario_by_id[qid][scenario_key(row)] += 1
        per_gt_by_id[qid][gt_bucket(row)] += 1
        correctness_by_id[qid][str(full_eval.get("is_correct"))] += 1
        difficulty_bucket_by_id[qid][row_difficulty_bucket(row)] += 1
        score_sums[qid]["difficulty"] += float(info.get("difficulty_score") or 0.0)
        score_sums[qid]["complexity"] += float(info.get("complexity_score") or 0.0)
        score_sums[qid]["simplicity"] += float(info.get("simplicity_score") or 0.0)
        scenes_by_id[qid].add(str(row.get("scene_id", "")))

    averages = {}
    for qid, count in per_id.items():
        averages[qid] = {
            "avg_difficulty_score": score_sums[qid]["difficulty"] / count,
            "avg_complexity_score": score_sums[qid]["complexity"] / count,
            "avg_simplicity_score": score_sums[qid]["simplicity"] / count,
            "distinct_scene_count": len(scenes_by_id[qid]),
        }

    return {
        "metadata": metadata,
        "total_questions": len(selected),
        "missing_response_count_in_full_questions": missing_response,
        "shortages": shortages,
        "per_question_type_counts": dict(sorted(per_id.items())),
        "per_task_counts": dict(sorted(per_task.items())),
        "full_eval_correctness_by_question_type": {qid: dict(c) for qid, c in sorted(correctness_by_id.items())},
        "selection_score_averages_by_question_type": dict(sorted(averages.items())),
        "difficulty_bucket_distribution_by_question_type": {
            qid: dict(c.most_common()) for qid, c in sorted(difficulty_bucket_by_id.items())
        },
        "scenario_distribution_by_question_type": {qid: dict(c.most_common()) for qid, c in sorted(per_scenario_by_id.items())},
        "top_ground_truth_distribution_by_question_type": {qid: dict(c.most_common(25)) for qid, c in sorted(per_gt_by_id.items())},
    }


def main() -> None:
    args = parse_args()
    questions_payload = read_json(args.questions_json)
    responses_payload = read_json(args.responses_json)
    questions = rows_from_payload(questions_payload, "questions")
    responses = rows_from_payload(responses_payload, "responses")
    response_by_qid = {
        str(row.get("question_id")): row
        for row in responses
        if row.get("question_id") not in (None, "")
    }

    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing_response = 0
    for question in questions:
        qid = str(question.get("question_id", ""))
        response = response_by_qid.get(qid)
        if response is None:
            missing_response += 1
            if not args.allow_missing_responses:
                continue
        row = attach_response(question, response)
        by_id[canonical_question_type(row)].append(row)

    rng = random.Random(args.seed)
    selected: list[dict[str, Any]] = []
    shortages: dict[str, int] = {}
    for qid in sorted(by_id):
        rows = by_id[qid]
        if len(rows) < args.per_question_type:
            shortages[qid] = len(rows)
        if args.policy == "balanced-diverse-hard":
            selected.extend(select_balanced_diverse_hard(rows, args.per_question_type, rng, args))
        elif args.policy == "diverse-hard":
            selected.extend(select_diverse_hard(rows, args.per_question_type, rng, args))
        else:
            selected.extend(select_legacy(rows, args.per_question_type, args.policy, rng))

    selected = sorted(selected, key=lambda r: (canonical_question_type(r), str(r.get("question_id", ""))))
    for row in selected:
        enforce_trj10_option_prompt(row)
    metadata = {
        "dataset": "waymo_validation",
        "benchmark": "waymo_validation_selected_miniset_from_full_results",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "questions_json": str(args.questions_json),
        "responses_json": str(args.responses_json),
        "per_question_type": args.per_question_type,
        "seed": args.seed,
        "policy": args.policy,
        "difficulty_weight": args.difficulty_weight,
        "complexity_weight": args.complexity_weight,
        "diversity_weight": args.diversity_weight,
        "max_simple_ratio": args.max_simple_ratio,
        "mcq_hard_ratio": args.mcq_hard_ratio,
        "mcq_correct_ratio": args.mcq_correct_ratio,
        "scenario_soft_cap_ratio": args.scenario_soft_cap_ratio,
        "allow_missing_responses": args.allow_missing_responses,
        "notes": [
            "Selection is per question type, targeting 100 rows each.",
            "Balanced-diverse-hard policy prioritizes scene novelty, scenario/GT diversity, and complex/corner-case scene keywords, with a moderate hard-example quota.",
            "TRJ-10 ground-truth text is normalized back to an option key when it exactly matches one of the options.",
        ],
    }
    output = {"metadata": metadata, "questions": selected}
    summary = summarize_selected(selected, missing_response, metadata, shortages)

    write_json(args.output_json, output)
    write_json(args.summary_json, summary)
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "summary_json": str(args.summary_json),
                "total_questions": len(selected),
                "question_type_count": len(summary["per_question_type_counts"]),
                "shortages": shortages,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
