#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from random import Random
from typing import Any


DEFAULT_INPUT_JSON = Path(
    "/home/rgao727/Spatial_Temporal_Intelligence/waymo_test/waymo_e2e_questions_5frame_stride5_full_gt_trj_completed.json"
)
DEFAULT_REVIEW_JSON = Path(
    "/home/rgao727/Spatial_Temporal_Intelligence/waymo_test/waymo_sc_trj_review_records.json"
)
DEFAULT_OUTPUT_JSON = Path(
    "/home/rgao727/Spatial_Temporal_Intelligence/waymo_test/waymo_e2e_questions_5frame_stride5_full_gt_review_merged_mcq4.json"
)

REVIEW_ACCEPT_STATUSES = {"approved", "revised_approved"}

SC1_LABEL_NORMALIZATION = {
    "busy intersection": "traffic intersection",
    "intersection": "traffic intersection",
    "traffic intersection": "traffic intersection",
    "night intersection": "traffic intersection",
    "night traffic intersection": "traffic intersection",
    "night time intersection": "traffic intersection",
    "turning left at traffic intersection": "left turn",
    "t-intersection": "t-intersection",
    "night road": "urban road",
    "night urban road": "urban road",
    "night rainy urban street": "urban road",
    "urban road": "urban road",
    "narrow urban road": "urban road",
    "residential street": "residential street",
    "residential road": "residential street",
    "narrow residential street": "residential street",
    "rural road": "rural road",
    "highway": "highway",
    "highway merge": "highway merge",
    "merging scenario": "highway merge",
    "construction zone": "construction zone",
    "left turn": "left turn",
    "right turn": "right turn",
    "pedestrian crossing": "pedestrian crossing",
    "parking lot": "parking lot",
    "single lane": "urban road",
}

SC1_LABEL_POOL = [
    "traffic intersection",
    "t-intersection",
    "urban road",
    "residential street",
    "rural road",
    "highway",
    "highway merge",
    "construction zone",
    "left turn",
    "right turn",
    "pedestrian crossing",
    "parking lot",
]

TRJ10_DISTRACTOR_POOL = [
    "It would likely crash into a pedestrian, cyclist, or another vulnerable road user",
    "It would likely crash into another car or vehicle",
    "It would likely hit a barrier, curb, cone, or roadside object",
    "It would likely violate a traffic light, sign, lane rule, or right-of-way rule",
    "It would likely enter opposing traffic or create a head-on conflict",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge approved review records back into GT and clean SC-1/TRJ-10 MCQ choices."
    )
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--review-json", type=Path, default=DEFAULT_REVIEW_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def stable_rng(key: str) -> Random:
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return Random(int(digest[:8], 16))


def normalize_sc1_label(text: str) -> str:
    raw = " ".join(str(text or "").strip().split())
    lowered = raw.lower()
    normalized = SC1_LABEL_NORMALIZATION.get(lowered, lowered)
    return normalized


def sc1_family(label: str) -> str:
    label = normalize_sc1_label(label)
    if label in {"urban road", "residential street", "rural road", "highway", "highway merge"}:
        return "road"
    if label in {"traffic intersection", "t-intersection"}:
        return "intersection"
    if label in {"left turn", "right turn"}:
        return "turn"
    if label in {"construction zone"}:
        return "construction"
    if label in {"pedestrian crossing"}:
        return "pedestrian"
    if label in {"parking lot"}:
        return "parking"
    return "other"


def normalize_trj10_text(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def trj10_family(text: str) -> str:
    lowered = normalize_trj10_text(text).lower()
    if any(token in lowered for token in ["pedestrian", "cyclist", "vulnerable road user"]):
        return "vru_collision"
    if any(token in lowered for token in ["another car", "another vehicle", "crash into another", "another road user"]):
        return "vehicle_collision"
    if any(token in lowered for token in ["barrier", "curb", "cone", "roadside object", "roadside wall", "vegetation"]):
        return "object_collision"
    if any(token in lowered for token in ["leave its lane", "wrong lane", "road edge", "road edge", "off the intended path", "lose lane control", "drift"]):
        return "lane_departure"
    if any(token in lowered for token in ["sudden braking", "sharp corrective", "reaction time"]):
        return "corrective_action"
    if any(token in lowered for token in ["stuck", "stop unnecessarily", "block traffic", "insufficient progress"]):
        return "blocked_progress"
    if any(token in lowered for token in ["traffic light", "sign", "right-of-way", "violate"]):
        return "rule_violation"
    return "other"


def build_four_choice_mcq(correct_text: str, pool: list[str], key: str, family_fn) -> tuple[dict[str, str], str]:
    correct_text = correct_text.strip()
    rng = stable_rng(key)
    distractor_candidates = [text for text in pool if text.strip() and text.strip() != correct_text]
    rng.shuffle(distractor_candidates)

    selected = [correct_text]
    used_families = {family_fn(correct_text)}
    for candidate in distractor_candidates:
        family = family_fn(candidate)
        if family not in used_families:
            selected.append(candidate)
            used_families.add(family)
        if len(selected) == 4:
            break

    if len(selected) < 4:
        for candidate in distractor_candidates:
            if candidate not in selected:
                selected.append(candidate)
            if len(selected) == 4:
                break

    rng.shuffle(selected)
    letters = ["A", "B", "C", "D"]
    choices = {letter: text for letter, text in zip(letters, selected)}
    ground_truth = next(letter for letter, text in choices.items() if text == correct_text)
    return choices, ground_truth


def build_sc1_choices(correct_text: str, key: str) -> tuple[dict[str, str], str]:
    correct_text = normalize_sc1_label(correct_text)
    rng = stable_rng(f"sc1:{key}")
    pool = list(SC1_LABEL_POOL)
    rng.shuffle(pool)
    correct_family = sc1_family(correct_text)

    selected = [correct_text]
    used_families = {correct_family}
    for candidate in pool:
        if candidate == correct_text:
            continue
        family = sc1_family(candidate)
        if correct_family == "road" and family == "road":
            continue
        if family in used_families:
            continue
        selected.append(candidate)
        used_families.add(family)
        if len(selected) == 4:
            break

    if len(selected) < 4:
        for candidate in pool:
            if candidate == correct_text or candidate in selected:
                continue
            if correct_family == "road" and sc1_family(candidate) == "road":
                continue
            selected.append(candidate)
            if len(selected) == 4:
                break

    rng.shuffle(selected)
    letters = ["A", "B", "C", "D"]
    choices = {letter: text for letter, text in zip(letters, selected)}
    ground_truth = next(letter for letter, text in choices.items() if text == correct_text)
    return choices, ground_truth


def pick_trj10_final_text(record: dict[str, Any], item: dict[str, Any]) -> str:
    revised_text = normalize_trj10_text(record.get("revised_ground_truth_text", ""))
    custom_text = normalize_trj10_text(record.get("revised_custom_choice_text", ""))
    original_text = normalize_trj10_text(record.get("original_ground_truth_text", item.get("ground_truth_text", "")))

    if custom_text:
        if not revised_text or revised_text == original_text or len(revised_text) < 20:
            return custom_text
    return revised_text or custom_text or normalize_trj10_text(item.get("ground_truth_text", ""))


def apply_review_record(item: dict[str, Any], record: dict[str, Any], stats: Counter) -> None:
    task_id = str(item.get("id", ""))
    item["review_status"] = record.get("review_status", "")
    item["review_note"] = record.get("review_note", "")

    if task_id == "SC-1":
        final_text = normalize_sc1_label(record.get("revised_ground_truth_text", item.get("ground_truth_text", "")))
        choices, ground_truth = build_sc1_choices(final_text, str(item.get("question_id", "")))
        item["choices"] = choices
        item["ground_truth"] = ground_truth
        item["ground_truth_text"] = final_text
        stats["sc1_applied"] += 1
        if final_text == "urban road":
            stats["sc1_night_road_removed"] += 1
        return

    if task_id == "TRJ-10":
        final_text = pick_trj10_final_text(record, item)
        choices, ground_truth = build_four_choice_mcq(
            final_text,
            TRJ10_DISTRACTOR_POOL,
            f"trj10:{item.get('question_id', '')}",
            trj10_family,
        )
        revised_question_text = str(record.get("revised_question_text", "")).strip()
        revised_action = str(
            record.get("revised_custom_action_text") or record.get("revised_alternative_maneuver") or ""
        ).strip()
        if revised_question_text:
            item["question"] = revised_question_text
        elif revised_action:
            item["question"] = f"If the ego vehicle {revised_action}, what is the most likely consequence?"
        item["choices"] = choices
        item["ground_truth"] = ground_truth
        item["ground_truth_text"] = final_text
        stats["trj10_applied"] += 1
        return

    revised_gt = record.get("revised_ground_truth")
    revised_gt_text = record.get("revised_ground_truth_text")
    if revised_gt not in (None, ""):
        item["ground_truth"] = revised_gt
    if revised_gt_text not in (None, ""):
        item["ground_truth_text"] = revised_gt_text
    stats["other_review_applied"] += 1


def rewrite_dataset(payload: dict[str, Any], review_records: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], Counter]:
    stats: Counter = Counter()

    for item in payload.get("questions", []):
        question_id = str(item.get("question_id", ""))
        task_id = str(item.get("id", ""))
        record = review_records.get(question_id)
        if not record or record.get("review_status") not in REVIEW_ACCEPT_STATUSES:
            continue
        apply_review_record(item, record, stats)
        stats["review_records_applied"] += 1
        if task_id in {"SC-1", "TRJ-10"}:
            stats["mcq_cleaned_to_4_options"] += 1

    stats["questions_out"] = len(payload.get("questions", []))
    return payload, stats


def main() -> None:
    args = parse_args()
    payload = load_json(args.input_json)
    review_records = load_json(args.review_json)

    rewritten_payload, stats = rewrite_dataset(payload, review_records)
    rewritten_payload.setdefault("rewrite_metadata", {})
    rewritten_payload["rewrite_metadata"]["merged_review_json"] = str(args.review_json)
    rewritten_payload["rewrite_metadata"]["mcq_choice_count"] = 4
    rewritten_payload["rewrite_metadata"]["sc1_night_road_policy"] = "normalize to urban road"
    rewritten_payload["rewrite_metadata"]["sc1_road_distractor_policy"] = (
        "if GT is road-family, exclude other road-family distractors"
    )

    dump_json(args.output_json, rewritten_payload)

    print("Wrote:", args.output_json)
    for key in sorted(stats):
        print(f"{key}: {stats[key]}")


if __name__ == "__main__":
    main()
