#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
WAYMO_TEST_DIR = SCRIPT_DIR.parent
MINISET_DIR = WAYMO_TEST_DIR / "miniset_version"
sys.path.insert(0, str(MINISET_DIR))

import build_waymo_validation_miniset as base  # noqa: E402


DEFAULT_OUTPUT_JSON = SCRIPT_DIR / "waymo_validation_full_set_questions.json"
DEFAULT_SUMMARY_JSON = SCRIPT_DIR / "waymo_validation_full_set_summary.json"
DEFAULT_SC_TRJ_REVIEW_JSON = WAYMO_TEST_DIR / "waymo_sc_trj_review_records.json"
APPROVED_REVIEW_STATUSES = {"approved", "revised_approved", "modified", "approved_modified"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the full Waymo validation eval question set with the current benchmark rules."
    )
    parser.add_argument("--input-json", type=Path, default=base.DEFAULT_INPUT_JSON)
    parser.add_argument("--sc-review-json", type=Path, default=base.DEFAULT_SC_REVIEW_JSON)
    parser.add_argument("--sc-trj-review-json", type=Path, default=DEFAULT_SC_TRJ_REVIEW_JSON)
    parser.add_argument("--full-scene-summary-json", type=Path, default=base.DEFAULT_FULL_SCENE_SUMMARY_JSON)
    parser.add_argument("--ego-status-dir", type=Path, default=base.DEFAULT_EGO_STATUS_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--seed", type=int, default=727)
    parser.add_argument(
        "--context-frame-stride",
        type=int,
        default=None,
        help="Frame spacing for non-SC visual context. If omitted, derive from context-window-seconds.",
    )
    parser.add_argument("--context-window-seconds", type=float, default=base.DEFAULT_CONTEXT_WINDOW_SECONDS)
    parser.add_argument("--frame-rate-hz", type=float, default=base.DEFAULT_FRAME_RATE_HZ)
    parser.add_argument(
        "--require-reviewed-sc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restrict SC-1..SC-4 to approved/revised review records.",
    )
    parser.add_argument(
        "--require-reviewed-trj",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restrict TRJ-* to approved/revised trajectory review bundles.",
    )
    parser.add_argument(
        "--task-ids",
        type=str,
        default="",
        help="Optional comma-separated question ids to build, e.g. SC-1,TRJ-5.",
    )
    parser.add_argument(
        "--limit-per-question-type",
        type=int,
        default=0,
        help="Optional deterministic sample size per question type. 0 means keep all candidates.",
    )
    parser.add_argument(
        "--max-total",
        type=int,
        default=0,
        help="Optional deterministic cap after all filtering. 0 means no cap.",
    )
    return parser.parse_args()


def load_review_records(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    raw = base.read_json(path)
    if not isinstance(raw, dict):
        raise ValueError(f"SC review JSON must be keyed by question_id: {path}")
    reviews = {
        qid: row
        for qid, row in raw.items()
        if row.get("review_status") in base.APPROVED_SC_STATUSES
    }
    by_bundle = {
        str(row.get("bundle_id", "")): row
        for row in reviews.values()
        if row.get("bundle_id")
    }
    return reviews, by_bundle


def load_sc_trj_review_records(path: Path) -> tuple[dict[str, dict[str, Any]], set[str]]:
    raw = base.read_json(path)
    rows = list(raw.values()) if isinstance(raw, dict) else raw
    if not isinstance(rows, list):
        raise ValueError(f"SC/TRJ review JSON must be a dict or list: {path}")
    approved: dict[str, dict[str, Any]] = {}
    approved_trj_bundles: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("review_status") not in APPROVED_REVIEW_STATUSES:
            continue
        question_id = str(row.get("question_id", ""))
        if question_id:
            approved[question_id] = row
        task_id = str(row.get("task_id", ""))
        bundle_id = str(row.get("bundle_id", ""))
        if task_id.startswith("TRJ") and bundle_id:
            approved_trj_bundles.add(bundle_id)
    return approved, approved_trj_bundles


def mark_reviewed_trj_bundle(row: dict[str, Any], reviewed_trj_bundles: set[str]) -> None:
    if not str(row.get("id", "")).startswith("TRJ"):
        return
    bundle_id = str(row.get("bundle_id", ""))
    if bundle_id not in reviewed_trj_bundles:
        return
    hidden = row.setdefault("hidden_metadata", {})
    hidden["trj_reviewed_bundle"] = True
    hidden["trj_reviewed_bundle_id"] = bundle_id


def apply_sc_trj_review(row: dict[str, Any], review_by_qid: dict[str, dict[str, Any]]) -> dict[str, Any]:
    review = review_by_qid.get(str(row.get("question_id", "")))
    if not review:
        return row
    row = copy_row(row)
    revised_gt = review.get("revised_ground_truth")
    revised_text = review.get("revised_ground_truth_text")
    original_gt = review.get("original_ground_truth")
    original_text = review.get("original_ground_truth_text")
    if revised_gt not in (None, ""):
        row["ground_truth"] = revised_gt
    elif original_gt not in (None, ""):
        row["ground_truth"] = original_gt
    if revised_text not in (None, ""):
        row["ground_truth_text"] = revised_text
    elif original_text not in (None, ""):
        row["ground_truth_text"] = original_text
    revised_question = review.get("revised_question_text")
    if revised_question not in (None, ""):
        row["question"] = revised_question
    hidden = row.setdefault("hidden_metadata", {})
    hidden["sc_trj_review_status"] = review.get("review_status", "")
    hidden["sc_trj_review_note"] = review.get("review_note", "")
    if review.get("revised_alternative_maneuver") not in (None, ""):
        hidden["revised_alternative_maneuver"] = review.get("revised_alternative_maneuver")
    return row


def copy_row(row: dict[str, Any]) -> dict[str, Any]:
    # Avoid importing copy in the hot path unless a reviewed row actually needs mutation.
    return json.loads(json.dumps(row, ensure_ascii=False))


def maybe_limit_rows(
    rows_by_id: dict[str, list[dict[str, Any]]],
    *,
    limit_per_question_type: int,
    max_total: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for qid in sorted(rows_by_id):
        rows = sorted(rows_by_id[qid], key=lambda r: str(r.get("question_id", "")))
        if limit_per_question_type > 0 and len(rows) > limit_per_question_type:
            rows = base.sample_rows(rows, limit_per_question_type, rng)
        selected.extend(rows)

    selected = sorted(selected, key=lambda r: (base.canonical_question_type(r), str(r.get("question_id", ""))))
    if max_total > 0 and len(selected) > max_total:
        selected = rng.sample(selected, max_total)
        selected = sorted(selected, key=lambda r: (base.canonical_question_type(r), str(r.get("question_id", ""))))
    return selected


def main() -> None:
    args = parse_args()
    context_stride = base.resolve_context_frame_stride(args)
    approx_context_seconds = base.context_window_seconds(context_stride, args.frame_rate_hz)

    payload = base.read_json(args.input_json)
    questions = payload.get("questions") if isinstance(payload, dict) else None
    if not isinstance(questions, list):
        raise ValueError(f"Input JSON must contain a questions list: {args.input_json}")

    allowed_ids = {x.strip() for x in args.task_ids.split(",") if x.strip()}
    sc_reviews, sc_reviews_by_bundle = load_review_records(args.sc_review_json)
    sc_trj_reviews_by_qid, reviewed_trj_bundles = load_sc_trj_review_records(args.sc_trj_review_json)
    full_scene_by_scene_id = base.load_full_scene_responses(args.full_scene_summary_json)

    rows_by_id: dict[str, list[dict[str, Any]]] = {}
    filtered_counts: Counter[str] = Counter()
    for source_row in questions:
        qid = base.canonical_question_type(source_row)
        if allowed_ids and qid not in allowed_ids:
            continue
        if args.require_reviewed_sc and qid in base.SC_IDS and str(source_row.get("bundle_id", "")) not in sc_reviews_by_bundle:
            filtered_counts[f"{qid}:missing_review"] += 1
            continue
        if args.require_reviewed_trj and qid.startswith("TRJ") and str(source_row.get("bundle_id", "")) not in reviewed_trj_bundles:
            filtered_counts[f"{qid}:missing_trj_review"] += 1
            continue
        if not source_row.get("ground_truth") and not source_row.get("ground_truth_text"):
            filtered_counts[f"{qid}:missing_gt"] += 1
            continue

        row = base.apply_reviewed_scene_gt(source_row, sc_reviews, sc_reviews_by_bundle, full_scene_by_scene_id)
        row = apply_sc_trj_review(row, sc_trj_reviews_by_qid)
        mark_reviewed_trj_bundle(row, reviewed_trj_bundles)
        if qid in base.SC_IDS:
            row = base.apply_full_scene_visual_context(
                row,
                full_scene_by_scene_id.get(str(row.get("scene_id", ""))),
                context_stride,
            )
        else:
            row = base.apply_strided_visual_context(row, context_stride)
            if row is None:
                filtered_counts[f"{qid}:missing_2s_context"] += 1
                continue
            if not base.refresh_trj_future_ground_truth(row, args.ego_status_dir):
                filtered_counts[f"{qid}:missing_future_gt"] += 1
                continue

        hidden = row.setdefault("hidden_metadata", {})
        hidden["context_frame_stride"] = context_stride
        hidden["frame_rate_hz"] = args.frame_rate_hz
        hidden["approx_context_window_seconds"] = approx_context_seconds
        base.refresh_trj_trajectory_prompt(row)
        base.enforce_trj10_option_only_prompt(row)
        rows_by_id.setdefault(qid, []).append(row)

    selected = maybe_limit_rows(
        rows_by_id,
        limit_per_question_type=args.limit_per_question_type,
        max_total=args.max_total,
        seed=args.seed,
    )
    out = {
        "metadata": {
            "dataset": "waymo_validation",
            "benchmark": "waymo_validation_full_set",
            "source_json": str(args.input_json),
            "sc_review_json": str(args.sc_review_json),
            "sc_trj_review_json": str(args.sc_trj_review_json),
            "full_scene_summary_json": str(args.full_scene_summary_json),
            "ego_status_dir": str(args.ego_status_dir),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "selection_method": "full_filtered_eval_set",
            "seed": args.seed,
            "require_reviewed_sc": args.require_reviewed_sc,
            "require_reviewed_trj": args.require_reviewed_trj,
            "context_frame_stride": context_stride,
            "frame_rate_hz": args.frame_rate_hz,
            "context_window_seconds": approx_context_seconds,
            "requested_context_window_seconds": args.context_window_seconds,
            "limit_per_question_type": args.limit_per_question_type,
            "max_total": args.max_total,
            "notes": [
                "SC-1..SC-4 keep the existing full-scene VLM selected clip behavior when available.",
                "TRJ-* rows are restricted to trajectory bundles with approved/revised human review by default.",
                "Non-SC visual context is anchored at the question's current frame and samples 5 past frames over about 2.0s by default.",
                "TRJ-5/6 future ground truth is resampled from the corresponding anchor-frame ego_status future states across the full future horizon.",
                "This full set is intended for running full eval first, then selecting a miniset from full-set results.",
            ],
            "filtered_counts": dict(sorted(filtered_counts.items())),
        },
        "questions": selected,
    }
    summary = base.build_summary(out, source_total=len(questions), review_count=len(sc_reviews))
    summary["metadata"]["per_question_type_available_before_limits"] = {
        qid: len(rows) for qid, rows in sorted(rows_by_id.items())
    }

    base.write_json(args.output_json, out)
    base.write_json(args.summary_json, summary)
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "summary_json": str(args.summary_json),
                "total_questions": len(selected),
                "question_type_count": len(summary["per_question_type_counts"]),
                "per_question_type_counts": summary["per_question_type_counts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
