#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_BASE_JSON = Path("/home/rgao727/Spatial_Temporal_Intelligence/waymo_test/waymo_e2e_questions_5frame_stride5_gt.json")
DEFAULT_RESPONSES_CSV = Path("/home/rgao727/Spatial_Temporal_Intelligence/waymo_test/trj_human_score_explanations_full.csv")
DEFAULT_OUTPUT_JSON = Path("/home/rgao727/Spatial_Temporal_Intelligence/waymo_test/waymo_e2e_questions_5frame_stride5_gt_with_trj9.json")

TRJ9_QUESTION = (
    "Given the past 5 frames and the three candidate future trajectories A, B, and C, "
    "what kind of driving behavior does each candidate represent, which trajectory is the best option, "
    "and why is it the best driving choice in this scene?"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge generated TRJ-9 free-response explanations back into the Waymo 5-frame GT JSON."
        )
    )
    parser.add_argument(
        "--base-json",
        type=Path,
        default=DEFAULT_BASE_JSON,
        help=f"Base GT JSON without TRJ-9 (default: {DEFAULT_BASE_JSON}).",
    )
    parser.add_argument(
        "--responses",
        type=Path,
        default=DEFAULT_RESPONSES_CSV,
        help=(
            "Model output file containing question_id -> explanation mapping. "
            "Supports .csv, .json, or .jsonl."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help=f"Output merged GT JSON (default: {DEFAULT_OUTPUT_JSON}).",
    )
    parser.add_argument(
        "--question-text",
        type=str,
        default=TRJ9_QUESTION,
        help="Benchmark question text used for TRJ-9 items.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Write indented JSON.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any], pretty: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        if pretty:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        else:
            json.dump(payload, f, ensure_ascii=False)


def load_responses(path: Path) -> dict[str, str]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return load_responses_csv(path)
    if suffix == ".json":
        return load_responses_json(path)
    if suffix == ".jsonl":
        return load_responses_jsonl(path)
    raise ValueError(f"Unsupported responses format: {path}")


def load_responses_csv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = str(row.get("question_id", "")).strip()
            text = str(row.get("model_output", "")).strip()
            if qid and text:
                out[qid] = text
    return out


def load_responses_json(path: Path) -> dict[str, str]:
    payload = load_json(path)
    rows: list[dict[str, Any]]
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("results") or payload.get("rows") or payload.get("data") or []
    else:
        rows = []
    out: dict[str, str] = {}
    for row in rows:
        qid = str(row.get("question_id", "")).strip()
        text = str(row.get("model_output", "") or row.get("answer", "") or row.get("output", "")).strip()
        if qid and text:
            out[qid] = text
    return out


def load_responses_jsonl(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = str(row.get("question_id", "")).strip()
            text = str(row.get("model_output", "") or row.get("answer", "") or row.get("output", "")).strip()
            if qid and text:
                out[qid] = text
    return out


def make_trj9_from_trj8(trj8_item: dict[str, Any], answer_text: str, question_text: str) -> dict[str, Any]:
    frame_name = str(trj8_item.get("frame_name", ""))
    question_id = str(trj8_item.get("question_id", ""))
    if question_id.endswith("_TRJ-8"):
        trj9_question_id = question_id[:-6] + "_TRJ-9"
    else:
        trj9_question_id = question_id + "_TRJ-9"

    out = {
        "question_id": trj9_question_id,
        "id": "TRJ-9",
        "task": "trajectory-prediction",
        "question_format": "FRQ",
        "frame_name": frame_name,
        "scene_id": trj8_item.get("scene_id"),
        "frame_index": trj8_item.get("frame_index"),
        "image_path": trj8_item.get("image_path"),
        "scenario_cluster": trj8_item.get("scenario_cluster"),
        "question": question_text,
        "ground_truth": answer_text,
        "ground_truth_text": answer_text,
        "model_response": "",
        "bundle_id": trj8_item.get("bundle_id"),
        "bundle_size": trj8_item.get("bundle_size"),
        "bundle_frame_names": trj8_item.get("bundle_frame_names"),
        "history_image_paths": trj8_item.get("history_image_paths"),
        "anchor_image_path": trj8_item.get("anchor_image_path"),
        "observed_trajectory": trj8_item.get("observed_trajectory"),
        "candidate_trajectories": trj8_item.get("candidate_trajectories"),
        "selected_candidate": trj8_item.get("ground_truth"),
        "hidden_metadata": {
            "source_task_id": "TRJ-8",
            "preference_scores": (trj8_item.get("hidden_metadata") or {}).get("preference_scores"),
            "score_rank_desc": (trj8_item.get("hidden_metadata") or {}).get("score_rank_desc"),
        },
    }
    return out


def main() -> None:
    args = parse_args()
    if not args.base_json.exists():
        raise FileNotFoundError(f"Base JSON not found: {args.base_json}")
    if not args.responses.exists():
        raise FileNotFoundError(f"Responses file not found: {args.responses}")

    payload = load_json(args.base_json)
    questions = list(payload.get("questions", []))
    responses = load_responses(args.responses)

    existing_qids = {str(q.get("question_id", "")) for q in questions}
    trj8_items = [q for q in questions if (q.get("id") or q.get("task_id")) == "TRJ-8"]

    added = 0
    missing = 0
    trj9_items: list[dict[str, Any]] = []
    for trj8 in trj8_items:
        source_qid = str(trj8.get("question_id", "")).strip()
        answer_text = responses.get(source_qid)
        if not answer_text:
            missing += 1
            continue
        trj9 = make_trj9_from_trj8(trj8, answer_text=answer_text, question_text=args.question_text)
        if trj9["question_id"] in existing_qids:
            continue
        trj9_items.append(trj9)
        existing_qids.add(trj9["question_id"])
        added += 1

    payload["questions"] = questions + trj9_items
    metadata = payload.setdefault("metadata", {})
    counts = metadata.setdefault("counts", {})
    counts["trj9_added"] = added
    counts["trj8_without_trj9_answer"] = missing
    counts["questions_generated"] = len(payload["questions"])
    metadata["trj9_question_text"] = args.question_text
    metadata["trj9_responses_source"] = str(args.responses)

    write_json(args.output_json, payload, args.pretty)
    print(
        json.dumps(
            {
                "base_json": str(args.base_json),
                "responses": str(args.responses),
                "output_json": str(args.output_json),
                "trj8_count": len(trj8_items),
                "trj9_added": added,
                "trj8_without_trj9_answer": missing,
                "total_questions": len(payload["questions"]),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
