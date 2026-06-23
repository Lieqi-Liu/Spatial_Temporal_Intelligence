#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BENCHMARK_JSON = SCRIPT_DIR / "waymo_validation_miniset_100_per_question_type_with_gt.json"
DEFAULT_RESPONSES_JSON = SCRIPT_DIR / "waymo_validation_miniset_vlm_responses.json"
DEFAULT_OUTPUT_JSON = SCRIPT_DIR / "waymo_validation_miniset_mcq_metrics.json"
DEFAULT_OUTPUT_CSV = SCRIPT_DIR / "waymo_validation_miniset_mcq_predictions.csv"
DEFAULT_OUTPUT_MD = SCRIPT_DIR / "waymo_validation_miniset_mcq_report.md"
DEFAULT_HF_HOME = Path("/local1/rgao727/huggingface")
DEFAULT_BLEURT_MODEL = "Elron/bleurt-base-512"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score MCQ accuracy on the Waymo validation miniset benchmark.")
    parser.add_argument("--benchmark-json", type=Path, default=DEFAULT_BENCHMARK_JSON)
    parser.add_argument("--responses-json", type=Path, default=DEFAULT_RESPONSES_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--enable-bleurt", action="store_true", help="Optionally score text FRQ answers with BLEURT.")
    parser.add_argument("--bleurt-model", type=str, default=DEFAULT_BLEURT_MODEL)
    parser.add_argument("--bleurt-batch-size", type=int, default=16)
    parser.add_argument("--bleurt-device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--hf-home", type=Path, default=DEFAULT_HF_HOME)
    parser.add_argument(
        "--bleurt-local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use only locally cached BLEURT model files unless set to --no-bleurt-local-files-only.",
    )
    parser.add_argument(
        "--assume-order",
        action="store_true",
        help="If response rows do not contain question_id, align MCQ benchmark rows to response rows by order.",
    )
    parser.add_argument(
        "--score-source",
        choices=["auto", "benchmark", "responses"],
        default="responses",
        help=(
            "Choose question metadata from the benchmark or from the response file. "
            "Default uses the response file so summaries match the actually generated responses."
        ),
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


def extract_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ["results", "responses", "questions", "predictions"]:
        value = payload.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]
    return []


def get_response_text(row: dict[str, Any]) -> str:
    for key in [
        "model_response",
        "model_response_text",
        "response",
        "raw_response",
        "text",
        "output",
        "answer",
    ]:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    parsed = row.get("parsed_response")
    if isinstance(parsed, dict):
        for key in ["answer", "choice", "predicted_option", "prediction"]:
            value = parsed.get(key)
            if value not in (None, ""):
                return str(value)
    return ""


def explicit_predicted_option(row: dict[str, Any], choices: dict[str, str]) -> str | None:
    valid = {str(k).strip().upper() for k in choices}
    for key in ["predicted_option", "predicted_key", "prediction", "answer_key", "model_answer"]:
        value = row.get(key)
        if value is None:
            continue
        candidate = str(value).strip().upper()
        if candidate in valid:
            return candidate
    parsed = row.get("parsed_response")
    if isinstance(parsed, dict):
        return explicit_predicted_option(parsed, choices)
    return None


def extract_answer_key(text: str, choices: dict[str, str]) -> str | None:
    valid = [str(k).strip().upper() for k in choices]
    if not text:
        return None
    raw = str(text).strip()
    upper = raw.upper()

    if upper in valid:
        return upper

    patterns = [
        r"(?:ANSWER|OPTION|CHOICE|SELECTED|PREDICTION)\s*(?:IS|:)?\s*[\(\[]?([A-Z])[\)\].:]?",
        r"^\s*[\(\[]?([A-Z])[\)\].:]\s",
        r"^\s*([A-Z])\s*$",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, upper):
            key = match.group(1)
            if key in valid:
                return key

    # Match option text. Prefer longer texts to avoid matching generic fragments.
    normalized = re.sub(r"\s+", " ", raw.lower())
    for key, value in sorted(choices.items(), key=lambda kv: len(str(kv[1])), reverse=True):
        choice_text = re.sub(r"\s+", " ", str(value).strip().lower())
        if choice_text and choice_text in normalized:
            return str(key).strip().upper()

    # Last-resort single-letter mention, only if unambiguous.
    mentioned = []
    for key in valid:
        if re.search(rf"(?<![A-Z0-9]){re.escape(key)}(?![A-Z0-9])", upper):
            mentioned.append(key)
    if len(set(mentioned)) == 1:
        return mentioned[0]
    return None


def random_baseline(choices: dict[str, str]) -> float:
    return 1.0 / len(choices) if choices else 0.0


def normalize_mcq_ground_truth(value: Any, choices: dict[str, str], ground_truth_text: Any = "") -> str:
    valid = {str(k).strip().upper(): str(v).strip() for k, v in choices.items()}
    raw = str(value or "").strip()
    upper = raw.upper()
    if upper in valid:
        return upper

    normalized_values = [raw, str(ground_truth_text or "").strip()]
    for candidate in normalized_values:
        candidate_norm = re.sub(r"\s+", " ", str(candidate).strip().lower())
        for key, choice_text in valid.items():
            choice_norm = re.sub(r"\s+", " ", str(choice_text).strip().lower())
            if choice_norm and candidate_norm == choice_norm:
                return key
    return upper


def extract_first_float(text: str) -> float | None:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(text or ""))
    if not match:
        return None
    try:
        return float(match.group(0))
    except Exception:
        return None


def parse_point_sequence(text: str) -> list[list[float]] | None:
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
            ok = True
            for item in payload:
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    ok = False
                    break
                try:
                    points.append([float(item[0]), float(item[1])])
                except Exception:
                    ok = False
                    break
            if ok:
                return points
    return None


def euclidean(p1: list[float], p2: list[float]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def numeric_distance_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    parsed = 0
    abs_errors: list[float] = []
    sq_errors: list[float] = []
    within_05 = 0
    within_1 = 0
    within_2 = 0
    for row in rows:
        pred = extract_first_float(row.get("model_response", ""))
        gt = extract_first_float(row.get("ground_truth", ""))
        if pred is None or gt is None:
            continue
        parsed += 1
        err = abs(pred - gt)
        abs_errors.append(err)
        sq_errors.append(err * err)
        within_05 += int(err <= 0.5)
        within_1 += int(err <= 1.0)
        within_2 += int(err <= 2.0)
    return {
        "count": len(rows),
        "parse_count": parsed,
        "parse_rate": parsed / len(rows) if rows else None,
        "mae_m": sum(abs_errors) / parsed if parsed else None,
        "rmse_m": math.sqrt(sum(sq_errors) / parsed) if parsed else None,
        "median_abs_error_m": sorted(abs_errors)[parsed // 2] if parsed else None,
        "within_0_5m_rate": within_05 / parsed if parsed else None,
        "within_1_0m_rate": within_1 / parsed if parsed else None,
        "within_2_0m_rate": within_2 / parsed if parsed else None,
    }


def trajectory_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    parsed = 0
    exact_len = 0
    ades: list[float] = []
    fdes: list[float] = []
    for row in rows:
        pred = parse_point_sequence(row.get("model_response", ""))
        gt = parse_point_sequence(row.get("ground_truth", ""))
        if pred is None or gt is None or not gt:
            continue
        parsed += 1
        exact_len += int(len(pred) == len(gt))
        compare_len = min(len(pred), len(gt))
        if compare_len <= 0:
            continue
        errors = [euclidean(pred[i], gt[i]) for i in range(compare_len)]
        ades.append(sum(errors) / compare_len)
        fdes.append(errors[-1])
    return {
        "count": len(rows),
        "parse_count": parsed,
        "parse_rate": parsed / len(rows) if rows else None,
        "exact_length_match_rate": exact_len / parsed if parsed else None,
        "mean_ade_m": sum(ades) / len(ades) if ades else None,
        "mean_fde_m": sum(fdes) / len(fdes) if fdes else None,
        "min_ade_m": min(ades) if ades else None,
        "min_fde_m": min(fdes) if fdes else None,
        "median_ade_m": sorted(ades)[len(ades) // 2] if ades else None,
        "median_fde_m": sorted(fdes)[len(fdes) // 2] if fdes else None,
    }


def text_frq_response_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    nonempty = 0
    errors = 0
    for row in rows:
        text = str(row.get("model_response", "")).strip()
        if text:
            nonempty += 1
        if text.startswith("[ERROR]"):
            errors += 1
    return {
        "count": len(rows),
        "nonempty_count": nonempty,
        "nonempty_rate": nonempty / len(rows) if rows else None,
        "empty_count": len(rows) - nonempty,
        "error_count": errors,
    }


def model_cache_dir_name(model: str) -> str:
    return f"models--{model.replace('/', '--')}"


def resolve_hf_cache_root(model: str, hf_home: Path, local_files_only: bool) -> Path:
    hf_home = hf_home.expanduser()
    model_dir = model_cache_dir_name(model)
    candidates = [
        hf_home / "hub",
        hf_home,
        Path.home() / ".cache" / "huggingface" / "hub",
        Path.home() / ".cache" / "huggingface",
    ]
    for candidate in candidates:
        if (candidate / model_dir).exists():
            return candidate
    if local_files_only:
        raise FileNotFoundError(
            f"Local BLEURT model cache not found for {model}. Checked: {', '.join(str(x) for x in candidates)}"
        )
    return hf_home / "hub"


def bleurt_metrics(
    rows: list[dict[str, Any]],
    *,
    model_name: str,
    hf_home: Path,
    local_files_only: bool,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    candidates = [
        row for row in rows
        if str(row.get("ground_truth", "")).strip()
        and str(row.get("model_response", "")).strip()
        and not str(row.get("model_response", "")).startswith("[ERROR]")
    ]
    if not candidates:
        return {"used": False, "reason": "no_text_frq_rows", "count": 0}

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    cache_root = resolve_hf_cache_root(model_name, hf_home, local_files_only)
    resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if resolved_device == "auto":
        resolved_device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=str(cache_root),
        local_files_only=local_files_only,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        cache_dir=str(cache_root),
        local_files_only=local_files_only,
    )
    model.to(resolved_device)
    model.eval()

    scores: list[float] = []
    refs = [str(row.get("ground_truth", "")).strip() for row in candidates]
    hyps = [str(row.get("model_response", "")).strip() for row in candidates]
    for start in range(0, len(candidates), batch_size):
        encoded = tokenizer(
            refs[start : start + batch_size],
            hyps[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {key: value.to(resolved_device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits.squeeze(-1)
        if logits.ndim == 0:
            scores.append(float(logits.detach().cpu().item()))
        else:
            scores.extend(float(x) for x in logits.detach().cpu().tolist())

    by_id: dict[str, list[float]] = defaultdict(list)
    for row, score in zip(candidates, scores):
        by_id[str(row.get("id", ""))].append(score)

    return {
        "used": True,
        "model": model_name,
        "cache_root": str(cache_root),
        "device": resolved_device,
        "count": len(candidates),
        "overall_mean": sum(scores) / len(scores) if scores else None,
        "by_question_type": {
            key: {
                "count": len(values),
                "mean": sum(values) / len(values) if values else None,
                "min": min(values) if values else None,
                "max": max(values) if values else None,
            }
            for key, values in sorted(by_id.items())
        },
    }


def summarize(rows: list[dict[str, Any]], group_key: str) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get(group_key, ""))].append(row)

    out: dict[str, Any] = {}
    for key, items in sorted(buckets.items()):
        total = len(items)
        correct = sum(1 for x in items if x["is_correct"])
        invalid = sum(1 for x in items if not x.get("predicted_option"))
        baseline = sum(float(x.get("random_baseline", 0.0)) for x in items) / total if total else None
        out[key] = {
            "count": total,
            "correct": correct,
            "accuracy": correct / total if total else None,
            "invalid_count": invalid,
            "invalid_rate": invalid / total if total else None,
            "random_guess_baseline": baseline,
            "ground_truth_distribution": dict(sorted(Counter(str(x.get("ground_truth")) for x in items).items())),
            "prediction_distribution": dict(sorted(Counter(str(x.get("predicted_option") or "__invalid__") for x in items).items())),
        }
    return out


def pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.2f}%"


def response_rows_have_question_metadata(rows: list[dict[str, Any]]) -> bool:
    return any(
        str(row.get("question_format", "")).upper() == "MCQ"
        and isinstance(row.get("choices"), dict)
        and row.get("choices")
        and row.get("ground_truth") not in (None, "")
        for row in rows
    )


def select_question_rows(
    *,
    args: argparse.Namespace,
    benchmark_questions: list[dict[str, Any]],
    response_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str, int, int]:
    benchmark_qids = {
        str(q.get("question_id"))
        for q in benchmark_questions
        if q.get("question_id") not in (None, "")
    }
    response_qids = {
        str(r.get("question_id"))
        for r in response_rows
        if r.get("question_id") not in (None, "")
    }
    matched_qids = len(benchmark_qids & response_qids)

    if args.score_source == "benchmark":
        return benchmark_questions, "benchmark", matched_qids, len(benchmark_qids)
    if args.score_source == "responses":
        return response_rows, "responses", matched_qids, len(benchmark_qids)

    if (
        response_rows_have_question_metadata(response_rows)
        and benchmark_qids
        and matched_qids < len(benchmark_qids)
    ):
        return response_rows, "responses", matched_qids, len(benchmark_qids)
    return benchmark_questions, "benchmark", matched_qids, len(benchmark_qids)


def write_markdown_report(path: Path, summary: dict[str, Any]) -> None:
    overall = summary["overall"]
    lines = [
        "# Waymo Miniset Report",
        "",
        f"- Benchmark: `{summary['metadata']['benchmark_json']}`",
        f"- Responses: `{summary['metadata']['responses_json']}`",
        f"- Score source: `{summary['metadata']['score_source']}`",
        f"- MCQ count: `{overall['mcq_count']}`",
        f"- Correct: `{overall['correct']}`",
        f"- Accuracy: **{pct(overall['accuracy'])}**",
        f"- Random baseline: `{pct(overall['random_guess_baseline'])}`",
        f"- Invalid/unparsed predictions: `{overall['invalid_count']}` ({pct(overall['invalid_rate'])})",
        f"- Missing responses: `{overall['missing_response_count']}`",
        "",
        "## Per Question Type",
        "",
        "| Question ID | Count | Accuracy | Random Baseline | Invalid |",
        "|---|---:|---:|---:|---:|",
    ]
    for qid, row in summary["per_question_type"].items():
        lines.append(
            f"| `{qid}` | {int(row['count'])} | {pct(row['accuracy'])} | "
            f"{pct(row['random_guess_baseline'])} | {int(row['invalid_count'])} |"
        )

    lines.extend([
        "",
        "## Per Task",
        "",
        "| Task | Count | Accuracy | Random Baseline | Invalid |",
        "|---|---:|---:|---:|---:|",
    ])
    for task, row in summary["per_task"].items():
        lines.append(
            f"| `{task}` | {int(row['count'])} | {pct(row['accuracy'])} | "
            f"{pct(row['random_guess_baseline'])} | {int(row['invalid_count'])} |"
        )

    special = summary.get("special_metrics", {})
    lines.extend(["", "## Distance And Trajectory Metrics", ""])
    if special.get("SP-3a"):
        m = special["SP-3a"]
        lines.extend([
            "### SP-3a Distance FRQ",
            "",
            f"- Count: `{m['count']}`",
            f"- Parse rate: `{pct(m['parse_rate'])}`",
            f"- MAE: `{m['mae_m']:.3f} m`" if m.get("mae_m") is not None else "- MAE: `n/a`",
            f"- RMSE: `{m['rmse_m']:.3f} m`" if m.get("rmse_m") is not None else "- RMSE: `n/a`",
            f"- Median AE: `{m['median_abs_error_m']:.3f} m`" if m.get("median_abs_error_m") is not None else "- Median AE: `n/a`",
            f"- Within 1m: `{pct(m['within_1_0m_rate'])}`",
            "",
        ])
    for key in ["TRJ-5", "TRJ-6"]:
        if special.get(key):
            m = special[key]
            lines.extend([
                f"### {key} Trajectory FRQ",
                "",
                f"- Count: `{m['count']}`",
                f"- Parse rate: `{pct(m['parse_rate'])}`",
                f"- Exact length match: `{pct(m['exact_length_match_rate'])}`",
                f"- Mean ADE: `{m['mean_ade_m']:.3f} m`" if m.get("mean_ade_m") is not None else "- Mean ADE: `n/a`",
                f"- Mean FDE: `{m['mean_fde_m']:.3f} m`" if m.get("mean_fde_m") is not None else "- Mean FDE: `n/a`",
                f"- Min FDE: `{m['min_fde_m']:.3f} m`" if m.get("min_fde_m") is not None else "- Min FDE: `n/a`",
                "",
            ])
    if special.get("TRJ-7"):
        m = special["TRJ-7"]
        lines.extend([
            "### TRJ-7 Text FRQ",
            "",
            f"- Count: `{m['count']}`",
            f"- Non-empty responses: `{m['nonempty_count']}` ({pct(m['nonempty_rate'])})",
            f"- Empty responses: `{m['empty_count']}`",
            f"- Error responses: `{m['error_count']}`",
            "- Automatic text accuracy: `not scored`",
            "",
        ])

    bleurt = summary.get("bleurt")
    if bleurt:
        lines.extend(["", "## BLEURT Text FRQ", ""])
        if not bleurt.get("used"):
            lines.append(f"- Not used: `{bleurt.get('reason', 'disabled')}`")
        else:
            lines.extend([
                f"- Model: `{bleurt.get('model')}`",
                f"- Count: `{bleurt.get('count')}`",
                f"- Overall mean: `{bleurt.get('overall_mean'):.4f}`" if bleurt.get("overall_mean") is not None else "- Overall mean: `n/a`",
                "",
                "| Question ID | Count | BLEURT Mean | Min | Max |",
                "|---|---:|---:|---:|---:|",
            ])
            for qid, row in bleurt.get("by_question_type", {}).items():
                lines.append(
                    f"| `{qid}` | {row.get('count')} | "
                    f"{row.get('mean'):.4f} | {row.get('min'):.4f} | {row.get('max'):.4f} |"
                )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    benchmark = read_json(args.benchmark_json)
    response_payload = read_json(args.responses_json)

    benchmark_questions = benchmark.get("questions", []) if isinstance(benchmark, dict) else []
    mcq_questions = [
        q for q in benchmark_questions
        if str(q.get("question_format", "")).upper() == "MCQ"
        and isinstance(q.get("choices"), dict)
        and q.get("choices")
    ]

    response_rows = extract_rows(response_payload)
    questions, score_source, matched_qids, benchmark_qid_count = select_question_rows(
        args=args,
        benchmark_questions=list(benchmark_questions),
        response_rows=response_rows,
    )
    mcq_questions = [
        q for q in questions
        if str(q.get("question_format", "")).upper() == "MCQ"
        and isinstance(q.get("choices"), dict)
        and q.get("choices")
    ]
    responses_by_qid = {
        str(r.get("question_id")): r
        for r in response_rows
        if r.get("question_id") not in (None, "")
    }

    def response_for(question: dict[str, Any], fallback_index: int) -> dict[str, Any] | None:
        qid = str(question.get("question_id", ""))
        response = responses_by_qid.get(qid)
        if response is None and args.assume_order and fallback_index < len(response_rows):
            response = response_rows[fallback_index]
        return response

    prediction_rows: list[dict[str, Any]] = []
    missing = 0
    for idx, question in enumerate(mcq_questions):
        qid = str(question.get("question_id", ""))
        response = response_for(question, idx)
        if response is None:
            missing += 1
            raw_text = ""
            pred = None
        else:
            raw_text = get_response_text(response)
            pred = explicit_predicted_option(response, question["choices"])
            if pred is None:
                pred = extract_answer_key(raw_text, question["choices"])

        gt = normalize_mcq_ground_truth(
            question.get("ground_truth", ""),
            question["choices"],
            question.get("ground_truth_text"),
        )
        is_correct = bool(pred and pred == gt)
        prediction_rows.append(
            {
                "question_id": qid,
                "id": question.get("id"),
                "task": question.get("task"),
                "question_format": question.get("question_format"),
                "scene_id": question.get("scene_id"),
                "bundle_id": question.get("bundle_id"),
                "question": question.get("question"),
                "ground_truth": gt,
                "ground_truth_text": question.get("ground_truth_text"),
                "predicted_option": pred,
                "predicted_text": question["choices"].get(pred) if pred else None,
                "is_correct": is_correct,
                "random_baseline": random_baseline(question["choices"]),
                "model_response": raw_text,
            }
        )

    special_rows: list[dict[str, Any]] = []
    text_frq_rows: list[dict[str, Any]] = []
    all_questions = list(questions)
    for idx, question in enumerate(all_questions):
        task_id = str(question.get("id", ""))
        qf = str(question.get("question_format", "")).upper()
        if qf != "FRQ":
            continue
        response = response_for(question, idx)
        raw_text = get_response_text(response) if response is not None else ""
        row = {
            "question_id": question.get("question_id"),
            "id": task_id,
            "task": question.get("task"),
            "question_format": question.get("question_format"),
            "scene_id": question.get("scene_id"),
            "bundle_id": question.get("bundle_id"),
            "question": question.get("question"),
            "ground_truth": question.get("ground_truth"),
            "ground_truth_text": question.get("ground_truth_text"),
            "model_response": raw_text,
            "missing_response": response is None,
        }
        if task_id in {"SP-3a", "TRJ-5", "TRJ-6", "TRJ-7"}:
            special_rows.append(row)
        else:
            text_frq_rows.append(row)

    special_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in special_rows:
        special_by_id[str(row.get("id", ""))].append(row)
    special_metrics = {
        "SP-3a": numeric_distance_metrics(special_by_id.get("SP-3a", [])),
        "TRJ-5": trajectory_metrics(special_by_id.get("TRJ-5", [])),
        "TRJ-6": trajectory_metrics(special_by_id.get("TRJ-6", [])),
        "TRJ-7": text_frq_response_metrics(special_by_id.get("TRJ-7", [])),
        "text_frq_note": "Text FRQ metrics are intentionally skipped for now.",
    }
    if args.enable_bleurt:
        try:
            bleurt_summary = bleurt_metrics(
                text_frq_rows,
                model_name=args.bleurt_model,
                hf_home=args.hf_home,
                local_files_only=args.bleurt_local_files_only,
                batch_size=args.bleurt_batch_size,
                device=args.bleurt_device,
            )
        except Exception as exc:
            bleurt_summary = {
                "used": False,
                "reason": f"{type(exc).__name__}: {exc}",
                "count": len(text_frq_rows),
            }
    else:
        bleurt_summary = {
            "used": False,
            "reason": "disabled; pass --enable-bleurt to score text FRQ",
            "count": len(text_frq_rows),
        }

    total = len(prediction_rows)
    correct = sum(1 for row in prediction_rows if row["is_correct"])
    invalid = sum(1 for row in prediction_rows if not row.get("predicted_option"))
    summary = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "benchmark_json": str(args.benchmark_json),
            "responses_json": str(args.responses_json),
            "assume_order": args.assume_order,
            "requested_score_source": args.score_source,
            "score_source": score_source,
            "matched_question_ids": matched_qids,
            "benchmark_question_id_count": benchmark_qid_count,
        },
        "overall": {
            "mcq_count": total,
            "correct": correct,
            "accuracy": correct / total if total else None,
            "invalid_count": invalid,
            "invalid_rate": invalid / total if total else None,
            "missing_response_count": missing,
            "random_guess_baseline": (
                sum(row["random_baseline"] for row in prediction_rows) / total if total else None
            ),
        },
        "per_question_type": summarize(prediction_rows, "id"),
        "per_task": summarize(prediction_rows, "task"),
        "special_metrics": special_metrics,
        "bleurt": bleurt_summary,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output_json, summary)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "question_id",
            "id",
            "task",
            "scene_id",
            "bundle_id",
            "ground_truth",
            "ground_truth_text",
            "predicted_option",
            "predicted_text",
            "is_correct",
            "model_response",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in prediction_rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    write_markdown_report(args.output_md, summary)

    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "output_csv": str(args.output_csv),
                "output_md": str(args.output_md),
                "mcq_count": total,
                "accuracy": summary["overall"]["accuracy"],
                "invalid_rate": summary["overall"]["invalid_rate"],
                "missing_response_count": missing,
                "special_metrics": {
                    key: value
                    for key, value in special_metrics.items()
                    if key in {"SP-3a", "TRJ-5", "TRJ-6", "TRJ-7"}
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
