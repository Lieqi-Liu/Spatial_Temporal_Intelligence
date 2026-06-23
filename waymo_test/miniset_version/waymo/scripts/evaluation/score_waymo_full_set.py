#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
WAYMO_TEST_DIR = SCRIPT_DIR.parent
MINISET_DIR = WAYMO_TEST_DIR / "miniset_version"
sys.path.insert(0, str(MINISET_DIR))

import score_waymo_miniset_mcq as scorer  # noqa: E402


scorer.DEFAULT_BENCHMARK_JSON = SCRIPT_DIR / "waymo_validation_full_set_questions.json"
scorer.DEFAULT_RESPONSES_JSON = SCRIPT_DIR / "waymo_validation_full_set_vlm_responses.json"
scorer.DEFAULT_OUTPUT_JSON = SCRIPT_DIR / "waymo_validation_full_set_metrics.json"
scorer.DEFAULT_OUTPUT_CSV = SCRIPT_DIR / "waymo_validation_full_set_predictions.csv"
scorer.DEFAULT_OUTPUT_MD = SCRIPT_DIR / "waymo_validation_full_set_report.md"


if __name__ == "__main__":
    scorer.main()
