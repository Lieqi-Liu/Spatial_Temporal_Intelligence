#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
WAYMO_TEST_DIR = SCRIPT_DIR.parent
MINISET_DIR = WAYMO_TEST_DIR / "miniset_version"
sys.path.insert(0, str(MINISET_DIR))

import run_waymo_miniset_vllm as runner  # noqa: E402


runner.DEFAULT_BENCHMARK_JSON = SCRIPT_DIR / "waymo_validation_full_set_questions.json"
runner.DEFAULT_OUTPUT_JSON = SCRIPT_DIR / "waymo_validation_full_set_vlm_responses.json"


if __name__ == "__main__":
    runner.main()
