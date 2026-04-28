"""Compatibility wrapper for older imports.

Prefer importing from ``answers.common.json_store``.
"""

from .common.json_store import dump_json, load_json

__all__ = ["dump_json", "load_json"]
