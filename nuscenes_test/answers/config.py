"""Compatibility wrapper for older imports.

Prefer importing from ``answers.common.generation_config``.
"""

from .common.generation_config import AnswerGenerationConfig

__all__ = ["AnswerGenerationConfig"]
