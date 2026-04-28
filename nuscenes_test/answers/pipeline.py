"""Compatibility wrapper for older imports.

Prefer importing from ``answers.workflows.generation_pipeline``.
"""

from .workflows.generation_pipeline import AnswerGenerationPipeline, PipelineState

__all__ = ["AnswerGenerationPipeline", "PipelineState"]
