"""New modular answer-generation package.

The legacy ``generate_answers.py`` remains untouched. New code should prefer
the functional layout under ``common/``, ``question_types/``, ``workflows/``,
and ``commands/``.
"""

from .common.generation_config import AnswerGenerationConfig
from .workflows.generation_pipeline import AnswerGenerationPipeline

__all__ = ["AnswerGenerationConfig", "AnswerGenerationPipeline"]
