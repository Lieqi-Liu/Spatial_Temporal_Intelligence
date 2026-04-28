"""Compatibility wrapper for older command path.

Prefer running ``python -m answers.commands.summarize_generation_inputs``.
"""

from .commands.summarize_generation_inputs import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
