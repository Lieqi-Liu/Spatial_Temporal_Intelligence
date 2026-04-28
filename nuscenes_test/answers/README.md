# answers

This folder is the new modular answer-generation flow.

The existing `generate_answers.py` remains untouched and continues to be the
stable legacy entrypoint.

Directory layout:

- `common/`: shared config and JSON utilities
- `question_types/`: logic grouped by question family
- `workflows/`: orchestration and pipeline assembly
- `commands/`: runnable entrypoints

Current question-type modules:

- `question_types/scene_level_tasks.py`
- `question_types/object_level_tasks.py`
- `question_types/trajectory_tasks.py`

Current command:

```bash
python -m answers.commands.summarize_generation_inputs \
  --root ../data/nuscenes-v1.0-mini \
  --formatted-scenes-dir ./formatted_scenes \
  --questions-json ./questions.json
```

Backward-compatible wrappers remain at the top level:

- `answers.cli`
- `answers.config`
- `answers.io`
- `answers.pipeline`
- `answers.scene_level`
