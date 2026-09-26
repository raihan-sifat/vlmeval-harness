"""Task registry.

Tasks and models are both looked up by name so that configs, the CLI and the
report layer never import a concrete class. Adding a task is one import plus one
decorator; nothing else in the codebase needs to know it exists.
"""

from .base import ItemOutcome, Query, Task
from .mcq import MCQTask
from .pope import POPETask

TASKS: dict[str, type[Task]] = {
    MCQTask.name: MCQTask,
    POPETask.name: POPETask,
}

#: Names that resolve to an existing task, for config readability.
TASK_ALIASES: dict[str, str] = {
    "letter_answer": MCQTask.name,
    "multiple_choice": MCQTask.name,
    "vqa_mcq": MCQTask.name,
    "pope": POPETask.name,
    "hallucination": POPETask.name,
    "object_existence": POPETask.name,
}

__all__ = ["TASKS", "TASK_ALIASES", "Task", "Query", "ItemOutcome", "MCQTask", "POPETask", "get_task"]
