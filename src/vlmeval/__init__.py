"""vlmeval: a reproducible evaluation harness for vision-language models.

The public surface is deliberately small:

>>> from vlmeval import build_model, build_task, load, Runner
>>> model = build_model("echo:demo")            # any backend, same interface
>>> task = build_task({"task": "mcq", "num_permutations": 2})
>>> examples = load("synthetic_mcq", limit=20)   # no network, no API key
>>> result = Runner(model, task).run(examples, dataset="synthetic_mcq")
>>> result.metrics["accuracy"].fmt(pct=True)
'70.0 [45.0, 90.0]'

Swap ``"echo:demo"`` for ``"openai:gpt-4o"`` and the rest of the pipeline is
unchanged: that is the property the whole design is built around.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .cache import ResponseCache
from .data import load
from .parsing import answers_match, extract_choice, extract_yes_no
from .prompts import PromptStyle
from .registry import available_models, available_tasks, build_model, build_task
from .report import leaderboard, write_leaderboard, write_run
from .runstate import Budget, RunConfig
from .runner import Runner, evaluate
from .stats import bootstrap_ci, mcnemar_test, paired_bootstrap_delta
from .tasks import MCQTask, POPETask
from .types import Example, MetricSummary, ModelResponse, RunResult

__all__ = [
    "__version__",
    "Example",
    "MetricSummary",
    "ModelResponse",
    "RunResult",
    "MCQTask",
    "POPETask",
    "Runner",
    "evaluate",
    "ResponseCache",
    "Budget",
    "RunConfig",
    "build_model",
    "build_task",
    "available_models",
    "available_tasks",
    "load",
    "write_run",
    "write_leaderboard",
    "leaderboard",
    "PromptStyle",
    "extract_choice",
    "extract_yes_no",
    "answers_match",
    "bootstrap_ci",
    "mcnemar_test",
    "paired_bootstrap_delta",
]
