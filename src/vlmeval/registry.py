"""Registries for models and tasks.

Configs and the CLI refer to everything by string, which keeps YAML free of
Python imports and lets a new backend ship without touching a single existing
call site.

Model specs accept two forms:

* ``"openai:gpt-4o"`` -- shorthand, split on the first colon.
* a mapping ``{backend: openai, model: gpt-4o, max_side: 1024}`` -- for
  anything that needs per-run options or a distinct report label.

The ``name`` defaults to the model id with separators replaced, so a spec never
has to supply one, and any spec that *does* supply one gets a directory and
table row named exactly as asked.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .models.base import ModelAdapter
from .models.echo import ECHO_PRESETS, EchoModel
from .tasks.base import Task
from .tasks.mcq import MCQTask
from .tasks.pope import POPETask

MODEL_BACKENDS: dict[str, type[ModelAdapter]] = {
    "openai": "vlmeval.models.openai_api:OpenAIModel",
    "anthropic": "vlmeval.models.anthropic_api:AnthropicModel",
    "google": "vlmeval.models.google_api:GoogleModel",
    "hf": "vlmeval.models.hf_local:HFLocalModel",
    "hf_local": "vlmeval.models.hf_local:HFLocalModel",
    "local": "vlmeval.models.hf_local:HFLocalModel",
    "echo": "vlmeval.models.echo:EchoModel",
    "dummy": "vlmeval.models.echo:EchoModel",
}

TASK_REGISTRY: dict[str, type[Task]] = {
    MCQTask.name: MCQTask,
    POPETask.name: POPETask,
}

TASK_ALIASES: dict[str, str] = {
    "letter_answer": MCQTask.name,
    "multiple_choice": MCQTask.name,
    "vqa_mcq": MCQTask.name,
    "pope": POPETask.name,
    "hallucination": POPETask.name,
    "object_existence": POPETask.name,
}


def _load_backend(path: str) -> type[ModelAdapter]:
    """Import a backend lazily by dotted path.

    Lazy because the cloud adapters import their SDK at construction time and a
    user evaluating only local models should not need the cloud SDKs present.
    """
    import importlib

    module_name, _, class_name = path.partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def sanitize(name: str) -> str:
    """Make a model id safe for use as a path component.

    Model ids and dataset names come from configuration files, and this function
    decides where the results they produce are written. Anything that could be
    read as path structure is removed rather than escaped: separators, and the
    ``..`` runs that survive naive substitution. A name that reduces to nothing
    useful becomes ``model``, so a path is never built from an empty string.
    """
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(name))
    while ".." in cleaned:
        cleaned = cleaned.replace("..", ".")
    cleaned = cleaned.strip("-. ")
    return cleaned or "model"


def build_model(spec: str | dict[str, Any]) -> ModelAdapter:
    """Instantiate a model adapter from a shorthand string or a mapping.

    Raises:
        ValueError: If the backend is unknown, listing the ones that exist.
    """
    if isinstance(spec, str):
        stripped = spec.strip()
        if stripped.startswith("{"):
            # A JSON object is a mapping written inline, which is the only way to
            # pass adapter options from a shell without a config file. Parsing it
            # here means `--model '{"backend": "echo", "skill": 0.5}'` behaves
            # exactly like the equivalent mapping.
            try:
                decoded = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Model spec looks like JSON but does not parse: {exc}"
                ) from exc
            # A string starting with "{" always decodes to an object, so there is
            # no container case to handle here.
            return build_model(decoded)
        backend, _, model_id = spec.partition(":")
        if not model_id:
            raise ValueError(
                f"Model shorthand must look like 'backend:model', got {spec!r}. "
                f"Backends: {sorted(set(MODEL_BACKENDS))}"
            )
        options: dict[str, Any] = {"model": model_id}
    elif isinstance(spec, dict):
        if "backend" not in spec:
            raise ValueError(f"Model mapping needs a 'backend' key: {spec}")
        if not spec.get("model"):
            raise ValueError(f"Model mapping needs a 'model' key: {spec}")
        options = {k: v for k, v in spec.items() if k != "backend"}
        backend = str(spec["backend"])
    else:
        raise TypeError(f"Model spec must be a string or mapping, got {type(spec).__name__}")

    backend = backend.lower()
    if backend not in MODEL_BACKENDS:
        raise ValueError(
            f"Unknown backend {backend!r}. Available: {sorted(set(MODEL_BACKENDS))}. "
            "Add an entry in vlmeval.registry.MODEL_BACKENDS to register a new one."
        )

    if backend in {"echo", "dummy"}:
        # A named echo preset sets the behaviour; anything given explicitly in
        # the spec still wins, so a preset can be used as a starting point.
        preset = ECHO_PRESETS.get(str(options.get("model", "")).lower())
        if preset:
            options = {**preset, **options}

    # Price overrides are handled here rather than in each adapter so they work
    # for every backend, including local models that have no list price at all.
    # Popped before construction because no adapter accepts these names.
    cost_in = options.pop("cost_in", None)
    cost_out = options.pop("cost_out", None)

    cls = _load_backend(MODEL_BACKENDS[backend])
    model = cls(**options)
    model.name = sanitize(model.name)

    if cost_in is not None or cost_out is not None:
        base_in, base_out = model.cost_per_million
        model.cost_per_million = (
            base_in if cost_in is None else float(cost_in),
            base_out if cost_out is None else float(cost_out),
        )
        # Recorded rather than silently applied: a cost figure in a results file
        # has to be traceable to the price that produced it.
        model.info["cost_override"] = {
            "cost_in_per_mtok": model.cost_per_million[0],
            "cost_out_per_mtok": model.cost_per_million[1],
            "list_in_per_mtok": base_in,
            "list_out_per_mtok": base_out,
        }
    return model


def build_task(spec: str | dict[str, Any], **overrides: Any) -> Task:
    """Instantiate a task from a name or a mapping of options.

    Keyword arguments override anything coming from `spec`, so all three of
    these are equivalent::

        build_task("mcq")
        build_task("mcq", num_permutations=3)
        build_task({"task": "mcq", "num_permutations": 3})

    Raises:
        ValueError: If the task name is unknown.
    """
    if isinstance(spec, str):
        name, options = spec, {}
    elif isinstance(spec, dict):
        options = dict(spec)
        name = options.pop("task", None) or options.pop("name", None)
        if not name:
            raise ValueError(f"Task mapping needs a 'task' or 'name' key: {spec}")
    else:
        raise TypeError(f"Task spec must be a string or mapping, got {type(spec).__name__}")

    options.update(overrides)

    key = TASK_ALIASES.get(str(name).lower(), str(name).lower())
    if key not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown task {name!r}. Available: {sorted(set(TASK_REGISTRY))}. "
            "Add an entry in vlmeval.registry.TASK_REGISTRY to register a new one."
        )
    return TASK_REGISTRY[key](**options)


def available_models() -> list[str]:
    """Backend names usable in a model spec."""
    return sorted(set(MODEL_BACKENDS))


def available_tasks() -> list[str]:
    """Task names usable in a task spec, aliases included."""
    return sorted(set(TASK_REGISTRY) | set(TASK_ALIASES))


__all__ = [
    "ECHO_PRESETS",
    "EchoModel",
    "MCQTask",
    "POPETask",
    "MODEL_BACKENDS",
    "TASK_REGISTRY",
    "TASK_ALIASES",
    "build_model",
    "build_task",
    "available_models",
    "available_tasks",
    "sanitize",
]
