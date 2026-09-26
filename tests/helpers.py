"""Shared helpers and fake models for the test suite."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from vlmeval.models.base import ModelAdapter  # noqa: E402
from vlmeval.types import Example, ModelResponse  # noqa: E402


class RecordingModel(ModelAdapter):
    """Captures every call so tests can assert on what a backend would see.

    Deliberately implements only :meth:`generate`. Because the harness routes
    harness-internal context through ``generate_with_context``, an adapter that
    does not override that method must never observe a gold label -- and
    ``contexts`` below stays empty forever. That is the property
    ``test_runner_never_leaks_gold_to_adapters`` pins down.
    """

    def __init__(self, reply: str = "A", name: str = "recorder", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.name = name
        self.model = name
        self.reply = reply
        self.prompts: list[str] = []
        self.systems: list[str | None] = []
        self.contexts: list[dict[str, Any] | None] = []
        self.calls = 0

    def generate_with_context(
        self,
        *,
        image: Any,
        prompt: str,
        system: str | None = None,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        # Recorded, then delegated, so the test can prove the context arrived
        # here and still did not reach the payload the provider would see.
        self.contexts.append(context)
        return self.generate(image=image, prompt=prompt, system=system, **kwargs)

    def generate(
        self,
        *,
        image: Any,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 16,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> ModelResponse:
        self.calls += 1
        self.prompts.append(prompt)
        self.systems.append(system)
        return ModelResponse(text=self.reply, prompt_tokens=1, completion_tokens=1)


class FlakyModel(ModelAdapter):
    """Fails the first `failures` attempts with a retryable error."""

    def __init__(self, failures: int = 2, name: str = "flaky", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.name = name
        self.model = name
        self.remaining = failures
        self.calls = 0

    def generate(self, *, image: Any, prompt: str, **kwargs: Any) -> ModelResponse:
        self.calls += 1
        if self.remaining > 0:
            self.remaining -= 1
            return ModelResponse(text="", error="429 rate limit exceeded")
        return ModelResponse(text="A", prompt_tokens=1, completion_tokens=1)


def mcq_example(uid: str = "e0", gold: str = "A", n_choices: int = 4) -> Example:
    choices = ["red", "blue", "green", "yellow", "black", "white"][:n_choices]
    return Example(
        uid=uid,
        question="Which colour is the large shape?",
        choices=choices,
        label=gold,
        image="img.png",
    )


def pope_example(uid: str = "p0", answer: str = "yes", category: str = "random") -> Example:
    return Example(
        uid=uid,
        question="Is there a car in the image?",
        label=answer,
        category=category,
        image="img.png",
    )
