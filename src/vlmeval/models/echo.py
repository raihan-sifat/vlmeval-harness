"""A deterministic, dependency-free stand-in for a real model.

This exists so the whole harness can be exercised with no API key, no GPU and no
network: unit tests, CI, and the committed example runs all go through it.

Three properties make it useful rather than merely present:

* **Deterministic.** Answers derive from a hash of the item id, so a test can
  assert exact metrics and a demo run produces the same numbers on every
  machine. Seeding on the item rather than the prompt is what makes the model
  *consistently* right or wrong across option permutations, the way a real
  model is.
* **It reproduces the failure modes that make real evaluations hard** -- verbose
  preambles, unparseable replies, and a position prior -- so the parsing and
  bias-audit code paths are genuinely exercised rather than assumed.
* **It receives the gold label**, through
  :meth:`~vlmeval.models.base.ModelAdapter.generate_with_context`. That is how
  `skill` can mean something, and it is why this adapter overrides the context
  hook while every real backend does not.
"""

from __future__ import annotations

import hashlib
from typing import Any

from ..types import ModelResponse
from .base import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, ModelAdapter

#: Display letters the model can pick from, in prompt order.
_POSITIONS = "ABCDEFGH"

#: Named behaviours for the ``echo:<name>`` shorthand.
#:
#: Without this table every ``echo:*`` spec would build the *same* model under a
#: different name, so a benchmark of three echo variants would report three
#: identical scores and look like a broken harness. Each entry is a genuinely
#: different model, and the two ends double as CI floors and ceilings:
#:
#: ``oracle``   always right, never verbose, never refuses -> accuracy exactly 1.0
#: ``bad``      never right, never verbose, never refuses -> accuracy near 0
#: ``blind``    always answers the first option, ignoring the image, which is
#:              what a model exploiting a position prior looks like; accuracy
#:              sits near chance while consistency collapses
#: ``chatty``   wraps every reply in prose, to exercise the answer parsers
#: ``refuses``  always returns an unparseable refusal, to exercise unparsed
#:              accounting and the strict-vs-parsed accuracy gap
ECHO_PRESETS: dict[str, dict[str, float]] = {
    "demo": {"skill": 0.70, "chatter": 0.15, "refusal": 0.05},
    "oracle": {"skill": 1.00, "chatter": 0.0, "refusal": 0.0},
    "bad": {"skill": 0.00, "chatter": 0.0, "refusal": 0.0},
    "blind": {"skill": 0.00, "chatter": 0.0, "refusal": 0.0, "position_bias": 1.0},
    "chatty": {"skill": 0.70, "chatter": 1.0, "refusal": 0.0},
    "refuses": {"skill": 1.00, "chatter": 0.0, "refusal": 1.0},
    "strong": {"skill": 0.90, "chatter": 0.10, "refusal": 0.0},
    "weak": {"skill": 0.30, "chatter": 0.30, "refusal": 0.10},
    # For POPE: asserts every object is present, which is what an object
    # hallucinating model does. Accuracy and precision collapse on the "no"
    # items while recall stays at 1.0, so the split metrics have to show it.
    "hallucinator": {"skill": 0.0, "chatter": 0.0, "refusal": 0.0, "yes_bias": 1.0},
}


class EchoModel(ModelAdapter):
    """Answers from a seeded hash, with configurable degradations.

    Args:
        model: Identifier for reports and cache keys. This is the field a
            model spec supplies, so it is the primary name.
        name: Alias for `model`, kept for symmetry with the other adapters.
        skill: Probability in [0, 1] of returning the gold answer.
        position_bias: Probability of ignoring the image and answering with a
            fixed display letter instead. This is what a model exploiting a
            position prior looks like, and setting it is how the
            order-consistency audit is tested: consistency drops towards
            1/num_choices while accuracy stays near chance.
        chatter: Probability of wrapping the answer in verbose prose, to
            exercise the answer parsers.
        refusal: Probability of returning an unparseable reply, to exercise
            unparsed-item accounting.
        yes_bias: Probability of answering "yes" to a binary existence question
            regardless of the label. This is the object-hallucination failure
            mode POPE was built to measure, so the harness needs a model that
            exhibits it on demand.
    """

    supports_images = True

    def __init__(
        self,
        model: str | None = None,
        name: str | None = None,
        skill: float = 0.7,
        position_bias: float = 0.0,
        chatter: float = 0.15,
        refusal: float = 0.05,
        yes_bias: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model or name or "echo-v1"
        self.name = self.model
        self.skill = float(skill)
        self.position_bias = float(position_bias)
        self.chatter = float(chatter)
        self.refusal = float(refusal)
        self.yes_bias = float(yes_bias)
        self.info = {
            "backend": "echo",
            "model": self.model,
            "skill": self.skill,
            "position_bias": self.position_bias,
            "chatter": self.chatter,
            "refusal": self.refusal,
            "yes_bias": self.yes_bias,
            "deterministic": True,
            "sees_gold": True,
        }

    def _unit(self, seed: str, salt: str = "") -> float:
        """A stable pseudo-random number in [0, 1) derived from `seed`."""
        digest = hashlib.sha256(f"{salt}|{seed}".encode()).hexdigest()
        return int(digest[:8], 16) / 0xFFFFFFFF

    def generate(
        self,
        *,
        image: Any,
        prompt: str,
        system: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        **kwargs: Any,
    ) -> ModelResponse:
        """Answer without a gold label.

        The base class requires this method, and it makes the adapter usable
        outside the runner. With no label to work from, replies are hash-derived
        and therefore effectively arbitrary -- useful for exercising parsers, not
        for measuring accuracy.
        """
        return self.generate_with_context(
            image=image,
            prompt=prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            context=None,
            **kwargs,
        )

    def generate_with_context(
        self,
        *,
        image: Any,
        prompt: str,
        system: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Answer using the gold label from `context`, then wrap it in noise.

        `context['gold']` is the answer in the *display* coordinate system, i.e.
        the letter the prompt is asking for. Reading it here rather than from
        the prompt keeps answer keys out of anything that gets logged.
        """
        context = context or {}
        # Answer the letter the prompt asks for. Under option permutation the
        # canonical gold is a different letter from the displayed one, and it is
        # the displayed one the model is being asked about. Mapping it back to
        # canonical is the harness's `decode` step, so doing it here too would
        # skip the very code path this adapter exists to exercise.
        gold = context.get("display_gold") or context.get("gold")
        num_choices = int(context.get("num_choices", 4) or 4)

        # Seed on the item, not the prompt. Every permutation of one item shares
        # a seed, so the model is *consistently* right or consistently wrong --
        # which is the behaviour a real model shows, and it keeps the
        # order-consistency audit measuring the model instead of the hash.
        # `position_bias` deliberately does not use the seed, since exploiting a
        # display position is prompt-dependent by nature.
        seed = str(context.get("uid") or prompt)

        prompt_tokens = max(1, len(prompt) // 4)

        if self._unit(seed, "refuse") < self.refusal:
            text = "I'm sorry, I cannot determine the answer from this image."
            finish = "stop"
        else:
            roll = self._unit(prompt, "choose")
            is_binary = str(gold).lower() in {"yes", "no"}
            if self.position_bias and roll < self.position_bias:
                # Ignore the image entirely: always the first option shown.
                answer = _POSITIONS[0]
            elif is_binary and self.yes_bias and self._unit(seed, "yesbias") < self.yes_bias:
                # Affirm everything. On POPE this is the hallucination
                # signature: high recall, poor precision on absent objects.
                answer = "yes"
            elif gold and self._unit(seed, "skill") < self.skill:
                answer = str(gold)
            else:
                answer = self._wrong_answer(seed, str(gold or ""), num_choices)

            if self._unit(seed, "chatter") < self.chatter:
                text = f"Looking at the image carefully, the answer is {answer}."
            else:
                text = answer
            finish = "stop"

        completion_tokens = max(1, len(text) // 4)
        if max_tokens and completion_tokens > max_tokens:
            text = text[: max(1, max_tokens * 4)]
            completion_tokens = max_tokens
            finish = "length"

        return ModelResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish,
        )

    def _wrong_answer(self, seed: str, avoid: str, num_choices: int) -> str:
        """Pick a plausible wrong answer of the same shape as `avoid`."""
        if avoid.lower() in {"yes", "no"}:
            return "no" if avoid.lower() == "yes" else "yes"

        available = [_POSITIONS[i] for i in range(min(num_choices, len(_POSITIONS)))]
        others = [c for c in available if c != avoid.upper()]
        if others:
            idx = int(self._unit(seed, "wrong") * len(others))
            return others[min(idx, len(others) - 1)]

        numbers = [n for n in range(1, 10) if str(n) != avoid]
        if numbers:
            idx = int(self._unit(seed, "wrong") * len(numbers))
            return str(numbers[min(idx, len(numbers) - 1)])
        return "unknown"
