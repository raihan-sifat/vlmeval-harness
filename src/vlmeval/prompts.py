"""Prompt templates.

Prompt wording is a confound in every VLM benchmark: a model that scores 8
points lower may have been asked a worse question, not seen the image worse.
Three rules keep the comparison defensible:

1. One template per task, shared by every model. Model-specific prompt tuning
   would make the leaderboard meaningless.
2. The output contract is stated explicitly ("answer with the letter only"),
   because a chatty model wastes tokens and breaks parsing.
3. Each template has a `system` variant that adds grounding pressure, so
   prompt sensitivity can be measured as a controlled variable rather than
   discovered as an unexplained score gap. See `PromptStyle`.

Answer keys never appear in a rendered prompt. The offline ``echo`` backend
receives the gold label through
:meth:`~vlmeval.models.base.ModelAdapter.generate_with_context`, which the
default implementation discards -- so no provider, and no committed result file,
can pick it up from the prompt text.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

LETTERS = "ABCDEFGHIJ"


class PromptStyle(str, Enum):
    """How much scaffolding to put around the question."""

    #: Just the question plus the answer-format contract. The default, and the
    #: one comparable to published benchmark numbers.
    PLAIN = "plain"
    #: Adds a system message instructing the model to answer from the image
    #: only and to admit uncertainty. Usually worth a few points on
    #: hallucination probes; the gap between styles is itself a result.
    GROUNDED = "grounded"


GROUNDED_SYSTEM = (
    "You are a careful visual assistant. Answer only from what is visible in the "
    "image. If the image does not show enough to decide, say you cannot tell. "
    "Keep your reply to the requested format and nothing else."
)

PLAIN_SYSTEM = (
    "You are a helpful visual assistant. Follow the requested answer format exactly."
)


@dataclass(frozen=True)
class PromptTemplate:
    """A question template plus the contract appended after the options."""

    name: str
    body: str
    instruction: str

    def render(self, **kwargs: object) -> str:
        return f"{self.body.format(**kwargs).strip()}\n\n{self.instruction.strip()}"


MCQ_TEMPLATE = PromptTemplate(
    name="mcq_letter",
    body="{question}\n\n{options}",
    instruction=(
        "Answer with the letter of the correct option only "
        "(for example: A). Do not explain your reasoning."
    ),
)

#: A variant that also forbids restating the option text. Some models otherwise
#: reply "A. a red triangle", which parses fine but costs more output tokens
#: than a single letter.
MCQ_TERSE_TEMPLATE = PromptTemplate(
    name="mcq_letter_terse",
    body="{question}\n\n{options}",
    instruction="Reply with exactly one letter and nothing else.",
)

BINARY_TEMPLATE = PromptTemplate(
    name="binary_yesno",
    body="{question}",
    instruction='Answer with a single word: "yes" or "no".',
)

BINARY_TERSE_TEMPLATE = PromptTemplate(
    name="binary_yesno_terse",
    body="{question}",
    instruction="Reply with yes or no. Nothing else.",
)

TEMPLATES: dict[str, PromptTemplate] = {
    t.name: t
    for t in (
        MCQ_TEMPLATE,
        MCQ_TERSE_TEMPLATE,
        BINARY_TEMPLATE,
        BINARY_TERSE_TEMPLATE,
    )
}


def format_options(choices: list[str], order: list[int] | None = None) -> str:
    """Render answer options as ``A. text`` lines.

    Args:
        choices: The option texts in their canonical order.
        order: Optional permutation. ``order[i]`` is the index of the
            canonical choice shown at display position ``i``. Permuting
            options is how the harness measures answer-position bias: a model
            that always picks the first plausible-looking letter will move its
            score when the order changes, and one that reads the image will
            not.
    """
    indices = order if order is not None else list(range(len(choices)))
    return "\n".join(f"{LETTERS[i]}. {choices[j]}" for i, j in enumerate(indices))


def system_prompt(style: PromptStyle | str) -> str:
    """Return the system message for a prompt style."""
    value = PromptStyle(style)
    return GROUNDED_SYSTEM if value is PromptStyle.GROUNDED else PLAIN_SYSTEM


def render_mcq(
    question: str,
    choices: list[str],
    order: list[int] | None = None,
    template: str = "mcq_letter",
) -> str:
    """Build a multiple-choice prompt with options in the given order."""
    return TEMPLATES[template].render(
        question=question, options=format_options(choices, order)
    )


def render_binary(question: str, template: str = "binary_yesno") -> str:
    """Build a yes/no prompt."""
    return TEMPLATES[template].render(question=question)
