"""The task contract.

A task owns three things and nothing else: how to turn an
:class:`~vlmeval.types.Example` into concrete prompts, how to turn a model reply
back into a comparable answer, and how to score the resulting outcomes. It
never calls a model and never touches the filesystem.

The indirection through :class:`Query` is what allows one item to produce more
than one request. Multiple-choice questions are asked under several option
orders to measure position bias; that is expressed as several `Query` objects
for one `Example`, and the task folds their answers back together. Adding a new
probe style therefore never requires changing the runner.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

from ..types import Example, ItemResult, MetricSummary


@dataclass
class Query:
    """One concrete request derived from an example.

    Attributes:
        prompt: The fully rendered prompt text.
        variant: Label for this variant, e.g. ``"base"`` or ``"perm2"``. Stored
            on every item result so per-variant analysis is possible later.
        decode: Maps a raw parsed answer to a canonical label, or ``None`` to
            use the parsed value directly. For a permuted MCQ this maps the
            displayed letter back to the letter of the original option order,
            which is what makes variants comparable.
        gold: The gold label *in the coordinate system `decode` produces*, so
            scoring is identical for every variant.
        meta: Anything else worth keeping, such as the permutation used.
    """

    prompt: str
    variant: str = "base"
    decode: dict[str, str] | None = None
    gold: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def apply(self, parsed: str | None) -> str | None:
        """Map a parsed raw answer into canonical space."""
        if parsed is None:
            return None
        if self.decode is None:
            return parsed
        return self.decode.get(parsed, parsed)


@dataclass
class ItemOutcome:
    """The scored result of one example, across all of its queries."""

    result: ItemResult
    #: Canonical answers per variant, for task-level aggregate metrics.
    decoded: dict[str, str | None]


class Task(abc.ABC):
    """Base class for evaluation tasks."""

    #: Identifier used in config, result filenames and reports.
    name: str = "base"
    #: Discriminator for grouping and for sanity checks against the data.
    kind: str = "mcq"
    #: Task-specific default reply cap. Single-token answers want a small cap.
    default_max_tokens: int = 16
    #: Human-readable one-liner for the report header.
    description: str = ""

    def __init__(self, **options: Any) -> None:
        self.options = options
        self.style = options.get("style", "plain")
        self.template = options.get("template")

    @abc.abstractmethod
    def queries(self, example: Example) -> list[Query]:
        """Build every request to issue for this example."""

    @abc.abstractmethod
    def parse(self, text: str, query: Query) -> str | None:
        """Extract a comparable answer from raw model output."""

    def system_prompt(self) -> str | None:
        """The system message to send, if any. Overridable per task."""
        from ..prompts import system_prompt

        return system_prompt(self.style)

    def max_tokens(self) -> int:
        """Reply cap for this task."""
        return int(self.options.get("max_tokens", self.default_max_tokens))

    @abc.abstractmethod
    def score(self, outcomes: list[ItemOutcome]) -> dict[str, MetricSummary]:
        """Aggregate item outcomes into named metrics with intervals."""

    def by_category(
        self, outcomes: list[ItemOutcome]
    ) -> dict[str, dict[str, MetricSummary]]:
        """Metrics computed separately per category.

        The default groups by ``ItemResult.category`` and recomputes the same
        metrics, which is what makes a POPE random/popular/adversarial split or
        an MMMU discipline breakdown available for free.

        Items are bucketed in a single pass by category. The obvious-looking
        alternative -- group the results, then filter the outcomes with
        ``o.result in items`` -- is wrong twice over: it is quadratic in the
        item count, and it matches on dataclass equality, so two distinct items
        with identical fields land in each other's buckets.
        """
        buckets: dict[str, list[ItemOutcome]] = {}
        for outcome in outcomes:
            buckets.setdefault(str(outcome.result.category or "all"), []).append(outcome)
        return {category: self.score(items) for category, items in buckets.items()}

    def extra_options(self) -> dict[str, Any]:
        """Task-specific settings to record in the manifest and fingerprint.

        Subclass constructors take their knobs as named parameters, so those
        values never reach ``self.options``. Without this hook they would be
        missing from the run manifest *and* from :meth:`fingerprint`, which
        would let a run change its permutations and still resume onto results
        computed with the old ones.
        """
        return {}

    def describe(self) -> dict[str, Any]:
        """Configuration snapshot for the run manifest."""
        return {
            "task": self.name,
            "kind": self.kind,
            "style": self.style,
            "template": self.template,
            "max_tokens": self.max_tokens(),
            "description": self.description,
            "options": {
                **self.extra_options(),
                **{k: v for k, v in self.options.items() if k not in {"style", "template"}},
            },
        }

    def fingerprint(self, length: int = 8) -> str:
        """A short hash of everything that changes what the model is asked.

        Changing the template, the prompt style, the token cap or the number of
        permutations changes the questions, so results computed under the old
        settings cannot answer the new ones. Keying the per-item result file on
        this hash means such a change starts a fresh file instead of resuming
        onto stale rows and quietly reporting yesterday's answers as today's.
        """
        from ..types import stable_hash

        return stable_hash(self.describe(), length=length)
