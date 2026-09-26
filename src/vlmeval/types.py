"""Core data structures shared by datasets, models, tasks and reports.

Everything here is a plain dataclass with JSON-serialisable fields so that a run
can be written to disk and re-read later without any custom decoding logic.
That property is what makes the harness auditable: a result file is a complete,
self-describing record of what was asked and what came back.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

# A dataset item points at an image on disk; synthetic fixtures may instead hold
# an in-memory PIL image. Keeping both in one union avoids a second code path
# through the whole harness for a difference that only matters at load time.
ImageRef = str | Path | Any

TaskKind = Literal["mcq", "binary"]


def utc_now() -> str:
    """ISO-8601 UTC timestamp, second resolution, used in every artifact."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stable_hash(payload: Any, length: int = 16) -> str:
    """Deterministic short hash of any JSON-serialisable payload.

    Used for cache keys and config fingerprints, so it must never depend on
    `hash()` (salted per process) or on dict ordering.
    """
    blob = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:length]


def file_digest(path: str | Path, length: int = 16) -> str:
    """Content hash of a file, used to invalidate cache entries on image edits."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:length]


@dataclass(slots=True)
class Example:
    """A single evaluation item, normalised away from any dataset's own format.

    Attributes:
        uid: Stable identifier. Used as the resume key, so it must be unique
            within a task and must not depend on row order.
        image: Image path, or an in-memory image for generated fixtures.
        question: The question text, without answer options.
        choices: Option strings for `mcq` tasks; `None` for `binary`.
        label: Gold answer after task normalisation: a letter such as ``"B"``
            for MCQ, or ``"yes"``/``"no"`` for binary tasks.
        category: Optional subgroup used for per-category breakdowns, e.g. the
            POPE split (random / popular / adversarial) or an MMMU discipline.
        meta: Anything else worth keeping: source row, image caption, counts.
    """

    uid: str
    image: ImageRef
    question: str
    label: str | None = None
    choices: list[str] | None = None
    category: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        if isinstance(self.image, (str, Path)):
            data["image"] = str(self.image)
        else:
            data["image"] = f"<in-memory:{type(self.image).__name__}>"
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Example:
        return cls(
            uid=data["uid"],
            image=data["image"],
            question=data["question"],
            label=data.get("label"),
            choices=data.get("choices"),
            category=data.get("category"),
            meta=data.get("meta") or {},
        )


@dataclass(slots=True)
class ModelResponse:
    """A single model reply plus the accounting needed to compare runs.

    Token counts and latency are captured per call rather than per run so that
    cost-per-item and latency percentiles stay meaningful when models are called
    concurrently and items have different difficulty.
    """

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    error: str | None = None
    finish_reason: str | None = None
    raw: Any = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_json(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_s": round(self.latency_s, 4),
            "error": self.error,
            "finish_reason": self.finish_reason,
        }


@dataclass(slots=True)
class ItemResult:
    """The full record for one evaluated item: prompt, reply, parse, score."""

    uid: str
    task: str
    model: str
    category: str | None
    gold: str | None
    pred: str | None
    correct: bool | None
    prompt: str
    response: ModelResponse
    variant: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "task": self.task,
            "model": self.model,
            "category": self.category,
            "gold": self.gold,
            "pred": self.pred,
            "correct": self.correct,
            "prompt": self.prompt,
            "response": self.response.to_json(),
            "variant": self.variant,
            "meta": self.meta,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> ItemResult:
        resp = data.get("response") or {}
        return cls(
            uid=data["uid"],
            task=data["task"],
            model=data["model"],
            category=data.get("category"),
            gold=data.get("gold"),
            pred=data.get("pred"),
            correct=data.get("correct"),
            prompt=data.get("prompt", ""),
            response=ModelResponse(
                text=resp.get("text", ""),
                prompt_tokens=resp.get("prompt_tokens", 0),
                completion_tokens=resp.get("completion_tokens", 0),
                latency_s=resp.get("latency_s", 0.0),
                error=resp.get("error"),
                finish_reason=resp.get("finish_reason"),
            ),
            variant=data.get("variant"),
            meta=data.get("meta") or {},
        )


@dataclass(slots=True)
class MetricSummary:
    """A metric name with its value, sample count and bootstrap interval.

    `ci_low`/`ci_high` are the bounds of a 95% percentile bootstrap CI over
    items. Shipping the interval with every number means no table in the report
    can accidentally present a noisy point estimate as if it were exact.
    """

    name: str
    value: float | None
    n: int = 0
    ci_low: float | None = None
    ci_high: float | None = None
    higher_is_better: bool = True

    def fmt(self, pct: bool = False, with_ci: bool = True) -> str:
        if self.value is None:
            return "n/a"
        if pct:
            body = f"{self.value * 100:.1f}"
            if with_ci and self.ci_low is not None and self.ci_high is not None:
                return f"{body} [{self.ci_low * 100:.1f}, {self.ci_high * 100:.1f}]"
            return body
        body = f"{self.value:.4f}"
        if with_ci and self.ci_low is not None and self.ci_high is not None:
            return f"{body} [{self.ci_low:.4f}, {self.ci_high:.4f}]"
        return body

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "n": self.n,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "higher_is_better": self.higher_is_better,
        }


@dataclass(slots=True)
class RunResult:
    """Everything about one (model, task, dataset) evaluation.

    This is the unit that `results/<run_id>/` is built around: metrics for the
    headline numbers, per-category breakdowns for the interesting slices, and
    item-level records for anyone who wants to re-analyse.
    """

    run_id: str
    task: str
    model: str
    dataset: str
    created_at: str = field(default_factory=utc_now)
    metrics: dict[str, MetricSummary] = field(default_factory=dict)
    by_category: dict[str, dict[str, MetricSummary]] = field(default_factory=dict)
    counts: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    items: list[ItemResult] = field(default_factory=list)

    def primary(self) -> MetricSummary | None:
        """The headline metric, used for leaderboard ordering."""
        for key in ("accuracy", "f1", "anls", "score"):
            if key in self.metrics:
                return self.metrics[key]
        return next(iter(self.metrics.values()), None)

    def to_json(self, include_items: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "run_id": self.run_id,
            "task": self.task,
            "model": self.model,
            "dataset": self.dataset,
            "created_at": self.created_at,
            "metrics": {k: v.to_json() for k, v in self.metrics.items()},
            "by_category": {
                cat: {k: v.to_json() for k, v in metrics.items()}
                for cat, metrics in self.by_category.items()
            },
            "counts": self.counts,
            "config": self.config,
        }
        if include_items:
            data["items"] = [item.to_json() for item in self.items]
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> RunResult:
        return cls(
            run_id=data["run_id"],
            task=data["task"],
            model=data["model"],
            dataset=data["dataset"],
            created_at=data.get("created_at", ""),
            metrics={
                k: MetricSummary(**v) for k, v in (data.get("metrics") or {}).items()
            },
            by_category={
                cat: {k: MetricSummary(**v) for k, v in metrics.items()}
                for cat, metrics in (data.get("by_category") or {}).items()
            },
            counts=data.get("counts") or {},
            config=data.get("config") or {},
            items=[ItemResult.from_json(i) for i in (data.get("items") or [])],
        )


def group_by(items: Iterable[ItemResult], key: str) -> dict[Any, list[ItemResult]]:
    """Group item results by an attribute, skipping items where it is unset."""
    buckets: dict[Any, list[ItemResult]] = {}
    for item in items:
        value = getattr(item, key, None)
        if value is None:
            continue
        buckets.setdefault(value, []).append(item)
    return buckets
