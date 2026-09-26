"""Run bookkeeping: configuration, budget, progress, and result persistence.

Split out from :mod:`vlmeval.runner` because these pieces are independently
testable and independently reusable -- the analysis scripts need to read a
partial run's results without importing the execution machinery.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .types import ItemResult, stable_hash

#: Error substrings that suggest a transient condition worth retrying. A 400,
#: a content-policy refusal or a bad request will not fix itself, so those are
#: recorded on the first attempt rather than burning three.
RETRYABLE_MARKERS = (
    "rate limit",
    "rate_limit",
    "429",
    "500",
    "502",
    "503",
    "504",
    "529",
    "overloaded",
    "timeout",
    "timed out",
    "connection",
    "temporarily unavailable",
    "internal server error",
    "bad gateway",
    "service unavailable",
)


def is_retryable(error: str | None) -> bool:
    """Whether an error message looks worth another attempt."""
    if not error:
        return False
    lowered = error.lower()
    return any(marker in lowered for marker in RETRYABLE_MARKERS)


@dataclass
class RunConfig:
    """Every setting that affects the numbers.

    Recorded verbatim in the run manifest. Reproducibility rests on this object:
    two runs sharing a :meth:`fingerprint` are directly comparable, and a
    mismatch explains why they are not.
    """

    model: str = ""
    task: str = ""
    dataset: str = ""
    limit: int | None = None
    max_tokens: int = 16
    temperature: float = 0.0
    prompt_style: str = "plain"
    seed: int = 12345
    num_permutations: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "task": self.task,
            "dataset": self.dataset,
            "limit": self.limit,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "prompt_style": self.prompt_style,
            "seed": self.seed,
            "num_permutations": self.num_permutations,
        }

    def fingerprint(self) -> str:
        """Short hash of the config, for cache keys and manifest comparison."""
        return stable_hash(self.to_dict())


@dataclass
class Budget:
    """A spend ceiling, checked as the run proceeds.

    A mis-typed ``--limit`` is the most expensive mistake available in a project
    like this, so a ceiling can be set in tokens or dollars. It is checked after
    each completed request, because a request's cost is not known until the
    provider answers.

    ``stopped_reason`` is written into the results: a truncated run must never
    be mistaken for a complete one, or a leaderboard ends up ranking models on
    a third of the data.

    ``max_requests`` caps the number of completed requests. Token and dollar
    ceilings are the real constraints, but they are both derived from what the
    provider reports, so neither stops a run that issues an unbounded number of
    free or near-free requests -- a mis-typed dataset path, say. This one is
    known before the first call.
    """

    max_tokens_total: int | None = None
    max_usd: float | None = None
    max_requests: int | None = None
    spent_tokens: int = 0
    spent_usd: float = 0.0
    spent_requests: int = 0
    stopped_reason: str | None = None

    def check(self) -> str | None:
        """Return a stop reason once a limit is reached, else ``None``."""
        if self.max_requests is not None and self.spent_requests >= self.max_requests:
            return "request_budget_exhausted"
        if self.max_tokens_total is not None and self.spent_tokens >= self.max_tokens_total:
            return "token_budget_exhausted"
        if self.max_usd is not None and self.spent_usd >= self.max_usd:
            return "usd_budget_exhausted"
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_tokens_total": self.max_tokens_total,
            "max_usd": self.max_usd,
            "max_requests": self.max_requests,
            "spent_tokens": self.spent_tokens,
            "spent_usd": round(self.spent_usd, 6),
            "spent_requests": self.spent_requests,
            "stopped_reason": self.stopped_reason,
        }


@dataclass
class Progress:
    """Live counters for the progress bar."""

    total: int = 0
    completed: int = 0
    cached: int = 0
    failed: int = 0
    retried: int = 0
    started_at: float = field(default_factory=time.time)
    stopped_reason: str | None = None

    def render(self) -> str:
        """One-line status, used for logs and non-interactive runs."""
        elapsed = max(1e-6, time.time() - self.started_at)
        rate = self.completed / elapsed
        pct = (100.0 * self.completed / self.total) if self.total else 0.0
        return (
            f"{self.completed}/{self.total} ({pct:5.1f}%) "
            f"cached={self.cached} failed={self.failed} retried={self.retried} "
            f"{rate:.1f} it/s"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "completed": self.completed,
            "cached": self.cached,
            "failed": self.failed,
            "retried": self.retried,
            "elapsed_s": round(time.time() - self.started_at, 2),
            "stopped_reason": self.stopped_reason,
        }


class JsonlSink:
    """Appends item results to a JSONL file as they complete.

    Per-item append rather than a single write at the end is what makes a run
    resumable: a killed process leaves a complete record of everything that
    finished, and :meth:`completed_keys` reads it back to skip on restart.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._handle = open(self.path, "a", encoding="utf-8")

    def write(self, result: ItemResult) -> None:
        with self._lock:
            self._handle.write(json.dumps(result.to_json(), ensure_ascii=False) + "\n")
            self._handle.flush()

    def completed_keys(self) -> set[tuple[str, str, str]]:
        """Which ``(uid, variant, model)`` triples are already recorded."""
        keys: set[tuple[str, str, str]] = set()
        if not self.path.exists():
            return keys
        with open(self.path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    # A torn final line from a hard kill: skip it rather than
                    # aborting the resume.
                    continue
                keys.add((row.get("uid", ""), row.get("variant") or "base", row.get("model", "")))
        return keys

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()
