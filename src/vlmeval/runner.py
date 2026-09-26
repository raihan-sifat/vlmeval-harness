"""The evaluation runner.

The one place where items become prompts, prompts become calls, replies become
answers and answers become metrics. What makes a long run survivable:

* **Concurrency** capped per backend, because a locally loaded model is not
  thread-safe and sixteen workers against a 4 GB GPU is an OOM, not a speedup.
* **Retries** with exponential backoff *and jitter*. Jitter is not cosmetic:
  without it every worker retries in lockstep and reproduces the burst that
  triggered the rate limit in the first place.
* **Resume** from the per-item JSONL sink, so an interrupted run continues.
* **Budget** enforcement in tokens or dollars.
* **Deterministic scoring.** Results are re-sorted into dataset order before
  metrics are computed, so worker interleaving can never change a number.
"""

from __future__ import annotations

import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .cache import ResponseCache
from .models.base import ModelAdapter
from .runstate import Budget, JsonlSink, Progress, RunConfig, is_retryable
from .tasks.base import ItemOutcome, Query, Task
from .types import Example, ItemResult, ModelResponse, RunResult, utc_now


@dataclass
class _Job:
    """One request: a single (example, query) pair."""

    example: Example
    query: Query
    index: int

    @property
    def key(self) -> tuple[str, str]:
        return (self.example.uid, self.query.variant)


@dataclass
class RunSummary:
    """Accounting for one run: throughput, spend, failures."""

    model: str
    task: str
    dataset: str
    n_items: int
    n_requests: int
    n_cached: int
    n_failed: int
    n_retried: int
    prompt_tokens: int
    completion_tokens: int
    est_cost_usd: float
    elapsed_s: float
    stopped_reason: str | None
    config_fingerprint: str
    model_info: dict[str, Any] = field(default_factory=dict)
    task_info: dict[str, Any] = field(default_factory=dict)
    started_at: str = utc_now()

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "task": self.task,
            "dataset": self.dataset,
            "n_items": self.n_items,
            "n_requests": self.n_requests,
            "n_cached": self.n_cached,
            "n_failed": self.n_failed,
            "n_retried": self.n_retried,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "est_cost_usd": round(self.est_cost_usd, 6),
            "elapsed_s": round(self.elapsed_s, 2),
            "stopped_reason": self.stopped_reason,
            "config_fingerprint": self.config_fingerprint,
            "model_info": self.model_info,
            "task_info": self.task_info,
            "started_at": self.started_at,
        }


class Runner:
    """Execute one (model, task, dataset) evaluation end to end.

    Args:
        model: The backend under test.
        task: Decides prompting, parsing and scoring.
        cache: Response cache. ``enabled=False`` forces fresh calls.
        workers: Requested concurrency. Clamped to 1 for backends that are not
            thread-safe, and never raised above the number of jobs.
        max_retries: Attempts *after* the first, and only for transient errors.
        backoff_base: Seconds for the first retry window; doubles each attempt.
        budget: Optional spend ceiling, checked as the run proceeds.
        sink_dir: Directory for the per-item JSONL. Enables resume when set.
        resume: Skip work already recorded in the sink.
        on_progress: Optional callback invoked with a :class:`Progress`.
    """

    def __init__(
        self,
        model: ModelAdapter,
        task: Task,
        *,
        cache: ResponseCache | None = None,
        workers: int = 4,
        max_retries: int = 3,
        backoff_base: float = 1.0,
        budget: Budget | None = None,
        sink_dir: str | Path | None = "results",
        resume: bool = True,
        on_progress: Any = None,
        seed: int = 12345,
    ) -> None:
        self.model = model
        self.task = task
        self.cache = cache if cache is not None else ResponseCache(enabled=False)
        self.workers = max(1, int(workers))
        self.max_retries = max(0, int(max_retries))
        self.backoff_base = backoff_base
        self.budget = budget or Budget()
        self.sink_dir = Path(sink_dir) if sink_dir else None
        self.resume = resume
        self.on_progress = on_progress
        self._rng = random.Random(seed)
        self._rng_lock = threading.Lock()
        self._progress = Progress()
        self._sink: JsonlSink | None = None
        self._spend_lock = threading.Lock()
        self._results_lock = threading.Lock()
        self._results: dict[tuple[str, str], ItemResult] = {}

    def sink_path(self, dataset: str) -> Path | None:
        """Where this (dataset, model, task-config) triple's item results live.

        The task fingerprint is part of the filename on purpose. Without it,
        re-running with a different prompt template or permutation count would
        resume onto rows produced by the old configuration and report them as
        this configuration's results -- a silent, badly-explained wrong answer
        rather than a crash.
        """
        if self.sink_dir is None:
            return None
        from .registry import sanitize

        stem = (
            f"{sanitize(dataset)}__{sanitize(self.model.name)}"
            f"__{sanitize(self.task.name)}-{self.task.fingerprint()}"
        )
        return self.sink_dir / "raw" / f"{stem}.jsonl"

    def build_jobs(self, examples: list[Example]) -> list[_Job]:
        """Expand examples into one job per (example, query) pair."""
        jobs: list[_Job] = []
        for i, example in enumerate(examples):
            for query in self.task.queries(example):
                jobs.append(_Job(example=example, query=query, index=i))
        return jobs

    def _effective_workers(self, n_jobs: int) -> int:
        """Clamp concurrency to what the backend can actually take."""
        workers = self.workers
        if not getattr(self.model, "parallel_safe", True):
            workers = 1
        return max(1, min(workers, n_jobs or 1))

    def _sleep_backoff(self, attempt: int) -> None:
        """Exponential backoff with full jitter.

        ``uniform(0, base * 2**attempt)`` rather than a fixed exponential: when a
        provider sheds load, synchronised retries are what keep it overloaded.
        """
        with self._rng_lock:
            delay = self._rng.uniform(0, self.backoff_base * (2**attempt))
        if delay > 0:
            time.sleep(delay)

    def _request_params(self) -> dict[str, Any]:
        return {
            "max_tokens": self.task.max_tokens(),
            "temperature": float(self.task.options.get("temperature", 0.0)),
            "system": self.task.system_prompt(),
        }

    def _call(self, job: _Job, params: dict[str, Any]) -> tuple[ModelResponse, bool]:
        """Issue one request, consulting the cache first.

        Returns:
            ``(response, from_cache)``. A permanently failed call returns a
            response carrying ``error`` rather than raising, so the item is
            recorded as a failure and counted -- never silently dropped.
        """
        key = ResponseCache.make_key(
            model_name=self.model.name,
            model_info=getattr(self.model, "info", {}),
            prompt=job.query.prompt,
            image=job.example.image,
            params=params,
            gold=job.query.gold,
        )
        cached = self.cache.get(key)
        if cached is not None:
            return cached, True

        # A text-only control model still runs, but must not be handed an image
        # it cannot read. The bare question then becomes a language-only probe,
        # which is what makes the vision/no-vision comparison informative.
        image = job.example.image if getattr(self.model, "supports_images", True) else None

        response = ModelResponse(text="", error="no attempt made")
        for attempt in range(self.max_retries + 1):
            # `context` carries the gold label for adapters that opt into it.
            # The default hook in ModelAdapter discards it, so a real provider
            # never receives the answer key.
            response = self.model.generate_with_context(
                image=image,
                prompt=job.query.prompt,
                system=params["system"],
                context={
                    "gold": job.query.gold,
                    # The letter the prompt actually asks for. These differ once
                    # options are permuted, and an adapter that wants to play
                    # the question honestly has to answer the display letter --
                    # decoding it back to canonical is the harness's job.
                    "display_gold": job.query.meta.get("display_gold", job.query.gold),
                    "uid": job.example.uid,
                    "category": job.example.category,
                    "num_choices": job.query.meta.get("num_choices"),
                },
            )
            if response.ok:
                self.cache.put(key, self.model.name, response)
                return response, False
            if not is_retryable(response.error) or attempt >= self.max_retries:
                break
            self._progress.retried += 1
            self._sleep_backoff(attempt + 1)
        return response, False

    def _record_spend(self, response: ModelResponse) -> None:
        with self._spend_lock:
            self.budget.spent_requests += 1
            self.budget.spent_tokens += response.total_tokens
            self.budget.spent_usd += self.model.estimate_cost(
                response.prompt_tokens, response.completion_tokens
            )

    def _process(
        self, job: _Job, params: dict[str, Any]
    ) -> tuple[ItemResult, str | None, bool]:
        """Run one job and turn it into a scored item result."""
        response, from_cache = self._call(job, params)
        parsed = self.task.parse(response.text, job.query)
        decoded = job.query.apply(parsed)
        gold = job.query.gold

        result = ItemResult(
            uid=job.example.uid,
            task=self.task.name,
            model=self.model.name,
            category=job.example.category,
            gold=gold,
            pred=decoded,
            correct=None if decoded is None else bool(gold) and decoded == gold,
            prompt=job.query.prompt,
            response=response,
            variant=job.query.variant,
            meta={**job.query.meta, "example_meta": job.example.meta},
        )
        return result, decoded, from_cache

    def run(self, examples: list[Example], dataset: str = "dataset") -> RunResult:
        """Evaluate every example and return the scored result.

        Args:
            examples: Items to evaluate, in dataset order. The order is
                restored before scoring, so metrics never depend on the order
                in which workers happened to finish.
            dataset: Label recorded in the result and used for filenames.

        Returns:
            A :class:`~vlmeval.types.RunResult` holding the headline metrics,
            per-category breakdowns, item-level records and the full run
            manifest. ``counts['stopped_reason']`` is set if a budget cut the
            run short.
        """
        started = time.time()
        jobs = self.build_jobs(examples)
        params = self._request_params()
        with self._results_lock:
            self._results.clear()

        sink_path = self.sink_path(dataset)
        done: set[tuple[str, str, str]] = set()
        if sink_path is not None:
            self._sink = JsonlSink(sink_path)
            if self.resume:
                done = self._sink.completed_keys()
        pending = [j for j in jobs if (j.example.uid, j.query.variant, self.model.name) not in done]
        self._progress = Progress(total=len(pending), cached=len(jobs) - len(pending))
        self.budget.stopped_reason = None

        try:
            self._execute(pending, params)
        finally:
            if self._sink is not None:
                self._sink.close()

        # Re-read the sink so a resumed run scores every item, not just the
        # ones executed in this process.
        results = self._collect(sink_path, jobs)
        outcomes = self._to_outcomes(results, examples)

        config = RunConfig(
            model=self.model.name,
            task=self.task.name,
            dataset=dataset,
            limit=len(examples),
            max_tokens=self.task.max_tokens(),
            temperature=float(self.task.options.get("temperature", 0.0)),
            prompt_style=str(self.task.style),
            seed=int(self.task.options.get("seed", 12345)),
            num_permutations=int(self.task.options.get("num_permutations", 0)),
        )

        return RunResult(
            run_id=f"{self.task.name}__{dataset}__{self.model.name}",
            task=self.task.name,
            model=self.model.name,
            dataset=dataset,
            created_at=utc_now(),
            metrics=self.task.score(outcomes),
            by_category=self.task.by_category(outcomes),
            counts=self._counts(results, examples, started, sink_path),
            config={
                **config.to_dict(),
                "config_fingerprint": config.fingerprint(),
                "model_info": getattr(self.model, "info", {}),
                "task_info": self.task.describe(),
                "budget": self.budget.to_dict(),
                "progress": self._progress.to_dict(),
                "workers_requested": self.workers,
                "workers_effective": self._effective_workers(len(jobs)),
                "cache": self.cache.stats(),
            },
            items=results,
        )

    def _execute(self, jobs: list[_Job], params: dict[str, Any]) -> None:
        """Issue all pending jobs, streaming each result to the sink."""
        if not jobs:
            return
        workers = self._effective_workers(len(jobs))

        if workers == 1:
            # Serial path avoids thread hand-off overhead entirely, which
            # matters for local models where generation dominates the runtime.
            for job in jobs:
                self._handle(*self._process(job, params))
                if self._budget_stopped():
                    return
            return

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(self._process, job, params): job for job in jobs}
            for future in as_completed(futures):
                try:
                    self._handle(*future.result())
                except Exception as exc:  # noqa: BLE001 - one bad job must not kill the run
                    job = futures[future]
                    self._handle(
                        ItemResult(
                            uid=job.example.uid,
                            task=self.task.name,
                            model=self.model.name,
                            category=job.example.category,
                            gold=job.query.gold,
                            pred=None,
                            correct=None,
                            prompt=job.query.prompt,
                            response=ModelResponse(text="", error=f"runner: {exc}"),
                            variant=job.query.variant,
                        ),
                        None,
                        False,
                    )
                if self._budget_stopped():
                    for pending in futures:
                        pending.cancel()
                    return

    def _handle(self, result: ItemResult, _decoded: str | None, from_cache: bool) -> None:
        """Record one finished item: spend, counters, sink, progress."""
        if not from_cache:
            self._record_spend(result.response)
        self._progress.completed += 1
        if from_cache:
            self._progress.cached += 1
        if not result.response.ok:
            self._progress.failed += 1
        # Keep the result in memory regardless of whether a sink is configured.
        # The JSONL file is a durability mechanism, not the source of truth for
        # scoring: reading results back only from the sink would make
        # `sink_dir=None` score nothing at all, silently.
        with self._results_lock:
            self._results[(result.uid, result.variant or "base")] = result
        if self._sink is not None:
            self._sink.write(result)
        if self.on_progress is not None:
            self.on_progress(self._progress)

    def _budget_stopped(self) -> bool:
        reason = self.budget.check()
        if reason:
            self.budget.stopped_reason = reason
            self._progress.stopped_reason = reason
            return True
        return False

    def _collect(self, sink_path: Path | None, jobs: list[_Job]) -> list[ItemResult]:
        """Read finished results back, ordered by dataset then variant.

        Ordering by the job list rather than by completion time is what makes a
        run reproducible: the same items in the same order produce the same
        metric inputs regardless of how the workers interleaved.

        Two sources are merged. The in-memory map holds everything this process
        computed. The sink additionally holds items completed by an *earlier*
        process, which is what makes ``--resume`` work across invocations, and
        it also holds items skipped by a budget stop on a previous attempt. A
        fresh in-memory result wins over a stale file row, so re-running a
        previously interrupted item always reflects the newest answer.
        """
        from .types import ItemResult as _ItemResult

        rows: dict[tuple[str, str], _ItemResult] = {}
        if sink_path is not None and Path(sink_path).exists():
            with open(sink_path, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = _ItemResult.from_json(json.loads(line))
                    except (json.JSONDecodeError, KeyError):
                        continue
                    rows[(row.uid, row.variant or "base")] = row
        with self._results_lock:
            rows.update(self._results)

        ordered: list[_ItemResult] = []
        for job in jobs:
            found = rows.get(job.key)
            if found is not None:
                ordered.append(found)
        return ordered

    def _to_outcomes(
        self, results: list[ItemResult], examples: list[Example]
    ) -> list[ItemOutcome]:
        """Group per-variant results back into per-item outcomes for the task.

        The task sees one :class:`ItemOutcome` per example carrying every
        variant's decoded answer, which is what lets the MCQ task compute
        order-consistency across variants.
        """
        by_uid: dict[str, dict[str, str | None]] = {}
        first: dict[str, ItemResult] = {}
        for item in results:
            by_uid.setdefault(item.uid, {})[item.variant or "base"] = item.pred
            first.setdefault(item.uid, item)

        outcomes: list[ItemOutcome] = []
        for example in examples:
            item = first.get(example.uid)
            if item is None:
                continue
            outcomes.append(ItemOutcome(result=item, decoded=by_uid[example.uid]))
        return outcomes

    def _counts(
        self,
        results: list[ItemResult],
        examples: list[Example],
        started: float,
        sink_path: Path | None,
    ) -> dict[str, Any]:
        """Assemble the run manifest's counts and provenance block."""
        prompt_tokens = sum(r.response.prompt_tokens for r in results)
        completion_tokens = sum(r.response.completion_tokens for r in results)
        return {
            "n_examples": len(examples),
            "n_results": len(results),
            "n_requests": len(results),
            "n_cached": self._progress.cached,
            "n_failed": self._progress.failed,
            "n_retried": self._progress.retried,
            "n_unparsed": sum(1 for r in results if r.pred is None),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "est_cost_usd": round(self.budget.spent_usd, 6),
            "elapsed_s": round(time.time() - started, 2),
            "stopped_reason": self.budget.stopped_reason,
            "sink": str(sink_path) if sink_path else None,
        }


def evaluate(
    model: ModelAdapter,
    task: Task,
    examples: list[Example],
    *,
    dataset: str = "dataset",
    **kwargs: Any,
) -> RunResult:
    """Convenience wrapper: build a :class:`Runner` and run it once."""
    runner = Runner(model, task, **kwargs)
    try:
        return runner.run(examples, dataset=dataset)
    finally:
        runner.model.close()

