"""Report and artifact generation.

A run is only useful if someone can read it without writing code, so every run
directory gets the same set of files:

``metrics.json``
    The full machine-readable result, including item-level records.
``summary.md``
    A human-readable report: headline table with confidence intervals, the
    per-category breakdown, answer-distribution diagnostics, and cost. Written
    to be pasted into an issue or a README.
``summary.csv``
    One row per model, for a spreadsheet or a plot.
``manifest.json``
    Versions, config, model identity, wall time and spend -- enough to
    reproduce or explain the run months later.

Markdown tables never show a bare point estimate. A leaderboard of bare numbers
is how a 2-point gap between two models gets reported as a finding.
"""

from __future__ import annotations

import csv
import json
import platform
import sys
from pathlib import Path
from typing import Any

from .registry import sanitize
from .types import RunResult, utc_now

#: Metrics worth showing in a per-category table, in preference order.
CATEGORY_METRICS = ("accuracy", "f1", "precision", "recall", "yes_ratio", "n")


def write_run(
    result: RunResult,
    out_dir: str | Path = "results",
    *,
    include_items: bool = False,
) -> Path:
    """Write every artifact for one run and return the run directory.

    The per-item records go to ``results.jsonl``, not into ``metrics.json``.
    Inlining them made a 260-item run's metrics file 750 KB, which is a real
    cost to anyone reading a score programmatically: a leaderboard over ten
    models had to parse seven megabytes to compare ten numbers. They are still
    committed and still complete -- just in a file you only open when you
    actually want item-level detail.
    """
    run_dir = Path(out_dir) / sanitize(result.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "metrics.json").write_text(
        json.dumps(result.to_json(include_items=include_items), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if result.items:
        with (run_dir / "results.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for item in result.items:
                handle.write(json.dumps(item.to_json(), ensure_ascii=False) + "\n")
    (run_dir / "summary.md").write_text(render_markdown(result), encoding="utf-8")
    write_csv(result, run_dir / "summary.csv")
    (run_dir / "manifest.json").write_text(
        json.dumps(build_manifest(result), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return run_dir


def load_run(run_dir: str | Path) -> RunResult:
    """Reassemble a run from ``metrics.json`` plus ``results.jsonl``.

    ``summarize`` only needs the metrics and can stop at the JSON; ``compare``
    needs the per-item outcomes for its paired tests, so it reads the JSONL
    alongside. Missing JSONL is not an error -- it just means item-level detail
    is unavailable and the run can still be summarised.
    """
    from .types import RunResult

    run_dir = Path(run_dir)
    data = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    result = RunResult.from_json(data)

    items_path = run_dir / "results.jsonl"
    if items_path.is_file():
        from .types import ItemResult

        items: list[ItemResult] = []
        for line in items_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                items.append(ItemResult.from_json(json.loads(line)))
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        result.items = items
    return result


def build_manifest(result: RunResult) -> dict[str, Any]:
    """Provenance block: enough to reproduce or explain a run later."""
    return {
        "run_id": result.run_id,
        "created_at": result.created_at,
        "generated_at": utc_now(),
        "task": result.task,
        "model": result.model,
        "dataset": result.dataset,
        "config": result.config,
        "counts": result.counts,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }


def _fmt(summary: Any, pct: bool = True, with_ci: bool = True) -> str:
    if summary is None or getattr(summary, "value", None) is None:
        return "-"
    return summary.fmt(pct=pct, with_ci=with_ci)


def render_markdown(result: RunResult, *, peer_results: list[RunResult] | None = None) -> str:
    """Render a run as Markdown.

    Args:
        result: The run to describe.
        peer_results: Other runs on the same task and dataset. When given, a
            comparison table is added with a paired significance test, because
            "is B actually better than A" cannot be answered by two numbers in
            separate tables.
    """
    lines: list[str] = []
    add = lines.append

    add(f"# {result.task} / {result.dataset} / `{result.model}`")
    add("")
    add(f"_Generated {utc_now()}_")
    add("")

    counts = result.counts
    add("## Run")
    add("")
    add("| Field | Value |")
    add("| --- | --- |")
    add(f"| Task | `{result.task}` |")
    add(f"| Dataset | `{result.dataset}` |")
    add(f"| Model | `{result.model}` |")
    add(f"| Items | {counts.get('n_examples', '-')} |")
    add(f"| Requests | {counts.get('n_requests', '-')} |")
    add(f"| From cache | {counts.get('n_cached', 0)} |")
    add(f"| Failed calls | {counts.get('n_failed', 0)} |")
    add(f"| Unparsed replies | {counts.get('n_unparsed', '-')} "
        f"({_pct(counts.get('n_unparsed'), counts.get('n_requests'))}) |")
    add(f"| Prompt tokens | {counts.get('prompt_tokens', 0):,} |")
    add(f"| Completion tokens | {counts.get('completion_tokens', 0):,} |")
    add(f"| Est. cost | ${counts.get('est_cost_usd', 0.0):.4f} |")
    add(f"| Wall time | {counts.get('elapsed_s', 0):.1f}s |")
    if counts.get("stopped_reason"):
        add(f"| **Stopped early** | **{counts['stopped_reason']}** |")
    add("")

    if counts.get("stopped_reason"):
        add("> This run was cut short by a budget limit. Treat every number below")
        add("> as a lower bound on coverage, not as a score on the full dataset.")
        add("")

    add("## Headline metrics")
    add("")
    add("Values are point estimates with 95% bootstrap confidence intervals.")
    add("")
    add("| Metric | Value | 95% CI | n | Better |")
    add("| --- | --- | --- | --- | --- |")
    for name, summary in result.metrics.items():
        if name.startswith("pick_rate_") or name in {"n_variants", "n_scored", "mean_latency_s"}:
            continue
        arrow = "higher" if summary.higher_is_better else "lower"
        add(
            f"| {name} | {_fmt(summary)} | "
            f"[{_pct(summary.ci_low)}, {_pct(summary.ci_high)}] | {summary.n} | {arrow} |"
        )
    add("")

    if result.by_category:
        add("## By category")
        add("")
        add("Categories are dataset-defined: MMMU disciplines, or the POPE")
        add("random / popular / adversarial splits. A single pooled number hides")
        add("the differences that matter, so the split is always shown.")
        add("")
        for category, metrics in sorted(result.by_category.items()):
            shown = [m for name, m in metrics.items() if name in CATEGORY_METRICS]
            if not shown:
                continue
            add(f"### {category}")
            add("")
            add("| Metric | Value | 95% CI | n |")
            add("| --- | --- | --- | --- |")
            for summary in shown:
                add(
                    f"| {summary.name} | {_fmt(summary)} | "
                    f"[{_pct(summary.ci_low)}, {_pct(summary.ci_high)}] | {summary.n} |"
                )
            add("")

    picks = {
        k.split("pick_rate_", 1)[1]: v
        for k, v in result.metrics.items()
        if k.startswith("pick_rate_")
    }
    if picks and any(k != "unparsed" for k in picks):
        add("## Answer distribution")
        add("")
        add("Where the model's answers actually land. A model that scores well")
        add("but piles its answers onto one letter is exploiting a position")
        add("prior rather than reading the image.")
        add("")
        add("| Bucket | Share |")
        add("| --- | --- |")
        for key, summary in sorted(picks.items()):
            add(f"| {key} | {_fmt(summary)} |")
        add("")

    if peer_results:
        lines.extend(_comparison_section(result, peer_results))

    lines.extend(_methodology_section(result))
    return "\n".join(lines) + "\n"


def _pct(value: Any, total: Any = None) -> str:
    if value is None:
        return "-"
    if total in (None, 0):
        return f"{float(value):.3f}"
    return f"{100.0 * float(value) / float(total):.1f}%"


def _comparison_section(result: RunResult, peers: list[RunResult]) -> list[str]:
    """A paired comparison against peer runs on the same task and dataset."""
    from .stats import mcnemar_test

    lines = ["## Comparison", ""]
    lines.append("Paired McNemar test on shared items. A difference is only")
    lines.append("meaningful when the intervals separate *and* the test rejects.")
    lines.append("")
    lines.append("| Model | Metric | 95% CI | vs this run | p |")
    lines.append("| --- | --- | --- | --- | --- |")

    mine = {i.uid: i for i in result.items}
    mine_flags = {i.uid: i.correct for i in result.items}

    for peer in peers:
        primary = peer.primary()
        if primary is None:
            continue
        shared = [uid for uid in mine if uid in {i.uid for i in peer.items}]
        if not shared:
            lines.append(f"| `{peer.model}` | {_fmt(primary)} | - | no shared items | - |")
            continue
        test = mcnemar_test(
            [mine_flags.get(uid) for uid in shared],
            [next(i.correct for i in peer.items if i.uid == uid) for uid in shared],
        )
        verdict = (
            f"{test['delta'] * 100:+.1f} pts"
            + (" (sig)" if test["significant"] else " (n.s.)")
        )
        lines.append(
            f"| `{peer.model}` | {_fmt(primary)} | "
            f"[{_pct(primary.ci_low)}, {_pct(primary.ci_high)}] | {verdict} | "
            f"{test['p_value']:.3f} |"
        )
    lines.append("")
    return lines


def _methodology_section(result: RunResult) -> list[str]:
    """Caveats that belong next to the numbers, not in a distant appendix."""
    lines = ["## Notes", ""]
    info = result.config.get("model_info") or {}
    if info:
        backend = info.get("backend", "?")
        lines.append(f"- Backend: `{backend}`")
        for key in ("model", "revision", "device", "dtype", "quantisation", "gpu"):
            if info.get(key):
                lines.append(f"- {key.replace('_', ' ').capitalize()}: `{info[key]}`")
    task_info = result.config.get("task_info") or {}
    if task_info.get("description"):
        lines.append(f"- Task: {task_info['description']}")
    if task_info.get("style"):
        lines.append(f"- Prompt style: `{task_info['style']}`")
    lines.append(
        "- Unparsed replies are excluded from the mean and reported separately; "
        "they are never scored as correct."
    )
    lines.append(
        "- Intervals are percentile bootstrap over items with a fixed seed, so "
        "re-running the report reproduces them exactly."
    )
    if result.config.get("num_permutations"):
        lines.append(
            f"- {result.config['num_permutations']} extra option-order permutation(s) "
            "per item; see the consistency and order_sensitivity metrics."
        )
    lines.append("")
    return lines


def write_csv(result: RunResult, path: str | Path) -> Path:
    """One row per metric, long format, for plotting or a spreadsheet."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for name, summary in result.metrics.items():
        rows.append(
            {
                "model": result.model,
                "task": result.task,
                "dataset": result.dataset,
                "category": "all",
                "metric": name,
                "value": summary.value,
                "ci_low": summary.ci_low,
                "ci_high": summary.ci_high,
                "n": summary.n,
            }
        )
    for category, metrics in result.by_category.items():
        for name, summary in metrics.items():
            rows.append(
                {
                    "model": result.model,
                    "task": result.task,
                    "dataset": result.dataset,
                    "category": category,
                    "metric": name,
                    "value": summary.value,
                    "ci_low": summary.ci_low,
                    "ci_high": summary.ci_high,
                    "n": summary.n,
                }
            )
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["metric"])
        writer.writeheader()
        writer.writerows(rows)
    return out


def leaderboard(results: list[RunResult], metric: str | None = None) -> str:
    """A cross-model table for one task and dataset, sorted by the headline metric.

    Args:
        results: Runs to compare. Runs on different tasks or datasets are
            grouped separately rather than ranked against each other, because
            accuracy on POPE and accuracy on MMMU are not the same quantity.
        metric: Metric to rank by; defaults to each run's primary metric.
    """
    groups: dict[tuple[str, str], list[RunResult]] = {}
    for result in results:
        groups.setdefault((result.task, result.dataset), []).append(result)

    lines: list[str] = []
    for (task, dataset), runs in sorted(groups.items()):
        lines.append(f"## {task} / {dataset}")
        lines.append("")
        lines.append("| Model | " + " | ".join(_metric_names(runs, metric)) + " | Cost | n |")
        header_metric = metric or "accuracy"
        lines.append("| --- | " + " | ".join(["---"] * (len(_metric_names(runs, metric)) + 2)) + " |")

        def sort_key(r: RunResult) -> float:
            summary = r.metrics.get(header_metric) or r.primary()
            return -(summary.value if summary and summary.value is not None else -1.0)

        for run in sorted(runs, key=sort_key):
            cells = []
            for name in _metric_names(runs, metric):
                summary = run.metrics.get(name)
                cells.append(_fmt(summary) if summary else "-")
            cost = run.counts.get("est_cost_usd", 0.0)
            lines.append(
                f"| `{run.model}` | " + " | ".join(cells) +
                f" | ${cost:.4f} | {run.counts.get('n_requests', '-')} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def _metric_names(runs: list[RunResult], metric: str | None) -> list[str]:
    """Metric columns to show: the requested one, else the common headline ones."""
    if metric:
        return [metric]
    common: set[str] = set()
    for run in runs:
        primary = run.primary()
        if primary:
            common.add(primary.name)
    preferred = [m for m in ("accuracy", "f1", "anls", "score") if m in common]
    return preferred or sorted(common) or ["accuracy"]


def write_leaderboard(
    results: list[RunResult], out_dir: str | Path = "results", filename: str = "leaderboard.md"
) -> Path:
    """Write the cross-model leaderboard, plus a combined CSV."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    header = (
        "# Leaderboard\n\n"
        "_Generated "
        + utc_now()
        + "_. Intervals are 95% bootstrap CIs over items; read them before\n"
        "reading the ranking.\n\n"
    )
    path.write_text(header + leaderboard(results), encoding="utf-8")
    return path
