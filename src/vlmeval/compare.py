"""Statistical comparison across runs.

This is the module that turns a pile of result files into a finding. The rules it
follows exist because leaderboards are usually read too confidently:

* Only runs on the **same task and dataset** are compared. Accuracy on POPE and
  accuracy on MMMU are different quantities, and a table that ranks them
  together is meaningless.
* Every claim is a **paired** test on shared items, not a difference of two
  independent accuracies.
* A family of comparisons is corrected with **Holm-Bonferroni**, because
  comparing a dozen models and reporting the smallest p-value uncorrected is a
  reliable way to publish a false positive.
* Effect sizes are printed next to p-values, since "significant" and "large
  enough to care about" are different questions.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Any

from .stats import holm_bonferroni, mcnemar_test, paired_bootstrap_delta
from .types import RunResult

#: Accuracy gaps below this are reported but flagged as not worth acting on.
#: On a 500-item benchmark a one-point gap is well inside noise even when a
#: test happens to call it significant.
PRACTICAL_THRESHOLD = 0.01


def _correctness_by_uid(result: RunResult) -> dict[str, bool | None]:
    """uid -> correct, for the canonical variant of each item.

    Non-canonical variants (option-order permutations) are excluded: including
    them would weight items with more permutations more heavily, quietly
    inflating every n and skewing the comparison.
    """
    return {
        item.uid: item.correct
        for item in result.items
        if (item.variant or "base") == "base"
    }


def compare_runs(
    results: list[RunResult],
    *,
    baseline: str | None = None,
    metric: str | None = None,
) -> str:
    """Render a Markdown comparison of every run against a baseline.

    Args:
        results: Runs to compare. Grouped by (task, dataset).
        baseline: Model name to compare against. Defaults to the best run in
            each group by the headline metric, which makes the strongest model
            the reference and frames the rest as deficits.
        metric: Metric for ranking. Defaults to each run's primary metric.

    Returns:
        Markdown. Groups with fewer than two runs are listed as skipped, with
        the reason, rather than silently omitted.
    """
    groups: dict[tuple[str, str], list[RunResult]] = defaultdict(list)
    for result in results:
        groups[(result.task, result.dataset)].append(result)

    lines: list[str] = ["# Model comparison", ""]
    lines.append(
        "Paired McNemar tests on shared items, with Holm-Bonferroni correction "
        "across each family of comparisons."
    )
    lines.append(
        "A p-value above 0.05 means the data cannot distinguish these two runs; "
        "it does not mean they are equally good."
    )
    lines.append("")

    skipped: list[str] = []
    for (task, dataset), runs in sorted(groups.items()):
        if len(runs) < 2:
            skipped.append(f"`{task}/{dataset}`: only {len(runs)} run")
            continue
        lines.extend(_group_section(task, dataset, runs, baseline, metric))

    if skipped:
        lines.append("## Skipped")
        lines.append("")
        for note in skipped:
            lines.append(f"- {note} (needs at least 2 runs to compare)")
        lines.append("")

    return "\n".join(lines) + "\n"


def _group_section(
    task: str,
    dataset: str,
    runs: list[RunResult],
    baseline: str | None,
    metric: str | None,
) -> list[str]:
    def score(run: RunResult) -> float:
        summary = run.metrics.get(metric) if metric else None
        summary = summary or run.primary()
        return summary.value if summary and summary.value is not None else -1.0

    ordered = sorted(runs, key=score, reverse=True)
    base = next((r for r in ordered if r.model == baseline), None) or ordered[0]

    lines = [f"## {task} / {dataset}", ""]
    lines.append(f"Baseline: `{base.model}`")
    lines.append("")
    lines.append("| Model | Metric | 95% CI | Delta | 95% CI of delta | p (adj) | Verdict |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")

    correctness = {run.model: _correctness_by_uid(run) for run in ordered}
    comparisons: list[tuple[str, RunResult, dict[str, Any], dict[str, Any]]] = []

    for run in ordered:
        summary = run.metrics.get(metric) if metric else None
        summary = summary or run.primary()
        if run.model == base.model:
            lines.append(
                f"| `{run.model}` | {_fmt(summary)} | {_ci(summary)} | — | — | — | baseline |"
            )
            continue

        shared = sorted(
            set(correctness[run.model]) & set(correctness[base.model]),
            key=lambda uid: uid,
        )
        if not shared:
            lines.append(
                f"| `{run.model}` | {_fmt(summary)} | {_ci(summary)} | — | — | — | "
                "no shared items |"
            )
            continue

        a = [correctness[run.model][uid] for uid in shared]
        b = [correctness[base.model][uid] for uid in shared]
        test = mcnemar_test(a, b)
        boot = paired_bootstrap_delta(a, b)
        comparisons.append((run.model, run, test, boot))

    # Correct across the whole family before deciding what to call significant.
    adjusted = holm_bonferroni({model: t["p_value"] for model, _, t, _ in comparisons})
    boot_by_model = {model: boot for model, _, _, boot in comparisons}
    test_by_model = {model: test for model, _, test, _ in comparisons}

    for run in ordered:
        if run.model == base.model or run.model not in test_by_model:
            continue
        test = test_by_model[run.model]
        boot = boot_by_model[run.model]
        adj = adjusted["adjusted"].get(run.model, 1.0)
        significant = adjusted["significant"].get(run.model, False)
        delta = test["delta"]
        verdict = _verdict(delta, significant)
        lines.append(
            f"| `{run.model}` | {_fmt(summary_of(run, metric))} | "
            f"{_ci(summary_of(run, metric))} | {delta * 100:+.1f} pts | "
            f"[{_pct(boot['ci_low'])}, {_pct(boot['ci_high'])}] | {adj:.3f} | {verdict} |"
        )

    lines.append("")
    lines.append(
        f"Correction applied across {adjusted['n_comparisons']} comparison(s) at "
        f"alpha={adjusted['alpha']}."
    )
    lines.append("")

    disagreements = _disagreement_examples(ordered, base)
    if disagreements:
        lines.append("### Where the models disagree most")
        lines.append("")
        lines.append("Categories where this run is furthest from the baseline.")
        lines.append("")
        lines.append("| Category | Metric | Baseline | This run | n |")
        lines.append("| --- | --- | --- | --- | --- |")
        for row in disagreements:
            lines.append(row)
        lines.append("")
    return lines


def summary_of(run: RunResult, metric: str | None) -> Any:
    """The summary a run should be ranked and reported by."""
    return (run.metrics.get(metric) if metric else None) or run.primary()


def _verdict(delta: float, significant: bool) -> str:
    """Turn a test outcome into a plain-language verdict."""
    if not significant:
        return "no detectable difference"
    if abs(delta) < PRACTICAL_THRESHOLD:
        return "significant but negligible"
    return "significant" if delta > 0 else "significantly worse"


def _disagreement_examples(runs: list[RunResult], base: RunResult) -> list[str]:
    """Per-category rows comparing each run to the baseline, worst first."""
    rows: list[str] = []
    for run in runs:
        if run.model == base.model:
            continue
        for category, metrics in run.by_category.items():
            base_metrics = base.by_category.get(category)
            if not base_metrics:
                continue
            key = "accuracy" if "accuracy" in metrics else next(iter(metrics), None)
            if not key or key not in base_metrics:
                continue
            mine = metrics[key].value
            theirs = base_metrics[key].value
            if mine is None or theirs is None:
                continue
            rows.append((abs(mine - theirs), category, key, theirs, mine, metrics[key].n))

    rows.sort(reverse=True)
    out = [
        f"| {cat} | {key} | {_pct(theirs)} | {_pct(mine)} | {n} |"
        for _, cat, key, theirs, mine, n in rows[:6]
    ]
    return out


def _fmt(summary: Any) -> str:
    if summary is None or getattr(summary, "value", None) is None:
        return "-"
    return f"{summary.value * 100:.1f}"


def _ci(summary: Any) -> str:
    if summary is None or summary.ci_low is None or summary.ci_high is None:
        return "-"
    return f"[{summary.ci_low * 100:.1f}, {summary.ci_high * 100:.1f}]"


def _pct(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100:.1f}"


def rank_runs(results: list[RunResult], metric: str | None = None) -> list[RunResult]:
    """Sort runs by a metric, highest first, grouped by task and dataset.

    Ties are broken by cost, which is a real tiebreak: between two models that
    score identically, the cheaper one is the better default.
    """
    def key(run: RunResult) -> tuple[float, float]:
        summary = summary_of(run, metric)
        value = summary.value if summary and summary.value is not None else -1.0
        return (-value, run.counts.get("est_cost_usd", 0.0))

    return sorted(results, key=key)
