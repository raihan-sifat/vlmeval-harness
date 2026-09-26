"""Figures.

Two plots, both chosen because they show something a table cannot:

* **Accuracy with confidence intervals.** A bar chart of point estimates is
  actively misleading when the differences are inside the noise, so every bar
  carries its bootstrap interval and overlapping intervals are visible as such.
* **Accuracy against cost.** Ranking models by score alone hides that the top
  two rows may differ by 20x in price. Plotting the trade-off makes the
  efficient frontier obvious: the point where a more expensive model stops
  buying a meaningful gain.

Matplotlib is an optional dependency; the module imports it lazily and the CLI
reports the install command rather than raising.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .types import RunResult

#: Colour-blind-safe qualitative palette (Okabe-Ito), so the figures stay
#: readable for the most common form of colour vision deficiency.
PALETTE = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
    "#000000",
)


def _pyplot() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ImportError(
            "Plotting needs matplotlib: pip install -e '.[analysis]'"
        ) from exc
    return plt


def _group(results: list[RunResult], metric: str | None) -> dict[tuple[str, str], list[RunResult]]:
    groups: dict[tuple[str, str], list[RunResult]] = {}
    for result in results:
        key = (result.task, result.dataset)
        groups.setdefault(key, []).append(result)
    return groups


def plot_accuracy_bars(
    results: list[RunResult],
    out_dir: str | Path,
    metric: str | None = None,
) -> list[Path]:
    """One bar chart per (task, dataset), with 95% intervals on every bar."""
    plt = _pyplot()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for (task, dataset), runs in sorted(_group(results, metric).items()):
        scored = []
        for run in runs:
            summary = run.metrics.get(metric) if metric else None
            summary = summary or run.primary()
            if summary and summary.value is not None:
                scored.append((run, summary))
        if not scored:
            continue
        scored.sort(key=lambda rs: rs[1].value)

        labels = [r.model for r, _ in scored]
        values = [s.value for _, s in scored]
        lows = [s.ci_low if s.ci_low is not None else s.value for _, s in scored]
        highs = [s.ci_high if s.ci_high is not None else s.value for _, s in scored]
        errors = [
            [v - lo for v, lo in zip(values, lows, strict=True)],
            [hi - v for v, hi in zip(values, highs, strict=True)],
        ]

        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.3), 4.2))
        ax.bar(
            labels,
            values,
            yerr=errors,
            capsize=4,
            color=[PALETTE[i % len(PALETTE)] for i in range(len(labels))],
            edgecolor="white",
        )
        ax.set_ylabel(_metric_label(metric, scored[0][1].name))
        ax.set_title(f"{task} / {dataset}")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.25, linestyle=":")
        ax.set_axisbelow(True)
        # Values are in percent; a formatter avoids "0.725" on a 0-1 axis.
        ax.yaxis.set_major_formatter(lambda v, _pos: f"{v * 100:.0f}%")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        for label, value in zip(ax.get_containers()[0].labels, values, strict=True):
            label.set_text(f"{value * 100:.1f}")
        fig.tight_layout()

        path = out / f"{task}__{dataset}__accuracy.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path)
    return written


def plot_accuracy_vs_cost(
    results: list[RunResult],
    out_dir: str | Path,
    metric: str | None = None,
) -> list[Path]:
    """Score against spend, marking the efficient frontier."""
    plt = _pyplot()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for (task, dataset), runs in sorted(_group(results, metric).items()):
        points: list[tuple[str, float, float]] = []
        for run in runs:
            summary = run.metrics.get(metric) if metric else None
            summary = summary or run.primary()
            cost = float(run.counts.get("est_cost_usd", 0.0) or 0.0)
            if summary and summary.value is not None:
                points.append((run.model, cost, summary.value))
        if len(points) < 2:
            # A single point cannot form a frontier; skip rather than imply one.
            continue
        points.sort(key=lambda p: p[1])

        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for i, (model, cost, value) in enumerate(points):
            ax.scatter(cost, value, s=70, color=PALETTE[i % len(PALETTE)], zorder=3)
            ax.annotate(
                model,
                (cost, value),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=8,
            )

        # Pareto frontier: no cheaper model scores higher.
        best = -1.0
        frontier: list[tuple[float, float]] = []
        for _model, cost, value in points:
            if value > best:
                frontier.append((cost, value))
                best = value
        if len(frontier) > 1:
            ax.plot(
                [p[0] for p in frontier],
                [p[1] for p in frontier],
                linestyle="--",
                color="0.5",
                linewidth=1,
                zorder=1,
                label="efficient frontier",
            )
            ax.legend(fontsize=8, loc="lower right")

        ax.set_xlabel("Estimated cost for this run (USD)")
        ax.set_ylabel(_metric_label(metric, "score"))
        ax.set_title(f"{task} / {dataset}: accuracy vs cost")
        ax.yaxis.set_major_formatter(lambda v, _pos: f"{v * 100:.0f}%")
        ax.grid(alpha=0.25, linestyle=":")
        ax.set_axisbelow(True)
        # Free models sit at exactly zero; a symlog axis keeps them on the plot
        # without letting them compress everything else.
        if any(c == 0.0 for _m, c, _v in points) and any(c > 0 for _m, c, _v in points):
            ax.set_xscale("symlog", linthresh=max(1e-4, min(c for _m, c, _v in points if c > 0) / 10))
        fig.tight_layout()

        path = out / f"{task}__{dataset}__accuracy_vs_cost.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path)
    return written


def plot_category_breakdown(
    results: list[RunResult],
    out_dir: str | Path,
    metric: str | None = None,
) -> list[Path]:
    """Grouped bars of one metric across categories, one series per model."""
    plt = _pyplot()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for (task, dataset), runs in sorted(_group(results, metric).items()):
        with_cats = [r for r in runs if r.by_category]
        if not with_cats:
            continue
        categories = sorted({c for r in with_cats for c in r.by_category})
        if len(categories) < 2:
            continue

        fig, ax = plt.subplots(figsize=(max(6, len(categories) * 1.6), 4.2))
        width = 0.8 / len(with_cats)
        for i, run in enumerate(with_cats):
            values = []
            for category in categories:
                metrics = run.by_category.get(category, {})
                summary = metrics.get(metric) if metric else None
                summary = summary or next(
                    (m for n, m in metrics.items() if n == "accuracy"), None
                )
                values.append(summary.value if summary and summary.value is not None else 0.0)
            offsets = [j - 0.4 + width * (i + 0.5) for j in range(len(categories))]
            ax.bar(
                offsets,
                values,
                width=width,
                label=run.model,
                color=PALETTE[i % len(PALETTE)],
                edgecolor="white",
            )

        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, rotation=20, ha="right")
        ax.set_ylabel(_metric_label(metric, "accuracy"))
        ax.set_title(f"{task} / {dataset}: per-category")
        ax.set_ylim(0, 1.0)
        ax.yaxis.set_major_formatter(lambda v, _pos: f"{v * 100:.0f}%")
        ax.grid(axis="y", alpha=0.25, linestyle=":")
        ax.set_axisbelow(True)
        if len(with_cats) <= 6:
            ax.legend(fontsize=8)
        fig.tight_layout()

        path = out / f"{task}__{dataset}__by_category.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path)
    return written


def plot_all(
    results: list[RunResult],
    out_dir: str | Path = "results/figures",
    metric: str | None = None,
) -> list[Path]:
    """Render every figure that applies to the given runs."""
    paths: list[Path] = []
    for fn in (plot_accuracy_bars, plot_accuracy_vs_cost, plot_category_breakdown):
        try:
            paths.extend(fn(results, out_dir, metric=metric))
        except ImportError:
            raise
        except Exception as exc:  # noqa: BLE001 - one bad figure shouldn't stop the rest
            print(f"warning: {fn.__name__} failed: {exc}")
    return paths


def _metric_label(metric: str | None, fallback: str) -> str:
    return (metric or fallback).replace("_", " ").capitalize()
