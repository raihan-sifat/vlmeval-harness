"""Yes/no (binary) metrics, following the POPE reporting convention.

Object-existence probes are scored on more than accuracy because accuracy is
nearly useless on a balanced-ish set: a model that answers "no" to everything
scores well whenever the negatives dominate, and scores terribly whenever they
do not. Precision, recall and the yes-ratio are what distinguish a model that
has learned to detect objects from one that has learned a safe default.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..stats import bootstrap_ci
from ..types import MetricSummary

POSITIVE = "yes"
NEGATIVE = "no"


def _confusion(
    predictions: Sequence[str | None], golds: Sequence[str | None]
) -> tuple[int, int, int, int, int]:
    """Return ``(tp, fp, tn, fn, unparsed)`` with "yes" as the positive class."""
    tp = fp = tn = fn = unparsed = 0
    for pred, gold in zip(predictions, golds, strict=True):
        if pred is None:
            unparsed += 1
            continue
        if pred == POSITIVE and gold == POSITIVE:
            tp += 1
        elif pred == POSITIVE and gold == NEGATIVE:
            fp += 1
        elif pred == NEGATIVE and gold == NEGATIVE:
            tn += 1
        elif pred == NEGATIVE and gold == POSITIVE:
            fn += 1
        else:
            unparsed += 1
    return tp, fp, tn, fn, unparsed


def _f1(tp: int, fp: int, fn: int) -> float | None:
    """F1 for the positive class, or ``None`` when it is undefined.

    Undefined when there are no predicted *or* no actual positives: precision
    and recall are then 0/0, and reporting 0.0 would read as "the model got
    every positive wrong" when the truth is "there was nothing to get".
    """
    denom = 2 * tp + fp + fn
    if denom == 0:
        return None
    return 2 * tp / denom


def yes_no_metrics(
    predictions: Sequence[str | None],
    golds: Sequence[str | None],
    *,
    n_boot: int = 2000,
    seed: int = 12345,
) -> dict[str, MetricSummary]:
    """Accuracy, precision, recall, F1, specificity and the yes-ratio.

    Args:
        predictions: Per-item parsed verdicts (``"yes"``, ``"no"`` or ``None``).
        golds: Per-item gold verdicts.
        n_boot: Bootstrap resamples for the intervals.
        seed: Bootstrap seed, fixed so reports are reproducible.

    Returns:
        ``yes_ratio`` is the fraction of *all* items answered yes, including
        unparsed ones excluded elsewhere. A value far from the gold positive
        rate indicates a response bias rather than a perception failure, and is
        the single most diagnostic number in a POPE-style run.
    """
    tp, fp, tn, fn, unparsed = _confusion(predictions, golds)
    scored = tp + fp + tn + fn

    # Unparsed items count as wrong for the rate metrics. Crediting a refusal
    # as neutral would let a model raise its score by abstaining, and the
    # abstention rate is reported separately so nothing is hidden.
    flags: list[bool] = []
    idx = 0
    for pred in predictions:
        gold = golds[idx]
        idx += 1
        flags.append(pred is not None and pred == gold)

    n_total = len(predictions) or 1
    yes_ratio = sum(1 for p in predictions if p == POSITIVE) / n_total
    gold_yes_ratio = (
        sum(1 for g in golds if g == POSITIVE) / (len(golds) or 1) if golds else None
    )

    out: dict[str, MetricSummary] = {
        "accuracy": _summary("accuracy", flags, n_boot, seed),
        "precision": MetricSummary(
            name="precision",
            value=(tp / (tp + fp)) if (tp + fp) else None,
            n=tp + fp,
        ),
        "recall": MetricSummary(name="recall", value=(tp / (tp + fn)) if (tp + fn) else None, n=tp + fn),
        "f1": MetricSummary(name="f1", value=_f1(tp, fp, fn), n=tp + fn + fp),
        "specificity": MetricSummary(
            name="specificity", value=(tn / (tn + fp)) if (tn + fp) else None, n=tn + fp
        ),
        "yes_ratio": MetricSummary(name="yes_ratio", value=yes_ratio, n=n_total),
        "unparsed_rate": MetricSummary(name="unparsed_rate", value=unparsed / n_total, n=n_total),
    }
    if gold_yes_ratio is not None:
        out["gold_yes_ratio"] = MetricSummary(
            name="gold_yes_ratio", value=gold_yes_ratio, n=len(golds)
        )
        out["yes_ratio_bias"] = MetricSummary(
            name="yes_ratio_bias", value=yes_ratio - gold_yes_ratio, n=n_total
        )
    if scored != n_total:
        out["n_scored"] = MetricSummary(name="n_scored", value=float(scored), n=scored)
    return out


def _summary(name: str, flags: Sequence[bool], n_boot: int, seed: int) -> MetricSummary:
    if not flags:
        return MetricSummary(name=name, value=None, n=0)
    low, high = bootstrap_ci(flags, n_boot=n_boot, seed=seed)
    return MetricSummary(
        name=name,
        value=sum(1.0 for f in flags if f) / len(flags),
        n=len(flags),
        ci_low=low,
        ci_high=high,
    )
