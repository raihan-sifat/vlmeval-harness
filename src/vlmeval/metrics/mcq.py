"""Multiple-choice metrics."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from ..stats import bootstrap_ci
from ..types import MetricSummary


def accuracy_metric(
    correct: Sequence[bool | None],
    *,
    name: str = "accuracy",
    n_boot: int = 2000,
    seed: int = 12345,
) -> MetricSummary:
    """Accuracy with unparsable replies counted as incorrect.

    The ``None`` entries -- replies the parser could not read -- score zero and
    stay in the denominator. This is the whole design of the function, so it is
    worth being explicit about why: dropping them instead would mean a model
    answering 50 of 100 items correctly and refusing the other 50 scores 100%,
    not 50%. Abstention would be the optimal strategy, and every benchmark
    would silently reward it. The cost is that a single odd reply format can
    move the headline number, which is why ``accuracy_parsed`` is reported
    alongside for diagnosis.

    Returns:
        A summary whose ``n`` is the number of items attempted and whose
        interval is a percentile bootstrap CI over them.
    """
    if not correct:
        return MetricSummary(name=name, value=None, n=0)
    flags = [bool(c) for c in correct]
    low, high = bootstrap_ci(flags, n_boot=n_boot, seed=seed)
    return MetricSummary(
        name=name,
        value=sum(1.0 for v in flags if v) / len(flags),
        n=len(flags),
        ci_low=low,
        ci_high=high,
    )


def parsed_accuracy_metric(
    correct: Sequence[bool | None],
    *,
    name: str = "accuracy_parsed",
    n_boot: int = 2000,
    seed: int = 12345,
) -> MetricSummary:
    """Accuracy over the items that produced a parsable answer.

    Diagnostic companion to :func:`accuracy_metric`. The gap between this and
    the strict accuracy *is* the unparsed rate, so the pair separates "cannot
    read the image" from "reads it and picks wrong" -- two failures that call for
    completely different fixes.
    """
    flags = [bool(c) for c in correct if c is not None]
    if not flags:
        return MetricSummary(name=name, value=None, n=0)
    low, high = bootstrap_ci(flags, n_boot=n_boot, seed=seed)
    return MetricSummary(
        name=name,
        value=sum(1.0 for v in flags if v) / len(flags),
        n=len(flags),
        ci_low=low,
        ci_high=high,
    )


def choice_distribution(
    predictions: Sequence[str | None],
    num_choices: int,
) -> dict[str, float]:
    """Fraction of answers falling on each letter, plus an unparsed bucket.

    A model that scores 45% but puts 60% of its answers on "A" is not reading
    the options, it is exploiting a position prior. This distribution is what
    makes that visible, and it is why the report prints it next to accuracy
    rather than hiding it in an appendix.

    The last key is ``unparsed``.
    """
    counts: Counter[str] = Counter()
    for pred in predictions:
        if pred:
            counts[pred] += 1
    total = len(predictions) or 1
    dist = {chr(ord("A") + i): counts.get(chr(ord("A") + i), 0) / total for i in range(num_choices)}
    dist["unparsed"] = sum(1 for p in predictions if not p) / total
    return dist


def consistency_metrics(
    per_item: dict[str, dict[str, str | None]],
    golds: dict[str, str],
) -> dict[str, MetricSummary]:
    """Agreement across option-order permutations of the same question.

    Each item is asked several times with the options in different orders, and
    every answer is mapped back to the canonical option index before this runs.
    A model that genuinely reads the image gives the same index every time; one
    that has learned "pick the second-looking option" does not.

    Args:
        per_item: ``uid -> {variant_name: decoded canonical option}``, where a
            decoded value is a letter. ``None`` marks an unparsed variant.
        golds: ``uid -> gold letter``.

    Returns:
        ``consistency``: fraction of items where every variant decoded to the
        same option. Often more informative than accuracy: a model can be
        accurate while unstable, which means part of its accuracy is a coin
        flip that will not transfer to differently-ordered options. Items with
        no parsable variant count as disagreements and stay in the denominator,
        so abstaining everywhere cannot flatter this number either.

        ``variant_accuracy``: accuracy pooled over all (item, variant) pairs,
        counting unparsed variants as incorrect.

        ``order_sensitivity``: canonical accuracy minus pooled variant accuracy
        -- the accuracy the model gives up purely because its options were
        reordered. A model that reads the image scores ~0 here no matter how
        good it is; a model riding a position prior scores high.
    """
    from ..prompts import LETTERS

    canonical_letters = set(LETTERS)
    agree_flags: list[bool] = []
    variant_flags: list[bool] = []
    canonical_flags: list[bool | None] = []

    for uid, variants in per_item.items():
        gold = golds.get(uid)
        decoded = [v for v in variants.values() if v in canonical_letters]
        base = variants.get("base", decoded[0] if decoded else None)
        canonical_flags.append(None if base is None else bool(gold) and base == gold)

        if not decoded:
            # No variant parsed: not evidence of agreement or disagreement.
            agree_flags.append(False)
            continue
        agree_flags.append(len(set(decoded)) == 1)

        if gold in canonical_letters:
            variant_flags.extend(
                (variants[name] == gold) for name in variants
            )

    consistency = accuracy_metric(agree_flags, name="consistency") if agree_flags else MetricSummary(
        name="consistency", value=None, n=0
    )
    variant_acc = (
        accuracy_metric(variant_flags, name="variant_accuracy")
        if variant_flags
        else MetricSummary(name="variant_accuracy", value=None, n=0)
    )
    canonical_acc = accuracy_metric(canonical_flags, name="accuracy")

    out = {
        "consistency": consistency,
        "variant_accuracy": variant_acc,
        "canonical_accuracy": canonical_acc,
    }
    if canonical_acc.value is not None and variant_acc.value is not None:
        out["order_sensitivity"] = MetricSummary(
            name="order_sensitivity",
            value=canonical_acc.value - variant_acc.value,
            n=min(canonical_acc.n, variant_acc.n),
        )
    return out

