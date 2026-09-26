"""Statistics for benchmark results.

A point estimate alone is not a result. On a 500-item benchmark the 95%
confidence interval on accuracy is roughly +/- 4 percentage points, which is wide
enough that two models 3 points apart have not been meaningfully separated. So
every metric in this project ships with an interval, and cross-model claims are
backed by a paired test rather than by comparing two numbers.

The tests are implemented directly rather than pulled from SciPy so that the
core harness has no scientific-computing dependency; :mod:`vlmeval.stats` is
importable with nothing but Pillow and PyYAML installed. SciPy, when present, is
used only to cross-check these implementations in the test suite.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Any

DEFAULT_BOOTSTRAP = 2000
DEFAULT_SEED = 12345


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean, or ``nan`` for an empty sequence."""
    return sum(values) / len(values) if values else float("nan")


def percentile(sorted_values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated percentile of an already-sorted sequence.

    Args:
        sorted_values: Values in ascending order.
        q: Target quantile in [0, 1]. Values outside that range are clamped to
            the nearest endpoint rather than raising, because a quantile that
            overshoots slightly should produce a boundary value and not take
            down a report that is otherwise complete.

    Returns:
        The interpolated quantile, or ``None`` for an empty sequence. ``None`` is
        deliberate rather than ``nan``: a NaN reaching a result file serialises
        as the bare token ``NaN``, which is not valid JSON and is rejected by
        ``JSON.parse``, taking the whole report down with it.
    """
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    q = min(1.0, max(0.0, float(q)))
    pos = q * (len(sorted_values) - 1)
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return float(sorted_values[int(pos)])
    frac = pos - low
    return float(sorted_values[low] * (1 - frac) + sorted_values[high] * frac)


def bootstrap_ci(
    values: Sequence[float | bool | None],
    *,
    n_boot: int = DEFAULT_BOOTSTRAP,
    alpha: float = 0.05,
    seed: int = DEFAULT_SEED,
) -> tuple[float | None, float | None]:
    """Percentile bootstrap confidence interval for a mean.

    Items are resampled with replacement, which is the right model here: the
    items are the random unit, and the question is "if we drew a different
    sample of items from this benchmark, how much would the mean move?"

    ``None`` entries (unparseable answers, failed calls) are dropped, so the
    interval is over the items that produced a usable answer. That is why the
    report always pairs the interval with the unparsed count -- an interval
    computed over 60% of the items needs that caveat to be read correctly.

    The seed is fixed by default so that re-running a report reproduces the
    same interval; comparison between two runs then reflects a change in the
    model, not in the resampling.

    Args:
        values: Per-item values. Bools are coerced to 0/1.
        n_boot: Number of resamples. 2000 gives interval endpoints stable to
            about +/- 0.5 percentage points, well below sampling noise.
        alpha: Two-sided significance level; 0.05 gives a 95% interval.
        seed: RNG seed.

    Returns:
        ``(low, high)``, or ``(None, None)`` if there is nothing to resample.
    """
    data = [1.0 if v is True else 0.0 if v is False else float(v) for v in values if v is not None]
    n = len(data)
    if n == 0:
        return (None, None)
    if n == 1:
        return (data[0], data[0])

    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_boot):
        total = 0.0
        for _ in range(n):
            total += data[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    return (percentile(means, alpha / 2), percentile(means, 1 - alpha / 2))


def mcnemar_test(
    correct_a: Sequence[bool | None],
    correct_b: Sequence[bool | None],
) -> dict[str, Any]:
    """Exact McNemar test for two models on the same items.

    The right test for this comparison. Both models answer the *same* items, so
    their outcomes are paired, and the only informative cells of the contingency
    table are the discordant ones: items A got right and B got wrong, and vice
    versa. An unpaired two-proportion test throws that pairing away and is
    markedly less powerful at the small accuracy gaps that model comparisons
    actually turn on.

    Uses the exact binomial (sign) test on the discordant pairs rather than the
    chi-square approximation, because the discordant count is usually small and
    the approximation is unreliable there.

    Args:
        correct_a: Per-item correctness for model A, in a fixed item order.
        correct_b: The same for model B.

    Returns:
        Dict with the discordant counts, the estimated accuracy difference
        (A minus B), the two-sided p-value, and `significant` at alpha=0.05.
    """
    if len(correct_a) != len(correct_b):
        raise ValueError(
            f"paired comparison needs equal lengths, got {len(correct_a)} and {len(correct_b)}"
        )

    both = only_a = only_b = neither = 0
    for a, b in zip(correct_a, correct_b, strict=True):
        a_ok, b_ok = bool(a), bool(b)
        if a_ok and b_ok:
            both += 1
        elif a_ok:
            only_a += 1
        elif b_ok:
            only_b += 1
        else:
            neither += 1

    n_discordant = only_a + only_b
    if n_discordant == 0:
        p_value = 1.0
    else:
        # P(X >= max(only_a, only_b)) * 2 under Binomial(n_discordant, 0.5).
        tail = sum(_binomial_pmf(k, n_discordant, 0.5) for k in range(max(only_a, only_b), n_discordant + 1))
        p_value = min(1.0, 2 * tail)

    n_shared = both + only_a + only_b + neither
    delta = (only_a - only_b) / n_shared if n_shared else 0.0
    return {
        "n_paired": n_shared,
        "both_correct": both,
        "only_a": only_a,
        "only_b": only_b,
        "neither": neither,
        "delta": delta,
        "p_value": p_value,
        "significant": p_value < 0.05,
    }


def _binomial_pmf(k: int, n: int, p: float) -> float:
    """Binomial probability mass, computed with logs to avoid overflow."""
    if k < 0 or k > n:
        return 0.0
    log_coeff = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    return math.exp(log_coeff + k * math.log(p) + (n - k) * math.log(1 - p))


def paired_bootstrap_delta(
    correct_a: Sequence[bool | None],
    correct_b: Sequence[bool | None],
    *,
    n_boot: int = DEFAULT_BOOTSTRAP,
    alpha: float = 0.05,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Bootstrap the accuracy difference between two models on shared items.

    Complements McNemar: the test answers "is there a difference?", this
    answers "how big is it, and how precisely do we know?". Resampling is done
    over *item indices* and both models are evaluated on each resample, which
    preserves the pairing and therefore keeps the interval narrow.
    """
    if len(correct_a) != len(correct_b):
        raise ValueError("paired comparison needs equal lengths")

    pairs = [
        (1.0 if bool(a) else 0.0, 1.0 if bool(b) else 0.0)
        for a, b in zip(correct_a, correct_b, strict=True)
        if a is not None and b is not None
    ]
    n = len(pairs)
    if n == 0:
        return {"delta": None, "ci_low": None, "ci_high": None, "n": 0}

    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(n_boot):
        total = 0.0
        for _ in range(n):
            a, b = pairs[rng.randrange(n)]
            total += a - b
        deltas.append(total / n)
    deltas.sort()

    observed = mean([a - b for a, b in pairs])
    return {
        "delta": observed,
        "ci_low": percentile(deltas, alpha / 2),
        "ci_high": percentile(deltas, 1 - alpha / 2),
        "n": n,
    }


def holm_bonferroni(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, Any]:
    """Holm-Bonferroni step-down correction for a family of comparisons.

    Comparing a dozen models pairwise and reporting the raw p-values invites the
    reader to accept the smallest one by chance. This adjusts the family so a
    claim of "significantly better than everything else" has to survive the
    number of comparisons actually made.

    Returns the per-comparison raw and adjusted p-values, and whether each
    comparison survives at `alpha`.
    """
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    rejected: dict[str, bool] = {}
    running_max = 0.0
    for rank, (key, p) in enumerate(items):
        # Step-down: each threshold shrinks with rank, and the running maximum
        # enforces monotonicity of the adjusted values.
        running_max = max(running_max, min(1.0, p * (m - rank)))
        adjusted[key] = running_max
        rejected[key] = running_max < alpha
    return {
        "adjusted": adjusted,
        "significant": rejected,
        "n_comparisons": m,
        "alpha": alpha,
    }
