"""Multiple-choice visual question answering.

The task covers the MMMU / MMBench / MME style of question: an image, a
question, and lettered options, scored by whether the model picks the right
letter.

Beyond plain accuracy it measures **option-order sensitivity**. Each item can
optionally be asked several times with the options permuted, and every answer is
mapped back to the canonical option index. Two models with identical accuracy
can be separated by this: the one that answers the same way regardless of
option order is reading the image, and the one that does not is partly
exploiting a position prior. That distinction decides whether a score will
survive a benchmark that happens to order its options differently.

The canonical (unpermuted) variant is what ``accuracy`` reports, so numbers stay
comparable with published results. Permutation variants only add diagnostics.
"""

from __future__ import annotations

import random
from typing import Any

from ..metrics.mcq import (
    accuracy_metric,
    choice_distribution,
    consistency_metrics,
    parsed_accuracy_metric,
)
from ..parsing import extract_choice
from ..prompts import LETTERS, render_mcq
from ..types import Example, MetricSummary
from .base import ItemOutcome, Query, Task


class MCQTask(Task):
    """Lettered multiple-choice questions.

    Args:
        num_permutations: How many *extra* option orders to ask beyond the
            canonical one. ``0`` (default) means one request per item, which
            is the cheapest setting and matches standard benchmark protocol.
            ``2`` triples the API cost and buys the bias diagnostics.
        seed: RNG seed for the permutations. Fixed by default so two models
            are compared under identical orders -- otherwise a difference in
            accuracy could be a difference in permutations.
        template: Prompt template name; defaults to the standard letter format.
        shuffle_choices: Shuffle the options in the canonical presentation too.
            Off by default: published datasets have a meaningful original
            order, and silently reordering makes results incomparable.
        max_choices: Sanity bound on option count.
    """

    name = "mcq"
    kind = "mcq"
    default_max_tokens = 8
    description = "Multiple-choice visual question answering (lettered options)"

    def __init__(
        self,
        num_permutations: int = 0,
        seed: int = 12345,
        template: str | None = None,
        shuffle_choices: bool = False,
        max_choices: int = 10,
        **options: Any,
    ) -> None:
        super().__init__(template=template, **options)
        self.num_permutations = max(0, int(num_permutations))
        self.seed = seed
        self.shuffle_choices = shuffle_choices
        self.max_choices = max_choices
        self._template = template or "mcq_letter"

    def extra_options(self) -> dict[str, Any]:
        return {
            "num_permutations": self.num_permutations,
            "seed": self.seed,
            "shuffle_choices": self.shuffle_choices,
            "max_choices": self.max_choices,
        }

    def _permutations(self, n: int, uid: str = "") -> list[list[int]]:
        """Canonical order first, then `num_permutations` seeded shuffles.

        Deduplicated: with 3 options there are only 6 orders, so asking for 5
        extra permutations would re-ask identical questions and pay for the
        privilege.

        The shuffles are seeded per item, not once for the whole run. A single
        global seed hands every item the *same* set of orders, which quietly
        biases the order audit: the gold answer then lands in the same display
        slot for every item, so "the model prefers slot C" and "the gold
        happened to sit in slot C twice" become indistinguishable, and the
        pooled variant accuracy stops being an unbiased estimate of how much a
        model leans on position. Varying the orders per item is what makes that
        number mean something.
        """
        orders = [list(range(n))]
        if self.num_permutations <= 0:
            return orders
        rng = random.Random(f"{self.seed}:{uid}")
        seen = {tuple(orders[0])}
        attempts = 0
        while len(orders) <= self.num_permutations and attempts < 50:
            attempts += 1
            candidate = list(range(n))
            rng.shuffle(candidate)
            key = tuple(candidate)
            if key in seen:
                continue
            seen.add(key)
            orders.append(candidate)
        return orders

    def queries(self, example: Example) -> list[Query]:
        if not example.choices:
            raise ValueError(
                f"MCQTask requires answer options, but example {example.uid!r} has none. "
                "Use a binary task for yes/no items, or check the dataset adapter."
            )
        if len(example.choices) > self.max_choices:
            raise ValueError(
                f"example {example.uid!r} has {len(example.choices)} options, "
                f"above the supported maximum of {self.max_choices}"
            )

        n = len(example.choices)
        gold_index = self._gold_index(example, n)
        # Indices shown at each display slot. `gold_slot` is where the correct
        # option ends up, which is what we map an answer back through.
        orders = self._permutations(n, example.uid)
        if self.shuffle_choices and self.num_permutations == 0:
            rng = random.Random(f"{self.seed}:{example.uid}")
            shuffled = list(range(n))
            rng.shuffle(shuffled)
            orders = [shuffled]

        out: list[Query] = []
        for i, order in enumerate(orders):
            gold_slot = order.index(gold_index) if gold_index is not None else None
            # `gold` is canonical on purpose. `decode` maps a displayed letter
            # back to canonical, so a canonical gold makes every variant
            # comparable; a display-relative gold would score `perm` variants
            # against the wrong key whenever the correct option moved.
            display_gold = LETTERS[gold_slot] if gold_slot is not None else None
            canonical_gold = LETTERS[gold_index] if gold_index is not None else None
            # letter -> canonical letter, so every variant is scored identically
            decode = {LETTERS[slot]: LETTERS[canonical] for slot, canonical in enumerate(order)}
            out.append(
                Query(
                    prompt=render_mcq(
                        example.question, example.choices, order=order, template=self._template
                    ),
                    variant="base" if i == 0 else f"perm{i}",
                    decode=decode,
                    gold=canonical_gold,
                    meta={
                        "order": order,
                        "num_choices": n,
                        "display_gold": display_gold,
                        "gold_slot": gold_slot,
                    },
                )
            )
        return out

    def _gold_index(self, example: Example, n: int) -> int | None:
        """Locate the gold option as an index into `example.choices`."""
        label = (example.label or "").strip()
        if not label:
            return None
        if len(label) == 1 and label.upper() in LETTERS[:n]:
            return LETTERS.index(label.upper())
        # Tolerate a gold answer written out as the option text.
        for i, choice in enumerate(example.choices):
            if label.strip().lower() == str(choice).strip().lower():
                return i
        return None

    def parse(self, text: str, query: Query) -> str | None:
        n = int(query.meta.get("num_choices", 4))
        return extract_choice(text, n)

    def score(self, outcomes: list[ItemOutcome]) -> dict[str, MetricSummary]:
        """Accuracy, unparsed rate, answer distribution and order diagnostics.

        ``accuracy`` covers the canonical variant only. ``n_variants`` reports
        how many requests per item were issued, so a reader can tell whether the
        order diagnostics were actually measured or are absent.
        """
        canonical = [o for o in outcomes if "base" in o.decoded]
        if not canonical:
            canonical = outcomes

        flags: list[bool | None] = []
        preds: list[str | None] = []
        for outcome in canonical:
            pred = outcome.decoded.get("base")
            gold = outcome.result.gold
            preds.append(pred)
            flags.append(None if pred is None else bool(gold) and pred == gold)

        n_choices = max(
            (int(o.result.meta.get("num_choices", 4)) for o in canonical), default=4
        )
        metrics: dict[str, MetricSummary] = {
            "accuracy": accuracy_metric(flags, name="accuracy"),
            "accuracy_parsed": parsed_accuracy_metric(flags, name="accuracy_parsed"),
            "unparsed_rate": MetricSummary(
                name="unparsed_rate",
                value=sum(1 for f in flags if f is None) / (len(flags) or 1),
                n=len(flags),
            ),
            "mean_latency_s": _mean_latency(canonical),
        }

        dist = choice_distribution(preds, n_choices)
        for letter, frac in dist.items():
            metrics[f"pick_rate_{letter}"] = MetricSummary(
                name=f"pick_rate_{letter}", value=frac, n=len(preds)
            )

        per_item = {o.result.uid: o.decoded for o in outcomes}
        golds = {o.result.uid: o.result.gold for o in outcomes}
        n_variants = max((len(o.decoded) for o in outcomes), default=1)
        metrics.update(consistency_metrics(per_item, golds))
        metrics["n_variants"] = MetricSummary(
            name="n_variants", value=float(n_variants), n=len(outcomes)
        )
        return metrics


def _mean_latency(outcomes: list[ItemOutcome]) -> MetricSummary:
    latencies = [o.result.response.latency_s for o in outcomes if o.result.response.latency_s]
    if not latencies:
        return MetricSummary(name="mean_latency_s", value=None, n=0)
    latencies.sort()
    from ..stats import percentile

    return MetricSummary(
        name="mean_latency_s",
        value=sum(latencies) / len(latencies),
        n=len(latencies),
        ci_low=percentile(latencies, 0.5),
        ci_high=percentile(latencies, 0.95),
    )


#: Convenience alias so configs can say `letter_answer` instead of `mcq`.
LetterAnswerTask = MCQTask
