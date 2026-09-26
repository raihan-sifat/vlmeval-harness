"""Object-existence probes: the POPE-style hallucination task.

Each item asks whether a named object is present. Half the objects are in the
image and half are not, which turns the task into a probe of a specific failure
mode: a model that recognises the queried object in a familiar context will
often assert it is there even when it is not. That is object hallucination, and
it is the failure that makes VLMs unsafe to build on -- confidently wrong
answers propagate into whatever system consumes them.

The dataset carries a ``category`` per item that is preserved all the way into
the report. The three POPE splits differ only in how the *absent* objects are
chosen, and they are not equally hard:

* ``random`` -- absent objects sampled uniformly. Easy; absence carries no cue.
* ``popular`` -- absent objects are the most frequent objects in the corpus, so
  the prior strongly favours "yes". Catches frequency bias.
* ``adversarial`` -- absent objects are the ones most often hallucinated given
  the present objects. Hardest, and the closest to a real deployment.

Reporting a single pooled number hides all of that, so per-category metrics are
computed by default and the pool is shown only alongside them.
"""

from __future__ import annotations

from typing import Any

from ..metrics.binary import yes_no_metrics
from ..parsing import extract_yes_no
from ..prompts import render_binary
from ..types import Example, MetricSummary
from .base import ItemOutcome, Query, Task

#: Accepted gold spellings, mapped onto the canonical polarity.
_POSITIVE = {
    "yes": "yes", "y": "yes", "true": "yes", "1": "yes",
    "present": "yes", "correct": "yes", "t": "yes",
}
_NEGATIVE = {
    "no": "no", "n": "no", "false": "no", "0": "no",
    "absent": "no", "incorrect": "no", "f": "no",
}


class POPETask(Task):
    """Binary "is X in the image?" questions.

    Args:
        template: Prompt template name; defaults to the single-word yes/no
            contract.
        max_tokens: Reply cap. Small on purpose: the model has one word to say,
            and a long reply only creates parsing ambiguity and cost.
        require_single_word: Adds an explicit "no explanation" clause. It
            reduces unparsed replies at a small risk of pushing some models
            toward bare guesses, so it is a knob rather than a default.

    Scoring counts an unparsable reply as incorrect and reports
    ``unparsed_rate`` alongside, so abstentions cannot quietly raise a score.
    """

    name = "pope"
    kind = "binary"
    default_max_tokens = 4
    description = "Object-existence probes for hallucination (POPE-style)"

    def __init__(
        self,
        template: str | None = None,
        require_single_word: bool = True,
        **options: Any,
    ) -> None:
        super().__init__(template=template, **options)
        self._template = template or "binary_yesno"
        self.require_single_word = require_single_word

    def extra_options(self) -> dict[str, Any]:
        return {"require_single_word": self.require_single_word}

    def queries(self, example: Example) -> list[Query]:
        if example.choices:
            raise ValueError(
                f"POPETask expects yes/no items without options, but {example.uid!r} "
                f"has {len(example.choices)}. Use MCQTask instead."
            )
        return [
            Query(
                prompt=render_binary(example.question, template=self._template),
                variant="base",
                gold=self._normalize(example.label),
                meta={"category": example.category},
            )
        ]

    def parse(self, text: str, query: Query) -> str | None:
        return extract_yes_no(text)

    @staticmethod
    def _normalize(label: str | None) -> str | None:
        """Map a gold label onto ``"yes"`` / ``"no"``.

        Accepts the many spellings real datasets use (``"Yes"``, ``"true"``,
        ``1``, ``"absent"``). An unrecognised label is a dataset bug, and
        returning ``None`` surfaces it in the unparsed count rather than
        scoring it as a miss.
        """
        if label is None:
            return None
        value = str(label).strip().lower()
        return _POSITIVE.get(value) or _NEGATIVE.get(value)

    def score(self, outcomes: list[ItemOutcome]) -> dict[str, MetricSummary]:
        preds = [o.decoded.get("base") for o in outcomes]
        golds = [o.result.gold for o in outcomes]
        metrics = yes_no_metrics(preds, golds)

        # Per-polarity accuracy: the pooled figure is dominated by whichever
        # polarity is more common, and a model can look fine overall while
        # answering "yes" to everything.
        for polarity in ("yes", "no"):
            flags: list[bool] = []
            for pred, gold in zip(preds, golds, strict=True):
                if gold != polarity:
                    continue
                flags.append(pred == polarity)
            if flags:
                metrics[f"recall_{polarity}"] = MetricSummary(
                    name=f"recall_{polarity}",
                    value=sum(1.0 for f in flags if f) / len(flags),
                    n=len(flags),
                )
        return metrics

    def by_category(
        self, outcomes: list[ItemOutcome]
    ) -> dict[str, dict[str, MetricSummary]]:
        """Per-split metrics, defaulting to the pooled set when unlabelled.

        Items with no category still need a number, so they land under
        ``"all"`` rather than being dropped from the report. The inherited
        implementation already does that, so this only documents the intent.
        """
        return super().by_category(outcomes)


#: Alias matching the metric family name used in the literature.
HallucinationTask = POPETask
