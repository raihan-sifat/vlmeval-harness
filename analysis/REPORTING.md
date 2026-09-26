# Reading a vlmeval report

A generated report is easy to misread. This note is the checklist for turning one
into a claim you would be willing to defend, and it is ordered by how often each
trap actually fires.

## 1. Read the intervals before the ranking

Every headline metric ships with a 95% bootstrap confidence interval over items.
A leaderboard sorted by point estimate is a leaderboard sorted by noise whenever
two intervals overlap.

```
| `strong` | 95.2 [90.5, 98.8] |
| `demo`   | 76.2 [66.7, 84.5] |
| `weak`   | 34.5 [23.8, 45.2] |
```

`strong` over `demo` is a real gap; the intervals do not come close. `demo` over
`weak` is also real. But two models at 76.2 [66.7, 84.5] and 78.0 [68.8, 85.1]
are not distinguishable on this dataset, and listing them as a ranking implies
otherwise.

The interval reflects item sampling only. It does not capture prompt phrasing,
option order, or the seed -- those are separate questions, and the report answers
some of them below.

## 2. Check `unparsed_rate` before celebrating an accuracy

`accuracy` counts an unparseable reply as **wrong**. That is the conservative
choice and it is the one to quote. The report also gives you the alternatives so
you can see what a model is doing:

| Metric | Meaning |
| --- | --- |
| `accuracy` | Unparsed counts as wrong. The number to quote. |
| `accuracy_parsed` | Unparsed items excluded. Always >= `accuracy`. |
| `unparsed_rate` | Fraction of replies that did not parse. |

If `accuracy_parsed` is much higher than `accuracy`, the model can answer when it
tries and is choosing not to emit a bare letter. That is a prompting or
instruction-following problem, not a capability ceiling, and the fix is different:
a stricter instruction, a different template, or `--style grounded`.

If `unparsed_rate` is high *and* the model is a paid one, check a few raw replies
in `results.jsonl` before blaming the model. A provider silently switching to a
refusal, a safety filter, or a reasoning preamble will all land here.

## 3. Take order effects seriously

With `--permutations N`, each item is re-asked with N extra option orders.

| Metric | Reading |
| --- | --- |
| `consistency` | Fraction of items where every order agreed. |
| `variant_accuracy` | Accuracy across the permuted prompts. |
| `order_sensitivity` | `accuracy` minus `variant_accuracy`, in points. |

- `consistency` near `1.0` with accuracy near chance means the model is answering
  from position, not from the image. `echo:blind` is built to look exactly like
  this.
- Large `order_sensitivity` means the headline number is partly an artifact of
  which letter the correct answer happened to land on. Quote `variant_accuracy`
  alongside, or raise `--permutations` and re-run.
- `order_sensitivity` slightly *negative* is normal and not a bug: it means the
  model did marginally better when the options moved.

Permutation only covers option order. It says nothing about robustness to
reworded questions or a different answer format, so do not present a low
`order_sensitivity` as general robustness.

## 4. On POPE, accuracy alone hides the interesting failure

Binary existence probes have a degenerate solution: answer "yes" to everything.
On a balanced set that scores 50%, which looks like a mediocre model rather than
a broken one.

```
  accuracy: 50.0 [30.0, 70.0]
  precision=0.500 recall=1.000 f1=0.667 yes_ratio=1.000
```

That is total hallucination: recall 1.0, specificity 0.0, `yes_ratio` 1.0. The
model asserts every object is present. Report the vector, not the scalar:

- `recall` high with `specificity` at 0.0 -- says yes to everything.
- `yes_ratio` far from the dataset's base rate -- a strong prior, and the number
  to compare against the base rate rather than against 0.5.
- Per-split accuracy -- adversarial splits exist to break the prior, and a model
  that only degrades there is exploiting the frequency of the object rather than
  looking at the image.

`echo:hallucinator` reproduces this signature on demand, which is how you check
that your own analysis would notice it.

## 5. Use the paired test, and believe its verdict column

`vlmeval compare` runs McNemar's test on shared items, so it only looks at the
items where the two runs disagree. That is far more sensitive than comparing two
independent proportions, and it is the only reason to believe a small gap.

- Read the **verdict**, not the p-value. "p = 0.049" is a statement about a
  threshold, not about magnitude.
- Read the **CI of the delta** as well. A significant delta of +0.4 points with a
  CI spanning [-2.0, +2.8] is not a finding.
- Comparisons are Holm-Bonferroni corrected across the family. Without the
  correction, a 20-model sweep finds a "significant" difference by chance roughly
  two times out of three. The corrected column already accounts for this.
- `p > 0.05` means *underpowered*, not *equivalent*. A 500-item run cannot
  distinguish two models that differ by 2 points. That is a statement about the
  dataset, and the fix is more items, not a different test.

## 6. Confirm the runs are actually comparable

Check `counts` in `metrics.json` before comparing two runs:

- `n_requests` equal -- otherwise you are comparing different item sets and the
  pairing is not doing what you think.
- `failed` zero. Failures are recorded, not dropped, but a run with failures has
  a lower effective `n` than its request count suggests.
- `stopped_reason` empty. A budget stop truncates the run, and a truncated run
  that happens to stop after the easy items will look better than a complete one.
- Task fingerprints matching. Result paths carry a fingerprint of the template,
  style, seed, and permutation count, so incompatible runs cannot overwrite each
  other -- but two runs in different directories can still be incomparable if you
  compare them by hand.

## 7. State the cost alongside the score

`counts.est_cost_usd` is computed from the token counts the provider actually
reported and the price table in `src/vlmeval/pricing.py`, snapshotted
2026-09-26. Check the price is current, or override it per model with
`cost_in` / `cost_out` in a config; the override is recorded in the run's
provenance so a cost figure is always traceable to the price that produced it.

The interesting result is usually the trade-off, not the maximum: a model 1.5
points better at 20x the cost is a decision, not a result.

## A defensible summary sentence

> On `<dataset>` (<n> items, `<task>`, `<template/style>`, `<k>` option
> permutations), `<model>` scored **<accuracy> [95% CI]**, with
> `<unparsed_rate>` unparsed and order sensitivity of `<x>` points. It is
> `<better/worse>` than `<baseline>` by `<delta>` points
> (95% CI `[lo, hi]`, McNemar p `<p>`, Holm-corrected across `<k>` comparisons),
> at an estimated cost of `$<x>`. `<Known caveat: order sensitivity / unparsed
> rate / dataset size.>`

If a sentence cannot be filled in from the report, the run is not finished.
