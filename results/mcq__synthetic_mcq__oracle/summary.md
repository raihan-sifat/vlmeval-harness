# mcq / synthetic_mcq / `oracle`

_Generated 2026-09-26T13:25:54Z_

## Run

| Field | Value |
| --- | --- |
| Task | `mcq` |
| Dataset | `synthetic_mcq` |
| Model | `oracle` |
| Items | 260 |
| Requests | 779 |
| From cache | 779 |
| Failed calls | 0 |
| Unparsed replies | 0 (0.0%) |
| Prompt tokens | 34,916 |
| Completion tokens | 779 |
| Est. cost | $0.0000 |
| Wall time | 1.4s |

## Headline metrics

Values are point estimates with 95% bootstrap confidence intervals.

| Metric | Value | 95% CI | n | Better |
| --- | --- | --- | --- | --- |
| accuracy | 100.0 [100.0, 100.0] | [1.000, 1.000] | 260 | higher |
| accuracy_parsed | 100.0 [100.0, 100.0] | [1.000, 1.000] | 260 | higher |
| unparsed_rate | 0.0 | [-, -] | 260 | higher |
| consistency | 100.0 [100.0, 100.0] | [1.000, 1.000] | 260 | higher |
| variant_accuracy | 100.0 [100.0, 100.0] | [1.000, 1.000] | 779 | higher |
| canonical_accuracy | 100.0 [100.0, 100.0] | [1.000, 1.000] | 260 | higher |
| order_sensitivity | 0.0 | [-, -] | 260 | higher |

## By category

Categories are dataset-defined: MMMU disciplines, or the POPE
random / popular / adversarial splits. A single pooled number hides
the differences that matter, so the split is always shown.

### color

| Metric | Value | 95% CI | n |
| --- | --- | --- | --- |
| accuracy | 100.0 [100.0, 100.0] | [1.000, 1.000] | 54 |

### count

| Metric | Value | 95% CI | n |
| --- | --- | --- | --- |
| accuracy | 100.0 [100.0, 100.0] | [1.000, 1.000] | 34 |

### shape

| Metric | Value | 95% CI | n |
| --- | --- | --- | --- |
| accuracy | 100.0 [100.0, 100.0] | [1.000, 1.000] | 121 |

### size

| Metric | Value | 95% CI | n |
| --- | --- | --- | --- |
| accuracy | 100.0 [100.0, 100.0] | [1.000, 1.000] | 51 |

## Answer distribution

Where the model's answers actually land. A model that scores well
but piles its answers onto one letter is exploiting a position
prior rather than reading the image.

| Bucket | Share |
| --- | --- |
| A | 31.5 |
| B | 27.3 |
| C | 28.8 |
| D | 12.3 |
| unparsed | 0.0 |

## Notes

- Backend: `echo`
- Model: `oracle`
- Task: Multiple-choice visual question answering (lettered options)
- Prompt style: `plain`
- Unparsed replies are excluded from the mean and reported separately; they are never scored as correct.
- Intervals are percentile bootstrap over items with a fixed seed, so re-running the report reproduces them exactly.

