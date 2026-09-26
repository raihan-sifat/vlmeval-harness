# mcq / synthetic_mcq / `bad`

_Generated 2026-09-26T13:25:59Z_

## Run

| Field | Value |
| --- | --- |
| Task | `mcq` |
| Dataset | `synthetic_mcq` |
| Model | `bad` |
| Items | 260 |
| Requests | 779 |
| From cache | 779 |
| Failed calls | 0 |
| Unparsed replies | 0 (0.0%) |
| Prompt tokens | 34,916 |
| Completion tokens | 779 |
| Est. cost | $0.0000 |
| Wall time | 1.5s |

## Headline metrics

Values are point estimates with 95% bootstrap confidence intervals.

| Metric | Value | 95% CI | n | Better |
| --- | --- | --- | --- | --- |
| accuracy | 0.0 [0.0, 0.0] | [0.000, 0.000] | 260 | higher |
| accuracy_parsed | 0.0 [0.0, 0.0] | [0.000, 0.000] | 260 | higher |
| unparsed_rate | 0.0 | [-, -] | 260 | higher |
| consistency | 6.9 [3.8, 10.0] | [0.038, 0.100] | 260 | higher |
| variant_accuracy | 0.0 [0.0, 0.0] | [0.000, 0.000] | 779 | higher |
| canonical_accuracy | 0.0 [0.0, 0.0] | [0.000, 0.000] | 260 | higher |
| order_sensitivity | 0.0 | [-, -] | 260 | higher |

## By category

Categories are dataset-defined: MMMU disciplines, or the POPE
random / popular / adversarial splits. A single pooled number hides
the differences that matter, so the split is always shown.

### color

| Metric | Value | 95% CI | n |
| --- | --- | --- | --- |
| accuracy | 0.0 [0.0, 0.0] | [0.000, 0.000] | 54 |

### count

| Metric | Value | 95% CI | n |
| --- | --- | --- | --- |
| accuracy | 0.0 [0.0, 0.0] | [0.000, 0.000] | 34 |

### shape

| Metric | Value | 95% CI | n |
| --- | --- | --- | --- |
| accuracy | 0.0 [0.0, 0.0] | [0.000, 0.000] | 121 |

### size

| Metric | Value | 95% CI | n |
| --- | --- | --- | --- |
| accuracy | 0.0 [0.0, 0.0] | [0.000, 0.000] | 51 |

## Answer distribution

Where the model's answers actually land. A model that scores well
but piles its answers onto one letter is exploiting a position
prior rather than reading the image.

| Bucket | Share |
| --- | --- |
| A | 23.1 |
| B | 26.2 |
| C | 31.2 |
| D | 19.6 |
| unparsed | 0.0 |

## Notes

- Backend: `echo`
- Model: `bad`
- Task: Multiple-choice visual question answering (lettered options)
- Prompt style: `plain`
- Unparsed replies are excluded from the mean and reported separately; they are never scored as correct.
- Intervals are percentile bootstrap over items with a fixed seed, so re-running the report reproduces them exactly.

