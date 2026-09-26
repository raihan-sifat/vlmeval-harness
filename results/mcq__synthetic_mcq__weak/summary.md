# mcq / synthetic_mcq / `weak`

_Generated 2026-09-26T13:25:58Z_

## Run

| Field | Value |
| --- | --- |
| Task | `mcq` |
| Dataset | `synthetic_mcq` |
| Model | `weak` |
| Items | 260 |
| Requests | 779 |
| From cache | 779 |
| Failed calls | 0 |
| Unparsed replies | 80 (10.3%) |
| Prompt tokens | 34,916 |
| Completion tokens | 4,184 |
| Est. cost | $0.0000 |
| Wall time | 1.3s |

## Headline metrics

Values are point estimates with 95% bootstrap confidence intervals.

| Metric | Value | 95% CI | n | Better |
| --- | --- | --- | --- | --- |
| accuracy | 24.2 [19.2, 29.6] | [0.192, 0.296] | 260 | higher |
| accuracy_parsed | 27.0 [21.5, 33.0] | [0.215, 0.330] | 233 | higher |
| unparsed_rate | 10.4 | [-, -] | 260 | higher |
| consistency | 27.7 [22.3, 33.5] | [0.223, 0.335] | 260 | higher |
| variant_accuracy | 26.7 [23.4, 30.1] | [0.234, 0.301] | 701 | higher |
| canonical_accuracy | 24.2 [19.2, 29.6] | [0.192, 0.296] | 260 | higher |
| order_sensitivity | -2.4 | [-, -] | 260 | higher |

## By category

Categories are dataset-defined: MMMU disciplines, or the POPE
random / popular / adversarial splits. A single pooled number hides
the differences that matter, so the split is always shown.

### color

| Metric | Value | 95% CI | n |
| --- | --- | --- | --- |
| accuracy | 25.9 [14.8, 38.9] | [0.148, 0.389] | 54 |

### count

| Metric | Value | 95% CI | n |
| --- | --- | --- | --- |
| accuracy | 11.8 [2.9, 23.5] | [0.029, 0.235] | 34 |

### shape

| Metric | Value | 95% CI | n |
| --- | --- | --- | --- |
| accuracy | 27.3 [19.8, 35.5] | [0.198, 0.355] | 121 |

### size

| Metric | Value | 95% CI | n |
| --- | --- | --- | --- |
| accuracy | 23.5 [11.8, 35.3] | [0.118, 0.353] | 51 |

## Answer distribution

Where the model's answers actually land. A model that scores well
but piles its answers onto one letter is exploiting a position
prior rather than reading the image.

| Bucket | Share |
| --- | --- |
| A | 22.3 |
| B | 23.8 |
| C | 28.8 |
| D | 14.6 |
| unparsed | 10.4 |

## Notes

- Backend: `echo`
- Model: `weak`
- Task: Multiple-choice visual question answering (lettered options)
- Prompt style: `plain`
- Unparsed replies are excluded from the mean and reported separately; they are never scored as correct.
- Intervals are percentile bootstrap over items with a fixed seed, so re-running the report reproduces them exactly.

