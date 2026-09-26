# mcq / synthetic_mcq / `strong`

_Generated 2026-09-26T13:25:55Z_

## Run

| Field | Value |
| --- | --- |
| Task | `mcq` |
| Dataset | `synthetic_mcq` |
| Model | `strong` |
| Items | 260 |
| Requests | 779 |
| From cache | 779 |
| Failed calls | 0 |
| Unparsed replies | 0 (0.0%) |
| Prompt tokens | 34,916 |
| Completion tokens | 1,769 |
| Est. cost | $0.0000 |
| Wall time | 1.4s |

## Headline metrics

Values are point estimates with 95% bootstrap confidence intervals.

| Metric | Value | 95% CI | n | Better |
| --- | --- | --- | --- | --- |
| accuracy | 92.7 [89.2, 95.8] | [0.892, 0.958] | 260 | higher |
| accuracy_parsed | 92.7 [89.2, 95.8] | [0.892, 0.958] | 260 | higher |
| unparsed_rate | 0.0 | [-, -] | 260 | higher |
| consistency | 93.1 [90.0, 96.2] | [0.900, 0.962] | 260 | higher |
| variant_accuracy | 92.7 [90.8, 94.4] | [0.908, 0.944] | 779 | higher |
| canonical_accuracy | 92.7 [89.2, 95.8] | [0.892, 0.958] | 260 | higher |
| order_sensitivity | 0.0 | [-, -] | 260 | higher |

## By category

Categories are dataset-defined: MMMU disciplines, or the POPE
random / popular / adversarial splits. A single pooled number hides
the differences that matter, so the split is always shown.

### color

| Metric | Value | 95% CI | n |
| --- | --- | --- | --- |
| accuracy | 92.6 [85.2, 98.1] | [0.852, 0.981] | 54 |

### count

| Metric | Value | 95% CI | n |
| --- | --- | --- | --- |
| accuracy | 79.4 [64.7, 91.2] | [0.647, 0.912] | 34 |

### shape

| Metric | Value | 95% CI | n |
| --- | --- | --- | --- |
| accuracy | 96.7 [93.4, 99.2] | [0.934, 0.992] | 121 |

### size

| Metric | Value | 95% CI | n |
| --- | --- | --- | --- |
| accuracy | 92.2 [84.3, 98.0] | [0.843, 0.980] | 51 |

## Answer distribution

Where the model's answers actually land. A model that scores well
but piles its answers onto one letter is exploiting a position
prior rather than reading the image.

| Bucket | Share |
| --- | --- |
| A | 31.5 |
| B | 27.7 |
| C | 30.4 |
| D | 10.4 |
| unparsed | 0.0 |

## Notes

- Backend: `echo`
- Model: `strong`
- Task: Multiple-choice visual question answering (lettered options)
- Prompt style: `plain`
- Unparsed replies are excluded from the mean and reported separately; they are never scored as correct.
- Intervals are percentile bootstrap over items with a fixed seed, so re-running the report reproduces them exactly.

