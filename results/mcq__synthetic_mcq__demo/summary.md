# mcq / synthetic_mcq / `demo`

_Generated 2026-09-26T13:25:57Z_

## Run

| Field | Value |
| --- | --- |
| Task | `mcq` |
| Dataset | `synthetic_mcq` |
| Model | `demo` |
| Items | 260 |
| Requests | 779 |
| From cache | 779 |
| Failed calls | 0 |
| Unparsed replies | 32 (4.1%) |
| Prompt tokens | 34,916 |
| Completion tokens | 2,537 |
| Est. cost | $0.0000 |
| Wall time | 1.3s |

## Headline metrics

Values are point estimates with 95% bootstrap confidence intervals.

| Metric | Value | 95% CI | n | Better |
| --- | --- | --- | --- | --- |
| accuracy | 65.4 [59.6, 71.2] | [0.596, 0.712] | 260 | higher |
| accuracy_parsed | 68.3 [62.2, 73.9] | [0.622, 0.739] | 249 | higher |
| unparsed_rate | 4.2 | [-, -] | 260 | higher |
| consistency | 67.3 [61.5, 73.1] | [0.615, 0.731] | 260 | higher |
| variant_accuracy | 68.4 [64.9, 71.6] | [0.649, 0.716] | 749 | higher |
| canonical_accuracy | 65.4 [59.6, 71.2] | [0.596, 0.712] | 260 | higher |
| order_sensitivity | -3.0 | [-, -] | 260 | higher |

## By category

Categories are dataset-defined: MMMU disciplines, or the POPE
random / popular / adversarial splits. A single pooled number hides
the differences that matter, so the split is always shown.

### color

| Metric | Value | 95% CI | n |
| --- | --- | --- | --- |
| accuracy | 70.4 [57.4, 81.5] | [0.574, 0.815] | 54 |

### count

| Metric | Value | 95% CI | n |
| --- | --- | --- | --- |
| accuracy | 55.9 [38.2, 73.5] | [0.382, 0.735] | 34 |

### shape

| Metric | Value | 95% CI | n |
| --- | --- | --- | --- |
| accuracy | 66.9 [58.7, 75.2] | [0.587, 0.752] | 121 |

### size

| Metric | Value | 95% CI | n |
| --- | --- | --- | --- |
| accuracy | 62.7 [49.0, 74.5] | [0.490, 0.745] | 51 |

## Answer distribution

Where the model's answers actually land. A model that scores well
but piles its answers onto one letter is exploiting a position
prior rather than reading the image.

| Bucket | Share |
| --- | --- |
| A | 25.8 |
| B | 26.9 |
| C | 28.5 |
| D | 14.6 |
| unparsed | 4.2 |

## Notes

- Backend: `echo`
- Model: `demo`
- Task: Multiple-choice visual question answering (lettered options)
- Prompt style: `plain`
- Unparsed replies are excluded from the mean and reported separately; they are never scored as correct.
- Intervals are percentile bootstrap over items with a fixed seed, so re-running the report reproduces them exactly.

