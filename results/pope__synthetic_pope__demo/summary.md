# pope / synthetic_pope / `demo`

_Generated 2026-09-26T13:29:31Z_

## Run

| Field | Value |
| --- | --- |
| Task | `pope` |
| Dataset | `synthetic_pope` |
| Model | `demo` |
| Items | 240 |
| Requests | 240 |
| From cache | 240 |
| Failed calls | 0 |
| Unparsed replies | 0 (0.0%) |
| Prompt tokens | 4,341 |
| Completion tokens | 748 |
| Est. cost | $0.0000 |
| Wall time | 0.2s |

## Headline metrics

Values are point estimates with 95% bootstrap confidence intervals.

| Metric | Value | 95% CI | n | Better |
| --- | --- | --- | --- | --- |
| accuracy | 65.8 [60.0, 72.1] | [0.600, 0.721] | 240 | higher |
| precision | 65.8 | [-, -] | 120 | higher |
| recall | 65.8 | [-, -] | 120 | higher |
| f1 | 65.8 | [-, -] | 161 | higher |
| specificity | 65.8 | [-, -] | 120 | higher |
| yes_ratio | 50.0 | [-, -] | 240 | higher |
| unparsed_rate | 0.0 | [-, -] | 240 | higher |
| gold_yes_ratio | 50.0 | [-, -] | 240 | higher |
| yes_ratio_bias | 0.0 | [-, -] | 240 | higher |
| recall_yes | 65.8 | [-, -] | 120 | higher |
| recall_no | 65.8 | [-, -] | 120 | higher |

## By category

Categories are dataset-defined: MMMU disciplines, or the POPE
random / popular / adversarial splits. A single pooled number hides
the differences that matter, so the split is always shown.

### adversarial

| Metric | Value | 95% CI | n |
| --- | --- | --- | --- |
| accuracy | 65.0 [55.0, 75.0] | [0.550, 0.750] | 80 |
| precision | 63.0 | [-, -] | 46 |
| recall | 72.5 | [-, -] | 40 |
| f1 | 67.4 | [-, -] | 57 |
| yes_ratio | 57.5 | [-, -] | 80 |

### popular

| Metric | Value | 95% CI | n |
| --- | --- | --- | --- |
| accuracy | 70.0 [60.0, 80.0] | [0.600, 0.800] | 80 |
| precision | 70.0 | [-, -] | 40 |
| recall | 70.0 | [-, -] | 40 |
| f1 | 70.0 | [-, -] | 52 |
| yes_ratio | 50.0 | [-, -] | 80 |

### random

| Metric | Value | 95% CI | n |
| --- | --- | --- | --- |
| accuracy | 62.5 [52.5, 73.8] | [0.525, 0.738] | 80 |
| precision | 64.7 | [-, -] | 34 |
| recall | 55.0 | [-, -] | 40 |
| f1 | 59.5 | [-, -] | 52 |
| yes_ratio | 42.5 | [-, -] | 80 |

## Notes

- Backend: `echo`
- Model: `demo`
- Task: Object-existence probes for hallucination (POPE-style)
- Prompt style: `plain`
- Unparsed replies are excluded from the mean and reported separately; they are never scored as correct.
- Intervals are percentile bootstrap over items with a fixed seed, so re-running the report reproduces them exactly.

