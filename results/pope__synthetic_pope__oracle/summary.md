# pope / synthetic_pope / `oracle`

_Generated 2026-09-26T13:26:00Z_

## Run

| Field | Value |
| --- | --- |
| Task | `pope` |
| Dataset | `synthetic_pope` |
| Model | `oracle` |
| Items | 240 |
| Requests | 240 |
| From cache | 240 |
| Failed calls | 0 |
| Unparsed replies | 0 (0.0%) |
| Prompt tokens | 4,341 |
| Completion tokens | 240 |
| Est. cost | $0.0000 |
| Wall time | 0.2s |

## Headline metrics

Values are point estimates with 95% bootstrap confidence intervals.

| Metric | Value | 95% CI | n | Better |
| --- | --- | --- | --- | --- |
| accuracy | 100.0 [100.0, 100.0] | [1.000, 1.000] | 240 | higher |
| precision | 100.0 | [-, -] | 120 | higher |
| recall | 100.0 | [-, -] | 120 | higher |
| f1 | 100.0 | [-, -] | 120 | higher |
| specificity | 100.0 | [-, -] | 120 | higher |
| yes_ratio | 50.0 | [-, -] | 240 | higher |
| unparsed_rate | 0.0 | [-, -] | 240 | higher |
| gold_yes_ratio | 50.0 | [-, -] | 240 | higher |
| yes_ratio_bias | 0.0 | [-, -] | 240 | higher |
| recall_yes | 100.0 | [-, -] | 120 | higher |
| recall_no | 100.0 | [-, -] | 120 | higher |

## By category

Categories are dataset-defined: MMMU disciplines, or the POPE
random / popular / adversarial splits. A single pooled number hides
the differences that matter, so the split is always shown.

### adversarial

| Metric | Value | 95% CI | n |
| --- | --- | --- | --- |
| accuracy | 100.0 [100.0, 100.0] | [1.000, 1.000] | 80 |
| precision | 100.0 | [-, -] | 40 |
| recall | 100.0 | [-, -] | 40 |
| f1 | 100.0 | [-, -] | 40 |
| yes_ratio | 50.0 | [-, -] | 80 |

### popular

| Metric | Value | 95% CI | n |
| --- | --- | --- | --- |
| accuracy | 100.0 [100.0, 100.0] | [1.000, 1.000] | 80 |
| precision | 100.0 | [-, -] | 40 |
| recall | 100.0 | [-, -] | 40 |
| f1 | 100.0 | [-, -] | 40 |
| yes_ratio | 50.0 | [-, -] | 80 |

### random

| Metric | Value | 95% CI | n |
| --- | --- | --- | --- |
| accuracy | 100.0 [100.0, 100.0] | [1.000, 1.000] | 80 |
| precision | 100.0 | [-, -] | 40 |
| recall | 100.0 | [-, -] | 40 |
| f1 | 100.0 | [-, -] | 40 |
| yes_ratio | 50.0 | [-, -] | 80 |

## Notes

- Backend: `echo`
- Model: `oracle`
- Task: Object-existence probes for hallucination (POPE-style)
- Prompt style: `plain`
- Unparsed replies are excluded from the mean and reported separately; they are never scored as correct.
- Intervals are percentile bootstrap over items with a fixed seed, so re-running the report reproduces them exactly.

