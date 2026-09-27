# pope / synthetic_pope / `hallucinator`

_Generated 2026-09-26T13:29:31Z_

## Run

| Field | Value |
| --- | --- |
| Task | `pope` |
| Dataset | `synthetic_pope` |
| Model | `hallucinator` |
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
| accuracy | 50.0 [43.3, 56.3] | [0.433, 0.563] | 240 | higher |
| precision | 50.0 | [-, -] | 240 | higher |
| recall | 100.0 | [-, -] | 120 | higher |
| f1 | 66.7 | [-, -] | 240 | higher |
| specificity | 0.0 | [-, -] | 120 | higher |
| yes_ratio | 100.0 | [-, -] | 240 | higher |
| unparsed_rate | 0.0 | [-, -] | 240 | higher |
| gold_yes_ratio | 50.0 | [-, -] | 240 | higher |
| yes_ratio_bias | 50.0 | [-, -] | 240 | higher |
| recall_yes | 100.0 | [-, -] | 120 | higher |
| recall_no | 0.0 | [-, -] | 120 | higher |

## By category

Categories are dataset-defined: MMMU disciplines, or the POPE
random / popular / adversarial splits. A single pooled number hides
the differences that matter, so the split is always shown.

### adversarial

| Metric | Value | 95% CI | n |
| --- | --- | --- | --- |
| accuracy | 50.0 [38.8, 61.3] | [0.388, 0.613] | 80 |
| precision | 50.0 | [-, -] | 80 |
| recall | 100.0 | [-, -] | 40 |
| f1 | 66.7 | [-, -] | 80 |
| yes_ratio | 100.0 | [-, -] | 80 |

### popular

| Metric | Value | 95% CI | n |
| --- | --- | --- | --- |
| accuracy | 50.0 [38.8, 61.3] | [0.388, 0.613] | 80 |
| precision | 50.0 | [-, -] | 80 |
| recall | 100.0 | [-, -] | 40 |
| f1 | 66.7 | [-, -] | 80 |
| yes_ratio | 100.0 | [-, -] | 80 |

### random

| Metric | Value | 95% CI | n |
| --- | --- | --- | --- |
| accuracy | 50.0 [38.8, 61.3] | [0.388, 0.613] | 80 |
| precision | 50.0 | [-, -] | 80 |
| recall | 100.0 | [-, -] | 40 |
| f1 | 66.7 | [-, -] | 80 |
| yes_ratio | 100.0 | [-, -] | 80 |

## Notes

- Backend: `echo`
- Model: `hallucinator`
- Task: Object-existence probes for hallucination (POPE-style)
- Prompt style: `plain`
- Unparsed replies are excluded from the mean and reported separately; they are never scored as correct.
- Intervals are percentile bootstrap over items with a fixed seed, so re-running the report reproduces them exactly.

