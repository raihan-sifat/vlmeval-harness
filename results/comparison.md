# Model comparison

Paired McNemar tests on shared items, with Holm-Bonferroni correction across each family of comparisons.
A p-value above 0.05 means the data cannot distinguish these two runs; it does not mean they are equally good.

## mcq / synthetic_mcq

Baseline: `demo`

| Model | Metric | 95% CI | Delta | 95% CI of delta | p (adj) | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| `demo` | 65.4 | [59.6, 71.2] | — | — | — | baseline |
| `oracle` | 100.0 | [100.0, 100.0] | +34.6 pts | [26.1, 37.8] | 0.000 | significant |

Correction applied across 1 comparison(s) at alpha=0.05.

### Where the models disagree most

Categories where this run is furthest from the baseline.

| Category | Metric | Baseline | This run | n |
| --- | --- | --- | --- | --- |
| count | accuracy | 55.9 | 100.0 | 34 |
| size | accuracy | 62.7 | 100.0 | 51 |
| shape | accuracy | 66.9 | 100.0 | 121 |
| color | accuracy | 70.4 | 100.0 | 54 |

## pope / synthetic_pope

Baseline: `demo`

| Model | Metric | 95% CI | Delta | 95% CI of delta | p (adj) | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| `demo` | 65.8 | [60.0, 72.1] | — | — | — | baseline |
| `hallucinator` | 50.0 | [43.3, 56.3] | -15.8 pts | [-24.2, -7.5] | 0.001 | significantly worse |

Correction applied across 1 comparison(s) at alpha=0.05.

### Where the models disagree most

Categories where this run is furthest from the baseline.

| Category | Metric | Baseline | This run | n |
| --- | --- | --- | --- | --- |
| popular | accuracy | 70.0 | 50.0 | 80 |
| adversarial | accuracy | 65.0 | 50.0 | 80 |
| random | accuracy | 62.5 | 50.0 | 80 |

