# Model comparison

Paired McNemar tests on shared items, with Holm-Bonferroni correction across each family of comparisons.
A p-value above 0.05 means the data cannot distinguish these two runs; it does not mean they are equally good.

## mcq / synthetic_mcq

Baseline: `demo`

| Model | Metric | 95% CI | Delta | 95% CI of delta | p (adj) | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| `demo` | 65.4 | [59.6, 71.2] | — | — | — | baseline |
| `oracle` | 100.0 | [100.0, 100.0] | +34.6 pts | [26.1, 37.8] | 0.000 | significant |
| `strong` | 92.7 | [89.2, 95.8] | +27.3 pts | [18.9, 29.7] | 0.000 | significant |
| `weak` | 24.2 | [19.2, 29.6] | -41.2 pts | [-47.2, -35.2] | 0.000 | significantly worse |
| `bad` | 0.0 | [0.0, 0.0] | -65.4 pts | [-73.9, -62.2] | 0.000 | significantly worse |

Correction applied across 4 comparison(s) at alpha=0.05.

### Where the models disagree most

Categories where this run is furthest from the baseline.

| Category | Metric | Baseline | This run | n |
| --- | --- | --- | --- | --- |
| color | accuracy | 70.4 | 0.0 | 54 |
| shape | accuracy | 66.9 | 0.0 | 121 |
| size | accuracy | 62.7 | 0.0 | 51 |
| count | accuracy | 55.9 | 0.0 | 34 |
| color | accuracy | 70.4 | 25.9 | 54 |
| count | accuracy | 55.9 | 100.0 | 34 |

## pope / synthetic_pope

Baseline: `demo`

| Model | Metric | 95% CI | Delta | 95% CI of delta | p (adj) | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| `demo` | 65.8 | [60.0, 72.1] | — | — | — | baseline |
| `oracle` | 100.0 | [100.0, 100.0] | +34.2 pts | [28.3, 40.0] | 0.000 | significant |
| `hallucinator` | 50.0 | [43.3, 56.3] | -15.8 pts | [-24.2, -7.5] | 0.001 | significantly worse |

Correction applied across 2 comparison(s) at alpha=0.05.

### Where the models disagree most

Categories where this run is furthest from the baseline.

| Category | Metric | Baseline | This run | n |
| --- | --- | --- | --- | --- |
| random | accuracy | 62.5 | 100.0 | 80 |
| adversarial | accuracy | 65.0 | 100.0 | 80 |
| popular | accuracy | 70.0 | 100.0 | 80 |
| popular | accuracy | 70.0 | 50.0 | 80 |
| adversarial | accuracy | 65.0 | 50.0 | 80 |
| random | accuracy | 62.5 | 50.0 | 80 |

