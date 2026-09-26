# Experiment log template

Copy this per experiment. Its purpose is to make a run reproducible from the
writeup alone, months later, without archaeology through shell history.

## Setup

| | |
| --- | --- |
| Date | |
| Commit | `git rev-parse --short HEAD` |
| Task / dataset | e.g. `mcq` / `synthetic_v1` |
| Items (`n`) | |
| Prompt template / style | e.g. default / `grounded` |
| Option permutations | e.g. `0` or `3` |
| Permutation seed | |
| Decoding | `temperature=0.0`, `max_tokens=<n>` |

## Models

| Model | Backend | Revision / date | Price used | Notes |
| --- | --- | --- | --- | --- |
| `gpt-4o` | openai | | from `pricing.py` | |
| | | | | |

## Results

| Model | Accuracy [95% CI] | Parsed acc | Unparsed | Consistency | Order sens. | Cost | n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| | | | | | | | |

Per category or split, if it matters:

| Model | `<category>` | `<category>` | `<category>` |
| --- | --- | --- | --- |
| | | | |

## Comparison

Baseline: `<model>`. `vlmeval compare --baseline <model>`, Holm-corrected across
`<k>` comparisons.

| Model | Delta [95% CI] | p (adj) | Verdict |
| --- | --- | --- | --- |
| | | | |

## Caveats

State the ones that apply. This section is the difference between a measurement
and a claim.

- [ ] Intervals overlap somewhere in the ranking
- [ ] `unparsed_rate` above `<x>`
- [ ] `order_sensitivity` above `<x>` points
- [ ] Dataset smaller than the gap being claimed
- [ ] Price table may be stale
- [ ] Prompt wording untested
- [ ] Model revision not pinned

## Verdict

One paragraph. What the numbers support, what they do not, and what would change
the conclusion. Include the next experiment worth running.
