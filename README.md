# vlmeval

A reproducible evaluation harness for vision-language models.

Point it at a model and a dataset, and it produces a number you can defend: a
point estimate, a confidence interval, a cost, and the raw per-item records it
was computed from. Comparing two models gives a paired significance test rather
than two numbers side by side.

It exists because "GPT-4o scored 78% on our benchmark" is usually not a
reproducible claim. The dataset is unstated, the prompts are unstated, the option
order was one arbitrary ordering, the items the model failed to answer were
quietly dropped from the denominator, and the comparison to the other model was
made on unpaired scores with no uncertainty at all.

```bash
git clone <this repo>
cd Vision-Language\ Model\ Evaluation

# No install, no API key, no GPU, no network.
python scripts/make_fixtures.py --n-items 120
python scripts/run_eval.py --model echo:demo --task mcq --source synthetic_mcq --permutations 2
```

That produces a real evaluation in a few seconds, because the offline `echo`
backend is a genuine model with deterministic behaviour -- not a stub that
always answers "C".

---

## Contents

- [Why this exists](#why-this-exists)
- [Tech stack](#tech-stack)
- [Install](#install)
- [The five-minute tour](#the-five-minute-tour)
- [Concepts](#concepts)
- [Tasks](#tasks)
- [Models](#models)
- [Datasets](#datasets)
- [Reading the results](#reading-the-results)
- [Reproducibility](#reproducibility)
- [Cost control](#cost-control)
- [Configuration files](#configuration-files)
- [Command reference](#command-reference)
- [Extending it](#extending-it)
- [Project layout](#project-layout)
- [Testing](#testing)
- [Limitations](#limitations)
- [License](#license)

---

## Why this exists

Most VLM evaluation code answers "what was the score?" and none of the harder
questions:

| Question                                          | What most harnesses do          | What this one does                                                                           |
| ------------------------------------------------- | ------------------------------- | -------------------------------------------------------------------------------------------- |
| What exactly was the model shown?                 | Whatever string was in the loop | One shared template per task, versioned by fingerprint, stored with every record             |
| Does the model just prefer option A?              | Ignored                         | Permutation audit: re-ask with shuffled options and report consistency and order sensitivity |
| What happened to replies that didn't parse?       | Dropped from the denominator    | Counted as **wrong**, and reported separately as `unparsed_rate`                             |
| Is 78% vs 76% a real difference?                  | Two numbers, no test            | Paired McNemar with a bootstrap CI on the delta, Holm-corrected across the family            |
| Could the answer key have leaked?                 | Unknown                         | Gold never enters a prompt or a provider payload; a test asserts it for every backend        |
| What did it cost?                                 | A guess                         | Per-item token counts and a price lookup, with a hard spend ceiling                          |
| Can I resume a 5,000-item run that died at 2,000? | Start over                      | JSONL sink, resume, and a content-addressed response cache                                   |

## Tech stack

Everything below is chosen so the harness runs on a fresh clone before anyone
resolves a heavy dependency graph, spends money, or touches a GPU. The core is
**pure Python plus Pillow** -- every provider SDK, dataset loader, and plotting
library is an optional extra.

**Core**

| Layer             | Choice                                           | Why                                                                                                             |
| ----------------- | ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| Language          | Python 3.10+                                     | `match`, `X \| None` unions, and modern typing throughout                                                       |
| Runtime deps      | [Pillow](https://python-pillow.org/) only        | Image loading, resizing, and base64 encoding for vision payloads                                                |
| Packaging         | `pyproject.toml` (PEP 517/518), setuptools       | One editable install exposes the `vlmeval` console script; extras gate everything else                          |
| CLI               | Standard-library `argparse` (subcommands)        | Zero-dependency, no framework to learn                                                                          |
| Concurrency       | `concurrent.futures.ThreadPoolExecutor`          | I/O-bound provider calls; the cache uses a single transactional writer                                          |
| Retries & budgets | Hand-rolled in `runner.py` / `runstate.py`       | Per-request retry, spend ceiling, and resume without an orchestration library                                   |
| Statistics        | Pure-Python stdlib (`math`)                      | Bootstrap CIs, McNemar's test, and Holm-Bonferroni implemented directly -- no NumPy/SciPy needed to score a run |
| Storage           | JSONL sinks + content-addressed **SQLite** cache | Crash-safe resume and free re-runs; `sqlite3` ships with Python                                                 |
| Config            | YAML with a built-in subset reader               | PyYAML when present, a strict fallback parser otherwise -- never guesses                                        |
| Reproducibility   | Content hashing (`hashlib`) + seeded `random`    | Byte-identical fixtures and fingerprinted outputs                                                               |

**Optional extras** (install only what you use -- `pip install -e ".[all]"` for everything)

| Extra       | Stack                                            | Purpose                                           |
| ----------- | ------------------------------------------------ | ------------------------------------------------- |
| `config`    | PyYAML                                           | Full YAML config files                            |
| `cli`       | tqdm                                             | Progress bars instead of periodic plain lines     |
| `openai`    | OpenAI Python SDK                                | `openai:gpt-4o` and compatible endpoints          |
| `anthropic` | Anthropic Python SDK                             | `anthropic:claude-haiku-4-5`                      |
| `google`    | google-genai SDK                                 | `google:gemini-2.5-flash`                         |
| `local`     | PyTorch, Transformers, Accelerate, qwen-vl-utils | Open-weight VLMs on local GPU (`hf:smolvlm-500m`) |
| `data`      | Hugging Face `datasets`                          | Load public benchmarks from the Hub               |
| `analysis`  | pandas, matplotlib, scipy                        | `vlmeval plot` figures and tabular analysis       |
| `dev`       | pytest, pytest-cov, ruff                         | Test suite and linting                            |

**Testing & quality:** 182 tests run with `unittest` / `pytest` against recording
stubs -- no network, no API keys, no GPU. Provider adapters are verified in
isolation so a request-shape bug cannot hide behind the offline path. Ruff
(`line-length = 100`) handles linting and import sorting.

**Architecture:** a small plugin model -- model backends and tasks register
through a central registry, specs like `openai:gpt-4o` and `mcq` resolve to
objects, and metrics are plain callables over per-item outcomes. Adding a
backend or task is a class plus one registry entry.

## Install

The harness runs with **Pillow as its only requirement**, which is deliberate: the
first thing a new user should be able to do is run a complete evaluation without
resolving a dependency graph.

```bash
pip install -e .                 # core: Pillow
pip install -e ".[all]"          # + every provider SDK, dataset loading, figures
```

Optional extras, install only what you need:

| Extra                             | Adds                | Needed for                                                         |
| --------------------------------- | ------------------- | ------------------------------------------------------------------ |
| `config`                          | PyYAML              | Richer config files (a built-in reader handles the shipped subset) |
| `cli`                             | tqdm                | Progress bars (otherwise periodic plain lines)                     |
| `openai` / `anthropic` / `google` | provider SDKs       | Those cloud backends                                               |
| `local`                           | torch, transformers | Local open-weight VLMs                                             |
| `data`                            | datasets            | Loading benchmarks from the Hugging Face Hub                       |
| `analysis`                        | pandas, matplotlib  | `vlmeval plot`                                                     |
| `dev`                             | pytest, ruff        | Tests and linting                                                  |

`python scripts/run_eval.py` works without installing anything at all; the
scripts put `src/` on the path themselves.

## The five-minute tour

**1. See what you can run.** Start here when something refuses to start.

```bash
python scripts/run_eval.py models   # backends, credentials, prices, local presets
python scripts/run_eval.py tasks    # tasks and prompt templates
python scripts/run_eval.py keys     # which provider credentials are present
```

**2. Make the offline benchmark.** Deterministic, byte-identical on every
machine, so the fixtures are committed and diffable.

```bash
python scripts/make_fixtures.py --n-items 120
```

**3. Look before you spend.** Every expensive mistake -- wrong dataset, wrong
task, a prompt that does not list the options -- is visible for free:

```bash
python scripts/run_eval.py --model openai:gpt-4o --task mcq \
    --source synthetic_mcq --permutations 2 --dry-run
```

```
Dry run: no model was contacted and nothing was charged.

  models    openai:gpt-4o
  task      mcq (Multiple-choice visual question answering, 2 permutations/item, seed 12345, <= 8 reply tokens)
  source    synthetic_mcq  path=None
  dataset   synthetic_mcq  (120 items, 360 requests)
  out       results
  estimate  <= $2.31 worst case at 4096 tokens/request

--- first prompt (mcq_0000_0) ---
What is the colour of the hexagon in the image?

A. yellow
B. purple
C. blue
D. red

Answer with the letter of the correct option only (for example: A). Do not explain your reasoning.
--- end prompt ---

Remove --dry-run to run for real.
```

**4. Run it for real.**

```bash
python scripts/run_eval.py --model echo:demo --task mcq --source synthetic_mcq --permutations 2
```

```
  mcq / synthetic_mcq / demo
  accuracy: 76.2 [66.7, 84.5]
  unparsed_rate=0.036 consistency=0.786
  n=251 failed=0 unparsed=9 cost=$0.0000 time=0.5s
  report: results/mcq__synthetic_v1__demo/summary.md
```

**5. Compare models properly.** Use `benchmark` rather than looping `run` by
hand; it is what makes the comparison section meaningful.

```bash
python scripts/run_eval.py benchmark --models echo:oracle echo:strong echo:demo echo:weak echo:bad \
    --task mcq --source synthetic_mcq --permutations 2
```

```
| Model     | accuracy          | Cost     | n   |
| --------- | ----------------- | -------- | --- |
| `oracle`  | 100.0 [100.0, 100.0] | $0.0000 | 251 |
| `strong`  | 95.2 [90.5, 98.8]    | $0.0000 | 251 |
| `demo`    | 76.2 [66.7, 84.5]    | $0.0000 | 251 |
| `blind`   | 36.9 [26.2, 47.6]    | $0.0000 | 251 |
| `weak`    | 34.5 [23.8, 45.2]    | $0.0000 | 251 |
| `bad`     | 0.0 [0.0, 0.0]       | $0.0000 | 251 |
```

```bash
python scripts/run_eval.py compare --results results --baseline demo
```

```
| Model     | Metric | 95% CI          | Delta     | 95% CI of delta | p (adj) | Verdict             |
| --------- | ------ | --------------- | --------- | --------------- | ------- | ------------------- |
| `oracle`  | 100.0  | [100.0, 100.0]  | +23.8 pts | [12.3, 29.6]    | 0.000   | significant         |
| `blind`   | 36.9   | [26.2, 47.6]    | -39.3 pts | [-55.6, -28.4]   | 0.000   | significantly worse |
| `weak`    | 34.5   | [23.8, 45.2]    | -41.7 pts | [-53.8, -30.8]   | 0.000   | significantly worse |

Correction applied across 5 comparison(s) at alpha=0.05.
```

A p-value above 0.05 means the data cannot distinguish two runs. It does not
mean they are equally good.

## Concepts

**One task, one prompt.** Every model sees the same template, the same
instruction, and the same option order for a given item. Model-specific
formatting lives in the adapter, not in the prompt, so a score difference is
attributable to the model rather than to the harness.

**Two accuracies, always.** `accuracy` counts an unparseable reply as wrong.
`accuracy_parsed` excludes those items, and `unparsed_rate` reports how many
there were. A model that refuses half the items cannot look good by having the
hard ones dropped.

**Permutation audit.** With `--permutations N`, each MCQ item is re-asked with
N extra option orders. The same item keeps the same seed, so the model is
_consistently_ right or consistently wrong, the way a real model behaves. Then:

- `consistency` -- how often all orders agreed. A model that only ever answers
  "A" scores near `1/num_choices` here while its accuracy sits near chance.
- `order_sensitivity` -- the gap between canonical and variant accuracy. Large
  values mean the score is partly an artifact of option order.
- `variant_accuracy` -- accuracy over the permuted prompts, which is the number
  to quote if order effects are large.

**Canonical vs display space.** The gold label is stored canonically, so
permuting options does not change the answer key. The letter a prompt asks for
is tracked separately as `display_gold` and is what an adapter is given. Decoding
a reply back to canonical space is the harness's job, and it is covered by tests.

**No gold leakage.** The answer key never enters a prompt or a provider payload.
The base adapter's `generate_with_context` hook drops the harness context
outright; only the offline `echo` backend opts in, because its whole purpose is
to be a model of known skill. There is a test per provider asserting the gold
label cannot appear in a request.

## Tasks

### `mcq` -- multiple-choice visual question answering

Lettered options, one correct answer, with the option-order audit described
above. Reports `accuracy`, `accuracy_parsed`, `consistency`, `variant_accuracy`,
`order_sensitivity`, `unparsed_rate`, and per-category breakdowns.

```bash
python scripts/run_eval.py --model hf:smolvlm-500m --task mcq \
    --source dir --path data/samples/synthetic_v1 --permutations 3 --style grounded
```

`--style grounded` adds a system prompt that pressures the model to answer only
from the image, which is how you find out whether a score came from the picture
or from the shape of the question.

### `pope` -- probing for object hallucination

Yes/no existence probes over random, popular, and adversarial splits. Reports
`accuracy`, `precision`, `recall`, `specificity`, `f1`, and `yes_ratio`, plus
per-split breakdowns.

```bash
python scripts/run_eval.py --model echo:hallucinator --task pope --source synthetic_pope
```

```
  accuracy: 50.0 [30.0, 70.0]
  precision=0.500 recall=1.000 f1=0.667 yes_ratio=1.000 unparsed_rate=0.000
```

That is the hallucination signature and it is worth reading carefully: recall
1.0 with specificity 0.0 and `yes_ratio` 1.0 means the model asserts every
object is present. Accuracy alone would hide that completely -- a model that
says "yes" to everything scores 50% on a balanced set.

## Models

| Backend   | Spec                                     | Credentials                          |
| --------- | ---------------------------------------- | ------------------------------------ |
| OpenAI    | `openai:gpt-4o`                          | `OPENAI_API_KEY`                     |
| Anthropic | `anthropic:claude-haiku-4-5`             | `ANTHROPIC_API_KEY`                  |
| Google    | `google:gemini-2.5-flash`                | `GOOGLE_API_KEY` or `GEMINI_API_KEY` |
| Local     | `hf:smolvlm-500m`, `hf:qwen2-vl-2b-4bit` | none (downloads weights)             |
| Offline   | `echo:demo`                              | none                                 |

Copy `.env.example` to `.env` and fill in what you need; the harness loads it on
start and real environment variables always win, so CI secrets are never
overwritten by a checked-in file.

### The offline `echo` backend

`echo` is a deterministic model with a seeded hash for its answers and knobs for
the ways real models fail. It is not a placeholder: `echo:oracle` scores exactly
100.0 and `echo:bad` exactly 0.0, which makes them usable as CI floors and
ceilings that catch a scoring regression immediately.

| Preset                      | Behaviour                                  | Use it to                             |
| --------------------------- | ------------------------------------------ | ------------------------------------- |
| `echo:oracle`               | always right, never verbose, never refuses | CI ceiling; expect accuracy 1.0       |
| `echo:bad`                  | never right                                | CI floor; expect accuracy 0.0         |
| `echo:strong` / `echo:weak` | high / low skill                           | a spread worth comparing              |
| `echo:demo`                 | 70% skill, some chatter                    | the default smoke test                |
| `echo:chatty`               | wraps every reply in prose                 | exercise the answer parsers           |
| `echo:refuses`              | always unparseable                         | exercise unparsed accounting          |
| `echo:blind`                | always answers the first option            | exercise the order-consistency audit  |
| `echo:hallucinator`         | asserts every object is present            | exercise POPE's whole reason to exist |

Presets are a starting point, not a ceiling -- any field can be overridden:

```bash
python scripts/run_eval.py --model '{"backend":"echo","model":"strong","chatter":0.9}'
```

## Datasets

```bash
--source synthetic_mcq     # the committed offline fixtures
--source synthetic_pope
--source dir  --path data/samples/synthetic_v1
--source jsonl --path my_items.jsonl
--source hf   --path pope          # via `pip install -e ".[data]"`
```

A JSONL item needs an `id`, an `image` path, a `question`, the `choices`, and
the `answer` index. Unknown fields are preserved in `meta` and end up in the
results, which is how per-item provenance survives into the report.

`--limit N` evaluates the first N items. Useful with a fixed seed, useless
without one, because "the first N" must mean the same N on every machine.

## Reading the results

Each run writes to `results/<task>__<dataset>__<model>/`:

| File            | Contents                                                                                       |
| --------------- | ---------------------------------------------------------------------------------------------- |
| `metrics.json`  | Every metric with its confidence interval, per-category breakdowns, counts, and the run config |
| `summary.md`    | The human-readable report                                                                      |
| `summary.csv`   | The same numbers as a table, for a spreadsheet                                                 |
| `results.jsonl` | One scored record per item, with the full prompt and raw reply                                 |
| `manifest.json` | Provenance: timestamps, config, environment, counts                                            |

`results/raw/` keeps the fingerprint-tagged raw trace used for resuming. It is
gitignored because the committed fixtures reproduce it byte for byte, and it is
several times larger than the reports.

`metrics.json` deliberately does **not** inline the per-item records. Reading one
score means parsing about 15 KB, not 750 KB; the item detail lives in
`results.jsonl` next to it, and `vlmeval compare` reads both.

The numbers are fractions in `[0, 1]` on disk and percentages when printed. That
is deliberate: JSON that means different things depending on who reads it is a
trap, and `Metric.value` is always the fraction.

Read `analysis/REPORTING.md` before writing up a comparison. The short version:
read the intervals before the ranking, check `unparsed_rate` before celebrating
an accuracy, and treat `order_sensitivity` above a few points as a caveat on the
headline number.

## Reproducibility

- **Deterministic fixtures.** Same seed, byte-identical files, safe to commit and
  diff.
- **Deterministic permutations.** Permutation seeds derive from the item id, so
  the same items get the same option orders on every machine and every rerun.
- **Fingerprinted outputs.** The task fingerprint covers the template, style,
  seed, permutation count, and options. Change any of them and the sink path
  changes too, so results from different configurations cannot be confused or
  resumed into one another.
- **Provenance in every record.** The prompt, the raw reply, the finish reason,
  token counts, latency, and the model info dict are stored per item.
- **Compatibility notes recorded.** When a provider rejects a parameter and the
  adapter retries without it, that appears in `info['compat']` and in the
  results, so a non-standard call is visible instead of silent.

## Cost control

```bash
--max-usd 5.00           # stop once estimated spend reaches this
--max-tokens-total 500000
--max-requests 500       # bounds wall-clock independently of what the provider reports
```

A budget stop is recorded in `counts['stopped_reason']` and printed as
`STOPPED EARLY`, so a truncated run is never mistaken for a complete one. Paired
with `--dry-run`'s worst-case estimate, the expensive mistakes are all visible
before the first request.

Responses are cached in a content-addressed SQLite database keyed by model,
config, prompt, image bytes, and parameters. Re-running the same evaluation
costs nothing, and a resumed run only pays for what it has not already fetched.

## Configuration files

A config file names complete, repeatable experiments rather than model ids:

```yaml
defaults:
    workers: 4
    temperature: 0.0
    max_usd: 5.0

models:
    - name: echo-strong
      spec: { backend: echo, model: strong }
      task: mcq
      source: synthetic_mcq

    - name: gpt-4o-audited
      spec: { backend: openai, model: gpt-4o }
      task: mcq
      source: synthetic_mcq
      permutations: 3
      style: grounded
```

```bash
python scripts/run_eval.py run --config configs/models.yaml --preset gpt-4o-audited
```

Precedence is **explicit flag > preset > `defaults`**. A flag you actually typed
always wins, even when its value happens to equal the default. Unknown preset
names and unknown keys are refused with the list of valid ones, because a config
typo discovered after the credits are spent is the failure this feature exists to
prevent.

PyYAML is used when installed. Without it a built-in reader handles the subset
above and refuses anything it does not implement rather than half-parsing it.

## Command reference

| Command     | Purpose                                               |
| ----------- | ----------------------------------------------------- |
| `run`       | One model against one task and dataset                |
| `benchmark` | Several models, then a leaderboard with paired tests  |
| `summarize` | Rebuild a leaderboard from existing runs              |
| `compare`   | Paired McNemar comparison between runs                |
| `plot`      | Render figures (needs `.[analysis]`)                  |
| `fixtures`  | Generate the offline synthetic benchmark              |
| `models`    | Backends, credentials, prices, local and echo presets |
| `tasks`     | Tasks and prompt templates                            |
| `keys`      | Which credentials are present                         |

Useful flags: `--dry-run`, `--limit`, `--permutations`, `--seed`, `--style`,
`--template`, `--workers`, `--max-retries`, `--no-cache`, `--no-resume`,
`--quiet`, `--max-usd`, `--max-tokens-total`, `--max-requests`.

Every subcommand prints a concrete next step on failure. A harness that exits
with a traceback has spent the user's attention for nothing.

## Extending it

A new backend is a class and a registry entry:

```python
from vlmeval.models.base import ModelAdapter
from vlmeval.types import ModelResponse

class MyModel(ModelAdapter):
    supports_images = True

    def generate(self, *, image, prompt, system=None,
                 max_tokens=None, temperature=None, **kwargs) -> ModelResponse:
        ...
        return ModelResponse(text=reply, prompt_tokens=a, completion_tokens=b)

    def estimate_cost(self, prompt_tokens, completion_tokens) -> float:
        ...
```

```python
# vlmeval/registry.py
MODEL_BACKENDS["mybackend"] = "my_package.my_module:MyModel"
```

```bash
python scripts/run_eval.py --model mybackend:some-model
```

Do not override `generate_with_context` unless the model genuinely needs the
answer key. Every real backend should inherit the base implementation, which
discards the harness context so a label cannot reach a paid endpoint.

A new task subclasses `Task`, implements `queries()` and `score()`, and registers
itself in `TASK_REGISTRY`. Metrics are plain callables over per-item outcomes,
so adding one does not require touching the runner.

## Project layout

```
src/vlmeval/
  cli.py            argparse front end for every subcommand
  config.py         .env loading, config files, preset resolution
  yamlite.py        YAML subset reader for the zero-dependency path
  registry.py       spec strings -> model and task objects
  runner.py         concurrency, retries, resume, scoring
  cache.py          content-addressed SQLite response cache
  runstate.py       progress, budgets, JSONL sink
  stats.py          bootstrap CIs, McNemar, Holm-Bonferroni
  report.py         metrics.json, summary.md, leaderboards
  compare.py        paired run comparison
  plots.py          figures
  prompts.py        the shared templates
  models/           one adapter per backend
  tasks/            mcq, pope
  metrics/          per-task scoring
  data/             loading and the synthetic generator
scripts/            thin wrappers that work without installing
tests/              182 tests, no network and no API keys
configs/            models.yaml, tasks.yaml
analysis/           how to write up a comparison
```

## Testing

```bash
python -m unittest discover -s tests -t .     # no dependencies at all
python -m compileall -q src scripts tests     # catches syntax errors anywhere
pytest                                        # if you have the dev extra
```

The suite runs with no network, no API keys, and no GPU. Provider adapters are
tested against recording stubs injected in place of the SDKs, which is how a
syntax error in `google_api.py` and an unreachable parameter-rename branch in
`openai_api.py` were both caught -- neither is reachable from the offline path.
`tests/test_providers.py` also asserts the gold label cannot appear in a request
payload, per backend.

The `echo` presets double as regression fixtures: `oracle` at 100.0 and `bad` at
0.0 are asserted in CI, so a change that breaks scoring fails loudly instead of
quietly shifting every number.

## Limitations

Worth stating plainly:

- **Two tasks.** MCQ and POPE-style probing. The design generalises, but they are
  the only implementations.
- **The offline benchmark is synthetic.** It is generated from shapes and
  colours, so it validates the harness rather than measuring model quality. Its
  question mix is also uneven -- shapes dominate -- so do not read it as a
  capability benchmark.
- **Prices are a snapshot.** `src/vlmeval/pricing.py` records list prices as of
  2026-09-26 and needs periodic review. Override per model with `cost_in` /
  `cost_out` in a config, or `cost_per_million` in a model mapping.
- **Bootstrap CIs assume item independence.** With clustered or near-duplicate
  items they will read narrow.
- **Cloud and local backends are unverified here.** The request and response
  shapes are covered by tests, and the adapters handle API drift, but no paid
  call or GPU inference has been run in this environment.
- **Option permutation covers order, not phrasing.** Robustness to reworded
  questions is not measured.

## License

MIT. See [LICENSE](LICENSE).
