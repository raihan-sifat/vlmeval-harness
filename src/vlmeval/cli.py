"""Command-line interface.

Subcommands, in the order you normally need them:

``vlmeval models`` / ``tasks`` / ``keys``
    What can be run, and whether the credentials for it exist. Run this first
    when something refuses to start; it names the fix.
``vlmeval fixtures``
    Generate the offline synthetic benchmark. No network, no API key.
``vlmeval run``
    One model against one task and dataset. This is the workhorse.
``vlmeval benchmark``
    Several models against one task and dataset, then a leaderboard with paired
    significance tests. Use this rather than looping ``run`` by hand: it is what
    makes the comparison section of the report meaningful.
``vlmeval summarize`` / ``plot`` / ``compare``
    Reporting over runs that already exist.

Every subcommand prints a concrete next step on failure. A harness that exits
with a traceback has spent the user's attention for nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .config import load_env, missing_keys
from .registry import available_models, available_tasks, build_model, build_task


def _explicit_options(argv: list[str], parser: argparse.ArgumentParser) -> set[str]:
    """Which options the user actually typed on the command line.

    Needed because a preset should fill in the settings a person did not
    mention, and a flag they *did* type should win. Comparing parsed values to
    parser defaults cannot tell "--workers 4" from the default 4, so the raw
    tokens are inspected instead.

    The option map has to be gathered from the subparsers as well: the parser
    handed to :func:`main` is the top-level one, and its actions contain only
    ``--help`` and the subcommand list. Reading it alone would report that
    nothing was ever typed, and a preset would silently win over every flag.
    """
    destinations: dict[str, str] = {}
    pending = [parser]
    while pending:
        current = pending.pop()
        for action in current._actions:  # noqa: SLF001 - no public equivalent
            if isinstance(action, argparse._SubParsersAction):  # noqa: SLF001
                pending.extend(action.choices.values())
                continue
            for option in action.option_strings:
                destinations[option] = action.dest
    given: set[str] = set()
    for token in argv:
        name = token.split("=", 1)[0]
        dest = destinations.get(name)
        if dest:
            given.add(dest)
    return given



def _apply_preset(args: argparse.Namespace) -> None:
    """Fold a config preset into `args`, without overriding explicit flags."""
    from .config import load_config, resolve_preset

    config = load_config(args.config)
    preset_name = args.preset
    if not preset_name:
        names = sorted(config["presets"])
        if len(names) == 1:
            preset_name = names[0]
        else:
            raise SystemExit(
                f"{args.config} defines {len(names)} presets. "
                f"Choose one with --preset: {', '.join(names)}"
            )

    merged = resolve_preset(config, preset_name)
    explicit = _explicit_options(args._argv, args._parser)

    spec = merged.get("spec")
    if isinstance(spec, dict):
        # A preset carries a full model mapping; record it so `run` can build
        # the adapter from it instead of re-deriving one from a string.
        args._spec_override = spec
        if "model" not in explicit:
            args.model = f"{spec.get('backend')}:{spec.get('model')}"

    for key, value in merged.items():
        if key in {"name", "spec", "_name"}:
            continue
        dest = key.replace("-", "_")
        if dest in explicit or not hasattr(args, dest):
            continue
        setattr(args, dest, value)
    args._preset_name = preset_name


def _resolve_config(args: argparse.Namespace) -> bool:
    """Apply `--config`/`--preset`, reporting problems without a traceback.

    Returns False when the configuration could not be used, so the caller can
    exit non-zero. A config typo discovered after the credits are spent is the
    failure mode this whole feature exists to prevent, so it gets a sentence
    naming the fix rather than a stack trace.
    """
    if not getattr(args, "config", None):
        return True
    try:
        _apply_preset(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Configuration problem: {exc}", file=sys.stderr)
        return False
    return True


def _add_config(parser: argparse.ArgumentParser) -> None:

    parser.add_argument(
        "--config",
        default=None,
        help="YAML file of named run presets, e.g. configs/models.yaml.",
    )
    parser.add_argument(
        "--preset",
        default=None,
        help="Name of a preset in --config. Fills in any setting not given "
        "as a flag; explicit flags still win.",
    )


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--limit", type=int, default=None, help="Evaluate at most N items.")
    parser.add_argument("--out", default="results", help="Output directory (default: results).")
    parser.add_argument("--cache", default="data/cache/responses.sqlite3", help="Cache DB path.")
    parser.add_argument("--no-cache", action="store_true", help="Bypass the response cache.")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent requests.")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries for transient errors.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Cap on reply length.")
    parser.add_argument(
        "--style",
        default="plain",
        choices=["plain", "grounded"],
        help="Prompt style. 'grounded' adds a system prompt that pressures "
        "the model to answer only from the image.",
    )
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing partial results.")
    parser.add_argument("--quiet", action="store_true", help="Suppress the progress bar.")


def _add_budget(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-usd", type=float, default=None, help="Stop once estimated spend reaches this."
    )
    parser.add_argument(
        "--max-tokens-total",
        type=int,
        default=None,
        help="Stop once total tokens (prompt + completion) reach this.",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help="Stop after this many completed requests. Bounds wall-clock and "
        "cost independently of what the provider reports back.",
    )


def _add_dry_run(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build everything, print the plan and the first prompt, then exit. "
        "Contacts no model and costs nothing.",
    )


def _progress_printer(quiet: bool) -> Any:
    if quiet:
        return None
    try:
        from tqdm import tqdm
    except ImportError:
        return None

    bar = tqdm(total=100, unit=" it", desc="eval", dynamic_ncols=True, leave=False)

    def report(progress: Any) -> None:
        if bar.total != progress.total:
            bar.reset(total=max(1, progress.total))
        bar.n = progress.completed
        bar.set_postfix_str(
            f"cached={progress.cached} failed={progress.failed} retried={progress.retried}"
        )
        bar.refresh()

    return report


def _load_examples(args: argparse.Namespace) -> tuple[list[Any], str]:
    from .data import load as load_data

    if args.source.startswith("synthetic"):
        path = args.path or "data/samples/synthetic_v1"
        return load_data(args.source, path=path, limit=args.limit), args.source
    if args.source == "hf":
        name = args.path or args.dataset or "pope"
        return load_data("hf", path=name, limit=args.limit), f"hf:{name}"
    if not args.path:
        raise SystemExit(
            f"--path is required for source {args.source!r}. "
            "Example: --source dir --path data/samples/synthetic_v1"
        )
    return load_data(args.source, path=args.path, limit=args.limit), Path(args.path).name


def _build_task(args: argparse.Namespace) -> Any:
    options: dict[str, Any] = {
        "style": args.style,
        "temperature": args.temperature,
    }
    if args.template:
        options["template"] = args.template
    if args.max_tokens:
        options["max_tokens"] = args.max_tokens
    if args.task in {"mcq", "letter_answer", "multiple_choice", "vqa_mcq"}:
        options["num_permutations"] = args.permutations
        if args.seed is not None:
            options["seed"] = args.seed
    return build_task({"task": args.task, **options})


def _runner_for(args: argparse.Namespace, model: Any, task: Any) -> Any:
    from .cache import ResponseCache
    from .runstate import Budget
    from .runner import Runner

    cache = ResponseCache(args.cache, enabled=not args.no_cache)
    budget = Budget(
        max_tokens_total=args.max_tokens_total,
        max_usd=args.max_usd,
        max_requests=getattr(args, "max_requests", None),
    )
    return Runner(
        model,
        task,
        cache=cache,
        workers=args.workers,
        max_retries=args.max_retries,
        budget=budget,
        sink_dir=args.out,
        resume=not args.no_resume,
        on_progress=_progress_printer(args.quiet),
    )


# --- subcommands -----------------------------------------------------------


def cmd_models(args: argparse.Namespace) -> int:
    from .models.echo import ECHO_PRESETS
    from .models.hf_local import LOCAL_PRESETS
    from .pricing import ANTHROPIC_PRICES, GOOGLE_PRICES, OPENAI_PRICES

    keys = missing_keys()
    print("Model backends and example specs\n")
    print(f"  {'backend':<12} {'credentials':<12} example")
    print(f"  {'-' * 12} {'-' * 12} {'-' * 40}")
    rows = [
        ("openai", keys["openai"], "openai:gpt-4o"),
        ("anthropic", keys["anthropic"], "anthropic:claude-haiku-4-5"),
        ("google", keys["google"], "google:gemini-2.5-flash"),
        ("hf", "n/a", "hf:HuggingFaceTB/SmolVLM-500M-Instruct"),
        ("echo", "n/a", "echo:demo"),
    ]
    for backend, has_key, example in rows:
        status = "ready" if has_key in (True, "n/a") else "MISSING KEY"
        print(f"  {backend:<12} {status:<12} {example}")

    print("\nOffline echo presets (no key, no GPU, deterministic). These are real")
    print("models with different behaviour, not aliases of one another:\n")
    for name, options in ECHO_PRESETS.items():
        knobs = " ".join(f"{k}={v:g}" for k, v in sorted(options.items()))
        print(f"  echo:{name:<12} {knobs}")
    print("\n  oracle is a CI ceiling (accuracy 1.0); bad is a CI floor.")

    print("\nLocal presets that fit a small GPU:")
    for name, spec in LOCAL_PRESETS.items():
        print(f"  {name:<16} {spec['model']}")
        print(f"  {'':<16} {spec['note']}")

    print("\nKnown list prices, USD per 1M tokens (prompt, completion).")
    print("Snapshot 2026-09-26; override per model with cost_in / cost_out in")
    print("configs/models.yaml, or in a model mapping's `cost_per_million`.\n")
    for label, table in (
        ("openai", OPENAI_PRICES),
        ("anthropic", ANTHROPIC_PRICES),
        ("google", GOOGLE_PRICES),
    ):
        print(f"  {label}:")
        for model, (in_rate, out_rate) in sorted(table.items()):
            print(f"    {model:<28} ${in_rate:>7.2f} ${out_rate:>8.2f}")
    return 0


def cmd_tasks(args: argparse.Namespace) -> int:
    from .prompts import TEMPLATES
    from .tasks import MCQTask, POPETask

    print("Tasks\n")
    for cls in (MCQTask, POPETask):
        print(f"  {cls.name:<8} {cls.description}")
        print(f"  {'':<8} default reply cap: {cls.default_max_tokens} tokens")
    print(f"\n  Names accepted: {', '.join(available_tasks())}")

    print("\nPrompt templates\n")
    for name, template in TEMPLATES.items():
        print(f"  {name}")
        print(f"    instruction: {template.instruction}")
    return 0


def cmd_keys(args: argparse.Namespace) -> int:
    print("Provider credentials\n")
    for provider, present in missing_keys().items():
        mark = "found" if present else "not set"
        print(f"  {provider:<12} {mark}")
    print("\nSet one in .env (copy .env.example) or export it in your shell.")
    print("The harness loads .env from the project root on start.")
    return 0


def cmd_fixtures(args: argparse.Namespace) -> int:
    from .data import synthetic

    out = Path(args.out)
    n = args.n_items
    seed = args.seed

    mcq = synthetic.generate_mcq(out, n_items=n, seed=seed, max_objects=args.max_objects)
    mcq_path = synthetic.write_jsonl(mcq, out / "mcq.jsonl")
    pope = synthetic.generate_pope(out, n_items=n, seed=seed + 1)
    pope_path = synthetic.write_jsonl(pope, out / "pope.jsonl")

    print(f"Wrote {len(mcq)} MCQ items  -> {mcq_path}")
    print(f"Wrote {len(pope)} POPE items -> {pope_path}")
    print(f"Images: {out / 'images'}")
    print(f"\nSeed {seed}: regeneration is byte-identical, so these fixtures are")
    print("safe to commit and safe to diff.")
    print("\nNext:  vlmeval run --model echo:demo --task mcq --source synthetic_mcq")
    return 0


def _specs_for(args: argparse.Namespace) -> list[str]:
    """Resolve the model specs a run should use, in order."""
    specs = list(getattr(args, "models", None) or [])
    if not specs and getattr(args, "model", None):
        specs = [args.model]
    return specs or ["echo:demo"]


def _cost_estimate(args: argparse.Namespace, n_jobs: int) -> float | None:
    """Rough worst-case spend, so --dry-run can warn before anything is sent.

    Prices the full per-request token cap in and out, which over- rather than
    under-estimates. A dry run that quoted an optimistic number would be worse
    than no number. Local and unknown models price at zero, so this only speaks
    when it actually knows a rate.
    """
    from .pricing import lookup

    per_request = getattr(args, "max_tokens", None) or 4096
    total = 0.0
    priced = False
    for spec in _specs_for(args):
        backend, _, model_id = spec.partition(":")
        if not model_id:
            continue
        cost_in, cost_out = lookup(backend, model_id)
        if cost_in or cost_out:
            priced = True
        # Rates are quoted per million tokens.
        total += (per_request * (cost_in + cost_out) / 1_000_000) * n_jobs
    return total if priced else None


def _describe_task(task: Any) -> str:
    """One readable line describing a task, hiding options that are defaults.

    A dry run is read by a person deciding whether to spend money, so the
    interesting settings are the ones that change the workload: how many
    permutations, what cap on the reply.
    """
    info = task.describe()
    parts = [str(info.get("description", info.get("task", "?")))]
    options = info.get("options") or {}
    if options.get("num_permutations"):
        parts.append(f"{options['num_permutations']} permutations/item")
    if options.get("seed") is not None:
        parts.append(f"seed {options['seed']}")
    if info.get("style") and info["style"] != "plain":
        parts.append(f"style {info['style']}")
    if info.get("max_tokens"):
        parts.append(f"<= {info['max_tokens']} reply tokens")
    return f"{info.get('task', '?')} ({', '.join(parts)})"


def cmd_dry_run(args: argparse.Namespace) -> int:
    """Show exactly what a real run would do, then stop.

    Worth having as a first-class flag rather than a script: the expensive
    mistakes in an evaluation -- wrong dataset, wrong task, a prompt that does
    not list the options -- are all visible here for free.
    """
    examples, dataset = _load_examples(args)
    if not examples:
        print(f"No examples loaded from {args.source}: {args.path}", file=sys.stderr)
        return 1

    task = _build_task(args)
    specs = _specs_for(args)
    try:
        jobs_n = sum(len(task.queries(e)) for e in examples)
    except Exception as exc:  # noqa: BLE001 - surface the misconfiguration now
        print(f"Cannot build prompts for this task/dataset combination: {exc}", file=sys.stderr)
        return 1

    print("Dry run: no model was contacted and nothing was charged.\n")
    print(f"  models    {', '.join(specs)}")
    print(f"  task      {_describe_task(task)}")
    print(f"  source    {args.source}  path={args.path}")
    print(f"  dataset   {dataset}  ({len(examples)} items, {jobs_n} requests)")
    print(f"  out       {args.out}")
    for label, value in (
        ("--max-usd", getattr(args, "max_usd", None)),
        ("--max-tokens-total", getattr(args, "max_tokens_total", None)),
        ("--max-requests", getattr(args, "max_requests", None)),
    ):
        if value is not None:
            print(f"  budget    {label} {value}")

    estimate = _cost_estimate(args, jobs_n)
    if estimate:
        print(f"  estimate  <= ${estimate:.2f} worst case at {args.max_tokens or 4096} tokens/request")

    example = examples[0]
    query = task.queries(example)[0]
    print(f"\n--- first prompt ({example.uid}) ---")
    print(query.prompt)
    print("--- end prompt ---\n")
    print(f"Expected reply: a single letter among "
          f"{sorted(query.decode or {}) or 'the offered options'}"
          f"{f', gold {query.gold}' if query.gold else ''}.")
    print("\nRemove --dry-run to run for real.")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    load_env()
    if not _resolve_config(args):
        return 1
    if not args.model and not getattr(args, "_spec_override", None):
        raise SystemExit(
            "No model given. Use --model backend:id (e.g. --model openai:gpt-4o), "
            "or --config configs/models.yaml --preset <name>. "
            "Run 'vlmeval models' to see what is available."
        )
    if getattr(args, "dry_run", False):
        return cmd_dry_run(args)
    examples, dataset = _load_examples(args)
    if not examples:
        print(f"No examples loaded from {args.source}: {args.path}", file=sys.stderr)
        return 1
    task = _build_task(args)
    spec = getattr(args, "_spec_override", None) or args.model
    try:
        model = build_model(spec)
    except (ValueError, TypeError) as exc:
        # A malformed or unavailable spec is a configuration mistake, not a
        # crash. `benchmark` already reports these per model and carries on;
        # `run` reports the one it was given.
        print(f"Cannot build model from {spec!r}: {exc}", file=sys.stderr)
        return 1
    runner = _runner_for(args, model, task)

    try:
        result = runner.run(examples, dataset=dataset)
    finally:
        model.close()
    run_dir = _write(args, result)
    _print_summary(result, run_dir, args.quiet)
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    load_env()
    from .report import write_leaderboard

    if not _resolve_config(args):
        return 1
    if not args.models:
        raise SystemExit(
            "No models given. Use --models backend:id [backend:id ...], "
            "or --config configs/models.yaml --preset <name>."
        )
    if getattr(args, "dry_run", False):
        return cmd_dry_run(args)
    examples, dataset = _load_examples(args)
    if not examples:
        print(f"No examples loaded from {args.source}: {args.path}", file=sys.stderr)
        return 1

    results = []

    failures: list[tuple[str, str]] = []
    for spec in args.models:
        try:
            model = build_model(spec)
        except Exception as exc:  # noqa: BLE001 - report and continue with the rest
            failures.append((spec, str(exc)))
            print(f"  skip {spec}: {exc}", file=sys.stderr)
            continue
        task = _build_task(args)
        runner = _runner_for(args, model, task)
        if not args.quiet:
            print(f"\n=== {model.name} ===")
        try:
            result = runner.run(examples, dataset=dataset)
        except Exception as exc:  # noqa: BLE001 - one bad model must not kill the sweep
            failures.append((spec, str(exc)))
            print(f"  FAILED {spec}: {exc}", file=sys.stderr)
            continue
        finally:
            model.close()
        results.append(result)
        _write(args, result)
        if not args.quiet:
            _print_summary(result, None, quiet=False, brief=True)

    if not results:
        print("No runs completed.", file=sys.stderr)
        for spec, err in failures:
            print(f"  {spec}: {err}", file=sys.stderr)
        return 1

    board = write_leaderboard(results, out_dir=args.out)
    print(f"\nLeaderboard -> {board}")
    if failures:
        print(f"\n{len(failures)} model(s) did not complete:", file=sys.stderr)
        for spec, err in failures:
            print(f"  {spec}: {err}", file=sys.stderr)
    return 0


def cmd_summarize(args: argparse.Namespace) -> int:
    from .report import write_leaderboard

    results = _load_results(args.results)
    if not results:
        print(f"No metrics.json files under {args.results}", file=sys.stderr)
        return 1
    board = write_leaderboard(results, out_dir=args.out)
    print(board.read_text(encoding="utf-8"))
    print(f"Written to {board}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from .compare import compare_runs

    results = _load_results(args.results)
    if len(results) < 2:
        print(f"Need at least 2 runs to compare, found {len(results)}.", file=sys.stderr)
        return 1
    report = compare_runs(results, baseline=args.baseline)
    out = Path(args.out) / "comparison.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"Written to {out}")
    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    from .plots import plot_all

    results = _load_results(args.results)
    if not results:
        print(f"No metrics.json files under {args.results}", file=sys.stderr)
        return 1
    try:
        paths = plot_all(results, out_dir=args.out, metric=args.metric)
    except ImportError as exc:
        # Plotting is the one optional feature that needs a third-party
        # dependency. Say so plainly instead of printing a traceback: the
        # evaluation itself already succeeded and its results are on disk.
        print(f"Cannot draw figures: {exc}", file=sys.stderr)
        print("Everything else still works; the metrics are in the reports.", file=sys.stderr)
        return 1
    for path in paths:
        print(f"Wrote {path}")
    return 0


# --- helpers ---------------------------------------------------------------


def _load_results(root: str) -> list[Any]:
    """Rebuild every run under `root`, with item-level detail when present."""
    from .report import load_run

    out: list[Any] = []
    for path in sorted(Path(root).rglob("metrics.json")):
        try:
            out.append(load_run(path.parent))
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            print(f"skipping {path}: {exc}", file=sys.stderr)
    return out


def _write(args: argparse.Namespace, result: Any) -> Path:
    from .report import write_run

    return write_run(result, out_dir=args.out)


def _print_summary(
    result: Any, run_dir: Path | None, quiet: bool, brief: bool = False
) -> None:
    if quiet:
        return
    primary = result.primary()
    print()
    print(f"  {result.task} / {result.dataset} / {result.model}")
    if primary:
        print(f"  {primary.name}: {primary.fmt(pct=True)}")
    extra = [
        f"{k}={v.value:.3f}"
        for k, v in result.metrics.items()
        if k in {"f1", "precision", "recall", "yes_ratio", "consistency", "unparsed_rate"}
        and v.value is not None
    ]
    if extra:
        print(f"  {' '.join(extra)}")
    counts = result.counts
    print(
        f"  n={counts.get('n_requests', 0)} "
        f"failed={counts.get('n_failed', 0)} "
        f"unparsed={counts.get('n_unparsed', 0)} "
        f"cost=${counts.get('est_cost_usd', 0.0):.4f} "
        f"time={counts.get('elapsed_s', 0):.1f}s"
    )
    if counts.get("stopped_reason"):
        print(f"  STOPPED EARLY: {counts['stopped_reason']}")
    if run_dir and not brief:
        print(f"  report: {run_dir / 'summary.md'}")


def _configure_console() -> None:
    """Make stdout/stderr UTF-8 so reports survive a non-UTF-8 console.

    The reports contain em-dashes and the prompts can contain non-ASCII text. On
    a Windows console defaulting to a legacy codepage those become mojibake or
    raise ``UnicodeEncodeError``. ``errors="replace"`` guarantees a readable
    character instead of a crash; the files written to disk are always UTF-8
    regardless of what the console can display.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover - exotic stream
                pass


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    _configure_console()
    parser = argparse.ArgumentParser(
        prog="vlmeval",
        description="Reproducible evaluation of vision-language models.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("models", help="List model backends, presets and prices.")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("tasks", help="List tasks and prompt templates.")
    p.set_defaults(func=cmd_tasks)

    p = sub.add_parser("keys", help="Check which provider credentials are available.")
    p.set_defaults(func=cmd_keys)

    p = sub.add_parser("fixtures", help="Generate the offline synthetic benchmark.")
    p.add_argument("--out", default="data/samples/synthetic_v1")
    p.add_argument("--n-items", type=int, default=120, dest="n_items")
    p.add_argument("--seed", type=int, default=20260926)
    p.add_argument("--max-objects", type=int, default=6, dest="max_objects")
    p.set_defaults(func=cmd_fixtures)

    p = sub.add_parser("run", help="Evaluate one model on one task and dataset.")
    p.add_argument("--model", default=None,
                   help="e.g. openai:gpt-4o, hf:<repo>, echo:demo. May come from "
                        "--config/--preset instead.")

    p.add_argument("--task", default="mcq", help="mcq or pope")
    p.add_argument("--source", default="synthetic_mcq",
                   help="synthetic_mcq, synthetic_pope, jsonl, dir, or hf")
    p.add_argument("--path", default=None, help="Dataset path, or Hub name when source=hf")
    p.add_argument("--template", default=None, help="Prompt template override.")
    p.add_argument("--permutations", type=int, default=0,
                   help="Extra option-order permutations per MCQ item (costs more).")
    p.add_argument("--seed", type=int, default=None, help="Seed for permutations.")
    _add_config(p)
    _add_common(p)
    _add_budget(p)
    _add_dry_run(p)
    p.set_defaults(func=cmd_run)


    p = sub.add_parser("benchmark", help="Evaluate several models and build a leaderboard.")
    p.add_argument("--models", nargs="+", default=None,
                   help="Model specs. May come from --config/--preset instead.")

    p.add_argument("--task", default="mcq")
    p.add_argument("--source", default="synthetic_mcq")
    p.add_argument("--path", default=None)
    p.add_argument("--template", default=None)
    p.add_argument("--permutations", type=int, default=0)
    p.add_argument("--seed", type=int, default=None)
    _add_config(p)
    _add_common(p)
    _add_budget(p)
    _add_dry_run(p)
    p.set_defaults(func=cmd_benchmark)


    p = sub.add_parser("summarize", help="Build a leaderboard from existing runs.")
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="results")
    p.set_defaults(func=cmd_summarize)

    p = sub.add_parser("compare", help="Paired statistical comparison between runs.")
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="results")
    p.add_argument("--baseline", default=None, help="Model name to compare everything against.")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("plot", help="Render figures from existing runs.")
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="results/figures")
    p.add_argument("--metric", default=None, help="Metric to plot (default: headline).")
    p.set_defaults(func=cmd_plot)

    args = parser.parse_args(argv)
    # Kept on the namespace so preset resolution can see which flags were typed.
    args._argv = list(argv if argv is not None else sys.argv[1:])
    args._parser = parser
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
