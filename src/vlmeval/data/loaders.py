"""Dataset loading and normalisation.

Three sources, one output type:

* ``synthetic`` -- generated locally by :mod:`vlmeval.data.synthetic`. No
  network, no licence, exact ground truth.
* ``jsonl`` -- a directory of JSON Lines files plus an images folder. The
  escape hatch for private or hand-built data, and the format this project's own
  fixtures use.
* ``hf`` -- a Hugging Face Hub dataset, via per-benchmark adapters that map
  each dataset's own column names and answer conventions onto
  :class:`~vlmeval.types.Example`.

Real benchmarks disagree about almost everything: the option field is called
``options``/``choices``/``candidates``, the answer is a letter, an index, or the
option text, and images arrive as paths, bytes or PIL objects. Each adapter
encodes those decisions once, with the reasoning in its docstring, so a dataset
quirk is fixed in one place instead of surfacing as a mysterious accuracy drop.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from ..prompts import LETTERS
from ..types import Example

#: Maps a gold answer onto a letter, for datasets that store the option text.
_TO_LETTER_CACHE: dict[tuple[int, str], str] = {}


def gold_to_letter(label: Any, choices: list[str] | None) -> str | None:
    """Normalise a gold answer to a letter, given the option list.

    Handles the three conventions in the wild: a bare letter, a zero- or
    one-based index, and the option text itself. Returns ``None`` when the
    answer cannot be resolved, so the item shows up as unparsed instead of
    being silently scored wrong.
    """
    if label is None:
        return None
    if choices is None:
        return str(label).strip() or None

    text = str(label).strip()
    if len(text) == 1 and text.upper() in LETTERS[: len(choices)]:
        return text.upper()

    if text.isdigit():
        idx = int(text)
        if 0 <= idx < len(choices):
            return LETTERS[idx]
        if 1 <= idx <= len(choices):
            return LETTERS[idx - 1]

    folded = text.casefold()
    for i, choice in enumerate(choices):
        if str(choice).strip().casefold() == folded:
            return LETTERS[i]
    return None


def _as_choices(value: Any) -> list[str] | None:
    """Coerce a dataset's option field into a list of strings."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return None
        value = parsed
    if isinstance(value, dict):
        # Some datasets use {"text": ..., "label": ...} or {"A": ..., ...}.
        if "text" in value:
            return [str(value["text"])]
        keys = [k for k in LETTERS if k in value]
        if keys:
            return [str(value[k]) for k in keys]
        return [str(v) for v in value.values()]
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            if isinstance(item, dict):
                out.append(str(item.get("text", item.get("answer", item.get("choice", "")))))
            else:
                out.append(str(item))
        return out
    return None


def load_jsonl(path: str | Path, image_root: str | Path | None = None) -> list[Example]:
    """Load examples from a JSON Lines file.

    Image paths are resolved relative to `image_root` when given, otherwise
    relative to the JSONL file's own directory. Relative paths in a committed
    fixture therefore keep working after a clone.
    """
    path = Path(path)
    root = Path(image_root) if image_root else path.parent
    out: list[Example] = []
    with open(path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} is not valid JSON: {exc}") from exc
            image = raw.get("image")
            if isinstance(image, str) and not Path(image).is_absolute():
                image = root / image
            choices = _as_choices(raw.get("choices", raw.get("options")))
            out.append(
                Example(
                    uid=str(raw.get("uid") or raw.get("id") or f"{path.stem}_{lineno}"),
                    image=image,
                    question=str(raw.get("question") or raw.get("prompt") or ""),
                    choices=choices,
                    label=gold_to_letter(raw.get("label", raw.get("answer")), choices)
                    if choices
                    else _normalise_binary(raw.get("label", raw.get("answer"))),
                    category=raw.get("category"),
                    meta=raw.get("meta") or {},
                )
            )
    return out


def _normalise_binary(label: Any) -> str | None:
    """Map a yes/no gold answer onto ``"yes"``/``"no"``."""
    if label is None:
        return None
    text = str(label).strip().lower()
    if text in {"yes", "y", "true", "1"}:
        return "yes"
    if text in {"no", "n", "false", "0"}:
        return "no"
    return text or None


def load_directory(
    directory: str | Path,
    pattern: str = "*.jsonl",
    image_root: str | Path | None = None,
) -> list[Example]:
    """Load and concatenate every JSONL file in `directory`, sorted by name."""
    directory = Path(directory)
    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No {pattern} files in {directory}. Generate the fixtures first: "
            "python scripts/make_fixtures.py"
        )
    out: list[Example] = []
    for file in files:
        out.extend(load_jsonl(file, image_root=image_root))
    return out


def _hf_image_to_path(image: Any, out_dir: Path, uid: str) -> Path:
    """Persist a Hub image to `out_dir` and return the local path.

    Caching to disk rather than keeping bytes in memory means a run can be
    resumed after a restart, and the exact bytes a model saw are preserved for
    later inspection.
    """
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{uid}.png"
    if path.exists():
        return path
    if isinstance(image, dict):
        # datasets >= 3 returns {"bytes": ..., "path": ...} for image features.
        image = image.get("bytes") or image.get("path")
    if isinstance(image, (bytes, bytearray)):
        from io import BytesIO

        img = Image.open(BytesIO(bytes(image)))
    elif isinstance(image, Image.Image):
        img = image
    else:
        img = Image.open(image)
    img.convert("RGB").save(path, format="PNG")
    return path


# --- Per-benchmark adapters -------------------------------------------------
# Each takes a raw dataset row plus an image cache directory and returns an
# Example. The docstring records the dataset's conventions, because those are
# the facts that break when a benchmark is re-released.


def adapt_mmlu_style(
    row: dict[str, Any],
    image_dir: Path,
    uid: str,
    *,
    question_key: str = "question",
    options_key: str = "options",
    answer_key: str = "answer",
    category_key: str | None = "category",
) -> Example:
    """MMMU / MMBench style rows.

    Conventions: ``options`` is a list of 2-10 strings, ``answer`` is the index
    as an *integer* (not a letter), and the discipline lives in a subject field
    used as the report category. MMBench's published ``answer`` is a letter
    instead, which :func:`gold_to_letter` handles either way.
    """
    choices = _as_choices(row.get(options_key))
    raw_answer = row.get(answer_key)
    if isinstance(raw_answer, int) and 0 <= raw_answer < len(choices or []):
        label = LETTERS[raw_answer]
    else:
        label = gold_to_letter(raw_answer, choices)
    return Example(
        uid=uid,
        image=_hf_image_to_path(row["image"], image_dir, uid),
        question=str(row.get(question_key, "")).strip(),
        choices=choices,
        label=label,
        category=str(row.get(category_key)) if category_key and row.get(category_key) else None,
        meta={"source": "hf", "raw_answer": raw_answer},
    )


def adapt_pope_hf(
    row: dict[str, Any],
    image_dir: Path,
    uid: str,
    *,
    question_key: str = "question",
    answer_key: str = "answer",
    category_key: str = "category",
) -> Example:
    """POPE rows from the Hub.

    The published POPE sets use ``answer`` as the *string* ``"yes"``/``"no"``
    and carry the split name in ``category`` (random / popular / adversarial),
    which maps directly onto this harness's per-split reporting.
    """
    return Example(
        uid=uid,
        image=_hf_image_to_path(row["image"], image_dir, uid),
        question=str(row.get(question_key, "")).strip(),
        choices=None,
        label=_normalise_binary(row.get(answer_key)),
        category=str(row.get(category_key)) if row.get(category_key) else None,
        meta={"source": "hf", "target": row.get("object") or row.get("category")},
    )


def adapt_short_answer(
    row: dict[str, Any],
    image_dir: Path,
    uid: str,
    *,
    question_key: str = "question",
    answer_key: str = "answer",
    category_key: str | None = None,
) -> Example:
    """Short-answer rows (ChartQA / DocVQA / TextVQA style).

    Answers may be a string or a list of acceptable variants; the list is kept
    whole in ``meta`` so a future lenient-scoring task can use it. Binary
    normalisation is applied because some of these rows are themselves yes/no.
    """
    raw = row.get(answer_key)
    answers = raw if isinstance(raw, list) else [raw]
    primary = answers[0] if answers else None
    return Example(
        uid=uid,
        image=_hf_image_to_path(row["image"], image_dir, uid),
        question=str(row.get(question_key, "")).strip(),
        choices=None,
        label=_normalise_binary(primary),
        category=str(row.get(category_key)) if category_key and row.get(category_key) else None,
        meta={"source": "hf", "answers": answers},
    )


#: Registry of Hub datasets this project knows how to normalise.
HF_DATASETS: dict[str, dict[str, Any]] = {
    "mmmu": {
        "repo": "lmms-lab/MMMU",
        "split": "validation",
        "config": "all",
        "adapter": adapt_mmlu_style,
        "task": "mcq",
        "limit_note": "MMMU validation is ~900 items across 30 subjects.",
    },
    "mmbench_en": {
        "repo": "lmms-lab/MMBench_EN",
        "split": "dev",
        "config": "en",
        "adapter": adapt_mmlu_style,
        "task": "mcq",
        "limit_note": "Use the dev split: test labels are withheld.",
    },
    "pope": {
        "repo": "lmms-lab/POPE",
        "split": "test",
        "adapter": adapt_pope_hf,
        "task": "pope",
        "limit_note": "Three splits (random/popular/adversarial) in one file.",
    },
    "chartqa": {
        "repo": "HuggingFaceM4/ChartQA",
        "split": "test",
        "adapter": adapt_short_answer,
        "task": "binary",
        "limit_note": "Short answers; needs the lenient short-answer scorer.",
    },
    "textvqa": {
        "repo": "lmms-lab/textvqa",
        "split": "val",
        "adapter": adapt_short_answer,
        "task": "binary",
        "limit_note": "Scene-text reading; needs the lenient short-answer scorer.",
    },
}


def load_hf(
    name: str,
    *,
    limit: int | None = None,
    image_dir: str | Path = "data/cache/hf_images",
    repo: str | None = None,
    split: str | None = None,
    config: str | None = None,
    streaming: bool = False,
) -> list[Example]:
    """Load a Hub dataset by short name, normalised to `Example` objects.

    Args:
        name: Key in :data:`HF_DATASETS`, or a full ``"repo:split"`` string for
            anything not registered.
        limit: Cap on rows, for a quick smoke run. Always set this before
            spending money on a full pass.
        image_dir: Where decoded images are cached.
        repo / split / config: Override the registered defaults.
        streaming: Avoid downloading the whole dataset. Useful for large sets
            when combined with `limit`.

    Raises:
        ImportError: If ``datasets`` is not installed, with the fix in the
            message rather than a bare module error.
    """
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ImportError(
            "Loading Hub datasets needs the datasets package: pip install -e '.[data]'"
        ) from exc

    spec = HF_DATASETS.get(name, {})
    repo = repo or spec.get("repo")
    split = split or spec.get("split", "test")
    config = config if config is not None else spec.get("config")
    adapter: Callable[..., Example] = spec.get("adapter", adapt_short_answer)
    image_path = Path(image_dir)

    if repo is None:
        raise ValueError(
            f"Unknown dataset {name!r}. Known: {sorted(HF_DATASETS)}; "
            "or pass repo='org/dataset' explicitly."
        )

    dataset = hf_load_dataset(repo, config, split=split, streaming=streaming)
    out: list[Example] = []
    for i, row in enumerate(dataset):
        if limit is not None and i >= limit:
            break
        row = dict(row)
        if "image" not in row:
            continue
        out.append(adapter(row, image_path, f"{name}_{i:06d}"))
    if not out:
        raise ValueError(
            f"No usable rows from {repo} ({config or 'default'}/{split}). "
            "Check the column names against the dataset card."
        )
    return out


def recommended_task(name: str) -> str | None:
    """The task a registered dataset should be run with, if known."""
    return HF_DATASETS.get(name, {}).get("task")


def load(
    source: str,
    *,
    path: str | Path | None = None,
    limit: int | None = None,
    **kwargs: Any,
) -> list[Example]:
    """Load examples from any supported source.

    Args:
        source: ``"synthetic_mcq"``, ``"synthetic_pope"``, ``"jsonl"``,
            ``"dir"`` or ``"hf"``.
        path: Filesystem location, required for the local sources.
        limit: Cap on examples, applied after loading.
    """
    if source == "synthetic_mcq":
        examples = _cached_synthetic("mcq", path, limit, kwargs)
    elif source == "synthetic_pope":
        examples = _cached_synthetic("pope", path, limit, kwargs)
    elif source == "jsonl":
        examples = load_jsonl(path, image_root=kwargs.get("image_root"))
    elif source == "dir":
        examples = load_directory(
            path,
            pattern=kwargs.get("pattern", "*.jsonl"),
            image_root=kwargs.get("image_root"),
        )
    elif source == "hf":
        examples = load_hf(str(path or kwargs.get("name", "")), limit=limit, **kwargs)
    else:
        raise ValueError(
            f"Unknown source {source!r}. Use synthetic_mcq, synthetic_pope, jsonl, dir or hf."
        )
    if limit is not None:
        examples = examples[:limit]
    return examples


def _cached_synthetic(
    kind: str, path: str | Path | None, limit: int | None, kwargs: dict[str, Any]
) -> list[Example]:
    """Generate a synthetic dataset once, then reload it from disk.

    Generation is deterministic given the seed, so regenerating would produce
    identical images. Writing them out anyway means a second run reads the same
    bytes the first run did, and a human can look at the images while debugging
    a model failure.
    """
    from . import synthetic

    out_dir = Path(path or f"data/samples/synthetic_v1")
    jsonl = out_dir / f"{kind}.jsonl"
    if jsonl.exists():
        return load_jsonl(jsonl, image_root=out_dir)

    n_items = kwargs.get("n_items", 120)
    if limit is not None:
        n_items = min(n_items, limit)
    seed = kwargs.get("seed", 20260926)
    if kind == "mcq":
        examples = synthetic.generate_mcq(out_dir, n_items=n_items, seed=seed)
    else:
        examples = synthetic.generate_pope(out_dir, n_items=n_items, seed=seed)
    synthetic.write_jsonl(examples, jsonl)
    return load_jsonl(jsonl, image_root=out_dir)
