"""Environment configuration.

Loads ``.env`` from the repository root without requiring ``python-dotenv``: the
file format is a dozen lines of ``KEY=value`` and a dependency is not worth it.
Existing environment variables always win, so an exported key or a CI secret
overrides the file, and ``.env`` can stay as a local convenience rather than
something that has to be kept in sync with the real environment.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

#: Variables the harness knows how to use, with whether each is required for
#: its provider. Only these are loaded, so a stray secret in ``.env`` is not
#: pulled into the process environment.
KNOWN_VARS = {
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "HF_TOKEN",
}

_ENV_LOADED = False


def repo_root() -> Path:
    """The project root, i.e. the directory containing ``pyproject.toml``."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return here.parents[2]


def load_env(path: str | Path | None = None, *, force: bool = False) -> dict[str, str]:
    """Load known variables from a ``.env`` file without clobbering the env.

    Handles the shapes that actually appear in a hand-written ``.env``:
    ``KEY=value``, ``export KEY=value``, quoted values, ``#`` comments and blank
    lines. Malformed lines are skipped rather than raising -- a typo in an
    optional variable should not stop an evaluation.
    """
    global _ENV_LOADED
    if _ENV_LOADED and not force:
        return {}
    _ENV_LOADED = True

    env_path = Path(path) if path else repo_root() / ".env"
    if not env_path.exists():
        return {}

    loaded: dict[str, str] = {}
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        if key not in KNOWN_VARS:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if not os.environ.get(key):
            os.environ[key] = value
            loaded[key] = value
    return loaded


def missing_keys() -> dict[str, bool]:
    """Which providers have credentials available, for a friendly startup check."""
    load_env()
    return {
        "openai": bool(os.environ.get("OPENAI_API_KEY")),
        "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "google": bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")),
        "local": True,
        "echo": True,
    }


#: Keys a preset entry may set, beyond the model ``spec`` itself. These are the
#: run-level settings a named entry is allowed to carry, so a config file can
#: describe a complete, repeatable experiment rather than just a model id.
PRESET_KEYS = (
    "task",
    "source",
    "path",
    "template",
    "style",
    "permutations",
    "seed",
    "workers",
    "max_retries",
    "temperature",
    "max_tokens",
    "max_usd",
    "max_tokens_total",
    "max_requests",
    "limit",
)


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a run configuration file.

    Expected shape, matching ``configs/models.yaml``::

        defaults:
          workers: 4
        models:
          - name: echo-strong
            spec: {backend: echo, model: strong}
            task: mcq
            source: synthetic_mcq

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the file has no ``models`` list, or a preset has no name.
            A silently ignored config file is the kind of mistake that gets
            discovered after the credits are spent.
    """
    from .yamlite import YamlSubsetError, load as load_yaml

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Config file not found: {path}. "
            "Pass --config with a path to a YAML file, or drop the flag to use "
            "model specs directly."
        )
    try:
        data = load_yaml(path)
    except YamlSubsetError as exc:
        raise ValueError(f"{path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")

    entries = data.get("models")
    if not isinstance(entries, list) or not entries:
        raise ValueError(
            f"{path}: expected a non-empty 'models:' list. "
            "Each entry needs a 'name' and a 'spec'."
        )

    presets: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: models[{index}] must be a mapping, got {entry!r}")
        name = entry.get("name")
        if not name:
            raise ValueError(f"{path}: models[{index}] is missing a 'name'")
        unknown = set(entry) - {"name", "spec"} - set(PRESET_KEYS)
        if unknown:
            raise ValueError(
                f"{path}: preset {name!r} has unknown key(s) {sorted(unknown)}. "
                f"Allowed: name, spec, {', '.join(PRESET_KEYS)}"
            )
        if not isinstance(entry.get("spec"), dict):
            raise ValueError(
                f"{path}: preset {name!r} needs a 'spec' mapping, "
                "e.g. spec: {backend: openai, model: gpt-4o}"
            )
        presets[str(name)] = entry

    defaults = data.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise ValueError(f"{path}: 'defaults' must be a mapping")

    return {"defaults": defaults, "presets": presets, "path": str(path)}


def resolve_preset(config: dict[str, Any], name: str) -> dict[str, Any]:
    """Merge a preset over the file's ``defaults``.

    The preset wins over ``defaults``, because it is the more specific
    statement. Unknown preset names list the ones that do exist, since a typo
    here is otherwise indistinguishable from a missing file.
    """
    presets = config.get("presets") or {}
    if name not in presets:
        raise ValueError(
            f"No preset named {name!r} in {config.get('path')}. "
            f"Available: {', '.join(sorted(presets))}"
        )
    merged: dict[str, Any] = dict(config.get("defaults") or {})
    merged.update({k: v for k, v in presets[name].items() if k != "name"})
    merged["_name"] = name
    return merged
