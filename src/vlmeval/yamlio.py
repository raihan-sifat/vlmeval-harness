"""Config file loading.

YAML is the config format because these files are edited by hand and reviewed in
diffs. PyYAML is a declared dependency, but the loader does not *require* it:
when it is missing, a small reader handles the subset these configs actually
use (nested mappings, lists of mappings, scalars, comments, quoted strings).

That fallback exists so the harness stays runnable in a locked-down or offline
environment -- an evaluation tool that refuses to start because a YAML parser is
absent is a worse tool. It is deliberately limited: anchors, multi-document
streams, flow-style nesting and complex keys are not supported and raise rather
than being silently misread.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML or JSON config into a dict.

    Args:
        path: File to read. ``.json`` is parsed with the stdlib; everything
            else goes through PyYAML, or the fallback reader.

    Returns:
        The parsed mapping. A file whose top level is not a mapping raises,
        because every consumer here expects named keys and a list would fail
        later with a much less obvious message.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config not found: {path}. Run from the repository root, or pass an absolute path."
        )
    text = path.read_text(encoding="utf-8")

    if path.suffix == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml
        except ImportError:
            data = _parse_simple_yaml(text, source=str(path))
        else:
            data = yaml.safe_load(text)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(
            f"{path}: top level must be a mapping of keys to values, got {type(data).__name__}"
        )
    return data


def _parse_simple_yaml(text: str, source: str = "<string>") -> Any:
    """A minimal YAML reader for the subset used by this project's configs.

    Supported: comments, nested block mappings by indentation, block sequences
    (``- scalar`` and ``- key: value``), quoted and bare scalars, and ``null`` /
    booleans / integers / floats. Anything else raises with a line number rather
    than guessing.
    """
    lines: list[tuple[int, int, str]] = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if raw.lstrip().startswith("---"):
            raise ValueError(
                f"{source}:{lineno}: multi-document YAML is not supported by the "
                "fallback parser. Install PyYAML for full YAML support."
            )
        indent = len(raw) - len(raw.lstrip(" "))
        lines.append((lineno, indent, raw.strip()))

    if not lines:
        return {}
    value, index = _parse_block(lines, 0, lines[0][1], source)
    if index != len(lines):
        raise ValueError(f"{source}:{lines[index][0]}: unexpected indentation")
    return value


def _parse_block(
    lines: list[tuple[int, int, str]], index: int, indent: int, source: str
) -> tuple[Any, int]:
    """Parse one block at `indent`, returning the value and the next index."""
    if lines[index][2].startswith("- "):
        return _parse_sequence(lines, index, indent, source)
    return _parse_mapping(lines, index, indent, source)


def _parse_mapping(
    lines: list[tuple[int, int, str]], index: int, indent: int, source: str
) -> tuple[dict[str, Any], int]:
    out: dict[str, Any] = {}
    while index < len(lines):
        lineno, line_indent, content = lines[index]
        if line_indent < indent:
            break
        if line_indent > indent:
            raise ValueError(f"{source}:{lineno}: unexpected indentation")
        key, sep, rest = _split_key(content, source, lineno)
        if not sep:
            raise ValueError(f"{source}:{lineno}: expected 'key: value', got {content!r}")
        rest = rest.strip()
        index += 1
        if rest:
            out[key] = _scalar(rest)
            continue
        # A nested block follows if the next line is deeper, or is a sequence
        # at the same indent (valid YAML for `key:` followed by `- item`).
        if index < len(lines) and (
            lines[index][1] > indent
            or (lines[index][1] == indent and lines[index][2].startswith("- "))
        ):
            child_indent = lines[index][1]
            out[key], index = _parse_block(lines, index, child_indent, source)
        else:
            out[key] = None
    return out, index


def _parse_sequence(
    lines: list[tuple[int, int, str]], index: int, indent: int, source: str
) -> tuple[list[Any], int]:
    out: list[Any] = []
    while index < len(lines):
        lineno, line_indent, content = lines[index]
        if line_indent < indent or not content.startswith("- "):
            break
        if line_indent > indent:
            raise ValueError(f"{source}:{lineno}: unexpected indentation in sequence")
        body = content[2:].strip()
        index += 1
        key, sep, rest = _split_key(body, source, lineno)
        if not sep:
            out.append(_scalar(body))
            continue
        # `- key: value` opens a mapping whose remaining keys are indented to
        # where the key started, not to the dash.
        item_indent = line_indent + 2
        entry: dict[str, Any] = {}
        rest = rest.strip()
        if rest:
            entry[key] = _scalar(rest)
        elif index < len(lines) and lines[index][1] > item_indent:
            entry[key], index = _parse_block(lines, index, lines[index][1], source)
        else:
            entry[key] = None
        while index < len(lines) and lines[index][1] == item_indent:
            more, index = _parse_mapping(lines, index, item_indent, source)
            entry.update(more)
        out.append(entry)
    return out, index


def _split_key(content: str, source: str, lineno: int) -> tuple[str, bool, str]:
    """Split ``key: value`` on the first colon that is not inside quotes."""
    in_single = in_double = False
    for i, ch in enumerate(content):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == ":" and not in_single and not in_double:
            after = content[i + 1 :]
            if after and not after.startswith(" "):
                continue
            return content[:i].strip().strip("\"'"), True, after
    return content, False, ""


def _scalar(token: str) -> Any:
    """Convert a YAML scalar token to a Python value."""
    token = token.strip()
    if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'":
        return token[1:-1]
    # Strip a trailing comment from an unquoted scalar.
    if " #" in token:
        token = token.split(" #", 1)[0].strip()
    lowered = token.lower()
    if lowered in {"null", "~", ""}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if token.startswith("[") and token.endswith("]"):
        inner = token[1:-1].strip()
        return [_scalar(p) for p in inner.split(",")] if inner else []
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        return token
