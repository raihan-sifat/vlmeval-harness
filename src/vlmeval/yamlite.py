"""A YAML subset loader, so configuration files work with zero dependencies.

The project promises that a fresh clone can run an evaluation with nothing
installed, and `pip install` is not always available. Requiring PyYAML for an
optional config file would break that promise for the sake of a feature most
users meet only after the offline path already works.

So: use PyYAML when it is importable, and otherwise fall back to the small
reader below. It handles the subset the shipped config files actually use --
nested mappings, sequences of mappings, scalars, comments, and quotes -- and it
raises on anything it does not understand rather than guessing. A config file
that silently half-parses is worse than one that refuses.

Anything outside that subset (anchors, multi-line scalars, flow collections,
multiple documents) is genuinely unsupported here; install PyYAML for those.
"""

from __future__ import annotations

from typing import Any

__all__ = ["load", "loads", "YamlSubsetError", "using_pyyaml"]


class YamlSubsetError(ValueError):
    """Raised when the input uses YAML this reader does not implement."""


def using_pyyaml() -> bool:
    """True when the real PyYAML is handling the parse."""
    try:
        import yaml  # noqa: F401
    except ImportError:
        return False
    return True


class _Line:
    """One significant line: its indentation and its content."""

    __slots__ = ("indent", "content", "number")

    def __init__(self, indent: int, content: str, number: int) -> None:
        self.indent = indent
        self.content = content
        self.number = number

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"_Line({self.indent}, {self.content!r}, line {self.number})"


def _strip_comment(text: str) -> str:
    """Remove a trailing ``#`` comment, respecting quotes.

    A ``#`` only starts a comment at the start of the line or after whitespace,
    which is what lets ``model: gpt#4o`` keep its value.
    """
    quote: str | None = None
    for i, ch in enumerate(text):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == "#" and (i == 0 or text[i - 1].isspace()):
            return text[:i]
    return text


def _scan(text: str) -> list[_Line]:
    lines: list[_Line] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        if "\t" in raw[: len(raw) - len(raw.lstrip())]:
            raise YamlSubsetError(
                f"line {number}: tabs cannot be used for indentation in YAML"
            )
        stripped = _strip_comment(raw).rstrip()
        if not stripped.strip():
            continue
        if stripped.lstrip().startswith("---"):
            if lines:
                raise YamlSubsetError(
                    f"line {number}: multiple documents are not supported; "
                    "use one file per configuration"
                )
            continue
        indent = len(stripped) - len(stripped.lstrip(" "))
        lines.append(_Line(indent, stripped.strip(), number))
    return lines


def _scalar(text: str) -> Any:
    text = text.strip()
    if not text:
        return None
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        inner = text[1:-1]
        return inner.replace('\\"', '"').replace("\\'", "'") if text[0] == '"' else inner
    if text[0] in "[{":
        raise YamlSubsetError(
            f"flow collections like {text!r} are not supported; "
            "use block style, or install PyYAML"
        )
    lowered = text.lower()
    if lowered in {"null", "~"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def _is_sequence_item(line: _Line) -> bool:
    return line.content == "-" or line.content.startswith("- ")


def _split_key(content: str) -> tuple[str, str] | None:
    """Split ``key: value`` outside quotes, or return None if not a mapping entry."""
    quote: str | None = None
    for i, ch in enumerate(content):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == ":" and (i + 1 == len(content) or content[i + 1].isspace()):
            return content[:i].strip(), content[i + 1 :].strip()
    return None


def _parse_block(lines: list[_Line], start: int, indent: int) -> tuple[Any, int]:
    if start >= len(lines):
        return None, start
    if _is_sequence_item(lines[start]):
        return _parse_sequence(lines, start, indent)
    return _parse_mapping(lines, start, indent)


def _parse_mapping(lines: list[_Line], start: int, indent: int) -> tuple[dict[str, Any], int]:
    result: dict[str, Any] = {}
    i = start
    while i < len(lines) and lines[i].indent == indent and not _is_sequence_item(lines[i]):
        pair = _split_key(lines[i].content)
        if pair is None:
            raise YamlSubsetError(
                f"line {lines[i].number}: expected 'key: value', got {lines[i].content!r}"
            )
        key, raw = pair
        i += 1
        if raw:
            result[key] = _scalar(raw)
        elif i < len(lines) and lines[i].indent > indent:
            value, i = _parse_block(lines, i, lines[i].indent)
            result[key] = value
        elif i < len(lines) and lines[i].indent == indent and _is_sequence_item(lines[i]):
            # A sequence may be written at the same indent as its key.
            value, i = _parse_sequence(lines, i, indent)
            result[key] = value
        else:
            result[key] = None
    return result, i


def _parse_sequence(lines: list[_Line], start: int, indent: int) -> tuple[list[Any], int]:
    items: list[Any] = []
    i = start
    while i < len(lines) and lines[i].indent == indent and _is_sequence_item(lines[i]):
        line = lines[i]
        rest = line.content[1:].strip()
        if not rest:
            i += 1
            if i < len(lines) and lines[i].indent > indent:
                value, i = _parse_block(lines, i, lines[i].indent)
                items.append(value)
            else:
                items.append(None)
            continue
        if _split_key(rest) is not None:
            # "- key: value" opens a mapping whose remaining keys are indented
            # to the column where `key` starts. Re-indent the dash line to that
            # column so the whole item parses as one block.
            content_col = line.indent + (len(line.content) - len(rest))
            block = [_Line(content_col, rest, line.number)]
            j = i + 1
            while j < len(lines) and lines[j].indent >= content_col:
                block.append(lines[j])
                j += 1
            value, _ = _parse_block(block, 0, content_col)
            items.append(value)
            i = j
        else:
            items.append(_scalar(rest))
            i += 1
    return items, i


def loads(text: str) -> Any:
    """Parse the supported YAML subset, preferring PyYAML when installed."""
    try:
        import yaml
    except ImportError:
        pass
    else:
        try:
            return yaml.safe_load(text)
        except yaml.YAMLError as exc:  # pragma: no cover - depends on env
            raise YamlSubsetError(f"invalid YAML: {exc}") from exc

    lines = _scan(text)
    if not lines:
        return None
    value, consumed = _parse_block(lines, 0, lines[0].indent)
    if consumed != len(lines):
        leftover = lines[consumed]
        raise YamlSubsetError(
            f"line {leftover.number}: unexpected indentation {leftover.content!r}"
        )
    return value


def load(path: Any) -> Any:
    """Parse a YAML file, preferring PyYAML when installed."""
    from pathlib import Path

    return loads(Path(path).read_text(encoding="utf-8"))
