"""Token price table used for cost accounting.

**This is a snapshot, not a source of truth.** Rates were captured on
2026-09-26 and providers change them without notice. Two things follow:

* Every rate can be overridden per model from ``configs/models.yaml``
  (``cost_in`` / ``cost_out``), so a stale table can never silently misreport
  spend on the models you actually care about.
* Cost is reported separately from accuracy in every table. A benchmark that
  only ranks models by score hides the fact that the top two rows may differ by
  20x in price.

Rates are USD per million tokens, ``(prompt, completion)``. Local and echo
backends are free, which is what makes the offline path genuinely free to run.
"""

from __future__ import annotations

#: USD per 1M prompt tokens, USD per 1M completion tokens.
OPENAI_PRICES: dict[str, tuple[float, float]] = {
    "gpt-5.6": (5.00, 30.00),
    "gpt-5.5": (5.00, 30.00),
    "gpt-5.4": (2.50, 15.00),
    "gpt-5.2": (1.75, 14.00),
    "gpt-5.1": (1.25, 10.00),
    "gpt-5": (1.25, 10.00),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "o3": (2.00, 8.00),
    "o4-mini": (1.10, 4.40),
}

ANTHROPIC_PRICES: dict[str, tuple[float, float]] = {
    "claude-fable-5": (10.00, 50.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-3-5-haiku": (0.80, 4.00),
}

GOOGLE_PRICES: dict[str, tuple[float, float]] = {
    "gemini-3-pro": (2.00, 12.00),
    "gemini-3-flash": (0.50, 3.00),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
}

_TABLES: dict[str, dict[str, tuple[float, float]]] = {
    "openai": OPENAI_PRICES,
    "anthropic": ANTHROPIC_PRICES,
    "google": GOOGLE_PRICES,
}


def lookup(backend: str, model: str) -> tuple[float, float]:
    """Best-matching price for `model`, or ``(0.0, 0.0)`` if unknown.

    Falls back to progressively shorter prefixes so that dated model ids
    (``claude-haiku-4-5-20251001``) and suffixed variants resolve to the base
    rate without enumerating every revision.
    """
    table = _TABLES.get(backend, {})
    if not table:
        return (0.0, 0.0)
    if model in table:
        return table[model]
    candidates = [key for key in table if model.startswith(key)]
    if candidates:
        return table[max(candidates, key=len)]
    return (0.0, 0.0)
