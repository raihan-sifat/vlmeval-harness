#!/usr/bin/env python
"""Generate the offline synthetic benchmark without installing the package.

A thin wrapper over ``vlmeval fixtures``::

    python scripts/make_fixtures.py --n-items 120

The generator itself lives in :mod:`vlmeval.data.synthetic`; this file only
exists so a fresh clone can produce fixtures with no install step.
"""

from __future__ import annotations

import sys

from _bootstrap import bootstrap

bootstrap()

from vlmeval.cli import main  # noqa: E402 - import must follow the path setup


def run(argv: list[str]) -> int:
    """Delegate to ``vlmeval fixtures``, defaulting to it when unqualified."""
    if not argv or argv[0].startswith("-"):
        argv = ["fixtures", *argv]
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1:]))
