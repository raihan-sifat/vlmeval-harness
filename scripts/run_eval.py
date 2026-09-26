#!/usr/bin/env python
"""Run one evaluation without installing the package.

This is a thin wrapper over ``vlmeval run`` so a fresh clone works with no setup
step::

    python scripts/run_eval.py --model echo:demo --task mcq --source synthetic_mcq

All behaviour lives in :mod:`vlmeval.cli`. Keeping a second implementation here
would only create a way for the two entry points to disagree, which is exactly
the kind of drift that makes a harness's numbers untrustworthy.
"""

from __future__ import annotations

import sys

from _bootstrap import bootstrap

bootstrap()

from vlmeval.cli import main  # noqa: E402 - import must follow the path setup

USAGE = "usage: python scripts/run_eval.py <vlmeval run arguments...>"


def run(argv: list[str]) -> int:
    """Delegate to ``vlmeval run``, inserting the subcommand if it is missing.

    ``run_eval.py --model echo:demo`` and ``run_eval.py run --model echo:demo``
    both work, and ``--help`` shows the ``run`` help rather than this wrapper's.
    """
    if not argv or argv[0].startswith("-"):
        argv = ["run", *argv]
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1:]))
