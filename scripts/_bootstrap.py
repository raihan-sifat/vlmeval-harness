"""Make ``src/`` importable without installing the package.

Importing this module is the first line of every script in this directory. A
fresh clone can therefore run ``python scripts/run_eval.py`` with no install
step at all, which matters for a project whose first command should be cheap to
try.

If the package *is* installed (``pip install -e .``), the installed copy wins:
``src/`` is appended rather than prepended, so a local checkout never shadows a
deliberately installed version.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
REPO_ROOT = SRC.parent


def bootstrap() -> Path:
    """Put the repository's ``src/`` on the import path and return the root."""
    if SRC.is_dir() and str(SRC) not in sys.path:
        sys.path.append(str(SRC))
    if str(REPO_ROOT) not in sys.path:
        sys.path.append(str(REPO_ROOT))
    return REPO_ROOT
