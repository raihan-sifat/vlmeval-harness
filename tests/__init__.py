"""Test suite for vlmeval.

Written against :mod:`unittest` rather than pytest so it runs with a bare
interpreter, on a machine with no network, and in CI. pytest collects
``unittest.TestCase`` classes natively, so ``pytest`` works too -- the only
thing lost is plain ``assert`` rewriting, which is not worth a hard dependency
on a package that will not install offline.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python -m unittest discover` to work from a source checkout without an
# editable install. Harmless when the package is installed properly.
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
