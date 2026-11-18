"""Make the repo root importable so ``app``, ``mdlab`` and
``derivatives_engine`` resolve when the suite runs from any working directory.

The three packages live at the repository root (no ``src/`` layout), so the
root must be on ``sys.path``. pytest already inserts the rootdir in the common
case; this keeps it working regardless of invocation or import mode.
"""

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
