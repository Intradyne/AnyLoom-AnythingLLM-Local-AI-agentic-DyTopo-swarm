"""Ensure src/ is on sys.path so that ``import dytopo`` resolves to
``src/dytopo/`` (the full package).
"""

import sys
from pathlib import Path

_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)
