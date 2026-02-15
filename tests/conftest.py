"""Ensure src/ is on sys.path before the project root so that
``import dytopo`` resolves to ``src/dytopo/`` (the full 8-module
package) instead of the legacy ``dytopo/`` directory at the repo root.
"""

import sys
from pathlib import Path

_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)
