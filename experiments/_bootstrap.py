"""
Make the repository root importable when an experiment is launched directly
(e.g. ``python experiments/main.py``) without installing the package.

Import this module first in every experiment script::

    import _bootstrap  # noqa: F401  (adds repo root to sys.path)

If you ``pip install -e .`` the project instead, this shim is a harmless no-op.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
