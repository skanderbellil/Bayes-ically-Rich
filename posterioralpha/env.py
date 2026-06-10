"""
Minimal ``.env`` loader (no python-dotenv dependency).

Secrets like ``FRED_API_KEY`` live in a gitignored ``<repo>/.env`` file::

    FRED_API_KEY=...

``load_env()`` parses it once and fills ``os.environ`` without overriding
variables that are already set (real environment wins over the file).
It is called automatically by ``experiments/_bootstrap.py`` and by the
FRED fetcher, so both experiment scripts and library use pick it up.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"

_loaded = False


def load_env(path: Path | None = None) -> None:
    """Load KEY=VALUE lines from ``.env`` into ``os.environ`` (idempotent)."""
    global _loaded
    if _loaded and path is None:
        return
    p = path or ENV_FILE
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value
        logger.debug("loaded environment from %s", p)
    _loaded = True
