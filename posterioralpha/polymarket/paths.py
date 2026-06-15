"""Repo-root-anchored data/result paths for the Polymarket pipeline.

Mirrors ``posterioralpha.pead.paths``: resolves runtime directories relative to
the repository root so the Polymarket experiments run from any working
directory.

These directories hold *runtime* artifacts (the CLOB price-history cache and
generated results); they are gitignored and created on demand.
"""
from pathlib import Path

# <repo>/posterioralpha/polymarket/paths.py  →  parents[2] == <repo>
REPO_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR     = REPO_ROOT / "data" / "raw" / "polymarket"   # CLOB/Gamma fetch cache
RESULTS_DIR = REPO_ROOT / "results" / "polymarket"        # generated outputs

MARKETS_CSV = RAW_DIR / "markets.csv"     # Gamma market metadata snapshot
PANEL_CSV   = RAW_DIR / "yes_panel.csv"   # daily YES-price panel (dates × markets)


def ensure_dirs() -> None:
    """Create the runtime directories if they don't already exist."""
    for d in (RAW_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
