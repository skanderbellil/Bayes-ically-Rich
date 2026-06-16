"""Repo-root-anchored data/result paths for the PEAD pipeline.

Mirrors ``posterioralpha.data.loaders``: resolves runtime directories relative
to the repository root so the PEAD experiments run from any working directory
(the original scripts used bare CWD-relative paths like ``"results/pead"``).

These directories hold *runtime* artifacts (downloaded universe dumps, the
yfinance fetch cache, and generated results); they are gitignored and created
on demand.
"""
from pathlib import Path

# <repo>/posterioralpha/pead/paths.py  →  parents[2] == <repo>
REPO_ROOT = Path(__file__).resolve().parents[2]

FINANCE_DATA_DIR = REPO_ROOT / "financial_data"          # FinanceDatabase dumps
RAW_DIR          = REPO_ROOT / "data" / "raw" / "pead"   # yfinance fetch cache
RESULTS_DIR      = REPO_ROOT / "results" / "pead"         # generated outputs

EQUITIES_CSV = FINANCE_DATA_DIR / "equities.csv"
LEDGER_CSV   = RAW_DIR / "fetch_ledger.csv"


def ensure_dirs() -> None:
    """Create the runtime directories if they don't already exist."""
    for d in (FINANCE_DATA_DIR, RAW_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
