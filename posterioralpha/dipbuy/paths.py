"""Repo-root-anchored paths for the dip-buy study (runtime data is gitignored)."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw" / "dipbuy"          # downloaded/parsed vendor files (gitignored)
CTS_DAY_DIR = RAW_DIR / "cts_days"                        # one parquet per trading day
XDP_DIR = RAW_DIR / "xdp"                                 # NYSE XDP sample files
YAHOO_DIR = RAW_DIR / "yahoo"                             # Yahoo chart-API cache
PANEL_PATH = RAW_DIR / "cts_panel.parquet"                # stacked daily panel
RESULTS_DIR = REPO_ROOT / "results" / "dipbuy"           # scratch outputs (gitignored)
DOCS_DIR = REPO_ROOT / "docs" / "dipbuy"                  # committed report, tables, charts


def ensure_dirs() -> None:
    for d in (RAW_DIR, CTS_DAY_DIR, XDP_DIR, YAHOO_DIR, RESULTS_DIR, DOCS_DIR):
        d.mkdir(parents=True, exist_ok=True)
