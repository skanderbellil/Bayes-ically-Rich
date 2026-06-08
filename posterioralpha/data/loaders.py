"""
Robust loaders for the bundled datasets (stage 1: data).

Resolves dataset paths relative to the repository root, so experiments work
regardless of the current working directory (previously the run scripts used
bare relative paths like ``pd.read_csv("portfolio_data.csv")`` and only worked
when launched from the repo root).

Bundled datasets (in ``<repo>/datasets/``)
------------------------------------------
  portfolio_data.csv          5 real ETF adjusted closes: SPY, TLT, GLD, EEM, VNQ
  sp500_top100_adj_close.csv  ~100 S&P 500 names, adjusted closes
"""
from pathlib import Path

import pandas as pd

# <repo>/posterioralpha/data/loaders.py  →  parents[2] == <repo>
DATASETS_DIR = Path(__file__).resolve().parents[2] / "datasets"

PORTFOLIO_CSV = DATASETS_DIR / "portfolio_data.csv"
SP500_CSV     = DATASETS_DIR / "sp500_top100_adj_close.csv"


def load_portfolio_prices() -> pd.DataFrame:
    """Adjusted-close prices for the 5 real ETFs (SPY, TLT, GLD, EEM, VNQ)."""
    return pd.read_csv(
        PORTFOLIO_CSV, parse_dates=["Date"], index_col="Date"
    ).sort_index()


def load_portfolio_returns() -> pd.DataFrame:
    """Daily arithmetic returns for the 5 real ETFs (NaNs dropped)."""
    return load_portfolio_prices().pct_change().dropna()


def load_sp500_prices() -> pd.DataFrame:
    """Adjusted-close prices for the ~100 S&P 500 names."""
    return pd.read_csv(
        SP500_CSV, parse_dates=["Date"], index_col="Date"
    ).sort_index()
