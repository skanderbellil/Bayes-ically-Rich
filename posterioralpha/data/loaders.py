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
  etf_universe_prices.csv     large liquid US ETF universe (adjusted closes)
  etf_universe_info.csv       financedatabase info for that universe (+ median ADV)
"""
from pathlib import Path

import pandas as pd

# <repo>/posterioralpha/data/loaders.py  →  parents[2] == <repo>
DATASETS_DIR = Path(__file__).resolve().parents[2] / "datasets"

PORTFOLIO_CSV = DATASETS_DIR / "portfolio_data.csv"
SP500_CSV     = DATASETS_DIR / "sp500_top100_adj_close.csv"
ETF_UNIVERSE_CSV      = DATASETS_DIR / "etf_universe_prices.csv.gz"
ETF_UNIVERSE_INFO_CSV = DATASETS_DIR / "etf_universe_info.csv"
EQUITY_UNIVERSE_CSV      = DATASETS_DIR / "equity_universe_prices.csv.gz"
EQUITY_UNIVERSE_INFO_CSV = DATASETS_DIR / "equity_universe_info.csv"
NET_LIQUIDITY_CSV        = DATASETS_DIR / "net_liquidity.csv"
FRED_MACRO_CSV           = DATASETS_DIR / "fred_macro.csv"


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


def _require(path: Path, builder_hint: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} not found in datasets/.  Build it once with:\n"
            f"    {builder_hint}"
        )
    return path


def load_etf_universe_prices() -> pd.DataFrame:
    """
    Adjusted-close prices for the large liquid US ETF universe.

    Built from financedatabase (universe + info) + yfinance (history); see
    ``posterioralpha.data.universe.build_etf_universe`` or
    ``experiments/build_etf_universe.py`` to (re)generate.
    """
    path = _require(ETF_UNIVERSE_CSV, "python experiments/build_etf_universe.py")
    return pd.read_csv(path, parse_dates=["Date"], index_col="Date").sort_index()


def load_etf_universe_returns() -> pd.DataFrame:
    """Daily arithmetic returns for the large ETF universe (NaNs dropped)."""
    return load_etf_universe_prices().pct_change().dropna(how="all")


def load_etf_universe_info() -> pd.DataFrame:
    """financedatabase instrument info for the ETF universe (+ median ADV)."""
    path = _require(ETF_UNIVERSE_INFO_CSV, "python experiments/build_etf_universe.py")
    return pd.read_csv(path, index_col="symbol")


def load_equity_universe_prices() -> pd.DataFrame:
    """
    Adjusted-close prices for the large liquid US equity universe.

    Built from financedatabase (universe + info) + yfinance (history); see
    ``posterioralpha.data.universe.build_equity_universe`` or
    ``experiments/build_equity_universe.py`` to (re)generate.
    """
    path = _require(EQUITY_UNIVERSE_CSV, "python experiments/build_equity_universe.py")
    return pd.read_csv(path, parse_dates=["Date"], index_col="Date").sort_index()


def load_equity_universe_returns() -> pd.DataFrame:
    """Daily arithmetic returns for the large equity universe (NaNs dropped)."""
    return load_equity_universe_prices().pct_change().dropna(how="all")


def load_equity_universe_info() -> pd.DataFrame:
    """financedatabase instrument info for the equity universe (+ median ADV)."""
    path = _require(EQUITY_UNIVERSE_INFO_CSV, "python experiments/build_equity_universe.py")
    return pd.read_csv(path, index_col="symbol")


def load_net_liquidity() -> pd.DataFrame:
    """
    Fed net liquidity (WALCL − TGA − RRP) in $bn, daily, with components.

    Built from FRED; see ``posterioralpha.data.macro.build_net_liquidity`` or
    ``experiments/run_net_liquidity.py`` to (re)generate.
    """
    path = _require(NET_LIQUIDITY_CSV, "python experiments/run_net_liquidity.py --build")
    return pd.read_csv(path, parse_dates=["date"], index_col="date").sort_index()


def load_fred_macro() -> pd.DataFrame:
    """
    Curated FRED macro panel (rates, curve, credit OAS, VIX, NFCI/STLFSI,
    breakevens, broad dollar, jobless claims), daily business-day index,
    publication-lagged so row t is information available at t.

    Built from FRED (keyed); see ``posterioralpha.data.macro.build_fred_macro``
    or ``experiments/build_fred_macro.py`` to (re)generate.
    """
    path = _require(FRED_MACRO_CSV, "python experiments/build_fred_macro.py")
    return pd.read_csv(path, parse_dates=["date"], index_col="date").sort_index()
