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

# Long single-name daily closes (Date,Close) — used by the kinematic
# state-space study where a clean, long history of ONE instrument matters more
# than cross-section. SPY (1993→), QQQ (1999→), GLD (2004→).
FRESH_CSV = {
    "SPY": DATASETS_DIR / "fresh_SPY.csv",
    "QQQ": DATASETS_DIR / "fresh_QQQ.csv",
    "GLD": DATASETS_DIR / "fresh_GLD.csv",
}


def load_fresh_prices(tickers=("SPY", "QQQ", "GLD")) -> pd.DataFrame:
    """
    Daily closes for the bundled long single-name series, aligned on a common
    DatetimeIndex (one column per ticker).

    Pass a single string for one name, or an iterable for several. Columns are
    outer-joined on date; rows where every requested name is NaN are dropped,
    so callers that need a fully overlapping window should ``.dropna()``.
    """
    if isinstance(tickers, str):
        tickers = (tickers,)
    cols = {}
    for t in tickers:
        if t not in FRESH_CSV:
            raise KeyError(f"{t!r} not bundled; available: {sorted(FRESH_CSV)}")
        s = pd.read_csv(FRESH_CSV[t], parse_dates=["Date"], index_col="Date")["Close"]
        cols[t] = s.sort_index()
    return pd.DataFrame(cols).dropna(how="all").sort_index()


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


# ── Factor datasets (built by experiments/build_factor_data.py) ──────────────

def load_ff_factors(freq: str = "monthly") -> pd.DataFrame:
    """
    Kenneth French US factors: Mkt-RF, SMB, HML, RMW, CMA, RF, Mom —
    decimals. ``freq`` ∈ {"monthly", "daily"}; daily is the one to regress
    daily strategy returns on (FF5+UMD attribution).
    """
    from posterioralpha.data.factors import (FF_FACTORS_DAILY_CSV,
                                             FF_FACTORS_MONTHLY_CSV)
    path = FF_FACTORS_DAILY_CSV if freq == "daily" else FF_FACTORS_MONTHLY_CSV
    _require(path, "python experiments/build_factor_data.py --source ff")
    return pd.read_csv(path, parse_dates=["Date"], index_col="Date").sort_index()


def load_ff_europe() -> pd.DataFrame:
    """French Europe 5 factors + momentum, monthly, decimals."""
    from posterioralpha.data.factors import FF_EUROPE_MONTHLY_CSV
    _require(FF_EUROPE_MONTHLY_CSV,
             "python experiments/build_factor_data.py --source ff")
    return pd.read_csv(FF_EUROPE_MONTHLY_CSV, parse_dates=["Date"],
                       index_col="Date").sort_index()


def load_ff_industries() -> pd.DataFrame:
    """French 12 value-weighted industry portfolios, monthly, decimals."""
    from posterioralpha.data.factors import FF_INDUSTRY12_CSV
    _require(FF_INDUSTRY12_CSV,
             "python experiments/build_factor_data.py --source ff")
    return pd.read_csv(FF_INDUSTRY12_CSV, parse_dates=["Date"],
                       index_col="Date").sort_index()


def load_jkp_factors(region: str = None) -> pd.DataFrame:
    """
    JKP theme-level monthly factor returns. Long format
    (location, name, date, ret); pass ``region`` (e.g. "usa") to get a
    wide date × theme frame for that region.
    """
    from posterioralpha.data.factors import JKP_FACTORS_CSV
    _require(JKP_FACTORS_CSV,
             "python experiments/build_factor_data.py --source jkp")
    panel = pd.read_csv(JKP_FACTORS_CSV, parse_dates=["date"])
    if region is None:
        return panel
    sub = panel[panel["location"] == region]
    return sub.pivot(index="date", columns="name", values="ret").sort_index()


def load_jkp_individual(region: str = None) -> pd.DataFrame:
    """
    JKP 153 INDIVIDUAL characteristic factors (premium-oriented). Long
    format (location, name, date, ret); pass ``region`` (e.g. "usa") for a
    wide date × factor frame. Matched-breadth panel for the seasonality
    OOS test (rivals OpenAP's 212 US predictors, unlike the 13 themes).
    """
    from posterioralpha.data.factors import JKP_INDIV_CSV
    _require(JKP_INDIV_CSV,
             "python experiments/build_factor_data.py --source jkp-indiv")
    panel = pd.read_csv(JKP_INDIV_CSV, parse_dates=["date"])
    if region is None:
        return panel
    sub = panel[panel["location"] == region]
    return sub.pivot(index="date", columns="name", values="ret").sort_index()


def load_openap_returns() -> pd.DataFrame:
    """Chen-Zimmermann monthly long-short predictor returns (wide, decimals)."""
    from posterioralpha.data.factors import OPENAP_LS_CSV
    _require(OPENAP_LS_CSV,
             "python experiments/build_factor_data.py --source openap")
    return pd.read_csv(OPENAP_LS_CSV, parse_dates=["date"],
                       index_col="date").sort_index()


def load_openap_doc() -> pd.DataFrame:
    """Chen-Zimmermann signal documentation (acronym, category, sample, t-stats)."""
    from posterioralpha.data.factors import OPENAP_DOC_CSV
    _require(OPENAP_DOC_CSV,
             "python experiments/build_factor_data.py --source openap")
    return pd.read_csv(OPENAP_DOC_CSV)
