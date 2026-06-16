"""Data fetching and preprocessing for PosteriorAlpha backtesting."""
import logging
from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Curated stable S&P 500 members as fallback (broad sector coverage)
STABLE_SP500 = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "INTC", "ADBE", "CRM", "TXN",
    "CSCO", "QCOM", "AMAT", "KLAC", "LRCX", "SNPS", "CDNS", "ANSS", "NOW", "INTU",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "AXP", "USB", "TFC", "PNC",
    "SPGI", "MCO", "ICE", "CME", "CB", "AIG", "MET", "PRU", "AFL", "ALL",
    # Healthcare
    "JNJ", "UNH", "PFE", "MRK", "ABBV", "TMO", "ABT", "DHR", "BMY", "AMGN",
    "GILD", "ISRG", "VRTX", "REGN", "IDXX", "IQV", "BDX", "ZBH", "SYK", "MDT",
    # Consumer
    "PG", "KO", "PEP", "WMT", "COST", "HD", "MCD", "NKE", "SBUX", "TGT",
    "LOW", "TJX", "ROST", "AZO", "ORLY", "DIS", "CMCSA", "VZ", "T", "NFLX",
    # Energy / Industrials / Materials
    "XOM", "CVX", "COP", "SLB", "EOG", "CAT", "DE", "HON", "MMM", "GE",
    "BA", "LMT", "RTX", "NOC", "GD", "UPS", "FDX", "EMR", "ETN", "PH",
    # REITs / Utilities
    "AMT", "PLD", "CCI", "EQIX", "PSA", "SPG", "NEE", "DUK", "SO", "D",
    # BRK
    "BRK-B", "V", "MA", "LIN", "APD",
]


_WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
# Wikipedia 403s the default urllib user-agent; send a browser-like one.
_HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; PosteriorAlpha/1.0)"}


def get_sp500_tickers() -> List[str]:
    """Fetch current S&P 500 tickers from Wikipedia. Falls back to curated list."""
    try:
        import requests
        from io import StringIO

        resp = requests.get(_WIKI_SP500_URL, headers=_HTTP_HEADERS, timeout=15)
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text))
        tickers = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
        logger.info(f"Fetched {len(tickers)} tickers from Wikipedia")
        return tickers
    except Exception as exc:
        logger.warning(f"Wikipedia fetch failed ({exc}); using fallback list of {len(STABLE_SP500)}")
        return STABLE_SP500


def sample_tickers(universe: List[str], n: int = 50, seed: Optional[int] = None) -> List[str]:
    """Return a random sample of n tickers from universe (reproducible via seed)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(universe), size=min(n, len(universe)), replace=False)
    return [universe[i] for i in sorted(idx)]


def download_returns(
    tickers: List[str],
    start: str,
    end: str,
    min_coverage: float = 0.85,
) -> pd.DataFrame:
    """
    Download adjusted daily prices for `tickers` and return simple (arithmetic) returns.

    Tickers with fewer than `min_coverage` non-NaN observations are dropped.
    Forward-fills small gaps (e.g. staggered exchange holidays) before computing returns.
    """
    logger.info(f"Downloading {len(tickers)} tickers  [{start} → {end}]")

    raw = yf.download(
        tickers, start=start, end=end,
        auto_adjust=True, progress=False, threads=True,
    )

    # yfinance returns MultiIndex columns when n_tickers > 1
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        col = "Close"
        prices = raw[[col]].rename(columns={col: tickers[0]})

    # Drop tickers with insufficient history
    coverage = prices.notna().mean()
    keep = coverage[coverage >= min_coverage].index.tolist()
    n_dropped = len(prices.columns) - len(keep)
    if n_dropped:
        logger.info(f"  Dropped {n_dropped} tickers (coverage < {min_coverage:.0%})")

    prices = prices[keep].ffill().dropna(how="all")
    returns = prices.pct_change().dropna()

    logger.info(f"  Dataset: {len(returns)} days × {len(keep)} assets")
    return returns
