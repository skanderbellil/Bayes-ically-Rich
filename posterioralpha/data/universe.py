"""
Large-universe ETF data access (stage 1: data).

The framework convention is to research on a *large, liquid* universe rather
than a hand-picked subset.  This module assembles one:

  - ``financedatabase`` supplies the investable universe **and instrument info**
    (name, family/issuer, category, exchange, ISIN, …) — tens of thousands of
    ETFs, filtered down to US-listed funds from the major issuers.
  - ``yfinance`` supplies adjusted price **history** for that universe.
  - The candidates are screened for data coverage and dollar-volume liquidity,
    and the result is cached under ``datasets/`` so studies run offline and
    reproducibly (same philosophy as the other bundled datasets).

Typical use::

    from posterioralpha.data import load_etf_universe_returns
    returns = load_etf_universe_returns()        # offline, from datasets/

To (re)build / refresh the cached universe from the network::

    from posterioralpha.data.universe import build_etf_universe
    prices, info = build_etf_universe(start="2010-01-01", top_n=250)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# <repo>/posterioralpha/data/universe.py  →  parents[2] == <repo>
DATASETS_DIR = Path(__file__).resolve().parents[2] / "datasets"
ETF_PRICES_CSV = DATASETS_DIR / "etf_universe_prices.csv.gz"
ETF_INFO_CSV = DATASETS_DIR / "etf_universe_info.csv"
EQUITY_PRICES_CSV = DATASETS_DIR / "equity_universe_prices.csv.gz"
EQUITY_INFO_CSV = DATASETS_DIR / "equity_universe_info.csv"

# Major US ETF listing venues (financedatabase ``exchange`` codes).
US_ETF_EXCHANGES: Tuple[str, ...] = ("PCX", "NMS", "NGM", "BTS", "ASE")

# Primary US *equity* listing venues — NYSE (NYQ), NASDAQ (NMS/NGM), NYSE
# American (ASE).  fd lists many foreign cross-listings (FRA, STU, …) for the
# same companies; restricting to these keeps the universe US-traded.
US_EQUITY_EXCHANGES: Tuple[str, ...] = ("NYQ", "NMS", "NGM", "ASE")

# Market-cap buckets to include for the equity universe by default.
LARGE_CAPS: Tuple[str, ...] = ("Mega Cap", "Large Cap")

# US-domiciled issuers whose ETFs are the liquid, tradeable ones.  (European
# issuers such as Xtrackers/Lyxor/Amundi/UBS are deliberately excluded — their
# listings are mostly LSE/EBS and out of scope for a US-traded universe.)
MAJOR_US_ETF_FAMILIES: Tuple[str, ...] = (
    "BlackRock Asset Management",            # iShares
    "State Street Global Advisors",          # SPDR
    "Vanguard Asset Management",
    "Invesco Investment Management",
    "Charles Schwab Investment Management",
    "First Trust Advisors",
    "WisdomTree Asset Management",
    "VanEck Asset Management",
    "Global X Funds",
    "JPMorgan Asset Management",
    "ProShares",
    "Direxion Funds",
)

# Leveraged / inverse products — excluded from a research universe by default.
_LEVERAGED_RE = r"(?i)\b(2x|3x|ultra|ultrapro|inverse|leveraged|-1x|short|bear)\b"


def etf_info(
    currency: str = "USD",
    exchanges: Sequence[str] = US_ETF_EXCHANGES,
    families: Optional[Sequence[str]] = MAJOR_US_ETF_FAMILIES,
    category_group: Optional[str] = None,
    exclude_leveraged: bool = True,
) -> pd.DataFrame:
    """
    Investable ETF universe **with instrument info**, from financedatabase.

    Returns a DataFrame indexed by ticker symbol with the fd metadata columns
    (name, family, category, category_group, exchange, mic, isin, summary),
    filtered to clean US-listed tickers from the requested issuers.
    """
    import financedatabase as fd

    df = fd.ETFs().select(currency=currency)
    df = df[df["exchange"].isin(list(exchanges))]
    if families:
        df = df[df["family"].isin(list(families))]
    if category_group:
        df = df[df["category_group"] == category_group]
    if exclude_leveraged:
        df = df[~df["name"].fillna("").str.contains(_LEVERAGED_RE, regex=True)]

    # Keep clean US tickers only (drop NaN and foreign/suffixed symbols).
    sym = df.index.to_series().astype(str)
    df = df[df.index.notna() & ~sym.str.contains(r"[.\-]", regex=True)]
    df = df[~df.index.duplicated(keep="first")]
    logger.info("financedatabase universe: %d ETFs after filters", len(df))
    return df.sort_index()


def equity_info(
    country: str = "United States",
    market_caps: Optional[Sequence[str]] = LARGE_CAPS,
    exchanges: Sequence[str] = US_EQUITY_EXCHANGES,
    sectors: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Investable equity universe **with instrument info**, from financedatabase.

    Returns a DataFrame indexed by ticker symbol with fd metadata (name,
    sector, industry, market_cap, country, exchange, …), filtered to clean
    US-listed tickers of the requested market-cap buckets and sectors.

    ⚠️ Survivorship bias: this is *current* membership — the large/mega-cap
    names as of today.  Backtesting their history therefore excludes companies
    that were delisted, shrank, or went bankrupt, biasing long-only and
    equal-weight results upward.  Treat equity-universe studies as exploratory.
    """
    import financedatabase as fd

    df = fd.Equities().select(country=country)
    df = df[df["exchange"].isin(list(exchanges))]
    if market_caps:
        df = df[df["market_cap"].isin(list(market_caps))]
    if sectors:
        df = df[df["sector"].isin(list(sectors))]

    # Keep clean tickers (drop NaN and foreign/dotted symbols; '-' class shares
    # such as BRK-B are kept — that's yfinance's own convention).
    sym = df.index.to_series().astype(str)
    df = df[df.index.notna() & ~sym.str.contains(r"[.]", regex=True)]
    df = df[~df.index.duplicated(keep="first")]
    logger.info("financedatabase equity universe: %d names after filters", len(df))
    return df.sort_index()


def fetch_prices_volume(
    symbols: Sequence[str],
    start: str,
    end: Optional[str] = None,
    batch: int = 200,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download adjusted closes and volumes for ``symbols`` via yfinance.

    Returns ``(close, volume)`` DataFrames (columns = symbols, index = dates).
    Symbols with no data are simply absent from the result.
    """
    import yfinance as yf

    symbols = list(symbols)
    closes, vols = [], []
    for i in range(0, len(symbols), batch):
        chunk = symbols[i : i + batch]
        logger.info(
            "yfinance %d–%d / %d", i + 1, min(i + batch, len(symbols)), len(symbols)
        )
        raw = yf.download(
            chunk, start=start, end=end,
            auto_adjust=True, progress=False, threads=True,
        )
        if raw is None or len(raw) == 0:
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            closes.append(raw["Close"])
            vols.append(raw["Volume"])
        else:  # single surviving symbol → flat columns
            closes.append(raw[["Close"]].rename(columns={"Close": chunk[0]}))
            vols.append(raw[["Volume"]].rename(columns={"Volume": chunk[0]}))

    if not closes:
        return pd.DataFrame(), pd.DataFrame()
    close = pd.concat(closes, axis=1).sort_index()
    vol = pd.concat(vols, axis=1).sort_index()
    # de-duplicate any symbol that appeared in two batches
    close = close.loc[:, ~close.columns.duplicated()]
    vol = vol.loc[:, ~vol.columns.duplicated()]
    return close, vol


def screen_liquidity(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    min_coverage: float = 0.9,
    top_n: int = 250,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Keep the ``top_n`` most liquid names with sufficient history.

    Drops symbols whose non-NaN coverage is below ``min_coverage``, ranks the
    survivors by **median daily dollar volume** (price × volume), and keeps the
    top ``top_n``.  Returns ``(prices, median_adv_usd)`` aligned on those names.
    """
    cov = close.notna().mean()
    keep = cov[cov >= min_coverage].index
    close = close[keep].ffill()
    volume = volume[keep]

    adv = (close * volume).median(skipna=True).replace(0, np.nan).dropna()
    top = adv.sort_values(ascending=False).head(top_n).index.tolist()
    prices = close[top].dropna(how="all")
    logger.info(
        "liquidity screen: %d/%d names pass coverage, keeping top %d by ADV",
        len(keep), close.shape[1], len(top),
    )
    return prices, adv.loc[top]


def build_etf_universe(
    start: str = "2010-01-01",
    end: Optional[str] = None,
    top_n: int = 250,
    min_coverage: float = 0.9,
    save: bool = True,
    **info_kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Assemble a large, liquid ETF universe (prices + info) and cache it.

    1. Discover the universe + info from financedatabase (``etf_info``).
    2. Download adjusted price history + volume from yfinance.
    3. Screen for coverage and dollar-volume liquidity, keeping ``top_n`` names.
    4. Persist to ``datasets/etf_universe_{prices,info}.csv`` for offline reuse.

    Returns ``(prices, info)``: ``prices`` is a (days × top_n) adjusted-close
    panel; ``info`` is the fd metadata for those names plus ``median_adv_usd``.
    """
    info = etf_info(**info_kwargs)
    return _assemble_universe(
        info, start, end, top_n, min_coverage, save,
        prices_path=ETF_PRICES_CSV, info_path=ETF_INFO_CSV, label="ETF",
    )


def build_equity_universe(
    start: str = "2010-01-01",
    end: Optional[str] = None,
    top_n: int = 500,
    min_coverage: float = 0.9,
    save: bool = True,
    **info_kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Assemble a large, liquid US equity universe (prices + info) and cache it.

    Same pipeline as :func:`build_etf_universe` but over financedatabase
    equities (``equity_info``): discover US large/mega-cap names + info,
    download history + volume, screen by coverage and dollar-volume liquidity,
    and persist to ``datasets/equity_universe_{prices,info}``.

    Returns ``(prices, info)``; ``info`` carries fd sector/industry/market_cap
    plus ``median_adv_usd``.
    """
    info = equity_info(**info_kwargs)
    return _assemble_universe(
        info, start, end, top_n, min_coverage, save,
        prices_path=EQUITY_PRICES_CSV, info_path=EQUITY_INFO_CSV, label="equity",
    )


def _assemble_universe(
    info: pd.DataFrame,
    start: str,
    end: Optional[str],
    top_n: int,
    min_coverage: float,
    save: bool,
    prices_path: Path,
    info_path: Path,
    label: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Shared discover→fetch→screen→cache backend for the universe builders."""
    symbols = info.index.astype(str).tolist()

    close, volume = fetch_prices_volume(symbols, start=start, end=end)
    if close.empty:
        raise RuntimeError(f"yfinance returned no data for the {label} universe")

    prices, adv = screen_liquidity(
        close, volume, min_coverage=min_coverage, top_n=top_n
    )

    info = info.loc[info.index.isin(prices.columns)].copy()
    info["median_adv_usd"] = adv.reindex(info.index)
    info = info.sort_values("median_adv_usd", ascending=False)

    if save:
        DATASETS_DIR.mkdir(exist_ok=True)
        prices.to_csv(prices_path, index_label="Date", compression="gzip")
        info.to_csv(info_path, index_label="symbol")
        logger.info(
            "saved %d×%d %s prices → %s  and info → %s",
            prices.shape[0], prices.shape[1], label, prices_path.name, info_path.name,
        )
    return prices, info
