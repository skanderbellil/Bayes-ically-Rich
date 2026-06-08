"""Build a survivorship-aware US common-stock universe.

Sources:
  - FinanceDatabase equities table -> live US primary-listed names + market_cap bucket.
  - Alpha Vantage LISTING_STATUS (delisted) -> dead names with delisting dates,
    used to quantify survivorship bias (yfinance cannot retrieve dead-ticker data).

Honest limitation: the *live* universe is survivorship-biased by construction
(delisted names lack retrievable price/earnings data from yfinance). The delisted
list is used for a bias sensitivity check, not to make the panel fully clean.
"""

from __future__ import annotations

import pandas as pd

# US primary exchanges whose tickers yfinance addresses with a bare symbol.
US_PRIMARY_EXCHANGES = ["NYQ", "NMS", "NGM", "NCM", "ASE", "PCX", "BTS"]

# Order matters: used for reporting from largest to smallest.
CAP_BUCKETS = ["Mega Cap", "Large Cap", "Mid Cap", "Small Cap", "Micro Cap", "Nano Cap"]


def load_live_universe(equities_csv: str) -> pd.DataFrame:
    """Load the live US primary-listed common-stock universe.

    Returns a frame indexed by ticker with columns: name, sector, industry,
    exchange, market_cap (bucket), country.
    """
    df = pd.read_csv(equities_csv, index_col=0, low_memory=False).T
    df.index.name = "ticker"

    us = df[df["country"] == "United States"]
    us = us[us["exchange"].isin(US_PRIMARY_EXCHANGES)]

    keep = ["name", "sector", "industry", "exchange", "market_cap", "country"]
    us = us[keep].copy()

    # Drop obvious non-common-stock instruments (warrants/units/rights) by ticker
    # convention: 5-char symbols ending in W (warrant), U (unit), R (right).
    is_derivative = us.index.to_series().str.match(r"^[A-Z]{4}[WUR]$")
    us = us[~is_derivative]

    return us


def stratified_sample(
    universe: pd.DataFrame, per_bucket: int, seed: int = 42
) -> pd.DataFrame:
    """Sample up to `per_bucket` names from each market_cap bucket.

    Sampling is the only way to keep the yfinance pull tractable under rate
    limits; it does not bias the cross-sectional IC test within a bucket.
    """
    parts = []
    for bucket in CAP_BUCKETS:
        sub = universe[universe["market_cap"] == bucket]
        if len(sub) == 0:
            continue
        n = min(per_bucket, len(sub))
        parts.append(sub.sample(n=n, random_state=seed))
    return pd.concat(parts)


def load_delisted(av_delisted_csv: str) -> pd.DataFrame:
    """Load the Alpha Vantage delisted-stock snapshot (for bias checks)."""
    d = pd.read_csv(av_delisted_csv)
    d = d[d["assetType"] == "Stock"]
    d["delistingDate"] = pd.to_datetime(d["delistingDate"], errors="coerce")
    d["ipoDate"] = pd.to_datetime(d["ipoDate"], errors="coerce")
    return d
