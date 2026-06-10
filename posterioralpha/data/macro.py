"""
Macro liquidity data (stage 1: data).

Net liquidity is a popular proxy for the dollar liquidity available to risk
assets.  The common construction (Fed balance sheet less the two big
liquidity "sinks") is::

    net_liquidity = Fed total assets (WALCL)
                  − Treasury General Account (WTREGEN)
                  − Overnight Reverse Repo (RRPONTSYD)

All three series come from FRED via the public no-key CSV endpoint.  WALCL and
WTREGEN are reported in $ millions, RRPONTSYD in $ billions — this module
aligns units to **$ billions**, forward-fills the weekly series onto a daily
business-day index, and caches the result under ``datasets/`` for offline use.

    from posterioralpha.data import load_net_liquidity     # offline, cached
    nl = load_net_liquidity()
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Dict

import pandas as pd

logger = logging.getLogger(__name__)

DATASETS_DIR = Path(__file__).resolve().parents[2] / "datasets"
NET_LIQUIDITY_CSV = DATASETS_DIR / "net_liquidity.csv"

# FRED deprecated unauthenticated access in Nov 2025: the no-key CSV endpoint
# now returns 503/4xx, and the v2 API requires a free key.  We try the keyed
# v2 JSON API first (set FRED_API_KEY — free at https://fred.stlouisfed.org/
# docs/api/api_key.html), then fall back to the legacy CSV for older envs.
_FRED_API = "https://api.stlouisfed.org/fred/series/observations"
_FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; PosteriorAlpha/1.0)"}

# series id -> unit scale to convert into $ billions
NET_LIQUIDITY_SERIES: Dict[str, float] = {
    "WALCL": 1e-3,       # Fed total assets, $mn -> $bn
    "WTREGEN": 1e-3,     # Treasury General Account, $mn -> $bn
    "RRPONTSYD": 1.0,    # Overnight Reverse Repo, already $bn
}


def _fred_via_api(series_id: str, api_key: str, timeout: int) -> pd.Series:
    import requests

    params = {"series_id": series_id, "api_key": api_key, "file_type": "json"}
    resp = requests.get(_FRED_API, params=params, headers=_HEADERS, timeout=timeout)
    resp.raise_for_status()
    obs = resp.json()["observations"]
    s = pd.Series(
        {pd.Timestamp(o["date"]): pd.to_numeric(o["value"], errors="coerce")
         for o in obs},
    ).dropna()
    s.index.name = "date"; s.name = series_id
    return s


def _fred_via_csv(series_id: str, timeout: int) -> pd.Series:
    import io
    import requests

    resp = requests.get(_FRED_CSV.format(sid=series_id), headers=_HEADERS, timeout=timeout)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), parse_dates=[0], index_col=0)
    s = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()
    s.index.name = "date"; s.name = series_id
    return s


def fetch_fred(series_id: str, retries: int = 4, timeout: int = 30) -> pd.Series:
    """
    Download one FRED series.

    Uses the keyed v2 JSON API when ``FRED_API_KEY`` is set (the supported path
    since FRED's Nov-2025 auth change), otherwise the legacy no-key CSV
    endpoint (often 503 now).  Retries with exponential backoff.
    """
    from posterioralpha.env import load_env
    load_env()
    api_key = os.environ.get("FRED_API_KEY")
    last_exc = None
    for attempt in range(retries):
        try:
            s = (_fred_via_api(series_id, api_key, timeout) if api_key
                 else _fred_via_csv(series_id, timeout))
            logger.info("FRED %s: %d obs [%s … %s]%s", series_id, len(s),
                        s.index.min().date(), s.index.max().date(),
                        "" if api_key else "  (no-key CSV — set FRED_API_KEY if this fails)")
            return s
        except Exception as exc:
            last_exc = exc
            wait = min(5 * 2 ** attempt, 60)
            logger.warning("FRED %s attempt %d failed (%s); retry in %ds",
                           series_id, attempt + 1, type(exc).__name__, wait)
            time.sleep(wait)
    hint = "" if api_key else (
        "  FRED now requires a (free) API key — set FRED_API_KEY "
        "(https://fred.stlouisfed.org/docs/api/api_key.html)."
    )
    raise RuntimeError(f"FRED fetch failed for {series_id}: {last_exc}.{hint}")


# ─────────────────────────────────────────────────────────────────────────────
# Broad macro panel
# ─────────────────────────────────────────────────────────────────────────────

FRED_MACRO_CSV = DATASETS_DIR / "fred_macro.csv"

# Curated FRED series: id → (publication lag in business days, description).
# Market-derived daily series (yields, spreads, vol, FX) are observable at the
# close with no revision → lag 1 (usable the next day, matching the miner's
# 1-day execution lag).  Weekly survey/aggregate series get their actual
# release delay so the panel stays causal when forward-filled.
MACRO_SERIES: Dict[str, tuple] = {
    # rates & curve
    "DFF":          (1, "Fed funds effective rate"),
    "DGS2":         (1, "2y Treasury yield"),
    "DGS10":        (1, "10y Treasury yield"),
    "T10Y2Y":       (1, "10y−2y curve slope"),
    "T10Y3M":       (1, "10y−3m curve slope"),
    # credit — ⚠️ FRED's API license-caps the ICE BofA OAS series to roughly
    # the last 3 years; BAA10Y is the full-history daily credit spread
    "BAA10Y":       (1, "Moody's Baa − 10y Treasury spread"),
    "BAMLH0A0HYM2": (1, "ICE BofA US High Yield OAS (API: ~3y depth)"),
    "BAMLC0A0CM":   (1, "ICE BofA US Corporate (IG) OAS (API: ~3y depth)"),
    # risk & conditions
    "VIXCLS":       (1, "CBOE VIX close"),
    "NFCI":         (5, "Chicago Fed National Financial Conditions (weekly)"),
    "STLFSI4":      (5, "St. Louis Fed Financial Stress (weekly)"),
    # inflation expectations
    "T5YIE":        (1, "5y breakeven inflation"),
    "T10YIE":       (1, "10y breakeven inflation"),
    # dollar
    "DTWEXBGS":     (1, "Broad trade-weighted dollar index"),
    # real economy (high frequency)
    "ICSA":         (5, "Initial jobless claims (weekly)"),
}


def build_fred_macro(
    start: str = "2010-01-01",
    save: bool = True,
) -> pd.DataFrame:
    """
    Fetch the curated macro panel from FRED (keyed API; set ``FRED_API_KEY``).

    Each series is shifted by its publication lag, reindexed to a daily
    business-day calendar and forward-filled, so ``panel.loc[t]`` only
    contains information that was actually available at t — safe to feed
    straight into causal signal construction.  Caches to
    ``datasets/fred_macro.csv`` when ``save``.
    """
    raw = {sid: fetch_fred(sid) for sid in MACRO_SERIES}
    end = max(s.index.max() for s in raw.values())
    idx = pd.date_range(start, end, freq="B")

    # align to the business-day calendar first (collapses weekend prints of
    # daily series like DFF), then apply the publication lag positionally
    panel = pd.DataFrame({
        sid: raw[sid].reindex(idx, method="ffill").shift(lag)
        for sid, (lag, _desc) in MACRO_SERIES.items()
    })
    panel.index.name = "date"

    if save:
        DATASETS_DIR.mkdir(exist_ok=True)
        panel.to_csv(FRED_MACRO_CSV, index_label="date")
        logger.info("saved FRED macro panel (%d days × %d series) → %s",
                    len(panel), panel.shape[1], FRED_MACRO_CSV.name)
    return panel


def build_net_liquidity(
    start: str = "2010-01-01",
    save: bool = True,
) -> pd.DataFrame:
    """
    Fetch the components from FRED and assemble net liquidity (in $ billions).

    Returns a DataFrame with the aligned component columns plus
    ``net_liquidity``, indexed on a daily business-day calendar (weekly series
    forward-filled).  Caches to ``datasets/net_liquidity.csv`` when ``save``.
    """
    raw = {sid: fetch_fred(sid) * scale for sid, scale in NET_LIQUIDITY_SERIES.items()}
    end = max(s.index.max() for s in raw.values())
    idx = pd.date_range(start, end, freq="B")

    comp = pd.DataFrame({sid: s.reindex(idx, method="ffill") for sid, s in raw.items()})
    comp["net_liquidity"] = comp["WALCL"] - comp["WTREGEN"] - comp["RRPONTSYD"]
    comp = comp.dropna(subset=["net_liquidity"])
    comp.index.name = "date"

    if save:
        DATASETS_DIR.mkdir(exist_ok=True)
        comp.to_csv(NET_LIQUIDITY_CSV, index_label="date")
        logger.info("saved net liquidity (%d days) → %s", len(comp), NET_LIQUIDITY_CSV.name)
    return comp
