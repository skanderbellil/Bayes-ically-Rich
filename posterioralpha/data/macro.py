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
