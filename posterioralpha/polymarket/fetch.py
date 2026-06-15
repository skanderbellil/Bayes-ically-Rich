"""Live Polymarket data access — Gamma (metadata) + CLOB (price history).

Polymarket is a binary prediction market: each market resolves Yes/No, and the
two outcome tokens trade between $0 and $1. The Yes-token price *is* the market's
implied probability of the event — that is the "price series" a cross-market
momentum strategy trades.

Two public, key-less REST APIs are used:

  * Gamma  (``gamma-api.polymarket.com/markets``) — market metadata: question,
    outcomes, ``clobTokenIds`` (the on-chain Yes/No token ids), volume, liquidity,
    start/end dates, resolution flag.
  * CLOB   (``clob.polymarket.com/prices-history``) — historical mid-price for a
    single token id: a list of ``{t: unix_seconds, p: price}`` points.

The panel builder pulls the most-traded *resolved* markets, fetches each Yes
token's price history at daily resolution, and outer-joins them into a
(dates × markets) DataFrame of implied probabilities — the input every signal
and backtest in this subpackage consumes.

Caching is checkpoint/resume, exactly like ``pead.fetch``: each token's daily
series is persisted under ``data/raw/polymarket/`` so a panel can be rebuilt
without re-hitting the API.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from .paths import RAW_DIR, MARKETS_CSV, PANEL_CSV, ensure_dirs

logger = logging.getLogger(__name__)

GAMMA_URL = "https://gamma-api.polymarket.com/markets"
CLOB_URL  = "https://clob.polymarket.com/prices-history"

_HEADERS = {"User-Agent": "posterioralpha-polymarket/1.0"}


# ---------------------------------------------------------------------------
# HTTP with retries
# ---------------------------------------------------------------------------

def _get(url: str, params: dict, retries: int = 4, timeout: int = 30) -> Optional[object]:
    """GET JSON with exponential backoff; returns parsed JSON or None."""
    delay = 1.0
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=_HEADERS, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()
            # 429 / 5xx → back off and retry
            if resp.status_code in (429, 500, 502, 503, 504):
                logger.debug("GET %s -> %s, retry in %.0fs", url, resp.status_code, delay)
                time.sleep(delay)
                delay *= 2
                continue
            logger.warning("GET %s -> HTTP %s", url, resp.status_code)
            return None
        except requests.RequestException as e:  # noqa: BLE001
            logger.debug("GET %s failed (%s), retry in %.0fs", url, type(e).__name__, delay)
            time.sleep(delay)
            delay *= 2
    return None


# ---------------------------------------------------------------------------
# Gamma — market metadata
# ---------------------------------------------------------------------------

def _yes_token_id(clob_token_ids: str, outcomes: str) -> Optional[str]:
    """Pick the Yes-outcome token id from the (JSON-string) Gamma fields."""
    try:
        tokens = json.loads(clob_token_ids)
        outs   = [o.strip().lower() for o in json.loads(outcomes)]
    except (json.JSONDecodeError, TypeError):
        return None
    if not tokens or len(tokens) != len(outs):
        return None
    for tok, out in zip(tokens, outs):
        if out in ("yes", "true"):
            return str(tok)
    # Non Yes/No market (e.g. multi-candidate) — treat first outcome as the "Yes leg"
    return str(tokens[0])


def fetch_markets(
    n_markets: int = 120,
    min_volume: float = 50_000.0,
    min_liquidity: float = 1_000.0,
    closed: bool = True,
    page_size: int = 100,
) -> pd.DataFrame:
    """Page through Gamma for the highest-volume markets matching the filters.

    Parameters
    ----------
    n_markets     : number of qualifying markets to return
    min_volume    : minimum lifetime USD volume (liquidity/coverage screen)
    min_liquidity : minimum current book liquidity
    closed        : if True, only *resolved* markets (clean backtest); if False,
                    only currently-open markets (live signal)

    Returns a DataFrame with one row per market: id, question, slug, yes_token,
    volume, liquidity, start/end dates, closed flag.
    """
    rows: list[dict] = []
    offset = 0
    while len(rows) < n_markets:
        batch = _get(GAMMA_URL, {
            "limit": page_size,
            "offset": offset,
            "closed": str(closed).lower(),
            "order": "volumeNum",
            "ascending": "false",
        })
        if not batch:
            break
        for m in batch:
            try:
                vol = float(m.get("volumeNum") or 0.0)
                liq = float(m.get("liquidityNum") or 0.0)
            except (TypeError, ValueError):
                continue
            # Resolved markets carry ~0 current book liquidity, so only screen on
            # liquidity for *open* markets; volume is the coverage screen for both.
            if vol < min_volume:
                continue
            if not closed and liq < min_liquidity:
                continue
            yes_token = _yes_token_id(m.get("clobTokenIds", ""), m.get("outcomes", ""))
            if yes_token is None:
                continue
            rows.append({
                "id":         str(m.get("id")),
                "question":   m.get("question"),
                "slug":       m.get("slug"),
                "yes_token":  yes_token,
                "volume":     vol,
                "liquidity":  liq,
                "start_date": m.get("startDate"),
                "end_date":   m.get("endDate"),
                "closed":     bool(m.get("closed")),
            })
            if len(rows) >= n_markets:
                break
        if len(batch) < page_size:
            break
        offset += page_size

    df = pd.DataFrame(rows)
    logger.info("fetch_markets: %d markets (vol≥%.0f, liq≥%.0f, closed=%s)",
                len(df), min_volume, min_liquidity, closed)
    return df


# ---------------------------------------------------------------------------
# CLOB — per-token price history
# ---------------------------------------------------------------------------

def _token_cache_path(token_id: str) -> Path:
    return RAW_DIR / f"token_{token_id}.csv"


def fetch_token_history(
    token_id: str,
    fidelity_minutes: int = 1440,
    use_cache: bool = True,
) -> pd.Series:
    """Daily Yes-price series for one token (indexed by UTC date).

    ``fidelity_minutes=1440`` requests daily bars over the token's full life
    (``interval=max``). The raw points are collapsed to one price per calendar
    date (last observation). Cached to CSV for checkpoint/resume.
    """
    cache = _token_cache_path(token_id)
    if use_cache and cache.exists():
        s = pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0]
        s.name = token_id
        return s

    data = _get(CLOB_URL, {
        "market": token_id,
        "interval": "max",
        "fidelity": fidelity_minutes,
    })
    history = (data or {}).get("history", []) if isinstance(data, dict) else []
    if not history:
        s = pd.Series(dtype=float, name=token_id)
    else:
        df = pd.DataFrame(history)
        df["date"] = pd.to_datetime(df["t"], unit="s", utc=True).dt.tz_localize(None).dt.normalize()
        # last price per day → clean daily mark
        s = df.groupby("date")["p"].last().astype(float)
        s.name = token_id

    if use_cache:
        ensure_dirs()
        s.to_frame(name="p").to_csv(cache)
    return s


# ---------------------------------------------------------------------------
# Panel builder
# ---------------------------------------------------------------------------

def build_price_panel(
    n_markets: int = 120,
    min_volume: float = 50_000.0,
    min_liquidity: float = 1_000.0,
    closed: bool = True,
    min_obs: int = 20,
    sleep: float = 0.2,
    use_cache: bool = True,
    refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a daily Yes-price panel across many markets.

    Returns
    -------
    panel : DataFrame (dates × market_id) of Yes-token prices (implied
            probabilities). NaN where a market was not yet open / already
            resolved.
    meta  : DataFrame of per-market metadata (question, volume, dates, ...),
            indexed by market_id and aligned to the panel columns.

    The fully-assembled panel + metadata are cached to ``data/raw/polymarket/``;
    pass ``refresh=True`` to rebuild from the API.
    """
    ensure_dirs()
    if use_cache and not refresh and PANEL_CSV.exists() and MARKETS_CSV.exists():
        panel = pd.read_csv(PANEL_CSV, index_col=0, parse_dates=True)
        meta  = pd.read_csv(MARKETS_CSV, index_col=0)
        meta.index = meta.index.astype(str)
        logger.info("build_price_panel: loaded cache — %d days × %d markets",
                    panel.shape[0], panel.shape[1])
        return panel, meta

    meta = fetch_markets(n_markets, min_volume, min_liquidity, closed)
    if meta.empty:
        raise RuntimeError("No markets returned from Gamma — check network/filters.")

    series: dict[str, pd.Series] = {}
    for i, row in enumerate(meta.itertuples(index=False), 1):
        s = fetch_token_history(row.yes_token, use_cache=use_cache)
        if len(s) >= min_obs:
            series[row.id] = s
        if i % 25 == 0:
            logger.info("  price history %d/%d (%d usable)", i, len(meta), len(series))
        if sleep and not (use_cache and _token_cache_path(row.yes_token).exists()):
            time.sleep(sleep)

    if not series:
        raise RuntimeError("No markets had >= min_obs price points.")

    panel = pd.DataFrame(series).sort_index()
    # daily calendar reindex so every column shares one date axis
    full_idx = pd.date_range(panel.index.min(), panel.index.max(), freq="D")
    panel = panel.reindex(full_idx)
    panel.index.name = "date"

    meta = meta.set_index("id").loc[list(series.keys())]
    meta.index = meta.index.astype(str)
    panel.columns = panel.columns.astype(str)

    if use_cache:
        panel.to_csv(PANEL_CSV)
        meta.to_csv(MARKETS_CSV)
    logger.info("build_price_panel: %d days × %d markets",
                panel.shape[0], panel.shape[1])
    return panel, meta
