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

def _yes_index(outcomes: str) -> int:
    """Index of the Yes/True leg in the (JSON-string) outcomes list (0 if absent)."""
    try:
        outs = [o.strip().lower() for o in json.loads(outcomes)]
    except (json.JSONDecodeError, TypeError):
        return 0
    for i, out in enumerate(outs):
        if out in ("yes", "true"):
            return i
    return 0


def _yes_token_id(clob_token_ids: str, outcomes: str) -> Optional[str]:
    """Pick the Yes-outcome token id from the (JSON-string) Gamma fields."""
    try:
        tokens = json.loads(clob_token_ids)
    except (json.JSONDecodeError, TypeError):
        return None
    if not tokens:
        return None
    idx = _yes_index(outcomes)
    return str(tokens[idx]) if idx < len(tokens) else str(tokens[0])


def _event_fields(m: dict) -> tuple[str, str]:
    """Parent-event ticker + title for a market (for categorisation); ('','') if absent."""
    ev = m.get("events")
    if isinstance(ev, str):
        try:
            ev = json.loads(ev)
        except (json.JSONDecodeError, TypeError):
            ev = None
    if isinstance(ev, list) and ev and isinstance(ev[0], dict):
        return str(ev[0].get("ticker") or ""), str(ev[0].get("title") or "")
    return "", ""


def _yes_outcome(outcome_prices: str, outcomes: str) -> float:
    """Settled Yes probability (0/1) for a resolved market; NaN if not settled.

    After resolution Gamma reports ``outcomePrices`` as the settled values, e.g.
    ``["1", "0"]`` (Yes happened) or ``["0", "1"]`` (No happened).
    """
    try:
        prices = [float(x) for x in json.loads(outcome_prices)]
    except (json.JSONDecodeError, TypeError, ValueError):
        return float("nan")
    idx = _yes_index(outcomes)
    if idx >= len(prices):
        return float("nan")
    val = prices[idx]
    # settled markets sit at ~0/1; anything in between isn't a clean label
    if val <= 0.02:
        return 0.0
    if val >= 0.98:
        return 1.0
    return float("nan")


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
            ev_ticker, ev_title = _event_fields(m)
            rows.append({
                "id":           str(m.get("id")),
                "condition_id": str(m.get("conditionId") or ""),
                "question":     m.get("question"),
                "slug":         m.get("slug"),
                "yes_token":    yes_token,
                "volume":       vol,
                "liquidity":    liq,
                "start_date":   m.get("startDate"),
                "end_date":     m.get("endDate"),
                "closed_time":  m.get("closedTime"),   # actual close (may be after endDate)
                "closed":       bool(m.get("closed")),
                "outcome":      _yes_outcome(m.get("outcomePrices", ""), m.get("outcomes", "")),
                "event_ticker": ev_ticker,
                "event_title":  ev_title,
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
# Trade history (data-api) — all fills for a market
# ---------------------------------------------------------------------------

def fetch_market_trades(
    condition_id: str,
    max_trades: int = 2000,
    sleep: float = 0.3,
) -> pd.DataFrame:
    """All fills for a given market (condition_id), oldest → newest.

    Paginates data-api.polymarket.com/trades?market=<condition_id>.
    Returns DataFrame[proxyWallet, side, asset, outcome, size, price, timestamp].
    Empty DataFrame if the market has no trades.
    """
    url  = "https://data-api.polymarket.com/trades"
    rows: list[dict] = []
    offset = 0
    limit  = 500
    while len(rows) < max_trades:
        try:
            r = _get(url, {"market": condition_id, "limit": limit, "offset": offset})
        except Exception as e:
            logger.warning("fetch_market_trades %s offset=%d: %s", condition_id[:12], offset, e)
            break
        batch = r if isinstance(r, list) else []
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < limit:
            break
        offset += limit
        time.sleep(sleep)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)[["proxyWallet", "side", "asset", "outcome", "size", "price", "timestamp"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["size"]      = df["size"].astype(float)
    df["price"]     = df["price"].astype(float)
    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info("fetch_market_trades %s: %d fills", condition_id[:16], len(df))
    return df


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


# ---------------------------------------------------------------------------
# Resolution outcomes (the "actual outcome" label)
# ---------------------------------------------------------------------------

def resolution_outcomes(meta: pd.DataFrame, panel: Optional[pd.DataFrame] = None) -> pd.Series:
    """Binary Yes-resolution label per market (1.0 = resolved Yes, 0.0 = No).

    Prefers Gamma's settled ``outcome`` column; falls back to the last clean
    panel price (≈0/1 at resolution) for rows where the column is absent — so a
    legacy cache without the column still yields labels. Markets that did not
    settle cleanly are dropped (NaN).
    """
    out = pd.Series(np.nan, index=meta.index, dtype=float)
    if "outcome" in meta.columns:
        out = pd.to_numeric(meta["outcome"], errors="coerce")

    if panel is not None:
        need = out.index[out.isna()]
        for mid in need:
            if mid in panel.columns:
                last = panel[mid].dropna()
                if len(last):
                    v = float(last.iloc[-1])
                    out.loc[mid] = 1.0 if v >= 0.9 else (0.0 if v <= 0.1 else np.nan)
    return out.dropna()


# ---------------------------------------------------------------------------
# CLOB order book (live snapshot — full depth, no history)
# ---------------------------------------------------------------------------

def fetch_order_book(token_id: str) -> dict:
    """Current order book for a token: ``{bids: [...], asks: [...]}`` with sizes.

    Live snapshot only — Polymarket exposes full current depth but no historical
    books, so this powers *live* microstructure signals, not the backtest.
    """
    data = _get("https://clob.polymarket.com/book", {"token_id": token_id})
    return data if isinstance(data, dict) else {}


def order_book_features(book: dict) -> dict:
    """Microstructure features from a CLOB book snapshot.

    Returns best bid/ask, spread, mid, size-weighted microprice, and top-of-book
    depth imbalance ∈ [-1, 1] (positive = more bid size → buy pressure).
    """
    def _levels(side):
        out = []
        for lv in book.get(side, []) or []:
            try:
                out.append((float(lv["price"]), float(lv["size"])))
            except (KeyError, TypeError, ValueError):
                continue
        return out

    bids, asks = _levels("bids"), _levels("asks")
    if not bids or not asks:
        return {}
    best_bid = max(bids, key=lambda x: x[0])
    best_ask = min(asks, key=lambda x: x[0])
    bp, bsz = best_bid
    ap, asz = best_ask
    mid = 0.5 * (bp + ap)
    micro = (bp * asz + ap * bsz) / (asz + bsz) if (asz + bsz) > 0 else mid
    imb = (bsz - asz) / (bsz + asz) if (bsz + asz) > 0 else 0.0
    return {
        "best_bid": bp, "best_ask": ap, "spread": ap - bp,
        "mid": mid, "microprice": micro, "depth_imbalance": imb,
    }
