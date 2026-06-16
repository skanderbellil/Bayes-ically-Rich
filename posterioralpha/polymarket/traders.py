"""Trader-level Polymarket data — leaderboard + per-wallet trade history.

The rest of this subpackage trades the *cross-section of market prices*. This
module opens a different door: the **flow of individual traders**. Polymarket
exposes two key-less endpoints that, together, let us ask "who has been winning,
and what are they buying?" — the raw material for a smart-money / copy-trading
strategy.

  * Leaderboard (``lb-api.polymarket.com/{profit,volume}``) — the top wallets by
    realised PnL or by traded volume over a rolling window (``1d`` / ``7d`` /
    ``30d`` / ``all``). PnL ranks the *winners*; volume ranks the *churners*
    (the market-maker screen leans on the latter).
  * Activity (``data-api.polymarket.com/activity?user=<wallet>&type=TRADE``) —
    every fill for one wallet: timestamp, ``conditionId`` (market), ``asset``
    (the specific outcome-token id), ``side`` (BUY/SELL), ``outcomeIndex``,
    ``price``, ``size`` (shares) and ``usdcSize`` (dollars). Newest-first,
    paginated by ``offset``.

Both are cached to ``data/raw/polymarket/traders/`` (checkpoint/resume, exactly
like ``fetch.py``) so a study can be rebuilt without re-hitting the API.

A note on survivorship — important enough to live in the data layer: a
leaderboard is computed *as of now*, so the candidate pool is unavoidably
enriched for traders who turned out to be good. We never use that ranking to
*time* anything (selection inside the backtest is strictly point-in-time on PnL
reconstructed from each wallet's own fills); the leaderboard only defines *which
wallets exist in the universe at all*. The residual enrichment of the pool is a
documented limitation, not a hidden lookahead.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from .fetch import _get, fetch_token_history  # reuse the retry/backoff GET + price cache
from .paths import RAW_DIR, ensure_dirs

logger = logging.getLogger(__name__)

LEADERBOARD_URL = "https://lb-api.polymarket.com"          # /profit , /volume
ACTIVITY_URL    = "https://data-api.polymarket.com/activity"

VALID_WINDOWS = ("1d", "7d", "30d", "all")

TRADERS_DIR = RAW_DIR / "traders"


def _traders_dir() -> Path:
    ensure_dirs()
    TRADERS_DIR.mkdir(parents=True, exist_ok=True)
    return TRADERS_DIR


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------

def fetch_leaderboard(
    window: str = "30d",
    kind: str = "profit",
    limit: int = 100,
) -> pd.DataFrame:
    """Top wallets by ``kind`` (``profit`` | ``volume``) over ``window``.

    Returns a DataFrame[wallet, name, amount, window, kind] sorted descending by
    amount (USD PnL or USD volume). Empty on failure.
    """
    if window not in VALID_WINDOWS:
        raise ValueError(f"window must be one of {VALID_WINDOWS}, got {window!r}")
    if kind not in ("profit", "volume"):
        raise ValueError("kind must be 'profit' or 'volume'")

    data = _get(f"{LEADERBOARD_URL}/{kind}", {"window": window, "limit": limit})
    rows = data if isinstance(data, list) else []
    out = []
    for r in rows:
        wallet = r.get("proxyWallet")
        if not wallet:
            continue
        out.append({
            "wallet": str(wallet).lower(),
            "name":   r.get("name") or r.get("pseudonym") or "",
            "amount": float(r.get("amount") or 0.0),
            "window": window,
            "kind":   kind,
        })
    df = pd.DataFrame(out)
    logger.info("fetch_leaderboard(%s, %s): %d wallets", kind, window, len(df))
    return df


def candidate_universe(
    profit_windows: Iterable[str] = ("7d", "30d", "all"),
    per_window: int = 80,
) -> pd.DataFrame:
    """Union of profit-leaderboard wallets across windows — the candidate pool.

    The pool is the set of wallets we *might* follow; who we actually follow at
    each date is decided point-in-time inside the backtest. A wallet is kept once
    (best PnL rank seen) with the windows it appeared in, so downstream code can
    see whether a name is a persistent winner (in ``all``) or a recent flyer
    (only in ``7d``).
    """
    frames = [fetch_leaderboard(w, "profit", per_window) for w in profit_windows]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=["wallet", "name", "amount", "windows"])
    allrows = pd.concat(frames, ignore_index=True)
    grp = allrows.groupby("wallet")
    pool = pd.DataFrame({
        "name":    grp["name"].first(),
        "amount":  grp["amount"].max(),                         # best window PnL
        "windows": grp["window"].apply(lambda s: ",".join(sorted(set(s)))),
    }).reset_index().sort_values("amount", ascending=False)
    logger.info("candidate_universe: %d unique wallets across %s",
                len(pool), list(profit_windows))
    return pool.reset_index(drop=True)


def fetch_volume_wallets(window: str = "all", limit: int = 120) -> set[str]:
    """Set of high-volume wallets (the market-maker watchlist).

    Heavy two-sided volume with little net PnL is the signature of a market
    maker; intersecting the candidate pool with this set is one input to the MM
    screen in ``smartmoney.py``.
    """
    df = fetch_leaderboard(window, "volume", limit)
    return set(df["wallet"]) if not df.empty else set()


# ---------------------------------------------------------------------------
# Per-wallet trade history
# ---------------------------------------------------------------------------

_TRADE_COLS = ["timestamp", "conditionId", "asset", "side", "outcomeIndex",
               "price", "size", "usdcSize", "title", "slug"]


def _trader_cache(wallet: str) -> Path:
    return _traders_dir() / f"trades_{wallet.lower()}.csv"


def fetch_trader_trades(
    wallet: str,
    max_trades: int = 4000,
    page_size: int = 500,
    sleep: float = 0.15,
    use_cache: bool = True,
    refresh: bool = False,
) -> pd.DataFrame:
    """All TRADE fills for one wallet, oldest→newest, cached to CSV.

    Columns: timestamp (datetime, UTC-naive), conditionId, asset (token id),
    side, outcomeIndex, price, size (shares), usdcSize (dollars), title, slug.
    Paginates by ``offset`` up to ``max_trades`` (the API returns newest-first).
    """
    wallet = wallet.lower()
    cache = _trader_cache(wallet)
    if use_cache and not refresh and cache.exists():
        # token ids / conditionIds are 70+ digit integers — force str so they
        # survive the CSV round-trip (pandas would otherwise read them as float
        # and silently lose precision, breaking every join on the token id).
        df = pd.read_csv(cache, parse_dates=["timestamp"],
                         dtype={"asset": str, "conditionId": str,
                                "slug": str, "title": str, "side": str})
        return df

    rows: list[dict] = []
    offset = 0
    while len(rows) < max_trades:
        batch = _get(ACTIVITY_URL, {
            "user": wallet, "type": "TRADE",
            "limit": page_size, "offset": offset,
        })
        if not isinstance(batch, list) or not batch:
            break
        for r in batch:
            rows.append({
                "timestamp":    int(r.get("timestamp") or 0),
                "conditionId":  r.get("conditionId") or "",
                "asset":        str(r.get("asset") or ""),
                "side":         r.get("side") or "",
                "outcomeIndex": int(r.get("outcomeIndex") or 0),
                "price":        float(r.get("price") or 0.0),
                "size":         float(r.get("size") or 0.0),
                "usdcSize":     float(r.get("usdcSize") or 0.0),
                "title":        r.get("title") or "",
                "slug":         r.get("slug") or "",
            })
        if len(batch) < page_size:
            break
        offset += page_size
        if sleep:
            time.sleep(sleep)

    if not rows:
        df = pd.DataFrame(columns=_TRADE_COLS)
    else:
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_localize(None)
        df = df.sort_values("timestamp").reset_index(drop=True)

    if use_cache:
        df.to_csv(cache, index=False)
    return df


def build_token_panel(
    token_ids: Iterable[str],
    min_obs: int = 15,
    sleep: float = 0.15,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Daily price panel (dates × token) for an arbitrary set of outcome tokens.

    Unlike ``fetch.build_price_panel`` — which pulls the highest-volume *markets*
    and keeps only their Yes leg — this prices *exactly the tokens the traders
    held* (either leg), since the follow-book marks each leader's position in the
    specific token they bought. Reuses the per-token CLOB cache from ``fetch.py``.
    """
    token_ids = [str(t) for t in dict.fromkeys(token_ids) if t]  # de-dupe, keep order
    series: dict[str, pd.Series] = {}
    for i, tok in enumerate(token_ids, 1):
        s = fetch_token_history(tok, use_cache=use_cache)
        if len(s) >= min_obs:
            series[tok] = s
        if i % 25 == 0:
            logger.info("  token price history %d/%d (%d usable)", i, len(token_ids), len(series))
        if sleep and not (use_cache and (RAW_DIR / f"token_{tok}.csv").exists()):
            time.sleep(sleep)
    if not series:
        raise RuntimeError("No tokens had >= min_obs price points.")
    panel = pd.DataFrame(series).sort_index()
    full_idx = pd.date_range(panel.index.min(), panel.index.max(), freq="D")
    panel = panel.reindex(full_idx)
    panel.index.name = "date"
    panel.columns = panel.columns.astype(str)
    logger.info("build_token_panel: %d days × %d tokens", panel.shape[0], panel.shape[1])
    return panel


def fetch_many_traders(
    wallets: Iterable[str],
    max_trades: int = 4000,
    sleep: float = 0.15,
    use_cache: bool = True,
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Trade history for many wallets → {wallet: DataFrame}. Skips empty wallets."""
    wallets = [w.lower() for w in wallets]
    out: dict[str, pd.DataFrame] = {}
    for i, w in enumerate(wallets, 1):
        df = fetch_trader_trades(w, max_trades=max_trades, sleep=sleep,
                                 use_cache=use_cache, refresh=refresh)
        if not df.empty:
            out[w] = df
        if i % 20 == 0:
            logger.info("  trader history %d/%d (%d non-empty)", i, len(wallets), len(out))
    logger.info("fetch_many_traders: %d/%d wallets with trades", len(out), len(wallets))
    return out
