"""Forward paper-trade tracker for the smart-money consensus signal.

The live, out-of-sample analog of the `ORDER_FLOW` / `PAYUP_FOLLOW` studies, and
the honest test of their one open question — *does the gross edge survive real
spread?* Each day it asks: which **currently-open** outcome tokens have several
**non-market-maker leaderboard wallets** recently buying them (consensus
breadth)? Those become new paper longs — and crucially each position is entered
at the **live CLOB ask**, not the mid, so the bid-ask spread the backtest could
only proxy is captured for real. Positions are held to resolution and marked
daily.

It mirrors ``papertrade.py`` (macro buy-leader): one idempotent CSV ledger at
``data/paper_trade/smart_flow_positions.csv``, additive — new consensus tokens are
appended, resolved ones marked closed, nothing deleted — designed to run as a
daily GitHub Actions cron.

State columns
-------------
token          : the outcome-token id (stable key)
condition_id   : parent market condition id
question       : market question text
domain         : coarse topic (politics/macro/sports/…)
entry_date     : date first logged
n_smart_buyers : distinct non-MM leaderboard wallets that bought it in the window
entry_mid      : CLOB mid at entry
entry_ask      : CLOB best-ask at entry (what we actually "pay" — captures spread)
spread         : entry_ask − entry_mid (the realised half-cost)
current_price  : latest mid (refreshed each run)
status         : open | won | lost
exit_date      : resolution date (blank while open)
outcome        : 1.0 won / 0.0 lost (blank while open)
pnl            : (outcome / entry_ask − 1) · fraction  (blank while open)
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from .categorize import market_category
from .fetch import fetch_order_book, order_book_features
from .traders import fetch_leaderboard, fetch_trader_trades

logger = logging.getLogger(__name__)

STATE_FILE = Path(__file__).resolve().parents[2] / "data" / "paper_trade" / "smart_flow_positions.csv"

_COLS = ["token", "condition_id", "question", "domain", "entry_date", "n_smart_buyers",
         "entry_mid", "entry_ask", "spread", "current_price",
         "status", "exit_date", "outcome", "pnl"]


# ---------------------------------------------------------------------------
# Smart, non-market-maker pool
# ---------------------------------------------------------------------------

def smart_pool(per_window: int = 60) -> list[str]:
    """Recent profit-leaderboard wallets minus the volume leaders (MM proxy).

    A light, cron-friendly screen: a wallet that ranks high on *profit* but also
    high on *volume* is most likely a market maker, so we drop the intersection.
    """
    winners: list[str] = []
    for w in ("7d", "30d"):
        df = fetch_leaderboard(w, "profit", per_window)
        if not df.empty:
            winners += df["wallet"].tolist()
    winners = list(dict.fromkeys(winners))
    vol = fetch_leaderboard("30d", "volume", per_window)
    mm = set(vol["wallet"]) if not vol.empty else set()
    pool = [w for w in winners if w not in mm]
    logger.info("smart_pool: %d winners, %d after dropping volume-leaders (MM proxy)",
                len(winners), len(pool))
    return pool


# ---------------------------------------------------------------------------
# Scan recent consensus buys on currently-open markets
# ---------------------------------------------------------------------------

def scan_smart_flow_entries(
    window_days: int = 7,
    min_buyers: int = 3,
    per_window: int = 60,
    min_price: float = 0.05,
    max_price: float = 0.90,
) -> list[dict]:
    """Open outcome tokens with ≥``min_buyers`` distinct non-MM wallets buying recently.

    Pulls each pool wallet's most recent fills (one page, uncached — this is a
    *live* signal), keeps BUYs in the trailing ``window_days`` window, counts
    distinct buyers per token, and for tokens clearing ``min_buyers`` fetches the
    live order book to confirm the market is open and price it (mid/ask/spread).
    """
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=window_days)
    pool = smart_pool(per_window)

    buys: dict[str, dict] = {}
    for w in pool:
        df = fetch_trader_trades(w, max_trades=500, use_cache=False)
        if df.empty:
            continue
        recent = df[(df["timestamp"] >= cutoff) & (df["side"] == "BUY")]
        for r in recent.itertuples(index=False):
            tok = str(r.asset)
            if not tok:
                continue
            e = buys.setdefault(tok, {"token": tok, "condition_id": r.conditionId,
                                      "question": r.title, "slug": r.slug, "buyers": set()})
            e["buyers"].add(w)

    candidates = []
    for tok, e in buys.items():
        n = len(e["buyers"])
        if n < min_buyers:
            continue
        book = fetch_order_book(tok)
        feat = order_book_features(book)
        if not feat:                       # no live book → resolved / untradeable
            continue
        mid, ask = feat["mid"], feat["best_ask"]
        if not (min_price <= mid <= max_price):
            continue
        candidates.append({
            "token": tok,
            "condition_id": e["condition_id"],
            "question": e["question"],
            "domain": market_category(e["question"] or ""),
            "n_smart_buyers": n,
            "entry_mid": round(mid, 4),
            "entry_ask": round(ask, 4),
            "spread": round(ask - mid, 4),
        })
    candidates.sort(key=lambda c: c["n_smart_buyers"], reverse=True)
    logger.info("scan_smart_flow_entries: %d tokens with ≥%d smart buyers (open)",
                len(candidates), min_buyers)
    return candidates


# ---------------------------------------------------------------------------
# Ledger I/O
# ---------------------------------------------------------------------------

def load_ledger() -> pd.DataFrame:
    if STATE_FILE.exists():
        return pd.read_csv(STATE_FILE, dtype=str)
    return pd.DataFrame(columns=_COLS)


def save_ledger(df: pd.DataFrame) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    df[_COLS].to_csv(STATE_FILE, index=False)


def _price(token: str) -> tuple[float | None, float | None]:
    """Live (mid, best_ask) for a token from the CLOB book; (None, None) if no book."""
    feat = order_book_features(fetch_order_book(token))
    if not feat:
        return None, None
    return feat["mid"], feat["best_ask"]


# ---------------------------------------------------------------------------
# Daily update
# ---------------------------------------------------------------------------

def update_ledger(bet_fraction: float = 0.10, min_buyers: int = 3,
                  window_days: int = 7) -> pd.DataFrame:
    """One daily cycle: append new consensus longs, refresh prices, mark resolutions.

    Idempotent on the token id. New positions enter at the live ask (spread paid);
    open positions are marked to the current mid; a token resolving to ≈0/1 is
    closed with realised PnL ``(outcome / entry_ask − 1) · bet_fraction``.
    """
    ledger = load_ledger()
    today = date.today().isoformat()
    held = set(ledger["token"].tolist()) if not ledger.empty else set()

    # ── 1. new consensus entries ──────────────────────────────────────────
    new_rows = []
    for c in scan_smart_flow_entries(window_days=window_days, min_buyers=min_buyers):
        if c["token"] in held:
            continue
        logger.info("NEW long: %s  (%d smart buyers)  mid %.3f ask %.3f",
                    (c["question"] or "")[:40], c["n_smart_buyers"], c["entry_mid"], c["entry_ask"])
        new_rows.append({
            "token": c["token"], "condition_id": c["condition_id"],
            "question": c["question"], "domain": c["domain"], "entry_date": today,
            "n_smart_buyers": str(c["n_smart_buyers"]),
            "entry_mid": str(c["entry_mid"]), "entry_ask": str(c["entry_ask"]),
            "spread": str(c["spread"]), "current_price": str(c["entry_mid"]),
            "status": "open", "exit_date": "", "outcome": "", "pnl": "",
        })
    if new_rows:
        ledger = pd.concat([ledger, pd.DataFrame(new_rows)], ignore_index=True)

    # ── 2. refresh + resolve open positions ───────────────────────────────
    for i, row in ledger[ledger["status"] == "open"].iterrows():
        mid, _ = _price(row["token"])
        if mid is None:
            continue
        ledger.at[i, "current_price"] = str(round(mid, 4))
        outcome = 1.0 if mid >= 0.99 else (0.0 if mid <= 0.01 else None)
        if outcome is not None:
            entry_ask = float(row["entry_ask"])
            pnl = (outcome / entry_ask - 1.0) * bet_fraction
            ledger.at[i, "status"]   = "won" if outcome == 1.0 else "lost"
            ledger.at[i, "exit_date"] = today
            ledger.at[i, "outcome"]  = str(outcome)
            ledger.at[i, "pnl"]      = str(round(pnl, 4))
            ledger.at[i, "current_price"] = str(outcome)
            logger.info("RESOLVED %s  %s  pnl=%.4f",
                        (row["question"] or "")[:40], ledger.at[i, "status"], pnl)

    save_ledger(ledger)
    return ledger
