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
domain         : coarse topic (politics/macro/sports/...)
entry_date     : date first logged
n_smart_buyers : distinct non-MM leaderboard wallets that bought it in the window
bet_fraction   : position size fraction actually used (after Kelly scaling)
entry_mid      : CLOB mid at entry
entry_ask      : CLOB best-ask at entry (what we actually "pay" -- captures spread)
spread         : entry_ask - entry_mid (the realised half-cost)
current_price  : latest mid (refreshed each run)
status         : open | won | lost | stopped | flipped
exit_date      : resolution date (blank while open)
outcome        : 1.0 won / 0.0 lost (blank while open)
pnl            : (outcome / entry_ask - 1) * fraction  (blank while open)
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


def kelly_fraction(
    n_smart_buyers: int,
    base_fraction: float = 0.10,
    ref_buyers: int = 4,
    cap_multiple: float = 2.0,
) -> float:
    """Scale bet fraction proportionally to smart-buyer conviction.

    3 buyers -> 7.5%, 4 -> 10%, 5 -> 12.5%, 6 -> 15%, capped at base * cap_multiple.
    """
    raw = base_fraction * (n_smart_buyers / ref_buyers)
    return round(min(raw, base_fraction * cap_multiple), 6)

_COLS = ["token", "condition_id", "question", "domain", "entry_date",
         "n_smart_buyers", "bet_fraction",
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

def _build_flow_index(pool: list[str], window_days: int) -> dict[str, dict]:
    """Fetch recent trades for all pool wallets; index by token with buyer/seller sets.

    Returns {token: {"token", "condition_id", "question", "slug", "buyers": set, "sellers": set}}.
    Fetching once and sharing between entry scan and exit check avoids duplicate API calls.
    """
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=window_days)
    index: dict[str, dict] = {}
    for w in pool:
        df = fetch_trader_trades(w, max_trades=500, use_cache=False)
        if df.empty:
            continue
        recent = df[df["timestamp"] >= cutoff]
        for r in recent.itertuples(index=False):
            tok = str(r.asset)
            if not tok:
                continue
            e = index.setdefault(tok, {
                "token": tok,
                "condition_id": getattr(r, "conditionId", ""),
                "question": getattr(r, "title", ""),
                "slug": getattr(r, "slug", ""),
                "buyers": set(),
                "sellers": set(),
            })
            side = getattr(r, "side", "")
            if side == "BUY":
                e["buyers"].add(w)
            elif side == "SELL":
                e["sellers"].add(w)
    logger.info("_build_flow_index: %d tokens seen across %d wallets", len(index), len(pool))
    return index


def scan_smart_flow_entries(
    window_days: int = 7,
    min_buyers: int = 3,
    per_window: int = 60,
    min_price: float = 0.05,
    max_price: float = 0.90,
    _flow_index: dict | None = None,
) -> list[dict]:
    """Open outcome tokens with >=``min_buyers`` distinct non-MM wallets buying recently.

    Pass a pre-built ``_flow_index`` (from ``_build_flow_index``) to avoid
    re-fetching trades when called from within ``update_ledger``.
    """
    if _flow_index is not None:
        flow = _flow_index
    else:
        pool = smart_pool(per_window)
        flow = _build_flow_index(pool, window_days)

    candidates = []
    for tok, e in flow.items():
        n = len(e["buyers"])
        if n < min_buyers:
            continue
        book = fetch_order_book(tok)
        feat = order_book_features(book)
        if not feat:                       # no live book -> resolved / untradeable
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
    logger.info("scan_smart_flow_entries: %d tokens with >=%d smart buyers (open)",
                len(candidates), min_buyers)
    return candidates


def consensus_reversal_check(
    token: str,
    flow_index: dict,
    flip_threshold: int = 1,
) -> bool:
    """Return True if smart-wallet net flow for this token has turned bearish.

    Fires when distinct sellers >= distinct buyers + flip_threshold in the
    trailing window, meaning the smart-money crowd has more sellers than buyers.
    Returns False silently if the token was not seen in any wallet's trades
    (no recent activity either way is not a reversal).
    """
    e = flow_index.get(token)
    if e is None:
        return False
    n_buy  = len(e.get("buyers",  set()))
    n_sell = len(e.get("sellers", set()))
    fired  = n_sell >= n_buy + flip_threshold
    if fired:
        logger.info("CONSENSUS FLIP candidate: token=%s  buy=%d sell=%d threshold=%d",
                    token[:20], n_buy, n_sell, flip_threshold)
    return fired


# ---------------------------------------------------------------------------
# Ledger I/O
# ---------------------------------------------------------------------------

def load_ledger() -> pd.DataFrame:
    if not STATE_FILE.exists():
        return pd.DataFrame(columns=_COLS)
    df = pd.read_csv(STATE_FILE, dtype=str)
    if "bet_fraction" not in df.columns:
        # Migrate legacy CSVs: assign flat-10% default so historical PnL is unchanged.
        df.insert(df.columns.get_loc("n_smart_buyers") + 1, "bet_fraction", "0.1")
    for col in _COLS:
        if col not in df.columns:
            df[col] = ""
    return df[_COLS]


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

def update_ledger(
    bet_fraction: float = 0.10,
    min_buyers: int = 3,
    window_days: int = 7,
    use_kelly: bool = False,
    ref_buyers: int = 4,
    kelly_cap: float = 2.0,
    stop_loss: float | None = None,
    consensus_exit: bool = False,
    flip_threshold: int = 1,
    per_window: int = 60,
) -> pd.DataFrame:
    """One daily cycle: append new consensus longs, refresh prices, mark resolutions.

    Parameters
    ----------
    bet_fraction : float
        Base bankroll fraction per position. When use_kelly=True this is the
        reference fraction for a ref_buyers-conviction trade; higher-conviction
        trades are sized proportionally larger (capped at bet_fraction * kelly_cap).
    use_kelly : bool
        If True, scale each position's fraction by n_smart_buyers / ref_buyers.
    ref_buyers : int
        Buyer count that maps to exactly bet_fraction (Kelly reference point).
    kelly_cap : float
        Maximum Kelly multiple over bet_fraction (prevents runaway sizing).
    stop_loss : float | None
        Exit an open position when current_price <= entry_ask * (1 - stop_loss).
        E.g. 0.35 exits when price falls to 65% of entry. None disables stops.
    consensus_exit : bool
        If True, exit positions where smart-wallet net flow has flipped bearish
        (more distinct sellers than buyers in the trailing window).
    flip_threshold : int
        Sellers must exceed buyers by at least this many to trigger consensus exit.
    per_window : int
        Number of wallets to fetch from the leaderboard for the smart pool.
    """
    ledger = load_ledger()
    today = date.today().isoformat()
    held = set(ledger["token"].tolist()) if not ledger.empty else set()

    # -- 1. build shared flow index (entries + exits share the same API fetch) -
    pool       = smart_pool(per_window)
    flow_index = _build_flow_index(pool, window_days)

    # -- 2. new consensus entries --
    new_rows = []
    for c in scan_smart_flow_entries(window_days=window_days, min_buyers=min_buyers,
                                     _flow_index=flow_index):
        if c["token"] in held:
            continue
        n = c["n_smart_buyers"]
        frac = kelly_fraction(n, bet_fraction, ref_buyers, kelly_cap) if use_kelly else bet_fraction
        logger.info("NEW long: %s  (%d buyers)  mid %.3f ask %.3f  frac=%.3f",
                    (c["question"] or "")[:40], n, c["entry_mid"], c["entry_ask"], frac)
        new_rows.append({
            "token": c["token"], "condition_id": c["condition_id"],
            "question": c["question"], "domain": c["domain"], "entry_date": today,
            "n_smart_buyers": str(n), "bet_fraction": str(frac),
            "entry_mid": str(c["entry_mid"]), "entry_ask": str(c["entry_ask"]),
            "spread": str(c["spread"]), "current_price": str(c["entry_mid"]),
            "status": "open", "exit_date": "", "outcome": "", "pnl": "",
        })
    if new_rows:
        ledger = pd.concat([ledger, pd.DataFrame(new_rows)], ignore_index=True)

    # -- 3. refresh + stop-loss + consensus-exit + resolve open positions --
    for i, row in ledger[ledger["status"] == "open"].iterrows():
        mid, _ = _price(row["token"])
        if mid is None:
            continue
        ledger.at[i, "current_price"] = str(round(mid, 4))
        entry_ask = float(row["entry_ask"])
        frac = float(row["bet_fraction"]) if row.get("bet_fraction") else bet_fraction

        # stop-loss: check before resolution so we don't double-fire
        if stop_loss is not None:
            stop_floor = entry_ask * (1.0 - stop_loss)
            if mid <= stop_floor:
                pnl = (mid / entry_ask - 1.0) * frac
                ledger.at[i, "status"]        = "stopped"
                ledger.at[i, "exit_date"]     = today
                ledger.at[i, "outcome"]       = ""
                ledger.at[i, "pnl"]           = str(round(pnl, 4))
                logger.info("STOPPED %s  mid=%.4f floor=%.4f pnl=%.4f",
                            (row["question"] or "")[:40], mid, stop_floor, pnl)
                continue

        # consensus exit: signal reversal
        if consensus_exit:
            if consensus_reversal_check(row["token"], flow_index, flip_threshold):
                pnl = (mid / entry_ask - 1.0) * frac
                ledger.at[i, "status"]    = "flipped"
                ledger.at[i, "exit_date"] = today
                ledger.at[i, "outcome"]   = ""
                ledger.at[i, "pnl"]      = str(round(pnl, 4))
                logger.info("FLIPPED  %s  mid=%.4f  pnl=%.4f",
                            (row["question"] or "")[:40], mid, pnl)
                continue

        # resolution: price collapsed to 0 or 1
        outcome = 1.0 if mid >= 0.99 else (0.0 if mid <= 0.01 else None)
        if outcome is not None:
            pnl = (outcome / entry_ask - 1.0) * frac
            ledger.at[i, "status"]        = "won" if outcome == 1.0 else "lost"
            ledger.at[i, "exit_date"]     = today
            ledger.at[i, "outcome"]       = str(outcome)
            ledger.at[i, "pnl"]          = str(round(pnl, 4))
            ledger.at[i, "current_price"] = str(outcome)
            logger.info("RESOLVED %s  %s  pnl=%.4f",
                        (row["question"] or "")[:40], ledger.at[i, "status"], pnl)

    save_ledger(ledger)
    return ledger
