"""Forward paper-trade tracker for the mid-priced-YES convergence signal.

The live, out-of-sample test of the strongest pattern found in the price-history
deep dive: **YES tokens trading in a mid/low band a few days before resolution
resolve YES more often than their price implies**, so buying them and holding to
resolution carries a positive edge.  The backtest (1,500 resolved markets, 9,111
obs) located the edge's sweet spot at roughly **5 days to resolution** in the
**[0.10, 0.50]** price band (n=263, win 44%, mean +53.6%, per-trade Sharpe 0.26).

This module runs that as a live ledger, mirroring ``smartflow_papertrade.py``:
each cycle it finds currently-open markets whose scheduled resolution is ~5 days
out and whose YES price sits in the band, logs a paper YES long **at the live
CLOB ask** (so the spread is paid for real), and holds every position to
resolution — no exits, because every exit rule we tested (stops, trailing,
decaying-horizon, timed) underperformed buy-and-hold.

The historical backtest is *baked in* (``historical_backtest``) so each run can
print the benchmark the live ledger is being measured against.

State columns
-------------
token            : YES outcome-token id (stable key)
condition_id     : parent market condition id
question         : market question text
end_date         : scheduled resolution timestamp (ISO)
entry_date       : date first logged
days_to_res      : days to scheduled resolution at entry (~5 by design)
bet_fraction     : bankroll fraction staked
entry_mid        : CLOB mid at entry
entry_ask        : CLOB best-ask at entry (what we "pay")
spread           : entry_ask − entry_mid
current_price    : latest mid (refreshed each run)
status           : open | won | lost
exit_date        : resolution date (blank while open)
outcome          : 1.0 won / 0.0 lost (blank while open)
pnl              : (outcome / entry_ask − 1) · bet_fraction
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .fetch import (
    fetch_markets,
    fetch_order_book,
    fetch_token_history,
    fetch_token_history_raw,
    order_book_features,
)

logger = logging.getLogger(__name__)

STATE_FILE = Path(__file__).resolve().parents[2] / "data" / "paper_trade" / "midprice_yes_positions.csv"
BENCH_FILE = Path(__file__).resolve().parents[2] / "data" / "paper_trade" / "midprice_yes_benchmark.csv"

_COLS = ["token", "condition_id", "question", "end_date", "entry_date",
         "days_to_res", "bet_fraction", "entry_mid", "entry_ask", "spread",
         "current_price", "status", "exit_date", "outcome", "pnl"]

# Strategy defaults — the backtest-optimal configuration
BAND_LO, BAND_HI = 0.10, 0.50
TARGET_DAYS = 5
# Entry window. The backtest showed a broad Sharpe plateau (~0.19–0.26) across
# 1–10 days to resolution, peaking near 5d; markets cluster at specific dates so
# a wide catch window is needed to actually find entries. [2,10] sits squarely
# inside the plateau and reliably has markets in it.
DAYS_MIN, DAYS_MAX = 2, 10


# ---------------------------------------------------------------------------
# Time-to-resolution helper
# ---------------------------------------------------------------------------

def _days_to_res(end_date: str | None) -> float | None:
    if not end_date:
        return None
    try:
        end = pd.to_datetime(end_date, utc=True)
    except (ValueError, TypeError):
        return None
    now = pd.Timestamp.now(tz="UTC")
    return (end - now).total_seconds() / 86400.0


# ---------------------------------------------------------------------------
# Scan open markets ~5d from resolution priced in band
# ---------------------------------------------------------------------------

def scan_midprice_entries(
    days_min: int = DAYS_MIN,
    days_max: int = DAYS_MAX,
    lo: float = BAND_LO,
    hi: float = BAND_HI,
    n_markets: int = 900,
    min_volume: float = 5_000.0,
    min_liquidity: float = 500.0,
) -> list[dict]:
    """Open markets resolving in [days_min, days_max] days with YES mid in [lo, hi)."""
    markets = fetch_markets(n_markets=n_markets, min_volume=min_volume,
                            min_liquidity=min_liquidity, closed=False)
    cands = []
    for m in markets.itertuples(index=False):
        dtr = _days_to_res(getattr(m, "end_date", None))
        if dtr is None or not (days_min <= dtr <= days_max):
            continue
        feat = order_book_features(fetch_order_book(m.yes_token))
        if not feat:
            continue
        mid, ask = feat["mid"], feat["best_ask"]
        if not (lo <= mid < hi):
            continue
        cands.append({
            "token": m.yes_token,
            "condition_id": getattr(m, "condition_id", ""),
            "question": m.question,
            "end_date": getattr(m, "end_date", ""),
            "days_to_res": round(dtr, 2),
            "entry_mid": round(mid, 4),
            "entry_ask": round(ask, 4),
            "spread": round(ask - mid, 4),
        })
    cands.sort(key=lambda c: c["days_to_res"])
    logger.info("scan_midprice_entries: %d open markets in band [%.2f,%.2f) ~%d–%dd out",
                len(cands), lo, hi, days_min, days_max)
    return cands


# ---------------------------------------------------------------------------
# Ledger I/O
# ---------------------------------------------------------------------------

def load_ledger(state_file: Path = STATE_FILE) -> pd.DataFrame:
    if not state_file.exists():
        return pd.DataFrame(columns=_COLS)
    df = pd.read_csv(state_file, dtype=str)
    for c in _COLS:
        if c not in df.columns:
            df[c] = ""
    return df[_COLS]


def save_ledger(df: pd.DataFrame, state_file: Path = STATE_FILE) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    df[_COLS].to_csv(state_file, index=False)


def _price(token: str) -> float | None:
    feat = order_book_features(fetch_order_book(token))
    if feat:
        return feat["mid"]
    # No active order book — market likely resolved. Fall back to CLOB history.
    try:
        hist = fetch_token_history_raw(token, fidelity_minutes=60, use_cache=False)
        if not hist.empty:
            return float(hist.iloc[-1])
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Daily update — enter new, mark open, resolve. NO exits (hold to resolution).
# ---------------------------------------------------------------------------

def update_ledger(
    bet_fraction: float = 0.10,
    days_min: int = DAYS_MIN,
    days_max: int = DAYS_MAX,
    lo: float = BAND_LO,
    hi: float = BAND_HI,
    min_volume: float = 5_000.0,
    state_file: Path = STATE_FILE,
    pre_scanned: list[dict] | None = None,
) -> pd.DataFrame:
    """One cycle: append new band entries, refresh prices, mark resolutions.

    pre_scanned: if provided (from a shared scan), filter it to this band instead
    of re-fetching markets from the API.
    """
    ledger = load_ledger(state_file=state_file)
    today = date.today().isoformat()
    held = set(ledger["token"].tolist()) if not ledger.empty else set()

    # 1. new entries — use pre-scanned list (filtered to band) or do a fresh scan
    if pre_scanned is not None:
        candidates = [c for c in pre_scanned if lo <= c["entry_mid"] < hi]
    else:
        candidates = scan_midprice_entries(days_min, days_max, lo, hi, min_volume=min_volume)

    new_rows = []
    for c in candidates:
        if c["token"] in held:
            continue
        logger.info("NEW YES long: %s  (%.1fd out)  mid %.3f ask %.3f",
                    (c["question"] or "")[:42], c["days_to_res"], c["entry_mid"], c["entry_ask"])
        new_rows.append({
            "token": c["token"], "condition_id": c["condition_id"],
            "question": c["question"], "end_date": c["end_date"], "entry_date": today,
            "days_to_res": str(c["days_to_res"]), "bet_fraction": str(bet_fraction),
            "entry_mid": str(c["entry_mid"]), "entry_ask": str(c["entry_ask"]),
            "spread": str(c["spread"]), "current_price": str(c["entry_mid"]),
            "status": "open", "exit_date": "", "outcome": "", "pnl": "",
        })
    if new_rows:
        ledger = pd.concat([ledger, pd.DataFrame(new_rows)], ignore_index=True)

    # 2. refresh + resolve open positions (hold to resolution — no exits)
    for i, row in ledger[ledger["status"] == "open"].iterrows():
        mid = _price(row["token"])
        if mid is None:
            continue
        ledger.at[i, "current_price"] = str(round(mid, 4))
        outcome = 1.0 if mid >= 0.99 else (0.0 if mid <= 0.01 else None)
        if outcome is not None:
            entry_ask = float(row["entry_ask"])
            frac = float(row["bet_fraction"]) if row.get("bet_fraction") else bet_fraction
            pnl = (outcome / entry_ask - 1.0) * frac
            ledger.at[i, "status"] = "won" if outcome == 1.0 else "lost"
            ledger.at[i, "exit_date"] = today
            ledger.at[i, "outcome"] = str(outcome)
            ledger.at[i, "pnl"] = str(round(pnl, 4))
            ledger.at[i, "current_price"] = str(outcome)
            logger.info("RESOLVED %s  %s  pnl=%.4f",
                        (row["question"] or "")[:42], ledger.at[i, "status"], pnl)

    save_ledger(ledger, state_file=state_file)
    return ledger


# ---------------------------------------------------------------------------
# Baked-in historical backtest (the benchmark the live ledger is judged against)
# ---------------------------------------------------------------------------

def historical_backtest(
    n_markets: int = 800,
    days_min: int = DAYS_MIN,
    days_max: int = DAYS_MAX,
    lo: float = BAND_LO,
    hi: float = BAND_HI,
    haircut: float = 0.02,
    min_volume: float = 1_000.0,
    use_cache: bool = True,
    sleep: float = 0.0,
) -> dict:
    """Backtest 'buy YES in [lo,hi) somewhere in the [days_min,days_max] window,
    hold to resolution'.

    Pools every integer day in the entry window as a separate observation, so the
    benchmark reflects the live strategy's behaviour (each market is entered once
    at some day within the window, ~uniformly as the cron rolls). Returns summary
    stats and caches them to BENCH_FILE. Set use_cache=False to recompute.
    """
    if use_cache and BENCH_FILE.exists():
        row = pd.read_csv(BENCH_FILE).iloc[0].to_dict()
        return row

    import time
    markets = fetch_markets(n_markets=n_markets, min_volume=min_volume, closed=True)
    markets = markets[markets["outcome"].isin([0.0, 1.0])].reset_index(drop=True)
    rets = []
    for i, m in enumerate(markets.itertuples(index=False), 1):
        s = fetch_token_history(m.yes_token, fidelity_minutes=1440)
        if len(s) < 3:
            continue
        t_res = s.index.max()
        for d in range(days_min, days_max + 1):
            sub = s[s.index <= t_res - pd.Timedelta(days=d)]
            if sub.empty:
                continue
            p0 = float(sub.iloc[-1])
            if not (lo <= p0 < hi):
                continue
            entry = min(p0 + haircut, 0.99)
            rets.append(float(m.outcome) / entry - 1.0)
        if sleep and i % 5 == 0:
            time.sleep(sleep)
    rets = np.array(rets)
    if len(rets) == 0:
        stats = {"n": 0, "win_rate": float("nan"), "mean_ret": float("nan"),
                 "median_ret": float("nan"), "sharpe": float("nan"), "total_pnl": 0.0}
    else:
        std = rets.std(ddof=1) if len(rets) > 1 else float("nan")
        stats = {
            "n": int(len(rets)),
            "win_rate": float((rets > 0).mean()),
            "mean_ret": float(rets.mean()),
            "median_ret": float(np.median(rets)),
            "sharpe": float(rets.mean() / std) if std and std > 0 else float("nan"),
            "total_pnl": float(rets.sum()),
        }
    stats.update({"band_lo": lo, "band_hi": hi, "days_min": days_min, "days_max": days_max,
                  "haircut": haircut, "computed": datetime.now(timezone.utc).isoformat()})
    BENCH_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([stats]).to_csv(BENCH_FILE, index=False)
    logger.info("historical_backtest: n=%d win=%.1f%% mean=%.1f%% sharpe=%.2f",
                stats["n"], stats["win_rate"] * 100 if stats["n"] else float("nan"),
                stats["mean_ret"] * 100 if stats["n"] else float("nan"), stats["sharpe"])
    return stats
