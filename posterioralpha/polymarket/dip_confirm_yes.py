"""Live paper-trade tracker for the dip-confirm YES entry strategy.

Observation from backtesting: markets in [0.10, 0.50] YES at 7d out that
show a ≥10% intra-day dip followed by recovery to ≥95% of watch price by
day 2 resolve YES 74-88% of the time vs a 38-40% base rate.

Two-stage lifecycle
-------------------
  watching : spotted at WATCH_MIN–WATCH_MAX days to resolution, price in band.
             True minimum tracked via 1-minute CLOB history each cycle.
  open     : dip-confirm triggered — dipped ≥ DIP_THRESH below watch_price
             and recovered to ≥ RECOVERY_FLOOR × watch_price.
             Entered at live ask, held to resolution.
  expired  : reached <CONFIRM_MIN_DAYS to resolution without triggering.
  won/lost : resolved.

State columns
-------------
token            : YES token id
condition_id     : market condition id
question         : market question
end_date         : scheduled resolution
watch_date       : date first spotted
watch_price      : YES mid when first spotted
min_price        : true minimum from 1-min CLOB history since watch_date
days_at_watch    : days-to-res when first spotted
entry_date       : date entered (after confirm fires)
days_at_entry    : days-to-res when entered
bet_fraction     : bankroll fraction staked
entry_mid        : mid at entry
entry_ask        : ask at entry (cost basis)
spread           : entry_ask − entry_mid
current_price    : latest mid (refreshed each cycle)
status           : watching | open | expired | won | lost
exit_date        : date resolved/expired
outcome          : 1.0 / 0.0 (blank while open/watching)
pnl              : (outcome / entry_ask − 1) × bet_fraction
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .fetch import fetch_markets, fetch_order_book, fetch_token_history_raw, order_book_features

logger = logging.getLogger(__name__)

STATE_FILE = Path(__file__).resolve().parents[2] / "data" / "paper_trade" / "dip_confirm_positions.csv"

# Strategy parameters (backtest-optimal)
BAND_LO, BAND_HI    = 0.10, 0.50
WATCH_MIN_DAYS      = 5     # start watching markets this many days out
WATCH_MAX_DAYS      = 9     # ... up to this many days out
CONFIRM_MIN_DAYS    = 2     # minimum days remaining to still enter after confirm
DIP_THRESH          = 0.10  # need ≥10% intra-period dip below watch_price
RECOVERY_FLOOR      = 0.95  # price must recover to ≥95% of watch_price to confirm

_COLS = [
    "token", "condition_id", "question", "end_date",
    "watch_date", "watch_price", "min_price", "days_at_watch",
    "entry_date", "days_at_entry", "bet_fraction",
    "entry_mid", "entry_ask", "spread", "current_price",
    "status", "exit_date", "outcome", "pnl",
]


def _days_to_res(end_date: str | None) -> float | None:
    if not end_date:
        return None
    try:
        end = pd.to_datetime(end_date, utc=True)
    except (ValueError, TypeError):
        return None
    return (end - pd.Timestamp.now(tz="UTC")).total_seconds() / 86400.0


def _mid(token: str) -> float | None:
    feat = order_book_features(fetch_order_book(token))
    return feat["mid"] if feat else None


def _ask(token: str) -> tuple[float, float] | None:
    feat = order_book_features(fetch_order_book(token))
    return (feat["mid"], feat["best_ask"]) if feat else None


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


def update_ledger(
    bet_fraction: float = 0.10,
    n_markets: int = 900,
    min_volume: float = 5_000.0,
    state_file: Path = STATE_FILE,
) -> pd.DataFrame:
    """One cycle: add new watch entries, check confirms, resolve open positions."""
    ledger = load_ledger(state_file)
    today = date.today().isoformat()
    held_tokens = set(ledger["token"].tolist()) if not ledger.empty else set()

    # ── 1. Scan for new markets to watch ─────────────────────────────────────
    markets = fetch_markets(n_markets=n_markets, min_volume=min_volume, closed=False)
    new_rows = []
    for m in markets.itertuples(index=False):
        dtr = _days_to_res(getattr(m, "end_date", None))
        if dtr is None or not (WATCH_MIN_DAYS <= dtr <= WATCH_MAX_DAYS):
            continue
        if m.yes_token in held_tokens:
            continue
        feat = order_book_features(fetch_order_book(m.yes_token))
        if not feat:
            continue
        mid = feat["mid"]
        if not (BAND_LO <= mid < BAND_HI):
            continue
        logger.info("WATCH: %s  %.1fd out  mid=%.3f", (m.question or "")[:42], dtr, mid)
        new_rows.append({
            "token": m.yes_token, "condition_id": getattr(m, "condition_id", ""),
            "question": m.question, "end_date": getattr(m, "end_date", ""),
            "watch_date": today, "watch_price": str(round(mid, 4)),
            "min_price": str(round(mid, 4)), "days_at_watch": str(round(dtr, 2)),
            "entry_date": "", "days_at_entry": "", "bet_fraction": str(bet_fraction),
            "entry_mid": "", "entry_ask": "", "spread": "", "current_price": str(round(mid, 4)),
            "status": "watching", "exit_date": "", "outcome": "", "pnl": "",
        })
        held_tokens.add(m.yes_token)

    if new_rows:
        ledger = pd.concat([ledger, pd.DataFrame(new_rows)], ignore_index=True)

    # ── 2. Update watching positions — check dip-confirm ─────────────────────
    for i, row in ledger[ledger["status"] == "watching"].iterrows():
        dtr = _days_to_res(row.get("end_date"))
        prices = _ask(row["token"])
        if prices is None:
            continue
        mid, ask = prices
        ledger.at[i, "current_price"] = str(round(mid, 4))

        watch_price = float(row["watch_price"])

        # Fetch full 1-minute price history since watch_date so we catch any
        # intraday dip that recovered within a single hourly polling cycle.
        try:
            hist = fetch_token_history_raw(row["token"], fidelity_minutes=1, use_cache=False)
            if not hist.empty and row.get("watch_date"):
                since = pd.Timestamp(row["watch_date"], tz="UTC")
                hist = hist[hist.index >= since]
            hist_min = float(hist.min()) if not hist.empty else mid
        except Exception:
            hist_min = mid

        # True minimum = lowest price seen in history OR current order-book mid
        min_price = min(hist_min, mid, float(row["min_price"]) if row.get("min_price") else mid)
        ledger.at[i, "min_price"] = str(round(min_price, 4))

        # expire if too close to resolution without confirming
        if dtr is not None and dtr < CONFIRM_MIN_DAYS:
            ledger.at[i, "status"] = "expired"
            ledger.at[i, "exit_date"] = today
            logger.info("EXPIRED (no confirm): %s", (row["question"] or "")[:42])
            continue

        # check dip-confirm: dipped enough AND recovered enough
        dipped    = min_price <= watch_price * (1.0 - DIP_THRESH)
        recovered = mid >= watch_price * RECOVERY_FLOOR
        if dipped and recovered:
            frac = float(row["bet_fraction"]) if row.get("bet_fraction") else bet_fraction
            ledger.at[i, "status"]        = "open"
            ledger.at[i, "entry_date"]    = today
            ledger.at[i, "days_at_entry"] = str(round(dtr, 2)) if dtr else ""
            ledger.at[i, "entry_mid"]     = str(round(mid, 4))
            ledger.at[i, "entry_ask"]     = str(round(ask, 4))
            ledger.at[i, "spread"]        = str(round(ask - mid, 4))
            logger.info("CONFIRM → OPEN: %s  min=%.3f watch=%.3f now=%.3f",
                        (row["question"] or "")[:42], min_price, watch_price, mid)

    # ── 3. Resolve open positions ─────────────────────────────────────────────
    for i, row in ledger[ledger["status"] == "open"].iterrows():
        mid = _mid(row["token"])
        if mid is None:
            continue
        ledger.at[i, "current_price"] = str(round(mid, 4))
        outcome = 1.0 if mid >= 0.99 else (0.0 if mid <= 0.01 else None)
        if outcome is not None:
            entry_ask = float(row["entry_ask"])
            frac = float(row["bet_fraction"]) if row.get("bet_fraction") else bet_fraction
            pnl = (outcome / entry_ask - 1.0) * frac
            ledger.at[i, "status"]        = "won" if outcome == 1.0 else "lost"
            ledger.at[i, "exit_date"]     = today
            ledger.at[i, "outcome"]       = str(outcome)
            ledger.at[i, "pnl"]          = str(round(pnl, 4))
            ledger.at[i, "current_price"] = str(outcome)
            logger.info("RESOLVED %s  %s  pnl=%.4f",
                        (row["question"] or "")[:42], ledger.at[i, "status"], pnl)

    save_ledger(ledger, state_file)
    return ledger
