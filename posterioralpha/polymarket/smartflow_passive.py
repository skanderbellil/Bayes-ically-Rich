"""Forward paper-trade tracker for *passively entered* smart-flow consensus.

Pre-registered successor to the independence experiment (``smartflow_independence``),
which was retired on 2026-09-19 when both its frozen kill criteria fired. See
``docs/polymarket/SMART_FLOW_PASSIVE.md`` for the frozen thresholds and kill criteria,
registered before this ledger's first data point.

Why this exists
---------------
Retiring the discriminator left one finding standing, and it is about execution, not
about signal. Measured on the 7,788 resolved rows of the independence ledger:

    edge/$1 at the ask (what the ledger paid)   -0.00730   event-clustered t = -1.93
    edge/$1 at the mid (what it could have)     +0.00862   event-clustered t = +2.36
    mean recorded half-spread                    0.01731

-0.0090 + 0.0173 = +0.0083. The consensus signal is not noise -- it is worth roughly
0.9c/$1 at mid, and the sleeve pays ~1.7c/$1 to cross the spread to get it. The loss
IS the spread. That reading is the most out-of-sample-stable pattern in the ledger:
edge at mid is roughly flat across spread buckets while edge at the ask collapses
monotonically as the spread widens, in-sample and out-of-sample alike (the >5c bucket
loses 12.0% per trade in-sample and 12.6% out-of-sample at the ask, but only ~0.7% at
the mid).

So this experiment changes exactly two things about the incumbent consensus rule, both
about how the position is acquired, neither about which token is chosen:

  1. SPREAD CAP -- skip any candidate whose half-spread exceeds ``MAX_HALF_SPREAD``.
  2. PASSIVE ENTRY -- post a limit at the mid and wait, instead of lifting the offer.
     The order works for ``WORK_HOURS``; if the book never comes to us it EXPIRES
     UNFILLED and is recorded as such.

The honest failure mode, stated up front
----------------------------------------
Passive fills are ADVERSELY SELECTED. A resting bid gets hit precisely when someone
wants to sell into it, which correlates with the price about to fall -- so the realised
edge on filled orders will be worse than the +0.0086 mid-edge measured above, which was
computed on fills we know we would have got by crossing. The entire question this
ledger answers is whether enough of that mid-edge survives adverse selection to pay for
itself. ``unfilled`` is therefore a first-class recorded outcome, not an error: a
strategy that only fills when it is wrong is worthless even if its fill price looks good,
and that shows up here as a high fill rate on losers and a low one on winners.

State columns
-------------
token          : the outcome-token id (stable key)
condition_id   : parent market condition id
question       : market question text
domain         : coarse topic (politics/macro/sports/...)
end_date       : market end date (Gamma)
scan_date      : date the consensus gate fired and the limit order was posted
work_deadline  : UTC timestamp after which an unfilled order is cancelled
n_smart_buyers : distinct non-MM leaderboard wallets that bought it in the window
buyers         : pipe-separated, sorted buyer wallet addresses at scan time
scan_mid       : CLOB mid when the order was posted
scan_ask       : CLOB best-ask when the order was posted (what the incumbent would pay)
scan_spread    : scan_ask - scan_mid, the half-spread the cap is applied to
limit_px       : the resting bid we posted (== scan_mid; frozen, see LIMIT_AT_MID)
entry_date     : date the order FILLED (blank while working/unfilled) -- named
                 entry_date, not fill_date, so generate_dashboard's loader picks it up
                 without a special case
fill_px        : price we actually paid (== limit_px on fill; blank otherwise)
bet_fraction   : position size fraction (flat -- no Kelly, no conviction scaling)
current_price  : latest mid (refreshed each run)
status         : working | unfilled | open | won | lost | flipped
exit_date      : resolution/cancel date (blank while working or open)
outcome        : 1.0 won / 0.0 lost (blank unless resolved)
pnl            : (outcome / fill_px - 1) * bet_fraction  (blank unless resolved)
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from .fetch import fetch_market_resolution
from .smartflow_papertrade import (
    _fetch_pool_trades,
    _flow_index_from,
    _price,
    consensus_reversal_check,
    scan_smart_flow_entries,
    smart_pool,
)

logger = logging.getLogger(__name__)

STATE_FILE = Path(__file__).resolve().parents[2] / "data" / "paper_trade" / "smart_flow_passive_positions.csv"

_COLS = ["token", "condition_id", "question", "domain", "end_date",
         "scan_date", "work_deadline", "n_smart_buyers", "buyers",
         "scan_mid", "scan_ask", "scan_spread", "limit_px",
         "entry_date", "fill_px", "bet_fraction", "current_price",
         "status", "exit_date", "outcome", "pnl"]

# ---------------------------------------------------------------------------
# PRE-REGISTERED, FROZEN (SMART_FLOW_PASSIVE.md). Registered 2026-09-19, before
# this ledger's first row. Do not retune after data arrives -- a retuned
# threshold is a new experiment on a new ledger, not an edit to this one.
# ---------------------------------------------------------------------------
MAX_HALF_SPREAD = 0.02   # skip candidates whose (ask - mid) exceeds this
WORK_HOURS = 24.0        # how long a resting limit order stays live before cancel
LIMIT_AT_MID = True      # post at the mid exactly; not bid+tick, not mid-epsilon

# Kill criteria (frozen). Evaluated only once MIN_RESOLVED filled positions exist.
MIN_RESOLVED = 200          # filled AND resolved positions before any verdict
MIN_FILL_RATE = 0.20        # of MIN_WORKED terminal orders, below this = untradeable
MIN_WORKED = 200            # terminal (filled or expired) orders before the fill-rate rule
KILL_T_MIN = 1.0            # event-clustered t on edge/$1 at the FILLED price


def fill_rate(ledger: pd.DataFrame) -> dict:
    """Share of terminal orders that actually filled.

    'Terminal' means the order is no longer working: it either filled (and is now
    open/won/lost/flipped) or expired unfilled. Orders still working are excluded --
    counting them as misses would understate the rate early in each order's life.
    """
    if ledger.empty:
        return {"worked": 0, "filled": 0, "unfilled": 0, "rate": float("nan")}
    st = ledger["status"].astype(str)
    filled = int(st.isin(["open", "won", "lost", "flipped"]).sum())
    unfilled = int((st == "unfilled").sum())
    worked = filled + unfilled
    return {"worked": worked, "filled": filled, "unfilled": unfilled,
            "rate": (filled / worked) if worked else float("nan")}


def kill_check(ledger: pd.DataFrame) -> dict:
    """Recompute the frozen kill criteria from the ledger. Reports, never asserts.

    Primary: event-clustered mean edge/$1 at the filled price, one mean per
    settlement date (the repo's standard unit of independent evidence -- this
    book settles ~120 correlated positions on a single day, so a per-trade t
    overstates the evidence by an order of magnitude). ``t < KILL_T_MIN``
    once MIN_RESOLVED positions have resolved retires the sleeve.
    Viability: if fewer than MIN_FILL_RATE of MIN_WORKED terminal orders filled,
    passive entry at this limit is untradeable regardless of the edge on the
    fills we did get.

    Returns counts, the fill rate, ``edge``/``t``, each boolean, and ``armed``
    (False while the sample floors are unmet -- the criteria deliberately do not
    bind before then).
    """
    fr = fill_rate(ledger)
    out = {"n_resolved": 0, "events": 0, "edge": float("nan"), "t": float("nan"),
           "fill_rate": fr["rate"], "worked": fr["worked"],
           "armed": False, "primary_failed": False, "viability_failed": False,
           "verdict": "insufficient data"}
    if not ledger.empty:
        res = ledger[ledger["status"].isin(["won", "lost"])].copy()
        out["n_resolved"] = len(res)
        if len(res) >= 2:
            px = pd.to_numeric(res["fill_px"], errors="coerce")
            oc = pd.to_numeric(res["outcome"], errors="coerce")
            day = res["exit_date"].where(res["exit_date"].astype(str) != "", res["entry_date"])
            ev = (oc - px).groupby(day).mean().dropna()
            out["events"] = len(ev)
            if len(ev) >= 2 and ev.std(ddof=1) > 0:
                out["edge"] = float(ev.mean())
                out["t"] = float(ev.mean() / (ev.std(ddof=1) / len(ev) ** 0.5))

    out["viability_failed"] = fr["worked"] >= MIN_WORKED and fr["rate"] < MIN_FILL_RATE
    out["armed"] = out["n_resolved"] >= MIN_RESOLVED or out["viability_failed"]
    if not out["armed"]:
        return out
    out["primary_failed"] = out["n_resolved"] >= MIN_RESOLVED and not (out["t"] >= KILL_T_MIN)
    out["verdict"] = "RETIRED" if (out["primary_failed"] or out["viability_failed"]) else "alive"
    return out


# ---------------------------------------------------------------------------
# Ledger I/O
# ---------------------------------------------------------------------------

def load_ledger(state_file: Path = STATE_FILE) -> pd.DataFrame:
    if not state_file.exists():
        return pd.DataFrame(columns=_COLS)
    df = pd.read_csv(state_file, dtype=str)
    for col in _COLS:
        if col not in df.columns:
            df[col] = ""
    return df[_COLS].fillna("")


def save_ledger(df: pd.DataFrame, state_file: Path = STATE_FILE) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    df[_COLS].to_csv(state_file, index=False)


# ---------------------------------------------------------------------------
# Hourly update
# ---------------------------------------------------------------------------

def update_ledger(
    bet_fraction: float = 0.10,
    min_buyers: int = 3,
    window_days: int = 7,
    consensus_exit: bool = False,
    flip_threshold: int = 1,
    per_window: int = 60,
    state_file: Path = STATE_FILE,
    now: datetime | None = None,
) -> pd.DataFrame:
    """One hourly cycle: post limits on new in-spread consensus candidates, then
    try to fill working orders, then mark/resolve/exit whatever has filled.

    Parameters
    ----------
    bet_fraction : float
        Flat bankroll fraction per position. No Kelly: this experiment is about
        execution, so sizing must not vary with conviction and confound it.
    now : datetime | None
        Point-in-time reference for order expiry. Defaults to "now"; pass an
        explicit value to replay deterministically in tests.
    """
    ledger = load_ledger(state_file)
    ts = (now or datetime.now(timezone.utc)).replace(tzinfo=None)
    today = date.today().isoformat() if now is None else ts.date().isoformat()
    known = set(ledger["token"].tolist()) if not ledger.empty else set()

    # -- 1. shared flow index, built once and reused (never fetch trades twice) --
    pool = smart_pool(per_window)
    trades_by_wallet = _fetch_pool_trades(pool)
    flow_index = _flow_index_from(trades_by_wallet, window_days)

    # -- 2. post a resting limit on each NEW candidate that clears the spread cap --
    new_rows, skipped_wide = [], 0
    for c in scan_smart_flow_entries(window_days=window_days, min_buyers=min_buyers,
                                     _flow_index=flow_index):
        if c["token"] in known:
            continue
        half_spread = float(c["spread"])          # scan records ask - mid
        if half_spread > MAX_HALF_SPREAD:
            skipped_wide += 1
            continue                      # too expensive to be worth working
        if not LIMIT_AT_MID:              # frozen True; guard documents the choice
            raise NotImplementedError("limit placement other than the mid is a new experiment")
        limit_px = float(c["entry_mid"])
        logger.info("POST limit %.3f (ask %.3f, half-spread %.4f): %s (%d buyers)",
                    limit_px, c["entry_ask"], half_spread,
                    (c["question"] or "")[:40], c["n_smart_buyers"])
        new_rows.append({
            "token": c["token"], "condition_id": c["condition_id"],
            "question": c["question"], "domain": c["domain"],
            "end_date": c.get("end_date", ""),
            "scan_date": today,
            "work_deadline": (ts + timedelta(hours=WORK_HOURS)).isoformat(timespec="seconds"),
            "n_smart_buyers": str(c["n_smart_buyers"]), "buyers": c.get("buyers", ""),
            "scan_mid": str(c["entry_mid"]), "scan_ask": str(c["entry_ask"]),
            "scan_spread": str(round(half_spread, 4)), "limit_px": str(round(limit_px, 4)),
            "entry_date": "", "fill_px": "", "bet_fraction": str(bet_fraction),
            "current_price": str(c["entry_mid"]),
            "status": "working", "exit_date": "", "outcome": "", "pnl": "",
        })
    if skipped_wide:
        logger.info("skipped %d candidate(s) over the %.3f half-spread cap",
                    skipped_wide, MAX_HALF_SPREAD)
    if new_rows:
        ledger = pd.concat([ledger, pd.DataFrame(new_rows)], ignore_index=True)

    # -- 3. WORKING orders: fill if the offer has come down to our limit, else
    #       cancel once the order has worked past its deadline. A fill needs the
    #       book to cross US -- best_ask <= limit means someone is willing to sell
    #       at or below our resting bid, which is exactly when we'd be hit.
    for i, row in ledger[ledger["status"] == "working"].iterrows():
        limit_px = float(row["limit_px"])
        mid, ask = _price(row["token"])
        if mid is not None:
            ledger.at[i, "current_price"] = str(round(mid, 4))
        if ask is not None and ask <= limit_px + 1e-9:
            ledger.at[i, "status"] = "open"
            ledger.at[i, "entry_date"] = today
            ledger.at[i, "fill_px"] = str(round(limit_px, 4))
            logger.info("FILLED at %.3f: %s", limit_px, (row["question"] or "")[:40])
            continue
        deadline = str(row["work_deadline"])
        if deadline and ts >= pd.Timestamp(deadline).to_pydatetime():
            ledger.at[i, "status"] = "unfilled"
            ledger.at[i, "exit_date"] = today
            logger.info("EXPIRED unfilled (limit %.3f): %s",
                        limit_px, (row["question"] or "")[:40])

    # -- 4. FILLED positions: mark, resolve, consensus-exit (incumbent mechanics) --
    for i, row in ledger[ledger["status"] == "open"].iterrows():
        fill_px = float(row["fill_px"])
        frac = float(row["bet_fraction"]) if row.get("bet_fraction") else bet_fraction

        # resolution first, from the AUTHORITATIVE CLOB market status -- catches
        # settled markets whose order book is already gone.
        res = fetch_market_resolution(row.get("condition_id", ""))
        market_closed = bool(res and res["closed"])
        outcome = None
        if market_closed:
            toks = res["tokens"]
            finalised = any(t["winner"] or t["price"] >= 0.99 for t in toks.values())
            t = toks.get(str(row["token"]))
            if finalised and t is not None:
                outcome = 1.0 if (t["winner"] or t["price"] >= 0.99) else 0.0

        mid, _ = _price(row["token"])
        if mid is not None:
            ledger.at[i, "current_price"] = str(round(mid, 4))
        if outcome is None and mid is None:
            continue           # neither resolution nor a fresh mark -- leave untouched

        # price-heuristic fallback only when the API confirms the market closed
        # but could not produce a clean per-token label.
        if outcome is None and market_closed and mid is not None:
            outcome = 1.0 if mid >= 0.99 else (0.0 if mid <= 0.01 else None)
        if outcome is not None:
            pnl = (outcome / fill_px - 1.0) * frac
            ledger.at[i, "status"] = "won" if outcome == 1.0 else "lost"
            ledger.at[i, "exit_date"] = today
            ledger.at[i, "outcome"] = str(outcome)
            ledger.at[i, "pnl"] = str(round(pnl, 4))
            ledger.at[i, "current_price"] = str(outcome)
            logger.info("RESOLVED %s %s pnl=%.4f", (row["question"] or "")[:40],
                        ledger.at[i, "status"], pnl)
            continue

        if consensus_exit and consensus_reversal_check(row["token"], flow_index, flip_threshold):
            pnl = (mid / fill_px - 1.0) * frac
            ledger.at[i, "status"] = "flipped"
            ledger.at[i, "exit_date"] = today
            ledger.at[i, "pnl"] = str(round(pnl, 4))
            logger.info("FLIPPED %s mid=%.4f pnl=%.4f", (row["question"] or "")[:40], mid, pnl)

    save_ledger(ledger, state_file)
    return ledger
