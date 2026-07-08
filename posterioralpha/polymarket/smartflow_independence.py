"""Forward paper-trade tracker for *independence-classified* smart-flow consensus.

A pre-registered discriminator experiment, not a new signal. The incumbent
``smartflow_papertrade`` consensus rule (`>= N distinct smart wallets bought this
token recently`) is forward-losing (-52% MTM, edge t~0.09 over 822 markets seen —
see ``docs/knowledge/epistemology/README.md`` Lesson 2). The diagnosis: counting
co-buying wallets cannot tell *independent discovery* (Condorcet's jury theorem —
many people reasoning from separate evidence, which is genuinely informative) apart
from an *information cascade* (people imitating the first few movers, which adds
no information at all — Condorcet's assumption of independence is exactly what
breaks). This module classifies every consensus candidate the incumbent would take
into ``independent`` / ``cascade`` using three cheap, pre-registered proxies computed
from the same wallet trade histories already being fetched, and logs **both** classes
to the **same** ledger at the **same** flat bet size — the experiment is entirely in
the split, so the trading mechanics must not differ by class. See
``docs/polymarket/SMART_FLOW_INDEPENDENCE.md`` for the frozen thresholds and kill
criteria (pre-registered before the first data point).

Reuses ``smartflow_papertrade``'s plumbing (smart pool, flow index, order-book entry,
resolution, consensus-exit) rather than duplicating it — only the independence
features/classification and the ledger are new.

State columns
-------------
token                 : the outcome-token id (stable key)
condition_id          : parent market condition id
question              : market question text
domain                : coarse topic (politics/macro/sports/...)
entry_date            : date first logged
n_smart_buyers        : distinct non-MM leaderboard wallets that bought it in the window
first_buy_span_hours  : hours between the earliest and latest buyer's FIRST buy of this
                        token in the window (wide span -> buyers arrived independently
                        over time; narrow span -> everyone piled in together)
price_chase           : price paid by the chronologically LAST first-buyer minus the
                        price paid by the EARLIEST first-buyer (positive -> later
                        entrants paid up after the move -- a cascade signature)
pairwise_jaccard      : mean pairwise Jaccard similarity of the buyers' full traded
                        conditionId histories (high -> the same clique trades
                        everything together -> not independent evidence)
indep_score           : count of the three component tests passed (0-3)
indep_class           : "independent" if indep_score >= 2 else "cascade"
bet_fraction          : position size fraction (flat -- no Kelly; both classes sized
                        identically so the paired comparison is clean)
entry_mid             : CLOB mid at entry
entry_ask             : CLOB best-ask at entry (what we actually "pay" -- captures spread)
spread                : entry_ask - entry_mid (the realised half-cost)
current_price         : latest mid (refreshed each run)
status                : open | won | lost | flipped
exit_date             : resolution date (blank while open)
outcome               : 1.0 won / 0.0 lost (blank while open)
pnl                   : (outcome / entry_ask - 1) * fraction  (blank while open)
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from itertools import combinations
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

STATE_FILE = Path(__file__).resolve().parents[2] / "data" / "paper_trade" / "smart_flow_indep_positions.csv"

_COLS = ["token", "condition_id", "question", "domain", "end_date", "entry_date",
         "n_smart_buyers",
         "first_buy_span_hours", "price_chase", "pairwise_jaccard",
         "indep_score", "indep_class",
         "bet_fraction",
         "entry_mid", "entry_ask", "spread", "current_price",
         "status", "exit_date", "outcome", "pnl"]

# Pre-registered, frozen component thresholds (SMART_FLOW_INDEPENDENCE.md). Do not
# retune after data arrives -- a retuned threshold is a new experiment on a new ledger.
TEMPORAL_SPAN_HOURS = 24.0   # first-buy span must be >= this to "pass" temporal
PRICE_CHASE_MAX = 0.05       # last-first-buyer must not have paid up more than this
JACCARD_MAX = 0.20           # mean pairwise history overlap must not exceed this


# ---------------------------------------------------------------------------
# Independence features + classification
# ---------------------------------------------------------------------------

def independence_features(
    token: str,
    buyers: set[str],
    trades_by_wallet: dict[str, pd.DataFrame],
    window_days: int,
    asof: datetime | None = None,
) -> dict:
    """Three Condorcet-independence proxies for one token's buyer set.

    - first_buy_span_hours: hours between the earliest and latest buyer's first
      buy of ``token`` within the trailing ``window_days``. A cascade shows
      everyone piling in within minutes/hours of each other; independent
      discovery is spread out.
    - price_chase: (price paid by the chronologically LAST first-buyer) minus
      (price paid by the EARLIEST first-buyer). Positive means later entrants
      paid up after the move -- the classic footprint of chasing a crowd rather
      than acting on separate information.
    - pairwise_jaccard: mean pairwise Jaccard similarity of the buyers' full
      *fetched-history* conditionId sets (not restricted to the window). High
      overlap means the same clique trades everything together -- correlated
      behaviour, not independent votes. With <2 buyers this is 0.0 (shouldn't
      arise given the caller's min_buyers >= 3 floor).

    All fields are computed from ``trades_by_wallet``, the same per-wallet trade
    DataFrames already fetched to build the flow index -- no extra API calls.

    Parameters
    ----------
    asof : datetime | None
        Point-in-time reference the trailing window is measured back from.
        Defaults to ``None``, which means "now" (``datetime.now(timezone.utc)``)
        -- the live hourly-update call path is unchanged. Pass an explicit
        ``asof`` to replay this feature at a historical point in time (e.g. a
        backtest walking day by day) without duplicating this function's logic.
    """
    now = asof if asof is not None else datetime.now(timezone.utc)
    if now.tzinfo is not None:
        now = now.astimezone(timezone.utc).replace(tzinfo=None)
    cutoff = now - timedelta(days=window_days)

    # -- first buy (timestamp, price) per buyer, within the trailing window --
    first_buys: dict[str, tuple[pd.Timestamp, float]] = {}
    for w in buyers:
        df = trades_by_wallet.get(w)
        if df is None or df.empty:
            continue
        mask = (
            (df["asset"].astype(str) == str(token))
            & (df["side"] == "BUY")
            & (df["timestamp"] >= cutoff)
        )
        sub = df[mask]
        if sub.empty:
            continue
        sub = sub.sort_values("timestamp")
        first_row = sub.iloc[0]
        first_buys[w] = (first_row["timestamp"], float(first_row["price"]))

    if len(first_buys) >= 2:
        ordered = sorted(first_buys.items(), key=lambda kv: kv[1][0])
        (_, (earliest_ts, earliest_px)) = ordered[0]
        (_, (latest_ts, latest_px)) = ordered[-1]
        span_hours = (latest_ts - earliest_ts).total_seconds() / 3600.0
        price_chase = latest_px - earliest_px
    else:
        span_hours = 0.0
        price_chase = 0.0

    # -- pairwise Jaccard over each buyer's full fetched-history conditionIds --
    cond_sets: dict[str, set] = {}
    for w in buyers:
        df = trades_by_wallet.get(w)
        cond_sets[w] = set(df["conditionId"].astype(str)) if df is not None and not df.empty else set()

    wallets = list(buyers)
    sims = []
    for a, b in combinations(wallets, 2):
        sa, sb = cond_sets[a], cond_sets[b]
        union = sa | sb
        sims.append(len(sa & sb) / len(union) if union else 0.0)
    pairwise_jaccard = float(sum(sims) / len(sims)) if sims else 0.0

    return {
        "first_buy_span_hours": round(span_hours, 4),
        "price_chase": round(price_chase, 4),
        "pairwise_jaccard": round(pairwise_jaccard, 4),
    }


def classify_independence(features: dict) -> tuple[int, str]:
    """Score the three pre-registered component tests and label the class.

    Each component is a simple pass/fail against a frozen threshold:
      temporal:    first_buy_span_hours >= 24.0
      price:       price_chase          <= 0.05
      co-movement: pairwise_jaccard     <= 0.20

    Returns (indep_score, indep_class): indep_score is the count of components
    passed (0-3); indep_class is "independent" if indep_score >= 2 else "cascade".
    """
    temporal_pass = features["first_buy_span_hours"] >= TEMPORAL_SPAN_HOURS
    price_pass = features["price_chase"] <= PRICE_CHASE_MAX
    comovement_pass = features["pairwise_jaccard"] <= JACCARD_MAX
    indep_score = int(temporal_pass) + int(price_pass) + int(comovement_pass)
    indep_class = "independent" if indep_score >= 2 else "cascade"
    return indep_score, indep_class


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
    return df[_COLS]


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
) -> pd.DataFrame:
    """One hourly cycle: scan consensus candidates exactly like the incumbent,
    annotate each NEW entry with its independence class, append, refresh marks,
    resolve, and apply consensus-exit -- identical mechanics for both classes.

    Parameters
    ----------
    bet_fraction : float
        Flat bankroll fraction per position. No Kelly scaling by design: the
        experiment is the independent-vs-cascade split, so sizing must not
        differ by class or by conviction (that would confound the comparison).
    min_buyers : int
        Distinct smart wallets required to enter (same consensus gate as the
        incumbent).
    window_days : int
        Recent-flow window for both the consensus gate and the independence
        features.
    consensus_exit : bool
        If True, exit positions where smart-wallet net flow has flipped
        bearish (more distinct sellers than buyers in the trailing window) --
        same semantics as the incumbent.
    flip_threshold : int
        Sellers must exceed buyers by at least this many to trigger consensus
        exit.
    per_window : int
        Number of wallets to fetch from the leaderboard for the smart pool.
    state_file : Path
        Ledger CSV to read/write. Defaults to STATE_FILE
        (``smart_flow_indep_positions.csv``), kept separate from every other
        smart-flow ledger so this forward record is never contaminated.
    """
    ledger = load_ledger(state_file)
    today = date.today().isoformat()
    held = set(ledger["token"].tolist()) if not ledger.empty else set()

    # -- 1. build shared flow index once; reused for entries, independence
    #       features, and consensus-exit checks (never fetch trades twice) --
    pool = smart_pool(per_window)
    trades_by_wallet = _fetch_pool_trades(pool)
    flow_index = _flow_index_from(trades_by_wallet, window_days)

    # -- 2. new consensus entries, classified independent vs. cascade --
    new_rows = []
    for c in scan_smart_flow_entries(window_days=window_days, min_buyers=min_buyers,
                                     _flow_index=flow_index):
        if c["token"] in held:
            continue
        buyers = flow_index.get(c["token"], {}).get("buyers", set())
        feats = independence_features(c["token"], buyers, trades_by_wallet, window_days)
        score, cls = classify_independence(feats)
        n = c["n_smart_buyers"]
        logger.info("NEW long [%s score=%d]: %s (%d buyers) mid %.3f ask %.3f",
                    cls, score, (c["question"] or "")[:40], n, c["entry_mid"], c["entry_ask"])
        new_rows.append({
            "token": c["token"], "condition_id": c["condition_id"],
            "question": c["question"], "domain": c["domain"],
            "end_date": c.get("end_date", ""), "entry_date": today,
            "n_smart_buyers": str(n),
            "first_buy_span_hours": str(feats["first_buy_span_hours"]),
            "price_chase": str(feats["price_chase"]),
            "pairwise_jaccard": str(feats["pairwise_jaccard"]),
            "indep_score": str(score), "indep_class": cls,
            "bet_fraction": str(bet_fraction),
            "entry_mid": str(c["entry_mid"]), "entry_ask": str(c["entry_ask"]),
            "spread": str(c["spread"]), "current_price": str(c["entry_mid"]),
            "status": "open", "exit_date": "", "outcome": "", "pnl": "",
        })
    if new_rows:
        ledger = pd.concat([ledger, pd.DataFrame(new_rows)], ignore_index=True)

    # -- 3. refresh + resolve + consensus-exit open positions (both classes,
    #       identical mechanics -- the incumbent's update_ledger, minus Kelly) --
    for i, row in ledger[ledger["status"] == "open"].iterrows():
        entry_ask = float(row["entry_ask"])
        frac = float(row["bet_fraction"]) if row.get("bet_fraction") else bet_fraction

        # resolution first, from the AUTHORITATIVE CLOB market status (closed +
        # per-token winner) -- catches settled markets whose order book is gone.
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
            continue  # neither resolution nor a fresh mark -- leave untouched this run

        # price-heuristic fallback only when the API confirms the market is
        # closed but couldn't produce a clean per-token label.
        if outcome is None and market_closed and mid is not None:
            outcome = 1.0 if mid >= 0.99 else (0.0 if mid <= 0.01 else None)
        if outcome is not None:
            pnl = (outcome / entry_ask - 1.0) * frac
            ledger.at[i, "status"]        = "won" if outcome == 1.0 else "lost"
            ledger.at[i, "exit_date"]     = today
            ledger.at[i, "outcome"]       = str(outcome)
            ledger.at[i, "pnl"]          = str(round(pnl, 4))
            ledger.at[i, "current_price"] = str(outcome)
            logger.info("RESOLVED [%s] %s %s pnl=%.4f", row.get("indep_class", ""),
                        (row["question"] or "")[:40], ledger.at[i, "status"], pnl)
            continue

        # consensus exit: signal reversal (same semantics as the incumbent)
        if consensus_exit:
            if consensus_reversal_check(row["token"], flow_index, flip_threshold):
                pnl = (mid / entry_ask - 1.0) * frac
                ledger.at[i, "status"]    = "flipped"
                ledger.at[i, "exit_date"] = today
                ledger.at[i, "outcome"]   = ""
                ledger.at[i, "pnl"]      = str(round(pnl, 4))
                logger.info("FLIPPED [%s] %s mid=%.4f pnl=%.4f", row.get("indep_class", ""),
                            (row["question"] or "")[:40], mid, pnl)
                continue

    save_ledger(ledger, state_file)
    return ledger
