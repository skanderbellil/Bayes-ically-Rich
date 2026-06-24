#!/usr/bin/env python3
"""
Mid-priced-YES — full-universe capital backtest (1% of capital per trade)
========================================================================

Backtests the mid-price band strategy ("buy an outcome token priced in [lo,hi)
some days before resolution, hold to settlement") across **every resolved market
we can pull**, as a real capital-constrained account staking **1% of equity per
trade**.

Two deliberate design choices the request called for:

* **All possible outcomes, not just the Yes leg.** Each market's *both* outcome
  tokens are candidates, so a cheap "No" is tradeable just like a cheap "Yes".
* **Never bet both sides.** For a binary market the band [0.10,0.50) already
  admits at most one leg (the other sits in (0.50,0.90]), but we still enforce
  **one position per market (condition_id)** — if both legs ever qualify we keep
  the lower-priced one. (Caveat: mutually-exclusive legs of the same multi-
  outcome neg-risk *event* are deduped per market, not per event.)

Data: daily marks by default (fast, reuses the token cache); pass
``--fidelity 60`` for hourly intraday (slower, separate cache). Use
``--collect-only`` to just warm the cache (for a background data-collection job)
without running the sim.

Usage
-----
  python experiments/run_midprice_full_backtest.py                      # daily, broad
  python experiments/run_midprice_full_backtest.py --n-markets 2000
  python experiments/run_midprice_full_backtest.py --fidelity 60        # intraday
  python experiments/run_midprice_full_backtest.py --collect-only       # warm cache
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from posterioralpha.polymarket.fetch import (
    GAMMA_URL,
    _get,
    fetch_token_history,
    fetch_token_history_raw,
)
from posterioralpha.polymarket.paths import RESULTS_DIR, ensure_dirs

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-5s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Universe: every outcome token of every resolved market, with its settlement
# ---------------------------------------------------------------------------

def fetch_resolved_tokens(n_markets: int, min_volume: float, page_size: int = 100) -> pd.DataFrame:
    """Per-outcome-token rows for resolved markets: token, settled outcome, market id."""
    rows, offset, seen = [], 0, 0
    while seen < n_markets:
        batch = _get(GAMMA_URL, {"limit": page_size, "offset": offset, "closed": "true",
                                 "order": "volumeNum", "ascending": "false"})
        if not batch:
            break
        for m in batch:
            seen += 1
            try:
                vol = float(m.get("volumeNum") or 0.0)
            except (TypeError, ValueError):
                continue
            if vol < min_volume:
                continue
            try:
                toks = json.loads(m.get("clobTokenIds") or "[]")
                prices = [float(x) for x in json.loads(m.get("outcomePrices") or "[]")]
                labels = json.loads(m.get("outcomes") or "[]")
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if len(toks) != len(prices) or not toks:
                continue
            for i, (tok, settled) in enumerate(zip(toks, prices)):
                # keep only cleanly-settled legs (0/1)
                oc = 1.0 if settled >= 0.98 else (0.0 if settled <= 0.02 else None)
                if oc is None:
                    continue
                # leg label for the yes/no decomposition: use the market's own
                # outcome name when present, else fall back to index (0=Yes, 1=No).
                leg = (str(labels[i]) if i < len(labels) else
                       ("Yes" if i == 0 else "No")).strip().lower()
                rows.append({
                    "token": str(tok), "outcome": oc, "leg": leg,
                    "condition_id": str(m.get("conditionId") or ""),
                    "neg_risk": bool(m.get("negRisk")),
                    "end_date": m.get("endDate"), "volume": vol,
                    "question": m.get("question") or "",
                })
            if seen >= n_markets:
                break
        if len(batch) < page_size:
            break
        offset += page_size
    df = pd.DataFrame(rows)
    logger.info("fetch_resolved_tokens: %d markets scanned → %d settled outcome tokens",
                seen, len(df))
    return df


# ---------------------------------------------------------------------------
# Entry detection: first day in the window where the token is in-band
# ---------------------------------------------------------------------------

def find_entry(token: str, lo: float, hi: float, days_min: int, days_max: int,
               haircut: float, fidelity: int, use_cache: bool):
    """Return (entry_ts, entry_price) for the earliest in-band day in the window, else None."""
    if fidelity >= 1440:
        s = fetch_token_history(token, fidelity_minutes=1440, use_cache=use_cache)
    else:
        s = fetch_token_history_raw(token, fidelity_minutes=fidelity, use_cache=use_cache)
    if s is None or len(s) < 3:
        return None
    s = s[~s.index.duplicated(keep="last")].sort_index()
    t_res = s.index.max()
    for d in range(days_max, days_min - 1, -1):          # earliest first
        sub = s[s.index <= t_res - pd.Timedelta(days=d)]
        if sub.empty:
            continue
        p0 = float(sub.iloc[-1])
        if lo <= p0 < hi:
            return sub.index[-1], min(p0 + haircut, 0.99)
    return None


# ---------------------------------------------------------------------------
# Capital simulation — 1% of equity per trade, hold to resolution
# ---------------------------------------------------------------------------

def simulate(entries: pd.DataFrame, capital: float, stake_frac: float) -> dict:
    """Chronological cash-constrained sim. entries: entry_date, entry_price, res_date, outcome."""
    ev = []
    for i, r in entries.iterrows():
        ev.append((r["entry_date"], 1, i))
        ev.append((r["res_date"], 0, i))
    ev.sort(key=lambda e: (e[0], e[1]))                  # exits before entries same ts
    cash, taken, skipped, realized = capital, 0, 0, 0.0
    open_pos: dict = {}
    eq_curve = []
    for ts, kind, i in ev:
        if kind == 0:
            pos = open_pos.pop(i, None)
            if pos:
                payout = pos["shares"] * float(entries.at[i, "outcome"])
                cash += payout
                realized += payout - pos["cost"]
        else:
            equity = cash + sum(v["cost"] for v in open_pos.values())
            stake = stake_frac * equity
            px = float(entries.at[i, "entry_price"])
            if stake > 0 and cash >= stake and px > 0:
                open_pos[i] = {"shares": stake / px, "cost": stake}
                cash -= stake
                taken += 1
            else:
                skipped += 1
        eq_curve.append((ts, cash + sum(v["cost"] for v in open_pos.values())))
    # any unresolved left (shouldn't happen on resolved universe) → mark at cost
    final_equity = cash + sum(v["cost"] for v in open_pos.values())
    eq = pd.Series([e for _, e in eq_curve], index=[t for t, _ in eq_curve]) if eq_curve \
        else pd.Series([capital])
    dd = float(((eq - eq.cummax()) / eq.cummax()).min()) if len(eq) else 0.0
    return {"final_equity": final_equity, "total_return": final_equity / capital - 1.0,
            "realized": realized, "max_dd": dd, "taken": taken, "skipped": skipped,
            "equity_curve": eq}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-markets", type=int, default=1200, help="resolved markets to scan")
    ap.add_argument("--min-volume", type=float, default=5_000.0)
    ap.add_argument("--lo", type=float, default=0.10)
    ap.add_argument("--hi", type=float, default=0.50)
    ap.add_argument("--days-min", type=int, default=2)
    ap.add_argument("--days-max", type=int, default=10)
    ap.add_argument("--haircut", type=float, default=0.01, help="entry spread added to mid (¢)")
    ap.add_argument("--capital", type=float, default=1000.0)
    ap.add_argument("--stake", type=float, default=0.01, help="fraction of equity per trade")
    ap.add_argument("--fidelity", type=int, default=1440, help="1440=daily, 60=hourly intraday")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--collect-only", action="store_true", help="warm price cache, skip sim")
    ap.add_argument("--side", choices=["both", "yes", "no"], default="both",
                    help="which outcome leg to trade: 'both' (any in-band leg, the "
                         "default), 'yes' (Yes/first leg only), 'no' (No/second leg only)")
    ap.add_argument("--sleep", type=float, default=0.05)
    args = ap.parse_args()
    ensure_dirs()
    use_cache = not args.no_cache

    uni = fetch_resolved_tokens(args.n_markets, args.min_volume)
    if uni.empty:
        raise SystemExit("No resolved markets returned.")

    # yes/no decomposition: restrict the candidate legs before entry detection.
    if args.side != "both":
        before = len(uni)
        uni = uni[uni["leg"] == args.side].reset_index(drop=True)
        logger.info("side=%s filter: %d → %d candidate outcome tokens", args.side, before, len(uni))
        if uni.empty:
            raise SystemExit(f"No '{args.side}' legs in the resolved universe.")

    rows = []
    for k, r in enumerate(uni.itertuples(index=False), 1):
        res = find_entry(r.token, args.lo, args.hi, args.days_min, args.days_max,
                         args.haircut, args.fidelity, use_cache)
        if k % 100 == 0:
            logger.info("  scanned %d/%d tokens (%d in-band)", k, len(uni), len(rows))
        if args.sleep:
            time.sleep(args.sleep)
        if res is None:
            continue
        entry_ts, entry_px = res
        rows.append({
            "token": r.token, "condition_id": r.condition_id, "question": r.question,
            "leg": r.leg, "entry_date": entry_ts, "entry_price": entry_px,
            "res_date": pd.Timestamp(r.end_date).tz_localize(None) if r.end_date else entry_ts,
            "outcome": r.outcome, "volume": r.volume,
        })

    if args.collect_only:
        logger.info("collect-only: cache warmed for %d tokens (%d in-band).", len(uni), len(rows))
        return

    cand = pd.DataFrame(rows)
    if cand.empty:
        raise SystemExit("No in-band entries found.")
    # res_date sanity: must be on/after entry
    cand["res_date"] = cand[["res_date", "entry_date"]].max(axis=1)

    # ── both-sides guard: one position per market (lowest entry price wins) ──
    before = len(cand)
    cand = (cand.sort_values("entry_price")
                .drop_duplicates(subset="condition_id", keep="first")
                .reset_index(drop=True))
    logger.info("both-sides guard: %d → %d entries (dropped %d duplicate-market legs)",
                before, len(cand), before - len(cand))

    sim = simulate(cand, args.capital, args.stake)
    r = (cand["outcome"] / cand["entry_price"] - 1.0)
    win = float((cand["outcome"] > 0).mean())
    edge = cand["outcome"] - cand["entry_price"]
    tstat = float(edge.mean() / (edge.std(ddof=1) / math.sqrt(len(edge)))) if len(edge) > 1 else float("nan")

    # persist (suffix by side so yes/no/both runs don't overwrite each other)
    tag = "" if args.side == "both" else f"_{args.side}"
    out_csv = RESULTS_DIR / f"midprice_full_backtest{tag}_trades.csv"
    cand.assign(trade_return=r).to_csv(out_csv, index=False)
    if len(sim["equity_curve"]) > 1:
        sim["equity_curve"].to_csv(RESULTS_DIR / f"midprice_full_backtest{tag}_equity.csv",
                                   header=["equity"])

    print(f"\n{'='*84}")
    print(f"  MID-PRICED — FULL BACKTEST  ·  side={args.side}  ·  band [{args.lo:.2f},{args.hi:.2f})  ·  "
          f"enter {args.days_min}-{args.days_max}d pre-res  ·  {args.haircut*100:.0f}¢ haircut")
    print(f"  ${args.capital:,.0f} start · {args.stake:.0%} of equity per trade · "
          f"{'daily' if args.fidelity>=1440 else str(args.fidelity)+'-min'} marks")
    print(f"{'='*84}")
    print(f"  Markets scanned (tokens) : {len(uni):,}")
    print(f"  In-band entries          : {before:,}  →  {len(cand):,} after both-sides dedupe")
    print(f"  Trades taken / skipped   : {sim['taken']:,} / {sim['skipped']:,} (capital-constrained)")
    print(f"  Win rate                 : {win*100:.1f}%")
    print(f"  Mean trade return        : {r.mean()*100:+.1f}%   median {r.median()*100:+.1f}%")
    print(f"  Fair-pricing edge        : {edge.mean():+.4f} per $1   t-stat {tstat:+.2f}")
    print(f"  ── Capital result ──")
    print(f"  Final equity             : ${sim['final_equity']:,.0f}   ({sim['total_return']*100:+.1f}%)")
    print(f"  Realized PnL             : ${sim['realized']:+,.0f}")
    print(f"  Max drawdown             : {sim['max_dd']*100:.1f}%")
    print(f"\n  Trades → {out_csv.relative_to(RESULTS_DIR.parent.parent)}")
    print("✓  Backtest complete.")


if __name__ == "__main__":
    main()
