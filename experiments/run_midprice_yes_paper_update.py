#!/usr/bin/env python3
"""
Polymarket — mid-priced-YES convergence paper-trade tracker (live update)
=========================================================================

Forward, out-of-sample test of the deep-dive's strongest finding: YES tokens
priced in [0.10, 0.50] about 5 days before resolution resolve YES more often
than priced, so buying and **holding to resolution** is positive-edge. Every
exit rule tested (stops, trailing, decaying-horizon, timed sells) underperformed
hold, so this tracker never exits — it only enters and waits.

Each run logs a paper YES long (at the live CLOB ask, so the spread is paid) in
every currently-open market resolving in ~3–6 days with YES mid in band, marks
open positions, and resolves them. The historical backtest is printed alongside
as the benchmark the live ledger is being measured against.

Usage
-----
  python experiments/run_midprice_yes_paper_update.py
  python experiments/run_midprice_yes_paper_update.py --dry-run
  python experiments/run_midprice_yes_paper_update.py --rebuild-backtest
"""
import argparse
import logging

import _bootstrap  # noqa: F401
import pandas as pd
from tabulate import tabulate

from posterioralpha.polymarket.midprice_yes import (
    BAND_HI,
    BAND_LO,
    DAYS_MAX,
    DAYS_MIN,
    historical_backtest,
    scan_midprice_entries,
    update_ledger,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-5s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fraction", type=float, default=0.10, help="bankroll fraction per position")
    ap.add_argument("--lo", type=float, default=BAND_LO, help="band lower bound")
    ap.add_argument("--hi", type=float, default=BAND_HI, help="band upper bound")
    ap.add_argument("--days-min", type=int, default=DAYS_MIN, help="min days to resolution at entry")
    ap.add_argument("--days-max", type=int, default=DAYS_MAX, help="max days to resolution at entry")
    ap.add_argument("--min-volume", type=float, default=10_000.0, help="market volume screen")
    ap.add_argument("--rebuild-backtest", action="store_true", help="recompute the baked-in benchmark from the API")
    ap.add_argument("--dry-run", action="store_true", help="scan and print, no write")
    args = ap.parse_args()

    print("""
╔══════════════════════════════════════════════════════════╗
║  MID-PRICED-YES CONVERGENCE  ·  paper-trade tracker       ║
║  buy YES in [lo,hi) ~5d to resolution → hold to close     ║
╚══════════════════════════════════════════════════════════╝""")

    # ── baked-in backtest benchmark ───────────────────────────────────────
    bench = historical_backtest(lo=args.lo, hi=args.hi,
                                use_cache=not args.rebuild_backtest)
    print(f"\n  BAKED-IN BACKTEST BENCHMARK  (band [{args.lo:.2f},{args.hi:.2f}), ~5d, hold to resolution)")
    if bench.get("n"):
        print(f"    historical: n={int(bench['n'])}  win {float(bench['win_rate'])*100:.0f}%  "
              f"mean {float(bench['mean_ret'])*100:+.1f}%  median {float(bench['median_ret'])*100:+.1f}%  "
              f"Sharpe {float(bench['sharpe']):.2f}")
        print(f"    → live ledger below is the out-of-sample test of this edge.")
    else:
        print("    (benchmark unavailable)")

    if args.dry_run:
        cands = scan_midprice_entries(args.days_min, args.days_max, args.lo, args.hi,
                                      min_volume=args.min_volume)
        print(f"\n  DRY RUN — {len(cands)} qualifying open markets:")
        for c in cands:
            print(f"    {(c['question'] or '')[:50]:50s}  {c['days_to_res']:.1f}d  "
                  f"mid {c['entry_mid']:.3f}  ask {c['entry_ask']:.3f}")
        return

    ledger = update_ledger(bet_fraction=args.fraction, days_min=args.days_min,
                           days_max=args.days_max, lo=args.lo, hi=args.hi,
                           min_volume=args.min_volume)
    open_pos = ledger[ledger["status"] == "open"]
    closed = ledger[ledger["status"].isin(["won", "lost"])]

    print(f"\n{'═'*86}")
    print(f"  OPEN POSITIONS  ({len(open_pos)})")
    print(f"{'═'*86}")
    if open_pos.empty:
        print("  (none yet — no open markets in band ~5d to resolution)")
    else:
        print(tabulate(
            [[(r["question"] or "")[:44], r["entry_date"], r["days_to_res"],
              f"{float(r['entry_ask']):.3f}", f"{float(r['current_price']):.3f}",
              f"{float(r['current_price'])-float(r['entry_ask']):+.3f}"]
             for _, r in open_pos.iterrows()],
            headers=["question", "entered", "d2res@entry", "ask", "now", "drift"],
            tablefmt="rounded_grid"))

    print(f"\n{'═'*86}")
    print(f"  RESOLVED POSITIONS  ({len(closed)})")
    print(f"{'═'*86}")
    if closed.empty:
        print("  (none resolved yet)")
    else:
        print(tabulate(
            [[(r["question"] or "")[:44], r["entry_date"], r["exit_date"],
              f"{float(r['entry_ask']):.3f}", r["status"].upper(),
              f"{float(r['pnl']):+.4f}" if r["pnl"] else "—"]
             for _, r in closed.iterrows()],
            headers=["question", "entered", "exited", "ask", "result", "PnL"],
            tablefmt="rounded_grid"))
        won = (closed["status"] == "won").sum()
        lost = (closed["status"] == "lost").sum()
        pnls = closed["pnl"].replace("", float("nan")).astype(float).dropna()
        live_win = won / (won + lost) if (won + lost) else float("nan")
        print(f"\n  LIVE: {won}W / {lost}L  (win {live_win*100:.0f}%)   "
              f"cumulative PnL (additive): {pnls.sum():+.4f}")
        if bench.get("n"):
            print(f"  vs BENCHMARK win {float(bench['win_rate'])*100:.0f}%  "
                  f"mean {float(bench['mean_ret'])*100:+.1f}%/trade")

    print(f"\n  Ledger → data/paper_trade/midprice_yes_positions.csv  ({len(ledger)} rows)")
    print("✓  Update complete.")


if __name__ == "__main__":
    main()
