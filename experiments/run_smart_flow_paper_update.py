#!/usr/bin/env python3
"""
Polymarket — smart-money consensus paper-trade tracker (daily update)
=====================================================================

The live, forward, out-of-sample test of the trader-behaviour signals
(`ORDER_FLOW` / `PAYUP_FOLLOW`). Each day it logs a paper LONG in every
currently-open outcome token that several non-market-maker leaderboard wallets
have recently bought (consensus breadth), **entering at the live CLOB ask** so the
real bid-ask spread — the thing that made the backtest execution-bound — is
captured for real. Positions are held to resolution and marked daily.

State persists to ``data/paper_trade/smart_flow_positions.csv``, committed back by
the GitHub Actions workflow so the ledger is always current. Run manually any time
to inspect, or ``--dry-run`` to scan without writing.

Usage
-----
  python experiments/run_smart_flow_paper_update.py [--fraction 0.10] [--min-buyers 3]
  python experiments/run_smart_flow_paper_update.py --dry-run
"""
import argparse
import logging

import _bootstrap  # noqa: F401
from tabulate import tabulate

from posterioralpha.polymarket.smartflow_papertrade import (
    scan_smart_flow_entries,
    update_ledger,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-5s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fraction",   type=float, default=0.10, help="bankroll fraction per position")
    ap.add_argument("--min-buyers", type=int,   default=3,    help="distinct smart buyers to enter")
    ap.add_argument("--window",     type=int,   default=7,    help="recent-flow window (days)")
    ap.add_argument("--dry-run",    action="store_true",      help="scan and print, no write")
    args = ap.parse_args()

    print("""
╔══════════════════════════════════════════════════════════╗
║  SMART-MONEY CONSENSUS PAPER-TRADE  ·  daily update       ║
║  long where ≥N non-MM leaders recently bought (open mkts) ║
║  enter at the live ask → captures the real spread         ║
╚══════════════════════════════════════════════════════════╝""")

    if args.dry_run:
        cands = scan_smart_flow_entries(window_days=args.window, min_buyers=args.min_buyers)
        logger.info("Found %d qualifying open consensus tokens", len(cands))
        for c in cands:
            print(f"  [{c['domain'][:10]:10s}] {(c['question'] or '')[:46]:46s}  "
                  f"buyers={c['n_smart_buyers']}  mid {c['entry_mid']:.3f}  "
                  f"ask {c['entry_ask']:.3f}  spread {c['spread']:+.3f}")
        return

    ledger = update_ledger(bet_fraction=args.fraction, min_buyers=args.min_buyers,
                           window_days=args.window)
    open_pos = ledger[ledger["status"] == "open"]
    closed   = ledger[ledger["status"].isin(["won", "lost"])]

    print(f"\n{'═'*82}")
    print(f"  OPEN POSITIONS  ({len(open_pos)})")
    print(f"{'═'*82}")
    if open_pos.empty:
        print("  (none yet — no qualifying smart-consensus tokens found)")
    else:
        print(tabulate(
            [[(r["question"] or "")[:40], r["domain"][:10], r["entry_date"],
              r["n_smart_buyers"], f"{float(r['entry_ask']):.3f}",
              f"{float(r['spread']):+.3f}", f"{float(r['current_price']):.3f}",
              f"{float(r['current_price'])-float(r['entry_ask']):+.3f}"]
             for _, r in open_pos.iterrows()],
            headers=["question", "domain", "entered", "buyers", "ask", "spread", "now", "drift"],
            tablefmt="rounded_grid"))

    print(f"\n{'═'*82}")
    print(f"  RESOLVED POSITIONS  ({len(closed)})")
    print(f"{'═'*82}")
    if closed.empty:
        print("  (none resolved yet)")
    else:
        print(tabulate(
            [[(r["question"] or "")[:40], r["entry_date"], r["exit_date"],
              f"{float(r['entry_ask']):.3f}", r["status"].upper(),
              f"{float(r['pnl']):+.4f}" if r["pnl"] else "—"]
             for _, r in closed.iterrows()],
            headers=["question", "entered", "exited", "ask", "result", f"PnL ({args.fraction:.0%})"],
            tablefmt="rounded_grid"))
        won  = (closed["status"] == "won").sum()
        lost = (closed["status"] == "lost").sum()
        pnls = closed["pnl"].replace("", float("nan")).astype(float).dropna()
        avg_spread = open_pos["spread"].replace("", float("nan")).astype(float).mean() if not open_pos.empty else float("nan")
        print(f"\n  W/L: {won}W / {lost}L   "
              f"cumulative PnL (additive, {args.fraction:.0%} bet): {pnls.sum():+.4f}")

    print(f"\n  Ledger → data/paper_trade/smart_flow_positions.csv  ({len(ledger)} rows)")
    print("✓  Update complete.")


if __name__ == "__main__":
    main()
