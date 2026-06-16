#!/usr/bin/env python3
"""
Polymarket — macro paper-trade tracker (daily update)
======================================================

Scans live Polymarket for open macro multi-outcome events, logs new
positions, refreshes current prices, and marks resolutions. State is
persisted to ``data/paper_trade/macro_positions.csv`` — committed back to
the repo by the GitHub Actions workflow so the ledger is always up to date.

Design
------
* Run daily via ```.github/workflows/paper_trade.yml``` (GitHub Actions cron).
* From your phone: open ``data/paper_trade/macro_positions.csv`` on GitHub,
  or ask Claude Code "show me the current paper trade state" and it will read
  the file and summarise it.
* To run manually (e.g. to inspect right now): just run this script.

Usage
-----
  python experiments/run_paper_trade_update.py [--fraction 0.10]
"""
import argparse
import logging
import sys

import _bootstrap  # noqa: F401
from posterioralpha.polymarket.papertrade import load_ledger, update_ledger
from tabulate import tabulate

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-5s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fraction", type=float, default=0.10,
                    help="Fraction of bankroll per bet for PnL accounting (default 10%%)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Scan and print without writing the ledger")
    args = ap.parse_args()

    print("""
╔══════════════════════════════════════════════════════╗
║  MACRO PAPER-TRADE TRACKER  ·  daily update          ║
║  strategy: buy field leader, hold to resolution      ║
║  sizing  : fraction of bankroll per position         ║
╚══════════════════════════════════════════════════════╝""")

    if args.dry_run:
        from posterioralpha.polymarket.papertrade import scan_open_macro_events
        events = scan_open_macro_events()
        logger.info("Found %d qualifying open macro multi-outcome events", len(events))
        for ev in events:
            print(f"  [{ev['event_ticker'][:30]}]  leader: {ev['leader_question'][:50]}"
                  f"  price {ev['entry_price']:.3f}  k={ev['n_candidates']}")
        return

    ledger = update_ledger(bet_fraction=args.fraction)

    open_pos = ledger[ledger["status"] == "open"]
    closed   = ledger[ledger["status"].isin(["won", "lost"])]

    # ── open positions ─────────────────────────────────────────────────────
    print(f"\n{'═'*76}")
    print(f"  OPEN POSITIONS  ({len(open_pos)})")
    print(f"{'═'*76}")
    if open_pos.empty:
        print("  (none yet — no qualifying macro multi-outcome fields found)")
    else:
        print(tabulate(
            [[r["event_ticker"][:28], r["event_title"][:30],
              r["entry_date"], f"{float(r['entry_price']):.3f}",
              f"{float(r['current_price']):.3f}",
              f"{float(r['current_price'])-float(r['entry_price']):+.3f}"]
             for _, r in open_pos.iterrows()],
            headers=["event", "title", "entered", "entry p", "now p", "drift"],
            tablefmt="rounded_grid"))

    # ── closed / resolved positions ─────────────────────────────────────────
    print(f"\n{'═'*76}")
    print(f"  RESOLVED POSITIONS  ({len(closed)})")
    print(f"{'═'*76}")
    if closed.empty:
        print("  (none resolved yet)")
    else:
        print(tabulate(
            [[r["event_ticker"][:28], r["entry_date"], r["exit_date"],
              f"{float(r['entry_price']):.3f}", r["status"].upper(),
              f"{float(r['pnl']):+.4f}" if r["pnl"] else "—"]
             for _, r in closed.iterrows()],
            headers=["event", "entered", "exited", "entry p", "result", f"PnL ({args.fraction:.0%} bet)"],
            tablefmt="rounded_grid"))
        won  = (closed["status"] == "won").sum()
        lost = (closed["status"] == "lost").sum()
        pnls = closed["pnl"].replace("", float("nan")).astype(float).dropna()
        print(f"\n  W/L: {won}W / {lost}L   "
              f"cumulative PnL (additive, {args.fraction:.0%} bet): {pnls.sum():+.4f}  "
              f"= {pnls.sum()*100:+.1f}¢ per $1 staked")

    print(f"\n  Ledger → data/paper_trade/macro_positions.csv  ({len(ledger)} total rows)")
    print("✓  Update complete.")


if __name__ == "__main__":
    main()
