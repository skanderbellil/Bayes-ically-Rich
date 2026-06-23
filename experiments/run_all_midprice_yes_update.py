#!/usr/bin/env python3
"""
Unified mid-priced-YES paper-trade update — all bands in one API scan.

Runs a single broad market scan then routes qualifying markets to each
band-specific ledger, avoiding redundant API calls when multiple bands
are tracked simultaneously.

Bands tracked
-------------
  [10, 50)  → data/paper_trade/midprice_yes_positions.csv         (primary)
  [10, 20)  → data/paper_trade/midprice_yes_10_20_positions.csv
  [10, 30)  → data/paper_trade/midprice_yes_10_30_positions.csv
  [20, 40)  → data/paper_trade/midprice_yes_20_40_positions.csv

Usage
-----
  python experiments/run_all_midprice_yes_update.py
  python experiments/run_all_midprice_yes_update.py --dry-run
"""
import argparse
import logging
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from tabulate import tabulate

from posterioralpha.polymarket.midprice_yes import (
    DAYS_MAX, DAYS_MIN,
    scan_midprice_entries,
    update_ledger,
    historical_backtest,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-5s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

DATA = Path("data/paper_trade")

# (lo, hi, ledger_file, label)
BANDS = [
    (0.10, 0.50, DATA / "midprice_yes_positions.csv",       "YES [10–50%)  PRIMARY"),
    (0.10, 0.20, DATA / "midprice_yes_10_20_positions.csv", "YES [10–20%)"),
    (0.10, 0.30, DATA / "midprice_yes_10_30_positions.csv", "YES [10–30%)"),
    (0.20, 0.40, DATA / "midprice_yes_20_40_positions.csv", "YES [20–40%)"),
]


def _print_ledger(ledger: pd.DataFrame, label: str, fraction: float) -> None:
    open_pos = ledger[ledger["status"] == "open"]
    closed   = ledger[ledger["status"].isin(["won", "lost"])]
    won  = (closed["status"] == "won").sum()
    lost = (closed["status"] == "lost").sum()
    live_wr = won / (won + lost) if (won + lost) else float("nan")
    mtm_vals = pd.to_numeric(open_pos["current_price"], errors="coerce")
    ask_vals = pd.to_numeric(open_pos["entry_ask"], errors="coerce")
    mtm_pct  = ((mtm_vals - ask_vals) / ask_vals).mean() * 100 if not open_pos.empty else 0.0
    print(f"  {label:28s}  open={len(open_pos)}  {won}W/{lost}L  "
          f"win={live_wr*100:.0f}%  MTM {mtm_pct:+.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fraction", type=float, default=0.10)
    ap.add_argument("--days-min", type=int, default=DAYS_MIN)
    ap.add_argument("--days-max", type=int, default=DAYS_MAX)
    ap.add_argument("--min-volume", type=float, default=10_000.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print("""
╔══════════════════════════════════════════════════════════╗
║  MID-PRICED-YES  ·  unified multi-band update             ║
╚══════════════════════════════════════════════════════════╝""")

    # One broad scan covering [0.10, 0.50) — the widest band we track.
    # sub-bands are filtered from this list inside update_ledger.
    logger.info("Scanning open markets [%.2f, %.2f) %d–%dd out …", 0.10, 0.50, args.days_min, args.days_max)
    all_candidates = scan_midprice_entries(
        args.days_min, args.days_max,
        lo=0.10, hi=0.50,
        min_volume=args.min_volume,
    )
    logger.info("Scan found %d qualifying markets", len(all_candidates))

    if args.dry_run:
        print(f"\n  DRY RUN — {len(all_candidates)} markets in [0.10,0.50) window:")
        for c in all_candidates:
            print(f"    {(c['question'] or '')[:50]:50s}  {c['days_to_res']:.1f}d  mid {c['entry_mid']:.3f}")
        return

    print(f"\n{'═'*80}")
    print(f"  BAND SUMMARY")
    print(f"{'═'*80}")

    for lo, hi, ledger_file, label in BANDS:
        ledger = update_ledger(
            bet_fraction=args.fraction,
            days_min=args.days_min,
            days_max=args.days_max,
            lo=lo, hi=hi,
            min_volume=args.min_volume,
            state_file=ledger_file,
            pre_scanned=all_candidates,
        )
        _print_ledger(ledger, label, args.fraction)

    # Detailed output for the primary band only
    lo, hi, ledger_file, label = BANDS[0]
    ledger = pd.read_csv(ledger_file, dtype=str)
    open_pos = ledger[ledger["status"] == "open"]
    closed   = ledger[ledger["status"].isin(["won", "lost"])]

    if not open_pos.empty:
        print(f"\n{'═'*80}")
        print(f"  OPEN — {label}")
        print(f"{'═'*80}")
        print(tabulate(
            [[(r["question"] or "")[:44], r["entry_date"], r["days_to_res"],
              f"{float(r['entry_ask']):.3f}", f"{float(r['current_price']):.3f}",
              f"{float(r['current_price'])-float(r['entry_ask']):+.3f}"]
             for _, r in open_pos.iterrows()],
            headers=["question", "entered", "d2res@entry", "ask", "now", "drift"],
            tablefmt="rounded_grid"))

    if not closed.empty:
        won  = (closed["status"] == "won").sum()
        lost = (closed["status"] == "lost").sum()
        pnl_vals = pd.to_numeric(closed["pnl"], errors="coerce").dropna()
        bankroll = float(np.prod(1.0 + pnl_vals.values)) if len(pnl_vals) else 1.0
        print(f"\n  RESOLVED — {won}W / {lost}L  additive {pnl_vals.sum():+.4f}  "
              f"compounding ×{bankroll:.4f}")

    # Benchmark (primary band [10,50) only — cached)
    bench = historical_backtest(lo=0.10, hi=0.50, use_cache=True)
    if bench.get("n"):
        print(f"\n  BENCHMARK [10,50) ~5d hold: n={int(bench['n'])}  "
              f"win {float(bench['win_rate'])*100:.0f}%  "
              f"mean {float(bench['mean_ret'])*100:+.1f}%  "
              f"Sharpe {float(bench['sharpe']):.2f}")

    print("\n✓  All bands updated.")


if __name__ == "__main__":
    main()
