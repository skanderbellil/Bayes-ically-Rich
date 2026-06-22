#!/usr/bin/env python3
"""
Polymarket — dip-confirm YES paper-trade tracker (live update)
===============================================================

Two-stage strategy: watch markets at 5–9d out in [0.10,0.50] band;
enter ONLY if they dip ≥10% then recover to ≥95% of watch price.
Backtest: 74-88% win rate vs 40% base (H=5-7d, full 2098-market universe).

Lifecycle: watching → (dip-confirm fires) → open → won/lost
                     → (expires at <2d without firing) → expired

Usage
-----
  python experiments/run_dip_confirm_paper_update.py
  python experiments/run_dip_confirm_paper_update.py --dry-run
"""
import argparse
import logging
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd
from tabulate import tabulate

from posterioralpha.polymarket.dip_confirm_yes import (
    BAND_HI, BAND_LO, CONFIRM_MIN_DAYS, DIP_THRESH,
    RECOVERY_FLOOR, WATCH_MAX_DAYS, WATCH_MIN_DAYS,
    update_ledger, load_ledger,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-5s  %(message)s",
                    datefmt="%H:%M:%S")

DEFAULT_LEDGER = "data/paper_trade/dip_confirm_positions.csv"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fraction", type=float, default=0.10)
    ap.add_argument("--min-volume", type=float, default=5_000.0)
    ap.add_argument("--ledger-file", type=str, default=DEFAULT_LEDGER)
    ap.add_argument("--dry-run", action="store_true", help="print watching candidates, no write")
    args = ap.parse_args()

    print("""
╔══════════════════════════════════════════════════════════╗
║  DIP-CONFIRM YES  ·  paper-trade tracker                  ║
║  watch → dip≥10% → recover → enter → hold to close       ║
╚══════════════════════════════════════════════════════════╝""")
    print(f"  band [{BAND_LO:.2f},{BAND_HI:.2f})  watch {WATCH_MIN_DAYS}–{WATCH_MAX_DAYS}d  "
          f"dip≥{DIP_THRESH:.0%}  recover≥{RECOVERY_FLOOR:.0%}  confirm_min {CONFIRM_MIN_DAYS}d\n")

    if args.dry_run:
        # just show current ledger state without updating
        ledger = load_ledger(Path(args.ledger_file))
        watching = ledger[ledger["status"] == "watching"]
        print(f"  WATCHING ({len(watching)} markets):")
        for _, r in watching.iterrows():
            wp = float(r["watch_price"]); mp = float(r["min_price"]); cp = float(r["current_price"])
            dip_pct = (wp - mp) / wp
            print(f"    {(r['question'] or '')[:50]:50s}  watch={wp:.3f}  min={mp:.3f}(−{dip_pct:.0%})  now={cp:.3f}")
        return

    ledger = update_ledger(
        bet_fraction=args.fraction,
        min_volume=args.min_volume,
        state_file=Path(args.ledger_file),
    )

    watching = ledger[ledger["status"] == "watching"]
    open_pos = ledger[ledger["status"] == "open"]
    closed   = ledger[ledger["status"].isin(["won", "lost"])]
    expired  = ledger[ledger["status"] == "expired"]

    # ── Watching ──────────────────────────────────────────────────────────────
    print(f"\n{'═'*90}")
    print(f"  WATCHING — waiting for dip≥{DIP_THRESH:.0%} then recover  ({len(watching)})")
    print(f"{'═'*90}")
    if watching.empty:
        print("  (none in watch window)")
    else:
        rows = []
        for _, r in watching.iterrows():
            wp  = float(r["watch_price"])
            mp  = float(r["min_price"])
            cp  = float(r["current_price"])
            dip = (wp - mp) / wp
            need = wp * (1 - 0.10)   # must dip to this
            rows.append([(r["question"] or "")[:44], r["watch_date"],
                         f"{wp:.3f}", f"{mp:.3f}(−{dip:.0%})", f"{cp:.3f}",
                         f"{need:.3f}", "✓" if mp <= need else "—"])
        print(tabulate(rows,
                       headers=["question", "watched", "watch_px", "min(dip)", "now", "dip_target", "dipped?"],
                       tablefmt="rounded_grid"))

    # ── Open ──────────────────────────────────────────────────────────────────
    print(f"\n{'═'*90}")
    print(f"  OPEN POSITIONS  ({len(open_pos)})")
    print(f"{'═'*90}")
    if open_pos.empty:
        print("  (no confirmed entries yet)")
    else:
        rows = []
        for _, r in open_pos.iterrows():
            ask = float(r["entry_ask"]); cp = float(r["current_price"])
            rows.append([(r["question"] or "")[:44], r["entry_date"],
                         r.get("days_at_entry", "—"),
                         f"{float(r['watch_price']):.3f}",
                         f"{ask:.3f}", f"{cp:.3f}", f"{cp - ask:+.3f}"])
        print(tabulate(rows,
                       headers=["question", "entered", "d2res", "watch_px", "ask", "now", "drift"],
                       tablefmt="rounded_grid"))

    # ── Resolved ──────────────────────────────────────────────────────────────
    print(f"\n{'═'*90}")
    print(f"  RESOLVED  ({len(closed)})   EXPIRED  ({len(expired)})")
    print(f"{'═'*90}")
    if closed.empty:
        print("  (none resolved yet)")
    else:
        rows = []
        for _, r in closed.iterrows():
            rows.append([(r["question"] or "")[:44], r["entry_date"], r["exit_date"],
                         f"{float(r['watch_price']):.3f}",
                         f"{float(r['entry_ask']):.3f}", r["status"].upper(),
                         f"{float(r['pnl']):+.4f}" if r["pnl"] else "—"])
        print(tabulate(rows,
                       headers=["question", "entered", "exited", "watch_px", "ask", "result", "PnL"],
                       tablefmt="rounded_grid"))
        won  = (closed["status"] == "won").sum()
        lost = (closed["status"] == "lost").sum()
        live_wr = won / (won + lost) if (won + lost) else float("nan")
        pnl_vals = pd.to_numeric(closed["pnl"], errors="coerce").dropna()
        print(f"\n  LIVE: {won}W / {lost}L  (win {live_wr*100:.0f}%)  "
              f"additive PnL: {pnl_vals.sum():+.4f}")
        print(f"  BACKTEST benchmark: 74% win rate (H=5d, dip≥10%, n=27)")

    print(f"\n  Ledger → {args.ledger_file}  ({len(ledger)} rows)")
    print("✓  Dip-confirm update complete.")


if __name__ == "__main__":
    main()
