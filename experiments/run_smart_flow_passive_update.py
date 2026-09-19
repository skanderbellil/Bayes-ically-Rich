#!/usr/bin/env python3
"""
Polymarket — passively entered smart-flow consensus (hourly update)
===========================================================================

Pre-registered forward experiment (see ``docs/polymarket/SMART_FLOW_PASSIVE.md``):
the incumbent smart-flow consensus signal is worth roughly +0.9c/$1 measured at the
mid but loses 0.7c/$1 as actually traded, because it lifts the offer and pays a ~1.7c
half-spread to do it. The loss IS the spread. This ledger changes only how the
position is acquired -- a 2c half-spread cap and a resting limit at the mid instead of
crossing -- and records ``unfilled`` as a first-class outcome, because the open
question is whether the mid-edge survives the adverse selection a passive bid invites.

Successor to ``run_smart_flow_indep_update.py``, whose discriminator was retired on
2026-09-19 when both its pre-registered kill criteria fired.

Usage
-----
  python experiments/run_smart_flow_passive_update.py [--fraction 0.10] [--min-buyers 3]
  python experiments/run_smart_flow_passive_update.py --dry-run
"""
import argparse
import logging

import _bootstrap  # noqa: F401
from tabulate import tabulate

from posterioralpha.polymarket.smartflow_passive import (
    MAX_HALF_SPREAD,
    MIN_FILL_RATE,
    MIN_RESOLVED,
    MIN_WORKED,
    STATE_FILE,
    WORK_HOURS,
    fill_rate,
    kill_check,
    update_ledger,
)
from posterioralpha.polymarket.smartflow_papertrade import (
    _build_flow_index,
    scan_smart_flow_entries,
    smart_pool,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-5s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fraction",   type=float, default=0.10, help="flat bankroll fraction per position")
    ap.add_argument("--min-buyers", type=int,   default=3,    help="distinct smart buyers to enter")
    ap.add_argument("--window",     type=int,   default=7,    help="recent-flow window (days)")
    ap.add_argument("--consensus-exit", action="store_true",
                    help="exit when smart-wallet net flow flips bearish (sellers > buyers)")
    ap.add_argument("--flip-threshold", type=int, default=1, metavar="N",
                    help="sellers must exceed buyers by at least N to trigger consensus exit")
    ap.add_argument("--dry-run", action="store_true", help="scan and print, no write")
    args = ap.parse_args()

    print(f"""
╔═══════════════════════════════════════════════════════════╗
║  PASSIVE SMART-FLOW CONSENSUS  ·  hourly                  ║
║  same consensus gate as Smart Flow, but: half-spread cap  ║
║  {MAX_HALF_SPREAD:.2f}, resting limit at the mid, {WORK_HOURS:.0f}h working, no chasing  ║
╚═══════════════════════════════════════════════════════════╝""")

    if args.dry_run:
        flow = _build_flow_index(smart_pool(), args.window)
        cands = scan_smart_flow_entries(window_days=args.window, min_buyers=args.min_buyers,
                                        _flow_index=flow)
        kept = [c for c in cands if float(c["spread"]) <= MAX_HALF_SPREAD]
        logger.info("%d qualifying consensus tokens; %d clear the %.2f half-spread cap",
                    len(cands), len(kept), MAX_HALF_SPREAD)
        for c in cands:
            hs = float(c["spread"])
            flag = "post " if hs <= MAX_HALF_SPREAD else "SKIP "
            print(f"  {flag}[{c['domain'][:10]:10s}] {(c['question'] or '')[:40]:40s}  "
                  f"buyers={c['n_smart_buyers']:2d}  mid {c['entry_mid']:.3f}  "
                  f"ask {c['entry_ask']:.3f}  half-spread {hs:.4f}")
        return

    ledger = update_ledger(
        bet_fraction=args.fraction,
        min_buyers=args.min_buyers,
        window_days=args.window,
        consensus_exit=args.consensus_exit,
        flip_threshold=args.flip_threshold,
        state_file=STATE_FILE,
    )

    working  = ledger[ledger["status"] == "working"]
    unfilled = ledger[ledger["status"] == "unfilled"]
    open_pos = ledger[ledger["status"] == "open"]
    closed   = ledger[ledger["status"].isin(["won", "lost"])]

    print(f"\n{'='*82}")
    print(f"  WORKING ORDERS  ({len(working)})  -- resting bids, not yet filled")
    print(f"{'='*82}")
    if working.empty:
        print("  (none)")
    else:
        print(tabulate(
            [[(r["question"] or "")[:36], r["limit_px"], r["scan_ask"], r["current_price"],
              r["work_deadline"][:16]] for _, r in working.iterrows()],
            headers=["question", "limit", "ask@scan", "mid now", "expires"],
            tablefmt="rounded_grid"))

    print(f"\n{'='*82}")
    print(f"  FILLED, OPEN  ({len(open_pos)})")
    print(f"{'='*82}")
    if open_pos.empty:
        print("  (none)")
    else:
        print(tabulate(
            [[(r["question"] or "")[:36], r["fill_px"], r["scan_ask"], r["current_price"],
              f"{float(r['current_price'])-float(r['fill_px']):+.3f}"]
             for _, r in open_pos.iterrows()],
            headers=["question", "filled at", "ask@scan", "now", "drift"],
            tablefmt="rounded_grid"))

    fr = fill_rate(ledger)
    print(f"\n{'='*82}")
    print("  EXECUTION  (the whole point of this experiment)")
    print(f"{'='*82}")
    saved = None
    if not closed.empty or not open_pos.empty:
        got = ledger[ledger["status"].isin(["open", "won", "lost", "flipped"])]
        px = got["fill_px"].astype(float)
        ask = got["scan_ask"].astype(float)
        saved = float((ask - px).mean())
    print(tabulate([
        ["terminal orders (filled + expired)", fr["worked"]],
        ["filled", fr["filled"]],
        ["expired unfilled", fr["unfilled"]],
        ["fill rate", f"{fr['rate']*100:.1f}%" if fr["worked"] else "--"],
        [f"fill-rate floor (kill below, after {MIN_WORKED} orders)", f"{MIN_FILL_RATE*100:.0f}%"],
        ["mean spread saved vs. crossing", f"{saved:+.4f}" if saved is not None else "--"],
    ], headers=["metric", "value"], tablefmt="rounded_grid"))

    print(f"\n{'='*82}")
    print(f"  RESOLVED  ({len(closed)})   ·   unfilled to date: {len(unfilled)}")
    print(f"{'='*82}")
    k = kill_check(ledger)
    print(tabulate([
        ["resolved fills", f"{k['n_resolved']} / {MIN_RESOLVED} needed"],
        ["settlement-day events", k["events"]],
        ["edge/$1 at fill", f"{k['edge']:+.5f}" if k["edge"] == k["edge"] else "--"],
        ["event-clustered t", f"{k['t']:+.2f}" if k["t"] == k["t"] else "--"],
        ["verdict", k["verdict"]],
    ], headers=["kill criterion", "value"], tablefmt="rounded_grid"))
    if k["verdict"] == "RETIRED":
        print("\n  !! A pre-registered kill criterion has fired. Freeze this ledger:")
        print("     stop opening, let the rest resolve, and do NOT retune in place.")

    print(f"\n  Ledger -> {STATE_FILE}  ({len(ledger)} rows)")
    print("✓  Update complete.")


if __name__ == "__main__":
    main()
