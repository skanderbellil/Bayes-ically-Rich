#!/usr/bin/env python3
"""
Polymarket — independence-classified smart-flow consensus (hourly update)
===========================================================================

Pre-registered forward experiment (see ``docs/polymarket/SMART_FLOW_INDEPENDENCE.md``):
does splitting smart-flow consensus into "independent discovery" vs. "information
cascade" (Condorcet independence, `docs/knowledge/epistemology/README.md` Lesson 2)
explain why the incumbent smart-flow consensus strategy is forward-losing (-52% MTM,
edge t~0.09)? Each run scans the same consensus candidates the incumbent would take
(``>= min-buyers`` non-market-maker leaderboard wallets recently buying an open
outcome token), classifies each one independent/cascade from three frozen proxies
(buyer-arrival spread, price-chase, buyer-history co-movement), and logs **both**
classes to the **same** ledger at the **same** flat bet size — the experiment is the
split itself, so nothing about entry/exit/sizing may differ by class.

State persists to ``data/paper_trade/smart_flow_indep_positions.csv``, committed back
by the GitHub Actions workflow so the ledger is always current. Run manually any time
to inspect, or ``--dry-run`` to scan without writing.

Usage
-----
  python experiments/run_smart_flow_indep_update.py [--fraction 0.10] [--min-buyers 3]
  python experiments/run_smart_flow_indep_update.py --dry-run
"""
import argparse
import logging

import _bootstrap  # noqa: F401
import pandas as pd
from tabulate import tabulate

from posterioralpha.polymarket.smartflow_independence import (
    STATE_FILE,
    classify_independence,
    independence_features,
    update_ledger,
)
from posterioralpha.polymarket.smartflow_papertrade import (
    scan_smart_flow_entries,
    smart_pool,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-5s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def _class_summary(df: pd.DataFrame) -> list[list]:
    """Per-``indep_class`` rollup: count, open, resolved, win rate, mean pnl."""
    rows = []
    if df.empty:
        return rows
    for cls, g in df.groupby(df["indep_class"].replace("", "(unclassified)")):
        resolved = g[g["status"].isin(["won", "lost"])]
        n_resolved = len(resolved)
        win_rate = (resolved["status"] == "won").mean() if n_resolved else float("nan")
        pnls = pd.to_numeric(g["pnl"], errors="coerce").dropna()
        mean_pnl = pnls.mean() if len(pnls) else float("nan")
        rows.append([
            cls, len(g), (g["status"] == "open").sum(), n_resolved,
            f"{win_rate*100:.0f}%" if n_resolved else "--",
            f"{mean_pnl:+.4f}" if len(pnls) else "--",
        ])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fraction",    type=float, default=0.10,  help="flat bankroll fraction per position (no Kelly)")
    ap.add_argument("--min-buyers", type=int,   default=3,     help="distinct smart buyers to enter")
    ap.add_argument("--window",     type=int,   default=7,     help="recent-flow window (days)")
    ap.add_argument("--consensus-exit", action="store_true",
                    help="exit when smart-wallet net flow flips bearish (sellers > buyers)")
    ap.add_argument("--flip-threshold", type=int, default=1, metavar="N",
                    help="sellers must exceed buyers by at least N to trigger consensus exit (default: 1)")
    ap.add_argument("--dry-run",    action="store_true",       help="scan and print, no write")
    args = ap.parse_args()

    print("""
╔══════════════════════════════════════════════════════════╗
║  INDEPENDENCE-CLASSIFIED SMART-FLOW CONSENSUS  ·  hourly  ║
║  same consensus gate as Smart Flow, split into            ║
║  independent-discovery vs. information-cascade             ║
╚══════════════════════════════════════════════════════════╝""")

    if args.dry_run:
        from posterioralpha.polymarket.smartflow_papertrade import _fetch_pool_trades, _flow_index_from
        pool = smart_pool()
        trades_by_wallet = _fetch_pool_trades(pool)   # fetched once, reused below
        flow_index = _flow_index_from(trades_by_wallet, args.window)
        cands = scan_smart_flow_entries(window_days=args.window, min_buyers=args.min_buyers,
                                        _flow_index=flow_index)
        logger.info("Found %d qualifying open consensus tokens", len(cands))
        for c in cands:
            buyers = flow_index.get(c["token"], {}).get("buyers", set())
            feats = independence_features(c["token"], buyers, trades_by_wallet, args.window)
            score, cls = classify_independence(feats)
            print(f"  [{cls:11s} {score}/3] [{c['domain'][:10]:10s}] {(c['question'] or '')[:40]:40s}  "
                  f"buyers={c['n_smart_buyers']}  span={feats['first_buy_span_hours']:.1f}h  "
                  f"chase={feats['price_chase']:+.3f}  jaccard={feats['pairwise_jaccard']:.2f}  "
                  f"ask {c['entry_ask']:.3f}")
        return

    ledger = update_ledger(
        bet_fraction=args.fraction,
        min_buyers=args.min_buyers,
        window_days=args.window,
        consensus_exit=args.consensus_exit,
        flip_threshold=args.flip_threshold,
        state_file=STATE_FILE,
    )
    open_pos = ledger[ledger["status"] == "open"]
    flipped  = ledger[ledger["status"] == "flipped"]
    closed   = ledger[ledger["status"].isin(["won", "lost"])]

    print(f"\n{'='*82}")
    print(f"  OPEN POSITIONS  ({len(open_pos)})")
    print(f"{'='*82}")
    if open_pos.empty:
        print("  (none yet -- no qualifying smart-consensus tokens found)")
    else:
        print(tabulate(
            [[(r["question"] or "")[:36], r["indep_class"], r["indep_score"],
              r["n_smart_buyers"], f"{float(r['entry_ask']):.3f}",
              f"{float(r['current_price']):.3f}",
              f"{float(r['current_price'])-float(r['entry_ask']):+.3f}"]
             for _, r in open_pos.iterrows()],
            headers=["question", "class", "score", "buyers", "ask", "now", "drift"],
            tablefmt="rounded_grid"))

    if not flipped.empty:
        print(f"\n{'='*82}")
        print(f"  FLIPPED POSITIONS  ({len(flipped)})  -- exited on consensus reversal")
        print(f"{'='*82}")
        print(tabulate(
            [[(r["question"] or "")[:36], r["indep_class"], r["entry_date"], r["exit_date"],
              f"{float(r['pnl']):+.4f}" if r["pnl"] else "--"]
             for _, r in flipped.iterrows()],
            headers=["question", "class", "entered", "exited", "PnL"],
            tablefmt="rounded_grid"))

    print(f"\n{'='*82}")
    print(f"  RESOLVED POSITIONS  ({len(closed)})")
    print(f"{'='*82}")
    if closed.empty:
        print("  (none resolved yet)")
    else:
        print(tabulate(
            [[(r["question"] or "")[:36], r["indep_class"], r["status"].upper(),
              f"{float(r['pnl']):+.4f}" if r["pnl"] else "--"]
             for _, r in closed.iterrows()],
            headers=["question", "class", "result", "PnL"],
            tablefmt="rounded_grid"))

    print(f"\n{'='*82}")
    print(f"  PER-CLASS SPLIT  (the whole point of this experiment)")
    print(f"{'='*82}")
    summary = _class_summary(ledger)
    if not summary:
        print("  (no positions logged yet)")
    else:
        print(tabulate(summary,
                        headers=["class", "n", "open", "resolved", "win rate", "mean pnl"],
                        tablefmt="rounded_grid"))

    print(f"\n  Ledger -> {STATE_FILE}  ({len(ledger)} rows)")
    print("✓  Update complete.")


if __name__ == "__main__":
    main()
