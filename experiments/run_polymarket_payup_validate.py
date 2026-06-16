#!/usr/bin/env python3
"""
Polymarket pay-up follow — does it survive realistic costs & walk-forward tuning?
================================================================================

`PAYUP_FOLLOW` showed the informed-flow book nets Sharpe ~0.70 (0.67 out of
sample) at a flat 50 bps half-spread. Two honest objections remained: (1) real
Polymarket cost is the **bid-ask spread**, wider on thin books than 50 bps; (2)
the lookback / top-frac were hand-picked. This study answers both.

  1. **Break-even cost sweep.** Re-run the book at flat half-spreads from 0.5¢ to
     5¢ and find the spread at which net Sharpe hits zero — the cost budget the
     edge can absorb.
  2. **Liquidity-scaled cost.** Charge each token its own half-spread ≈
     k/√(cohort notional) (≈0.5¢ on liquid headline books, up to 4¢ on thin
     ones) — a realistic, per-market cost — and report the net Sharpe.
  3. **Walk-forward hyper-parameter selection.** Roll a train/test window through
     time; on each train block pick the (lookback, top-frac) with the best net
     Sharpe *under the liquidity cost*, apply it to the next test block, stitch
     the test returns. The honest out-of-sample number — no hyper-parameter is
     ever chosen with future data.

Usage
-----
  python experiments/run_polymarket_payup_validate.py
  python experiments/run_polymarket_payup_validate.py --train 200 --test 90 --refresh
"""
import argparse
import logging

import numpy as np
import pandas as pd
from tabulate import tabulate

import _bootstrap  # noqa: F401
from posterioralpha.polymarket import (
    FlowParams,
    build_token_panel,
    candidate_universe,
    cohort_token_notional,
    fetch_many_traders,
    fetch_volume_wallets,
    flow_signal_panels,
    liquidity_spread,
    run_flow_book,
    screen_market_makers,
    select_universe,
)
from posterioralpha.polymarket.paths import ensure_dirs
from posterioralpha.validation import compute_metrics

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-5s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

LOOKBACKS = [7, 10, 14, 21]
TOP_FRACS = [0.1, 0.2, 0.3]


def sharpe(r: pd.Series) -> float:
    return compute_metrics(r.dropna(), rf=0.0).get("Sharpe", 0.0) if len(r.dropna()) > 5 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-window", type=int, default=120)
    ap.add_argument("--top-tokens", type=int, default=600)
    ap.add_argument("--max-trades", type=int, default=3000)
    ap.add_argument("--min-obs",    type=int, default=10)
    ap.add_argument("--train",      type=int, default=180, help="walk-forward train window (days)")
    ap.add_argument("--test",       type=int, default=90,  help="walk-forward test window (days)")
    ap.add_argument("--refresh",    action="store_true")
    args = ap.parse_args()

    ensure_dirs()
    print("""
╔══════════════════════════════════════════════════════════╗
║   P O L Y M A R K E T   ·   pay-up follow — validation     ║
║   realistic spread + walk-forward hyper-parameter choice   ║
╚══════════════════════════════════════════════════════════╝""")

    pool = candidate_universe(("1d", "7d", "30d", "all"), per_window=args.per_window)
    wallets = list(dict.fromkeys(pool["wallet"].tolist()
                                 + sorted(fetch_volume_wallets("all", 150))))
    traders = fetch_many_traders(wallets, max_trades=args.max_trades, refresh=args.refresh)
    if not traders:
        raise SystemExit("No trader histories returned.")
    universe = select_universe(traders, top_k=args.top_tokens)
    panel = build_token_panel(universe, min_obs=args.min_obs, use_cache=not args.refresh)
    universe = [t for t in universe if t in panel.columns]
    is_mm = screen_market_makers(traders, panel, set(fetch_volume_wallets("all", 150)))
    notional = cohort_token_notional(traders)
    cost_vec = liquidity_spread(panel, notional)
    logger.info("Panel %d×%d, pool %d (%d MM); liq half-spread median %.3f¢, mean %.3f¢",
                panel.shape[0], panel.shape[1], len(traders), sum(is_mm.values()),
                100 * cost_vec.median(), 100 * cost_vec.mean())

    # precompute the informed-flow signal per lookback, and the full-sample book
    # under the liquidity cost for every (lookback, top_frac) combo
    sig = {lb: flow_signal_panels(traders, panel, is_mm, lookback=lb)["informed"] for lb in LOOKBACKS}
    books = {}
    for lb in LOOKBACKS:
        for tf in TOP_FRACS:
            r = run_flow_book(panel, sig[lb],
                              FlowParams(signal="informed", lookback=lb, top_frac=tf),
                              cost_by_token=cost_vec)
            books[(lb, tf)] = r.returns

    # ── 1+2: break-even cost sweep (fixed 14/0.2) + liquidity cost ────────────
    base_sig = sig[14]
    print("\n" + "═" * 64)
    print("  1) BREAK-EVEN COST SWEEP  ·  informed, lookback=14, top-frac=0.2")
    print("═" * 64)
    rows = []
    for hs in (0.005, 0.01, 0.015, 0.02, 0.03, 0.05):
        r = run_flow_book(panel, base_sig, FlowParams(signal="informed", lookback=14,
                                                      top_frac=0.2, cost=hs)).returns
        rows.append([f"{hs*100:.1f}¢", f"{sharpe(r):+.2f}", f"{r.sum():+.2f}"])
    print(tabulate(rows, headers=["half-spread", "net Sharpe", "net PnL"], tablefmt="rounded_grid"))
    liq_r = books[(14, 0.2)]
    print(f"  liquidity-scaled cost (per-token ≈k/√notional): "
          f"net Sharpe {sharpe(liq_r):+.2f}, PnL {liq_r.sum():+.2f}")

    # ── 3: walk-forward hyper-parameter selection (under liquidity cost) ──────
    dates = panel.index
    chosen, wf_chunks, fixed_chunks = [], [], []
    start = args.train
    while start + args.test <= len(dates):
        tr = dates[start - args.train:start]
        te = dates[start:start + args.test]
        # pick combo by train net Sharpe
        best = max(books, key=lambda c: sharpe(books[c].loc[tr]))
        chosen.append((te[0].date(), best, round(sharpe(books[best].loc[tr]), 2)))
        wf_chunks.append(books[best].loc[te])
        fixed_chunks.append(books[(14, 0.2)].loc[te])
        start += args.test

    wf = pd.concat(wf_chunks) if wf_chunks else pd.Series(dtype=float)
    fx = pd.concat(fixed_chunks) if fixed_chunks else pd.Series(dtype=float)
    print("\n" + "═" * 64)
    print(f"  3) WALK-FORWARD HP SELECTION  ·  train={args.train}d / test={args.test}d, "
          f"liquidity cost")
    print("═" * 64)
    for d, c, s in chosen:
        print(f"    test from {d}: chose lookback={c[0]:>2d}, top_frac={c[1]:.1f}  (train Sh {s:+.2f})")
    print(f"\n  walk-forward (HP chosen on train only):  net Sharpe {sharpe(wf):+.2f}, "
          f"PnL {wf.sum():+.2f}, n={len(wf)}d")
    print(f"  fixed 14/0.2 over the same test union:   net Sharpe {sharpe(fx):+.2f}, "
          f"PnL {fx.sum():+.2f}")

    print("\n✓  Polymarket pay-up validation complete.")


if __name__ == "__main__":
    main()
