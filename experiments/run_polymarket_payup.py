#!/usr/bin/env python3
"""
Polymarket pay-up follow book — does the informed-flow synthesis survive as a book?
==================================================================================

`TRADE_QUALITY` found that **paying up** through the mid is the dominant tell of
an informed fill, and `ORDER_FLOW` found that **breadth** (consensus) predicts the
cross-section. This study fuses them into one signal — an **informed-flow** panel
that sums the pool's directional dollars *weighted by pay-up urgency* over a
trailing window — and trades it dollar-neutral (long top / short bottom), so the
favorite-drift beta cancels and only the signal pays out. It is benchmarked
against the plain breadth and imbalance books.

Crucially it also reports an **out-of-sample slice** (``--start``): the priced
universe is dominated by the 2024 election complex, so the strong signals are
re-scored on the post-election period to check they aren't a one-cycle artefact.

Usage
-----
  python experiments/run_polymarket_payup.py
  python experiments/run_polymarket_payup.py --lookback 14 --start 2025-01-01
  python experiments/run_polymarket_payup.py --refresh
"""
import argparse
import logging

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

import _bootstrap  # noqa: F401
from posterioralpha.polymarket import (
    FlowParams,
    build_token_panel,
    candidate_universe,
    fetch_many_traders,
    fetch_volume_wallets,
    flow_signal_panels,
    run_flow_book,
    screen_market_makers,
    select_universe,
)
from posterioralpha.polymarket.paths import RESULTS_DIR, ensure_dirs
from posterioralpha.validation import compute_metrics

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-5s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def slice_metrics(res, start=None):
    r, g = res.returns, res.gross_returns
    if start is not None:
        r, g = r.loc[start:], g.loc[start:]
    eq = r.fillna(0.0).cumsum()
    return (compute_metrics(g, rf=0.0).get("Sharpe", 0.0),
            compute_metrics(r, rf=0.0).get("Sharpe", 0.0),
            float(eq.iloc[-1]) if len(eq) else 0.0,
            float((eq - eq.cummax()).min()) if len(eq) else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-window", type=int, default=120)
    ap.add_argument("--top-tokens", type=int, default=600)
    ap.add_argument("--max-trades", type=int, default=3000)
    ap.add_argument("--min-obs",    type=int, default=10)
    ap.add_argument("--lookback",   type=int, default=14, help="flow window (days)")
    ap.add_argument("--top-frac",   type=float, default=0.2)
    ap.add_argument("--cost",       type=float, default=0.005)
    ap.add_argument("--start",      type=str, default="2025-01-01", help="out-of-sample slice start")
    ap.add_argument("--refresh",    action="store_true")
    ap.add_argument("--no-plots",   action="store_true")
    args = ap.parse_args()

    ensure_dirs()
    print("""
╔══════════════════════════════════════════════════════════╗
║   P O L Y M A R K E T   ·   pay-up follow book            ║
║   informed-flow = consensus × urgency, beta-neutral        ║
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
    logger.info("Panel %d×%d, pool %d (%d MM)", panel.shape[0], panel.shape[1],
                len(traders), sum(is_mm.values()))

    sigs = flow_signal_panels(traders, panel, is_mm, lookback=args.lookback)
    books = {s: run_flow_book(panel, sigs[s],
                              FlowParams(signal=s, lookback=args.lookback,
                                         top_frac=args.top_frac, cost=args.cost))
             for s in ("informed", "breadth", "imbalance")}

    rows = []
    for name, res in books.items():
        gf, nf, pf, df = slice_metrics(res, None)
        go, no, po, do = slice_metrics(res, args.start)
        rows.append([name, f"{gf:>5.2f}", f"{nf:>5.2f}", f"{pf:>+5.2f}", f"{df:>5.2f}",
                     f"{go:>5.2f}", f"{no:>5.2f}", f"{po:>+5.2f}"])

    print("\n" + "═" * 80)
    print(f"  POLYMARKET PAY-UP FOLLOW  ·  $1 gross dollar-neutral, lookback={args.lookback}d, "
          f"top/bottom {args.top_frac:.0%}, cost={args.cost:.1%}/turn")
    print(f"  universe={panel.shape[1]} tokens, pool={len(traders)} ({sum(is_mm.values())} MM)")
    print(f"  full sample {panel.index.min().date()}→{panel.index.max().date()}  |  "
          f"OOS slice from {args.start}")
    print("═" * 80)
    print(tabulate(rows, headers=["signal", "full\ngSh", "full\nnSh", "full\nnPnL",
                                  "full\nnDD", "OOS\ngSh", "OOS\nnSh", "OOS\nnPnL"],
                   tablefmt="rounded_grid"))
    print("  informed = pay-up-weighted directional flow (TRADE_QUALITY × ORDER_FLOW)")

    if not args.no_plots:
        fig, ax = plt.subplots(figsize=(11, 6))
        for name, res in books.items():
            eq = res.returns.fillna(0.0).cumsum()
            ax.plot(eq.index, eq.values, label=name, lw=1.6)
        if args.start:
            ax.axvline(pd.Timestamp(args.start), color="k", ls="--", lw=0.8, alpha=0.6,
                       label=f"OOS {args.start}")
        ax.axhline(0.0, color="k", lw=0.6, alpha=0.4)
        ax.set_title("Polymarket pay-up follow — cumulative PnL (net, $1 gross, dollar-neutral)")
        ax.set_ylabel("cumulative PnL ($/$1 gross)")
        ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=0.25)
        fig.tight_layout()
        out = RESULTS_DIR / "polymarket_payup.png"
        fig.savefig(out, dpi=130)
        logger.info("Saved dashboard → %s", out)

    print("\n✓  Polymarket pay-up follow study complete.")


if __name__ == "__main__":
    main()
