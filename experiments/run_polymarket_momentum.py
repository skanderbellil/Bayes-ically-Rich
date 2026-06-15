#!/usr/bin/env python3
"""
Polymarket cross-market momentum — does probability drift persist or revert?
============================================================================

A prediction-market study, deliberately *outside* the equity/ETF portfolio work
that fills the rest of the repo. Each Polymarket market resolves Yes/No and its
Yes-token price is the implied probability; we trade the **cross-section of those
probabilities** in log-odds space.

The experiment pits four books against each other on real, resolved markets
pulled live from Polymarket (Gamma metadata + CLOB price history, cached under
``data/raw/polymarket/``):

  xs_momentum  — long the markets drifting up the most vs their peers, short the
                 ones drifting down the most (dollar-neutral). The hero.
  xs_reversal  — the contrarian leg. Prediction markets are famous for the
                 favorite-longshot bias / mean-reversion, so this is the null
                 the momentum hypothesis has to beat.
  ts_momentum  — per-market time-series momentum (sign of own trailing move).
  long_all     — equal-weight long every active market (drift / yield baseline).

All signals are causal (formed from prices up to t, held into t+1) and
Bayesian-shrunk by each market's own log-odds noise. Costs are charged on
turnover. Output: a metrics table + an equity-curve / diagnostics dashboard.

Usage
-----
  python experiments/run_polymarket_momentum.py
  python experiments/run_polymarket_momentum.py --n-markets 200 --lookback 5
  python experiments/run_polymarket_momentum.py --refresh        # re-pull live data
  python experiments/run_polymarket_momentum.py --no-plots
"""
import argparse
import logging

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate

import _bootstrap  # noqa: F401  (adds repo root to sys.path)
from posterioralpha.polymarket import PMParams, build_price_panel, run_polymarket_momentum
from posterioralpha.polymarket.paths import RESULTS_DIR, ensure_dirs
from posterioralpha.validation import compute_metrics

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-5s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

STRATEGIES = ["xs_momentum", "xs_reversal", "ts_momentum", "long_all"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-markets",  type=int,   default=150,   help="markets to pull (default 150)")
    ap.add_argument("--min-volume", type=float, default=50_000, help="min lifetime USD volume")
    ap.add_argument("--lookback",   type=int,   default=7,     help="momentum window in days")
    ap.add_argument("--holding",    type=int,   default=1,     help="rebalance every N days")
    ap.add_argument("--top-frac",   type=float, default=0.2,   help="fraction per long/short leg")
    ap.add_argument("--cost",       type=float, default=0.005, help="turnover cost per unit notional")
    ap.add_argument("--no-bayes",   action="store_true",       help="disable Bayesian shrinkage")
    ap.add_argument("--refresh",    action="store_true",       help="re-pull live data (ignore cache)")
    ap.add_argument("--no-plots",   action="store_true",       help="skip the dashboard")
    args = ap.parse_args()

    ensure_dirs()

    banner = """
╔══════════════════════════════════════════════════════════╗
║   P O L Y M A R K E T   ·   cross-market momentum          ║
║   prediction-market probabilities, traded in log-odds      ║
╚══════════════════════════════════════════════════════════╝"""
    print(banner)

    # ── Stage 1: DATA ─────────────────────────────────────────────────────
    logger.info("Building Yes-price panel from Polymarket (live + cached) …")
    panel, meta = build_price_panel(
        n_markets=args.n_markets,
        min_volume=args.min_volume,
        closed=True,
        refresh=args.refresh,
    )
    logger.info("Panel: %d days (%s → %s) × %d markets",
                panel.shape[0], panel.index.min().date(), panel.index.max().date(),
                panel.shape[1])
    logger.info("Median markets live per day: %.0f", panel.notna().sum(axis=1).median())

    # ── Stages 2+3: research signal → backtest ────────────────────────────
    results = {}
    for strat in STRATEGIES:
        params = PMParams(
            strategy=strat,
            lookback=args.lookback,
            holding=args.holding,
            top_frac=args.top_frac,
            cost=args.cost,
            bayesian=not args.no_bayes,
        )
        results[strat] = run_polymarket_momentum(panel, params)

    # ── Stage 4: VALIDATION ───────────────────────────────────────────────
    # Report gross *and* net: with daily rebalancing, turnover cost is the binding
    # constraint, so the honest read separates "is there signal?" (gross) from
    # "does it survive frictions?" (net). These are fixed-$1-notional, marked-to-
    # market PnL streams, so we accumulate them ADDITIVELY (cumulative PnL on $1
    # gross) rather than compounding — compounding would inject spurious vol-drag
    # from resolution jumps. Sharpe is taken from compute_metrics (scale-invariant).
    def cum_pnl_dd(r: pd.Series) -> tuple[float, float]:
        eq = r.fillna(0.0).cumsum()
        dd = float((eq - eq.cummax()).min())   # max drawdown of the $1-gross PnL curve
        return float(eq.iloc[-1]), dd

    rows = []
    for strat in STRATEGIES:
        res = results[strat]
        g_sharpe = compute_metrics(res.gross_returns, rf=0.0).get("Sharpe", 0.0)
        n_sharpe = compute_metrics(res.returns, rf=0.0).get("Sharpe", 0.0)
        g_pnl, _      = cum_pnl_dd(res.gross_returns)
        n_pnl, n_dd   = cum_pnl_dd(res.returns)
        rows.append([
            strat,
            f"{g_sharpe:>6.2f}", f"{g_pnl:>+7.2f}",
            f"{n_sharpe:>6.2f}", f"{n_pnl:>+7.2f}", f"{n_dd:>7.2f}",
            f"{res.turnover.mean():>6.3f}",
        ])

    print("\n" + "═" * 78)
    print(f"  POLYMARKET MOMENTUM  ·  $1 gross, lookback={args.lookback}d, "
          f"rebalance={args.holding}d, cost={args.cost:.1%}/turn")
    print("  (Return/DD = cumulative PnL per $1 gross notional, additive)")
    print("═" * 78)
    print(tabulate(rows,
                   headers=["strategy", "gross\nSharpe", "gross\nPnL",
                            "net\nSharpe", "net\nPnL", "net\nmaxDD", "avg\nturn"],
                   tablefmt="rounded_grid"))
    print("  Note: xs_reversal is the exact negative book of xs_momentum (gross),")
    print("  so opposite-signed gross Sharpes/PnL confirm the cross-sectional signal's direction.")

    # ── Dashboard ─────────────────────────────────────────────────────────
    if not args.no_plots:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8),
                                       gridspec_kw={"height_ratios": [3, 1]})
        for strat in STRATEGIES:
            eq = results[strat].returns.fillna(0.0).cumsum()   # additive PnL on $1 gross
            ax1.plot(eq.index, eq.values, label=strat, lw=1.6)
        ax1.axhline(0.0, color="k", lw=0.6, alpha=0.4)
        ax1.set_title("Polymarket cross-market momentum — cumulative PnL (net, $1 gross)")
        ax1.set_ylabel("cumulative PnL ($/$1 gross)")
        ax1.legend(loc="upper left", fontsize=9)
        ax1.grid(alpha=0.25)

        ax2.plot(panel.index, panel.notna().sum(axis=1).values, color="tab:gray", lw=1.2)
        ax2.set_title("markets live per day")
        ax2.set_ylabel("# active")
        ax2.grid(alpha=0.25)

        fig.tight_layout()
        out = RESULTS_DIR / "polymarket_momentum.png"
        fig.savefig(out, dpi=130)
        logger.info("Saved dashboard → %s", out)

    print("\n✓  Polymarket momentum study complete.")


if __name__ == "__main__":
    main()
