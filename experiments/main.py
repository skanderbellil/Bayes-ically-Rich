#!/usr/bin/env python3
"""
PosteriorAlpha — Bayesian Adaptive Portfolio Backtester
=======================================================

Backtests four strategies on 10 years of real S&P 500 data.
Runs multiple independent random samples of 50 stocks to test robustness.

Strategies compared
-------------------
  bayesian        — Bayesian shrinkage portfolio (our hero)
  equal_weight    — 1/N rebalanced monthly
  min_variance    — long-only global minimum variance
  hist_max_sharpe — max Sharpe on full historical prior (no adaptation)

Usage
-----
  python main.py                         # 5 seeds × 50 stocks, 2014–2024
  python main.py --seeds 10 --n-stocks 50
  python main.py --start 2012-01-01 --end 2022-12-31
  python main.py --no-plots              # skip matplotlib output
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from tabulate import tabulate

import _bootstrap  # noqa: F401  (adds repo root to sys.path)
from posterioralpha.backtest.bayesian import STRATEGIES, BacktestResult, run_all_strategies
from posterioralpha.data.market import download_returns, get_sp500_tickers, sample_tickers
from posterioralpha.validation.metrics import compute_metrics
from posterioralpha.validation.plots import plot_lambda_heatmap, plot_multi_seed, plot_single_run

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

RESULTS_DIR = _bootstrap.ROOT / "results"
WARMUP_YEARS = 2    # years of history before backtest window (prior warm-up)


# ---------------------------------------------------------------------------
# Single-seed runner
# ---------------------------------------------------------------------------

def run_seed(
    universe: List[str],
    seed: int,
    n_stocks: int,
    bt_start: str,
    bt_end: str,
    rf: float,
    sensitivity: float,
    max_weight: float,
) -> Optional[Dict[str, BacktestResult]]:
    """Download data and backtest all strategies for one random seed."""
    sep = "─" * 60
    logger.info(f"\n{sep}")
    logger.info(f"  SEED {seed}  ·  sampling {n_stocks} tickers from {len(universe)}")
    logger.info(sep)

    tickers = sample_tickers(universe, n=n_stocks, seed=seed)
    logger.info(f"  Sample: {tickers[:6]} … (+{len(tickers)-6} more)")

    # Pull extra warm-up data so the prior is meaningful from day 1
    data_start = f"{int(bt_start[:4]) - WARMUP_YEARS}{bt_start[4:]}"

    returns = download_returns(tickers, start=data_start, end=bt_end)

    # Keep only assets present for the full span
    returns = returns.dropna(axis=1, how="any")

    if returns.shape[1] < 10:
        logger.warning(f"  Only {returns.shape[1]} clean assets — skipping seed {seed}")
        return None

    logger.info(f"  Clean assets: {returns.shape[1]}  ·  Days: {returns.shape[0]}")

    results = run_all_strategies(
        returns,
        rebalance_freq="ME",
        min_history=252,
        recent_window=60,
        rf=rf,
        sensitivity=sensitivity,
        max_weight=max_weight,
    )

    # Log per-strategy summary
    header = f"  {'Strategy':<22} {'CAGR':>8} {'Sharpe':>8} {'Max DD':>8} {'Calmar':>8}"
    logger.info(header)
    logger.info(f"  {'─'*56}")
    for strat, res in results.items():
        m = compute_metrics(res.returns)
        logger.info(
            f"  {strat:<22} "
            f"{m.get('CAGR', 0):>8.2%} "
            f"{m.get('Sharpe', 0):>8.2f} "
            f"{m.get('Max DD', 0):>8.2%} "
            f"{m.get('Calmar', 0):>8.2f}"
        )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PosteriorAlpha — Bayesian Adaptive Portfolio Backtester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--seeds",       type=int,   default=5,          help="Number of random seeds (default: 5)")
    parser.add_argument("--n-stocks",    type=int,   default=50,         help="Stocks per run (default: 50)")
    parser.add_argument("--start",       type=str,   default="2014-01-01", help="Backtest start date")
    parser.add_argument("--end",         type=str,   default="2024-12-31", help="Backtest end date")
    parser.add_argument("--rf",          type=float, default=0.04,       help="Annual risk-free rate (default: 0.04)")
    parser.add_argument("--sensitivity", type=float, default=1.5,        help="λ sigmoid sensitivity (default: 1.5)")
    parser.add_argument("--max-weight",  type=float, default=0.20,       help="Max per-asset weight (default: 0.20)")
    parser.add_argument("--no-plots",    action="store_true",            help="Skip plot generation")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    banner = """
╔══════════════════════════════════════════════════════════╗
║          P O S T E R I O R A L P H A                    ║
║    Bayesian Adaptive Portfolio  ·  10-Year Backtest      ║
╚══════════════════════════════════════════════════════════╝"""
    print(banner)
    logger.info(f"Period  : {args.start} → {args.end}  ({WARMUP_YEARS}y warm-up)")
    logger.info(f"Seeds   : {args.seeds}  ·  Stocks/run: {args.n_stocks}")
    logger.info(f"rf={args.rf:.2%}  sensitivity={args.sensitivity}  max_weight={args.max_weight:.0%}")

    # ── Universe ──────────────────────────────────────────────────────────
    logger.info("\nFetching S&P 500 universe …")
    universe = get_sp500_tickers()

    # ── SPY benchmark ─────────────────────────────────────────────────────
    logger.info("Downloading SPY benchmark …")
    spy_raw = yf.download(
        "SPY", start=args.start, end=args.end,
        auto_adjust=True, progress=False,
    )
    spy_prices  = spy_raw["Close"].squeeze()
    spy_returns = spy_prices.pct_change().dropna()
    spy_returns.name = "SPY"
    spy_metrics = compute_metrics(spy_returns, rf=args.rf)

    # ── Per-seed runs ─────────────────────────────────────────────────────
    all_seed_results: Dict[int, Dict[str, BacktestResult]] = {}
    metrics_by_strat: Dict[str, List[Dict]] = {s: [] for s in STRATEGIES}
    lambda_by_seed:   Dict[int, pd.Series]  = {}

    for seed in range(args.seeds):
        seed_res = run_seed(
            universe,
            seed=seed,
            n_stocks=args.n_stocks,
            bt_start=args.start,
            bt_end=args.end,
            rf=args.rf,
            sensitivity=args.sensitivity,
            max_weight=args.max_weight,
        )
        if seed_res is None:
            continue

        all_seed_results[seed] = seed_res

        for strat, res in seed_res.items():
            m = compute_metrics(res.returns, benchmark=spy_returns, rf=args.rf)
            metrics_by_strat[strat].append(m)

        if "bayesian" in seed_res and seed_res["bayesian"].lambdas is not None:
            lambda_by_seed[seed] = seed_res["bayesian"].lambdas

        # Per-seed dashboard
        if not args.no_plots:
            fig_path = RESULTS_DIR / f"seed_{seed:02d}_dashboard.png"
            plot_single_run(seed_res, spy_returns, seed=seed, save_path=fig_path)
            logger.info(f"  Saved: {fig_path}")

    if not all_seed_results:
        logger.error("All seeds failed — check your internet connection and try again.")
        sys.exit(1)

    # ── Multi-seed summary plots ──────────────────────────────────────────
    if not args.no_plots:
        fig_path = RESULTS_DIR / "multi_seed_summary.png"
        plot_multi_seed(metrics_by_strat, spy_metrics=spy_metrics, save_path=fig_path)
        logger.info(f"Saved: {fig_path}")

        if lambda_by_seed:
            fig_path = RESULTS_DIR / "lambda_heatmap.png"
            plot_lambda_heatmap(lambda_by_seed, save_path=fig_path)
            logger.info(f"Saved: {fig_path}")

    # ── Aggregate results table ───────────────────────────────────────────
    print("\n" + "═" * 72)
    print("  MULTI-SEED SUMMARY  ·  mean ± std across random stock samples")
    print("═" * 72)

    table_rows = []
    col_keys   = ["CAGR", "Sharpe", "Sortino", "Max DD", "Calmar"]

    for strat in STRATEGIES:
        mlist = metrics_by_strat[strat]
        if not mlist:
            continue
        row = [strat.replace("_", " ").title()]
        for k in col_keys:
            vals = [m[k] for m in mlist if k in m]
            if not vals:
                row.append("—")
                continue
            mu, sd = np.mean(vals), np.std(vals)
            row.append(f"{mu:.2%} ±{sd:.2%}" if k in ("CAGR", "Max DD") else f"{mu:.2f} ±{sd:.2f}")
        table_rows.append(row)

    # SPY row (single value, no std)
    spy_row = ["S&P 500 (SPY)"]
    for k in col_keys:
        v = spy_metrics.get(k, 0)
        spy_row.append(f"{v:.2%}" if k in ("CAGR", "Max DD") else f"{v:.2f}")
    table_rows.append(spy_row)

    print(tabulate(table_rows, headers=["Strategy"] + col_keys, tablefmt="rounded_grid"))

    # ── Save raw metrics CSV ──────────────────────────────────────────────
    rows = []
    for seed, seed_res in all_seed_results.items():
        for strat, res in seed_res.items():
            m          = compute_metrics(res.returns, benchmark=spy_returns, rf=args.rf)
            m["seed"]  = seed
            m["strat"] = strat
            rows.append(m)

    csv_path = RESULTS_DIR / "all_metrics.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logger.info(f"\nRaw metrics → {csv_path}")
    logger.info(f"All plots   → {RESULTS_DIR.resolve()}/")

    print("\n✓  PosteriorAlpha backtest complete.")


if __name__ == "__main__":
    main()
