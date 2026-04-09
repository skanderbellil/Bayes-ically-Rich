#!/usr/bin/env python3
"""
Run PosteriorAlpha backtest on the local portfolio_data.csv.

Assets  : SPY (equities), TLT (bonds), GLD (gold), EEM (EM), VNQ (real estate)
Benchmark: SPY  (buy-and-hold)
Period  : 2007-01-01 → 2024-12-31  (2 years warm-up from 2005)
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tabulate import tabulate

from src.backtest import STRATEGIES, run_all_strategies
from src.metrics import compute_metrics
from src.visualization import plot_single_run, plot_multi_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("portfolio_data.csv", parse_dates=["Date"], index_col="Date")
df = df.sort_index()

print(f"\nLoaded: {df.shape[0]} days × {df.shape[1]} assets  "
      f"({df.index[0].date()} → {df.index[-1].date()})")
print(f"Assets: {df.columns.tolist()}\n")

# Arithmetic daily returns from price levels
returns = df.pct_change().dropna()

# SPY is both a portfolio candidate AND the benchmark
spy_returns = returns["SPY"].rename("SPY")

# ── Run all strategies on full 5-ETF universe ──────────────────────────────
logger.info("Running all strategies on 5-ETF universe …")
results = run_all_strategies(
    returns,
    rebalance_freq="ME",
    min_history=252,      # 1 year warm-up within the backtest
    recent_window=60,     # ~3 months 'recent' window
    rf=0.04,
    sensitivity=1.5,
    max_weight=0.60,      # allow concentration (5 assets, not 50)
)

# ── Metrics ────────────────────────────────────────────────────────────────
print("=" * 70)
print("  RESULTS  ·  5-ETF Multi-Asset Portfolio  (2006 → 2024)")
print("=" * 70)

rows = []
col_keys = ["CAGR", "Sharpe", "Sortino", "Max DD", "Calmar", "Alpha", "Beta"]
for strat, res in results.items():
    m = compute_metrics(res.returns, benchmark=spy_returns, rf=0.04)
    row = [strat.replace("_", " ").title()]
    for k in col_keys:
        v = m.get(k, float("nan"))
        row.append(f"{v:.2%}" if k in ("CAGR", "Max DD") else f"{v:.2f}")
    rows.append(row)

# SPY buy-and-hold
m_spy = compute_metrics(spy_returns, rf=0.04)
spy_row = ["SPY Buy & Hold"]
for k in col_keys:
    v = m_spy.get(k, float("nan"))
    spy_row.append(f"{v:.2%}" if k in ("CAGR", "Max DD") else f"{v:.2f}")
rows.append(spy_row)

print(tabulate(rows, headers=["Strategy"] + col_keys, tablefmt="rounded_grid"))

# ── Plots ──────────────────────────────────────────────────────────────────
logger.info("Generating plots …")

plot_single_run(
    results, spy_returns, seed=0,
    save_path=RESULTS_DIR / "dashboard.png",
)
logger.info("Saved: results/dashboard.png")

# Multi-seed summary reused as single-run bar chart
metrics_by_strat = {
    s: [compute_metrics(r.returns, benchmark=spy_returns, rf=0.04)]
    for s, r in results.items()
}
plot_multi_seed(
    metrics_by_strat,
    spy_metrics=m_spy,
    save_path=RESULTS_DIR / "metrics_bars.png",
)
logger.info("Saved: results/metrics_bars.png")

# Save per-strategy daily returns
returns_out = pd.DataFrame({s: r.returns for s, r in results.items()})
returns_out["SPY_benchmark"] = spy_returns
returns_out.to_csv(RESULTS_DIR / "strategy_returns.csv")
logger.info("Saved: results/strategy_returns.csv")

print("\n✓  Done.  Check the results/ folder for plots and CSV.")
