#!/usr/bin/env python3
"""
Large-universe study — works on the full liquid ETF universe, not a subset.

Data comes from the cached financedatabase + yfinance universe
(``datasets/etf_universe_*``; rebuild with ``experiments/build_etf_universe.py``).

Two parts:
  1. Universe summary, straight from the financedatabase instrument info
     (composition by asset class, most-liquid names by median dollar volume).
  2. A diversified cross-asset allocation backtest on a basket constructed
     *from that info* — the single most-liquid ETF in each fd category group —
     comparing the framework's portfolio engines against an SPY benchmark.

Run offline (uses the bundled dataset):

    python experiments/run_large_universe.py
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tabulate import tabulate

import _bootstrap  # noqa: F401  (adds repo root to sys.path)
from posterioralpha.data import (
    load_etf_universe_prices,
    load_etf_universe_returns,
    load_etf_universe_info,
)
from posterioralpha.backtest import run_backtest, run_amr_backtest
from posterioralpha.validation import compute_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S", stream=sys.stdout,
)
logger = logging.getLogger(__name__)
RESULTS_DIR = _bootstrap.ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

BENCH = "SPY"
RF = 0.04

# Universe-agnostic engines (skip bocpd_amr_v2/v3/v4 — those hardcode SPY/TLT/GLD
# for their regime signal).  Monthly Bayesian family + weekly AMR baselines.
MONTHLY = ["equal_weight", "min_variance", "hist_max_sharpe", "bayesian", "bayesian_full"]
WEEKLY  = ["inv_vol", "amr", "hmm3_amr"]


def diversified_basket(info: pd.DataFrame, prices: pd.DataFrame) -> list[str]:
    """Most-liquid ETF in each fd category group → a diversified cross-asset
    basket.  Uses the financedatabase ``category_group`` + ``median_adv_usd``
    info to span equities, fixed income, commodities, real estate, etc."""
    have = info.loc[info.index.isin(prices.columns)].copy()
    have = have.dropna(subset=["median_adv_usd"])
    picks = (
        have.sort_values("median_adv_usd", ascending=False)
            .groupby("category_group", sort=False)
            .head(1)
            .index.tolist()
    )
    if BENCH not in picks and BENCH in prices.columns:
        picks.append(BENCH)
    return sorted(set(picks))


def main() -> None:
    prices = load_etf_universe_prices()
    info = load_etf_universe_info()
    rets_all = load_etf_universe_returns()

    print("\n" + "=" * 74)
    print(f"  LARGE ETF UNIVERSE  ·  {prices.shape[1]} liquid US ETFs  ·  "
          f"{prices.index.min().date()} → {prices.index.max().date()}")
    print("  source: financedatabase (universe + info) + yfinance (history)")
    print("=" * 74)

    print("\nComposition by asset class (financedatabase category_group):")
    print(info["category_group"].value_counts().to_string())

    print("\nMost-liquid names (median daily $ volume):")
    top = info[["name", "category_group", "median_adv_usd"]].head(10).copy()
    top["median_adv_usd"] = (top["median_adv_usd"] / 1e9).round(2).astype(str) + "B"
    print(tabulate(top, headers=["symbol", "name", "asset class", "ADV $"], tablefmt="simple"))

    # ── Diversified cross-asset basket from fd info ──────────────────────────
    basket = diversified_basket(info, prices)
    rets = rets_all[basket].dropna(how="any")
    logger.info("Diversified basket (%d assets): %s", len(basket), basket)
    logger.info("Backtest window: %s → %s (%d days)",
                rets.index.min().date(), rets.index.max().date(), len(rets))

    # ── Run the engines ──────────────────────────────────────────────────────
    rows, curves = [], {}
    for strat in MONTHLY:
        res = run_backtest(rets, strategy=strat, rf=RF, max_weight=0.30)
        m = compute_metrics(res.returns, rf=RF)
        rows.append((strat, "ME", m))
        curves[strat] = res.returns
    for strat in WEEKLY:
        res = run_amr_backtest(rets, strategy=strat)
        m = compute_metrics(res.returns, rf=RF)
        rows.append((strat, "W", m))
        curves[strat] = res.returns

    # SPY benchmark over the same window
    spy = rets[BENCH]
    bench_m = compute_metrics(spy, rf=RF)
    rows.append((f"{BENCH} buy&hold", "—", bench_m))
    curves[f"{BENCH} buy&hold"] = spy

    # ── Metrics table ────────────────────────────────────────────────────────
    keys = ["CAGR", "Volatility", "Sharpe", "Sortino", "Max DD", "Calmar"]
    table = [[name, freq] + [f"{m.get(k, float('nan')):.2f}" for k in keys]
             for name, freq, m in rows]
    print("\n" + "=" * 74)
    print("  DIVERSIFIED CROSS-ASSET ALLOCATION  (basket from fd info)")
    print("=" * 74)
    print(tabulate(table, headers=["strategy", "reb"] + keys, tablefmt="github",
                   floatfmt=".2f"))

    # ── Equity curves ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, r in curves.items():
        eq = (1 + r.reindex(rets.index).fillna(0)).cumprod()
        ax.plot(eq.index, eq.values, label=name, lw=1.6 if name.startswith(BENCH) else 1.1,
                alpha=0.9, color="black" if name.startswith(BENCH) else None)
    ax.set_yscale("log")
    ax.set_title(f"Diversified cross-asset basket ({len(basket)} ETFs from fd info) "
                 f"vs {BENCH}")
    ax.set_ylabel("growth of $1 (log)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = RESULTS_DIR / "large_universe_allocation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    logger.info("Saved: %s", out)

    metrics_df = pd.DataFrame(
        {name: m for name, _, m in rows}
    ).T
    metrics_df.to_csv(RESULTS_DIR / "large_universe_metrics.csv")
    logger.info("Saved: %s", RESULTS_DIR / "large_universe_metrics.csv")
    print("\n✓  Done.")


if __name__ == "__main__":
    main()
