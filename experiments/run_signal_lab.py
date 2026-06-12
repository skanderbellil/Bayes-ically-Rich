#!/usr/bin/env python3
"""
IC lab on the S&P-100 panel: do the per-stock BOCPD families carry
cross-sectional information?

Question
--------
The miner already searches a market-level regime-gated momentum family.
This study asks the prior question for the two NEW per-stock families
(`stock_regime_momentum`, `regime_age`): is there any cross-sectional
information coefficient at all, benchmarked against plain momentum,
short-term reversal, and low-vol — evaluated the Grinold & Kahn way
(monthly rank IC vs. forward returns), before any of them is allowed to
consume a trial in the miner's gauntlet.

Pass bar: |mean IC| > 0.02 with |t| > 2 (interesting), |t| > 3
(publishable, Harvey et al.). Every signal here counts as a trial.

⚠️ current-membership S&P-100 → survivorship-biased; treat positive
results as upper bounds (the low-vol row makes this bias visible).
"""
import logging
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401  (repo root on sys.path)
import pandas as pd
from tabulate import tabulate

from posterioralpha.data.loaders import load_sp500_prices
from posterioralpha.mining.ic import evaluate_signals
from posterioralpha.mining.signals import (
    SignalContext, low_volatility, mean_reversion, momentum,
    regime_age, stock_regime_momentum,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)
logger = logging.getLogger(__name__)
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def main():
    prices = load_sp500_prices()
    prices = prices.loc[:, prices.notna().mean() > 0.90]   # drop late IPOs
    logger.info(f"Universe: {prices.shape[1]} names, "
                f"{prices.index[0].date()} → {prices.index[-1].date()}")

    ctx = SignalContext(prices=prices, returns=prices.pct_change())

    logger.info("Computing per-stock BOCPD ERL panel (slow first call) …")
    t0 = time.time()
    _ = ctx.erl_panel
    logger.info(f"  done in {time.time() - t0:.0f}s")

    signals = {
        "momentum_12_1":         momentum(ctx, lookback=252, skip=21),
        "reversal_1m":           mean_reversion(ctx, lookback=21),
        "low_vol_1y":            low_volatility(ctx, lookback=252),
        "regime_age":            regime_age(ctx, mature_days=126.0),
        "stock_regime_momentum": stock_regime_momentum(
            ctx, lookback=252, skip=21, mature_days=126.0),
    }

    logger.info("Evaluating monthly rank ICs at 21/63/126-day horizons …")
    report = evaluate_signals(prices, signals, horizons=(21, 63, 126))

    out = RESULTS_DIR / "signal_lab_ic.csv"
    report.to_csv(out, index=False)
    logger.info(f"Saved {out}")

    show = report.copy()
    for c in ("mean_ic", "ic_vol", "t_stat", "hit_rate",
              "ann_spread", "spread_sharpe"):
        show[c] = show[c].astype(float).round(3)
    print()
    print(tabulate(show, headers="keys", tablefmt="github", showindex=False))
    print()
    print("Pass bar: |mean IC| > 0.02 with |t| > 2; trials ledger += 5.")


if __name__ == "__main__":
    main()
