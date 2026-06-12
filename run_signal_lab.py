#!/usr/bin/env python3
"""
Signal Lab run — IC evaluation on the S&P-100 stock panel.

First application of the knowledge-base hygiene to this repo
(docs/knowledge/systematic-equity-strategies.md §5 rule 1):
evaluate cross-sectional signals by IC BEFORE building any backtest.

Universe : sp500_top100_adj_close.csv (98 names, 2016-04 → 2026-04)
Signals  :
  classic      — momentum_12_1, reversal_1m, low_vol
  novel        — erl_stability (per-stock BOCPD regime age)
               — regime_mom    (momentum gated by regime stability)
Output   : IC table (mean IC / t-stat / hit rate per horizon) printed and
           saved to results/signal_lab_ic.csv

Pass criteria (Grinold & Kahn + Harvey et al.):
  interesting  — |mean IC| > 0.02 and |t| > 2
  publishable  — |t| > 3 (multiple-testing bar)
Every signal evaluated here counts as a trial for future DSR accounting.
"""
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tabulate import tabulate

from src.signal_lab import (
    momentum_12_1, reversal_1m, low_vol,
    erl_panel, regime_stability, regime_mom,
    evaluate_signals,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)
logger = logging.getLogger(__name__)
RESULTS_DIR = Path("results"); RESULTS_DIR.mkdir(exist_ok=True)

CSV = "sp500_top100_adj_close.csv"


def main():
    prices = pd.read_csv(CSV, index_col=0, parse_dates=True).sort_index()
    # Drop names with sparse history (late IPOs) — survivors of the coverage
    # filter still carry survivorship caveats, noted in the report.
    coverage = prices.notna().mean()
    prices = prices.loc[:, coverage > 0.90]
    logger.info(f"Universe: {prices.shape[1]} names, "
                f"{prices.index[0].date()} → {prices.index[-1].date()}")

    logger.info("Computing classic signals …")
    sig_mom = momentum_12_1(prices)
    sig_rev = reversal_1m(prices)
    sig_lv  = low_vol(prices)

    logger.info("Computing per-stock BOCPD ERL (this is the slow part) …")
    t0 = time.time()
    erl = erl_panel(prices, hazard=1 / 126)
    logger.info(f"  BOCPD panel done in {time.time() - t0:.0f}s")

    stab = regime_stability(erl, mature_days=126.0)
    sig_rmom = regime_mom(sig_mom, stab)

    signals = {
        "momentum_12_1": sig_mom,
        "reversal_1m":   sig_rev,
        "low_vol":       sig_lv,
        "erl_stability": stab,
        "regime_mom":    sig_rmom,
    }

    logger.info("Evaluating ICs at 21/63/126-day horizons …")
    report = evaluate_signals(prices, signals, horizons=(21, 63, 126))

    out = RESULTS_DIR / "signal_lab_ic.csv"
    report.to_csv(out, index=False)
    logger.info(f"Saved {out}")

    show = report.copy()
    for c in ("mean_ic", "ic_vol", "t_stat", "hit_rate",
              "ann_spread", "spread_sharpe"):
        if c in show.columns:
            show[c] = show[c].astype(float).round(3)
    print()
    print(tabulate(show, headers="keys", tablefmt="github", showindex=False))
    print()
    print("Pass bar: |mean IC| > 0.02 with |t| > 2 (interesting), |t| > 3 (publishable).")
    print("NOTE: 5 signals tested = 5 trials. Universe is current S&P-100 →")
    print("survivorship-biased; treat positive results as upper bounds.")


if __name__ == "__main__":
    main()
