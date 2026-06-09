#!/usr/bin/env python3
"""
Net liquidity × price trend — does confirmation beat either signal alone?

Net liquidity is a slow macro regime dial (de-risking, not forecasting); a
200-day trend filter is a fast price-confirmation signal.  This study combines
them on SPY and compares against each standalone signal and buy & hold:

  trend       : SPY above its 200-day moving average
  liquidity   : net liquidity expanding (6-month change > 0), publication-lagged
  AND         : invested only when BOTH agree (defensive)
  vote (0/½/1): exposure = fraction of the two signals that are bullish

All signals are causal (lagged a day; liquidity also lagged for FRED's release
delay).  Runs offline on the bundled datasets.

    python experiments/run_liquidity_trend.py
"""
import logging
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tabulate import tabulate

import _bootstrap  # noqa: F401
from posterioralpha.data import load_net_liquidity, load_etf_universe_prices
from posterioralpha.validation import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)
logger = logging.getLogger(__name__)
RESULTS_DIR = _bootstrap.ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RF = 0.04
DAILY_RF = RF / 252
H = 126          # 6-month liquidity change (peak predictive horizon)
SMA = 200        # trend window
PUB_LAG = 5


def main() -> None:
    nl = load_net_liquidity()["net_liquidity"]
    spy = load_etf_universe_prices()["SPY"]
    idx = nl.dropna().index.intersection(spy.dropna().index)
    nl, spy = nl.reindex(idx), spy.reindex(idx)
    spy_ret = spy.pct_change()

    # Signals (causal: shift by a day; liquidity also by the publication lag)
    trend_on = (spy > spy.rolling(SMA).mean()).astype(float).shift(1).fillna(0.0)
    liq_on = ((nl.diff(H) > 0).astype(float)
              .shift(PUB_LAG).shift(1).fillna(0.0))

    def ret(expo):
        return pd.Series(expo * spy_ret + (1 - expo) * DAILY_RF, index=idx)

    both = trend_on * liq_on
    vote = 0.5 * (trend_on + liq_on)

    rows = [
        ["Trend only (SPY>200d)", ret(trend_on), trend_on],
        ["Liquidity only (Δ6m>0)", ret(liq_on), liq_on],
        ["AND (both agree)", ret(both), both],
        ["Vote (0/½/1 exposure)", ret(vote), vote],
        ["SPY buy & hold", spy_ret, pd.Series(1.0, index=idx)],
    ]

    print("\n" + "=" * 78)
    print(f"  NET LIQUIDITY × PRICE TREND ON SPY  ·  {idx.min().date()} → {idx.max().date()}")
    print("=" * 78)
    keys = ["CAGR", "Volatility", "Sharpe", "Sortino", "Max DD", "Calmar"]
    table = [[name] + [f"{compute_metrics(r.dropna(), rf=RF).get(k, float('nan')):.2f}" for k in keys]
             for name, r, _ in rows]
    print(tabulate(table, headers=["strategy"] + keys, tablefmt="github"))
    print(f"\n  avg exposure — trend {float(trend_on.mean()):.0%} | "
          f"liquidity {float(liq_on.mean()):.0%} | "
          f"AND {float(both.mean()):.0%} | vote {float(vote.mean()):.0%}")
    # agreement diagnostics
    agree = float((trend_on == liq_on).mean())
    print(f"  signals agree {agree:.0%} of the time")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, r, _ in rows:
        eq = (1 + r.reindex(idx).fillna(0)).cumprod()
        ax.plot(eq.index, eq.values, lw=1.5 if "buy" in name else 1.2,
                color="black" if "buy" in name else None,
                alpha=0.85, label=name)
    ax.set_yscale("log"); ax.set_ylabel("growth of $1 (log)")
    ax.set_title("Net liquidity × 200-day trend on SPY")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    out = RESULTS_DIR / "liquidity_trend.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    logger.info("Saved: %s", out)
    print("\n✓  Done.")


if __name__ == "__main__":
    main()
