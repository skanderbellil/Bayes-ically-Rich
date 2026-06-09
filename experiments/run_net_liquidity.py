#!/usr/bin/env python3
"""
Net liquidity and the effect on equities.

Net liquidity (Fed assets − Treasury General Account − Reverse Repo) is a proxy
for the dollar liquidity sloshing into risk assets.  This study:

  1. Quantifies the relationship between net-liquidity changes and SPY returns
     (contemporaneous and lead-lag).
  2. Tests a simple liquidity-regime timing rule on SPY: hold equities while
     net liquidity is expanding, step to cash while it contracts.

Publication lag is respected — the FRED series are released with a delay, so
the signal is lagged a week before it drives any position (no lookahead).

    python experiments/run_net_liquidity.py            # offline (cached FRED)
    python experiments/run_net_liquidity.py --build     # refresh from FRED (network)
"""
import argparse
import logging
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tabulate import tabulate

import _bootstrap  # noqa: F401  (adds repo root to sys.path)
from posterioralpha.data import load_net_liquidity, load_etf_universe_prices
from posterioralpha.validation import compute_metrics

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S", stream=sys.stdout,
)
logger = logging.getLogger(__name__)
RESULTS_DIR = _bootstrap.ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RF = 0.04
PUB_LAG = 5          # business days: FRED release delay (avoid lookahead)
CHG_WIN = 65         # ~13-week change horizon for the liquidity signal
DAILY_RF = RF / 252


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true", help="refresh FRED data (network)")
    args = ap.parse_args()

    if args.build:
        from posterioralpha.data.macro import build_net_liquidity
        nl_df = build_net_liquidity()
    else:
        nl_df = load_net_liquidity()

    nl = nl_df["net_liquidity"]
    spy = load_etf_universe_prices()["SPY"].reindex(nl.index).ffill()
    common = nl.dropna().index.intersection(spy.dropna().index)
    nl, spy = nl.reindex(common), spy.reindex(common)
    spy_ret = spy.pct_change()

    print("\n" + "=" * 74)
    print(f"  NET LIQUIDITY vs EQUITIES  ·  {common.min().date()} → {common.max().date()}")
    print("  net liquidity = Fed assets (WALCL) − TGA (WTREGEN) − RRP (RRPONTSYD)")
    print("=" * 74)
    print(f"\n  net liquidity now: ${nl.iloc[-1]/1000:.2f}T   "
          f"({CHG_WIN}d change: {nl.diff(CHG_WIN).iloc[-1]:+,.0f}bn)")

    # ── 1. Relationship ──────────────────────────────────────────────────────
    d_nl = nl.diff(CHG_WIN)
    fwd_ret = spy.pct_change(CHG_WIN).shift(-CHG_WIN)        # next 13wk SPY return
    con_ret = spy.pct_change(CHG_WIN)                        # same 13wk SPY return
    def corr(a, b):
        m = a.notna() & b.notna()
        return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 10 else float("nan")
    print("\nCorrelation of 13-week net-liquidity change with SPY return:")
    print(f"  contemporaneous : {corr(d_nl, con_ret):+.3f}")
    print(f"  next 13 weeks   : {corr(d_nl, fwd_ret):+.3f}   (predictive lead)")

    # ── 2. Liquidity-regime timing on SPY ────────────────────────────────────
    # Signal: is net liquidity expanding? (13wk change > 0).  Lag by the
    # publication delay, then use yesterday's reading to set today's position.
    expanding = (d_nl > 0).astype(float)
    signal = expanding.shift(PUB_LAG).shift(1).fillna(0.0)   # causal

    strat_ret = np.where(signal > 0, spy_ret, DAILY_RF)       # equities on / cash off
    strat = pd.Series(strat_ret, index=common, name="liq_timing").dropna()
    bh = spy_ret.reindex(strat.index)

    keys = ["CAGR", "Volatility", "Sharpe", "Sortino", "Max DD", "Calmar"]
    rows = [
        ["Net-liquidity timing", strat],
        ["SPY buy & hold", bh],
    ]
    table = [[name] + [f"{compute_metrics(r.dropna(), rf=RF).get(k, float('nan')):.2f}" for k in keys]
             for name, r in rows]
    print("\n" + "=" * 74)
    print("  LIQUIDITY-REGIME TIMING ON SPY  (long when net liquidity expanding)")
    print("=" * 74)
    print(tabulate(table, headers=["strategy"] + keys, tablefmt="github"))
    exposure = float((signal > 0).mean())
    print(f"\n  time invested in equities: {exposure:.0%}   "
          f"(signal lagged {PUB_LAG}d for publication delay)")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    eq_s = (1 + strat).cumprod(); eq_b = (1 + bh.fillna(0)).cumprod()
    ax1.plot(eq_s.index, eq_s, label="Net-liquidity timing", lw=1.3)
    ax1.plot(eq_b.index, eq_b, label="SPY buy & hold", lw=1.3, color="black", alpha=0.8)
    ax1.set_yscale("log"); ax1.set_ylabel("growth of $1 (log)")
    ax1.legend(); ax1.grid(alpha=0.3)
    ax1.set_title("Net-liquidity regime timing vs SPY")
    ax2.plot(nl.index, nl / 1000, color="tab:green", lw=1.0)
    ax2.set_ylabel("net liquidity ($T)"); ax2.grid(alpha=0.3)
    fig.tight_layout()
    out = RESULTS_DIR / "net_liquidity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    logger.info("Saved: %s", out)
    print("\n✓  Done.")


if __name__ == "__main__":
    main()
