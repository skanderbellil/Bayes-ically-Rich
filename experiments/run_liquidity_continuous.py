#!/usr/bin/env python3
"""
Continuous liquidity×VIX-TS×credit vote — bigger universe + rolling OOS.

Upgrades the binary three-way vote (run_liquidity_predictive.py) along three
axes requested of a robust strategy:

  1. CONTINUOUS, still fit-free.  Each layer's strength is a "soft sign" of its
     raw value:  strength = clip(0.5 + raw / (4·MAD₇₅₆), 0, 1) — preserves the
     natural zero threshold (raw 0 → 0.5), saturates at ±2 MADs, and never
     optimises on returns.  Exposure = mean of the three strengths.
       liq_raw    = 6-month Δ net-liquidity        (publication-lagged)
       vix_raw    = VIX3M/VIX − 1                  (contango degree)
       credit_raw = HYG−IEF 21-day relative return (credit appetite)
  2. BIGGER UNIVERSE.  The same overlay is applied to SPY and to an equal-
     weight cyclical sector basket (XLY/XLK/XLF/XLE/XLI) — the sleeve the
     sensitivity map says liquidity hits hardest.  Basket chosen economically,
     not fitted.
  3. ROLLING OOS.  Rolling 3-year Sharpe of overlay vs buy & hold (win rate =
     fraction of windows where the overlay's Sharpe is higher) and a per-year
     return table — consistency across sub-periods, not one full-sample number.

    python experiments/run_liquidity_continuous.py
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
VIX_CSV = _bootstrap.ROOT / "datasets" / "vix_term_structure.csv"

RF = 0.04
DAILY_RF = RF / 252
H = 126
PUB_LAG = 5
CREDIT_WIN = 21
MAD_WIN = 756            # trailing window for the MAD scale (≈3y)
ROLL = 756               # rolling-OOS window (≈3y)
CYCLICALS = ["XLY", "XLK", "XLF", "XLE", "XLI"]


def soft_sign(raw: pd.Series, mad_win: int = MAD_WIN) -> pd.Series:
    """clip(0.5 + raw/(4·MAD), 0, 1): continuous strength, no return fitting."""
    mad = raw.rolling(mad_win, min_periods=252).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=True)
    return (0.5 + raw / (4.0 * mad.replace(0, np.nan))).clip(0.0, 1.0)


def main() -> None:
    nl = load_net_liquidity()["net_liquidity"]
    px = load_etf_universe_prices()
    vix = pd.read_csv(VIX_CSV, parse_dates=["Date"], index_col="Date").sort_index()
    spy = px["SPY"]
    idx = (nl.dropna().index.intersection(spy.dropna().index)
           .intersection(vix.dropna().index)
           .intersection(px[CYCLICALS + ["HYG", "IEF"]].dropna().index))
    nl, spy, vix = nl.reindex(idx), spy.reindex(idx), vix.reindex(idx)
    spy_ret = spy.pct_change()
    cyc_ret = px[CYCLICALS].reindex(idx).pct_change().mean(axis=1)

    # ── Raw signals → continuous strengths (all causal) ──────────────────────
    liq_raw = nl.diff(H).shift(PUB_LAG)
    vix_raw = vix["VIX3M"] / vix["VIX"] - 1.0
    credit_raw = (px["HYG"].reindex(idx).pct_change(CREDIT_WIN)
                  - px["IEF"].reindex(idx).pct_change(CREDIT_WIN))

    strengths = pd.DataFrame({
        "liq": soft_sign(liq_raw),
        "vix_ts": soft_sign(vix_raw),
        "credit": soft_sign(credit_raw),
    }).shift(1)                              # trade on yesterday's reading
    cont = strengths.mean(axis=1)

    # Binary vote for comparison (same layers, hard thresholds)
    binary = (pd.DataFrame({
        "liq": (liq_raw > 0), "vix_ts": (vix_raw > 0), "credit": (credit_raw > 0),
    }).astype(float).shift(1).mean(axis=1))

    def overlay(expo, under_ret):
        e = expo.fillna(0.0)
        return pd.Series(e * under_ret + (1 - e) * DAILY_RF, index=idx)

    rows = [
        ["SPY buy & hold", spy_ret, None],
        ["Binary vote on SPY", overlay(binary, spy_ret), binary],
        ["Continuous vote on SPY", overlay(cont, spy_ret), cont],
        ["Cyclical basket buy & hold", cyc_ret, None],
        ["Continuous vote on cyclicals", overlay(cont, cyc_ret), cont],
    ]

    print("\n" + "=" * 80)
    print(f"  CONTINUOUS LIQUIDITY VOTE  ·  {idx.min().date()} → {idx.max().date()}")
    print(f"  layers: net-liquidity Δ6m · VIX3M/VIX−1 · HYG−IEF {CREDIT_WIN}d  "
          f"(soft-sign, fit-free)")
    print("=" * 80)
    keys = ["CAGR", "Volatility", "Sharpe", "Sortino", "Max DD", "Calmar"]
    table = [[n] + [f"{compute_metrics(r.dropna(), rf=RF).get(k, float('nan')):.2f}" for k in keys]
             for n, r, _ in rows]
    print(tabulate(table, headers=["strategy"] + keys, tablefmt="github"))
    print(f"\n  avg exposure — binary {float(binary.mean()):.2f} | "
          f"continuous {float(cont.mean()):.2f}")

    # ── Rolling OOS: 3y Sharpe of overlay vs its underlying ──────────────────
    def roll_sharpe(r):
        ex = r - DAILY_RF
        return (ex.rolling(ROLL).mean() / ex.rolling(ROLL).std()) * np.sqrt(252)

    print("\n" + "=" * 80)
    print(f"  ROLLING OOS  ·  {ROLL}d (~3y) windows, step daily")
    print("=" * 80)
    pairs = [
        ("Continuous vote vs SPY", overlay(cont, spy_ret), spy_ret),
        ("Binary vote vs SPY", overlay(binary, spy_ret), spy_ret),
        ("Continuous vote vs cyclicals", overlay(cont, cyc_ret), cyc_ret),
    ]
    for name, s, b in pairs:
        ds = roll_sharpe(s) - roll_sharpe(b)
        ds = ds.dropna()
        print(f"  {name:32s} win rate {float((ds > 0).mean()):.0%} of windows   "
              f"median ΔSharpe {float(ds.median()):+.2f}")

    # per-year returns
    yr = pd.DataFrame({
        "cont vote SPY": overlay(cont, spy_ret),
        "SPY": spy_ret,
    }).dropna()
    ann = yr.groupby(yr.index.year).apply(lambda d: (1 + d).prod() - 1)
    ann["edge"] = ann["cont vote SPY"] - ann["SPY"]
    print("\nPer-year returns (continuous vote vs SPY):")
    print(tabulate((ann * 100).round(1), headers=["year", "vote %", "SPY %", "edge %"],
                   tablefmt="github"))

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    for n, r, _ in rows:
        eq = (1 + r.reindex(idx).fillna(0)).cumprod()
        ax1.plot(eq.index, eq.values, lw=1.5 if "buy" in n else 1.2,
                 color="black" if n.startswith("SPY") else None,
                 alpha=0.85, label=n)
    ax1.set_yscale("log"); ax1.set_ylabel("growth of $1 (log)")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    ax1.set_title("Continuous fit-free liquidity vote — SPY & cyclical basket")
    d1 = (roll_sharpe(overlay(cont, spy_ret)) - roll_sharpe(spy_ret))
    ax2.plot(d1.index, d1.values, lw=1.0, color="tab:purple")
    ax2.axhline(0, color="grey", lw=0.6)
    ax2.set_ylabel(f"Δ rolling {ROLL}d Sharpe\n(vote − SPY)")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    out = RESULTS_DIR / "liquidity_continuous.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    logger.info("Saved: %s", out)
    print("\n✓  Done.")


if __name__ == "__main__":
    main()
