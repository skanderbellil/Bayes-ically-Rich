#!/usr/bin/env python3
"""3% stop PEAD market-neutral vs SPY — full-history OOS + rolling stability.

The strategy rule (rank on raw SUE, t+1 entry, 3% stop-from-entry, hold-to-next-earnings,
Mega+Large, dollar-neutral) has NO fitted parameters, so the entire 2000-2026 history is a
legitimate out-of-sample test — far more data than a single 2014 cutoff. We report the
full-period track, a rolling 12-quarter Sharpe for stability, and plot equity + drawdown
vs SPY (fetched live for the full window).

Two fills for the tight 3% stop:
  - idealised : exit at exactly -3% (optimistic; assumes the stop fills at the level)
  - realistic : exit at the CLOSE that first breaches -3% (captures gap-through)

Produces results/pead/stop3_vs_spy.png. Requires data/raw/pead/ (run run_pead.py first).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _bootstrap  # noqa: F401
from posterioralpha.pead import paths
from posterioralpha.backtest.alpaca_costs import AlpacaCosts
from experiments.pead_dynamic_exit import build_paths, _stats, COST

STOP, HOLD = 3.0, 63
AC = AlpacaCosts()


def _exit(c, realistic):
    hit = np.where(c <= -STOP)[0]
    if not len(hit):
        return c[-1]
    return float(c[hit[0]]) if realistic else -STOP    # close-at-breach vs exact level


def mn_quarterly(P, M, realistic=True):
    out = {}
    for s, g in M.groupby("season"):
        if len(g) < 10:
            continue
        hi, lo = g["sue"].quantile(0.8), g["sue"].quantile(0.2)
        def leg(sub, sign):
            v = []
            for i in sub.index:
                c = sign * P[i, 1:HOLD + 1]; c = c[~np.isnan(c)]
                if len(c):
                    v.append(_exit(c, realistic))
            return np.nanmean(v) if v else np.nan
        L = leg(g[g["sue"] >= hi], +1.0)
        S = leg(g[g["sue"] <= lo], -1.0)
        if np.isfinite(L) and np.isfinite(S):
            out[s] = (L + S) / 100 - COST
    return pd.Series(out).sort_index()


def mn_quarterly_nostop(P, M):
    out = {}
    for s, g in M.groupby("season"):
        if len(g) < 10:
            continue
        hi, lo = g["sue"].quantile(0.8), g["sue"].quantile(0.2)
        def leg(sub, sign):
            v = [sign * P[i, 1:HOLD + 1][~np.isnan(P[i, 1:HOLD + 1])][-1]
                 for i in sub.index if np.isfinite(P[i, 1:HOLD + 1]).any()]
            return np.nanmean(v) if v else np.nan
        L = leg(g[g["sue"] >= hi], +1.0); S = leg(g[g["sue"] <= lo], -1.0)
        if np.isfinite(L) and np.isfinite(S):
            out[s] = (L + S) / 100 - COST
    return pd.Series(out).sort_index()


def fetch_spy_quarterly():
    import yfinance as yf
    px = yf.download("SPY", start="1999-10-01", end="2026-07-01",
                     auto_adjust=True, progress=False)["Close"]
    r = np.log(px).diff().dropna()
    q = (np.exp(r.resample("QE").sum()) - 1)
    q.index = q.index.to_period("Q")
    return q.squeeze()


def main():
    P, M = build_paths(); M = M.reset_index(drop=True)
    mn_real = mn_quarterly(P, M, realistic=True)
    mn_ideal = mn_quarterly(P, M, realistic=False)
    mn_nostop = mn_quarterly_nostop(P, M)
    spy = fetch_spy_quarterly()
    common = mn_real.index.intersection(spy.index)
    mn_real, mn_ideal, mn_nostop, spy = (mn_real.loc[common], mn_ideal.loc[common],
                                         mn_nostop.loc[common], spy.loc[common])
    blend = 0.6 * spy + 0.4 * mn_real          # recommended unlevered deployment

    print(f"FULL-HISTORY OOS (fixed-rule, {common[0]}–{common[-1]}, {len(common)} quarters), 3% stop:")
    print(f"  {'Series':<34}{'CAGR':>8}{'Sharpe':>8}{'MaxDD':>8}{'corr SPY':>9}")
    for name, s in [("PEAD MN 3% (realistic fill)", mn_real),
                    ("PEAD MN 3% (idealised fill)", mn_ideal),
                    ("PEAD MN no-stop (fill-robust)", mn_nostop),
                    ("60% SPY + 40% MN (realistic)", blend),
                    ("SPY buy & hold", spy)]:
        a, sh, dd, _ = _stats(s.values)
        c = np.corrcoef(s.values, spy.values)[0, 1]
        print(f"  {name:<34}{a:>+8.1%}{sh:>8.2f}{dd:>+8.1%}{c:>9.2f}")

    # OOS by era (rolling/expanding sanity)
    print("\n  Rolling 5-year OOS Sharpe (PEAD MN 3%, realistic fill):")
    for lo in range(2001, 2026, 5):
        sub = mn_real[(mn_real.index.year >= lo) & (mn_real.index.year < lo + 5)]
        if len(sub) >= 8:
            print(f"    {lo}-{lo+4}: Sharpe {_stats(sub.values)[1]:.2f}  ({len(sub)} qtrs)")

    # ---- plot: equity, drawdown, rolling Sharpe ----
    def dd_curve(r):
        eq = np.cumprod(1 + np.asarray(r)); return (eq / np.maximum.accumulate(eq) - 1) * 100
    series = {"SPY buy & hold": (spy, "#2ca02c", "--", 1.8),
              "PEAD MN 3% (idealised fill)": (mn_ideal, "#9467bd", ":", 2.0),
              "PEAD MN 3% (realistic fill)": (mn_real, "#d62728", "-", 2.2),
              "PEAD MN no-stop (fill-robust)": (mn_nostop, "#1f77b4", "-", 2.2)}
    fig, ax = plt.subplots(3, 1, figsize=(13, 12), sharex=True,
                           gridspec_kw={"height_ratios": [3, 2, 2]})
    x = common.to_timestamp()
    for name, (s, c, ls, lw) in series.items():
        a, sh, dd, _ = _stats(s.values)
        ax[0].plot(x, 100_000 * np.cumprod(1 + s.values), color=c, ls=ls, lw=lw,
                   label=f"{name}  (CAGR {a:+.1%}, Sharpe {sh:.2f}, MaxDD {dd:+.0%})")
        ax[1].plot(x, dd_curve(s.values), color=c, ls=ls, lw=lw)
    ax[0].set_yscale("log"); ax[0].set_ylabel("Value ($, log)")
    ax[0].set_title(f"PEAD market-neutral (3% stop) vs SPY — full-history OOS "
                    f"{common[0].year}–{common[-1].year}, net of Alpaca costs", fontweight="bold")
    ax[0].legend(fontsize=9, loc="upper left"); ax[0].grid(alpha=.3)
    ax[1].set_title("Drawdown (underwater)", fontweight="bold"); ax[1].set_ylabel("%")
    ax[1].axhline(0, color="k", lw=.6); ax[1].grid(alpha=.3)
    # rolling 12-quarter (3yr) annualised Sharpe
    for name, (s, c, ls, lw) in series.items():
        roll = s.rolling(12).apply(lambda w: np.sqrt(4) * w.mean() / w.std() if w.std() else 0)
        ax[2].plot(x, roll.values, color=c, ls=ls, lw=lw)
    ax[2].axhline(0, color="k", lw=.6); ax[2].axhline(1, color="grey", ls=":", lw=.8)
    ax[2].set_title("Rolling 3-year Sharpe", fontweight="bold"); ax[2].set_ylabel("Sharpe")
    ax[2].set_xlabel("Year"); ax[2].grid(alpha=.3)
    plt.tight_layout()
    out = paths.RESULTS_DIR / "stop3_vs_spy.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nSaved plot: {out}")


if __name__ == "__main__":
    main()
