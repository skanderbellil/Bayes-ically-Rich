#!/usr/bin/env python3
"""Rolling walk-forward PEAD market-neutral with an intraday-stop fill model.

Two corrections over the prior analysis:

1. INTRADAY-STOP FILL. A real intraday stop order on a liquid large-cap catches most
   -stop% breaches near the level (gradual intraday crossing), filling at ~-stop% plus a
   small slippage. Only a big single-day plunge (overnight/intraday gap) fills at the
   worse close. We only have daily closes, so we approximate: on the first day the
   cumulative return closes <= -stop, if that day's drop is <= the stop size it was a
   gradual cross -> fill at -(stop+SLIP); if the day's drop exceeds the stop size it
   gapped -> fill at that close. This sits between the optimistic exit-at-level and the
   pessimistic exit-at-close bounds.

2. ROLLING WALK-FORWARD. Train on a moving W-year window, SELECT the stop level that
   maximised train Sharpe, apply it to the next S-year out-of-sample block, roll forward.
   Concatenate the OOS blocks into a single walk-forward track. This tests whether the
   stop choice is learnable/stable, with no look-ahead.

Mega+Large, t+1 entry, hold-to-next-earnings, net of Alpaca costs. Plots vs SPY.
Requires data/raw/pead/ (run experiments/run_pead.py first).
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

HOLD = 63
SLIP = 0.15                         # bps-of-price slippage past an intraday stop (%)
STOP_GRID = [3, 4, 5, 6, 8, 10]
TRAIN_Y, TEST_Y = 8, 2             # rolling train / test window lengths (years)


def exit_ret(c, stop, mode):
    hit = np.where(c <= -stop)[0]
    if not len(hit):
        return c[-1]
    d = hit[0]
    if mode == "ideal":
        return -stop
    if mode == "close":
        return float(c[d])
    # intraday: gradual cross fills near the level; big single-day plunge gaps through
    prev = c[d - 1] if d > 0 else 0.0
    day_drop = prev - c[d]
    return -(stop + SLIP) if day_drop <= stop else float(c[d])


def mn_quarterly(P, M, stop, mode="intraday"):
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
                    v.append(exit_ret(c, stop, mode) if stop else c[-1])
            return np.nanmean(v) if v else np.nan
        L = leg(g[g["sue"] >= hi], +1.0); S = leg(g[g["sue"] <= lo], -1.0)
        if np.isfinite(L) and np.isfinite(S):
            out[s] = (L + S) / 100 - COST
    return pd.Series(out).sort_index()


def fetch_spy_quarterly():
    import yfinance as yf
    px = yf.download("SPY", start="1999-10-01", end="2026-07-01",
                     auto_adjust=True, progress=False)["Close"]
    q = np.exp(np.log(px).diff().dropna().resample("QE").sum()) - 1
    q.index = q.index.to_period("Q")
    return q.squeeze()


def main():
    P, M = build_paths(); M = M.reset_index(drop=True)
    # precompute the quarterly series for every stop on the grid (intraday fill)
    grid = {stop: mn_quarterly(P, M, stop, "intraday") for stop in STOP_GRID}
    years = sorted({q.year for q in grid[STOP_GRID[0]].index})

    # ---- rolling walk-forward: pick best-train stop, apply OOS ----
    wf, picks = {}, []
    y = years[0]
    while y + TRAIN_Y + TEST_Y <= years[-1] + 1:
        tr = lambda s: s[(s.index.year >= y) & (s.index.year < y + TRAIN_Y)]
        te = lambda s: s[(s.index.year >= y + TRAIN_Y) & (s.index.year < y + TRAIN_Y + TEST_Y)]
        best = max(STOP_GRID, key=lambda st: _stats(tr(grid[st]).values)[1]
                   if len(tr(grid[st])) >= 8 else -9)
        for q, r in te(grid[best]).items():
            wf[q] = r
        picks.append((y + TRAIN_Y, best))
        y += TEST_Y
    wf = pd.Series(wf).sort_index()

    spy = fetch_spy_quarterly()
    common = wf.index.intersection(spy.index)
    wf, spy = wf.loc[common], spy.loc[common]
    fixed3 = grid[3].loc[common]
    blend = 0.6 * spy + 0.4 * wf

    print(f"ROLLING WALK-FORWARD ({TRAIN_Y}y train → {TEST_Y}y OOS, stop re-selected each step)")
    print(f"  OOS span {common[0]}–{common[-1]} ({len(common)} quarters), intraday-stop fill\n")
    print("  Selected stop per step:  " + "  ".join(f"{yr}:{st}%" for yr, st in picks))
    print(f"\n  {'Series':<34}{'CAGR':>8}{'Sharpe':>8}{'MaxDD':>8}{'corrSPY':>9}")
    for name, s in [("Walk-forward (selected stop)", wf),
                    ("Fixed 3% stop (intraday)", fixed3),
                    ("60% SPY + 40% walk-forward", blend),
                    ("SPY buy & hold", spy)]:
        a, sh, dd, _ = _stats(s.values)
        print(f"  {name:<34}{a:>+8.1%}{sh:>8.2f}{dd:>+8.1%}{np.corrcoef(s.values, spy.values)[0,1]:>9.2f}")

    # fill-sensitivity of the fixed-3% book over the same window (bounds)
    print("\n  Fixed 3% stop fill sensitivity (same OOS span):")
    for mode in ("ideal", "intraday", "close"):
        s = mn_quarterly(P, M, 3, mode).loc[common]
        a, sh, dd, _ = _stats(s.values)
        print(f"    {mode:<9}: CAGR {a:+.1%}, Sharpe {sh:.2f}, MaxDD {dd:+.1%}")

    # ---- plot ----
    x = common.to_timestamp()
    series = {"SPY buy & hold": (spy, "#2ca02c", "--", 1.8),
              "Walk-forward (intraday stop)": (wf, "#1f77b4", "-", 2.4),
              "60% SPY + 40% WF": (blend, "#000000", "-", 2.2)}
    fig, ax = plt.subplots(2, 1, figsize=(13, 9), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    for name, (s, c, ls, lw) in series.items():
        a, sh, dd, _ = _stats(s.values)
        ax[0].plot(x, 100_000 * np.cumprod(1 + s.values), color=c, ls=ls, lw=lw,
                   label=f"{name}  (CAGR {a:+.1%}, Sharpe {sh:.2f}, MaxDD {dd:+.0%})")
        roll = s.rolling(12).apply(lambda w: np.sqrt(4) * w.mean() / w.std() if w.std() else 0)
        ax[1].plot(x, roll.values, color=c, ls=ls, lw=lw)
    ax[0].set_yscale("log"); ax[0].set_ylabel("Value ($, log)")
    ax[0].set_title(f"Rolling walk-forward PEAD market-neutral (intraday stop) vs SPY — "
                    f"OOS {common[0].year}–{common[-1].year}", fontweight="bold")
    ax[0].legend(fontsize=9, loc="upper left"); ax[0].grid(alpha=.3)
    ax[1].axhline(0, color="k", lw=.6); ax[1].axhline(1, color="grey", ls=":", lw=.8)
    ax[1].set_title("Rolling 3-year Sharpe", fontweight="bold")
    ax[1].set_ylabel("Sharpe"); ax[1].set_xlabel("Year"); ax[1].grid(alpha=.3)
    plt.tight_layout()
    out = paths.RESULTS_DIR / "walkforward_vs_spy.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nSaved plot: {out}")


if __name__ == "__main__":
    main()
