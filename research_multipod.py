#!/usr/bin/env python3
"""Multi-pod ETF strategy: dynamic selection + risk-parity weighting + vol targeting.

Combines the two findings that proved robust:
  - cross-asset diversification (ETFs across asset classes), and
  - dynamic second-moment control (vol targeting).

Each ETF is a "pod". Three causal (past-only) layers:
  1. SELECT: pod is ON only if price > its 10-month moving average (time-series
     momentum -- robust for asset allocation; not single-stock picking).
  2. WEIGHT: among ON pods, inverse-vol (risk parity) from trailing 12m vol.
  3. SIZE:  scale total exposure so ex-ante portfolio vol = target, capped at 1.0
     (no leverage). Off-pods and unused exposure sit in cash (~0).

Rebalanced monthly. Compared to SPY and static baselines, with half-sample robustness.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ASSETS = ["SPY", "TLT", "IEF", "GLD", "DBC", "VNQ", "LQD", "EEM"]
TARGET_VOL = 0.10          # annual
COST_BPS = 2.0             # ETF round-trip per unit turnover


def load():
    px = pd.read_csv("results/pead/etf_monthly.csv", index_col=0, parse_dates=True)
    cols = [c for c in ASSETS if c in px.columns and px[c].notna().sum() > 150]
    return px[cols]


def ann(r):  return (1 + r.mean()) ** 12 - 1
def shp(r):  return np.sqrt(12) * r.mean() / r.std() if r.std() else 0.0
def mdd(r):
    eq = np.cumprod(1 + r.dropna().values)
    return float(((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min())


def multipod(px, use_trend=True, use_riskparity=True, use_voltarget=True):
    ret = px.pct_change()
    sma10 = px.rolling(10).mean()
    idx = ret.index
    out, wprev = [], pd.Series(0.0, index=px.columns)
    turnover = []
    for i in range(13, len(idx)):
        t = idx[i]; tm1 = idx[i - 1]
        elig = [c for c in px.columns if not np.isnan(px[c].iloc[i - 1])
                and ret[c].iloc[i - 12:i].notna().all()]
        if not elig:
            out.append((t, 0.0)); continue
        # 1) selection (trend)
        on = [c for c in elig if (px[c].iloc[i - 1] > sma10[c].iloc[i - 1])] if use_trend else elig
        w = pd.Series(0.0, index=px.columns)
        if on:
            trail = ret.loc[idx[i - 12:i], on]
            if use_riskparity:
                iv = 1.0 / trail.std()
                w[on] = (iv / iv.sum()).values
            else:
                w[on] = 1.0 / len(on)
            # 3) portfolio vol target (no leverage)
            if use_voltarget:
                cov = trail.cov().values
                wv = w[on].values
                exante = np.sqrt(max(wv @ cov @ wv, 1e-12)) * np.sqrt(12)
                k = min(1.0, TARGET_VOL / exante)
                w = w * k
        # realize next month
        rp = float((w * ret.loc[t]).sum(skipna=True))
        tc = float(np.abs(w - wprev).sum()) * COST_BPS / 1e4
        turnover.append(float(np.abs(w - wprev).sum()))
        out.append((t, rp - tc)); wprev = w
    s = pd.Series(dict(out)).sort_index()
    s.attrs["turnover"] = np.mean(turnover) if turnover else 0
    return s


def main():
    px = load()
    ret = px.pct_change()
    spy = ret["SPY"].dropna()
    ew = ret.mean(axis=1).dropna()

    full = multipod(px)
    no_trend = multipod(px, use_trend=False)
    no_vt = multipod(px, use_voltarget=False)

    print("=" * 70)
    print("MULTI-POD ETF STRATEGY  (dynamic select + risk-parity + vol target)")
    print("=" * 70)
    print(f"assets={list(px.columns)}  target_vol={TARGET_VOL:.0%}  monthly rebalance\n")
    rows = {"SPY (buy & hold)": spy, "Equal-weight ETFs": ew,
            "Multi-pod: no trend filter": no_trend,
            "Multi-pod: no vol target": no_vt,
            "Multi-pod (full)": full}
    print(f"{'Strategy':30s} {'Annual':>8s} {'Sharpe':>7s} {'MaxDD':>7s} {'Turnov':>7s}")
    for n, r in rows.items():
        to = r.attrs.get("turnover", np.nan)
        print(f"{n:30s} {ann(r)*100:+7.1f}% {shp(r):7.2f} {mdd(r)*100:6.0f}% "
              f"{('%.2f'%to) if to==to else '   -':>7s}")

    # robustness: halves
    print("\n=== Robustness (halves), Multi-pod full vs SPY ===")
    f = full.dropna(); n = len(f)
    for nm, seg in [("H1", f.iloc[:n // 2]), ("H2", f.iloc[n // 2:])]:
        sp = spy.reindex(seg.index).dropna()
        print(f"  {nm}: pod Sharpe {shp(seg):.2f} (ann {ann(seg)*100:+.1f}%, DD {mdd(seg)*100:.0f}%) "
              f"| SPY Sharpe {shp(sp):.2f} (DD {mdd(sp)*100:.0f}%)")

    # plot
    fig, ax = plt.subplots(figsize=(13, 7))
    for n, r, c, lw in [("SPY", spy, "#000", 1.5), ("Equal-weight", ew, "#888", 1.3),
                        ("Multi-pod (full)", full, "#d62728", 2.2)]:
        r = r.dropna()
        ax.plot(r.index, 1e5 * np.cumprod(1 + r.values), color=c, lw=lw,
                label=f"{n}: {ann(r)*100:+.1f}%/yr, Sharpe {shp(r):.2f}, DD {mdd(r)*100:.0f}%")
    ax.set_yscale("log"); ax.grid(True, alpha=0.3, which="both")
    ax.set_ylabel("Growth of $100k (log)", fontweight="bold")
    ax.set_title("Multi-pod ETF: dynamic selection + risk parity + vol target", fontweight="bold")
    ax.legend(fontsize=10)
    out = Path("results/pead/multipod.png"); fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
