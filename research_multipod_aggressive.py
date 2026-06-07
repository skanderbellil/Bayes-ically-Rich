#!/usr/bin/env python3
"""Aggressive multi-pod: momentum-tilt + leveraged ETFs (cheap embedded leverage).

The attempt to beat SPY on BOTH return and Sharpe. Adds leveraged equity/bond ETFs
(SSO/UPRO/QLD/TQQQ/UBT/TMF/...) so the book can lever cheaply via the products (no
margin account), weights by trailing momentum (return-seeking), and uses the trend
filter + vol target to manage decay/crash risk.

Honest verdict (see REVIEW.md): beats SPY on both GROSS and IN-SAMPLE, but the edge
does NOT survive (a) leveraged-ETF expense ratios + realistic spreads [Sharpe -> 0.79
vs SPY 0.78], (b) short-term tax in a taxable account [+12% -> +3.7%], or (c) the
recent half [2015-26 bull: SPY wins]. The durable benefit is crash protection
(GFC -22% vs SPY -51%). Useful for crisis defense / tax-advantaged accounts, not a
clean return+Sharpe win.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings("ignore")
import research_multipod_wide as mw

LB = 12
CASH = ["BIL", "SHY", "MBB"]


def load():
    base = pd.read_csv("results/pead/etf_wide_monthly.csv", index_col=0, parse_dates=True)
    lev = pd.read_csv("results/pead/etf_lev_monthly.csv", index_col=0, parse_dates=True)
    base = base[[c for c in base.columns if c not in CASH]]
    comb = base.join(lev, how="outer").sort_index()
    return comb, base["SPY"].pct_change().dropna(), set(lev.columns)


def run(px, levset, target_vol=0.16, expense=0.0095, cost_bps=5.0, st_tax=0.0):
    ret = px.pct_change(); sma = px.rolling(10).mean(); idx = ret.index
    out, wprev = [], pd.Series(0.0, index=px.columns)
    for i in range(LB + 1, len(idx)):
        t = idx[i]
        elig = [c for c in px.columns if not np.isnan(px[c].iloc[i - 1]) and ret[c].iloc[i - LB:i].notna().all()]
        on = [c for c in elig if px[c].iloc[i - 1] > sma[c].iloc[i - 1]]
        w = pd.Series(0.0, index=px.columns)
        if on:
            R = ret.loc[idx[i - LB:i], on]
            raw = (px[on].iloc[i - 1] / px[on].iloc[i - LB] - 1).clip(lower=0)
            raw = raw.replace([np.inf, -np.inf], 0).fillna(0)
            ww = raw / raw.sum() if raw.sum() > 0 else pd.Series(1 / len(on), index=on)
            S = np.cov(R.values, rowvar=False) if len(on) > 1 else np.atleast_2d(R.var().values)
            exante = np.sqrt(max(ww.values @ S @ ww.values, 1e-12)) * np.sqrt(12)
            w[on] = ww.values * min(1.0, target_vol / exante)
        gross = float((w * ret.loc[t]).sum(skipna=True))
        lev_w = float(w[[c for c in w.index if c in levset]].sum())
        net = gross - lev_w * expense / 12 - float(np.abs(w - wprev).sum()) * cost_bps / 1e4
        if st_tax > 0 and net > 0:
            net *= (1 - st_tax)
        out.append((t, net)); wprev = w
    return pd.Series(dict(out)).sort_index()


def main():
    comb, spy, levset = load()
    st = lambda r: (mw.ann(r), mw.shp(r), mw.mdd(r))
    base = run(comb, levset, 0.16, expense=0.0, cost_bps=2.0)
    net = run(comb, levset, 0.16, expense=0.0095, cost_bps=5.0)
    tax = run(comb, levset, 0.16, expense=0.0095, cost_bps=5.0, st_tax=0.30)
    sp = spy.reindex(base.index).dropna()
    print("Aggressive multi-pod (momentum-tilt + leveraged ETFs), target 16% vol\n")
    print(f"{'Accounting':28s} {'Annual':>8s} {'Sharpe':>7s} {'MaxDD':>7s}")
    for n, r in [("SPY buy & hold", sp), ("Strategy gross (2bps)", base),
                 ("+ ETF fees + 5bps spread", net), ("+ 30% short-term tax", tax)]:
        a, s, d = st(r)
        print(f"{n:28s} {a*100:+7.1f}% {s:7.2f} {d*100:6.0f}%")
    print("\nVerdict: gross beats SPY on both; net-of-fees ties (Sharpe 0.79 vs 0.78);")
    print("tax destroys it; only wins 2005-15 half. Durable value = crash protection.")


if __name__ == "__main__":
    main()
