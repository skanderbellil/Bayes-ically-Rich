#!/usr/bin/env python3
"""Wide multi-pod with (a) idle cash earning the risk-free rate and (b) small dynamic
leverage via the vol target. Addresses two fair critiques:

  - "Why exclude cash?"  Cash stays out of the OPTIMIZER (so inverse-vol can't hide in
    T-bills), but idle capital (when de-risked) now EARNS rf instead of 0% -- holding
    cash is a legitimate, paid decision. rf proxied by BIL (T-bill ETF).
  - "Allow small leverage." The vol-target cap is raised above 1.0, so the book levers
    UP only when forecast vol is low and de-levers when it spikes (dynamic leverage).
    Borrowing costs rf + spread; idle cash earns rf. Long-only Ledoit-Wolf weights.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import research_multipod_wide as mw

CASH_PROXIES = ["BIL", "SHY", "MBB"]
MARGIN_SPREAD = 0.015     # broker margin = rf + 1.5% (IBKR-like); Alpaca is higher (see note)
TARGET_VOL = 0.10
COST_BPS = 2.0
LOOKBACK = 12


def run_levered(px, rf, method="lw", lev_cap=1.0, rf_on_cash=True, margin_spread=MARGIN_SPREAD):
    ret = px.pct_change()
    sma = px.rolling(10).mean()
    idx = ret.index
    out, wprev, turns = [], pd.Series(0.0, index=px.columns), []
    for i in range(LOOKBACK + 1, len(idx)):
        t = idx[i]
        rf_t = rf.get(t, 0.0) if rf_on_cash else 0.0
        elig = [c for c in px.columns
                if not np.isnan(px[c].iloc[i - 1]) and ret[c].iloc[i - LOOKBACK:i].notna().all()]
        on = [c for c in elig if px[c].iloc[i - 1] > sma[c].iloc[i - 1]]
        w = pd.Series(0.0, index=px.columns); k = 0.0
        if on:
            R = ret.loc[idx[i - LOOKBACK:i], on]
            wv, S = mw.weights(method, R)
            wser = pd.Series(wv, index=on)
            exante = np.sqrt(max(wser.values @ S @ wser.values, 1e-12)) * np.sqrt(12)
            k = min(lev_cap, TARGET_VOL / exante)
            w[on] = wser.values * k
        rr = float((w * ret.loc[t]).sum(skipna=True))        # return on deployed risk
        if k <= 1.0:
            port = rr + (1.0 - k) * rf_t                      # idle cash earns rf
        else:
            port = rr - (k - 1.0) * (rf_t + margin_spread / 12)  # borrow at rf+spread
        tc = float(np.abs(w - wprev).sum()) * COST_BPS / 1e4
        turns.append(float(np.abs(w - wprev).sum()))
        out.append((t, port - tc)); wprev = w
    s = pd.Series(dict(out)).sort_index()
    s.attrs["turnover"] = float(np.mean(turns))
    return s


def main():
    raw = pd.read_csv("results/pead/etf_wide_monthly.csv", index_col=0, parse_dates=True)
    rf = raw["BIL"].pct_change().reindex(raw.index).fillna(0.0) if "BIL" in raw else pd.Series(0.0, index=raw.index)
    px = raw[[c for c in raw.columns if c not in CASH_PROXIES]]
    spy = raw["SPY"].pct_change().dropna()

    idx0 = run_levered(px, rf, "lw", 1.0).index
    sp = spy.reindex(idx0).dropna()
    rf_ann = (1 + rf.reindex(idx0).fillna(0)).prod() ** (12 / len(idx0)) - 1
    print(f"Fair roster {px.shape[1]} ETFs; idle cash earns rf (~{rf_ann*100:.1f}%/yr avg); "
          f"margin = rf+{MARGIN_SPREAD*100:.1f}%\n")

    print(f"{'Config':36s} {'Annual':>8s} {'Sharpe':>7s} {'MaxDD':>7s} {'AvgLev':>7s}")
    print(f"{'SPY (buy & hold)':36s} {mw.ann(sp)*100:+7.1f}% {mw.shp(sp):7.2f} {mw.mdd(sp)*100:6.0f}%")

    # 1) show the rf-on-cash effect at cap 1.0
    no_rf = run_levered(px, rf, "lw", 1.0, rf_on_cash=False)
    yes_rf = run_levered(px, rf, "lw", 1.0, rf_on_cash=True)
    print(f"{'LW cap1.0, idle cash=0% (old)':36s} {mw.ann(no_rf)*100:+7.1f}% {mw.shp(no_rf):7.2f} {mw.mdd(no_rf)*100:6.0f}%")
    print(f"{'LW cap1.0, idle cash=rf (fixed)':36s} {mw.ann(yes_rf)*100:+7.1f}% {mw.shp(yes_rf):7.2f} {mw.mdd(yes_rf)*100:6.0f}%")

    # 2) leverage sweep (idle cash earns rf)
    curves = {"SPY": sp, "LW cap1.0": yes_rf}
    for cap in [1.25, 1.5, 2.0]:
        r = run_levered(px, rf, "lw", cap)
        avglev = None
        print(f"{f'LW dynamic leverage cap {cap:.2f}x':36s} {mw.ann(r)*100:+7.1f}% {mw.shp(r):7.2f} {mw.mdd(r)*100:6.0f}%")
        curves[f"LW cap{cap:.2f}x"] = r

    # robustness for the 1.5x config
    r15 = run_levered(px, rf, "lw", 1.5); n = len(r15)
    h1, h2 = r15.iloc[:n//2], r15.iloc[n//2:]
    print(f"\nLW cap1.5x half-sample Sharpe: H1 {mw.shp(h1):.2f} / SPY {mw.shp(spy.reindex(h1.index).dropna()):.2f}"
          f"   H2 {mw.shp(h2):.2f} / SPY {mw.shp(spy.reindex(h2.index).dropna()):.2f}")

    # plot
    fig, ax = plt.subplots(figsize=(13, 7))
    colors = {"SPY": "#000", "LW cap1.0": "#1f77b4", "LW cap1.25x": "#2ca02c",
              "LW cap1.50x": "#ff7f0e", "LW cap2.00x": "#d62728"}
    for n_, r in curves.items():
        r = r.dropna()
        ax.plot(r.index, 1e5*np.cumprod(1+r.values), lw=2, color=colors.get(n_, "#777"),
                label=f"{n_}: {mw.ann(r)*100:+.1f}%/yr, Sharpe {mw.shp(r):.2f}, DD {mw.mdd(r)*100:.0f}%")
    ax.set_yscale("log"); ax.grid(True, alpha=0.3, which="both")
    ax.set_ylabel("Growth of $100k (log)", fontweight="bold")
    ax.set_title("Wide multi-pod: idle cash earns rf + small dynamic leverage (Ledoit-Wolf)",
                 fontweight="bold")
    ax.legend(fontsize=9)
    out = Path("results/pead/multipod_levered.png"); fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")
    print("\nNote: margin = rf+1.5% (IBKR-like). Alpaca's retail margin (~5.75-8% flat) is")
    print("higher; at that rate the leverage benefit shrinks materially (sensitivity below).")
    # Alpaca-rate sensitivity for cap 1.5
    r15_alp = run_levered(px, rf, "lw", 1.5, margin_spread=0.06)
    print(f"  LW cap1.5x at Alpaca ~rf+6% margin: {mw.ann(r15_alp)*100:+.1f}%/yr, Sharpe {mw.shp(r15_alp):.2f}, DD {mw.mdd(r15_alp)*100:.0f}%")


if __name__ == "__main__":
    main()
