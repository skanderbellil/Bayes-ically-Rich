#!/usr/bin/env python3
"""Wide multi-pod ETF strategy + covariance-estimator comparison.

51-ETF roster spanning US factors/sectors, international, fixed income, commodities.
Dynamic selection (10-month trend) + portfolio vol target (no leverage), monthly.
Compares three ways to set weights among selected pods:
  A. Inverse-vol  (diagonal; ignores correlations)
  B. Min-variance, SAMPLE covariance      (uses correlations; singular when n_on>12)
  C. Min-variance, LEDOIT-WOLF shrinkage  (Bayesian-shrunk covariance)

With up to 51 pods and only 12 monthly obs, the sample covariance is rank-deficient,
so this is the natural test of whether shrinkage earns its keep.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TARGET_VOL = 0.10
COST_BPS = 2.0
LOOKBACK = 12

try:
    from sklearn.covariance import LedoitWolf
    HAVE_SK = True
except Exception:
    HAVE_SK = False


def ann(r):  return (1 + r.mean()) ** 12 - 1
def shp(r):  return np.sqrt(12) * r.mean() / r.std() if r.std() else 0.0
def mdd(r):
    eq = np.cumprod(1 + r.dropna().values)
    return float(((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min())


def ledoit_wolf_cov(X):
    """Ledoit-Wolf shrinkage covariance (monthly). X: T x N returns."""
    if HAVE_SK and X.shape[0] > 2:
        try:
            return LedoitWolf().fit(X).covariance_
        except Exception:
            pass
    # fallback: shrink sample toward diagonal with fixed intensity
    S = np.cov(X, rowvar=False)
    F = np.diag(np.diag(S))
    return 0.5 * S + 0.5 * F


def weights(method, R):
    """Long-only weights from a T x N return block R (DataFrame)."""
    cols = R.columns
    n = len(cols)
    if n == 1:
        return np.array([1.0]), np.atleast_2d(R.var().values)
    if method == "invvol":
        iv = 1.0 / R.std()
        w = iv / iv.sum()
        return w.reindex(cols).fillna(0).values, np.diag(R.var().values)
    # covariance methods
    if method == "sample":
        S = np.atleast_2d(np.cov(R.values, rowvar=False))
    elif method == "lw":
        S = np.atleast_2d(ledoit_wolf_cov(R.values))
    Sinv = np.linalg.pinv(S)
    ones = np.ones(len(cols))
    w = Sinv @ ones
    w = np.clip(w, 0, None)          # long-only
    w = w / w.sum() if w.sum() > 0 else np.ones(len(cols)) / len(cols)
    return w, S


def run(px, method):
    ret = px.pct_change()
    sma = px.rolling(10).mean()
    idx = ret.index
    out, wprev, turns = [], pd.Series(0.0, index=px.columns), []
    for i in range(LOOKBACK + 1, len(idx)):
        t = idx[i]
        elig = [c for c in px.columns
                if not np.isnan(px[c].iloc[i - 1]) and ret[c].iloc[i - LOOKBACK:i].notna().all()]
        on = [c for c in elig if px[c].iloc[i - 1] > sma[c].iloc[i - 1]]
        w = pd.Series(0.0, index=px.columns)
        if on:
            R = ret.loc[idx[i - LOOKBACK:i], on]
            wv, S = weights(method, R)
            wser = pd.Series(wv, index=on)
            exante = np.sqrt(max(wser.values @ S @ wser.values, 1e-12)) * np.sqrt(12)
            k = min(1.0, TARGET_VOL / exante)
            w[on] = wser.values * k
        rp = float((w * ret.loc[t]).sum(skipna=True))
        tc = float(np.abs(w - wprev).sum()) * COST_BPS / 1e4
        turns.append(float(np.abs(w - wprev).sum()))
        out.append((t, rp - tc)); wprev = w
    s = pd.Series(dict(out)).sort_index()
    s.attrs["turnover"] = float(np.mean(turns))
    return s


CASH_PROXIES = ["BIL", "SHY", "MBB"]  # near-cash; excluded so inverse-vol can't hide in T-bills


def main():
    px = pd.read_csv("results/pead/etf_wide_monthly.csv", index_col=0, parse_dates=True)
    px = px[[c for c in px.columns if c not in CASH_PROXIES]]
    print(f"Fair roster: {px.shape[1]} ETFs (cash proxies {CASH_PROXIES} excluded), "
          f"{px.index.min().date()}..{px.index.max().date()}")
    print(f"sklearn LedoitWolf available: {HAVE_SK}\n")
    spy = px["SPY"].pct_change().dropna()

    res = {m: run(px, m) for m in ["invvol", "sample", "lw"]}
    names = {"invvol": "A. Inverse-vol (diagonal)",
             "sample": "B. Min-var, sample cov",
             "lw": "C. Min-var, Ledoit-Wolf"}
    print(f"{'Method':30s} {'Annual':>8s} {'Sharpe':>7s} {'MaxDD':>7s} {'Turn':>6s}")
    sp = spy.reindex(res['invvol'].index).dropna()
    print(f"{'SPY (buy & hold)':30s} {ann(sp)*100:+7.1f}% {shp(sp):7.2f} {mdd(sp)*100:6.0f}%")
    for m in ["invvol", "sample", "lw"]:
        r = res[m]
        print(f"{names[m]:30s} {ann(r)*100:+7.1f}% {shp(r):7.2f} {mdd(r)*100:6.0f}% {r.attrs['turnover']:6.2f}")

    # robustness halves
    print("\nHalf-sample Sharpe (pod / SPY):")
    for m in ["invvol", "sample", "lw"]:
        r = res[m].dropna(); n = len(r)
        h1, h2 = r.iloc[:n // 2], r.iloc[n // 2:]
        s1 = spy.reindex(h1.index).dropna(); s2 = spy.reindex(h2.index).dropna()
        print(f"  {names[m]:30s} H1 {shp(h1):.2f}/{shp(s1):.2f}   H2 {shp(h2):.2f}/{shp(s2):.2f}")

    # plot
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.plot(sp.index, 1e5*np.cumprod(1+sp.values), color="#000", lw=1.4,
            label=f"SPY: Sharpe {shp(sp):.2f}, DD {mdd(sp)*100:.0f}%")
    for m, c in [("invvol", "#1f77b4"), ("sample", "#ff7f0e"), ("lw", "#d62728")]:
        r = res[m].dropna()
        ax.plot(r.index, 1e5*np.cumprod(1+r.values), color=c, lw=2,
                label=f"{names[m]}: Sharpe {shp(r):.2f}, DD {mdd(r)*100:.0f}%")
    ax.set_yscale("log"); ax.grid(True, alpha=0.3, which="both")
    ax.set_ylabel("Growth of $100k (log)", fontweight="bold")
    ax.set_title("Wide multi-pod (51 ETFs): covariance-estimator comparison", fontweight="bold")
    ax.legend(fontsize=9)
    out = Path("results/pead/multipod_wide.png"); fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
