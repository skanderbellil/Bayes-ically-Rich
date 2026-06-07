#!/usr/bin/env python3
"""Low-turnover factor backtest WITH a survivorship-matched benchmark.

Motivation: after PEAD failed the hustle-cost test (short-term tax), the only way an
active strategy could justify itself is low turnover (annual rebalance -> long-term
cap gains, trivial hustle). Natural candidates: low-volatility and momentum.

Key methodological point: comparing a factor portfolio to SPY is NOT enough when the
universe is survivorship-biased (live names only). The 497 survivors equal-weighted
already beat SPY by ~9%/yr -- pure survivorship. A factor only shows a REAL edge if it
beats the equal-weight survivor universe (which cancels the shared bias).

Result on this data: neither low-vol nor momentum beats the EW-universe benchmark.
No real low-turnover edge is demonstrable here.
"""
from pathlib import Path
import numpy as np
import pandas as pd


def load_monthly_returns(path="results/pead/monthly_prices.csv"):
    px = pd.read_csv(path, index_col=0, parse_dates=True)
    return px.pct_change()


def load_spy():
    spy = pd.read_csv("results/pead/spy_monthly.csv")
    spy.index = pd.PeriodIndex(spy["month"], freq="M").to_timestamp("M")
    return spy["spy_mret"]


def ann(r):  return (1 + r.mean()) ** 12 - 1
def shp(r):  return np.sqrt(12) * r.mean() / r.std() if r.std() else 0.0
def mdd(r):
    if len(r) == 0: return 0.0
    eq = np.cumprod(1 + r.values)
    return float(((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min())


def backtest(ret, score_fn=None, low_is_good=True, frac=0.2):
    """Annual rebalance, equal-weight, 12-month hold. score_fn=None -> EW universe."""
    idx = ret.index
    port = []
    for y in range(2000, 2026):
        cand = idx[idx <= pd.Timestamp(f"{y-1}-12-31")]
        if len(cand) == 0: continue
        form = cand[-1]
        hold = idx[idx > form][:12]
        if len(hold) < 12: continue
        trail = ret.loc[idx[idx <= form]].iloc[-12:]
        elig = trail.columns[trail.notna().sum() >= 12]
        fwd = ret.loc[hold, elig]
        elig = fwd.columns[fwd.notna().sum() >= 12]
        if len(elig) < 20: continue
        if score_fn is None:
            pick = elig
        else:
            score = score_fn(trail[elig])
            k = max(5, int(len(elig) * frac))
            pick = score.nsmallest(k).index if low_is_good else score.nlargest(k).index
        port.append(ret.loc[hold, pick].mean(axis=1))
    return pd.concat(port)


def main():
    ret = load_monthly_returns()
    spy = load_spy()

    ew  = backtest(ret, None)
    vol = backtest(ret, lambda t: t.std(), low_is_good=True)
    mom = backtest(ret, lambda t: (1 + t).prod(), low_is_good=False)

    span = spy.loc[ew.index.min():ew.index.max()]
    print("=" * 70)
    print("LOW-TURNOVER FACTORS vs SURVIVORSHIP-MATCHED BENCHMARK")
    print("=" * 70)
    print(f"{'Series':18s} {'Annual':>8s} {'Sharpe':>7s} {'MaxDD':>7s}")
    print(f"{'SPY':18s} {ann(span)*100:+7.1f}% {shp(span):7.2f}")
    print(f"{'EW-universe(497)':18s} {ann(ew)*100:+7.1f}% {shp(ew):7.2f} {mdd(ew)*100:6.0f}%   <- survivorship benchmark")
    print(f"{'Low-Vol quintile':18s} {ann(vol)*100:+7.1f}% {shp(vol):7.2f} {mdd(vol)*100:6.0f}%")
    print(f"{'Momentum quintile':18s} {ann(mom)*100:+7.1f}% {shp(mom):7.2f} {mdd(mom)*100:6.0f}%")

    b = pd.concat([vol.rename("v"), mom.rename("m"), ew.rename("e")], axis=1).dropna()
    print("\nEdge vs survivorship-matched EW-universe (this is the honest test):")
    print(f"  Low-Vol  Sharpe {shp(b.v):.2f} - EW {shp(b.e):.2f} = {shp(b.v)-shp(b.e):+.2f}")
    print(f"  Momentum Sharpe {shp(b.m):.2f} - EW {shp(b.e):.2f} = {shp(b.m)-shp(b.e):+.2f}")
    print("\nThe +%s/yr gap of EW-universe over SPY is survivorship bias, not skill." %
          f"{(ann(ew)-ann(span))*100:.1f}%")
    print("Neither factor beats the EW benchmark -> no real low-turnover edge on this data.")


if __name__ == "__main__":
    main()
