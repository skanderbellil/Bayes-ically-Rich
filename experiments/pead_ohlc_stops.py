#!/usr/bin/env python3
"""Exact intraday-stop fills for PEAD market-neutral, using real daily OHLC.

With open/high/low we can model the stop precisely instead of bounding it:
  LONG, stop -S%:   if the day's OPEN gapped to <= -S%   -> fill at the open (gap-through)
                    elif the day's LOW touched <= -S%     -> fill at exactly -S% (intraday)
                    else continue; at the hold cap        -> exit at the close
  SHORT, stop +S%:  symmetric on OPEN/HIGH (loss when price rises).

This collapses the earlier Sharpe ~0.2-0.7 fill-uncertainty range into one defensible
number per stop. Mega+Large, t+1 entry, hold-to-next-earnings, net of Alpaca costs.
Requires data/raw/pead/<TICKER>__ohlc.csv (run experiments/refetch_ohlc.py first).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from posterioralpha.pead import paths, signals
from experiments.pead_dynamic_exit import _stats, COST

HOLD = 63


def load_ohlc(p):
    if not p.exists():
        return None
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    need = {"Open", "High", "Low", "Close"}
    if not need.issubset(df.columns):
        return None
    return df.sort_index()


def build_ohlc_paths():
    panel = pd.read_csv(paths.RESULTS_DIR / "signal_panel.csv")
    bucket = panel.drop_duplicates("ticker").set_index("ticker")["market_cap"].to_dict()
    names = [t for t, b in bucket.items() if b in ("Mega Cap", "Large Cap")]
    O, Lo, Hi, C, meta = [], [], [], [], []
    for t in names:
        e = signals.load_earnings_signal(paths.RAW_DIR / f"{t}__earnings.csv")
        oh = load_ohlc(paths.RAW_DIR / f"{t}__ohlc.csv")
        if e is None or oh is None:
            continue
        for ann, row in e.iterrows():
            sue = row.get("Surprise(%)")
            if pd.isna(sue):
                continue
            fut = oh[oh.index > ann]
            if len(fut) < 3:
                continue
            entry = fut["Close"].iloc[1]
            n = min(HOLD, len(fut) - 1)
            o = np.full(HOLD + 1, np.nan); lo = o.copy(); hi = o.copy(); c = o.copy()
            for d in range(1, n + 1):
                o[d] = np.log(fut["Open"].iloc[d] / entry) * 100
                lo[d] = np.log(fut["Low"].iloc[d] / entry) * 100
                hi[d] = np.log(fut["High"].iloc[d] / entry) * 100
                c[d] = np.log(fut["Close"].iloc[d] / entry) * 100
            O.append(o); Lo.append(lo); Hi.append(hi); C.append(c)
            meta.append((pd.Timestamp(ann).tz_localize(None), t, sue))
    M = pd.DataFrame(meta, columns=["ann", "ticker", "sue"])
    M["season"] = M["ann"].dt.to_period("Q"); M["year"] = M["ann"].dt.year
    return (np.array(O), np.array(Lo), np.array(Hi), np.array(C), M.reset_index(drop=True))


def exact_exit(o, lo, hi, c, sign, stop):
    """Signal-direction exit return (%) using real OHLC. Stop checked from day 2."""
    valid = np.where(~np.isnan(c))[0]
    if len(valid) < 2:
        return np.nan
    last = valid[-1]
    if stop is not None:
        for d in range(2, last + 1):
            if np.isnan(c[d]):
                break
            if sign > 0:                       # long: loss on the downside
                if o[d] <= -stop:              # gapped below at the open
                    return float(o[d])
                if lo[d] <= -stop:             # touched intraday -> fill at the level
                    return -stop
            else:                              # short: loss when price rises
                if o[d] >= stop:
                    return float(-o[d])
                if hi[d] >= stop:
                    return -stop
    return float(sign * c[last])               # exit at the hold cap


def mn_series(O, Lo, Hi, C, M, years, stop):
    out = {}
    for s, g in M[M["year"].isin(years)].groupby("season"):
        if len(g) < 10:
            continue
        hi_q, lo_q = g["sue"].quantile(0.8), g["sue"].quantile(0.2)
        def leg(sub, sign):
            v = [exact_exit(O[i], Lo[i], Hi[i], C[i], sign, stop) for i in sub.index]
            v = [x for x in v if np.isfinite(x)]
            return np.mean(v) if v else np.nan
        L = leg(g[g["sue"] >= hi_q], +1.0); S = leg(g[g["sue"] <= lo_q], -1.0)
        if np.isfinite(L) and np.isfinite(S):
            out[s] = (L + S) / 100 - COST
    return pd.Series(out).sort_index()


def main():
    O, Lo, Hi, C, M = build_ohlc_paths()
    print(f"OHLC paths: {len(M)} positions over {M['ticker'].nunique()} Mega+Large names "
          f"({M['ann'].min().date()}–{M['ann'].max().date()})\n")
    allyrs = list(range(2000, 2027)); oos = list(range(2014, 2027)); full_oos = list(range(2001, 2027))

    print("EXACT intraday-stop fills (real OHLC) — net of Alpaca cost:")
    print(f"  {'stop':>5}{'OOS≥14 Shrp':>12}{'OOS≥14 Ann':>11}{'FULL Shrp':>10}{'FULL Ann':>9}{'FULL MaxDD':>11}")
    for stop in [None, 3, 4, 5, 6, 8, 10]:
        b = mn_series(O, Lo, Hi, C, M, oos, stop)
        f = mn_series(O, Lo, Hi, C, M, full_oos, stop)
        lbl = "none" if stop is None else f"{stop}%"
        sb, sf = _stats(b.values), _stats(f.values)
        print(f"  {lbl:>5}{sb[1]:>12.2f}{sb[0]:>+11.1%}{sf[1]:>10.2f}{sf[0]:>+9.1%}{sf[2]:>+11.1%}")

    # how often does the stop gap through vs fill at the level? (3% long legs)
    gap = touch = none = 0
    for i in M.index:
        c = C[i]; valid = np.where(~np.isnan(c))[0]
        if len(valid) < 2:
            continue
        hit = False
        for d in range(2, valid[-1] + 1):
            if np.isnan(c[d]):
                break
            if O[i][d] <= -3:
                gap += 1; hit = True; break
            if Lo[i][d] <= -3:
                touch += 1; hit = True; break
        if not hit:
            none += 1
    tot = gap + touch + none
    print(f"\n3% long-leg stop fills: {touch/tot:.0%} intraday-at-level, {gap/tot:.0%} gapped-through-open, "
          f"{none/tot:.0%} never hit")
    print("(gap-through is the part that fills WORSE than -3%; intraday-at-level fills exactly -3%.)")


if __name__ == "__main__":
    main()
