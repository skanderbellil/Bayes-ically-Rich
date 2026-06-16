#!/usr/bin/env python3
"""Stop-check FREQUENCY sweep — does smoothing the stop (weekly/biweekly) help?

A 3% stop over a ~63-day hold is inside the daily noise band, so a daily/intraday stop
fires on transient dips (and 52% of those recover). Checking the stop less often — only
at weekly / biweekly / monthly closes — lets intra-week dips pass and only acts on
genuine end-of-period deterioration. This sweep asks whether that recovers any value.

Finding: smoothing helps a lot vs tight daily stops (biweekly 8% OOS Sharpe 0.39 vs
daily 3% 0.05), but NO stop configuration beats the no-stop book (OOS 0.54). The stop's
ceiling is "do no harm." Mega+Large, t+1 entry, hold-to-next-earnings, net of costs.
Requires data/raw/pead/ (run experiments/run_pead.py first).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np

import _bootstrap  # noqa: F401
from experiments.pead_dynamic_exit import build_paths, _stats, COST

HOLD = 63


def leg(members, P, sign, stop, every):
    """Stop checked only at every-th-day closes; exit at the breaching close."""
    out = []
    for i in members:
        c = sign * P[i, 1:HOLD + 1]; c = c[~np.isnan(c)]
        if len(c) == 0:
            continue
        if stop is None:
            out.append(c[-1]); continue
        exited = False
        for d in range(every - 1, len(c), every):
            if c[d] <= -stop:
                out.append(c[d]); exited = True; break
        if not exited:
            out.append(c[-1])
    return np.nanmean(out) if out else np.nan


def run(P, M, years, stop, every):
    sp = []
    for _, g in M[M["year"].isin(years)].groupby("season"):
        if len(g) < 10:
            continue
        hi, lo = g["sue"].quantile(0.8), g["sue"].quantile(0.2)
        L = leg(g[g["sue"] >= hi].index, P, +1.0, stop, every)
        S = leg(g[g["sue"] <= lo].index, P, -1.0, stop, every)
        if np.isfinite(L) and np.isfinite(S):
            sp.append((L + S) / 100 - COST)
    return _stats(np.array(sp))


def main():
    P, M = build_paths(); M = M.reset_index(drop=True)
    oos, full = list(range(2014, 2027)), list(range(2001, 2027))
    print("Stop-check frequency sweep (exit at the breaching close), net of Alpaca cost:")
    print(f"  {'check':>9}{'stop':>6}{'OOS Shrp':>10}{'OOS Ann':>9}{'FULL Shrp':>11}")
    b = run(P, M, oos, None, 5); f = run(P, M, full, None, 5)
    print(f"  {'(none)':>9}{'-':>6}{b[1]:>10.2f}{b[0]:>+9.1%}{f[1]:>11.2f}   <- ceiling")
    for every, lbl in [(1, "daily"), (5, "weekly"), (10, "biweekly"), (21, "monthly")]:
        for stop in (3, 5, 8, 10, 15):
            b = run(P, M, oos, stop, every); f = run(P, M, full, stop, every)
            print(f"  {lbl:>9}{stop:>5}%{b[1]:>10.2f}{b[0]:>+9.1%}{f[1]:>11.2f}")
        print()
    print("Takeaway: smoothing the stop check cuts the noise-whipsaw damage, but no stop")
    print("setting beats the no-stop book. The stop cannot add value to this strategy.")


if __name__ == "__main__":
    main()
