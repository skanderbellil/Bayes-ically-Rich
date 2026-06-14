#!/usr/bin/env python3
"""
PRE-REGISTERED out-of-sample test of the vol-management mechanism.

WHY: every analysis this session (a) searched ~30+ configs and (b) used data
from 2011-01-01 onward. The resulting alpha (t~2.1) failed our own t>3 bar
once multiple testing was counted. The only honest way to earn a trustworthy
result is to FIX one mechanism with canonical parameters and test it ONCE on
data the session never touched. This file is committed BEFORE the result is
seen (see git history) so the pre-registration is auditable.

────────────────────────────────────────────────────────────────────────────
PRE-REGISTRATION  (fixed before running — NO parameter search)
────────────────────────────────────────────────────────────────────────────
MECHANISM:  vol-managed 2x equity.
            exposure_t = clip( TARGET_VOL / (2 * EWMA_vol_{t-1}(underlying)),
                               0, 1 )  held in the 2x ETF, remainder in CASH.
PARAMETERS: EWMA halflife = 20 trading days   (RiskMetrics convention)
            TARGET_VOL    = 0.25              (the deployment-spec "moderate")
            cap           = 1.0               (no leverage beyond the 2x ETF)
            switch cost   = 10 bp on exposure turnover
            — identical to docs/.../DEPLOYMENT-vol-managed-leverage.md;
              NOT re-optimized for this window.
ASSETS:     QLD (2x QQQ) and SSO (2x SPY) — the only 2x ETFs that exist
            through the GFC. Safe asset = cash (managed-futures ETFs did not
            exist pre-2019, so the simplest mechanism is used).
TEST DATA:  2006-07-01 → 2010-12-31  — the GFC era. EVERY other experiment
            this session started 2011-01-01, so this window is untouched.
BENCHMARK:  the 1x underlying (QQQ / SPY) buy & hold over the same window.

HYPOTHESES (directional, stated before seeing results):
  H1 (primary):  vol-managed 2x equity has a SHALLOWER max drawdown than the
                 1x underlying buy & hold — because the 2008 volatility spike
                 forces de-risking the passive holder cannot do.
  H2 (secondary):vol-managed 2x equity has a HIGHER Sharpe than the 1x
                 underlying over the full window.
PASS = both H1 and H2 hold for BOTH QLD and SSO (4 of 4). This is the single
       test; no configs are swept, no thresholds tuned.
────────────────────────────────────────────────────────────────────────────
"""
import sys
import numpy as np
import pandas as pd
from tabulate import tabulate
import _bootstrap  # noqa: F401

DATA = _bootstrap.ROOT / "datasets"
ANN, RF, SC = 252, 0.04, 10 / 1e4
HL, TARGET, CAP = 20, 0.25, 1.0
WIN = ("2006-07-01", "2010-12-31")


def stats(r, rf=RF):
    r = r.dropna()
    mu, sd = r.mean() * ANN, r.std() * np.sqrt(ANN)
    eq = (1 + r).cumprod()
    return dict(cagr=eq.iloc[-1] ** (ANN / len(r)) - 1, vol=sd,
                sharpe=(mu - rf) / (sd + 1e-12),
                maxdd=float((eq / eq.cummax() - 1).min()))


def vol_managed(lev2x, underlying, rf_d):
    v = (underlying.ewm(halflife=HL).std() * np.sqrt(ANN)).shift(1)
    w = (TARGET / (2 * v)).clip(0, CAP)
    r = w * lev2x + (1 - w) * rf_d
    return (r - w.diff().abs().fillna(0) * SC).dropna()


def main():
    px = pd.read_csv(DATA / "levered_stacked.csv", index_col=0, parse_dates=True)
    rets = px.pct_change()
    rf_d = RF / ANN
    print(f"\nPRE-REGISTERED OOS TEST — GFC era {WIN[0]} → {WIN[1]} "
          f"(untouched; every other experiment started 2011-01-01).")
    print("Mechanism + params fixed in the committed pre-registration above.\n")
    rows, passes = [], []
    for lev, und in [("QLD", "QQQ"), ("SSO", "SPY")]:
        vm = vol_managed(rets[lev], rets[und], rf_d).loc[WIN[0]:WIN[1]]
        bh = rets[und].loc[vm.index]
        lv = rets[lev].loc[vm.index]
        sv, sb, sl = stats(vm), stats(bh), stats(lv)
        h1 = sv["maxdd"] > sb["maxdd"]      # shallower (less negative) DD than 1x
        h2 = sv["sharpe"] > sb["sharpe"]
        passes += [h1, h2]
        rows.append([f"vol-managed {lev}", f"{sv['cagr']:+.1%}", f"{sv['vol']:.0%}",
                     f"{sv['sharpe']:.2f}", f"{sv['maxdd']:.0%}",
                     f"H1 {'PASS' if h1 else 'FAIL'} / H2 {'PASS' if h2 else 'FAIL'}"])
        rows.append([f"  {und} buy&hold (1x bench)", f"{sb['cagr']:+.1%}",
                     f"{sb['vol']:.0%}", f"{sb['sharpe']:.2f}", f"{sb['maxdd']:.0%}", ""])
        rows.append([f"  {lev} buy&hold (2x, unmanaged)", f"{sl['cagr']:+.1%}",
                     f"{sl['vol']:.0%}", f"{sl['sharpe']:.2f}", f"{sl['maxdd']:.0%}", ""])
    print(tabulate(rows, headers=["book", "CAGR", "vol", "Sharpe", "maxDD",
                                  "pre-registered test"], tablefmt="github"))
    n_pass = sum(passes)
    print(f"\nRESULT: {n_pass}/4 pre-registered predictions held "
          f"(H1+H2 for QLD and SSO).")
    print("PASS" if n_pass == 4 else "PARTIAL/FAIL", "— single test, no config "
          "search, on data the session never saw. This is the honest verdict on "
          "the MECHANISM (vol-managing leverage), independent of the in-sample "
          "alpha that failed t>3.")


if __name__ == "__main__":
    main()
