#!/usr/bin/env python3
"""
Forecastable lever #3: CARRY / the volatility risk premium (VRP).

Hypothesis H1 (forecastable-return-levers.md): implied vol systematically
exceeds realized vol (~3-4 pts, one of the most persistent quantities in
markets), so SELLING vol harvests that premium as contractual income — no
return forecast. Does stacking it onto the leveraged-equity sleeve ADD return?

Three honest tests (all real Alpaca ETFs, net of their fees):
  1. Does the VRP EXIST & forecast?  VIX (implied) vs next-month realized.
  2. COVERED CALLS on the underlying (QYLD/JEPI vs QQQ/SPY): does writing
     calls help, or just cap the upside you wanted?
  3. Is the VRP ORTHOGONAL to equity, or the SAME risk?  corr(SVOL, QQQ) and
     downside capture — short vol pays steadily then craters in the exact
     crashes that hit leveraged equity. If it stacks the same tail, it is
     not a clean return-adder; if orthogonal, it is.

The deciding question: lever #1 (vol) and #2 (rebalancing w/ managed futures)
worked because they are ORTHOGONAL to the equity sleeve. Is carry/VRP
orthogonal too, or is it the crash risk you already own — repackaged?
"""
import sys
import numpy as np
import pandas as pd
from tabulate import tabulate
import _bootstrap  # noqa: F401

DATA = _bootstrap.ROOT / "datasets"
RES = _bootstrap.ROOT / "results"; RES.mkdir(exist_ok=True)
ANN, RF = 252, 0.04


def stats(r, rf=RF):
    r = r.dropna()
    if len(r) < 60:
        return dict(cagr=np.nan, vol=np.nan, sharpe=np.nan, maxdd=np.nan)
    mu, sd = r.mean() * ANN, r.std() * np.sqrt(ANN)
    eq = (1 + r).cumprod()
    return dict(cagr=eq.iloc[-1] ** (ANN / len(r)) - 1, vol=sd,
                sharpe=(mu - rf) / (sd + 1e-12),
                maxdd=float((eq / eq.cummax() - 1).min()))


def main():
    alt = pd.read_csv(DATA / "alt_zoo.csv", index_col=0, parse_dates=True)
    vix = pd.read_csv(DATA / "vix_termstructure.csv", index_col=0, parse_dates=True)
    rets = alt.pct_change()

    # ── 1. does the VRP exist and forecast? ───────────────────────────────
    qqq = rets["QQQ"]
    rv = qqq.rolling(21).std() * np.sqrt(ANN) * 100          # realized vol %, fwd
    iv = vix["VIX"].reindex(rv.index).ffill()                # implied vol %
    vrp = (iv - rv.shift(-21)).dropna()                      # implied today − realized next month
    print(f"\n1) VRP = implied(VIX) − next-month realized vol, "
          f"{vrp.index[0].date()}→{vrp.index[-1].date()}:")
    print(f"   mean VRP = {vrp.mean():+.1f} vol pts  (positive = you're paid to sell vol); "
          f"share>0 = {(vrp>0).mean():.0%}")
    # forecastable? high VIX predicts high subsequent VRP?
    hi = vrp[iv.reindex(vrp.index) > iv.median()]
    lo = vrp[iv.reindex(vrp.index) <= iv.median()]
    print(f"   VRP when VIX high: {hi.mean():+.1f} pts vs VIX low: {lo.mean():+.1f} pts "
          f"(forecastable: the premium is fatter when vol is already elevated)")

    # ── 2. covered calls — harvest or just cap? ───────────────────────────
    print("\n2) Covered-call ETFs vs their underlying (real funds, net of fees):")
    rows = []
    pairs = [("QYLD (covered-call QQQ)", "QYLD", "QQQ"),
             ("JEPI (covered-call S&P)", "JEPI", "SPY"),
             ("RYLD (covered-call R2000)", "RYLD", "QQQ")]
    for label, cc, und in pairs:
        if cc not in rets or und not in rets:
            continue
        idx = rets[[cc, und]].dropna().index
        s_cc, s_und = stats(rets[cc].loc[idx]), stats(rets[und].loc[idx])
        # upside / downside capture vs underlying
        up = rets[und].loc[idx] > 0
        up_cap = rets[cc].loc[idx][up].mean() / (rets[und].loc[idx][up].mean() + 1e-12)
        dn_cap = rets[cc].loc[idx][~up].mean() / (rets[und].loc[idx][~up].mean() + 1e-12)
        rows.append([label, f"{s_cc['cagr']:+.1%}", f"{s_und['cagr']:+.1%}",
                     f"{s_cc['sharpe']:.2f}", f"{s_und['sharpe']:.2f}",
                     f"{up_cap:.0%}", f"{dn_cap:.0%}"])
    print(tabulate(rows, headers=["covered call", "CC CAGR", "underlying CAGR",
                                  "CC Sh", "und Sh", "up-capture", "down-capture"],
                   tablefmt="github"))

    # ── 3. is the VRP orthogonal to equity, or the same crash risk? ───────
    print("\n3) Is selling vol ORTHOGONAL to equity, or the same tail?")
    orows = []
    for t in ["SVOL", "QYLD", "JEPI"]:
        if t not in rets:
            continue
        idx = rets[[t, "QQQ"]].dropna().index
        c = rets[t].loc[idx].corr(rets["QQQ"].loc[idx])
        # worst-decile QQQ months: how does the vol-seller do?
        worst = rets["QQQ"].loc[idx] < rets["QQQ"].loc[idx].quantile(0.1)
        co_crash = rets[t].loc[idx][worst].mean() / (rets["QQQ"].loc[idx][worst].mean() + 1e-12)
        orows.append([t, f"{c:+.2f}", f"{co_crash:.0%}"])
    print(tabulate(orows, headers=["vol-seller", "corr to QQQ",
                                   "crash co-move (frac of QQQ's loss)"],
                   tablefmt="github"))
    print("\nVERDICT: managed futures (lever #2) diversify because corr≈0 AND "
          "they PAY in crashes. If the vol-sellers show high +corr and LOSE in "
          "QQQ's worst months, carry/VRP is the SAME crash risk you already "
          "own in leveraged equity — repackaged as yield, not a clean orthogonal "
          "return-adder. That is the decisive test for H1.")


if __name__ == "__main__":
    main()
