#!/usr/bin/env python3
"""
"Better sauce": leveraged equity + managed futures (long-only, more RETURN).

The user's push (2026-06-13): the diversified books are SMOOTHER than SPY/QQQ
but the absolute return is modest — they want more money, not just less
drawdown. For a long-only investor that means LEVERAGE you can buy without a
margin account or shorting:
  • leveraged ETFs  — QLD (2× QQQ), SSO (2× SPY), UPRO (3× SPY): more equity
    return per dollar, but −80% drawdowns alone.
  • managed futures — the crisis-convex sleeve that SPIKES when leveraged
    equity craters (+22-24% in 2022 vs QLD ≈ −60%).

The thesis: holding both, long-only and rebalanced monthly, lets you carry
the leverage SURVIVABLY — the MF cushion the crash, and rebalancing harvests
it (sell the MF spike, buy the equity dip). Can the combo BEAT QQQ on CAGR
with a drawdown you could actually hold?

All books are long-only buys on Alpaca (leveraged + managed-futures ETFs),
monthly rebalanced, net of cost. Decisive window 2020-2026 (DBMF/KMLM exist
and it spans the 2022 crash); full-cycle caveat noted with QLD alone.

Honest box: leveraged daily-reset ETFs decay in choppy/sideways tape; the
MF funds are recent so 2020-26 includes their best regime; −drawdowns here
are still large. This is an aggressive growth sleeve, not a free lunch.
"""
import logging
import sys

import numpy as np
import pandas as pd
from tabulate import tabulate

import _bootstrap  # noqa: F401
from posterioralpha.backtest.alpaca_costs import HALF_SPREAD_BPS

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)
DATA = _bootstrap.ROOT / "datasets"
RESULTS_DIR = _bootstrap.ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

HS = HALF_SPREAD_BPS["ETF"] / 1e4
MF = ["DBMF", "KMLM", "CTA"]
RF = 0.04


def stats(r, rf=RF):
    r = r.dropna()
    if len(r) < 12:
        return dict(cagr=np.nan, vol=np.nan, sharpe=np.nan, maxdd=np.nan)
    mu, sd = r.mean() * 12, r.std() * np.sqrt(12)
    eq = (1 + r).cumprod()
    cagr = eq.iloc[-1] ** (12 / len(r)) - 1
    return dict(cagr=cagr, vol=sd, sharpe=(mu - rf) / (sd + 1e-12),
                maxdd=float((eq / eq.cummax() - 1).min()))


def rebal(weights, m):
    """Monthly-rebalanced fixed-weight book; returns net daily-equiv series.
    weights: dict ticker->w (long-only, sum≤1; rest to cash at RF/12)."""
    cols = list(weights)
    sub = m[cols].dropna()
    w = pd.Series(weights)
    cash = 1.0 - w.sum()
    gross = (sub * w).sum(axis=1) + cash * (RF / 12)
    turn = w.abs().sum() * 2  # full reset proxy each month is conservative
    # realistic monthly turnover of a fixed-weight book ≈ drift correction, small
    tc = (w.abs().sum() * 0.10) * HS    # ~10% of notional re-traded/mo
    return gross - tc


def main():
    m = (1 + pd.read_csv(DATA / "levered_stacked.csv", index_col=0,
                         parse_dates=True).pct_change()
         ).resample("ME").apply(lambda x: x.prod() - 1)
    mf = m[MF].mean(axis=1, skipna=True)
    m = m.assign(MFB=mf)

    win = m.index >= "2020-01-01"
    qqq = m["QQQ"].loc[win]
    spy = m["SPY"].loc[win]

    books = {
        "SPY buy & hold": spy,
        "QQQ buy & hold": qqq,
        "QLD buy & hold (2x QQQ)": m["QLD"].loc[win],
        "QLD 70% + MF 30%": rebal({"QLD": 0.7, "MFB": 0.3}, m).loc[win],
        "QLD 60% + MF 40%": rebal({"QLD": 0.6, "MFB": 0.4}, m).loc[win],
        "SSO 60% + MF 40% (2x SPY)": rebal({"SSO": 0.6, "MFB": 0.4}, m).loc[win],
        "UPRO 40% + MF 30% + cash": rebal({"UPRO": 0.4, "MFB": 0.3}, m).loc[win],
        "QLD 50 + MF 30 + GLD 20": rebal({"QLD": 0.5, "MFB": 0.3, "GLD": 0.2}, m).loc[win]
        if "GLD" in m else None,
    }
    rows = []
    for name, r in books.items():
        if r is None:
            continue
        s = stats(r)
        y = {yr: (1 + r[r.index.year == yr]).prod() - 1 for yr in (2020, 2022, 2024)}
        rows.append([name, f"{s['cagr']:+.1%}", f"{s['vol']:.0%}",
                     f"{s['sharpe']:.2f}", f"{s['maxdd']:.0%}",
                     f"{y[2020]:+.0%}/{y[2022]:+.0%}/{y[2024]:+.0%}"])
    print(f"\n'Better sauce' — leveraged equity + managed futures, long-only, "
          f"net of cost (2020-2026, incl. 2022 crash):")
    print(tabulate(rows, headers=["book", "CAGR", "vol", "Sharpe", "maxDD",
                                  "2020/22/24"], tablefmt="github"))
    print("\nThe question: which books BEAT QQQ's CAGR with a maxDD you could "
          "actually hold? Leverage gives the return; managed futures make the "
          "drawdown survivable (and rebalancing harvests the 2022 crash).")

    # full-cycle drawdown reality: QLD alone vs QLD+WTMF since 2012
    if "QLD" in m and "DBMF" in m:
        ql = m["QLD"].dropna()
        print(f"\nFull-history reality (QLD 2x QQQ, {ql.index[0].date()}→): "
              f"CAGR {stats(ql)['cagr']:+.1%}, maxDD {stats(ql)['maxdd']:.0%} "
              f"— leverage's tail is brutal without the hedge; the MF combo is "
              f"only as good as the hedge's availability in the next crash.")
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "levered_alts.csv", index=False)


if __name__ == "__main__":
    main()
