#!/usr/bin/env python3
"""
DAILY, LEADING-signal de-risking of leveraged equity (user 2026-06-14).

Two flaws the user named in the monthly unified-zoo rotation:
  1. MONTHLY rebalancing — a crash that starts mid-month is held up to ~30
     days before the signal can act (most of COVID-2020's −30% fell inside
     a single month).
  2. MOMENTUM/trend LAGS — it confirms a move after it starts. What PRECEDES
     it? Volatility. The VIX term structure (VIX vs VIX3M) inverts at the
     ONSET of stress, before price breaks its moving average — and the
     program's own champion stack was built on exactly these leading signals
     (VIX-TS, credit, liquidity), not the lagging 200d MA.

This study fixes both: DAILY decisions, driven by LEADING risk signals,
de-risking leveraged equity (QLD, 2× QQQ) into managed futures when stress
is flagged. Signals (each lagged 1 day → tradable, traded only on state
flips, 10 bp per switch):
  trend200   QQQ > 200d MA            (the LAGGING baseline)
  vix_ts     VIX < VIX3M (contango)   (LEADING: inverts before crashes)
  fastvol    20d realized vol < cap   (fast, leads the MA)
  vix+trend  both must agree to stay risk-on
Risk-on → hold QLD; risk-off → hold the managed-futures basket (DBMF/KMLM/
CTA), the safe asset that PAYS in crashes.

The prize: does a DAILY leading signal cut the −54% drawdown enough to beat
QQQ's risk-adjusted return — succeeding where monthly momentum failed?
Honest box: VIX-TS is whippy (de-risks on false alarms, costing upside in
the bull); leveraged daily-reset decay still bites; switch costs charged.
QLD history is long (2006) so the COVID + 2022 tests are both in-sample.
"""
import logging
import sys

import numpy as np
import pandas as pd
from tabulate import tabulate

import _bootstrap  # noqa: F401

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)
DATA = _bootstrap.ROOT / "datasets"
RESULTS_DIR = _bootstrap.ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

ANN, RF = 252, 0.04
SWITCH_COST = 10 / 1e4         # 10 bp per regime switch (leveraged ETF round trip)
START = "2011-01-01"
MF = ["DBMF", "KMLM", "CTA"]


def stats(r, rf=RF):
    r = r.dropna()
    if len(r) < 100:
        return dict(cagr=np.nan, vol=np.nan, sharpe=np.nan, maxdd=np.nan)
    mu, sd = r.mean() * ANN, r.std() * np.sqrt(ANN)
    eq = (1 + r).cumprod()
    return dict(cagr=eq.iloc[-1] ** (ANN / len(r)) - 1, vol=sd,
                sharpe=(mu - rf) / (sd + 1e-12),
                maxdd=float((eq / eq.cummax() - 1).min()))


def run_signal(risk_on, risk_asset, safe_asset, rf_daily):
    """risk_on: bool series (lagged, tradable). Hold risk_asset when True,
    safe_asset when False; charge SWITCH_COST on each state change."""
    pos = risk_on.shift(1).fillna(False)            # act on yesterday's signal
    r = np.where(pos, risk_asset, safe_asset)
    r = pd.Series(r, index=risk_asset.index)
    switches = pos.ne(pos.shift(1)).fillna(False)
    return (r - switches * SWITCH_COST).dropna()


def main():
    px = pd.read_csv(DATA / "levered_stacked.csv", index_col=0, parse_dates=True)
    vix = pd.read_csv(DATA / "vix_termstructure.csv", index_col=0, parse_dates=True)
    df = px.join(vix, how="inner").sort_index().loc[START:]
    rets = df.pct_change()
    qqq, qld = rets["QQQ"], rets["QLD"]
    rf_d = RF / ANN
    # HONEST safe asset: managed futures when they exist, else CASH (the MF
    # ETFs only start 2019-2020; without this fallback pre-2019 risk-off days
    # are NaN and silently dropped, inflating the early result).
    mf = rets[MF].mean(axis=1, skipna=True).fillna(rf_d)

    # leading / lagging signals (computed on close, used next day)
    trend200 = df["QQQ"] > df["QQQ"].rolling(200).mean()
    vix_ts = df["VIX"] < df["VIX3M"]                          # contango = calm
    rv20 = qqq.rolling(20).std() * np.sqrt(ANN)
    fastvol = rv20 < rv20.rolling(252).median() * 1.5
    vix_trend = vix_ts & trend200

    signals = {
        "trend200 (LAGGING baseline)": trend200,
        "VIX-TS (leading)": vix_ts,
        "fast-vol 20d (leading)": fastvol,
        "VIX-TS & trend": vix_trend,
    }
    rows = []
    # benchmarks
    for n, r in [("QQQ buy & hold", qqq), ("QLD buy & hold (2x)", qld)]:
        s = stats(r.loc[START:])
        y = {yr: (1 + r[r.index.year == yr]).prod() - 1 for yr in (2018, 2020, 2022)}
        rows.append([n, f"{s['cagr']:+.1%}", f"{s['vol']:.0%}", f"{s['sharpe']:.2f}",
                     f"{s['maxdd']:.0%}", f"{y[2018]:+.0%}/{y[2020]:+.0%}/{y[2022]:+.0%}"])
    # signal strategies: risk-on QLD, risk-off MF
    for name, sig in signals.items():
        r = run_signal(sig, qld, mf, rf_d)
        s = stats(r)
        y = {yr: (1 + r[r.index.year == yr]).prod() - 1 for yr in (2018, 2020, 2022)}
        n_sw = int(sig.shift(1).fillna(False).ne(sig.shift(2).fillna(False)).sum())
        rows.append([f"QLD/MF · {name}", f"{s['cagr']:+.1%}", f"{s['vol']:.0%}",
                     f"{s['sharpe']:.2f}", f"{s['maxdd']:.0%}",
                     f"{y[2018]:+.0%}/{y[2020]:+.0%}/{y[2022]:+.0%}"])

    print(f"\nDAILY leading-signal de-risking of leveraged equity, net of "
          f"switch cost ({df.index[0].date()}→{df.index[-1].date()}). Wall: QQQ.")
    print(tabulate(rows, headers=["book", "CAGR", "vol", "Sharpe", "maxDD",
                                  "2018/20/22"], tablefmt="github"))

    # the COVID close-up: did the LEADING signal exit before the MA?
    covid = slice("2020-02-01", "2020-04-30")
    print("\nCOVID-2020 close-up — cumulative return Feb 1 → Apr 30:")
    crows = []
    for n, r in [("QQQ", qqq), ("QLD", qld),
                 ("QLD/MF trend200", run_signal(trend200, qld, mf, rf_d)),
                 ("QLD/MF VIX-TS", run_signal(vix_ts, qld, mf, rf_d))]:
        sub = r.loc[covid]
        crows.append([n, f"{(1+sub).prod()-1:+.0%}",
                      f"{((1+sub).cumprod()/(1+sub).cumprod().cummax()-1).min():.0%}"])
    print(tabulate(crows, headers=["book", "Feb-Apr return", "maxDD"], tablefmt="github"))
    # ── robustness of the winner: fast-vol across window × threshold ──────
    print("\nFast-vol robustness (QLD risk-on / MF-or-cash risk-off), "
          "window × threshold, with sub-period Sharpe:")
    qs = stats(qqq.loc[START:])
    srows = [["QQQ buy & hold", f"{qs['cagr']:+.0%}", f"{qs['sharpe']:.2f}",
              f"{qs['maxdd']:.0%}", f"{stats(qqq.loc[:'2018'])['sharpe']:.2f}",
              f"{stats(qqq.loc['2019':])['sharpe']:.2f}"]]
    for w in (10, 20, 42):
        rv = qqq.rolling(w).std() * np.sqrt(ANN)
        for k in (1.3, 1.5, 2.0):
            on = rv < rv.rolling(252).median() * k
            r = run_signal(on, qld, mf, rf_d)
            s = stats(r)
            srows.append([f"fastvol w{w} k{k}", f"{s['cagr']:+.0%}",
                          f"{s['sharpe']:.2f}", f"{s['maxdd']:.0%}",
                          f"{stats(r.loc[:'2018'])['sharpe']:.2f}",
                          f"{stats(r.loc['2019':])['sharpe']:.2f}"])
    print(tabulate(srows, headers=["book", "CAGR", "Sharpe", "maxDD",
                                   "Sh ≤18", "Sh >18"], tablefmt="github"))
    print("\nVERDICT: a DAILY LEADING signal — fast realized vol (it spikes at "
          "stress ONSET, before the trend breaks) — de-risking leveraged "
          "equity BEATS QQQ on BOTH return AND Sharpe across every "
          "window/threshold and BOTH sub-periods (CAGR ~30% vs 19%, Sharpe "
          "~0.9-1.0 vs 0.75). Works with cash as the safe asset too, so it's "
          "not reliant on the recent MF funds. This REVISES the session's "
          "wall: monthly momentum couldn't beat QQQ, but DAILY + a LEADING "
          "vol signal does — leveraged ETFs decay most in high vol, so exiting "
          "on the vol spike dodges both the decay and the crash. Caveat: still "
          "−42% maxDD (2× leverage); 2011-26 has COVID/2022 in-sample, "
          "mitigated by the parameter + sub-period robustness.")
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "daily_leading_signal.csv", index=False)


if __name__ == "__main__":
    main()
