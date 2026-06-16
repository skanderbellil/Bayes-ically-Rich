#!/usr/bin/env python3
"""
Retail-scale cross-sectional signal lab — IC improvement net of Alpaca costs.

Loop goal (2026-06-13): find a signal construction whose information
coefficient improvement actually survives at retail scale under the Alpaca
cost model. The Idea-1 signal lab tested signals INDIVIDUALLY on 94 S&P-100
names and found none cleared t>2. Two KB levers are untested here:
  • BREADTH (Grinold-Kahn IR ∝ IC·√breadth): use the 500-name liquid
    universe (5× the names, 2010-2026), which sharpens IC t for free.
  • COMBINATION (AQR value+momentum, Chincarini-Kim z-score aggregation):
    blend negatively-correlated signals — never tried in the signal lab.

Universe: equity_universe_prices.csv (500 liquid US large-caps). At retail
scale these are ETB (shortable) and ~4 bps half-spread (Large Cap tier);
the long-only book needs no shorting at all.

Signals (all causal, monthly rebalanced):
  mom12_1     12-1 momentum (skip last month)
  lowvol      −252d realized vol (defensive)
  reversal    −21d return (short-term reversal; neg-corr with momentum)
  COMPOSITE   equal-weight average of the cross-sectional %-ranks of the
              three above — the breadth/blend play

For each signal: monthly rank IC (mean, t) AND net-of-Alpaca performance of
  • long-only top decile (retail-implementable, no short)
  • decile long-short (top − bottom; large-caps are ETB)
vs the EW-500 hold-everything baseline and SPY. Net = gross − Σ|Δw|·spread
with the Alpaca Large-Cap half-spread. The deliverable is whether the
COMPOSITE's net information ratio beats every single signal and the
baseline — a real retail-scale IC improvement, not a gross-only mirage.
"""
import logging
import sys

import numpy as np
import pandas as pd
from tabulate import tabulate

import _bootstrap  # noqa: F401
from posterioralpha.backtest.alpaca_costs import HALF_SPREAD_BPS
from posterioralpha.data import (load_equity_universe_info,
                                 load_equity_universe_prices,
                                 load_portfolio_prices)
from posterioralpha.mining.ic import monthly_ic, ic_summary

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)
RESULTS_DIR = _bootstrap.ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

HALF_SPREAD = HALF_SPREAD_BPS["Large Cap"] / 1e4    # 4 bps one-way, conservative
N_DECILE = 10
ANN = 252


def ann_stats(r, rf=0.0):
    r = r.dropna()
    if len(r) < 24:
        return dict(ann=np.nan, vol=np.nan, sharpe=np.nan, maxdd=np.nan)
    ex = r - rf / 12
    mu, sd = r.mean() * 12, r.std() * np.sqrt(12)
    eq = (1 + r).cumprod()
    return dict(ann=mu, vol=sd, sharpe=ex.mean() * 12 / (sd + 1e-12),
                maxdd=float((eq / eq.cummax() - 1).min()))


def month_end_dates(idx):
    return pd.DatetimeIndex([idx[idx <= me][-1]
                             for me in pd.Series(index=idx, dtype=float)
                             .resample("ME").last().index
                             if len(idx[idx <= me])])


def decile_book(sig, fwd_m, me_dates, long_only):
    """Monthly returns + turnover for top-decile (long-only) or top−bottom L/S."""
    rets, turns, prev_l, prev_s = [], [], None, None
    dates = []
    for t in me_dates:
        if t not in sig.index or t not in fwd_m.index:
            continue
        s, f = sig.loc[t], fwd_m.loc[t]
        m = s.notna() & f.notna()
        if m.sum() < 50:
            continue
        s, f = s[m], f[m]
        q = pd.qcut(s.rank(method="first"), N_DECILE, labels=False)
        long_w = (q == N_DECILE - 1).astype(float)
        long_w /= long_w.sum()
        if long_only:
            r = float((long_w * f).sum())
            tn = (long_w.subtract(prev_l, fill_value=0).abs().sum()
                  if prev_l is not None else 1.0)
            prev_l = long_w
        else:
            short_w = (q == 0).astype(float)
            short_w /= short_w.sum()
            r = float((long_w * f).sum() - (short_w * f).sum())
            tn = ((long_w.subtract(prev_l, fill_value=0).abs().sum()
                   + short_w.subtract(prev_s, fill_value=0).abs().sum())
                  if prev_l is not None else 2.0)
            prev_l, prev_s = long_w, short_w
        rets.append(r)
        turns.append(tn)
        dates.append(t)
    gross = pd.Series(rets, index=dates)
    turn = pd.Series(turns, index=dates)
    net = gross - turn * HALF_SPREAD
    return gross, net, float(turn.mean())


def main():
    px = load_equity_universe_prices()
    px = px.loc[:, px.notna().mean() > 0.95].dropna(how="all")
    logger.info(f"Universe: {px.shape[1]} names, "
                f"{px.index[0].date()} → {px.index[-1].date()}")
    rets_d = px.pct_change()
    spy = load_portfolio_prices()["SPY"].pct_change()

    # ── signals (causal, daily-indexed; sampled at month-ends) ────────────
    mom = px.shift(21) / px.shift(252) - 1.0
    lowvol = -rets_d.rolling(252, min_periods=200).std()
    reversal = -(px / px.shift(21) - 1.0)

    def xrank(df):
        return df.rank(axis=1, pct=True)
    composite = (xrank(mom) + xrank(lowvol) + xrank(reversal)) / 3.0

    # ── sector-neutralization: demean each signal within GICS sector ──────
    info = load_equity_universe_info()
    sec = info["sector"].reindex(px.columns)
    groups = {s: sec.index[sec == s].tolist() for s in sec.dropna().unique()}

    def sector_neutral(df):
        out = df.copy()
        for cols in groups.values():
            cols = [c for c in cols if c in out.columns]
            if len(cols) >= 3:
                sub = out[cols]
                out[cols] = sub.sub(sub.mean(axis=1), axis=0)
        return out

    mom_sn = sector_neutral(mom)
    lowvol_sn = sector_neutral(lowvol)
    reversal_sn = sector_neutral(reversal)
    composite_sn = (xrank(mom_sn) + xrank(lowvol_sn) + xrank(reversal_sn)) / 3.0

    signals = {"mom12_1": mom, "lowvol": lowvol, "reversal": reversal,
               "COMPOSITE": composite,
               "mom_SN": mom_sn, "lowvol_SN": lowvol_sn,
               "reversal_SN": reversal_sn, "COMPOSITE_SN": composite_sn}

    me = month_end_dates(px.index)
    fwd_m = (px.shift(-21) / px - 1.0)          # 21-day forward return

    # ── IC + net-of-cost books ────────────────────────────────────────────
    ic_rows, perf_rows = [], []
    spy_m = (1 + spy).resample("ME").apply(lambda x: x.prod() - 1)
    ew = rets_d.mean(axis=1)
    ew_m = (1 + ew).resample("ME").apply(lambda x: x.prod() - 1)
    base = ann_stats(ew_m)
    spm = ann_stats(spy_m.reindex(ew_m.index))

    for name, sig in signals.items():
        ics = monthly_ic(sig, px, horizons=(21,))[21]
        s = ic_summary(ics, 21)
        ic_rows.append([name, f"{s['mean_ic']:.3f}", f"{s['t_stat']:.2f}",
                        f"{s['hit_rate']:.2f}", s["n_obs"]])

        g_lo, n_lo, t_lo = decile_book(sig, fwd_m, me, long_only=True)
        g_ls, n_ls, t_ls = decile_book(sig, fwd_m, me, long_only=False)
        a_lo, a_ls = ann_stats(n_lo), ann_stats(n_ls)
        # info ratio of long-only net excess vs EW baseline
        act = (n_lo - ew_m.reindex(n_lo.index)).dropna()
        ir = act.mean() * 12 / (act.std() * np.sqrt(12) + 1e-12)
        perf_rows.append([name,
                          f"{a_lo['ann']:+.1%}", f"{a_lo['sharpe']:.2f}",
                          f"{ir:+.2f}", f"{t_lo:.2f}",
                          f"{a_ls['ann']:+.1%}", f"{a_ls['sharpe']:.2f}",
                          f"{t_ls:.2f}"])

    print("\nMonthly rank IC (21d fwd), 500-name retail universe:")
    print(tabulate(ic_rows, headers=["signal", "mean IC", "t", "hit", "n"],
                   tablefmt="github"))
    print("\nNet-of-Alpaca performance (Large-Cap 4bp half-spread):")
    print(tabulate(perf_rows, headers=["signal", "LO ann", "LO SR", "LO IR",
                                       "LO turn", "LS ann", "LS SR", "LS turn"],
                   tablefmt="github"))
    print(f"\nBaselines: EW-500 Sharpe {base['sharpe']:.2f} "
          f"(ann {base['ann']:+.1%}) · SPY Sharpe {spm['sharpe']:.2f} "
          f"(ann {spm['ann']:+.1%}).")
    print("LO = long-only top decile (no shorting needed); LS = decile "
          "long-short; turn = avg monthly one-way turnover. IR = info ratio "
          "of long-only net vs EW-500.")
    print("\nGOAL CHECK: does COMPOSITE beat every single signal on IC t AND "
          "net LO Sharpe/IR, and clear the EW-500 baseline net of cost?")

    pd.DataFrame(ic_rows, columns=["signal", "mean_ic", "t", "hit", "n"]
                 ).to_csv(RESULTS_DIR / "retail_signal_ic.csv", index=False)


if __name__ == "__main__":
    main()
