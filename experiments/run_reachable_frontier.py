#!/usr/bin/env python3
"""
The reachable absolute-return frontier — what you can actually buy.

Idea (user thread: reachable absolute return, beta or alpha)
-----------------------------------------------------------
The arc landed three facts: (1) the levered EW zoo is the absolute-return
king but unreachable (+10.9%/yr, needs 150 shorts + leverage); (2) no
long-only book can TRACK it (run_zoo_tracking — it's the orthogonal
complement of beta); (3) yet it's NEGATIVELY correlated with equities
(SPY −0.42), i.e. a diversifier. So the practical question is not "rebuild
the zoo" but "what is the best reachable long-only, no-leverage book", and
does the program's own lesson — diversify broadly, then vol-target —
beat just holding SPY?

Reachable universe (20 liquid ETFs, 2011-06→, no levered/inverse funds),
grouped so neither the 7 equity sleeves nor the low-vol bond sleeves
dominate by count:
  US large  SPY QQQ RSP IWD VYM SPLV SPHQ      US small  IWM
  Intl      EFA EEM                            Real      GLD DBC VNQ
  Rates     IEF TLT SHY TIP                    Credit    LQD HYG
  Dollar    UUP

Books (all causal — trailing 12m vol, weights applied next month, 5bp
turnover cost), long-only, Σw ≤ 1, NO leverage (vol-target only scales
toward cash):
  SPY / 60-40              reachability baselines
  inverse-vol (flat)       1/σ across all 20 — the bond cluster wins
  group risk parity        equal risk per group, 1/σ within — the honest
                           "diversify everything" book
  GRP @ vol-cap            same, scaled down toward cash to a 10% vol budget
  GRP defensive tilt       group RP with risk budget shifted to the
                           low-vol/quality/real/rates groups (the zoo's
                           defensive character, expressed long-only)

Honesty box
-----------
• These are real funds with real prices — costs are charged (5bp×turnover).
  Unlike every prior zoo study this is a NET, reachable number.
• No leverage: the 10% vol target can only de-risk (add cash), never
  borrow — so "@vol-cap" books sit at ≤10% vol, not exactly 10%.
• Short ETF history (181 months from 2011) = one big regime (the post-GFC
  bull + 2022). Read Sharpes as indicative, not as the 25-year evidence
  the OpenAP panel carries. Group definitions pre-declared, not searched.
"""
import logging
import sys

import numpy as np
import pandas as pd
from tabulate import tabulate

import _bootstrap  # noqa: F401
from posterioralpha.data import load_openap_doc, load_openap_returns

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)
RESULTS_DIR = _bootstrap.ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
DATA = _bootstrap.ROOT / "datasets"

RF = 0.04
VOL_WIN, VOL_MIN = 12, 6
TARGET_VOL = 0.10
TC = 0.0005
MIN_ELIGIBLE = 30

GROUPS = {
    "us_large": ["SPY", "QQQ", "RSP", "IWD", "VYM", "SPLV", "SPHQ"],
    "us_small": ["IWM"], "intl": ["EFA", "EEM"],
    "real": ["GLD", "DBC", "VNQ"], "rates": ["IEF", "TLT", "SHY", "TIP"],
    "credit": ["LQD", "HYG"], "dollar": ["UUP"],
}
DEFENSIVE_BUDGET = {"us_large": 0.10, "us_small": 0.05, "intl": 0.10,
                    "real": 0.25, "rates": 0.25, "credit": 0.15, "dollar": 0.10}


def metrics(r: pd.Series, rf_m: pd.Series) -> dict:
    r = r.dropna()
    mu, sd = r.mean() * 12, r.std() * np.sqrt(12)
    eq = (1 + r).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    # Sharpe on the REALIZED monthly risk-free rate — fair to low-vol books
    # across the 2011-21 zero-rate decade (flat 4% would penalize them).
    excess = (r - rf_m.reindex(r.index).fillna(0.0))
    sharpe = excess.mean() / (excess.std() + 1e-12) * np.sqrt(12)
    return {"ann_ret": mu, "ann_vol": sd, "sharpe": sharpe,
            "sharpe_rf4": (mu - RF) / (sd + 1e-12),
            "max_dd": dd, "calmar": mu / (abs(dd) + 1e-12)}


def zoo_target() -> pd.Series:
    rets = load_openap_returns()
    doc = load_openap_doc().set_index("Acronym")
    meta = doc.loc[doc.index.intersection(rets.columns), ["Year"]].astype(float)
    rets = rets[meta.index]
    age = rets.index.year.values[:, None] - meta.Year.values
    pub = pd.DataFrame(age >= 1, index=rets.index, columns=rets.columns)
    elig = (pub & rets.notna()).shift(1).fillna(False)
    n = (elig.sum(axis=1) >= MIN_ELIGIBLE)
    ew = rets.where(elig).mean(axis=1)[n]
    lev = (TARGET_VOL / (ew.rolling(12).std() * np.sqrt(12))).clip(upper=3).shift(1)
    return (ew * lev).dropna()


def backtest(weights: pd.DataFrame, rets: pd.DataFrame) -> pd.Series:
    """weights known at month t, earn over t+1; 5bp on traded notional."""
    w = weights.shift(1).reindex(rets.index).fillna(0.0)
    gross = (w * rets).sum(axis=1)
    cash = (1.0 - w.sum(axis=1)).clip(lower=0) * RF / 12
    turn = w.diff().abs().sum(axis=1).fillna(0.0)
    return gross + cash - TC * turn


def main():
    px = pd.read_csv(DATA / "etf_universe_prices.csv.gz", index_col=0,
                     parse_dates=True)
    tickers = [t for g in GROUPS.values() for t in g]
    rets = px[tickers].resample("ME").last().pct_change().iloc[1:]
    rets = rets.loc[rets.notna().all(axis=1).idxmax():]
    logger.info(f"reachable universe: {len(tickers)} ETFs, "
                f"{rets.index[0].date()} → {rets.index[-1].date()}")

    inv = (1.0 / rets.rolling(VOL_WIN, min_periods=VOL_MIN).std())

    # flat inverse-vol
    w_flat = inv.div(inv.sum(axis=1), axis=0)

    # group risk parity (+ defensive variant): equal/budgeted risk per
    # group, inverse-vol within group
    def grp_weights(budget):
        W = pd.DataFrame(0.0, index=rets.index, columns=tickers)
        for g, members in GROUPS.items():
            gi = inv[members]
            within = gi.div(gi.sum(axis=1), axis=0)
            W[members] = within.mul(budget[g], axis=0)
        return W.div(W.sum(axis=1), axis=0)

    eq_budget = {g: 1.0 / len(GROUPS) for g in GROUPS}
    w_grp = grp_weights(eq_budget)
    w_def = grp_weights(DEFENSIVE_BUDGET)

    books = {
        "SPY buy & hold": backtest(
            pd.DataFrame({"SPY": 1.0}, index=rets.index).reindex(
                columns=tickers).fillna(0.0), rets),
        "60/40 (SPY/IEF)": backtest(
            pd.DataFrame({"SPY": 0.6, "IEF": 0.4}, index=rets.index).reindex(
                columns=tickers).fillna(0.0), rets),
        "inverse-vol (flat)": backtest(w_flat, rets),
        "group risk parity": backtest(w_grp, rets),
        "GRP defensive tilt": backtest(w_def, rets),
    }
    # vol-capped (long-only: scale down only) versions of the two RP books
    for name, W in [("group risk parity", w_grp),
                    ("GRP defensive tilt", w_def)]:
        base = backtest(W, rets)
        scale = (TARGET_VOL / (base.rolling(VOL_WIN, min_periods=VOL_MIN).std()
                               * np.sqrt(12))).clip(upper=1.0).shift(1)
        books[f"{name} @vol-cap"] = backtest(W.mul(scale, axis=0), rets)

    ff = pd.read_csv(DATA / "ff_factors_monthly.csv", index_col=0,
                     parse_dates=True)
    rf_m = ff["RF"].reindex(rets.index).fillna(0.0)
    target = zoo_target().reindex(rets.index)
    spy = books["SPY buy & hold"]
    rows, raw = [], []
    for name, r in books.items():
        m = metrics(r, rf_m)
        zc = r.corr(target)
        sc = r.corr(spy) if name != "SPY buy & hold" else 1.0
        rows.append([name, f"{m['ann_ret']:+.2%}", f"{m['ann_vol']:.2%}",
                     f"{m['sharpe']:.2f}", f"{m['sharpe_rf4']:.2f}",
                     f"{m['max_dd']:.0%}", f"{m['calmar']:.2f}",
                     f"{sc:+.2f}", f"{zc:+.2f}"])
        raw.append({"book": name, **m, "corr_spy": sc, "corr_zoo": zc})

    print("\nReachable long-only books (NET of 5bp turnover, no leverage):")
    print(tabulate(rows, headers=["book", "ann ret", "vol", "Sharpe*",
                                  "SR@rf4", "maxDD", "Calmar", "ρ SPY",
                                  "ρ zoo"], tablefmt="github"))
    print("  * Sharpe on realized monthly RF (fair across the zero-rate "
          "decade); SR@rf4 = flat-4% convention for comparison")
    print(f"\n  (ρ zoo = corr with the unreachable levered EW zoo over the "
          f"overlap; RF {RF:.0%}, period {rets.index[0].date()}→"
          f"{rets.index[-1].date()})")

    # 2022 stress year — where diversification should show
    y22 = {n: (1 + r[r.index.year == 2022]).prod() - 1 for n, r in books.items()}
    print("\n2022 (the rate shock, diversification's exam):")
    print("  " + "   ".join(f"{n.split(' @')[0][:14]}: {v:+.0%}"
                            for n, v in y22.items()))

    out = RESULTS_DIR / "reachable_frontier.csv"
    pd.DataFrame(raw).to_csv(out, index=False)
    logger.info(f"\nSaved {out}")
    print("\n⚠ 2011→ ETF history = one regime + 2022; Sharpes indicative. "
          "NET of costs and fully reachable — unlike the gross zoo studies.")


if __name__ == "__main__":
    main()
