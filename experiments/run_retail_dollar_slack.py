#!/usr/bin/env python3
"""
The dollar slack at retail scale — three tickers, no margin account.

run_retail_implementation found the simplest implementation tested was the
best implementation tested: w = vote in QLD (real 2× daily-reset prints,
financing embedded) + (1−w) in cash.  run_dollar_slack then promoted the
risk-off-leg finding: the slack belongs in UUP, not cash.  This study closes
the loop at retail scale — QLD + UUP + nothing else, all real fund prints:

  w = vote in QLD   ·   (1−w) in UUP (or ½ UUP / ½ cash)

The UUP leg is itself an ETF, so the whole strategy is THREE tickers and
one weight — still no margin call, now with the risk-off asset that the
fleeing flows actually flow into.  GFC-inclusive (signals live 2008-04),
same A/B on SPY/SSO, ceiling ×½ shown as a stacked variant only (its edge
is concentrated — see run_adaptive_vote).

    python experiments/run_retail_dollar_slack.py
"""
import logging
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tabulate import tabulate

import _bootstrap  # noqa: F401
from posterioralpha.data import load_net_liquidity
from posterioralpha.research import soft_sign
from posterioralpha.validation import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)
logger = logging.getLogger(__name__)
RESULTS_DIR = _bootstrap.ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
PANEL_CSV = _bootstrap.ROOT / "datasets" / "overnight_panel.csv.gz"
STRESS_CSV = _bootstrap.ROOT / "datasets" / "stress_panel.csv.gz"
LEV_CSV = _bootstrap.ROOT / "datasets" / "levered_etfs.csv.gz"

RF = 0.04
DAILY_RF = RF / 252
TC = 0.0002
ROLL = 756
H_LIQ, PUB_LAG, CREDIT_WIN = 126, 5, 21

DEBT_CEILING = ["2011-08-02", "2013-10-17", "2014-02-15", "2015-11-02",
                "2017-09-08", "2018-02-09", "2019-08-02", "2021-10-14",
                "2021-12-16", "2023-06-03"]   # complete through Jun-2023


def roll_cagr(r):
    return (1 + r).rolling(ROLL).apply(np.prod, raw=True) ** (252 / ROLL) - 1


def roll_sharpe(r):
    ex = r - DAILY_RF
    return (ex.rolling(ROLL).mean() / ex.rolling(ROLL).std()) * np.sqrt(252)


def main() -> None:
    nl = load_net_liquidity()["net_liquidity"]
    p = pd.read_csv(STRESS_CSV, parse_dates=["Date"], index_col="Date").sort_index()
    panel = pd.read_csv(PANEL_CSV, parse_dates=["Date"], index_col="Date").sort_index()
    lev = pd.read_csv(LEV_CSV, parse_dates=["Date"], index_col="Date").sort_index()
    idx = (nl.dropna().index
           .intersection(p[["SPY", "QQQ", "HYG", "IEF", "UUP", "VIX", "VIX3M"]]
                         .dropna().index)
           .intersection(panel.dropna().index)
           .intersection(lev.dropna().index))
    nl, p, lev = nl.reindex(idx), p.reindex(idx), lev.reindex(idx)

    liq = soft_sign(nl.diff(H_LIQ).shift(PUB_LAG)).shift(1)
    vix_ts = soft_sign(p["VIX3M"] / p["VIX"] - 1.0).shift(1)
    credit = soft_sign(p["HYG"].pct_change(CREDIT_WIN)
                       - p["IEF"].pct_change(CREDIT_WIN)).shift(1)
    dollar = soft_sign(-p["UUP"].pct_change(63)).shift(1)

    def vote_for(t):
        o, c = panel[f"{t}_Open"].reindex(idx), panel[f"{t}_Close"].reindex(idx)
        r_on, r_id = o / c.shift(1) - 1.0, c / o - 1.0
        tug = soft_sign(((1 + r_on).rolling(21).apply(np.prod, raw=True)
                         - (1 + r_id).rolling(21).apply(np.prod, raw=True))).shift(1)
        return pd.DataFrame({"liq": liq, "vix_ts": vix_ts, "credit": credit,
                             "dollar": dollar, "tug": tug}).mean(axis=1)

    v_spy, v_qqq = vote_for("SPY"), vote_for("QQQ")
    r_spy, r_qqq = p["SPY"].pct_change(), p["QQQ"].pct_change()
    r_sso, r_qld = lev["SSO"].pct_change(), lev["QLD"].pct_change()
    r_uup = p["UUP"].pct_change()

    dc_mask = pd.Series(False, index=idx)
    for d in pd.to_datetime(DEBT_CEILING):
        i = idx.searchsorted(d)
        if 0 < i < len(idx):
            dc_mask.iloc[i:min(i + 61, len(idx))] = True
    dc_mask = dc_mask.shift(1).fillna(False)

    def letf_run(vote, r_letf, frac_uup=0.0, ceiling=False):
        w = vote.clip(0.0, 1.0)
        if ceiling:
            w = w * np.where(dc_mask, 0.5, 1.0)
        w = pd.Series(w, index=idx).fillna(0.0)
        w_uup = (1.0 - w) * frac_uup
        gross = w * r_letf + w_uup * r_uup + (1.0 - w - w_uup) * DAILY_RF
        costs = TC * (w.diff().abs().fillna(0.0) + w_uup.diff().abs().fillna(0.0))
        return gross - costs

    live = max(v_spy.first_valid_index(), v_qqq.first_valid_index())
    strat = {
        "SPY buy & hold": r_spy,
        "QQQ buy & hold": r_qqq,
        "QQQ/QLD + cash (prior best)": letf_run(v_qqq, r_qld, 0.0),
        "QQQ/QLD + ½UUP": letf_run(v_qqq, r_qld, 0.5),
        "QQQ/QLD + UUP": letf_run(v_qqq, r_qld, 1.0),
        "QQQ/QLD + UUP + ceiling ×½": letf_run(v_qqq, r_qld, 1.0, ceiling=True),
        "SPY/SSO + cash": letf_run(v_spy, r_sso, 0.0),
        "SPY/SSO + UUP": letf_run(v_spy, r_sso, 1.0),
    }
    strat = {k: v.loc[live:] for k, v in strat.items()}
    spy_live = r_spy.loc[live:]

    print("\n" + "=" * 100)
    print(f"  THE DOLLAR SLACK AT RETAIL SCALE  ·  real QLD/SSO/UUP prints  ·  "
          f"{live.date()} → {idx[-1].date()}  (GFC-inclusive)")
    print("=" * 100)
    keys = ["CAGR", "Volatility", "Sharpe", "Sortino", "Max DD", "Calmar"]
    table = []
    for name, r in strat.items():
        m = compute_metrics(r.dropna(), rf=RF)
        gfc = (1 + r.loc["2008-09-01":"2009-06-30"]).prod() - 1
        ret22 = (1 + r[r.index.year == 2022]).prod() - 1
        dc_ = (roll_cagr(r) - roll_cagr(spy_live)).dropna()
        ds_ = (roll_sharpe(r) - roll_sharpe(spy_live)).dropna()
        table.append([name] + [f"{m.get(k, float('nan')):.2f}" for k in keys]
                     + [f"{gfc:+.0%}", f"{ret22:+.0%}",
                        f"{float((dc_ > 0).mean()):.0%}",
                        f"{float((ds_ > 0).mean()):.0%}"])
    print(tabulate(table, headers=["strategy"] + keys
                   + ["GFC", "2022", "CAGRwin", "SRwin"], tablefmt="github"))
    print("  GFC = Sep-08→Jun-09  ·  win rates vs SPY B&H, rolling 3y windows")

    yr = pd.DataFrame({
        "QLD+cash": strat["QQQ/QLD + cash (prior best)"],
        "QLD+UUP": strat["QQQ/QLD + UUP"],
        "QLD+UUP+ceil": strat["QQQ/QLD + UUP + ceiling ×½"],
        "QQQ B&H": strat["QQQ buy & hold"]}).dropna()
    ann = yr.groupby(yr.index.year).apply(lambda d: (1 + d).prod() - 1)
    print("\nPer-year returns (%):")
    print(tabulate((ann * 100).round(1), headers=["year"] + list(ann.columns),
                   tablefmt="github"))

    # ── Graph ────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    show = [("QQQ/QLD + UUP + ceiling ×½", "#B71C1C", 2.0),
            ("QQQ/QLD + UUP", "#1A237E", 1.8),
            ("QQQ/QLD + cash (prior best)", "#00695C", 1.4),
            ("QQQ buy & hold", "grey", 1.3),
            ("SPY buy & hold", "black", 1.3)]
    for name, c, lw in show:
        eq = (1 + strat[name].fillna(0)).cumprod()
        ax1.plot(eq.index, eq.values, color=c, lw=lw,
                 label=f"{name}  (${eq.iloc[-1]:,.0f})")
    ax1.set_yscale("log"); ax1.set_ylabel("growth of $1 (log)")
    ax1.set_title("Three tickers, one weight: vote in QLD, slack in UUP "
                  "(real prints, GFC-inclusive)", fontweight="bold")
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3)
    for name, c, lw in show:
        eq = (1 + strat[name].fillna(0)).cumprod()
        dd = eq / eq.cummax() - 1
        ax2.plot(dd.index, dd.values, color=c, lw=lw)
    ax2.set_ylabel("drawdown"); ax2.grid(alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    fig.tight_layout()
    out = RESULTS_DIR / "retail_dollar_slack.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    logger.info("Saved: %s", out)
    print("\n✓  Done.")


if __name__ == "__main__":
    main()
