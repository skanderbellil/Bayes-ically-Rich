#!/usr/bin/env python3
"""
Retail reality check — the 5-layer vote on SPY and QQQ, three ways to get 2×.

Everything so far finances leverage at RF+50bp — an institutional fiction.
A retail account gets 2× exposure three ways, each with different costs:

  margin 50bp   : the thread's standard (futures-like; kept as the benchmark)
  margin 150bp  : realistic retail broker margin (IBKR-tier)
  levered ETF   : hold w = vote in SSO/QLD (2× daily reset) + (1−w) in cash.
                  exposure = 2×vote exactly, financing + expense ratio are
                  *embedded in the real fund prints* (actual SSO/QLD history,
                  2006→), daily-reset variance drag included by construction.
                  This is the simplest thing a retail account can actually do:
                  two tickers, one weight, no margin call.

Underlyings: SPY and QQQ, each with its own tug layer (the other four layers
are shared).  GFC-inclusive sample (stress panel, signals live 2008-04).
The ceiling overlay is shown only as a variant — its edge is concentrated
(see run_adaptive_vote autopsy) so the core claim must stand without it.

    python experiments/run_retail_implementation.py
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
LEV_CAP = 2.0
ROLL = 756
H_LIQ, PUB_LAG, CREDIT_WIN = 126, 5, 21


def margin_run(expo, r_under, borrow_spread):
    expo = expo.clip(0.0, LEV_CAP).fillna(0.0)
    gross = expo * r_under
    financing = np.maximum(expo - 1.0, 0.0) * (RF + borrow_spread) / 252
    cash = np.maximum(1.0 - expo, 0.0) * DAILY_RF
    costs = TC * expo.diff().abs().fillna(0.0)
    return pd.Series(gross - financing + cash - costs, index=r_under.index)


def letf_run(vote, r_letf):
    """w = vote in the 2× fund, 1−w in cash earning RF.  Costs on Δw."""
    w = vote.clip(0.0, 1.0).fillna(0.0)
    gross = w * r_letf + (1.0 - w) * DAILY_RF
    costs = TC * w.diff().abs().fillna(0.0)
    return pd.Series(gross - costs, index=r_letf.index)


def roll_sharpe(r):
    ex = r - DAILY_RF
    return (ex.rolling(ROLL).mean() / ex.rolling(ROLL).std()) * np.sqrt(252)


def roll_cagr(r):
    return (1 + r).rolling(ROLL).apply(np.prod, raw=True) ** (252 / ROLL) - 1


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

    # Shared layers
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
    r_spy = p["SPY"].pct_change()
    r_qqq = p["QQQ"].pct_change()
    r_sso = lev["SSO"].pct_change()
    r_qld = lev["QLD"].pct_change()

    live = max(v_spy.first_valid_index(), v_qqq.first_valid_index())
    strat = {
        "SPY buy & hold": r_spy,
        "QQQ buy & hold": r_qqq,
        "SPY: 2×vote margin 50bp": margin_run(2 * v_spy, r_spy, 0.005),
        "SPY: 2×vote margin 150bp": margin_run(2 * v_spy, r_spy, 0.015),
        "SPY: vote in SSO + cash": letf_run(v_spy, r_sso),
        "QQQ: 2×vote margin 50bp": margin_run(2 * v_qqq, r_qqq, 0.005),
        "QQQ: 2×vote margin 150bp": margin_run(2 * v_qqq, r_qqq, 0.015),
        "QQQ: vote in QLD + cash": letf_run(v_qqq, r_qld),
    }
    strat = {k: v.loc[live:] for k, v in strat.items()}
    spy_live = r_spy.loc[live:]

    print("\n" + "=" * 100)
    print(f"  RETAIL IMPLEMENTATION  ·  5-layer vote  ·  {live.date()} → "
          f"{idx[-1].date()}  (GFC-inclusive)")
    print("=" * 100)
    keys = ["CAGR", "Volatility", "Sharpe", "Sortino", "Max DD", "Calmar"]
    table = []
    for name, r in strat.items():
        m = compute_metrics(r.dropna(), rf=RF)
        gfc = (1 + r.loc["2008-09-01":"2009-06-30"]).prod() - 1
        ret22 = (1 + r[r.index.year == 2022]).prod() - 1
        dc_ = (roll_cagr(r) - roll_cagr(spy_live)).dropna()
        table.append([name] + [f"{m.get(k, float('nan')):.2f}" for k in keys]
                     + [f"{gfc:+.0%}", f"{ret22:+.0%}",
                        f"{float((dc_ > 0).mean()):.0%}"])
    print(tabulate(table, headers=["strategy"] + keys
                   + ["GFC", "2022", "CAGRwin"], tablefmt="github"))
    print("  GFC = Sep-08→Jun-09  ·  CAGRwin = rolling 3y windows beating SPY B&H")

    # Cost of retail vs fiction, in CAGR
    print("\n  What retail costs you (CAGR give-up vs the 50bp fiction):")
    for t in ["SPY", "QQQ"]:
        c50 = compute_metrics(strat[f"{t}: 2×vote margin 50bp"].dropna(), rf=RF)["CAGR"]
        c150 = compute_metrics(strat[f"{t}: 2×vote margin 150bp"].dropna(), rf=RF)["CAGR"]
        cl = compute_metrics(strat[f"{t}: vote in {'SSO' if t == 'SPY' else 'QLD'} + cash"]
                             .dropna(), rf=RF)["CAGR"]
        print(f"    {t}: margin150bp {1e2 * (c150 - c50):+.1f}pp  ·  "
              f"levered-ETF {1e2 * (cl - c50):+.1f}pp")

    # Per-year for the headline retail configs
    yr = pd.DataFrame({
        "QQQ/QLD retail": strat["QQQ: vote in QLD + cash"],
        "SPY/SSO retail": strat["SPY: vote in SSO + cash"],
        "QQQ B&H": strat["QQQ buy & hold"],
        "SPY B&H": strat["SPY buy & hold"]}).dropna()
    ann = yr.groupby(yr.index.year).apply(lambda d: (1 + d).prod() - 1)
    print("\nPer-year returns (%):")
    print(tabulate((ann * 100).round(1), headers=["year"] + list(ann.columns),
                   tablefmt="github"))

    # ── Graph ────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    show = [("QQQ: vote in QLD + cash", "#1A237E", 2.0),
            ("SPY: vote in SSO + cash", "#00695C", 1.6),
            ("QQQ: 2×vote margin 150bp", "#7B1FA2", 1.2),
            ("QQQ buy & hold", "grey", 1.3),
            ("SPY buy & hold", "black", 1.3)]
    for name, c, lw in show:
        eq = (1 + strat[name].fillna(0)).cumprod()
        ax1.plot(eq.index, eq.values, color=c, lw=lw,
                 label=f"{name}  (${eq.iloc[-1]:,.0f})")
    ax1.set_yscale("log"); ax1.set_ylabel("growth of $1 (log)")
    ax1.set_title("Retail-accessible implementations of the 5-layer vote "
                  "(GFC-inclusive, real fund prints)", fontweight="bold")
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3)
    for name, c, lw in show:
        eq = (1 + strat[name].fillna(0)).cumprod()
        dd = eq / eq.cummax() - 1
        ax2.plot(dd.index, dd.values, color=c, lw=lw)
    ax2.set_ylabel("drawdown"); ax2.grid(alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    fig.tight_layout()
    out = RESULTS_DIR / "retail_implementation.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    logger.info("Saved: %s", out)
    print("\n✓  Done.")


if __name__ == "__main__":
    main()
