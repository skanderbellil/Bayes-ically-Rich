#!/usr/bin/env python3
"""
The dollar slack — promotion test for the risk-off leg finding.

run_riskoff_leg found that parking the champion's slack in UUP instead of
cash upgrades every tail metric and survives an out-of-selection
GFC-inclusive check.  Before promotion, three questions must close:

  1. REDUNDANCY     The vote already has a dollar LAYER — soft-signed −UUP
                    63d momentum as a risk signal.  Is the dollar SLACK
                    (holding USD as an asset when de-risked) the same
                    information twice?  Four cells: {4-layer no-dollar,
                    5-layer} × {slack cash, slack UUP}.  If the slack helps
                    without the layer AND the layer helps with the slack,
                    they are complementary (flow signal vs asset holding).
  2. THE FULL STACK Project-best config (5-layer + debt-ceiling ×½) with
                    the dollar slack.  Note the interaction is mechanism-
                    aligned: ceiling windows halve exposure, which GROWS the
                    slack exactly when TGA rebuilds strengthen the dollar.
  3. REPLICATION    Same A/B on SPY as the underlying (GFC-inclusive), plus
                    a half-slack (50% UUP / 50% cash) implementation-
                    robustness row.

    python experiments/run_dollar_slack.py
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

RF = 0.04
DAILY_RF = RF / 252
BORROW_SPREAD = 0.005
TC = 0.0002
LEV_CAP = 2.0
ROLL = 756
H_LIQ, PUB_LAG, CREDIT_WIN = 126, 5, 21

DEBT_CEILING = ["2011-08-02", "2013-10-17", "2014-02-15", "2015-11-02",
                "2017-09-08", "2018-02-09", "2019-08-02", "2021-10-14",
                "2021-12-16", "2023-06-03"]   # complete through Jun-2023


def roll_sharpe(r):
    ex = r - DAILY_RF
    return (ex.rolling(ROLL).mean() / ex.rolling(ROLL).std()) * np.sqrt(252)


def roll_cagr(r):
    return (1 + r).rolling(ROLL).apply(np.prod, raw=True) ** (252 / ROLL) - 1


def main() -> None:
    nl = load_net_liquidity()["net_liquidity"]
    p = pd.read_csv(STRESS_CSV, parse_dates=["Date"], index_col="Date").sort_index()
    panel = pd.read_csv(PANEL_CSV, parse_dates=["Date"], index_col="Date").sort_index()
    idx = (nl.dropna().index
           .intersection(p[["SPY", "QQQ", "HYG", "IEF", "UUP", "VIX", "VIX3M"]]
                         .dropna().index)
           .intersection(panel.dropna().index))
    nl, p = nl.reindex(idx), p.reindex(idx)
    r_qqq = p["QQQ"].pct_change()
    r_spy = p["SPY"].pct_change()
    r_uup = p["UUP"].pct_change()

    # ── The layers (house construction) ──────────────────────────────────────
    liq = soft_sign(nl.diff(H_LIQ).shift(PUB_LAG)).shift(1)
    vix_ts = soft_sign(p["VIX3M"] / p["VIX"] - 1.0).shift(1)
    credit = soft_sign(p["HYG"].pct_change(CREDIT_WIN)
                       - p["IEF"].pct_change(CREDIT_WIN)).shift(1)
    dollar = soft_sign(-p["UUP"].pct_change(63)).shift(1)
    q_o, q_c = panel["QQQ_Open"].reindex(idx), panel["QQQ_Close"].reindex(idx)
    r_on, r_id = q_o / q_c.shift(1) - 1.0, q_c / q_o - 1.0
    tug = soft_sign(((1 + r_on).rolling(21).apply(np.prod, raw=True)
                     - (1 + r_id).rolling(21).apply(np.prod, raw=True))).shift(1)
    vote5 = pd.DataFrame({"liq": liq, "vix_ts": vix_ts, "credit": credit,
                          "dollar": dollar, "tug": tug}).mean(axis=1)
    vote4 = pd.DataFrame({"liq": liq, "vix_ts": vix_ts, "credit": credit,
                          "tug": tug}).mean(axis=1)        # no dollar layer

    dc_mask = pd.Series(False, index=idx)
    for d in pd.to_datetime(DEBT_CEILING):
        i = idx.searchsorted(d)
        if 0 < i < len(idx):
            dc_mask.iloc[i:min(i + 61, len(idx))] = True
    dc_mask = dc_mask.shift(1).fillna(False)

    def run(vote, r_under, slack_frac_uup=0.0, ceiling=False):
        e = (2.0 * vote).clip(0.0, LEV_CAP)
        if ceiling:
            e = e * np.where(dc_mask, 0.5, 1.0)
        e = pd.Series(e, index=idx).fillna(0.0)
        slack = np.maximum(1.0 - e, 0.0)
        w_uup = slack * slack_frac_uup
        gross = (e * r_under + w_uup * r_uup
                 + (slack - w_uup) * DAILY_RF)
        financing = np.maximum(e - 1.0, 0.0) * (RF + BORROW_SPREAD) / 252
        turn = e.diff().abs().fillna(0.0) + w_uup.diff().abs().fillna(0.0)
        return gross - financing - TC * turn

    live = max(vote5.first_valid_index(), vote4.first_valid_index())

    def report(strat, bench_ret, title):
        bench_live = bench_ret.loc[live:]
        print(f"\n  {title}  ·  common live {live.date()}:")
        keys = ["CAGR", "Volatility", "Sharpe", "Sortino", "Max DD", "Calmar"]
        table = []
        for name, r in strat.items():
            r = r.loc[live:]
            m = compute_metrics(r.dropna(), rf=RF)
            gfc = (1 + r.loc["2008-09-01":"2009-06-30"]).prod() - 1
            ret22 = (1 + r[r.index.year == 2022]).prod() - 1
            dc = (roll_cagr(r) - roll_cagr(bench_live)).dropna()
            ds = (roll_sharpe(r) - roll_sharpe(bench_live)).dropna()
            table.append([name] + [f"{m.get(k, float('nan')):.2f}" for k in keys]
                         + [f"{gfc:+.0%}", f"{ret22:+.0%}",
                            f"{float((dc > 0).mean()):.0%}",
                            f"{float((ds > 0).mean()):.0%}"])
        print(tabulate(table, headers=["strategy"] + keys
                       + ["GFC", "2022", "CAGRwin", "SRwin"], tablefmt="github"))

    print("\n" + "=" * 100)
    print(f"  THE DOLLAR SLACK — PROMOTION TEST  ·  GFC-inclusive  ·  "
          f"{live.date()} → {idx[-1].date()}")
    print("=" * 100)

    # ── 1. Redundancy: layer × slack four cells ──────────────────────────────
    strat = {
        "QQQ buy & hold": r_qqq,
        "4L (no dollar layer) · slack cash": run(vote4, r_qqq, 0.0),
        "4L (no dollar layer) · slack UUP": run(vote4, r_qqq, 1.0),
        "5L (dollar layer)   · slack cash": run(vote5, r_qqq, 0.0),
        "5L (dollar layer)   · slack UUP": run(vote5, r_qqq, 1.0),
    }
    report(strat, r_spy, "1. REDUNDANCY — dollar layer × dollar slack (QQQ)")
    print("  read: slack effect without the layer (row 2 vs 1) and layer "
          "effect with the slack (row 4 vs 2)")

    # ── 2. The full stack ────────────────────────────────────────────────────
    strat = {
        "5L · slack cash (champion)": run(vote5, r_qqq, 0.0),
        "5L + ceiling ×½ · slack cash (best)": run(vote5, r_qqq, 0.0, ceiling=True),
        "5L + ceiling ×½ · slack UUP": run(vote5, r_qqq, 1.0, ceiling=True),
        "5L + ceiling ×½ · slack ½UUP": run(vote5, r_qqq, 0.5, ceiling=True),
    }
    report(strat, r_spy, "2. FULL STACK — project-best config + dollar slack (QQQ)")

    # ── 3. SPY replication ───────────────────────────────────────────────────
    strat = {
        "SPY buy & hold": r_spy,
        "SPY 5L · slack cash": run(vote5, r_spy, 0.0),
        "SPY 5L · slack UUP": run(vote5, r_spy, 1.0),
        "SPY 5L · slack ½UUP": run(vote5, r_spy, 0.5),
    }
    report(strat, r_spy, "3. REPLICATION — same A/B on SPY as underlying")

    # ── Graph ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    ax = axes[0]
    show = {"QQQ buy & hold": r_qqq,
            "5L · slack cash (champion)": run(vote5, r_qqq, 0.0),
            "5L + ceiling ×½ · slack cash (best)": run(vote5, r_qqq, 0.0, ceiling=True),
            "5L + ceiling ×½ · slack UUP": run(vote5, r_qqq, 1.0, ceiling=True)}
    for name, r in show.items():
        eq = (1 + r.loc[live:].fillna(0)).cumprod()
        ax.plot(eq.index, eq.values, lw=1.8 if "UUP" in name else 1.1,
                color="grey" if "buy" in name else None, label=name)
    ax.set_yscale("log"); ax.set_ylabel("growth of $1 (log)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("The dollar slack stacked on the project-best config")
    ax = axes[1]
    for name, r in show.items():
        eq = (1 + r.loc[live:].fillna(0)).cumprod()
        dd = eq / eq.cummax() - 1
        ax.plot(dd.index, dd.values, lw=1.1,
                color="grey" if "buy" in name else None, label=name)
    ax.set_ylabel("drawdown"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("Drawdowns — GFC and 2022")
    fig.tight_layout()
    out = RESULTS_DIR / "dollar_slack.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    logger.info("Saved: %s", out)
    print("\n✓  Done.")


if __name__ == "__main__":
    main()
