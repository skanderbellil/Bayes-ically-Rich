#!/usr/bin/env python3
"""
Fragility-WEIGHTED liquidity vote — feed the conditional insight into construction.

The interaction study (run_liquidity_interaction) found that net liquidity only
predicts forward equity returns when the market is FRAGILE: corr(Δ6m liquidity,
next-6m SPY) = +0.34 in high-VIX states vs -0.11 when calm.  Equal-weighting
liquidity with the other layers therefore dilutes a good signal in stress and
adds a bad one in calm.  The fix the mechanism dictates is to make liquidity's
*vote weight* equal to its reliability — i.e. to fragility itself:

    vote = (1 - f)·mean(vix_ts, credit, dollar)  +  f·liq

  f = 0  (calm)    → liquidity gets ZERO weight  (matches corr -0.11)
  f = 1  (fragile) → vote leans on liquidity      (matches corr +0.34)

There is no tunable parameter: the weight *is* the regime probability.  This is
also exactly what the funding-liquidity literature says — high VIX = low funding
liquidity, and the VIX curve inverts into backwardation under stress — so we
build f from the two cited mechanisms, both causal:

  vix_pct   : rolling 3y percentile rank of VIX        (level: high = fragile)
  backwrd   : 1 - soft_sign(VIX3M/VIX - 1)             (curve: inverted = fragile)
  f = ½·vix_pct + ½·backwrd                            (lagged 1 day)

Tested head-to-head against the incumbent equal-weight 4-layer vote, levered 2x
on QQQ with honest financing / trading costs, on the 2007→ stress panel (so the
GFC and the 2022 grind — the incumbent's weak spot — are both in view).

    python experiments/run_liquidity_fragility_weighted.py
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
from posterioralpha.research import liquidity_vote, soft_sign
from posterioralpha.validation import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)
logger = logging.getLogger(__name__)
RESULTS_DIR = _bootstrap.ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
PANEL_CSV = _bootstrap.ROOT / "datasets" / "stress_panel.csv.gz"

RF = 0.04
DAILY_RF = RF / 252
BORROW_SPREAD = 0.005
TC = 0.0002
LEV_CAP = 2.0
ROLL = 756
VIX_PCT_WIN = 756       # 3y rolling percentile for the VIX level


def apply_2xvote(under_ret, vote):
    expo = (2.0 * vote).clip(0.0, LEV_CAP).fillna(0.0)
    gross = expo * under_ret
    fin = np.maximum(expo - 1.0, 0.0) * (RF + BORROW_SPREAD) / 252
    cash = np.maximum(1.0 - expo, 0.0) * DAILY_RF
    costs = TC * expo.diff().abs().fillna(0.0)
    return pd.Series(gross - fin + cash - costs, index=under_ret.index), expo


def roll_cagr(r):
    return (1 + r).rolling(ROLL).apply(np.prod, raw=True) ** (252 / ROLL) - 1


def roll_sharpe(r):
    ex = r - DAILY_RF
    return (ex.rolling(ROLL).mean() / ex.rolling(ROLL).std()) * np.sqrt(252)


def metrics_row(name, r):
    m = compute_metrics(r.dropna(), rf=RF)
    keys = ["CAGR", "Volatility", "Sharpe", "Sortino", "Max DD", "Calmar"]
    gfc = (1 + r[(r.index >= "2008-09-01") & (r.index <= "2009-06-30")]).prod() - 1
    y22 = (1 + r[r.index.year == 2022]).prod() - 1
    return [name] + [f"{m.get(k, float('nan')):.2f}" for k in keys] + \
           [f"{gfc:+.0%}", f"{y22:+.0%}"]


def main() -> None:
    nl = load_net_liquidity()["net_liquidity"]
    p = pd.read_csv(PANEL_CSV, parse_dates=["Date"], index_col="Date").sort_index()
    idx = nl.dropna().index.intersection(
        p.dropna(subset=["SPY", "QQQ", "HYG", "IEF", "UUP", "VIX", "VIX3M"]).index)
    nl, p = nl.reindex(idx), p.reindex(idx)

    # ── Causal layers ─────────────────────────────────────────────────────────
    base = liquidity_vote(nl, p["VIX"], p["VIX3M"], p["HYG"], p["IEF"])
    dollar = soft_sign(-p["UUP"].pct_change(63)).shift(1)
    liq = base["liq"]
    market = pd.concat([base["vix_ts"], base["credit"], dollar.rename("d")],
                       axis=1).mean(axis=1)          # always-on risk gauges

    # ── Fragility f ∈ [0,1] (the two cited mechanisms, both causal) ───────────
    vix_pct = p["VIX"].rolling(VIX_PCT_WIN, min_periods=252).apply(
        lambda x: (x < x[-1]).mean(), raw=True)       # level: high = fragile
    backwrd = 1.0 - base["vix_ts"]                     # curve inverted = fragile
    frag = (0.5 * vix_pct + 0.5 * backwrd.reindex(idx)).shift(1).clip(0, 1)

    # ── Votes ─────────────────────────────────────────────────────────────────
    equal_vote = pd.concat([liq, market.rename("m")], axis=1).mean(axis=1)  # incumbent feel
    # incumbent is the literal 4-layer equal mean of liq,vix_ts,credit,dollar:
    equal_vote = pd.concat([base[["liq", "vix_ts", "credit"]],
                            dollar.rename("d")], axis=1).mean(axis=1)
    frag_vote = ((1 - frag) * market + frag * liq)     # fragility-weighted

    qqq_ret, spy_ret = p["QQQ"].pct_change(), p["SPY"].pct_change()
    live = pd.concat([equal_vote, frag_vote, frag], axis=1).dropna().index.min()
    ev = idx[idx >= live]
    logger.info("both votes live (signals matured): %s", live.date())

    lev_eq, e_eq = apply_2xvote(qqq_ret, equal_vote)
    lev_fr, e_fr = apply_2xvote(qqq_ret, frag_vote)
    lev_fr_spy, _ = apply_2xvote(spy_ret, frag_vote)

    rets = {
        "SPY buy & hold": spy_ret.reindex(ev),
        "QQQ buy & hold": qqq_ret.reindex(ev),
        "2×equal-weight vote on QQQ (incumbent)": lev_eq.reindex(ev),
        "2×fragility-weighted vote on QQQ": lev_fr.reindex(ev),
        "2×fragility-weighted vote on SPY": lev_fr_spy.reindex(ev),
    }

    print("\n" + "=" * 96)
    print(f"  FRAGILITY-WEIGHTED LIQUIDITY VOTE  ·  {live.date()} → {ev.max().date()}")
    print("  vote = (1-f)·mean(vix_ts,credit,dollar) + f·liq   ·   f = ½ VIX-pct + ½ backwardation")
    print("=" * 96)
    keys = ["CAGR", "Volatility", "Sharpe", "Sortino", "Max DD", "Calmar"]
    print(tabulate([metrics_row(n, r) for n, r in rets.items()],
                   headers=["strategy"] + keys + ["GFC", "2022"], tablefmt="github"))
    print(f"\n  avg exposure — equal {float(e_eq.reindex(ev).mean()):.2f}x | "
          f"fragility-weighted {float(e_fr.reindex(ev).mean()):.2f}x")
    print(f"  liquidity's effective weight — calm(f<0.33) {float(frag[frag < 0.33].mean()):.2f} | "
          f"fragile(f>0.67) {float(frag[frag > 0.67].mean()):.2f}")

    # ── Rolling OOS vs SPY (the bar) ─────────────────────────────────────────
    print("\n  Rolling 3y win vs SPY (CAGR / Sharpe):")
    for n in ["2×equal-weight vote on QQQ (incumbent)", "2×fragility-weighted vote on QQQ"]:
        dc = (roll_cagr(rets[n]) - roll_cagr(rets["SPY buy & hold"])).dropna()
        ds = (roll_sharpe(rets[n]) - roll_sharpe(rets["SPY buy & hold"])).dropna()
        print(f"    {n:42s} CAGR {float((dc > 0).mean()):.0%} (med {float(dc.median()):+.1%}/yr)"
              f"   Sharpe {float((ds > 0).mean()):.0%} (med {float(ds.median()):+.2f})")

    # ── Graphs ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 13))
    ax = axes[0]
    for n, r in rets.items():
        eqc = (1 + r.fillna(0)).cumprod()
        ax.plot(eqc.index, eqc.values, lw=1.6 if "buy" in n else 1.3,
                color={"SPY buy & hold": "black", "QQQ buy & hold": "grey"}.get(n),
                alpha=0.9, label=n)
    ax.set_yscale("log"); ax.set_ylabel("growth of $1 (log)")
    ax.set_title("Fragility-weighted vs equal-weight liquidity vote (2× on QQQ)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    for n in ["SPY buy & hold", "2×equal-weight vote on QQQ (incumbent)",
              "2×fragility-weighted vote on QQQ"]:
        eqc = (1 + rets[n].fillna(0)).cumprod()
        dd = eqc / eqc.cummax() - 1
        ax.plot(dd.index, dd.values * 100, lw=1.1, label=n)
    ax.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31"),
               color="red", alpha=0.08, label="2022 grind")
    ax.set_ylabel("drawdown (%)"); ax.legend(fontsize=8)
    ax.set_title("Drawdowns — does fragility-weighting patch the 2022 hole?")
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(frag.reindex(ev).index, frag.reindex(ev).values, lw=0.8,
            color="tab:red", label="fragility f  (= liquidity's vote weight)")
    ax.plot(e_fr.reindex(ev).index, (e_fr.reindex(ev) / 2).values, lw=0.7,
            color="tab:blue", alpha=0.7, label="frag-vote exposure ÷2")
    for span in [("2008-09-01", "2009-06-30"), ("2022-01-01", "2022-12-31")]:
        ax.axvspan(pd.Timestamp(span[0]), pd.Timestamp(span[1]), color="red", alpha=0.08)
    ax.set_ylabel("weight / exposure"); ax.legend(fontsize=8)
    ax.set_title("Fragility f over time (GFC & 2022 shaded) — liquidity earns weight only in stress")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = RESULTS_DIR / "liquidity_fragility_weighted.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    logger.info("Saved: %s", out)
    print("\n✓  Done.")


if __name__ == "__main__":
    main()
