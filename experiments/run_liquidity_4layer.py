#!/usr/bin/env python3
"""
Fourth layer for the liquidity vote — closing the grinding-bear (2022) hole.

The 3-layer vote (liquidity × VIX-TS × credit) levered 2x on QQQ beats SPY in
84% of rolling 3y windows but suffers a -48% drawdown: 2022 was a low-vol grind
where all three signals sat near neutral.  This adds a fourth, *orthogonal*
risk gauge and tests — in one shot — whether it patches 2022 without giving
back the upside:

  dollar  : −UUP 63-day return  (a rising USD is risk-off; ripped through 2022)
  breadth : fraction of the 500-equity universe with positive 6-month returns,
            minus ½  (internal participation; deteriorates in grinding bears)

Each is mapped by the same fit-free soft sign as the other layers.  Compared on
the 2x-vote-on-QQQ configuration with honest financing / trading costs.

    python experiments/run_liquidity_4layer.py
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
from posterioralpha.data import (
    load_net_liquidity, load_etf_universe_prices, load_equity_universe_prices,
)
from posterioralpha.research import liquidity_vote, soft_sign
from posterioralpha.validation import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)
logger = logging.getLogger(__name__)
RESULTS_DIR = _bootstrap.ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
VIX_CSV = _bootstrap.ROOT / "datasets" / "vix_term_structure.csv"

RF = 0.04
DAILY_RF = RF / 252
BORROW_SPREAD = 0.005
TC = 0.0002
LEV_CAP = 2.0
ROLL = 756


def apply_2xvote(under_ret: pd.Series, vote: pd.Series):
    expo = (2.0 * vote).clip(0.0, LEV_CAP).fillna(0.0)
    gross = expo * under_ret
    financing = np.maximum(expo - 1.0, 0.0) * (RF + BORROW_SPREAD) / 252
    cash = np.maximum(1.0 - expo, 0.0) * DAILY_RF
    costs = TC * expo.diff().abs().fillna(0.0)
    return pd.Series(gross - financing + cash - costs, index=under_ret.index), expo


def roll_sharpe(r):
    ex = r - DAILY_RF
    return (ex.rolling(ROLL).mean() / ex.rolling(ROLL).std()) * np.sqrt(252)


def roll_cagr(r):
    return (1 + r).rolling(ROLL).apply(np.prod, raw=True) ** (252 / ROLL) - 1


def main() -> None:
    nl = load_net_liquidity()["net_liquidity"]
    px = load_etf_universe_prices()
    eq = load_equity_universe_prices()
    vix = pd.read_csv(VIX_CSV, parse_dates=["Date"], index_col="Date").sort_index()
    idx = (nl.dropna().index
           .intersection(px[["SPY", "QQQ", "HYG", "IEF", "UUP"]].dropna().index)
           .intersection(vix.dropna().index))
    nl, vix = nl.reindex(idx), vix.reindex(idx)
    qqq_ret = px["QQQ"].reindex(idx).pct_change()
    spy_ret = px["SPY"].reindex(idx).pct_change()

    # 3 base strengths (already causal: publication-lag + 1d)
    base = liquidity_vote(nl, vix["VIX"], vix["VIX3M"],
                          px["HYG"].reindex(idx), px["IEF"].reindex(idx))
    s3 = base[["liq", "vix_ts", "credit"]]

    # Layer 4a — dollar (rising USD = risk-off → negate)
    dollar_raw = -px["UUP"].reindex(idx).pct_change(63)
    dollar = soft_sign(dollar_raw).shift(1)

    # Layer 4b — equity breadth (fraction of names with positive 6m return − ½)
    breadth_raw = (eq.reindex(idx).pct_change(126) > 0).mean(axis=1) - 0.5
    breadth = soft_sign(breadth_raw).shift(1)

    votes = {
        "3-layer (liq·vix·credit)": s3.mean(axis=1),
        "4-layer +dollar": pd.concat([s3, dollar.rename("d")], axis=1).mean(axis=1),
        "4-layer +breadth": pd.concat([s3, breadth.rename("b")], axis=1).mean(axis=1),
        "5-layer +both": pd.concat([s3, dollar.rename("d"), breadth.rename("b")],
                                   axis=1).mean(axis=1),
    }

    rets = {"SPY buy & hold": spy_ret, "QQQ buy & hold": qqq_ret}
    expos = {}
    for name, v in votes.items():
        r, e = apply_2xvote(qqq_ret, v)
        rets[f"2×{name} on QQQ"] = r
        expos[name] = e

    # ── Metrics ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 84)
    print(f"  FOUR-LAYER LIQUIDITY VOTE (2× on QQQ)  ·  {idx.min().date()} → {idx.max().date()}")
    print("=" * 84)
    keys = ["CAGR", "Volatility", "Sharpe", "Sortino", "Max DD", "Calmar"]
    table = []
    for name, r in rets.items():
        m = compute_metrics(r.dropna(), rf=RF)
        ret22 = (1 + r[r.index.year == 2022]).prod() - 1
        table.append([name] + [f"{m.get(k, float('nan')):.2f}" for k in keys]
                     + [f"{ret22:+.0%}"])
    print(tabulate(table, headers=["strategy"] + keys + ["2022"], tablefmt="github"))

    # ── Rolling OOS vs SPY (the bar) ─────────────────────────────────────────
    print("\n" + "=" * 84)
    print(f"  ROLLING OOS vs SPY  ·  {ROLL}d (~3y) windows")
    print("=" * 84)
    for name in votes:
        r = rets[f"2×{name} on QQQ"]
        ds = (roll_sharpe(r) - roll_sharpe(spy_ret)).dropna()
        dc = (roll_cagr(r) - roll_cagr(spy_ret)).dropna()
        print(f"  {name:28s} CAGR win {float((dc > 0).mean()):.0%} "
              f"(median {float(dc.median()):+.1%})   "
              f"Sharpe win {float((ds > 0).mean()):.0%} "
              f"(median {float(ds.median()):+.2f})")

    # ── Graphs ────────────────────────────────────────────────────────────────
    best = "4-layer +dollar"           # winner of the head-to-head (see table)
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    # (1) equity curves
    ax = axes[0]
    for name in ["SPY buy & hold", "QQQ buy & hold",
                 "2×3-layer (liq·vix·credit) on QQQ", f"2×{best} on QQQ"]:
        eqc = (1 + rets[name].reindex(idx).fillna(0)).cumprod()
        ax.plot(eqc.index, eqc.values, lw=1.6 if "buy" in name else 1.3,
                color="black" if name.startswith("SPY") else
                ("grey" if name.startswith("QQQ") else None),
                alpha=0.9, label=name)
    ax.set_yscale("log"); ax.set_ylabel("growth of $1 (log)")
    ax.set_title("Levered 4-layer liquidity vote vs SPY / QQQ"); ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    # (2) rolling 3y CAGR edge vs SPY
    ax = axes[1]
    for name in ["2×3-layer (liq·vix·credit) on QQQ", f"2×{best} on QQQ"]:
        dc = (roll_cagr(rets[name]) - roll_cagr(spy_ret))
        ax.plot(dc.index, dc.values * 100, lw=1.2, label=name)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_ylabel("rolling 3y CAGR edge\nvs SPY (pp)"); ax.legend(fontsize=8)
    ax.set_title("Rolling 3-year CAGR edge over SPY"); ax.grid(alpha=0.3)
    # (3) exposure: 3-layer vs best 4-layer (2022 shaded)
    ax = axes[2]
    ax.plot(expos["3-layer (liq·vix·credit)"].index,
            expos["3-layer (liq·vix·credit)"].values, lw=0.8, label="3-layer exposure")
    ax.plot(expos[best].index, expos[best].values, lw=0.8, label=f"{best} exposure")
    ax.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31"),
               color="red", alpha=0.08, label="2022 grind")
    ax.axhline(1.0, color="grey", lw=0.6)
    ax.set_ylabel("exposure (x)"); ax.legend(fontsize=8)
    ax.set_title("Exposure — does the 4th layer cut risk through 2022?")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = RESULTS_DIR / "liquidity_4layer.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    logger.info("Saved: %s", out)
    print("\n✓  Done.")


if __name__ == "__main__":
    main()
