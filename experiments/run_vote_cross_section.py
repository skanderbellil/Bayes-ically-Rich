#!/usr/bin/env python3
"""
Vote-beta — where in the 250-ETF cross-section does the vote edge live?

Every champion variant applies the macro vote to a hand-picked underlying
(QQQ, chosen long ago because it worked).  The 250-ETF universe has never
been used cross-sectionally with the vote.  Two questions:

  1. CONCENTRATION  Define each asset's VOTE-BETA: trailing 3y corr of its
                    daily return with the (causal) 4-layer vote level.  Does
                    the vote's timing edge — E[r | vote high] − E[r | vote low],
                    measured OUT of the estimation window — increase in
                    vote-beta?  (If yes, the edge is a property of the asset's
                    macro exposure, not of QQQ per se.)
  2. SELECTION      Can a causal selector beat the hand-pick?  Monthly:
                    top-10 vote-beta ETFs, inverse-vol weighted, exposure
                    2×vote, honest costs — vs 2×vote on QQQ and B&H.
                    run_liquidity_stress showed *diluting* the underlying
                    fails; this tests *selecting* it instead.

The tug layer needs per-asset OHLC the universe doesn't carry, so the vote
here is the 4-layer macro version (liq, VIX-TS, credit, dollar) for ALL
assets — one dial, many underlyings.  ⚠️ current-membership ETF universe →
survivorship-biased level returns; the A/B vs QQQ on the same universe is
the meaningful comparison, not absolute CAGRs.

    python experiments/run_vote_cross_section.py
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
STRESS_CSV = _bootstrap.ROOT / "datasets" / "stress_panel.csv.gz"
ETF_CSV = _bootstrap.ROOT / "datasets" / "etf_universe_prices.csv.gz"
INFO_CSV = _bootstrap.ROOT / "datasets" / "etf_universe_info.csv"

RF = 0.04
DAILY_RF = RF / 252
BORROW_SPREAD = 0.005
TC = 0.0002
LEV_CAP = 2.0
ROLL = 756
H_LIQ, PUB_LAG, CREDIT_WIN = 126, 5, 21
BETA_WIN, REFIT, TOP_K = 756, 21, 10


def roll_cagr(r):
    return (1 + r).rolling(ROLL).apply(np.prod, raw=True) ** (252 / ROLL) - 1


def roll_sharpe(r):
    ex = r - DAILY_RF
    return (ex.rolling(ROLL).mean() / ex.rolling(ROLL).std()) * np.sqrt(252)


def main() -> None:
    nl = load_net_liquidity()["net_liquidity"]
    p = pd.read_csv(STRESS_CSV, parse_dates=["Date"], index_col="Date").sort_index()
    etf = pd.read_csv(ETF_CSV, parse_dates=["Date"], index_col="Date").sort_index()
    info = pd.read_csv(INFO_CSV, index_col=0)
    idx = (nl.dropna().index
           .intersection(p[["SPY", "QQQ", "HYG", "IEF", "UUP", "VIX", "VIX3M"]]
                         .dropna().index)
           .intersection(etf.index))
    nl, p, etf = nl.reindex(idx), p.reindex(idx), etf.reindex(idx)
    rets = etf.pct_change()
    r_qqq = p["QQQ"].pct_change()
    r_spy = p["SPY"].pct_change()

    # ── The 4-layer macro vote (one dial for every asset) ────────────────────
    liq = soft_sign(nl.diff(H_LIQ).shift(PUB_LAG)).shift(1)
    vix_ts = soft_sign(p["VIX3M"] / p["VIX"] - 1.0).shift(1)
    credit = soft_sign(p["HYG"].pct_change(CREDIT_WIN)
                       - p["IEF"].pct_change(CREDIT_WIN)).shift(1)
    dollar = soft_sign(-p["UUP"].pct_change(63)).shift(1)
    vote = pd.DataFrame({"liq": liq, "vix_ts": vix_ts,
                         "credit": credit, "dollar": dollar}).mean(axis=1)

    # ── Vote-beta per asset (trailing, causal) ───────────────────────────────
    logger.info("Rolling vote-beta for %d ETFs …", rets.shape[1])
    vbeta = rets.rolling(BETA_WIN, min_periods=378).corr(vote).shift(1)

    # ── 1. Concentration: timing edge by vote-beta quintile, out of window ───
    # daily: bucket every asset by its TRAILING vote-beta (refreshed monthly),
    # then pool forward 21d returns conditional on the (causal) vote state.
    # edge_q = E[fwd21 | vote high, asset ∈ q] − E[fwd21 | vote low, asset ∈ q]
    vmed = vote.expanding(min_periods=252).median().shift(1)
    hi_day = (vote >= vmed) & vmed.notna()
    lo_day = (vote < vmed) & vmed.notna()
    month = pd.Series(idx.month, index=idx)
    m_end = month != month.shift(-1)
    fwd21 = etf.pct_change(21).shift(-21)
    qmask = pd.DataFrame(np.nan, index=idx, columns=rets.columns)
    cur_q = None
    for i, d in enumerate(idx):
        if m_end.iloc[i] or cur_q is None:
            vb = vbeta.loc[d].dropna()
            if len(vb) >= 100:
                cur_q = pd.qcut(vb, 5, labels=False, duplicates="drop")
        if cur_q is not None:
            qmask.loc[d, cur_q.index] = cur_q.values
    qmask = qmask.shift(1)
    rows, edge_vals = [], []
    for q in range(5):
        inq = qmask == q
        eh = fwd21[inq & np.broadcast_to(hi_day.values[:, None], fwd21.shape)]
        el = fwd21[inq & np.broadcast_to(lo_day.values[:, None], fwd21.shape)]
        mh, ml = float(np.nanmean(eh.values)), float(np.nanmean(el.values))
        edge_vals.append((mh - ml) * 12)
        rows.append([f"Q{q + 1}" + (" (low β)" if q == 0 else
                                    " (high β)" if q == 4 else ""),
                     f"{mh * 12:+.1%}", f"{ml * 12:+.1%}", f"{(mh - ml) * 12:+.1%}",
                     int(inq.values.sum())])
    start = vbeta.dropna(how="all").index[0]
    print("\n" + "=" * 96)
    print(f"  VOTE-BETA CROSS-SECTION  ·  250 ETFs  ·  {start.date()} → {idx[-1].date()}")
    print("=" * 96)
    print("\n  Forward 21d return (ann.) by causal vote state × trailing "
          "vote-beta quintile:")
    print(tabulate(rows, headers=["quintile", "vote high", "vote low",
                                  "timing edge", "asset-days"], tablefmt="github"))
    qqq_vb = r_qqq.rolling(BETA_WIN, min_periods=378).corr(vote).shift(1)
    ends = idx[m_end.values & vbeta.notna().any(axis=1).values]
    qb = qqq_vb.reindex(ends).dropna()
    pct = [(vbeta.loc[d].dropna() < qb.loc[d]).mean() for d in qb.index]
    print(f"  QQQ's own vote-beta percentile in the universe: "
          f"median {np.median(pct):.0%}  [{np.percentile(pct, 10):.0%}, "
          f"{np.percentile(pct, 90):.0%}]")

    # ── 2. Selection: top-K vote-beta basket vs hand-picked QQQ ─────────────
    logger.info("Building the monthly top-%d basket …", TOP_K)
    vol = rets.rolling(63).std().shift(1)
    w = pd.DataFrame(0.0, index=idx, columns=rets.columns)
    cur = None
    for i, d in enumerate(idx):
        if m_end.iloc[i] or cur is None:
            vb = vbeta.loc[d].dropna()
            if len(vb) >= 100:
                top = vb.nlargest(TOP_K).index
                iv = (1.0 / vol.loc[d, top]).replace([np.inf, -np.inf], np.nan).dropna()
                if len(iv):
                    cur = (iv / iv.sum())
        if cur is not None:
            w.loc[d, cur.index] = cur.values
    w = w.shift(1).fillna(0.0)
    r_basket = (w * rets).sum(axis=1)

    def run(expo, r_under):
        expo = expo.clip(0.0, LEV_CAP).fillna(0.0)
        gross = expo * r_under
        financing = np.maximum(expo - 1.0, 0.0) * (RF + BORROW_SPREAD) / 252
        cash = np.maximum(1.0 - expo, 0.0) * DAILY_RF
        costs = TC * expo.diff().abs().fillna(0.0)
        return gross - financing + cash - costs

    basket_turn = (w.diff().abs().sum(axis=1)).fillna(0.0)
    live = max(vote.first_valid_index(),
               (w.sum(axis=1) > 0).idxmax())
    strat = {
        "SPY buy & hold": r_spy,
        "QQQ buy & hold": r_qqq,
        "2×vote on QQQ (hand-pick)": run(2.0 * vote, r_qqq),
        "top-10 vote-beta basket B&H": r_basket - TC * basket_turn,
        "2×vote on top-10 basket": run(2.0 * vote, r_basket) - TC * basket_turn,
    }
    strat = {k: v.loc[live:] for k, v in strat.items()}
    spy_live = r_spy.loc[live:]

    print(f"\n  Selection head-to-head  ·  honest costs  ·  common live {live.date()}:")
    keys = ["CAGR", "Volatility", "Sharpe", "Sortino", "Max DD", "Calmar"]
    table = []
    for name, r in strat.items():
        m = compute_metrics(r.dropna(), rf=RF)
        ret22 = (1 + r[r.index.year == 2022]).prod() - 1
        dc = (roll_cagr(r) - roll_cagr(spy_live)).dropna()
        ds = (roll_sharpe(r) - roll_sharpe(spy_live)).dropna()
        table.append([name] + [f"{m.get(k, float('nan')):.2f}" for k in keys]
                     + [f"{ret22:+.0%}", f"{float((dc > 0).mean()):.0%}",
                        f"{float((ds > 0).mean()):.0%}"])
    print(tabulate(table, headers=["strategy"] + keys + ["2022", "CAGRwin", "SRwin"],
                   tablefmt="github"))
    print("  ⚠️ survivorship-biased universe — judge the basket vs QQQ spread, "
          "not absolute levels")

    last = vbeta.dropna(how="all").index[-1]
    top_now = vbeta.loc[last].dropna().nlargest(TOP_K)
    fam = info["family"] if "family" in info.columns else None
    print(f"\n  Current top-{TOP_K} vote-beta ETFs ({last.date()}):")
    for t, v in top_now.items():
        name = str(info.loc[t, "name"])[:60] if t in info.index and "name" in info.columns else ""
        print(f"    {t:6s} {v:+.2f}  {name}")

    # ── Graph ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    ax = axes[0]
    qs = [f"Q{q + 1}" for q in range(5)]
    ax.bar(qs, edge_vals, color="tab:blue", alpha=0.8)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_ylabel("ann. timing edge")
    ax.set_title("Next-month vote-timing edge by trailing vote-beta quintile "
                 "(low → high)")
    ax.grid(alpha=0.3)
    ax = axes[1]
    for name, r in strat.items():
        eq = (1 + r.fillna(0)).cumprod()
        ax.plot(eq.index, eq.values, lw=1.4,
                color="black" if name.startswith("SPY") else
                ("grey" if name.startswith("QQQ b") else None), label=name)
    ax.set_yscale("log"); ax.set_ylabel("growth of $1 (log)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("Causal top-10 vote-beta basket vs the hand-picked underlying")
    fig.tight_layout()
    out = RESULTS_DIR / "vote_cross_section.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    logger.info("Saved: %s", out)
    print("\n✓  Done.")


if __name__ == "__main__":
    main()
