#!/usr/bin/env python3
"""
Primary-dealer inventories — the real fragility conditioner?

When the fragility-weighted vote needed a dealer-stress gauge we used a VIX
percentile because we had no positioning history.  The NY Fed publishes the
real thing weekly since 1998: primary dealers' net outright positions in
Treasury coupons.  High inventories = congested balance sheets = impaired
intermediation capacity (He-Kelly-Manela intermediary asset pricing).  Our
series (2013→, four survey formats) shows the congestion trade tripling:
mean $32bn (2013-14) → $101bn (2015-21) → $115bn (2022-24) → $301bn (2024-26).

  fragility_t = causal rank of total coupon net position, computed WITHIN
  each survey-format segment (level breaks never enter the rank), published
  with a conservative 8-business-day availability lag + 1d trade lag.

  Tests (the house template)
  -----
  1. Forward QQQ return / vol / left tail by inventory tercile.
  2. The conditioning claim: is the liquidity layer more informative when
     dealers are congested?  (run_liquidity_interaction found VIX-fragility
     conditioning real but unmonetizable; this is the mechanism-true version.)
  3. Regime-weighted vote head-to-head: w_liq = 2×fragility with DEALER rank
     vs the original VIX rank, on the champion config, honest costs.

Sample limitation: 2013→ (older survey formats use undocumented codes), so
the GFC is out of reach; 2020, 2022 and the 2024-26 congestion are in.

    python experiments/run_dealer_fragility.py
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
PD_CSV = _bootstrap.ROOT / "datasets" / "dealer_positions.csv.gz"
PANEL_CSV = _bootstrap.ROOT / "datasets" / "overnight_panel.csv.gz"
STRESS_CSV = _bootstrap.ROOT / "datasets" / "stress_panel.csv.gz"

RF = 0.04
DAILY_RF = RF / 252
BORROW_SPREAD = 0.005
TC = 0.0002
LEV_CAP = 2.0
ROLL = 756
H_LIQ, PUB_LAG, CREDIT_WIN = 126, 5, 21
AVAIL_LAG = 8        # business days from as-of Wednesday to publication


def main() -> None:
    pdpos = pd.read_csv(PD_CSV, parse_dates=["Date"], index_col="Date").sort_index()
    panel = pd.read_csv(PANEL_CSV, parse_dates=["Date"], index_col="Date").sort_index()
    idx = panel[["QQQ_Close", "SPY_Close"]].dropna().index
    qqq = panel["QQQ_Close"].reindex(idx)
    r_qqq = qqq.pct_change()

    # within-segment causal rank (rolling 156w, min 52w) on the weekly series
    frag_w = pd.Series(dtype=float)
    for seg, g in pdpos.groupby("seg"):
        s = g["coupon_net"]
        rk = s.rolling(156, min_periods=52).apply(
            lambda x: (x < x[-1]).mean(), raw=True)
        frag_w = pd.concat([frag_w, rk])
    frag_w = frag_w.sort_index()
    # availability: as-of Wednesday → published ~8bd later, then daily ffill
    avail = frag_w.copy()
    avail.index = avail.index + pd.offsets.BDay(AVAIL_LAG)
    frag = avail.reindex(idx).ffill(limit=10).shift(1)
    lo, hi = frag <= 0.33, frag >= 0.67
    mid = ~lo & ~hi & frag.notna()
    start = frag.dropna().index[0]

    print("\n" + "=" * 96)
    print(f"  DEALER-INVENTORY FRAGILITY  ·  {start.date()} → {idx[-1].date()}  "
          f"(weekly, {AVAIL_LAG}bd availability lag)")
    print("=" * 96)

    fwd21 = qqq.pct_change(21).shift(-21)
    fwd_vol = (r_qqq.rolling(21).std() * np.sqrt(252)).shift(-21)
    rows = []
    for lab, m in [("low inventory", lo), ("mid", mid), ("high (congested)", hi)]:
        mm = m & fwd21.notna()
        rows.append([lab, f"{float(fwd21[mm].mean()) * 12:+.1%}",
                     f"{float(fwd21[mm].quantile(0.05)):+.1%}",
                     f"{float(fwd_vol[m & fwd_vol.notna()].mean()):.1%}",
                     int(m.sum())])
    print("  Forward 21d QQQ by dealer-inventory tercile (within-segment rank):")
    print(tabulate(rows, headers=["tercile", "fwd ret (ann.)", "5th pct",
                                  "fwd vol", "days"], tablefmt="github"))

    # ── 2. liquidity-layer conditioning ──────────────────────────────────────
    nl_full = load_net_liquidity()["net_liquidity"].reindex(idx).ffill()
    d_nl = nl_full.diff(H_LIQ).shift(PUB_LAG)
    fwd63 = qqq.pct_change(63).shift(-63)
    print("\n  corr(Δ6m net-liquidity, fwd 63d QQQ) by dealer-inventory state:")
    rows = []
    for lab, m in [("low", lo), ("mid", mid), ("high (congested)", hi)]:
        mm = m & d_nl.notna() & fwd63.notna()
        c = float(np.corrcoef(d_nl[mm], fwd63[mm])[0, 1]) if mm.sum() > 100 else np.nan
        rows.append([lab, f"{c:+.2f}", int(mm.sum())])
    print(tabulate(rows, headers=["state", "corr", "days"], tablefmt="github"))

    # ── 3. regime-weighted vote: dealer vs VIX fragility ─────────────────────
    p = pd.read_csv(STRESS_CSV, parse_dates=["Date"], index_col="Date").sort_index()
    nl = load_net_liquidity()["net_liquidity"]
    vidx = (nl.dropna().index
            .intersection(p[["SPY", "QQQ", "HYG", "IEF", "UUP", "VIX", "VIX3M"]]
                          .dropna().index)
            .intersection(panel.dropna().index))
    nlv, pv = nl.reindex(vidx), p.reindex(vidx)
    liq = soft_sign(nlv.diff(H_LIQ).shift(PUB_LAG)).shift(1)
    vix_ts = soft_sign(pv["VIX3M"] / pv["VIX"] - 1.0).shift(1)
    credit = soft_sign(pv["HYG"].pct_change(CREDIT_WIN)
                       - pv["IEF"].pct_change(CREDIT_WIN)).shift(1)
    dollar = soft_sign(-pv["UUP"].pct_change(63)).shift(1)
    o, cq = panel["QQQ_Open"].reindex(vidx), panel["QQQ_Close"].reindex(vidx)
    r_on, r_id = o / cq.shift(1) - 1.0, cq / o - 1.0
    tug = soft_sign(((1 + r_on).rolling(21).apply(np.prod, raw=True)
                     - (1 + r_id).rolling(21).apply(np.prod, raw=True))).shift(1)
    s5 = pd.DataFrame({"liq": liq, "vix_ts": vix_ts, "credit": credit,
                       "dollar": dollar, "tug": tug})
    vix_frag = pv["VIX"].rolling(756, min_periods=252).apply(
        lambda x: (x < x[-1]).mean(), raw=True).shift(1)
    dlr_frag = frag.reindex(vidx)

    def weighted_vote(w_liq):
        w = pd.DataFrame(1.0, index=s5.index, columns=s5.columns)
        w["liq"] = w_liq
        w = w.where(s5.notna())
        return (s5 * w).sum(axis=1) / w.sum(axis=1)

    rq = pv["QQQ"].pct_change()

    def run(expo):
        expo = expo.clip(0.0, LEV_CAP).fillna(0.0)
        return (expo * rq - np.maximum(expo - 1, 0) * (RF + BORROW_SPREAD) / 252
                + np.maximum(1 - expo, 0) * DAILY_RF
                - TC * expo.diff().abs().fillna(0))

    votes = {"equal (champion)": s5.mean(axis=1),
             "w_liq = 2×VIX fragility": weighted_vote(2.0 * vix_frag),
             "w_liq = 2×DEALER fragility": weighted_vote(2.0 * dlr_frag)}
    live = max([v.first_valid_index() for v in votes.values()]
               + [dlr_frag.dropna().index[0], vix_frag.dropna().index[0]])
    strat = {"QQQ buy & hold": rq.loc[live:]}
    for name, v in votes.items():
        strat[f"2×{name}"] = run(2 * v).loc[live:]
    spy_live = pv["SPY"].pct_change().loc[live:]

    def roll_cagr(r):
        return (1 + r).rolling(ROLL).apply(np.prod, raw=True) ** (252 / ROLL) - 1

    keys = ["CAGR", "Volatility", "Sharpe", "Sortino", "Max DD", "Calmar"]
    table = []
    for name, r in strat.items():
        m = compute_metrics(r.dropna(), rf=RF)
        ret22 = (1 + r[r.index.year == 2022]).prod() - 1
        dc = (roll_cagr(r) - roll_cagr(spy_live)).dropna()
        table.append([name] + [f"{m.get(k, float('nan')):.2f}" for k in keys]
                     + [f"{ret22:+.0%}", f"{float((dc > 0).mean()):.0%}"])
    print(f"\n  Regime-weighted vote head-to-head  ·  common live {live.date()}:")
    print(tabulate(table, headers=["strategy"] + keys + ["2022", "CAGRwin"],
                   tablefmt="github"))

    # ── Graph ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    ax = axes[0]
    tot = pdpos["coupon_net"] / 1e3
    ax.plot(tot.index, tot.values, lw=1.0, color="tab:red",
            label="dealer net coupon positions ($bn)")
    for seg, g in pdpos.groupby("seg"):
        ax.axvline(g.index[0], color="grey", lw=0.6, ls=":")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("Primary-dealer Treasury coupon inventories "
                 "(survey-format breaks dotted)")
    ax = axes[1]
    for name, r in strat.items():
        eq = (1 + r.fillna(0)).cumprod()
        ax.plot(eq.index, eq.values, lw=1.3,
                color="grey" if "buy" in name else None, label=name)
    ax.set_yscale("log"); ax.set_ylabel("growth of $1 (log)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("Fragility-weighted votes: dealer vs VIX conditioner")
    fig.tight_layout()
    out = RESULTS_DIR / "dealer_fragility.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    logger.info("Saved: %s", out)
    print("\n✓  Done.")


if __name__ == "__main__":
    main()
