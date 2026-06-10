#!/usr/bin/env python3
"""
Questioning the winner — GFC stress, Bayesian layer weights, ceiling autopsy.

Three open questions from the exploration lab, on the GFC-inclusive sample
(stress_panel has VIX/VIX3M/UUP from 2007; tug OHLC from 1999; liquidity
from 2003 — signals live ~2008-04, five months before Lehman):

  1. GFC STRESS    The 5-layer (+tug) winner has only ever been tested from
                   2011.  2008-2010 is fresh data for it — and the tug
                   fresh-sample check says 2004-2013 was tug's dead decade,
                   so the GFC should show the layer at its worst.

  2. BAYESIAN      The equal-weight vote is dogma.  In a project called
     WEIGHTS       Bayes-ically Rich, each layer should carry the weight its
                   trailing evidence earns:

                     z_i(t) = trailing-3y corr(stance_i, same-day return) × √n
                     w_i(t) = 1 + max(z_i, 0)        (equal-weight prior
                                                      pseudo-count + positive
                                                      evidence, lagged 1d)
                     vote   = Σ w_i s_i / Σ w_i

                   Parameter-free posterior-ish weighting.  The sharp test:
                   does it auto-suppress tug through its dead 2008-2013 decade
                   and auto-revive it after 2014?  If yes, episodic layers
                   stop being a promotion blocker — the vote becomes
                   self-curating.

  3. CEILING       The debt-ceiling overlay rests on n=10 episodes.  Per-
     AUTOPSY       episode breakdown of what the ×½ halving contributed —
                   is the edge broad or one lucky episode?

    python experiments/run_adaptive_vote.py
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
EVID_WIN = 756

DEBT_CEILING = ["2011-08-02", "2013-10-17", "2014-02-15", "2015-11-02",
                "2017-09-08", "2018-02-09", "2019-08-02", "2021-10-14",
                "2021-12-16", "2023-06-03"]   # complete through Jun-2023


def run(expo, r_cc):
    expo = expo.clip(0.0, LEV_CAP).fillna(0.0)
    gross = expo * r_cc
    financing = np.maximum(expo - 1.0, 0.0) * (RF + BORROW_SPREAD) / 252
    cash = np.maximum(1.0 - expo, 0.0) * DAILY_RF
    costs = TC * expo.diff().abs().fillna(0.0)
    return pd.Series(gross - financing + cash - costs, index=r_cc.index)


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
    r_cc = p["QQQ"].pct_change()
    spy_ret = p["SPY"].pct_change()

    # ── Five layer stances (same constructions as the thread) ───────────────
    liq = soft_sign(nl.diff(H_LIQ).shift(PUB_LAG)).shift(1)
    vix_ts = soft_sign(p["VIX3M"] / p["VIX"] - 1.0).shift(1)
    credit = soft_sign(p["HYG"].pct_change(CREDIT_WIN)
                       - p["IEF"].pct_change(CREDIT_WIN)).shift(1)
    dollar = soft_sign(-p["UUP"].pct_change(63)).shift(1)
    q_o, q_c = panel["QQQ_Open"].reindex(idx), panel["QQQ_Close"].reindex(idx)
    r_on, r_id = q_o / q_c.shift(1) - 1.0, q_c / q_o - 1.0
    tug = soft_sign(((1 + r_on).rolling(21).apply(np.prod, raw=True)
                     - (1 + r_id).rolling(21).apply(np.prod, raw=True))).shift(1)
    s5 = pd.DataFrame({"liq": liq, "vix_ts": vix_ts, "credit": credit,
                       "dollar": dollar, "tug": tug})
    vote4 = s5[["liq", "vix_ts", "credit", "dollar"]].mean(axis=1)
    vote5 = s5.mean(axis=1)

    # ── Bayesian evidence weights ────────────────────────────────────────────
    # z_i = trailing corr(stance, same-day ret) × √n  (stances already causal);
    # weight = equal-weight prior pseudo-count (1) + positive evidence, lagged.
    z = pd.DataFrame(index=idx, columns=s5.columns, dtype=float)
    for c in s5.columns:
        corr = s5[c].rolling(EVID_WIN, min_periods=252).corr(r_cc)
        n = s5[c].notna().rolling(EVID_WIN, min_periods=252).sum()
        z[c] = corr * np.sqrt(n)
    w = (1.0 + z.clip(lower=0.0)).shift(1)
    w = w.where(s5.notna())
    vote_b = (s5 * w).sum(axis=1) / w.sum(axis=1)

    # ── Debt-ceiling mask ────────────────────────────────────────────────────
    dc_mask = pd.Series(False, index=idx)
    for d in pd.to_datetime(DEBT_CEILING):
        i = idx.searchsorted(d)
        if 0 < i < len(idx):
            dc_mask.iloc[i:min(i + 61, len(idx))] = True
    dc_mask = dc_mask.shift(1).fillna(False)

    live = vote_b.first_valid_index()
    strat = {
        "SPY buy & hold": spy_ret,
        "QQQ buy & hold": r_cc,
        "2×4-layer equal": run(2.0 * vote4, r_cc),
        "2×5-layer equal (+tug)": run(2.0 * vote5, r_cc),
        "2×5-layer Bayesian weights": run(2.0 * vote_b, r_cc),
        "2×5-layer equal + ceiling ×½": run(2.0 * vote5 * np.where(dc_mask, 0.5, 1.0), r_cc),
        "2×5-layer Bayes + ceiling ×½": run(2.0 * vote_b * np.where(dc_mask, 0.5, 1.0), r_cc),
    }
    strat = {k: v.loc[live:] for k, v in strat.items()}
    spy_live = spy_ret.loc[live:]

    # ── Part 1+2 results ─────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print(f"  GFC-INCLUSIVE TEST + BAYESIAN LAYER WEIGHTS  ·  QQQ, honest costs  ·  "
          f"{live.date()} → {idx[-1].date()}")
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
                        f"{float((dc_ > 0).mean()):.0%}", f"{float((ds_ > 0).mean()):.0%}"])
    print(tabulate(table, headers=["strategy"] + keys
                   + ["GFC", "2022", "CAGRwin", "SRwin"], tablefmt="github"))
    print("  GFC = Sep-2008 → Jun-2009 total return  ·  win rates vs SPY, 3y windows")

    # The Bayesian warm-up pushes the common live date past Lehman, so the GFC
    # column above is only the recovery tail.  Run the equal-weight votes from
    # their own (earlier) live date for a real GFC reading.
    live_eq = vote5.first_valid_index()
    if live_eq < pd.Timestamp("2008-09-01"):
        print(f"\n  TRUE GFC TEST (equal votes live from {live_eq.date()}, "
              "before Lehman):")
        for name, v in [("2×4-layer equal", vote4), ("2×5-layer equal", vote5)]:
            r = run(2.0 * v, r_cc).loc[live_eq:]
            gfc = (1 + r.loc["2008-09-01":"2009-06-30"]).prod() - 1
            print(f"    {name:24s} GFC {gfc:+.0%}   "
                  f"(QQQ {(1 + r_cc.loc['2008-09-01':'2009-06-30']).prod() - 1:+.0%}, "
                  f"SPY {(1 + spy_ret.loc['2008-09-01':'2009-06-30']).prod() - 1:+.0%})")
    else:
        print(f"\n  (equal votes only live from {live_eq.date()} — GFC untestable)")

    print("\n  Mean Bayesian weight per layer (share of vote):")
    wn = (w.div(w.sum(axis=1), axis=0)).loc[live:]
    rows = []
    for lab, a, b in [("2008-2013 (tug dead decade)", "2008", "2013"),
                      ("2014-2026 (tug alive)", "2014", "2026")]:
        rows.append([lab] + [f"{float(wn[c].loc[a:b].mean()):.1%}" for c in s5.columns])
    rows.append(["equal weight"] + ["20.0%"] * 5)
    print(tabulate(rows, headers=["period"] + list(s5.columns), tablefmt="github"))

    # ── Part 3: debt-ceiling per-episode autopsy ─────────────────────────────
    print("\n" + "=" * 100)
    print("  DEBT-CEILING AUTOPSY  ·  per-episode effect of the ×½ halving "
          "(5-layer equal vote)")
    print("=" * 100)
    base_r = strat["2×5-layer equal (+tug)"]
    ovl_r = strat["2×5-layer equal + ceiling ×½"]
    rows = []
    for d in pd.to_datetime(DEBT_CEILING):
        i = idx.searchsorted(d)
        if not (0 < i < len(idx) and idx[i] >= live):
            continue
        win = idx[i:min(i + 61, len(idx))]
        qqq_w = (1 + r_cc.loc[win]).prod() - 1
        b_w = (1 + base_r.loc[win]).prod() - 1
        o_w = (1 + ovl_r.loc[win]).prod() - 1
        rows.append([str(d.date()), f"{qqq_w:+.1%}", f"{b_w:+.1%}", f"{o_w:+.1%}",
                     f"{o_w - b_w:+.1%}"])
    helped = sum(1 for r_ in rows if float(r_[4].rstrip("%")) > 0)
    print(tabulate(rows, headers=["resolution", "QQQ [0,+60]", "champion",
                                  "with ×½", "overlay Δ"], tablefmt="github"))
    print(f"  overlay helped in {helped}/{len(rows)} episodes")

    # ── Graphs ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 13))
    ax = axes[0]
    for name, r in strat.items():
        eq = (1 + r.fillna(0)).cumprod()
        ax.plot(eq.index, eq.values,
                lw=1.8 if "Bayes + ceiling" in name else 1.1,
                color="black" if name.startswith("SPY") else
                ("grey" if name.startswith("QQQ") else None), label=name)
    ax.axvspan(pd.Timestamp("2008-09-01"), pd.Timestamp("2009-06-30"),
               color="red", alpha=0.08)
    ax.set_yscale("log"); ax.set_ylabel("growth of $1 (log)")
    ax.set_title("GFC-inclusive: equal vs Bayesian-weighted vote (GFC shaded)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[1]
    for c in s5.columns:
        ax.plot(wn.index, wn[c].rolling(63).mean(), lw=1.2, label=c)
    ax.axhline(0.2, color="grey", lw=0.8, ls="--", label="equal weight")
    ax.set_ylabel("share of vote (63d MA)")
    ax.set_title("Bayesian layer weights — does tug get auto-suppressed pre-2014?")
    ax.legend(fontsize=8, ncol=6); ax.grid(alpha=0.3)

    ax = axes[2]
    for name in ["QQQ buy & hold", "2×5-layer equal (+tug)",
                 "2×5-layer Bayes + ceiling ×½"]:
        eq = (1 + strat[name].fillna(0)).cumprod()
        dd = eq / eq.cummax() - 1
        ax.plot(dd.index, dd.values, lw=1.1,
                color="grey" if "buy" in name else None, label=name)
    ax.set_ylabel("drawdown"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("Drawdown through the GFC")
    fig.tight_layout()
    out = RESULTS_DIR / "adaptive_vote.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    logger.info("Saved: %s", out)
    print("\n✓  Done.")


if __name__ == "__main__":
    main()
