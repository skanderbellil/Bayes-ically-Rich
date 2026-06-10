#!/usr/bin/env python3
"""
Decomposing the founding layer — do the Fed's three taps flow at one speed?

Net liquidity = WALCL − TGA − RRP has been a single composite since the
thread began.  But its components are different machines: WALCL is *policy*
(QE/QT, slow, persistent), the TGA is *Treasury cash management* (mechanical,
calendar-driven, mean-reverting — the calendar study showed only debt-ceiling
rebuilds matter), and the RRP is *money-fund behaviour* (era-dependent; the
floor forensics showed scarcity AMPLIFIES drains rather than buffering them).
There is no reason these share a lead time — or even a sign.  If they
disagree, the composite Δ6m is averaging away information.  The shadow test
just showed the layer's value is plumbing-specific; this asks WHICH plumbing.

  Signals (each the component's contribution to the composite, signed as it
  enters net liquidity, Δ126d, publication-lagged, soft-signed):

    dW = +Δ126 WALCL      dT = −Δ126 TGA      dR = −Δ126 RRP

  Tests
  -----
  1. LEAD-LAG       corr(Δh signal, fwd 63d QQQ) for h ∈ {21, 63, 126, 252}
                    per component vs the composite — one horizon fits all?
  2. INFORMATION    full-sample + sub-period corr and causal-tercile forward
                    returns per component (Δ126).
  3. HEAD-TO-HEAD   The champion with its liq layer = composite (baseline),
                    component-mean, each component alone, and an
                    evidence-weighted component blend (w_i = 1 + positive
                    trailing-corr z, the run_adaptive_vote construction,
                    applied WITHIN the layer).  Honest costs, win rates.

    python experiments/run_liquidity_components.py
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


def causal_rank(s):
    return s.rolling(ROLL, min_periods=252).apply(
        lambda x: (x < x[-1]).mean(), raw=True)


def main() -> None:
    nld = load_net_liquidity()
    p = pd.read_csv(STRESS_CSV, parse_dates=["Date"], index_col="Date").sort_index()
    panel = pd.read_csv(PANEL_CSV, parse_dates=["Date"], index_col="Date").sort_index()
    idx = (nld.dropna().index
           .intersection(p[["SPY", "QQQ", "HYG", "IEF", "UUP", "VIX", "VIX3M"]]
                         .dropna().index)
           .intersection(panel.dropna().index))
    nld, p = nld.reindex(idx), p.reindex(idx)
    qqq = p["QQQ"]
    r_cc = qqq.pct_change()
    spy_ret = p["SPY"].pct_change()

    comp_raw = {"WALCL (policy)": nld["WALCL"],
                "−TGA (Treasury)": -nld["WTREGEN"],
                "−RRP (money funds)": -nld["RRPONTSYD"],
                "composite": nld["net_liquidity"]}

    # ── 1. Lead-lag scan ─────────────────────────────────────────────────────
    fwd63 = qqq.pct_change(63).shift(-63)
    print("\n" + "=" * 96)
    print(f"  LIQUIDITY COMPONENT DECOMPOSITION  ·  {idx[0].date()} → {idx[-1].date()}")
    print("=" * 96)
    print("\n  Lead-lag: corr(Δh signal, fwd 63d QQQ) — one horizon does not fit all?")
    horizons = [21, 63, 126, 252]
    rows = []
    for name, s in comp_raw.items():
        row = [name]
        for h in horizons:
            d = s.diff(h).shift(PUB_LAG)
            m = d.notna() & fwd63.notna()
            row.append(f"{float(np.corrcoef(d[m], fwd63[m])[0, 1]):+.2f}")
        rows.append(row)
    print(tabulate(rows, headers=["signal"] + [f"Δ{h}d" for h in horizons],
                   tablefmt="github"))

    # ── 2. Information at the house horizon (Δ126) ──────────────────────────
    sigs = {k: s.diff(H_LIQ).shift(PUB_LAG) for k, s in comp_raw.items()}
    print("\n  corr(Δ126 signal, fwd 63d QQQ) — sub-periods:")
    spans = [("full", None, None), ("2008-2014", "2008", "2014"),
             ("2015-2019", "2015", "2019"), ("2020-2026", "2020", "2026")]
    rows = []
    for name, d in sigs.items():
        row = [name]
        for _, a, b in spans:
            m = d.notna() & fwd63.notna()
            dd_, ff = (d[m], fwd63[m]) if a is None else (d[m].loc[a:b], fwd63[m].loc[a:b])
            row.append(f"{float(np.corrcoef(dd_, ff)[0, 1]):+.2f}" if len(dd_) > 100
                       else "—")
        rows.append(row)
    print(tabulate(rows, headers=["signal"] + [s[0] for s in spans],
                   tablefmt="github"))

    print("\n  Forward 63d QQQ (ann.) by causal signal tercile (Δ126):")
    rows = []
    for name, d in sigs.items():
        rk = causal_rank(d)
        lo, hi = rk <= 0.33, rk >= 0.67
        mid = ~lo & ~hi & rk.notna()
        row = [name]
        for m in (lo, mid, hi):
            mm = m & fwd63.notna()
            row.append(f"{float(fwd63[mm].mean()) * 4:+.1%}" if mm.sum() > 50 else "—")
        mm_hi, mm_lo = hi & fwd63.notna(), lo & fwd63.notna()
        if mm_hi.sum() > 50 and mm_lo.sum() > 50:
            row.append(f"{(float(fwd63[mm_hi].mean()) - float(fwd63[mm_lo].mean())) * 4:+.1%}")
        else:
            row.append("—")
        rows.append(row)
    print(tabulate(rows, headers=["signal", "low", "mid", "high", "hi−lo"],
                   tablefmt="github"))

    # ── 3. Champion head-to-head ─────────────────────────────────────────────
    vix_ts = soft_sign(p["VIX3M"] / p["VIX"] - 1.0).shift(1)
    credit = soft_sign(p["HYG"].pct_change(CREDIT_WIN)
                       - p["IEF"].pct_change(CREDIT_WIN)).shift(1)
    dollar = soft_sign(-p["UUP"].pct_change(63)).shift(1)
    q_o, q_c = panel["QQQ_Open"].reindex(idx), panel["QQQ_Close"].reindex(idx)
    r_on, r_id = q_o / q_c.shift(1) - 1.0, q_c / q_o - 1.0
    tug = soft_sign(((1 + r_on).rolling(21).apply(np.prod, raw=True)
                     - (1 + r_id).rolling(21).apply(np.prod, raw=True))).shift(1)
    base = pd.DataFrame({"vix_ts": vix_ts, "credit": credit,
                         "dollar": dollar, "tug": tug})

    sW = soft_sign(sigs["WALCL (policy)"]).shift(1)
    sT = soft_sign(sigs["−TGA (Treasury)"]).shift(1)
    sR = soft_sign(sigs["−RRP (money funds)"]).shift(1)
    sC = soft_sign(sigs["composite"]).shift(1)
    comp3 = pd.DataFrame({"W": sW, "T": sT, "R": sR})

    # evidence-weighted blend WITHIN the layer (run_adaptive_vote construction)
    z = pd.DataFrame(index=idx, columns=comp3.columns, dtype=float)
    for c in comp3.columns:
        corr = comp3[c].rolling(EVID_WIN, min_periods=252).corr(r_cc)
        n = comp3[c].notna().rolling(EVID_WIN, min_periods=252).sum()
        z[c] = corr * np.sqrt(n)
    w = (1.0 + z.clip(lower=0.0)).shift(1).where(comp3.notna())
    s_evid = (comp3 * w).sum(axis=1) / w.sum(axis=1)

    liq_variants = {"composite (champion)": sC,
                    "component mean (W,T,R)": comp3.mean(axis=1),
                    "WALCL only": sW, "−TGA only": sT, "−RRP only": sR,
                    "evidence-weighted W/T/R": s_evid}
    votes = {k: pd.concat([base, v.rename("liq")], axis=1).mean(axis=1)
             for k, v in liq_variants.items()}
    live = max([v.first_valid_index() for v in votes.values()]
               + [v.first_valid_index() for v in liq_variants.values()])
    strat = {"SPY buy & hold": spy_ret, "QQQ buy & hold": r_cc}
    strat.update({f"2×5L · liq = {k}": run(2.0 * v, r_cc) for k, v in votes.items()})
    strat = {k: v.loc[live:] for k, v in strat.items()}
    spy_live = spy_ret.loc[live:]

    print(f"\n  Champion head-to-head  ·  QQQ, honest costs  ·  common live {live.date()}:")
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
    print("  win rates vs SPY, rolling 3y windows")

    wn = w.div(w.sum(axis=1), axis=0).loc[live:]
    print("\n  Mean evidence weight within the layer (share):")
    rows = []
    for lab, a, b in [("2010-2014", "2010", "2014"), ("2015-2019", "2015", "2019"),
                      ("2020-2026", "2020", "2026")]:
        rows.append([lab] + [f"{float(wn[c].loc[a:b].mean()):.1%}" for c in comp3.columns])
    print(tabulate(rows, headers=["period", "WALCL", "−TGA", "−RRP"],
                   tablefmt="github"))

    # ── Graph ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 13))
    ax = axes[0]
    for name, s in comp_raw.items():
        if name == "composite":
            continue
        d = s.diff(H_LIQ)
        ax.plot(d.index, d.values, lw=1.0, label=f"Δ126 {name}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_ylabel("$bn / 126d")
    ax.set_title("The three taps: Δ6m of each net-liquidity component")

    ax = axes[1]
    for name, r in strat.items():
        eq = (1 + r.fillna(0)).cumprod()
        ax.plot(eq.index, eq.values,
                lw=1.8 if "evidence" in name else 1.1,
                color="black" if name.startswith("SPY") else
                ("grey" if name.startswith("QQQ") else None), label=name)
    ax.set_yscale("log"); ax.set_ylabel("growth of $1 (log)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax.set_title("Champion with the liq layer decomposed")

    ax = axes[2]
    for c in comp3.columns:
        ax.plot(wn.index, wn[c].rolling(63).mean(), lw=1.2,
                label={"W": "WALCL", "T": "−TGA", "R": "−RRP"}[c])
    ax.axhline(1 / 3, color="grey", lw=0.8, ls="--", label="equal")
    ax.set_ylabel("share of layer (63d MA)")
    ax.legend(fontsize=8, ncol=4); ax.grid(alpha=0.3)
    ax.set_title("Evidence weights within the layer — which tap earns the weight, when?")
    fig.tight_layout()
    out = RESULTS_DIR / "liquidity_components.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    logger.info("Saved: %s", out)
    print("\n✓  Done.")


if __name__ == "__main__":
    main()
