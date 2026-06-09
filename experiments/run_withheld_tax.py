#!/usr/bin/env python3
"""
Withheld-tax nowcast — zero-lag labor income as an equity signal.

Large firms remit employee withholding to the Treasury the day after payday;
the DTS publishes the total the day after that.  Daily withheld income/
employment taxes are therefore a ~2-day-lag, zero-survey-error read on
economy-wide labor income — available decades before payrolls revisions
settle.  Economists use it to nowcast wages (taxtracking.com, Fed research);
our June-2026 literature check found no public test of it as an equity
signal.  That ambiguity is real: in Fed-dominated eras "good news" can be
bad for multiples — so the sign is an empirical question, tested honestly.

  nowcast_t = YoY growth of the trailing 63d sum of daily withheld taxes,
              lagged 3 business days (2d publication + 1d trade)

Data: datasets/withheld_taxes.csv.gz — DTS Table IV ("Withheld Income and
Employment Taxes", 2005-10 → 2023-02) spliced with the post-restructuring
Table II line ("Taxes - Withheld Individual/FICA", 2023-02 →).  The splice
is seamless on consecutive business days; continuity is checked below.

  Tests
  -----
  1. Sanity — landmarks (2009 trough, 2020 crash, 2021 boom) + splice check.
  2. Signal — fwd 21d/63d QQQ by causal nowcast tercile; corr; plus the
     acceleration variant (nowcast minus its value 63d ago).
  3. 6th vote layer on the champion, honest costs.

    python experiments/run_withheld_tax.py
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
WT_CSV = _bootstrap.ROOT / "datasets" / "withheld_taxes.csv.gz"
PANEL_CSV = _bootstrap.ROOT / "datasets" / "overnight_panel.csv.gz"
STRESS_CSV = _bootstrap.ROOT / "datasets" / "stress_panel.csv.gz"

RF = 0.04
DAILY_RF = RF / 252
BORROW_SPREAD = 0.005
TC = 0.0002
LEV_CAP = 2.0
ROLL = 756
H_LIQ, PUB_LAG, CREDIT_WIN = 126, 5, 21
LAG = 3                       # 2d publication + 1d trade
SPLICE = pd.Timestamp("2023-02-14")


def causal_rank(s, win=756):
    return s.rolling(win, min_periods=252).apply(
        lambda x: (x < x[-1]).mean(), raw=True).shift(1)


def main() -> None:
    wt = pd.read_csv(WT_CSV, parse_dates=["Date"], index_col="Date")["withheld"]
    panel = pd.read_csv(PANEL_CSV, parse_dates=["Date"], index_col="Date").sort_index()
    idx = panel[["QQQ_Close", "SPY_Close"]].dropna().index
    qqq = panel["QQQ_Close"].reindex(idx)
    r_qqq = qqq.pct_change()

    wt_d = wt.reindex(idx).fillna(0.0)        # weekends/holidays in DTS dates
    # The Feb-2023 DTS restructuring changed the withheld definition by ~10%
    # in level; rescale the old segment so YoY is clean across the splice.
    pre90 = float(wt_d.loc[SPLICE - pd.Timedelta(days=90):SPLICE].mean())
    post90 = float(wt_d.loc[SPLICE:SPLICE + pd.Timedelta(days=90)].mean())
    wt_d.loc[:SPLICE] *= post90 / pre90
    ws63 = wt_d.rolling(63).sum()
    nowcast = ((ws63 / ws63.shift(252) - 1.0).shift(LAG)
               .replace([np.inf, -np.inf], np.nan))
    accel = (nowcast - nowcast.shift(63))

    print("\n" + "=" * 96)
    print(f"  WITHHELD-TAX NOWCAST  ·  {nowcast.dropna().index[0].date()} → "
          f"{idx[-1].date()}")
    print("=" * 96)
    for d, lab in [("2009-07-01", "2009 trough"), ("2020-06-01", "2020 crash"),
                   ("2021-07-01", "2021 boom"), (str(idx[-1].date()), "latest")]:
        v = nowcast.loc[:d].dropna()
        print(f"  {lab:12s} YoY {float(v.iloc[-1]):+.1%}")
    pre = float(wt_d.loc[SPLICE - pd.Timedelta(days=90):SPLICE].mean())
    post = float(wt_d.loc[SPLICE:SPLICE + pd.Timedelta(days=90)].mean())
    print(f"  splice continuity (mean daily, 90d either side of {SPLICE.date()}): "
          f"{pre:,.0f} vs {post:,.0f}  (ratio {post / pre:.2f})")

    # ── Signal tests ─────────────────────────────────────────────────────────
    fwd21 = qqq.pct_change(21).shift(-21)
    fwd63 = qqq.pct_change(63).shift(-63)
    for lab, sig in [("nowcast level", nowcast), ("acceleration (Δ63d)", accel)]:
        rk = causal_rank(sig)
        rows = []
        for b, m in [("low", rk <= 1 / 3), ("mid", (rk > 1 / 3) & (rk < 2 / 3)),
                     ("high", rk >= 2 / 3)]:
            rows.append([b, f"{float(fwd21[m & fwd21.notna()].mean()) * 12:+.1%}",
                         f"{float(fwd63[m & fwd63.notna()].mean()) * 4:+.1%}",
                         int(m.sum())])
        m = sig.notna() & fwd21.notna()
        c21 = float(np.corrcoef(sig[m], fwd21[m])[0, 1])
        print(f"\n  {lab} → forward QQQ (causal 3y rank terciles):")
        print(tabulate(rows, headers=["tercile", "fwd 21d (ann.)",
                                      "fwd 63d (ann.)", "days"], tablefmt="github"))
        print(f"  corr with fwd 21d: {c21:+.3f}")

    # ── 6th layer on the champion ────────────────────────────────────────────
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
    # demean the nowcast against its own trailing year so soft_sign sees the
    # cycle, not the ~4% nominal trend
    wt_layer = soft_sign((nowcast - nowcast.rolling(252).mean()).reindex(vidx))
    s5 = pd.DataFrame({"liq": liq, "vix_ts": vix_ts, "credit": credit,
                       "dollar": dollar, "tug": tug})
    v5 = s5.mean(axis=1)
    v6 = pd.concat([s5, wt_layer.rename("wage")], axis=1).mean(axis=1)
    rq = pv["QQQ"].pct_change()

    def run(expo):
        expo = expo.clip(0.0, LEV_CAP).fillna(0.0)
        return (expo * rq - np.maximum(expo - 1, 0) * (RF + BORROW_SPREAD) / 252
                + np.maximum(1 - expo, 0) * DAILY_RF
                - TC * expo.diff().abs().fillna(0))

    live = max(v5.first_valid_index(), v6.first_valid_index())
    strat = {"QQQ buy & hold": rq, "2×5-layer vote": run(2 * v5),
             "2×6-layer (+wage nowcast)": run(2 * v6)}
    strat = {k: v.loc[live:] for k, v in strat.items()}
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
    print("\n" + tabulate(table, headers=["strategy"] + keys + ["2022", "CAGRwin"],
                          tablefmt="github"))
    corr = pd.concat([s5, wt_layer.rename("wage")], axis=1).corr()["wage"]
    print("  corr(wage layer, others): "
          + "  ".join(f"{k} {v:+.2f}" for k, v in corr.drop("wage").items()))

    # ── Graph ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    ax = axes[0]
    ax.plot(nowcast.index, nowcast.values * 100, lw=1.0, color="tab:blue",
            label="withheld-tax YoY (63d sum)")
    ax.axhline(0, color="grey", lw=0.6)
    ax.axvline(SPLICE, color="orange", lw=0.8, ls=":", label="DTS format splice")
    for a, b in [("2007-12-01", "2009-06-30"), ("2020-02-15", "2020-04-30")]:
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="red", alpha=0.08)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("Zero-lag labor income: daily withheld taxes, YoY (recessions shaded)")
    ax = axes[1]
    for name, r in strat.items():
        eq = (1 + r.fillna(0)).cumprod()
        ax.plot(eq.index, eq.values, lw=1.3,
                color="grey" if "buy" in name else None, label=name)
    ax.set_yscale("log"); ax.set_ylabel("growth of $1 (log)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("Wage-nowcast layer on the champion")
    fig.tight_layout()
    out = RESULTS_DIR / "withheld_tax.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    logger.info("Saved: %s", out)
    print("\n✓  Done.")


if __name__ == "__main__":
    main()
