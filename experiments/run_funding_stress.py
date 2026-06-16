#!/usr/bin/env python3
"""
Funding-rate spreads — the daily reserve-scarcity nowcast.

The RRP-floor study (run_rrp_floor) ended with a registered prediction and a
crude quarterly trigger (63d median RRP < $100bn).  The money market prints a
better gauge every morning: when the secured overnight rate trades at or
above the Fed's administered floor, reserves are scarce — cash is worth more
in the market than at the central bank.  Sep-2019 (SOFR 315bp over IOER),
Mar-2020, and the 2024-26 creep are all visible in one series.

Everyone watches this spread (Fed notes, dashboards); our June-2026
literature check found NO public backtest of it against equities — it is
universally called "a plumbing health check, not a trading signal" without
anyone testing whether that's true.  This is also the original rack item
"our own financing cost is a signal."

  scarcity_t = 5d MA of (market rate − admin rate), causal (shift 1)
      market: SOFR (2018→), EFFR before that (2008→, vs IOER)
      admin : IORB stitched with IOER
  stressed_t = scarcity_t ≥ 0bp   (market at/above the floor — a-priori,
                                   mechanical threshold, no fitting)

  Tests
  -----
  1. Forward QQQ return and realized vol by scarcity state (2008→).
  2. Liquidity-layer interaction: is the Δ6m net-liquidity signal more
     informative when reserves are scarce?  (the RRP study said yes by era;
     this asks the daily gauge.)
  3. Tax-window drag by scarcity state — the registered prediction's
     upgraded trigger.
  4. Stress throttle on the champion (×½ when stressed), honest costs.

Data: datasets/funding_rates.csv.gz via posterioralpha fetch_fred
(FRED_API_KEY env var → keyed API; --refresh refetches).

    python experiments/run_funding_stress.py
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
FUND_CSV = _bootstrap.ROOT / "datasets" / "funding_rates.csv.gz"
PANEL_CSV = _bootstrap.ROOT / "datasets" / "overnight_panel.csv.gz"
STRESS_CSV = _bootstrap.ROOT / "datasets" / "stress_panel.csv.gz"

RF = 0.04
DAILY_RF = RF / 252
BORROW_SPREAD = 0.005
TC = 0.0002
LEV_CAP = 2.0
ROLL = 756
H_LIQ, PUB_LAG, CREDIT_WIN = 126, 5, 21
TAX_WIN = (0, 10)


def refresh() -> None:
    from posterioralpha.data.macro import fetch_fred
    df = pd.DataFrame({s: fetch_fred(s) for s in ["SOFR", "IORB", "IOER", "EFFR"]})
    df.index.name = "Date"
    df.to_csv(FUND_CSV)
    logger.info("Rebuilt %s %s", FUND_CSV, df.shape)


def main() -> None:
    if "--refresh" in sys.argv:
        refresh()
    fr = pd.read_csv(FUND_CSV, parse_dates=["Date"], index_col="Date").sort_index()
    panel = pd.read_csv(PANEL_CSV, parse_dates=["Date"], index_col="Date").sort_index()
    idx = panel[["QQQ_Close", "SPY_Close"]].dropna().index
    fr = fr.reindex(idx).ffill(limit=5)
    qqq = panel["QQQ_Close"].reindex(idx)
    r_qqq = qqq.pct_change()

    admin = fr["IORB"].fillna(fr["IOER"])
    market = fr["SOFR"].fillna(fr["EFFR"])
    spread_bp = (market - admin) * 100.0
    scarcity = spread_bp.rolling(5).mean().shift(1)
    stressed = scarcity >= 0.0
    valid = scarcity.notna()
    start = scarcity.dropna().index[0]

    print("\n" + "=" * 96)
    print(f"  FUNDING-RATE SCARCITY NOWCAST  ·  {start.date()} → {idx[-1].date()}")
    print(f"  spread = (SOFR|EFFR) − (IORB|IOER)  ·  stressed = 5d MA ≥ 0bp")
    print("=" * 96)
    print(f"  landmarks: Sep-17-2019 {float(spread_bp.loc['2019-09-17']):+.0f}bp · "
          f"Mar-17-2020 {float(spread_bp.loc['2020-03-17']):+.0f}bp · "
          f"latest {float(spread_bp.dropna().iloc[-1]):+.0f}bp")
    yr_share = stressed[valid].groupby(stressed[valid].index.year).mean()
    print("  share of stressed days by year: "
          + "  ".join(f"{y}:{v:.0%}" for y, v in yr_share.items() if v > 0))

    # ── 1. Forward return / vol by state ─────────────────────────────────────
    fwd21 = qqq.pct_change(21).shift(-21)
    fwd_vol = (r_qqq.rolling(21).std() * np.sqrt(252)).shift(-21)
    rows = []
    for lab, m in [("calm (spread<0)", valid & ~stressed),
                   ("stressed (≥0bp)", valid & stressed)]:
        rows.append([lab, f"{float(fwd21[m & fwd21.notna()].mean()) * 12:+.1%}",
                     f"{float(fwd21[m & fwd21.notna()].quantile(0.05)):+.1%}",
                     f"{float(fwd_vol[m & fwd_vol.notna()].mean()):.1%}",
                     int(m.sum())])
    print("\n  Forward 21d QQQ by scarcity state:")
    print(tabulate(rows, headers=["state", "fwd ret (ann.)", "5th pct",
                                  "fwd vol", "days"], tablefmt="github"))

    # ── 2. Liquidity-layer interaction ───────────────────────────────────────
    nl_full = load_net_liquidity()["net_liquidity"].reindex(idx).ffill()
    d_nl = nl_full.diff(H_LIQ).shift(PUB_LAG)
    fwd63 = qqq.pct_change(63).shift(-63)
    print("\n  corr(Δ6m net-liquidity, fwd 63d QQQ) by scarcity state:")
    rows = []
    for lab, m in [("calm", valid & ~stressed), ("stressed", valid & stressed)]:
        mm = m & d_nl.notna() & fwd63.notna()
        c = float(np.corrcoef(d_nl[mm], fwd63[mm])[0, 1]) if mm.sum() > 100 else np.nan
        rows.append([lab, f"{c:+.2f}", int(mm.sum())])
    print(tabulate(rows, headers=["state", "corr", "days"], tablefmt="github"))

    # ── 3. Tax-window drag by state (upgraded trigger for the prediction) ───
    in_tax = pd.Series(False, index=idx)
    for y in range(start.year, idx[-1].year + 1):
        for mth in (1, 4, 6, 9):
            i = idx.searchsorted(pd.Timestamp(y, mth, 15))
            if 0 < i < len(idx) - TAX_WIN[1]:
                in_tax.iloc[i:i + TAX_WIN[1] + 1] = True
    print("\n  QQQ annualized in tax windows, by scarcity state (2008→):")
    rows = []
    for lab, m in [("calm", valid & ~stressed), ("stressed", valid & stressed)]:
        t = in_tax & m
        o = ~in_tax & m
        rows.append([lab, f"{float(r_qqq[t].mean()) * 252:+.1%}",
                     f"{float(r_qqq[o].mean()) * 252:+.1%}",
                     f"{float(r_qqq[t].mean() - r_qqq[o].mean()) * 252:+.1%}",
                     int(t.sum())])
    print(tabulate(rows, headers=["state", "tax windows", "other days",
                                  "drag", "tax days"], tablefmt="github"))

    # ── 4. Stress throttle on the champion ───────────────────────────────────
    p = pd.read_csv(STRESS_CSV, parse_dates=["Date"], index_col="Date").sort_index()
    vidx = (load_net_liquidity()["net_liquidity"].dropna().index
            .intersection(p[["SPY", "QQQ", "HYG", "IEF", "UUP", "VIX", "VIX3M"]]
                          .dropna().index)
            .intersection(panel.dropna().index))
    nlv, pv = load_net_liquidity()["net_liquidity"].reindex(vidx), p.reindex(vidx)
    liq = soft_sign(nlv.diff(H_LIQ).shift(PUB_LAG)).shift(1)
    vix_ts = soft_sign(pv["VIX3M"] / pv["VIX"] - 1.0).shift(1)
    credit = soft_sign(pv["HYG"].pct_change(CREDIT_WIN)
                       - pv["IEF"].pct_change(CREDIT_WIN)).shift(1)
    dollar = soft_sign(-pv["UUP"].pct_change(63)).shift(1)
    o, cq = panel["QQQ_Open"].reindex(vidx), panel["QQQ_Close"].reindex(vidx)
    r_on, r_id = o / cq.shift(1) - 1.0, cq / o - 1.0
    tug = soft_sign(((1 + r_on).rolling(21).apply(np.prod, raw=True)
                     - (1 + r_id).rolling(21).apply(np.prod, raw=True))).shift(1)
    vote = pd.DataFrame({"liq": liq, "vix_ts": vix_ts, "credit": credit,
                         "dollar": dollar, "tug": tug}).mean(axis=1)
    rq = pv["QQQ"].pct_change()
    stress_v = stressed.reindex(vidx).fillna(False)

    def run(expo):
        expo = expo.clip(0.0, LEV_CAP).fillna(0.0)
        return (expo * rq - np.maximum(expo - 1, 0) * (RF + BORROW_SPREAD) / 252
                + np.maximum(1 - expo, 0) * DAILY_RF
                - TC * expo.diff().abs().fillna(0))

    live = max(vote.first_valid_index(), scarcity.dropna().index[0])
    strat = {
        "QQQ buy & hold": rq,
        "2×5-layer vote (champion)": run(2 * vote),
        "champion ×½ when stressed": run(2 * vote * np.where(stress_v, 0.5, 1.0)),
    }
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
    print(f"  stressed days in strategy window: "
          f"{float(stress_v.loc[live:].mean()):.0%}")

    # ── Graph ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    ax = axes[0]
    ax.plot(spread_bp.index, spread_bp.values, lw=0.8, color="tab:blue",
            label="(SOFR|EFFR) − (IORB|IOER), bp")
    ax.axhline(0, color="red", lw=0.8, ls="--", label="scarcity threshold")
    ax.set_ylim(-40, 60)
    for d, lab in [("2019-09-17", "Sep-19 repo spike"), ("2020-03-17", "Mar-20"),
                   ("2025-09-18", "post-buffer era")]:
        ax.axvline(pd.Timestamp(d), color="grey", lw=0.7, alpha=0.7)
        ax.text(pd.Timestamp(d), 50, lab, fontsize=7, rotation=90, va="top")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("The daily scarcity gauge (clipped; Sep-2019 printed +315bp)")
    ax = axes[1]
    for name, r in strat.items():
        eq = (1 + r.fillna(0)).cumprod()
        ax.plot(eq.index, eq.values, lw=1.3,
                color="grey" if "buy" in name else None, label=name)
    ax.set_yscale("log"); ax.set_ylabel("growth of $1 (log)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("Stress throttle on the champion")
    fig.tight_layout()
    out = RESULTS_DIR / "funding_stress.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    logger.info("Saved: %s", out)
    print("\n✓  Done.")


if __name__ == "__main__":
    main()
