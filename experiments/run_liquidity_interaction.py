#!/usr/bin/env python3
"""
Is net liquidity a STATE-DEPENDENT signal? — liquidity × volatility regime.

Marginal "net liquidity predicts equities" is a crowded thesis.  The less-
explored question: does its predictive power *depend on the regime*?  Dealer
fragility (short gamma) would be the ideal conditioner, but historical options
positioning isn't freely available — so we use the volatility regime as the
fragility proxy, which we have full history for (VIX, VIX term structure).

Hypothesis: liquidity's lead over forward equity returns is concentrated in
FRAGILE states (elevated / backwardated VIX) and near-zero when calm — i.e. the
macro tide matters most when the market is already on edge.  If true:
  • it's a genuine conditional structure, not just another marginal macro signal;
  • it explains 2022 (a fragile grind) being the weak spot of the flat overlay;
  • the actionable rule is: trust liquidity when fragile, just hold when calm.

Easy data only: FRED net liquidity + yfinance SPY/VIX/VIX3M (from datasets/).

    python experiments/run_liquidity_interaction.py
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
from posterioralpha.validation import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)
logger = logging.getLogger(__name__)
RESULTS_DIR = _bootstrap.ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
PANEL_CSV = _bootstrap.ROOT / "datasets" / "stress_panel.csv.gz"

RF = 0.04
DAILY_RF = RF / 252
H = 126            # 6-month horizon
PUB_LAG = 5


def _corr(a, b):
    m = a.notna() & b.notna()
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 30 else np.nan


def main() -> None:
    nl = load_net_liquidity()["net_liquidity"]
    p = pd.read_csv(PANEL_CSV, parse_dates=["Date"], index_col="Date").sort_index()
    idx = nl.dropna().index.intersection(p.dropna(subset=["SPY", "VIX"]).index)
    nl, p = nl.reindex(idx), p.reindex(idx)
    spy = p["SPY"]; vix = p["VIX"]
    spy_ret = spy.pct_change()

    d_nl = nl.diff(H).shift(PUB_LAG)                 # causal Δ-liquidity
    fwd = spy.pct_change(H).shift(-H)                # next-6m SPY return

    # Causal volatility-regime label: rolling 3y percentile rank of VIX (lagged)
    vix_rank = vix.rolling(756, min_periods=252).apply(
        lambda x: (x < x[-1]).mean(), raw=True).shift(1)
    calm = vix_rank <= 0.33
    fragile = vix_rank >= 0.67

    # ── 1. Conditional predictive power ──────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"  LIQUIDITY × VOL REGIME  ·  {idx.min().date()} → {idx.max().date()}")
    print("  corr(Δ6m net-liquidity, next-6m SPY return) by VIX regime")
    print("=" * 80)
    rows = []
    for lab, mask in [("Calm  (VIX low tercile)", calm),
                      ("Mid", ~calm & ~fragile),
                      ("Fragile (VIX high tercile)", fragile),
                      ("ALL", pd.Series(True, index=idx))]:
        m = mask.reindex(idx).fillna(False)
        c = _corr(d_nl[m], fwd[m])
        up = fwd[m & (d_nl > 0)].mean()
        dn = fwd[m & (d_nl < 0)].mean()
        rows.append([lab, f"{c:+.3f}", f"{up:+.1%}", f"{dn:+.1%}",
                     f"{(up - dn):+.1%}", int(m.sum())])
    print(tabulate(rows, headers=["VIX regime", "corr", "fwd|Δliq>0",
                                  "fwd|Δliq<0", "spread", "days"], tablefmt="github"))
    print("  (overlapping 6m windows — descriptive, not t-tested)")

    # ── 2. Does conditioning ADD value? ──────────────────────────────────────
    # liquidity expanding signal, causal
    liq_on = (d_nl > 0).astype(float).shift(1).fillna(0.0)
    fragile_c = fragile.reindex(idx).fillna(False).shift(1).fillna(False)

    def ret(expo):
        return pd.Series(expo * spy_ret + (1 - expo) * DAILY_RF, index=idx)

    # A: liquidity timing always   B: liquidity only when fragile, else hold
    always = ret(liq_on)
    cond = ret(np.where(fragile_c, liq_on, 1.0))
    bh = spy_ret

    print("\n" + "=" * 80)
    print("  CONDITIONAL STRATEGY  ·  'trust liquidity only when fragile, else hold'")
    print("=" * 80)
    keys = ["CAGR", "Volatility", "Sharpe", "Sortino", "Max DD", "Calmar"]
    strat = [["Liquidity timing ALWAYS", always],
             ["Liquidity ONLY when fragile (else hold)", cond],
             ["SPY buy & hold", bh]]
    table = []
    for n, r in strat:
        m = compute_metrics(r.dropna(), rf=RF)
        y22 = (1 + r[r.index.year == 2022]).prod() - 1
        table.append([n] + [f"{m.get(k, float('nan')):.2f}" for k in keys] + [f"{y22:+.0%}"])
    print(tabulate(table, headers=["strategy"] + keys + ["2022"], tablefmt="github"))

    # ── Graphs ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    ax = axes[0]
    labs = ["Calm", "Mid", "Fragile", "ALL"]
    corrs = [float(r[1]) for r in rows]
    spreads = [float(r[4].rstrip("%")) for r in rows]
    x = np.arange(len(labs)); w = 0.38
    ax.bar(x - w / 2, corrs, w, label="corr(Δliq, fwd 6m ret)", color="tab:blue")
    ax.bar(x + w / 2, [s / 100 for s in spreads], w,
           label="fwd-return spread (Δliq>0 − <0)", color="tab:orange")
    ax.set_xticks(x); ax.set_xticklabels(labs)
    ax.axhline(0, color="grey", lw=0.6); ax.legend(fontsize=9)
    ax.set_title("Net-liquidity predictive power by volatility regime "
                 "(stronger when fragile)")
    ax.grid(alpha=0.3, axis="y")
    ax = axes[1]
    for n, r in [("Liquidity ALWAYS", always),
                 ("Liquidity ONLY when fragile", cond), ("SPY buy & hold", bh)]:
        eqc = (1 + r.fillna(0)).cumprod()
        ax.plot(eqc.index, eqc.values, lw=1.5 if "hold" in n else 1.3,
                color="black" if "hold" in n else None, alpha=0.9, label=n)
    ax.set_yscale("log"); ax.set_ylabel("growth of $1 (log)"); ax.legend(fontsize=9)
    ax.set_title("Conditional vs unconditional liquidity timing on SPY")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = RESULTS_DIR / "liquidity_interaction.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    logger.info("Saved: %s", out)
    print("\n✓  Done.")


if __name__ == "__main__":
    main()
