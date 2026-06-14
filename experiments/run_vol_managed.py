#!/usr/bin/env python3
"""
More return WITHOUT estimating returns — pure volatility management.

User insight (2026-06-14): returns can't be forecast (noise), so strip out
return estimation entirely. The only input here is VOLATILITY, which IS
predictable (it clusters), and the COMPOUNDING math turns that into more
terminal wealth — no return forecast anywhere.

The mechanism is variance drain: geometric (compound) growth
    g  ≈  μ  −  σ²/2
so cutting the σ² SPIKES lifts g even if the average return μ is unchanged.
For a 2× leveraged ETF the drain is ~4× worse (σ→2σ ⇒ σ²→4σ²), which is
exactly why leveraged ETFs "decay" — and why volatility-managing them pays.

Strategy (continuous, no return estimation, daily, long-only, Alpaca):
    exposure_t = clip( target_vol / EWMA_vol_{t-1}(QQQ) , 0, 1 ) held in QLD
    (2× QQQ); the rest in the managed-futures basket (or cash before 2019).
When vol is calm → full 2× exposure; when vol spikes → dial down to the safe
asset. The ONLY estimate is volatility.

Outputs the return/vol/drawdown table plus a decomposition of arithmetic
return − variance drain = geometric growth, and a chart dashboard.
"""
import sys
import numpy as np
import pandas as pd
from tabulate import tabulate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _bootstrap  # noqa: F401

DATA = _bootstrap.ROOT / "datasets"
RES = _bootstrap.ROOT / "results"
RES.mkdir(exist_ok=True)
ANN, RF = 252, 0.04
SC = 10 / 1e4
START = "2011-01-01"


def decompose(r):
    r = r.dropna()
    arith = r.mean() * ANN
    sd = r.std() * np.sqrt(ANN)
    geo = (1 + r).prod() ** (ANN / len(r)) - 1
    eq = (1 + r).cumprod()
    return dict(arith=arith, vol=sd, drain=arith - geo, geo=geo,
                sharpe=(arith - RF) / (sd + 1e-12),
                maxdd=float((eq / eq.cummax() - 1).min()))


def ewma_vol(r, halflife=20):
    return r.ewm(halflife=halflife).std() * np.sqrt(ANN)


def vol_target(qld, safe, qqq, target, cap=1.0):
    """exposure = target/EWMA_vol(QQQ), held in QLD; rest in safe. Daily."""
    v = ewma_vol(qqq).shift(1)
    w = (target / (2 * v)).clip(0, cap)          # QLD is 2× → its vol ≈ 2·qqq vol
    r = w * qld + (1 - w) * safe
    turn = w.diff().abs().fillna(0)
    return (r - turn * SC).dropna(), w


def main():
    px = pd.read_csv(DATA / "levered_stacked.csv", index_col=0, parse_dates=True).loc[START:]
    rets = px.pct_change()
    qqq, qld = rets["QQQ"], rets["QLD"]
    rf_d = RF / ANN
    mf = rets[["DBMF", "KMLM", "CTA"]].mean(axis=1, skipna=True).fillna(rf_d)

    books = {"QQQ buy & hold": (qqq, None), "QLD buy & hold (2x)": (qld, None)}
    weights = {}
    for tv in (0.20, 0.25, 0.30):
        r, w = vol_target(qld, mf, qqq, tv)
        books[f"vol-managed QLD @ {int(tv*100)}%"] = (r, w)
        weights[tv] = w

    rows = []
    for n, (r, _) in books.items():
        d = decompose(r.reindex(qld.index).dropna() if isinstance(r, pd.Series) else r)
        rows.append([n, f"{d['arith']:+.1%}", f"{d['vol']:.0%}", f"−{d['drain']:.1%}",
                     f"{d['geo']:+.1%}", f"{d['sharpe']:.2f}", f"{d['maxdd']:.0%}"])
    print(f"\nMore return WITHOUT return estimation — pure vol management "
          f"({px.index[0].date()}→{px.index[-1].date()}):")
    print(tabulate(rows, headers=["book", "arith ret", "vol", "var drain",
                                  "GEO (compound)", "Sharpe", "maxDD"],
                   tablefmt="github"))
    print("\nThe lever: only VOLATILITY is estimated. Stabilising it shrinks "
          "the σ²/2 variance drain → higher GEO growth — no return forecast.")

    # ── dashboard ─────────────────────────────────────────────────────────
    idx = books["vol-managed QLD @ 25%"][0].index
    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(3, 2, hspace=0.33, wspace=0.22, height_ratios=[2, 1, 1])
    ax = fig.add_subplot(gs[0, :])
    for n, c, lw in [("QQQ buy & hold", "#888", 1.6), ("QLD buy & hold (2x)", "#ccc", 1.2),
                     ("vol-managed QLD @ 25%", "#B71C1C", 2.4)]:
        eq = (1 + books[n][0].reindex(idx).fillna(0)).cumprod()
        d = decompose(books[n][0])
        ax.plot(eq.index, eq.values, color=c, lw=lw,
                label=f"{n}  (GEO {d['geo']:+.0%}, Sharpe {d['sharpe']:.2f}, maxDD {d['maxdd']:.0%})")
    ax.set_yscale("log"); ax.set_ylabel("growth of $1 (log)")
    ax.legend(loc="upper left", fontsize=10); ax.grid(alpha=0.3, which="both")
    ax.set_title("Volatility-managed leveraged equity — no return estimation, only vol",
                 fontweight="bold", fontsize=13)

    # variance-drain decomposition bars
    ax = fig.add_subplot(gs[1, 0])
    names = ["QQQ buy & hold", "QLD buy & hold (2x)", "vol-managed QLD @ 25%"]
    ar = [decompose(books[n][0])["arith"]*100 for n in names]
    dr = [decompose(books[n][0])["drain"]*100 for n in names]
    ge = [decompose(books[n][0])["geo"]*100 for n in names]
    x = np.arange(len(names))
    ax.bar(x, ar, 0.5, label="arithmetic return", color="#90CAF9")
    ax.bar(x, [-d for d in dr], 0.5, bottom=ar, label="− variance drain (σ²/2)", color="#EF9A9A")
    ax.plot(x, ge, "ko", ms=9, label="= GEO (compound) growth")
    ax.set_xticks(x); ax.set_xticklabels(["QQQ", "QLD 2x", "vol-mgd"], fontsize=9)
    ax.set_ylabel("% / yr"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("Variance drain: QLD's leverage bleeds to σ²/2;\nvol management recovers it", fontsize=10.5)

    # rolling realized vol — vol-managed is STABILIZED
    ax = fig.add_subplot(gs[1, 1])
    for n, c in [("QLD buy & hold (2x)", "#ccc"), ("vol-managed QLD @ 25%", "#B71C1C")]:
        rv = books[n][0].reindex(idx).rolling(42).std() * np.sqrt(ANN)
        ax.plot(rv.index, rv.values*100, color=c, lw=1.4, label=n.split(" buy")[0])
    ax.axhline(25, color="k", ls="--", lw=0.8, label="25% target")
    ax.set_ylabel("rolling 42d vol %"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("Realized vol: managed book holds ~target;\nQLD's vol explodes in crises", fontsize=10.5)

    # exposure dial over time
    ax = fig.add_subplot(gs[2, 0])
    w = weights[0.25].reindex(idx)
    ax.fill_between(w.index, w.values, 0, color="#1565C0", alpha=0.4)
    ax.set_ylabel("QLD weight (0-1)"); ax.grid(alpha=0.3); ax.set_ylim(0, 1)
    ax.set_title("The vol dial: leverage UP when calm, OFF when vol spikes", fontsize=10.5)

    # drawdowns
    ax = fig.add_subplot(gs[2, 1])
    for n, c in [("QQQ buy & hold", "#888"), ("vol-managed QLD @ 25%", "#B71C1C")]:
        eq = (1 + books[n][0].reindex(idx).fillna(0)).cumprod(); dd = eq/eq.cummax()-1
        ax.fill_between(dd.index, dd.values, 0, color=c, alpha=0.35, label=n.split(" buy")[0])
    ax.set_ylabel("drawdown"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_title("Drawdowns", fontsize=10.5)

    out = RES / "vol_managed_dashboard.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved {out}")
    pd.DataFrame(rows).to_csv(RES / "vol_managed.csv", index=False)


if __name__ == "__main__":
    main()
