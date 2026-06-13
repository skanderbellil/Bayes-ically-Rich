#!/usr/bin/env python3
"""Charts for the retail-deployable orthogonal-alpha ensemble (Idea 18)."""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _bootstrap  # noqa: F401
from posterioralpha.data import load_etf_universe_prices  # noqa: F401
from run_retail_alts_ensemble import (build_etfmom, risk_parity, to_monthly,
                                      ALTS, FACTORS, DATA, RESULTS_DIR, sl)

SPLIT = "2019-07-01"


def main():
    alts = pd.read_csv(DATA / "retail_alts.csv", index_col=0, parse_dates=True)
    m = to_monthly(alts)
    spy = m["SPY"].rename("core")
    zoo_a = m[ALTS].mean(axis=1).rename("zoo_a").dropna()
    etf = build_etfmom()

    df = pd.concat([spy, etf, zoo_a], axis=1).dropna()
    ens = risk_parity(df, ["core", "etfmom", "zoo_a"])
    ens_2 = risk_parity(df, ["core", "etfmom"])
    idx = ens.index
    spy_a = df["core"].reindex(idx)

    fig = plt.figure(figsize=(13, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.35, wspace=0.22)

    # 1. cumulative growth (log)
    ax = fig.add_subplot(gs[0, :])
    for s, lab, c, lw in [(spy_a, "SPY only (Sharpe 0.99)", "#888", 1.6),
                          (ens_2, "SPY + ETF-momentum (1.23)", "#1565C0", 1.8),
                          (ens, "SPY + ETF-mom + liquid-alts (1.42)", "#B71C1C", 2.4)]:
        eq = (1 + s.reindex(idx).fillna(0)).cumprod()
        ax.plot(eq.index, eq.values, color=c, lw=lw, label=f"{lab}  →  ${eq.iloc[-1]:.1f}")
    ax.set_yscale("log"); ax.set_ylabel("growth of $1 (log)")
    ax.set_title("Retail orthogonal-alpha ensemble vs SPY  (all Alpaca-tradable, net of cost)",
                 fontweight="bold", fontsize=13)
    ax.legend(loc="upper left", fontsize=10); ax.grid(alpha=0.3, which="both")
    ax.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31"),
               color="orange", alpha=0.12)

    # 2. drawdowns
    ax = fig.add_subplot(gs[1, :])
    for s, lab, c in [(spy_a, "SPY", "#888"), (ens, "retail ensemble", "#B71C1C")]:
        eq = (1 + s.reindex(idx).fillna(0)).cumprod()
        dd = eq / eq.cummax() - 1
        ax.fill_between(dd.index, dd.values, 0, color=c, alpha=0.35, label=lab)
    ax.set_ylabel("drawdown"); ax.legend(loc="lower left", fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.grid(alpha=0.3); ax.set_title("Drawdowns — ensemble −14% max vs SPY −24%", fontsize=11)

    # 3. the three orthogonal sleeves (cumulative)
    ax = fig.add_subplot(gs[2, 0])
    for s, lab, c in [(spy, "SPY core", "#888"),
                      (etf, "ETF-momentum L/S", "#1565C0"),
                      (zoo_a, "liquid-alts (BTAL/MNA/DBMF)", "#2E7D32")]:
        eq = (1 + s.reindex(idx).fillna(0)).cumprod()
        ax.plot(eq.index, eq.values, color=c, lw=1.6, label=lab)
    ax.set_yscale("log"); ax.legend(fontsize=8.5); ax.grid(alpha=0.3, which="both")
    ax.set_title("The 3 orthogonal sleeves (corr ≈ 0 to each other)", fontsize=10.5)

    # 4. rolling 24m corr of liquid-alts to SPY (the orthogonality, stable)
    ax = fig.add_subplot(gs[2, 1])
    rc = zoo_a.reindex(idx).rolling(24).corr(spy_a)
    ax.plot(rc.index, rc.values, color="#2E7D32", lw=1.8)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.fill_between(rc.index, rc.values, 0, where=(rc < 0), color="#2E7D32", alpha=0.2)
    ax.set_ylim(-1, 1); ax.grid(alpha=0.3)
    ax.set_title("Liquid-alts corr to SPY (24m roll) — stably negative", fontsize=10.5)

    fig.suptitle("", fontsize=1)
    out = RESULTS_DIR / "retail_ensemble_dashboard.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved {out}")

    # per-year bar chart (separate, clean)
    fig2, ax2 = plt.subplots(figsize=(12, 4.5))
    yr_s = spy_a.groupby(spy_a.index.year).apply(lambda x: (1 + x).prod() - 1)
    yr_e = ens.groupby(ens.index.year).apply(lambda x: (1 + x).prod() - 1)
    yrs = sorted(set(yr_s.index) & set(yr_e.index))
    x = np.arange(len(yrs)); w = 0.4
    ax2.bar(x - w/2, [yr_s.get(y, 0)*100 for y in yrs], w, label="SPY", color="#888")
    ax2.bar(x + w/2, [yr_e.get(y, 0)*100 for y in yrs], w, label="retail ensemble", color="#B71C1C")
    ax2.set_xticks(x); ax2.set_xticklabels(yrs, rotation=0)
    ax2.axhline(0, color="k", lw=0.8); ax2.set_ylabel("annual return %")
    ax2.legend(fontsize=10); ax2.grid(alpha=0.3, axis="y")
    ax2.set_title("Calendar-year returns — ensemble cuts the down years, gives up little upside",
                  fontweight="bold", fontsize=12)
    out2 = RESULTS_DIR / "retail_ensemble_yearly.png"
    fig2.savefig(out2, dpi=140, bbox_inches="tight")
    print(f"saved {out2}")

    print(f"ensemble Sharpe {sl(ens)['sharpe']:.2f}  ann {sl(ens)['ann']:+.1%}  "
          f"maxDD {sl(ens)['maxdd']:.0%}  |  SPY Sharpe {sl(spy_a)['sharpe']:.2f}")


if __name__ == "__main__":
    main()
