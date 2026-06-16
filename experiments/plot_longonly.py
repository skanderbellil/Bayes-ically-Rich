#!/usr/bin/env python3
"""Charts for the long-only / daily / no-leverage retail books."""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _bootstrap  # noqa: F401
from posterioralpha.data import load_etf_universe_prices
from posterioralpha.backtest.alpaca_costs import HALF_SPREAD_BPS
from run_retail_longonly_daily import (month_ends, monthly_rebal, TREND_ASSETS,
                                       ALTS, DATA, RESULTS_DIR, ANN)


def main():
    uni = load_etf_universe_prices()
    alts = pd.read_csv(DATA / "retail_alts.csv", index_col=0, parse_dates=True)
    px = pd.concat([uni[["SPY", "TLT", "GLD", "DBC", "IEF", "SHY"]],
                    alts[["BTAL", "MNA", "DBMF", "KMLM"]]], axis=1).sort_index()
    rets = px.pct_change(); me = month_ends(px.index); ma = px.rolling(200).mean()

    def w_trend(t):
        on = [a for a in TREND_ASSETS if px.loc[t, a] > ma.loc[t, a]]
        return pd.Series({a: 1/len(on) for a in on}) if on else pd.Series({"SHY": 1.0})

    def w_alts(t):
        av = [a for a in ALTS if not np.isnan(px.loc[t, a])]
        return pd.Series({a: 1/len(av) for a in av}) if av else pd.Series({"SHY": 1.0})

    tr = monthly_rebal(w_trend, rets, me)[0]
    al = monthly_rebal(w_alts, rets, me)[0]
    spy = rets["SPY"]
    df = pd.concat([spy.rename("spy"), tr.rename("tr"), al.rename("al")], axis=1).dropna()
    books = {
        "SPY buy & hold": df["spy"],
        "70/20/10 (eq/trend/alts)": 0.7*df["spy"] + 0.2*df["tr"] + 0.1*df["al"],
        "60/20/20 (defensive)": 0.6*df["spy"] + 0.2*df["tr"] + 0.2*df["al"],
    }
    idx = df.index
    cols = {"SPY buy & hold": "#888", "70/20/10 (eq/trend/alts)": "#B71C1C",
            "60/20/20 (defensive)": "#1565C0"}

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.28)
    ax = fig.add_subplot(gs[0])
    for n, r in books.items():
        eq = (1 + r).cumprod()
        sr = (r.mean()-0.04/ANN)*ANN/(r.std()*np.sqrt(ANN))
        dd = (eq/eq.cummax()-1).min()
        ax.plot(eq.index, eq.values, color=cols[n], lw=2.2 if "70" in n else 1.7,
                label=f"{n}  —  Sharpe {sr:.2f}, maxDD {dd:.0%}, ${eq.iloc[-1]:.0f}")
    ax.set_yscale("log"); ax.set_ylabel("growth of $1 (log)")
    ax.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31"), color="orange", alpha=0.12)
    ax.set_title("Long-only, daily, NO leverage — all Alpaca ETFs, no shorting (net of cost)",
                 fontweight="bold", fontsize=13)
    ax.legend(loc="upper left", fontsize=10); ax.grid(alpha=0.3, which="both")

    ax = fig.add_subplot(gs[1])
    for n, r in books.items():
        eq = (1 + r).cumprod(); dd = eq/eq.cummax()-1
        ax.fill_between(dd.index, dd.values, 0, color=cols[n], alpha=0.3, label=n)
    ax.set_ylabel("drawdown"); ax.legend(loc="lower left", fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.grid(alpha=0.3)
    ax.set_title("Drawdowns — diversified books cut the worst loss by a third; 2022 (shaded) is much shallower", fontsize=11)

    out = RESULTS_DIR / "longonly_dashboard.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
