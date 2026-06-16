#!/usr/bin/env python3
"""Charts: managed-futures convexity + the long-only SPY+MF books."""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _bootstrap  # noqa: F401
from run_liquid_alts_basket import DATA, RESULTS_DIR, MF, basket

ANN = 12


def main():
    wide = pd.read_csv(DATA / "liquid_alts_wide.csv", index_col=0, parse_dates=True)
    m = (1 + wide.pct_change()).resample("ME").apply(lambda x: x.prod() - 1)
    spy = m["SPY"]
    mf = basket(m, MF).rename("MF")
    d = pd.concat([spy.rename("SPY"), mf], axis=1).dropna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    # 1. convexity scatter: MF monthly return vs SPY monthly return
    ax = axes[0]
    ax.scatter(d["SPY"]*100, d["MF"]*100, s=28, alpha=0.6, color="#1565C0",
               edgecolor="none")
    z = np.polyfit(d["SPY"], d["MF"], 2)
    xs = np.linspace(d["SPY"].min(), d["SPY"].max(), 100)
    ax.plot(xs*100, np.polyval(z, xs)*100, color="#B71C1C", lw=2.2,
            label="quadratic fit (convex)")
    ax.axhline(0, color="k", lw=0.7); ax.axvline(0, color="k", lw=0.7)
    ax.set_xlabel("SPY monthly return %"); ax.set_ylabel("managed-futures basket %")
    ax.set_title("Convexity: managed futures rise when stocks fall\n(the orthogonal, crisis-paying return)",
                 fontweight="bold", fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # 2. growth of $1, 2020-2026, SPY vs SPY+MF books
    ax = axes[1]
    win = d.index >= "2020-01-01"
    books = {"SPY": d["SPY"], "SPY + 15% MF": 0.85*d["SPY"]+0.15*d["MF"],
             "SPY + 30% MF": 0.70*d["SPY"]+0.30*d["MF"]}
    cols = {"SPY": "#888", "SPY + 15% MF": "#B71C1C", "SPY + 30% MF": "#1565C0"}
    for n, r in books.items():
        r = r.loc[win]
        eq = (1 + r).cumprod()
        sr = (r.mean()-0.04/12)*12/(r.std()*np.sqrt(12))
        dd = (eq/eq.cummax()-1).min()
        ax.plot(eq.index, eq.values, color=cols[n], lw=2.2 if "15" in n else 1.7,
                label=f"{n}  (Sharpe {sr:.2f}, maxDD {dd:.0%})")
    ax.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31"), color="orange", alpha=0.13)
    ax.set_ylabel("growth of $1"); ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_title("Long-only SPY + managed futures (2020-26)\n2022 shaded — MF cushioned the crash",
                 fontweight="bold", fontsize=11)

    fig.tight_layout()
    out = RESULTS_DIR / "liquid_alts_dashboard.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
