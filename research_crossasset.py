#!/usr/bin/env python3
"""Cross-asset ETF portfolios vs holding SPY — the survivorship-clean question.

After single-stock strategies failed (PEAD = untradeable jump + tax; factors = no
edge vs survivorship-matched benchmark), the honest remaining avenue is multi-asset
diversification using ETFs, which barely delist (no survivorship bias), trade with
tight spreads, and rebalance annually (LTCG, near-zero hustle).

Finding: a diversified portfolio robustly IMPROVES risk-adjusted return (Sharpe
0.78 -> 0.92) and halves drawdown (-51% -> -29%), but earns LESS in absolute terms.
Levering it to equity risk does NOT beat SPY once 6% retail margin cost is paid.
=> It is a risk-reducer, not a return-beater, for a retail investor.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ASSETS = ["SPY", "TLT", "IEF", "GLD", "DBC", "VNQ", "LQD", "EEM"]
MARGIN_ANNUAL = 0.06


def load():
    m = pd.read_csv("results/pead/etf_monthly.csv", index_col=0, parse_dates=True)
    ret = m.pct_change()
    cols = [c for c in ASSETS if c in ret.columns and ret[c].notna().sum() > 200]
    ret = ret[cols].dropna(subset=["SPY", "TLT", "GLD"])
    return ret[ret.index >= ret[["SPY", "TLT", "GLD"]].dropna().index.min()], cols


def ann(r): return (1 + r.mean()) ** 12 - 1
def shp(r): return np.sqrt(12) * r.mean() / r.std()
def mdd(r):
    eq = np.cumprod(1 + r.dropna().values)
    return float(((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min())


def inv_vol_annual(ret, cols):
    idx = ret.index; out = []
    for y in range(idx.min().year + 1, 2027):
        cand = idx[idx <= pd.Timestamp(f"{y-1}-12-31")]
        if len(cand) == 0: continue
        form = cand[-1]; hold = idx[idx > form][:12]
        trail = ret.loc[idx[idx <= form], cols].iloc[-12:]
        usable = [c for c in cols if trail[c].notna().sum() >= 12]
        if len(usable) < 2 or len(hold) < 1: continue
        w = 1 / trail[usable].std(); w /= w.sum()
        out.append((ret.loc[hold, usable] * w).sum(axis=1))
    return pd.concat(out)


def main():
    ret, cols = load()
    spy = ret["SPY"].dropna()
    sixty = (0.6 * ret["SPY"] + 0.4 * ret["IEF"]).dropna()
    ew = ret[cols].mean(axis=1).dropna()
    rp = inv_vol_annual(ret, cols)

    print("=" * 66)
    print("CROSS-ASSET ETF PORTFOLIOS vs HOLDING SPY")
    print("=" * 66)
    print(f"window {ret.index.min().date()}..{ret.index.max().date()}   assets={cols}\n")
    rows = {"SPY (100% equity)": spy, "60/40 (SPY/IEF)": sixty,
            "Equal-weight": ew, "Risk-parity (inv-vol)": rp}
    print(f"{'Portfolio':24s} {'Annual':>8s} {'Sharpe':>7s} {'MaxDD':>7s}")
    for n, r in rows.items():
        print(f"{n:24s} {ann(r)*100:+7.1f}% {shp(r):7.2f} {mdd(r)*100:6.0f}%")

    # Levered-to-SPY-vol test (net of margin)
    print("\nLevered to SPY vol, net of 6% margin cost:")
    for n, r in [("60/40", sixty), ("Risk-parity", rp)]:
        r = r.dropna(); L = spy.std() / r.std()
        lev = (L * r - max(L - 1, 0) * MARGIN_ANNUAL / 12).reindex(spy.index).dropna()
        print(f"  {n:14s} L={L:.2f}x  ann {ann(lev)*100:+6.1f}%  Sharpe {shp(lev):.2f}  maxDD {mdd(lev)*100:5.0f}%")
    print(f"  {'SPY':14s}          ann {ann(spy)*100:+6.1f}%  Sharpe {shp(spy):.2f}  maxDD {mdd(spy)*100:5.0f}%")

    # Plot equity curves
    fig, ax = plt.subplots(figsize=(13, 7))
    for n, r, c, lw in [("SPY", spy, "#000000", 2.5), ("60/40", sixty, "#1f77b4", 2),
                        ("Risk-parity", rp, "#2ca02c", 2.2)]:
        r = r.dropna()
        ax.plot(r.index, 100_000 * np.cumprod(1 + r.values), color=c, lw=lw,
                label=f"{n}: {ann(r)*100:+.1f}%/yr, Sharpe {shp(r):.2f}, DD {mdd(r)*100:.0f}%")
    ax.set_yscale("log"); ax.grid(True, alpha=0.3, which="both")
    ax.set_ylabel("Growth of $100k (log)", fontweight="bold")
    ax.set_title("Cross-asset diversification: higher Sharpe, lower drawdown, lower return",
                 fontweight="bold")
    ax.legend(fontsize=10)
    out = Path("results/pead/crossasset_vs_spy.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
