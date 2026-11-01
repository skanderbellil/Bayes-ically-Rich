"""Stand-alone dip strategy (no SPY core): Q1-2 gap-down signals, buy at entry, sell at the official close.

full  : on a signal day 100% of capital is split equally across that day's trades; cash (0%) otherwise
cap10 : each trade gets 1/max(N,10) of capital (<=10% per name), rest in cash
Panel A: 2015-2026 by execution model (official auction prints). Panel B: 2025-2026 by entry time on the
common Yahoo-matched signal set (live tickers only) + the pre-market signal.
"""
import _bootstrap  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from posterioralpha.dipbuy import backtest as bt, stats
from posterioralpha.dipbuy.paths import DOCS_DIR, RAW_DIR

spy = pd.read_parquet(RAW_DIR / "spy.parquet")
D = pd.read_parquet(RAW_DIR / "dip_trades.parquet")
D = D[D.Q <= 2]


def full(T, col, days):
    t = T[["date", col]].dropna()
    return t.groupby("date")[col].mean().reindex(days).fillna(0.0)


# ---- panel A inputs
daysA = spy.loc["2015-01-01":"2026-10-30"].index
A = {"auction open (E1)": "E1_auction", "strict limit (E2)": "E2_strict_limit",
     "other-venue entry (E3)": "E3_other_venue_entry", "auction +1% slippage": "E5_slip_1.0%"}
# ---- panel B inputs (same construction as run_dipbuy_growth_recent.py)
START = "2025-01-01"
P = pd.read_parquet(RAW_DIR / "panel.parquet"); P = P[P.date >= START]
H = pd.read_parquet(RAW_DIR / "intraday_60m.parquet"); spyH = H[H.sym == "SPY"].set_index("date")
P = P.merge(H[H.sym != "SPY"], on=["sym", "date"], how="inner"); P = P[(P.y_open / P.open - 1).abs() < 0.15]
P["spy_pre_gap"] = P.date.map(spyH.pre_last) / P.date.map(spy.close.shift(1)) - 1
thr = np.maximum(bt.THR_FLOOR, bt.THR_SD * P.sd60); el = bt.eligible(P) & (P.Q <= 2)
real = P[el & (P.agap <= -thr)].copy()
pre = P[el & ((P.pre_last / P.prev_close - 1 - P.spy_pre_gap) <= -thr) & P.pre_last.notna()].copy()
c = bt.COST
real["auction open (E1)"] = real.close / real.open - 1 - c
real["strict limit (E2)"] = real["auction open (E1)"].where(real.cons_lo < real.open * (1 - 1e-6))
real["auction +1% slippage"] = real.close / (real.open * 1.01) - 1 - c
real["enter ~10:30"] = real.close / real.h1_close - 1 - c
real["first-hour VWAP"] = real.close / real.h1_typ - 1 - c
pre["pre-market signal, MOO"] = pre.close / pre.open - 1 - c
daysB = spy.loc[START:"2026-10-30"].index

pal = {"auction open (E1)": "#2a78d6", "strict limit (E2)": "#eb6834", "other-venue entry (E3)": "#1baf7a",
       "auction +1% slippage": "#6250d6", "enter ~10:30": "#1baf7a", "first-hour VWAP": "#eda100", "pre-market signal, MOO": "#e34948"}
rows = []
fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), dpi=140)
for ax, panel, items, days in ((axes[0], "2015-2026", [(k, D, v) for k, v in A.items()], daysA),
                               (axes[1], "2025-2026", [(k, real, k) for k in ["auction open (E1)", "strict limit (E2)", "auction +1% slippage",
                                                                               "enter ~10:30", "first-hour VWAP"]] + [("pre-market signal, MOO", pre, "pre-market signal, MOO")], daysB)):
    ends = []
    for lab, T, col in items:
        r = full(T, col, days)
        r10 = bt.sleeve(T, col, days, per_name=0.10)
        for sizing, x in (("full", r), ("cap10", r10)):
            p = stats.perf(x, spy.ret.reindex(days))
            p.update(panel=panel, strategy=lab, sizing=sizing, trades=int(T[col].notna().sum()),
                     days_active=float((x != 0).mean()))
            if panel == "2015-2026":
                for per, sl in (("IS", slice("2015", "2021")), ("OOS", slice("2022", "2026"))):
                    p[f"CAGR {per}"] = stats.perf(x.loc[sl])["CAGR"]
            rows.append(p)
        cc = (1 + r10).cumprod()
        ax.plot(cc.index, cc.values, color=pal[lab], lw=1.8, label=lab)
        ends.append((cc.iloc[-1], lab))
    ends.sort(); last = None
    for v, lab in ends:
        y = v if last is None else max(v, last * 1.04); last = y
        ax.annotate(f"{v:.2f}x", (days[-1], y), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8, color="#0b0b0b")
    ax.set_yscale("log"); ax.axhline(1, color="#c3c2b7", lw=0.8)
    if panel == "2015-2026":
        ax.axvline(pd.Timestamp("2022-01-01"), color="#52514e", lw=1, ls=":")
        ax.text(pd.Timestamp("2022-02-01"), ax.get_ylim()[1] * 0.85, "out-of-sample →", fontsize=8, color="#52514e")
    ax.set_title(f"{panel}: dip strategy alone (no SPY): ≤10% per Q1–Q2 dip, rest in cash",
                 fontsize=8.5, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left"); ax.set_ylabel("growth of $1 (log)", fontsize=8)
    ax.set_xlim(days[0], days[-1] + (days[-1] - days[0]) * 0.07)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(DOCS_DIR / "standalone_growth.png")
R = pd.DataFrame(rows).set_index(["panel", "sizing", "strategy"])
R.to_csv(DOCS_DIR / "standalone.csv")
print(R.drop(columns=["x$1"]).to_string(float_format=lambda x: f"{x:.3f}"))
