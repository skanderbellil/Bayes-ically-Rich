"""Growth of $1, SPY + Q1-2 dip overlay, by ENTRY model, on a common signal set (window default 2025-01 -> 2026-10).

Entries: official auction open (E1), strict limit at the open (E2), auction +1% (E5), ~10:30 (end of first hourly
bar), first-hour VWAP proxy, and the tradable pre-market signal (gap from the last pre-market print, bought in the
opening auction). Exit always at the official close. Signals restricted to those with Yahoo intraday bars, so all
lines trade the same names (live tickers only). For late entries the slice sits in cash between 09:30 and entry.
"""
import sys

import _bootstrap  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from posterioralpha.dipbuy import backtest as bt, stats
from posterioralpha.dipbuy.paths import DOCS_DIR, RAW_DIR

START = sys.argv[1] if len(sys.argv) > 1 else "2025-01-01"
P = pd.read_parquet(RAW_DIR / "panel.parquet")
P = P[P.date >= START]
H = pd.read_parquet(RAW_DIR / "intraday_60m.parquet")
spy = pd.read_parquet(RAW_DIR / "spy.parquet")
spyH = H[H.sym == "SPY"].set_index("date")
P = P.merge(H[H.sym != "SPY"], on=["sym", "date"], how="inner")
P = P[(P.y_open / P.open - 1).abs() < 0.15]
P["spy_pre_gap"] = P.date.map(spyH.pre_last) / P.date.map(spy.close.shift(1)) - 1
thr = np.maximum(bt.THR_FLOOR, bt.THR_SD * P.sd60)
el = bt.eligible(P) & (P.Q <= 2)
real = P[el & (P.agap <= -thr)].copy()
pre = P[el & ((P.pre_last / P.prev_close - 1 - P.spy_pre_gap) <= -thr) & P.pre_last.notna()].copy()
c = bt.COST
real["auction open (E1)"] = real.close / real.open - 1 - c
real["strict limit (E2)"] = real["auction open (E1)"].where(real.cons_lo < real.open * (1 - 1e-6))
real["auction +1% slippage"] = real.close / (real.open * 1.01) - 1 - c
real["enter ~10:30"] = real.close / real.h1_close - 1 - c
real["first-hour VWAP"] = real.close / real.h1_typ - 1 - c
pre["pre-market signal, MOO"] = pre.close / pre.open - 1 - c

days = spy.loc[START:"2026-10-30"].index
curves = {"SPY": spy.ret.reindex(days).fillna(0)}
rows = {"SPY": stats.perf(curves["SPY"])}
for col, T in [("auction open (E1)", real), ("strict limit (E2)", real), ("auction +1% slippage", real),
               ("enter ~10:30", real), ("first-hour VWAP", real), ("pre-market signal, MOO", pre)]:
    ov = bt.overlay(T, col, spy, days)
    curves[col] = ov
    pt = stats.clustered_mean(T[col], T.date)
    rows[col] = {**stats.perf(ov, curves["SPY"]), "trades": int(T[col].notna().sum()), "mean/trade": pt["mean"], "t": pt["t"]}
S = pd.DataFrame(rows).T
print(S.to_string(float_format=lambda x: f"{x:.3f}"))
S.to_csv(DOCS_DIR / f"growth_recent_{START[:4]}.csv")

pal = {"SPY": "#8a8984", "auction open (E1)": "#2a78d6", "strict limit (E2)": "#eb6834", "auction +1% slippage": "#6250d6",
       "enter ~10:30": "#1baf7a", "first-hour VWAP": "#eda100", "pre-market signal, MOO": "#e34948"}
fig, ax = plt.subplots(figsize=(11, 5.6), dpi=140)
ends = []
for k, r in curves.items():
    cc = (1 + r).cumprod()
    ax.plot(cc.index, cc.values, color=pal[k], lw=2.4 if k == "SPY" else 1.8, ls="--" if k == "SPY" else "-", label=k)
    ends.append((cc.iloc[-1], k))
# spread end labels so they don't collide
ends.sort()
ypos, last = [], None
for v, k in ends:
    y = v if last is None else max(v, last * 1.018)
    ypos.append((y, v, k)); last = y
for y, v, k in ypos:
    ax.annotate(f"{k}  {v:.3f}x", (days[-1], y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color="#0b0b0b")
ax.set_yscale("log"); ax.set_xlim(days[0], days[-1] + pd.Timedelta(days=170))
ax.set_ylabel("growth of $1 (log)", fontsize=9); ax.grid(axis="y", color="#e6e5e0", lw=0.8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.set_title(f"SPY + Q1–Q2 dip overlay by entry model, {START[:7]} → 2026-10 (same {len(real)} signals; pre-market line: {len(pre)} signals)",
             fontsize=9, loc="left")
fig.tight_layout(); fig.savefig(DOCS_DIR / f"growth_recent_{START[:4]}.png"); print("chart written")
