"""Europe, simple timed-entry check (Yahoo hourly bars, 2025-01 -> 2026-10, live tickers only).

Same frozen signal on the European daily panel. Exit at the daily close (closing auction on these venues).
Entries: daily open (auction print), first regular hourly bar close (~1 h after the open), first-hour typical price,
strict limit (filled only if the day's low < open).
"""
from concurrent.futures import ThreadPoolExecutor

import _bootstrap  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from posterioralpha.dipbuy import backtest as bt, europe, stats, yahoo
from posterioralpha.dipbuy.paths import DOCS_DIR, RAW_DIR

raw = pd.read_parquet(RAW_DIR / "europe_raw.parquet"); U = pd.read_parquet(RAW_DIR / "europe_universe.parquet")
idx = pd.read_parquet(RAW_DIR / "europe_index.parquet")
P = europe.build(raw, idx, U)
thr = np.maximum(bt.THR_FLOOR, bt.THR_SD * P.sd60)
el = (P.date >= "2025-01-01") & P.Q.notna() & (P.prev_close >= 5) & P.sd60.notna() & (P.gap.abs() < 0.5) & ~P.open_bad
S = P[el & (P.agap.abs() >= thr)].copy()
S["side"] = np.where(S.agap < 0, "down", "up")
syms = sorted(S.sym.unique())
print(len(S), "signals,", len(syms), "symbols", flush=True)


def first_hour(s):
    d = yahoo.chart(s, "60m", rng="730d")
    if d.empty:
        return None
    d = d.dropna(subset=["close"])
    d["date"] = d.index.tz_localize(None).normalize()
    f = d.groupby("date").first()
    return pd.DataFrame({"sym": s, "date": f.index, "h_open": f.open.values, "h1_close": f.close.values,
                         "h1_typ": ((f.high + f.low + f.close) / 3).values})


with ThreadPoolExecutor(4) as ex:
    H = pd.concat([x for x in ex.map(first_hour, syms) if x is not None], ignore_index=True)
S = S.merge(H, on=["sym", "date"], how="inner")
S = S[(S.h_open / S.open - 1).abs() < 0.15]
print(len(S), "signals with hourly bars; hourly-bar open == daily open:", np.isclose(S.h_open, S.open, rtol=1e-4).mean().round(3))
S["stale_1h"] = np.isclose(S.h1_close, S.open, rtol=1e-6)   # no price change in the first hour (thin names)
S.to_parquet(RAW_DIR / "europe_intraday_signals.parquet", index=False)
rows = []
for side, g in S.groupby("side"):
    sg = 1 if side == "down" else -1
    ent = {"open print": g.open, "first-hour close (~+1h)": g.h1_close, "first-hour avg price": g.h1_typ}
    fill = (g.low < g.open * (1 - 1e-6)) if side == "down" else (g.high > g.open * (1 + 1e-6))
    for lab, px in ent.items():
        r = sg * (g.close / px - 1) - bt.COST
        for liq, m in (("Q1-2", g.Q <= 2), ("Q3-5", g.Q >= 3)):
            c = stats.clustered_mean(r[m], g.date[m]); c.update(side=side, entry=lab, liq=liq); rows.append(c)
    r = (sg * (g.close / g.open - 1) - bt.COST).where(fill)
    for liq, m in (("Q1-2", g.Q <= 2), ("Q3-5", g.Q >= 3)):
        c = stats.clustered_mean(r[m], g.date[m]); c.update(side=side, entry="strict limit at open", liq=liq); rows.append(c)
    ns = ~g.stale_1h
    r = sg * (g.close / g.h1_close - 1) - bt.COST
    for liq, m in (("Q1-2", g.Q <= 2), ("Q3-5", g.Q >= 3)):
        mm = m & ns
        c = stats.clustered_mean(r[mm], g.date[mm]); c.update(side=side, entry="first-hour close, price moved in 1st hour", liq=liq,
                                                             stale_share=float(g.stale_1h[m].mean())); rows.append(c)
R = pd.DataFrame(rows)
R.to_csv(DOCS_DIR / "europe_intraday_2025.csv", index=False)
print(R.pivot_table(index=["side", "entry"], columns="liq", values=["N", "mean", "t"], aggfunc="first", sort=False).to_string(float_format=lambda x: f"{x:.4f}"))

order = ["open print", "strict limit at open", "first-hour close (~+1h)", "first-hour close, price moved in 1st hour", "first-hour avg price"]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), dpi=140, sharey=True)
for ax, liq in zip(axes, ("Q1-2", "Q3-5")):
    for k, (side, col, lab) in enumerate((("down", "#2a78d6", "gap-down: buy, sell at close"), ("up", "#eb6834", "gap-up: short, cover at close"))):
        x = R[(R.side == side) & (R.liq == liq)].set_index("entry").reindex(order)
        pos = np.arange(len(order)) + (k - 0.5) * 0.38
        ax.bar(pos, x["mean"] * 100, 0.36, color=col, label=lab)
        ax.errorbar(pos, x["mean"] * 100, yerr=1.96 * x["se"] * 100, fmt="none", ecolor="#52514e", lw=0.8)
    n_dn = int(R[(R.side == "down") & (R.liq == liq)].N.max()); n_up = int(R[(R.side == "up") & (R.liq == liq)].N.max())
    ax.axhline(0, color="#52514e", lw=0.8); ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=20, ha="right", fontsize=8)
    ax.set_title(f"Europe {liq}: {n_dn} dips, {n_up} gap-ups", fontsize=10, loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel("mean per trade to close, % (net 0.10%, 95% CI by date)", fontsize=8); axes[0].legend(frameon=False, fontsize=8)
fig.suptitle("European small/mid caps (call-auction venues), 2025-01 → 2026-10: return by entry", fontsize=9.5, x=0.01, ha="left")
fig.tight_layout(); fig.savefig(DOCS_DIR / "europe_intraday_2025.png"); print("chart written")
