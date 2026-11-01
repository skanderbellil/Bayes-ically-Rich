"""E6 (suspects A and D, time-stamped): Yahoo intraday bars, recent window, live tickers.

Same official-auction signals; exits at the official close; entries at:
  official open (E1, same subset) | Yahoo first regular trade | 09:35 | 09:45 | 5/15/30-min VWAP | ~10:30 | first-hour VWAP proxy
Pre-market (tradable) signal: gap from the last pre-market print before 09:30 (stock and SPY), entry at the
official open via a market-on-open order (E1) - precision/recall against the auction-based signal.
Window: 60m bars 2023-12 -> 2026-10 (inside the OOS period); 5m bars ~ last 60 sessions.
"""
import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from posterioralpha.dipbuy import backtest as bt, stats
from posterioralpha.dipbuy.paths import DOCS_DIR, RAW_DIR

import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
START = sys.argv[1] if len(sys.argv) > 1 else "2023-12-01"   # e.g. 2025-01-01
P = pd.read_parquet(RAW_DIR / "panel.parquet")
H = pd.read_parquet(RAW_DIR / "intraday_60m.parquet")
_f5 = RAW_DIR / "intraday_5m.parquet"
F = pd.read_parquet(_f5) if _f5.exists() else pd.DataFrame(columns=["sym", "date", "pre5_last", "y5_open", "vwap5", "vwap15", "vwap30", "px0935", "px0945"])
F["date"] = pd.to_datetime(F.date)
spyH = H[H.sym == "SPY"].set_index("date")
spyF = F[F.sym == "SPY"].set_index("date")
P = P[P.date >= max(pd.Timestamp(START), H.date.min())].merge(H[H.sym != "SPY"], on=["sym", "date"], how="inner")
P = P.merge(F[F.sym != "SPY"], on=["sym", "date"], how="left")
# guard against ticker reuse / bad bars: Yahoo first trade within 15% of the official open
P = P[(P.y_open / P.open - 1).abs() < 0.15]
spy = pd.read_parquet(RAW_DIR / "spy.parquet")
P["spy_prev_close"] = P.date.map(spy.close.shift(1))
P["spy_pre_gap"] = P.date.map(spyH.pre_last) / P.spy_prev_close - 1
thr = np.maximum(bt.THR_FLOOR, bt.THR_SD * P.sd60)
el = bt.eligible(P)
out = []

print("Yahoo first regular trade == official auction open:", np.isclose(P.y_open, P.open, rtol=1e-6, atol=1e-4).groupby(P.Q).mean().round(3).to_dict())
eqo = np.isclose(P.y_open, P.open, rtol=1e-6, atol=1e-4).groupby(P.Q).mean().rename("yahoo_open_equals_auction")
eqo.to_csv(DOCS_DIR / "yahoo_open_vs_auction.csv")

entries = {"official open (E1)": "open", "Yahoo first trade": "y_open", "09:35": "px0935", "09:45": "px0945",
           "VWAP 5m": "vwap5", "VWAP 15m": "vwap15", "VWAP 30m": "vwap30", "~10:30": "h1_close", "1st-hour VWAP proxy": "h1_typ"}
for side in ("down", "up"):
    S = P[el & ((P.agap <= -thr) if side == "down" else (P.agap >= thr))]
    sgn = 1 if side == "down" else -1
    for win, sub in ((f"60m window {START[:7]}..2026-10", S), ("5m window (last ~60 sessions)", S[S.vwap5.notna()])):
        for lab, col in entries.items():
            if sub[col].isna().all():
                continue
            r = sgn * (sub.close / sub[col] - 1) - bt.COST
            for liq, m in (("Q1-2", sub.Q <= 2), ("Q3-5", sub.Q >= 3)):
                c = stats.clustered_mean(r[m], sub.date[m]); c.update(side=side, window=win, entry=lab, liq=liq); out.append(c)
R = pd.DataFrame(out)
R.to_csv(DOCS_DIR / "intraday_entries.csv", index=False)
print(R.pivot_table(index=["side", "window", "entry"], columns="liq", values=["N", "mean", "t"], aggfunc="first", sort=False)
      .to_string(float_format=lambda x: f"{x:.4f}"))

# pre-market signal (tradable before the auction)
P["agap_pre"] = P.pre_last / P.prev_close - 1 - P.spy_pre_gap
real = el & (P.agap <= -thr)
pre = el & (P.agap_pre <= -thr) & P.pre_last.notna()
rows = {}
for liq, m in (("Q1-2", P.Q <= 2), ("Q3-5", P.Q >= 3), ("all", P.Q > 0)):
    rr = real & m; pp = pre & m
    e1 = P.intra - bt.COST
    rows[liq] = {"has pre-market print": P.pre_last.notna()[m & el].mean(), "auction signals": int(rr.sum()), "pre-market signals": int(pp.sum()),
                 "recall": (rr & pp).sum() / max(rr.sum(), 1), "precision": (rr & pp).sum() / max(pp.sum(), 1),
                 **{f"pre-mkt signal, MOO entry: {k}": v for k, v in stats.clustered_mean(e1[pp], P.date[pp]).items() if k in ("N", "mean", "t")},
                 **{f"auction signal (same window): {k}": v for k, v in stats.clustered_mean(e1[rr], P.date[rr]).items() if k in ("N", "mean", "t")},
                 **{f"auction signal NOT seen pre-market: {k}": v for k, v in stats.clustered_mean(e1[rr & ~pp], P.date[rr & ~pp]).items() if k in ("N", "mean", "t")}}
PR = pd.DataFrame(rows)
print(PR.to_string(float_format=lambda x: f"{x:.4f}")); PR.to_csv(DOCS_DIR / "premarket_signal.csv")

# ---------------------------------------------------------------- chart
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), dpi=140, sharey=True)
order = ["official open (E1)", "Yahoo first trade", "09:35", "09:45", "VWAP 5m", "VWAP 15m", "VWAP 30m", "~10:30", "1st-hour VWAP proxy"]
for ax, liq in zip(axes, ("Q1-2", "Q3-5")):
    for k, (side, col, lab) in enumerate((("down", "#2a78d6", "gap-down: buy, sell at close"), ("up", "#eb6834", "gap-up: short, cover at close"))):
        x = R[(R.side == side) & (R.liq == liq)].copy()
        x["key"] = np.where(x.window.str.startswith("5m") & ~x.entry.str.startswith("VWAP") & ~x.entry.isin(["09:35", "09:45"]), None, x.entry)
        x = x[x.key.notna()]
        x = x[(x.window.str.startswith("60m") & ~x.entry.isin(["09:35", "09:45", "VWAP 5m", "VWAP 15m", "VWAP 30m"])) |
              (x.window.str.startswith("5m") & x.entry.isin(["09:35", "09:45", "VWAP 5m", "VWAP 15m", "VWAP 30m"]))]
        x = x.set_index("entry").reindex(order)
        pos = np.arange(len(order)) + (k - 0.5) * 0.38
        ax.bar(pos, x["mean"] * 100, 0.36, color=col, label=lab)
        ax.errorbar(pos, x["mean"] * 100, yerr=1.96 * x["se"].astype(float) * 100, fmt="none", ecolor="#52514e", lw=0.8)
        for p_, n_ in zip(pos, x["N"]):
            if n_ == n_:
                ax.text(p_, ax.get_ylim()[0] if False else 0, f"{int(n_)}", ha="center", va="top", fontsize=6, color="#52514e", rotation=90)
    ax.axhline(0, color="#52514e", lw=0.8)
    ax.axvline(6.5, color="#c3c2b7", lw=0.8, ls=":"); ax.axvline(1.5, color="#c3c2b7", lw=0.8, ls=":")
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=35, ha="right", fontsize=8)
    ax.set_title(f"{liq}: net return to the official close by ENTRY time", fontsize=10, loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel("mean per trade, % (net 0.10%, 95% CI clustered by date)", fontsize=8)
axes[0].legend(frameon=False, fontsize=8)
fig.suptitle(f"US NYSE/NYSE American, official-auction gap signals, {START[:7]} → 2026-10 (09:35–VWAP30: last ~60 sessions only)", fontsize=9, x=0.01, ha="left")
fig.tight_layout(); fig.savefig(DOCS_DIR / f"intraday_entries_{START[:4]}.png"); print("chart written")
