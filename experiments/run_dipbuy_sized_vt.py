"""SPY core (optionally vol-targeted) + Q1-2 common-stock dips sized relative to the expected opening auction.

Daily, in $: equity E. Exposure L = 1 (plain) or min(1.5, 15% / SPY 20d realised vol, lagged). Each dip gets
w_i = min(10%, 10% x expected auction $ / E). Return = L x SPY_ret + sum w_i x (r_i - SPY_intraday).
Impact variant: entry price worse by sd60 x sqrt(participation), participation = w_i E / expected auction.
See docs/dipbuy/DESIGN.md addendum 1.
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

CAP0 = float(sys.argv[1]) if len(sys.argv) > 1 else 100_000.0
spy = pd.read_parquet(RAW_DIR / "spy.parquet")
D = pd.read_parquet(RAW_DIR / "dip_trades.parquet")
D = D[(D.Q <= 2) & D.equity_ok].copy()
share = pd.read_csv(DOCS_DIR / "auction_share_of_venue_volume.csv", index_col=0).iloc[:, 0].to_dict()
D["exp_auction"] = D.prim.map(share) * D.pvm60 * D.prev_close
D = D[D.exp_auction > 0]

rv = spy.ret.rolling(20).std() * np.sqrt(252)
L_vt = (0.15 / rv.shift(1)).clip(upper=1.5).fillna(1.0)


def run(days, model, vt, impact, cap0=CAP0, part=0.10, per_name=0.10):
    T = D[D.date.isin(days)].dropna(subset=[model])
    by_day = {d: g for d, g in T.groupby("date")}
    E = cap0; out = []; tickets = []
    for d in days:
        L = L_vt.get(d, 1.0) if vt else 1.0
        r = L * spy.ret.get(d, 0.0)
        g = by_day.get(d)
        if g is not None and len(g):
            w = np.minimum(per_name, part * g.exp_auction.values / E)
            if w.sum() > 1:
                w = w / w.sum()
            ri = g[model].values
            if impact:
                pi = (w * E) / g.exp_auction.values
                ri = (1 + ri) / (1 + g.sd60.values * np.sqrt(pi)) - 1
            r += float(np.sum(w * (ri - spy.intra.get(d, 0.0))))
            tickets.append(pd.DataFrame({"date": d, "usd": w * E, "w": w, "q": g.Q.values}))
        E *= 1 + r; out.append(r)
    tk = pd.concat(tickets) if tickets else pd.DataFrame(columns=["date", "usd", "w", "q"])
    return pd.Series(out, index=days), tk


variants = [("SPY", None, False, False), ("SPY vol-targeted", None, True, False),
            ("auction open, auction-sized", "E1_auction", False, False),
            ("auction open, auction-sized + VT", "E1_auction", True, False),
            ("auction open, auction-sized + VT + impact", "E1_auction", True, True),
            ("strict limit, auction-sized + VT", "E2_strict_limit", True, False),
            ("auction +1% slip, auction-sized + VT", "E5_slip_1.0%", True, False)]
pal = {"SPY": "#8a8984", "SPY vol-targeted": "#52514e", "auction open, auction-sized": "#2a78d6",
       "auction open, auction-sized + VT": "#1baf7a", "auction open, auction-sized + VT + impact": "#6250d6",
       "strict limit, auction-sized + VT": "#eb6834", "auction +1% slip, auction-sized + VT": "#e34948"}
rows = []
fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), dpi=140)
for ax, (lab_p, a, b) in zip(axes, (("2015-2026", "2015-01-01", "2026-10-30"), ("2025-2026", "2025-01-01", "2026-10-30"))):
    days = spy.loc[a:b].index
    ends = []
    for name, model, vt, imp in variants:
        if model is None:
            r = (L_vt.reindex(days) if vt else 1.0) * spy.ret.reindex(days).fillna(0); tk = None
        else:
            r, tk = run(days, model, vt, imp)
        p = stats.perf(r, spy.ret.reindex(days))
        p.update(period=lab_p, strategy=name)
        if tk is not None and len(tk):
            p["trades"] = len(tk); p["median ticket $"] = tk.usd.median(); p["median Q1 ticket $"] = tk[tk.q == 1].usd.median()
            p["mean dip gross on signal days"] = tk.groupby("date").w.sum().mean()
        rows.append(p)
        cc = (1 + r).cumprod()
        ax.plot(cc.index, cc.values, color=pal[name], lw=2.2 if name.startswith("SPY") else 1.8,
                ls="--" if name.startswith("SPY") else "-", label=name)
        ends.append((cc.iloc[-1], name))
    ends.sort(); last = None
    for v, n in ends:
        y = v if last is None else max(v, last * (1.035 if lab_p == "2015-2026" else 1.012)); last = y
        ax.annotate(f"{v:.2f}x", (days[-1], y), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8)
    ax.set_yscale("log"); ax.axhline(1, color="#c3c2b7", lw=0.8)
    if lab_p == "2015-2026":
        ax.axvline(pd.Timestamp("2022-01-01"), color="#52514e", lw=1, ls=":")
    ax.set_title(f"{lab_p}: ${CAP0/1e3:,.0f}k account, Q1–Q2 common-stock dips, ticket ≤ 10% of expected auction", fontsize=8.5, loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left"); ax.set_ylabel("growth of $1 (log)", fontsize=8)
    ax.set_xlim(days[0], days[-1] + (days[-1] - days[0]) * 0.08)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(DOCS_DIR / f"sized_vt_{int(CAP0/1e3)}k.png")
R = pd.DataFrame(rows).set_index(["period", "strategy"])
R.to_csv(DOCS_DIR / f"sized_vt_{int(CAP0/1e3)}k.csv")
print(R.drop(columns=["x$1"]).to_string(float_format=lambda x: f"{x:,.3f}"))
