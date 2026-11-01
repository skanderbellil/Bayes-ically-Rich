"""Portfolios for the dip buy (run after run_dipbuy_backtest.py and run_dipbuy_microstructure.py).

* SPY + overlay: 100% SPY; on signal days sell up to 20% of SPY at the open (<= 2% per name), buy the dips,
  sell at the close, rebuy SPY at the close. Unfilled limit orders leave the slice in SPY.
* Static 80/20: 80% SPY, 20% stand-alone dip sleeve (1/max(N,10) per name, cash otherwise), rebalanced daily.
* Fixed-$ tickets: $100k account in SPY, tickets of $2k/$5k/$20k per signal (max 10 names/day), each capped at
  10% of the stock's *estimated opening-auction size* (= listing-venue auction share of venue volume, measured
  on XDP sample days, x prior-60-day median listing-venue volume x price). P&L is not compounded into tickets.
"""
import _bootstrap  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from posterioralpha.dipbuy import backtest as bt, stats, yahoo
from posterioralpha.dipbuy.paths import DOCS_DIR, RAW_DIR, RESULTS_DIR

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
D = pd.read_parquet(RAW_DIR / "dip_trades.parquet")
spy = pd.read_parquet(RAW_DIR / "spy.parquet")
days = spy.loc["2015-01-01":"2026-10-30"].index
models = ["E1_auction", "E2_strict_limit", "E3_other_venue_entry", "E5_slip_1.0%"]
labels = {"E1_auction": "E1 auction→auction", "E2_strict_limit": "E2 strict limit", "E3_other_venue_entry": "E3 other-venue entry",
          "E5_slip_1.0%": "E5 auction +1% slippage"}
out_lines = []


def block(name, r):
    res = {}
    for per, sl in (("full 2015-26", slice("2015", "2026")), ("IS 2015-21", slice("2015", "2021")), ("OOS 2022-26", slice("2022", "2026"))):
        x = r.loc[sl]
        res[per] = stats.perf(x, spy.ret.loc[sl])
    return pd.DataFrame(res).T.assign(strategy=name)


tabs, curves = [], {"SPY": spy.ret.reindex(days).fillna(0)}
tabs.append(block("SPY buy & hold", curves["SPY"]))
for univ, T in (("Q1-2", D[D.Q <= 2]), ("all Q", D)):
    for m in models:
        ov = bt.overlay(T, m, spy, days)
        sl = bt.sleeve(T, m, days)
        s8020 = 0.8 * curves["SPY"] + 0.2 * sl
        tabs.append(block(f"{univ} | {labels[m]} | SPY + overlay", ov))
        tabs.append(block(f"{univ} | {labels[m]} | 80/20", s8020))
        if univ == "Q1-2":
            curves[labels[m]] = ov
PT = pd.concat(tabs).reset_index().rename(columns={"index": "period"}).set_index(["strategy", "period"])
print(PT.to_string(float_format=lambda x: f"{x:.3f}")); out_lines.append(PT.to_string(float_format=lambda x: f"{x:.3f}"))
PT.to_csv(DOCS_DIR / "portfolios.csv")

# activity / turnover of the Q1-2 overlay
T = D[D.Q <= 2]
n = T.groupby("date").size().reindex(days).fillna(0)
w = (0.20 / n.replace(0, np.nan)).clip(upper=0.02).fillna(0) * n
act = pd.Series({"share of days with >=1 signal": (n > 0).mean(), "trades per active day (mean)": n[n > 0].mean(),
                 "trades per active day (max)": n.max(), "mean overlay gross on active days": w[n > 0].mean(),
                 "annual one-way turnover of the overlay (x capital, 2 legs/day)": (2 * w).sum() / (len(days) / 252),
                 "E2 fill rate": T.E2_strict_limit.notna().mean()})
print(act); out_lines.append(act.to_string()); act.to_csv(DOCS_DIR / "overlay_activity.csv")

# fixed-$ tickets with auction participation cap
sh = pd.read_csv(DOCS_DIR / "auction_share_of_venue_volume.csv", index_col=0).iloc[:, 0].to_dict()
D["est_auction_usd"] = D.prim.map(sh) * D.pvm60 * D.prev_close
D["cap_usd"] = (0.10 * D.est_auction_usd).fillna(0)
eurusd = yahoo.daily("EURUSD=X", "2014-06-01").close.reindex(days).ffill().bfill()
fx = []
for univ, TT in (("Q1-2", D[D.Q <= 2]), ("all Q", D)):
    for m in ("E1_auction", "E2_strict_limit", "E3_other_venue_entry"):
        for tk in (2000, 5000, 20000):
            for capped in (True, False):
                f = bt.fixed_ticket_pnl(TT.assign(cap_usd=TT.cap_usd if capped else 1e12), m, tk).reindex(days).fillna(0)
                eq = [100_000.0]; rr = []
                for dte, row in f.iterrows():
                    r = curves["SPY"].loc[dte] + row.pnl / eq[-1]; rr.append(r); eq.append(eq[-1] * (1 + r))
                r = pd.Series(rr, index=days)
                p = stats.perf(r, curves["SPY"])
                pnl_eur = (f.pnl / eurusd)
                fx.append({"universe": univ, "model": m, "ticket $": tk, "auction cap 10%": capped,
                           "CAGR": p["CAGR"], "Sharpe": p["Sharpe"], "maxDD": p["maxDD"],
                           "overlay P&L $/yr": f.pnl.sum() / (len(days) / 252), "overlay P&L EUR/yr": pnl_eur.sum() / (len(days) / 252),
                           "OOS overlay P&L $/yr": f.pnl.loc["2022":].sum() / (len(days[days >= "2022"]) / 252),
                           "mean $ deployed / active day": f.deployed[f.n > 0].mean(),
                           "share of tickets cut by cap": (f.capped * f.n).sum() / max(f.n.sum(), 1)})
FX = pd.DataFrame(fx)
print(FX.to_string(float_format=lambda x: f"{x:,.3f}")); out_lines.append(FX.to_string(float_format=lambda x: f"{x:,.3f}"))
FX.to_csv(DOCS_DIR / "fixed_ticket.csv", index=False)
# capacity: $ that could be deployed per signal at 10% of the estimated auction
capq = D.groupby(["Q", "prim"]).cap_usd.median().unstack()
print("median per-trade capacity at 10% of est. opening auction ($):\n", capq.round(0)); capq.to_csv(DOCS_DIR / "capacity_per_trade.csv")

# ---------------------------------------------------------------- growth-of-$1 chart (log)
pal = {"SPY": "#8a8984", "E1 auction→auction": "#2a78d6", "E2 strict limit": "#eb6834", "E3 other-venue entry": "#1baf7a",
       "E5 auction +1% slippage": "#6250d6"}
fig, ax = plt.subplots(figsize=(10, 5.6), dpi=150)
for k, r in curves.items():
    c = (1 + r).cumprod()
    ax.plot(c.index, c.values, lw=2 if k != "SPY" else 2.4, color=pal[k], label=k, ls="-" if k != "SPY" else "--")
    ax.annotate(f"{k}  {c.iloc[-1]:.2f}x", (c.index[-1], c.iloc[-1]), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=8, color="#0b0b0b")
ax.set_yscale("log"); ax.axvline(pd.Timestamp("2022-01-01"), color="#52514e", lw=1, ls=":")
ax.text(pd.Timestamp("2022-02-01"), ax.get_ylim()[1] * 0.92, "out-of-sample →", fontsize=8, color="#52514e")
ax.set_title("Growth of $1: SPY + same-day dip overlay (Q1–Q2, ≤20% gross, ≤2%/name), official auction prints, NYSE/NYSE American",
             fontsize=9, loc="left")
ax.set_ylabel("growth of $1 (log scale)", fontsize=9)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color="#e6e5e0", lw=0.8); ax.tick_params(labelsize=8)
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.set_xlim(days[0], days[-1] + pd.Timedelta(days=700))
fig.tight_layout(); fig.savefig(DOCS_DIR / "growth_overlay.png"); print("chart written")
(RESULTS_DIR / "portfolio_output.txt").write_text("\n\n".join(out_lines))
