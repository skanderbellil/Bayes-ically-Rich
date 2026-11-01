"""Closed-end funds (XDP security type 'U'): does the gap-down bounce survive realistic fills?

Same frozen signal (agap <= -max(3%, 3 sd60), prev close >= $5) on CEFs instead of common stocks.
Q buckets are the common-stock universe's prior-year $-volume quintiles (CEFs inherit them via the panel).
Part 1: per-trade by fill model, IS/OOS, gap-down vs gap-up. Part 2: expected opening-auction size.
Part 3 (if Yahoo NAV tickers exist: X<ticker>X): discount at the open vs prior-day NAV.
"""
import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from posterioralpha.dipbuy import backtest as bt, stats
from posterioralpha.dipbuy.paths import DOCS_DIR, RAW_DIR

pd.set_option("display.width", 220)
P = pd.read_parquet(RAW_DIR / "panel.parquet")
P["period"] = bt.period(P.date)
P["is_cef"] = P.stypes.fillna("").str.contains("U") & ~P.stypes.fillna("").str.contains("C")
P2 = P.copy(); P2["equity_ok"] = True                       # bypass the common-stock filter
el = bt.eligible(P2) & P2.is_cef
thr = np.maximum(bt.THR_FLOOR, bt.THR_SD * P2.sd60)
print("CEF symbols in panel:", P2[P2.is_cef].sym.nunique(), " eligible CEF stock-days:", int(el.sum()))
out = []
for side in ("down", "up"):
    S = P2[el & ((P2.agap <= -thr) if side == "down" else (P2.agap >= thr))]
    R = bt.trade_returns(S, side).join(S[["date", "period", "Q"]])
    for m in ("E1_auction", "E2_strict_limit", "E3_other_venue_entry", "E5_slip_1.0%"):
        for per, g in R.groupby("period"):
            r = stats.clustered_mean(g[m], g.date); r.update(side=side, model=m, period=per); out.append(r)
        r = stats.clustered_mean(R[m], R.date); r.update(side=side, model=m, period="all"); out.append(r)
    if side == "down":
        DN = S.join(R[["E1_auction", "E2_strict_limit"]])
T = pd.DataFrame(out)
print(T.pivot_table(index=["side", "model"], columns="period", values=["N", "mean", "t"], aggfunc="first").to_string(float_format=lambda x: f"{x:.4f}"))
T.to_csv(DOCS_DIR / "cef_per_trade.csv", index=False)
share = pd.read_csv(DOCS_DIR / "auction_share_of_venue_volume.csv", index_col=0).iloc[:, 0].to_dict()
DN["exp_auction_usd"] = DN.prim.map(share) * DN.pvm60 * DN.prev_close
print("\nCEF gap-down trades: expected opening auction $ quantiles:", DN.exp_auction_usd.quantile([.25, .5, .75]).round(0).to_dict())
print("median prior-year $vol/day:", round(DN.prior_dvol.median()), " trades/yr:", round(len(DN) / 11.8, 1))
print("open == consolidated low share:", np.isclose(DN.open, DN.cons_lo, rtol=1e-6, atol=1e-4).mean().round(3))
DN.to_parquet(RAW_DIR / "cef_dips.parquet", index=False)

# ---------------------------------------------------------------- Part 3: NAV / discount
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from posterioralpha.dipbuy import yahoo  # noqa: E402

syms = sorted(DN.sym.unique())
with ThreadPoolExecutor(4) as ex:
    navs = dict(zip(syms, ex.map(lambda s: yahoo.daily(f"X{s}X", "2014-01-01"), syms)))
rows = []
for s, nav in navs.items():
    if nav.empty:
        continue
    n = nav.close
    g = P2[P2.sym == s].set_index("date")[["prev_close", "close"]]
    disc = (g.close / n.reindex(g.index) - 1)
    prior_disc = disc.shift(1)
    disc_avg60 = disc.shift(1).rolling(60, min_periods=40).mean()
    nav_prev = n.reindex(g.index).shift(1)
    for i, t in DN[DN.sym == s].iterrows():
        d = t.date
        if d not in g.index or pd.isna(nav_prev.get(d)) or pd.isna(n.get(d)):
            continue
        rows.append({"sym": s, "date": d, "period": t.period, "E1": t.E1_auction, "E2": t.E2_strict_limit,
                     "disc_prev_close": prior_disc.get(d), "disc_avg60": disc_avg60.get(d),
                     "disc_open_vs_prevNAV": t.open / nav_prev[d] - 1,          # known at the open
                     "disc_open_vs_NAV_t": t.open / n[d] - 1,                   # ex post
                     "disc_close": t.close / n[d] - 1, "nav_ret": n[d] / nav_prev[d] - 1})
N = pd.DataFrame(rows)
N["excess_open_disc"] = N.disc_open_vs_prevNAV - N.disc_avg60     # pre-open: how much wider than usual
print(f"\nNAV matched for {N.sym.nunique()} of {len(syms)} CEFs, {len(N)} of {len(DN)} dip trades")
print("mean discounts on dip days (prev close / open vs NAV_t / close):",
      N[["disc_prev_close", "disc_open_vs_NAV_t", "disc_close", "nav_ret"]].mean().round(4).to_dict())
N["xbin"] = pd.qcut(N.excess_open_disc, 3, labels=["least excess discount", "mid", "most excess discount"])
tab = pd.concat({m: stats.table(N, m, ["period", "xbin"])[["N", "mean", "t"]] for m in ("E1", "E2")}, axis=1)
print("\nCEF dips by excess discount at the open (pre-open info: open / prior NAV vs own 60d avg discount):")
print(tab.to_string(float_format=lambda x: f"{x:.4f}"))
tab.to_csv(DOCS_DIR / "cef_by_excess_discount.csv")
N.to_parquet(RAW_DIR / "cef_nav_trades.parquet", index=False)
