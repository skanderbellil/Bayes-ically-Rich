"""Main dip-buy re-test on official auction prints (NYSE / NYSE American, 2015-2026, delisted included).

Sections (all variants pre-listed in docs/dipbuy/DESIGN.md):
  0. sample description
  1. suspect A on ordinary days: how often is the official open the consolidated high/low?
  2. per-trade returns by execution model x liquidity quintile x IS/OOS, gap-down and gap-up mirror
  3. E4 cross-measurement (signal on other venues' first prints, entry at the official open)
  4. cross-sections
  5. portfolios: SPY + overlay, 80/20, fixed-$ tickets with an auction participation cap
Writes tables to docs/dipbuy/*.csv, the chart to docs/dipbuy/growth_overlay.png and a text log
to results/dipbuy/backtest_output.txt.
"""
import io
import sys
from contextlib import redirect_stdout

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from posterioralpha.dipbuy import backtest as bt
from posterioralpha.dipbuy import stats
from posterioralpha.dipbuy.paths import DOCS_DIR, RAW_DIR, RESULTS_DIR, ensure_dirs

ensure_dirs()
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
buf = io.StringIO()


def show(title, df, fname=None, fmt="{:.4f}"):
    s = f"\n=== {title} ===\n" + df.to_string(float_format=lambda x: fmt.format(x))
    print(s); buf.write(s + "\n")
    if fname:
        df.to_csv(DOCS_DIR / fname)


P = pd.read_parquet(RAW_DIR / "panel.parquet")
spy = pd.read_parquet(RAW_DIR / "spy.parquet")
ev_path = RAW_DIR / "yahoo_events.parquet"
EV = pd.read_parquet(ev_path) if ev_path.exists() else None
P["period"] = bt.period(P.date)
el = bt.eligible(P)

# ---------------------------------------------------------------- 0. sample
E = P[el]
last = P.groupby("sym").date.max()
s0 = pd.DataFrame({
    "stock-days": E.groupby("period").size(),
    "symbols": E.groupby("period").sym.nunique(),
    "symbols later delisted/renamed": E.groupby("period").sym.apply(lambda s: (s.drop_duplicates().map(last) < "2026-10-01").sum()),
    "NYSE American share": E.groupby("period").prim.apply(lambda s: (s == "AMEX").mean()),
}).T
show("0. Eligible sample", s0, "sample.csv", "{:,.3f}")
qd = E.groupby("Q").agg(med_prior_dvol_musd=("prior_dvol", lambda x: x.median() / 1e6), med_price=("close", "median"),
                        amex_share=("prim", lambda s: (s == "AMEX").mean()))
show("0b. Liquidity quintiles (prior-year median $ volume, $M/day)", qd, "quintiles.csv", "{:,.3f}")

# ---------------------------------------------------------------- 1. suspect A, ordinary days
thr = np.maximum(bt.THR_FLOOR, bt.THR_SD * P.sd60)
ordinary = el & (P.agap.abs() < thr)
O = P[ordinary]
tol = lambda a, b: np.isclose(a, b, rtol=1e-6, atol=1e-4)
o1 = pd.DataFrame({
    "open == consolidated low": O.groupby("Q").apply(lambda d: tol(d.open, d.cons_lo).mean()),
    "open == consolidated high": O.groupby("Q").apply(lambda d: tol(d.open, d.cons_hi).mean()),
    "open == cons. high or low": O.groupby("Q").apply(lambda d: (tol(d.open, d.cons_lo) | tol(d.open, d.cons_hi)).mean()),
    "open == listing-venue low or high": O.groupby("Q").apply(lambda d: (tol(d.open, d.p_lo) | tol(d.open, d.p_hi)).mean()),
    "zero-return days": O.groupby("Q").apply(lambda d: (d.ret == 0).mean()),
    "median |open/other-venue first print - 1|": O.groupby("Q").apply(lambda d: (d.open / d.alt_open_med - 1).abs().median()),
})
show("1. Ordinary days: is the official open an extreme of the day? (first pass, Kaggle open: Q1 61%, Q5 8%)", o1, "suspectA_ordinary_days.csv")

# ---------------------------------------------------------------- 2. per-trade returns
D = P[bt.signals(P, "down")].copy(); U = P[bt.signals(P, "up")].copy()
RD = bt.trade_returns(D, "down"); RU = bt.trade_returns(U, "up")
D = D.join(RD); U = U.join(RU)
if EV is not None:
    D = D.merge(EV, on=["sym", "date"], how="left"); U = U.merge(EV, on=["sym", "date"], how="left")
models = list(RD.columns)
rows = []
for side, T in (("gap-down LONG", D), ("gap-up SHORT", U)):
    for m in models:
        for (per, q), g in T.groupby(["period", "Q"]):
            r = stats.clustered_mean(g[m], g.date); r.update(side=side, model=m, period=per, Q=int(q)); rows.append(r)
        for per, g in T.groupby("period"):
            for lab, gg in (("Q1-2", g[g.Q <= 2]), ("all", g)):
                r = stats.clustered_mean(gg[m], gg.date); r.update(side=side, model=m, period=per, Q=lab); rows.append(r)
R = pd.DataFrame(rows)
R["fill_rate"] = np.nan
for side, T in (("gap-down LONG", D), ("gap-up SHORT", U)):
    fr = T.groupby(["period", "Q"]).E2_strict_limit.apply(lambda s: s.notna().mean())
    for (per, q), v in fr.items():
        R.loc[(R.side == side) & (R.model == "E2_strict_limit") & (R.period == per) & (R.Q == int(q)), "fill_rate"] = v
R.to_csv(DOCS_DIR / "per_trade_by_model.csv", index=False)
for side in ("gap-down LONG", "gap-up SHORT"):
    t = R[R.side == side].pivot_table(index=["model", "Q"], columns="period", values=["mean", "t", "N"], aggfunc="first")
    show(f"2. Per-trade net return (0.10% cost), {side}: mean / date-clustered t / N", t)

# where does E1 come from? split by whether the auction print was the day's consolidated low (high for gap-ups)
D["open_is_extreme"] = np.isclose(D.open, D.cons_lo, rtol=1e-6, atol=1e-4)
U["open_is_extreme"] = np.isclose(U.open, U.cons_hi, rtol=1e-6, atol=1e-4)
sp = pd.concat({side: stats.table(T.assign(liq=np.where(T.Q <= 2, "Q1-2", "Q3-5")), "E1_auction", ["period", "liq", "open_is_extreme"])[["N", "mean", "t"]]
                for side, T in (("gap-down long", D), ("gap-up short", U))})
show("2b. E1 split by whether the auction print was the day's extreme (= not fillable by a resting limit)", sp, "e1_by_open_extreme.csv")

# ---------------------------------------------------------------- 3. cross-measurement
agap_alt = P.alt_open_med / P.prev_close - 1 - P.spy_gap
X = P[(agap_alt <= -thr) & el].copy()
X["E4_official_entry"] = X.close / X.open - 1 - bt.COST
X["E4b_alt_entry"] = X.close / X.alt_open_med - 1 - bt.COST
x3 = pd.concat({k: stats.table(X, k, ["period", "Q"]) [["N", "mean", "t"]] for k in ("E4_official_entry", "E4b_alt_entry")}, axis=1)
show("3. Signal from OTHER venues' first prints (untimed: may be pre-market or after the open), entry at official open vs at those prints", x3, "cross_measurement.csv")
both = D.assign(alt_sig=(D.alt_open_med / D.prev_close - 1 - D.spy_gap) <= -np.maximum(bt.THR_FLOOR, bt.THR_SD * D.sd60))
x3b = stats.table(both, "E1_auction", ["alt_sig", "Q"])[["N", "mean", "t"]]
show("3b. Auction-signal trades split by whether other venues ALSO showed the gap", x3b, "cross_measurement_confirm.csv")

# ---------------------------------------------------------------- 4. cross-sections (E1 and E2)
D["gap_bucket"] = pd.cut(-D.agap, [0, .05, .10, .20, 1], labels=["3-5%", "5-10%", "10-20%", ">20%"])
D["index_gap"] = np.where(D.spy_gap < -0.005, "SPY gap < -0.5%", "SPY flat/up")
D["price"] = pd.cut(D.prev_close, [5, 10, 20, 1e9], labels=["$5-10", "$10-20", ">=$20"], right=False)
D["venue"] = D.prim
D["liq"] = np.where(D.Q <= 2, "Q1-2", "Q3-5")
cuts = ["venue", "gap_bucket", "index_gap", "price", "year"]
if EV is not None:
    D["news"] = np.select([D.yahoo_match != True, D.earnings_day == True], ["unmatched (delisted/renamed)", "earnings day"], "no earnings")  # noqa: E712
    D["coverage"] = np.select([D.yahoo_match != True, D.eps_covered == True, D.eps_covered == False],  # noqa: E712
                              ["unmatched", "EPS estimate (>=1 analyst)", "no EPS estimate"], "unknown")
    cuts += ["news", "coverage"]
for c in cuts:
    t = pd.concat({m: stats.table(D, m, [c, "liq"])[["N", "mean", "t"]] for m in ("E1_auction", "E2_strict_limit")}, axis=1)
    show(f"4. Cross-section by {c} (gap-down long, full sample 2015-2026)", t, f"xsec_{c}.csv")
    if c != "year":
        t2 = stats.table(D[D.period == "OOS 2022-26"], "E1_auction", [c, "liq"])[["N", "mean", "t"]]
        show(f"4. OOS only, by {c} (E1)", t2, f"xsec_{c}_oos.csv")

D.to_parquet(RAW_DIR / "dip_trades.parquet", index=False); U.to_parquet(RAW_DIR / "gapup_trades.parquet", index=False)
(RESULTS_DIR / "backtest_output.txt").write_text(buf.getvalue())
print("\nwrote", RESULTS_DIR / "backtest_output.txt")
