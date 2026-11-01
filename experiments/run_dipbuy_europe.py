"""European small/mid caps (Yahoo daily, call-auction venues): same frozen signal and execution models.

Survivorship-biased (live tickers only) and without auction flags. Read as a secondary check.
"""
import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from posterioralpha.dipbuy import backtest as bt, europe, stats
from posterioralpha.dipbuy.paths import DOCS_DIR, RAW_DIR

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
raw = pd.read_parquet(RAW_DIR / "europe_raw.parquet")
U = pd.read_parquet(RAW_DIR / "europe_universe.parquet")
idx = pd.read_parquet(RAW_DIR / "europe_index.parquet")
idx = idx[idx.volume > 0] if "volume" in idx else idx
P = europe.build(raw, idx, U)
P["period"] = bt.period(P.date)
thr = np.maximum(bt.THR_FLOOR, bt.THR_SD * P.sd60)
el = (P.year >= 2015) & P.Q.notna() & (P.prev_close >= 5) & P.sd60.notna() & (P.gap.abs() < 0.5) & ~P.open_bad
print(f"eligible stock-days {el.sum():,}, symbols {P[el].sym.nunique():,}")
qd = P[el].groupby("Q").prior_dvol.median() / 1e6
print("median prior-year local-currency $vol (M/day) by Q:", qd.round(3).to_dict())
tol = lambda a, b: np.isclose(a, b, rtol=1e-6, atol=1e-4)
O = P[el & (P.agap.abs() < thr)]
o1 = O.groupby("Q").apply(lambda d: pd.Series({"open == low": tol(d.open, d.low).mean(), "open == high": tol(d.open, d.high).mean(),
                                                 "zero-return days": (d.ret == 0).mean()}))
print(o1); o1.to_csv(DOCS_DIR / "europe_suspectA_ordinary_days.csv")

rows = []
for side in ("down", "up"):
    S = P[el & ((P.agap <= -thr) if side == "down" else (P.agap >= thr))].copy()
    sgn = 1 if side == "down" else -1
    S["E1_auction"] = sgn * (S.close / S.open - 1) - bt.COST
    fill = (S.low < S.open * (1 - 1e-6)) if side == "down" else (S.high > S.open * (1 + 1e-6))
    S["E2_strict_limit"] = S.E1_auction.where(fill)
    S["E5_slip_1.0%"] = sgn * (S.close / (S.open * (1 + sgn * 0.01)) - 1) - bt.COST
    for m in ("E1_auction", "E2_strict_limit", "E5_slip_1.0%"):
        for (per, q), g in S.groupby(["period", "Q"]):
            r = stats.clustered_mean(g[m], g.date); r.update(side=side, model=m, period=per, Q=int(q)); rows.append(r)
        for per, g in S.groupby("period"):
            gg = g[g.Q <= 2]; r = stats.clustered_mean(gg[m], gg.date); r.update(side=side, model=m, period=per, Q="Q1-2"); rows.append(r)
    if side == "down":
        Dn = S
R = pd.DataFrame(rows)
R.to_csv(DOCS_DIR / "europe_per_trade.csv", index=False)
print(R.pivot_table(index=["side", "model", "Q"], columns="period", values=["mean", "t", "N"], aggfunc="first").to_string(float_format=lambda x: f"{x:.4f}"))
ex = stats.table(Dn.assign(liq=np.where(Dn.Q <= 2, "Q1-2", "Q3-5")), "E1_auction", ["exchange", "liq"])[["N", "mean", "t"]]
print(ex); ex.to_csv(DOCS_DIR / "europe_by_exchange.csv")
ig = stats.table(Dn.assign(ix=np.where(Dn.idx_gap < -0.005, "index gap < -0.5%", "flat/up"), liq=np.where(Dn.Q <= 2, "Q1-2", "Q3-5")),
                 "E1_auction", ["ix", "liq"])[["N", "mean", "t"]]
print(ig); ig.to_csv(DOCS_DIR / "europe_by_index_gap.csv")
yr = stats.table(Dn[Dn.Q <= 2], "E1_auction", "year")[["N", "mean", "t"]]
print(yr); yr.to_csv(DOCS_DIR / "europe_by_year.csv")

# core + overlay on the STOXX 600 ETF
core = pd.DataFrame({"ret": idx.adjclose.pct_change(), "intra": idx.close / idx.open - 1})
days = core.loc["2015-01-01":"2026-10-30"].index
res = {"STOXX 600 ETF (EXSA)": stats.perf(core.ret.reindex(days).fillna(0))}
curves = {}
for m in ("E1_auction", "E2_strict_limit", "E5_slip_1.0%"):
    T = Dn[Dn.Q <= 2]
    ov = bt.overlay(T, m, core, days)
    for per, sl in (("full", slice("2015", "2026")), ("OOS 2022-26", slice("2022", "2026"))):
        res[f"Q1-2 {m} overlay | {per}"] = stats.perf(ov.loc[sl], core.ret.reindex(days).loc[sl])
    curves[m] = ov
PT = pd.DataFrame(res).T
print(PT.to_string(float_format=lambda x: f"{x:.3f}")); PT.to_csv(DOCS_DIR / "europe_portfolios.csv")
pd.DataFrame({k: (1 + v).cumprod() for k, v in curves.items()} | {"EXSA": (1 + core.ret.reindex(days).fillna(0)).cumprod()}).to_parquet(RAW_DIR / "europe_curves.parquet")
