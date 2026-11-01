"""ETFs (NYSE Arca / Cboe / NYSE American listed), 2025-01 -> 2026-10: same frozen gap signal.

Prices: Yahoo daily open/close (consolidated OFFICIAL open/close; the Arca venue open in the CTA file is often an
early-session print for thin ETFs, and its last print can be a late-session trade). CTA files give the ETF universe
(incl. since-closed ETFs), consolidated $ volume for liquidity quintiles (prior calendar year, within ETFs) and the
consolidated regular-session low/high for the strict-limit fill. Ex-distribution days (Yahoo) are excluded.
Leveraged / inverse ETFs flagged by name (financedatabase). Entries: official open, strict limit, ~1h close,
first-hour average; the 1h-close entry is also shown only on days the price moved in the first hour.
"""
import re
from concurrent.futures import ThreadPoolExecutor

import _bootstrap  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from posterioralpha.dipbuy import backtest as bt, sectype, stats, yahoo
from posterioralpha.dipbuy.paths import DOCS_DIR, RAW_DIR, XDP_DIR

pd.set_option("display.width", 220)
files = sorted((RAW_DIR / "cts_days_etf").glob("*.parquet"))
C = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
C["sym"] = C.sym.astype(str)
ST = sectype.build(XDP_DIR)
C = C[C.sym.map(ST.stypes).fillna("").str.contains("E")]
C["dvol"] = C.cons_vol * C.cons_last
C["year"] = C.date.dt.year
ys = C.groupby(["sym", "year"]).agg(med_dvol=("dvol", "median"), med_px=("cons_last", "median"), n=("dvol", "size")).reset_index()
ys["year"] += 1
ys = ys[(ys.med_px >= 5) & (ys.n >= 200) & (ys.med_dvol > 0)]
ys["Q"] = ys.groupby("year").med_dvol.transform(lambda s: pd.qcut(s, 5, labels=False) + 1)
univ = ys[ys.year.isin([2025, 2026])]
last_seen = C.groupby("sym").date.max()
print(f"ETF universe 2025-26: {univ.sym.nunique()} ETFs; closed/delisted before 2026-10: {(last_seen.reindex(univ.sym.unique()) < '2026-10-01').sum()}")


def lev_flag():
    try:
        import financedatabase as fd
        e = fd.ETFs().select().reset_index()
        pat = re.compile(r"(\b[23]x\b|ultra|leveraged|inverse|bear\b|short\b|-1x|-2x|-3x|daily .* bull|2x|3x)", re.I)
        return set(e[e.name.fillna("").str.contains(pat)].symbol)
    except Exception:
        return set()


LEV = lev_flag()
syms = sorted(univ.sym.unique())


def ydaily(s):
    d = yahoo.daily(s.replace("/", "-"), "2024-09-01")
    if d.empty:
        return None
    divs = {pd.Timestamp(int(k), unit="s").tz_localize("UTC").tz_convert("America/New_York").normalize().tz_localize(None)
            for k in d.attrs.get("events", {}).get("dividends", {})}
    o = d[["open", "close"]].copy(); o["exdiv"] = o.index.isin(divs); o["sym"] = s
    return o.rename_axis("date").reset_index()


with ThreadPoolExecutor(4) as ex:
    Y = pd.concat([x for x in ex.map(ydaily, syms) if x is not None], ignore_index=True)
print(f"Yahoo daily found for {Y.sym.nunique()} of {len(syms)} ETFs")
spy = pd.read_parquet(RAW_DIR / "spy.parquet")
Y = Y.sort_values(["sym", "date"])
g = Y.groupby("sym")
Y["prev_close"] = g.close.shift(1)
Y["gap"] = Y.open / Y.prev_close - 1; Y["ret"] = Y.close / Y.prev_close - 1
Y = Y.join(spy[["gap", "ret", "intra"]].add_prefix("spy_"), on="date")
Y["agap"] = Y.gap - Y.spy_gap; Y["ar"] = Y.ret - Y.spy_ret
Y["sd60"] = Y.groupby("sym").ar.transform(lambda x: x.shift(1).rolling(60, min_periods=50).std())
Y["year"] = Y.date.dt.year
Y = Y.merge(univ[["sym", "year", "Q", "med_dvol"]], on=["sym", "year"], how="inner")
Y = Y.merge(C[["sym", "date", "cons_lo", "cons_hi"]], on=["sym", "date"], how="left")
Y["lev"] = Y.sym.isin(LEV)
thr = np.maximum(bt.THR_FLOOR, bt.THR_SD * Y.sd60)
el = (Y.date >= "2025-01-01") & (Y.prev_close >= 5) & Y.sd60.notna() & (Y.gap.abs() < 0.5) & ~Y.exdiv & Y.open.notna()
S = Y[el & (Y.agap.abs() >= thr)].copy()
S["side"] = np.where(S.agap < 0, "down", "up")
print(f"signals: {len(S)} ({(S.side=='down').sum()} down / {(S.side=='up').sum()} up); leveraged/inverse share {S.lev.mean():.2f}")


def first_hour(s):
    d = yahoo.chart(s.replace("/", "-"), "60m", rng="730d")
    if d.empty:
        return None
    d = d.dropna(subset=["close"]); d["date"] = d.index.tz_localize(None).normalize()
    f = d.groupby("date").first()
    return pd.DataFrame({"sym": s, "date": f.index, "h_open": f.open.values, "h1_close": f.close.values,
                         "h1_typ": ((f.high + f.low + f.close) / 3).values})


with ThreadPoolExecutor(4) as ex:
    H = pd.concat([x for x in ex.map(first_hour, sorted(S.sym.unique())) if x is not None], ignore_index=True)
S = S.merge(H, on=["sym", "date"], how="left")
S["stale_1h"] = np.isclose(S.h1_close, S.open, rtol=1e-6)
S.to_parquet(RAW_DIR / "etf_signals.parquet", index=False)

rows = []
for (side, lev), g in S.groupby(["side", "lev"]):
    sg = 1 if side == "down" else -1
    fill = (g.cons_lo < g.open * (1 - 1e-6)) if side == "down" else (g.cons_hi > g.open * (1 + 1e-6))
    ents = {"official open": sg * (g.close / g.open - 1) - bt.COST}
    ents["strict limit at open"] = ents["official open"].where(fill & g.cons_lo.notna())
    ents["first-hour close (~10:30)"] = sg * (g.close / g.h1_close - 1) - bt.COST
    ents["first-hour close, price moved"] = ents["first-hour close (~10:30)"].where(~g.stale_1h)
    ents["first-hour avg price"] = sg * (g.close / g.h1_typ - 1) - bt.COST
    for lab, r in ents.items():
        for liq, m in (("Q1-2", g.Q <= 2), ("Q3-5", g.Q >= 3)):
            c = stats.clustered_mean(r[m], g.date[m]); c.update(side=side, lev=lev, entry=lab, liq=liq,
                                                                stale=float(g.stale_1h[m].mean())); rows.append(c)
R = pd.DataFrame(rows)
R.to_csv(DOCS_DIR / "etf_2025.csv", index=False)
print(R[~R.lev].pivot_table(index=["side", "entry"], columns="liq", values=["N", "mean", "t"], aggfunc="first", sort=False).to_string(float_format=lambda x: f"{x:.4f}"))
print("\nleveraged/inverse only:")
print(R[R.lev].pivot_table(index=["side", "entry"], columns="liq", values=["N", "mean", "t"], aggfunc="first", sort=False).to_string(float_format=lambda x: f"{x:.4f}"))
print("\nmedian prior-yr $vol/day by Q:", (univ.groupby("Q").med_dvol.median() / 1e6).round(3).to_dict())

order = ["official open", "strict limit at open", "first-hour close (~10:30)", "first-hour close, price moved", "first-hour avg price"]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=140, sharey=True)
RR = R[~R.lev]
for ax, liq in zip(axes, ("Q1-2", "Q3-5")):
    for k, (side, col, lab) in enumerate((("down", "#2a78d6", "gap-down: buy, sell at close"), ("up", "#eb6834", "gap-up: short, cover at close"))):
        x = RR[(RR.side == side) & (RR.liq == liq)].set_index("entry").reindex(order)
        pos = np.arange(len(order)) + (k - 0.5) * 0.38
        ax.bar(pos, x["mean"] * 100, 0.36, color=col, label=lab)
        ax.errorbar(pos, x["mean"] * 100, yerr=1.96 * x["se"] * 100, fmt="none", ecolor="#52514e", lw=0.8)
    nd = int(RR[(RR.side == "down") & (RR.liq == liq)].N.max()); nu = int(RR[(RR.side == "up") & (RR.liq == liq)].N.max())
    ax.axhline(0, color="#52514e", lw=0.8); ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=20, ha="right", fontsize=8)
    ax.set_title(f"ETFs {liq} (excl. leveraged/inverse): {nd} dips, {nu} gap-ups", fontsize=10, loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel("mean per trade to official close, %\n(net 0.10%, 95% CI by date)", fontsize=8); axes[0].legend(frameon=False, fontsize=8)
fig.suptitle("US-listed ETFs (Arca / Cboe / NYSE American), 2025-01 → 2026-10: return by entry", fontsize=9.5, x=0.01, ha="left")
fig.tight_layout(); fig.savefig(DOCS_DIR / "etf_2025.png"); print("chart written")
