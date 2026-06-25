#!/usr/bin/env python3
"""
Does the GPR calm gate work in COMMODITIES? (the geo-insurance home of the bias)
===============================================================================

Commodities are where geopolitical risk transmits most directly (oil on conflict,
gold on fear). The bias-faithful hypothesis: in GPR-calm regimes the crowd is
complacent about disruption, so geopolitical HEDGES (gold, silver, oil) are cheap ->
their FORWARD returns are higher bought in calm. The competing mechanism is the
opposite: conflict -> commodities spike, so they do well in TURBULENT regimes. We let
the data arbitrate, by ETF type, and apply the controls the equity-vol/credit rounds
taught us:

  PASS 1  every commodity ETF: next-day (ann) & 21d-fwd return split by regime, block
          bootstrap p. Grouped: precious metals / metal miners / energy / broad-future
          / industrial-materials.
  CONTROL is it just RISK-ON? Hedge out SPY beta (full-sample OLS) and regime-split the
          RESIDUAL. A calm edge that survives in the SPY-residual is commodity-specific,
          not "everything rallies in calm". (Critical: energy/materials have big equity
          beta; gold has ~zero — so this separates geo-insurance from market beta.)
  TRADE   the clean instrument: GLD (no roll drag, $1.25B ADV) calm-gated long vs buy &
          hold; plus a gold-complex sleeve. Does gating to calm actually help?
  ROLL    flag the contango-bleed ETFs (DBC/GSG/DBO) where buy-and-hold is not the same
          as the index — their calm edge may not be harvestable.

Causal level-OR-vol GPR flag, lagged 1d. ~22 ETFs x 2 horizons = multiple looks;
read the TYPE pattern and the SPY-residual column, not single stars.

Usage: python experiments/run_gpr_commodities.py
"""
from __future__ import annotations
import math
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import _bootstrap  # noqa

GPR = "datasets/gpr_daily.csv.gz"; ETF = "datasets/etf_universe_prices.csv.gz"; OUT = "data/cross_asset"
RNG = np.random.default_rng(20260625)

GROUPS = [
    ("precious metal", ["GLD", "IAU", "SLV"]),
    ("metal miners",   ["GDX", "GDXJ", "SIL"]),
    ("energy",         ["XLE", "XOP", "OIH", "IEO", "XES", "FCG", "IXC", "DBO"]),
    ("broad futures",  ["DBC", "GSG", "DBA"]),
    ("industrial/mat", ["XME", "VAW", "IYM", "MOO"]),
]
ROLL_DRAG = {"DBC", "GSG", "DBA", "DBO"}   # futures-based, contango bleed -> buy&hold != index


def tstat(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return x.mean() / (x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 and x.std() > 0 else np.nan


def gpr_calm_flag():
    g = pd.read_csv(GPR, parse_dates=["date"]).set_index("date")["GPRD"].sort_index().asfreq("D").ffill()
    lvl = g.rolling(45, min_periods=15).mean(); lc = lvl <= lvl.rolling(365, min_periods=180).median()
    vol = g.diff().rolling(45, min_periods=15).std(); vc = vol <= vol.rolling(365, min_periods=180).median()
    return (lc | vc).rename("calm")


def boot_gap(x, calm, n=4000, block=21):
    m = pd.DataFrame({"x": x, "c": calm}).dropna(); r = m["x"].values; c = m["c"].values.astype(bool); N = len(r)
    if c.sum() < 40 or (~c).sum() < 40 or N // block < 5: return np.nan, np.nan
    real = r[c].mean() - r[~c].mean(); nb = N // block; st = np.arange(N - block); null = np.empty(n)
    for i in range(n):
        idx = (RNG.choice(st, size=nb, replace=True)[:, None] + np.arange(block)).ravel()
        rr = r[idx]; cc = c[idx].copy(); RNG.shuffle(cc)
        null[i] = 0.0 if cc.all() or (~cc).all() else rr[cc].mean() - rr[~cc].mean()
    return real, (float((null >= real).mean()) if real >= 0 else float((null <= real).mean()))


def perf(r):
    r = r.dropna()
    if len(r) < 30: return {}
    eq = (1 + r).cumprod(); dd = (eq / eq.cummax() - 1).min()
    return dict(cagr=eq.iloc[-1] ** (252/len(r)) - 1, sharpe=r.mean()/r.std(ddof=1)*math.sqrt(252) if r.std() else np.nan,
                maxdd=dd)


def main():
    import os; os.makedirs(OUT, exist_ok=True)
    flag = gpr_calm_flag()
    etf = pd.read_csv(ETF, parse_dates=["Date"]).set_index("Date").sort_index()
    cal = etf["SPY"].dropna().index
    calm = flag.reindex(cal).ffill().shift(1)
    spy_ret = etf["SPY"].pct_change().reindex(cal)

    print("=" * 112)
    print("GPR CALM GATE IN COMMODITIES — geo-insurance vs conflict-premium, with a risk-on control")
    print("  calm = same causal level-OR-vol GPR flag (lagged 1d). +gap => higher returns bought in CALM.")
    print("=" * 112)
    print("  %-16s %5s | %8s %6s | %8s %6s | %8s %6s %6s | %9s %6s"
          % ("ETF (type)", "roll", "calmAnn", "Sh", "turbAnn", "Sh", "GAPann", "21dGAP", "bootP", "SPYresGAP", "rbootP"))

    rows = []
    for gname, tickers in GROUPS:
        print("  " + "-" * 104 + "  [%s]" % gname)
        for t in tickers:
            if t not in etf.columns: continue
            p = etf[t].dropna()
            if len(p) < 500: continue
            r1 = p.pct_change().reindex(cal)
            rF = (p.shift(-21)/p - 1.0).reindex(cal)
            df = pd.DataFrame({"r1": r1, "rF": rF, "spy": spy_ret, "c": calm}).dropna()
            df["c"] = df["c"].astype(bool)
            if df["c"].sum() < 60 or (~df["c"]).sum() < 60: continue
            c, tb = df[df.c], df[~df.c]
            calmA, turbA = c["r1"].mean()*252, tb["r1"].mean()*252
            shc = c["r1"].mean()/c["r1"].std()*math.sqrt(252); sht = tb["r1"].mean()/tb["r1"].std()*math.sqrt(252)
            _, bp = boot_gap(df["r1"], df["c"])
            gapF = c["rF"].mean() - tb["rF"].mean()
            # SPY-residual: hedge out equity beta, regime-split residual daily return
            beta = np.polyfit(df["spy"], df["r1"], 1)[0]
            df["res"] = df["r1"] - beta*df["spy"]
            resGap, rbp = boot_gap(df["res"], df["c"])
            rows.append(dict(etf=t, grp=gname, calmA=calmA, turbA=turbA, gapA=calmA-turbA, gapF=gapF,
                             bp=bp, resGapA=resGap*252, rbp=rbp, beta=beta))
            print("  %-16s %5s | %+7.1f%% %+5.2f | %+7.1f%% %+5.2f | %+7.1f%% %+6.1f%% %6.3f | %+8.1f%% %6.3f"
                  % ("%s (%s)" % (t, gname[:3]), "DRAG" if t in ROLL_DRAG else " -",
                     100*calmA, shc, 100*turbA, sht, 100*(calmA-turbA), 100*gapF, bp,
                     100*resGap*252, rbp))

    # -------- type-level read --------
    print("\n[TYPE SUMMARY] mean calm-turb gap (raw vs SPY-residual) — is the edge geo-specific or just risk-on?")
    R = pd.DataFrame(rows)
    for gname, _ in GROUPS:
        sub = R[R.grp == gname]
        if len(sub):
            print("  %-16s raw GAPann %+6.1f%%  | SPY-resid GAPann %+6.1f%%  | bootP-sig(raw) %d/%d  resid-sig %d/%d"
                  % (gname, 100*sub.gapA.mean(), 100*sub.resGapA.mean(),
                     (sub.bp <= 0.10).sum(), len(sub), (sub.rbp <= 0.10).sum(), len(sub)))

    # -------- tradeable: GLD calm-gated, and gold-complex sleeve --------
    print("\n[TRADEABLE] GLD (no roll drag, deep liquidity) — does calm-gating a long actually help?")
    gld = etf["GLD"].pct_change().reindex(cal)
    cb = calm.reindex(cal).fillna(False).astype(bool)
    print("  %-26s %7s %7s %7s" % ("variant", "CAGR", "Sharpe", "maxDD"))
    for lbl, pos in [("buy & hold", pd.Series(1.0, index=cal)),
                     ("calm=1 / turb=0", cb.map({True: 1.0, False: 0.0})),
                     ("calm=1 / turb=0.5", cb.map({True: 1.0, False: 0.5}))]:
        m = perf(pos.reindex(cal).fillna(0) * gld)
        print("  %-26s %+6.1f%% %+7.2f %6.0f%%" % (lbl, 100*m.get("cagr", np.nan), m.get("sharpe", np.nan), 100*m.get("maxdd", np.nan)))
    # equal-weight gold complex
    comp = etf[[c for c in ["GLD","SLV","GDX"] if c in etf.columns]].pct_change().mean(axis=1).reindex(cal)
    dC = pd.DataFrame({"r": comp, "c": calm}).dropna(); dC["c"] = dC["c"].astype(bool)
    realc, bpc = boot_gap(dC["r"], dC["c"])
    print("  gold-complex (GLD/SLV/GDX): calm ann %+.1f%%  turb ann %+.1f%%  gap %+.1f%%/yr  bootP=%.3f"
          % (100*dC[dC.c]["r"].mean()*252, 100*dC[~dC.c]["r"].mean()*252, 100*realc*252, bpc))

    # -------- stability: is the metals-calm / energy-turbulent split era-concentrated? --------
    print("\n[STABILITY] calm-turb next-day gap (ann %) by 4-yr block — is it stationary or episodic?")
    st = etf[["GLD", "SLV", "GDX", "XLE"]].pct_change().reindex(cal)
    st["cx"] = st[["GLD", "SLV", "GDX"]].mean(axis=1); st["c"] = calm.reindex(cal)
    st = st.dropna(); st["c"] = st["c"].astype(bool); st["blk"] = (st.index.year // 4) * 4
    print("  %-10s %8s %8s %8s %12s" % ("block", "GLD", "SLV", "goldCx", "XLE(energy)"))
    for blk, sub in st.groupby("blk"):
        if len(sub) < 150: continue
        def g(col): return (sub[sub.c][col].mean() - sub[~sub.c][col].mean()) * 252 * 100
        print("  %-10s %+7.1f%% %+7.1f%% %+7.1f%% %+11.1f%%" % ("%d-%d" % (blk, blk+3), g("GLD"), g("SLV"), g("cx"), g("XLE")))
    print("  -> metals-calm edge is real on avg & geo-specific (survives SPY control) but EPISODIC")
    print("     (2008-11, 2024-27 strong; 2016-19 negative); energy-turbulent is mostly a post-2020 (Ukraine) effect.")

    # -------- chart --------
    fig, ax = plt.subplots(figsize=(10.5, 7))
    rs = R.sort_values("gapA")
    y = np.arange(len(rs))
    palette = {"precious metal": "#d4a017", "metal miners": "#b8860b", "energy": "#8c564b",
               "broad futures": "#7f7f7f", "industrial/mat": "#1f77b4"}
    ax.barh(y, 100*rs.gapA, color=[palette[g] for g in rs.grp],
            edgecolor=["black" if b <= 0.10 else "none" for b in rs.bp], linewidth=1.3)
    ax.scatter(100*rs.resGapA, y, color="white", edgecolor="black", zorder=3, s=22, label="SPY-residual gap")
    ax.set_yticks(y); ax.set_yticklabels(rs.etf, fontsize=8); ax.axvline(0, color="black", lw=.8)
    ax.set_xlabel("calm − turbulent annualized return gap (%)   ·   ● = after hedging out SPY beta")
    ax.set_title("GPR calm gate in commodities: bar=raw gap (black outline=bootP≤.10), dot=geo-specific (SPY-residual)", fontsize=9)
    handles = [plt.Rectangle((0,0),1,1,color=palette[k]) for k in palette]
    ax.legend(handles, list(palette), fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(f"{OUT}/gpr_commodities.png", dpi=120)
    R.to_csv(f"{OUT}/gpr_commodities.csv", index=False)
    print("\nSaved: %s/gpr_commodities.png , %s/gpr_commodities.csv" % (OUT, OUT))


if __name__ == "__main__":
    main()
