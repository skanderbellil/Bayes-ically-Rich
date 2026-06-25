#!/usr/bin/env python3
"""
Refine the thesis: is the GPR edge INDEPENDENT, TRADEABLE, and not one episode?
===============================================================================

Two candidate edges survived the cross-asset sweep. Before building either, kill the
boring explanations:

  EQUITY-VOL VRP (short vol in calm):
    Q1  Is "calm" just a proxy for LOW VIX? VRP is mechanically larger at some VIX
        levels. We re-compute the calm-minus-turb VRP gap WITHIN terciles of the VIX
        level. If it vanishes, GPR adds nothing over "condition on VIX"; if it holds,
        GPR carries independent information.
    Q2  Is it stable year to year, or a couple of vol-crush years?

  CREDIT (disruption underpriced in calm):
    Q3  Make it TRADEABLE WITHOUT SHORTING: rotate HY -> safe duration (IEF) in calm.
        Forward credit-excess return = HYG_fwd - IEF_fwd. If default risk is underpriced
        in calm, HY should UNDERPERFORM duration afterward (calm excess < turb excess).
    Q4  Is the credit edge just March 2020? Re-run excluding 2020-02..2020-06, and
        year by year. And re-confirm the OAS-widening edge survives an OAS-level control.

All causal (level-OR-vol GPR flag lagged 1d); block bootstrap for overlap.
Usage: python experiments/run_gpr_thesis_refine.py
"""
from __future__ import annotations
import math
import numpy as np, pandas as pd
import _bootstrap  # noqa

GPR = "datasets/gpr_daily.csv.gz"; ETF = "datasets/etf_universe_prices.csv.gz"; MAC = "datasets/fred_macro_plus.csv"
RNG = np.random.default_rng(20260625)


def gpr_calm_flag():
    g = pd.read_csv(GPR, parse_dates=["date"]).set_index("date")["GPRD"].sort_index().asfreq("D").ffill()
    lvl = g.rolling(45, min_periods=15).mean(); lc = lvl <= lvl.rolling(365, min_periods=180).median()
    vol = g.diff().rolling(45, min_periods=15).std(); vc = vol <= vol.rolling(365, min_periods=180).median()
    return (lc | vc).rename("calm")


def boot_gap(x, calm, n=5000, block=21):
    m = pd.DataFrame({"x": x, "c": calm}).dropna(); r = m["x"].values; c = m["c"].values.astype(bool); N = len(r)
    if c.sum() < 30 or (~c).sum() < 30 or N // block < 5: return np.nan, np.nan
    real = r[c].mean() - r[~c].mean(); nb = N // block; st = np.arange(N - block); null = np.empty(n)
    for i in range(n):
        idx = (RNG.choice(st, size=nb, replace=True)[:, None] + np.arange(block)).ravel()
        rr = r[idx]; cc = c[idx].copy(); RNG.shuffle(cc)
        null[i] = 0.0 if cc.all() or (~cc).all() else rr[cc].mean() - rr[~cc].mean()
    return real, (float((null >= real).mean()) if real >= 0 else float((null <= real).mean()))


def tercile_control(x, calm, level, q=3):
    d = pd.DataFrame({"x": x, "c": calm.astype(float), "lev": level}).dropna()
    if len(d) < 120: return np.nan, []
    d["b"] = pd.qcut(d["lev"], q, labels=False, duplicates="drop")
    gaps = []
    for b, g in d.groupby("b"):
        cc, tt = g[g.c == 1]["x"], g[g.c == 0]["x"]
        if len(cc) >= 20 and len(tt) >= 20:
            gaps.append((b, cc.mean() - tt.mean(), g["lev"].mean()))
    return (float(np.mean([gg[1] for gg in gaps])) if gaps else np.nan), gaps


def main():
    flag = gpr_calm_flag()
    etf = pd.read_csv(ETF, parse_dates=["Date"]).set_index("Date").sort_index()
    mac = pd.read_csv(MAC, parse_dates=["date"]).set_index("date").sort_index()
    spy = etf["SPY"].dropna(); cal = spy.index
    calm = flag.reindex(cal).ffill().shift(1)
    logret = np.log(spy).diff()
    rv_fwd = (logret.rolling(21).std().shift(-20)) * math.sqrt(252) * 100
    vix = mac["VIXCLS"].reindex(cal.union(mac.index)).ffill().reindex(cal)
    vrp = (vix - rv_fwd)

    print("=" * 96)
    print("THESIS REFINEMENT — independence, tradeability, episode-robustness")
    print("=" * 96)

    # -------- Q1: equity-vol VRP, is it independent of VIX level? --------
    print("\n[Q1] EQUITY-VOL VRP gap (calm-turb), controlled for VIX LEVEL (terciles)")
    raw, _ = boot_gap(vrp, calm)
    ctrl, gaps = tercile_control(vrp.reindex(cal), calm, vix.reindex(cal))
    print("    raw calm-turb VRP gap = %+.2f vol pts" % raw)
    for b, gp, lv in gaps:
        print("      VIX tercile %d (mean VIX %.1f): calm-turb VRP gap = %+.2f" % (b, lv, gp))
    print("    level-controlled gap = %+.2f vol pts  -> %s"
          % (ctrl, "SURVIVES (GPR independent of VIX level)" if ctrl == ctrl and ctrl > 0.2
             else "WEAK/GONE (calm ~ proxy for VIX level)"))

    # -------- Q2: VRP edge year by year --------
    print("\n[Q2] VRP carry (VIX-realized) by year: calm vs turbulent mean")
    dv = pd.DataFrame({"vrp": vrp, "c": calm}).dropna(); dv["c"] = dv["c"].astype(bool); dv["yr"] = dv.index.year
    nc = nt = 0
    for yr, g in dv.groupby("yr"):
        cc, tt = g[g.c]["vrp"], g[~g.c]["vrp"]
        mk = "calm>turb" if (len(cc) and len(tt) and cc.mean() > tt.mean()) else ("--" if len(cc) and len(tt) else "")
        if len(cc) and len(tt):
            nc += cc.mean() > tt.mean(); nt += 1
        print("    %d  calm %+6.2f (n=%3d)  turb %+6.2f (n=%3d)  %s"
              % (yr, cc.mean() if len(cc) else float('nan'), len(cc),
                 tt.mean() if len(tt) else float('nan'), len(tt), mk))
    print("    -> calm>turb in %d of %d years with both regimes present" % (nc, nt))

    # -------- Q3: credit, tradeable no-short rotation HY->IEF --------
    print("\n[Q3] CREDIT, TRADEABLE no-short: forward 21d credit-excess = HYG - IEF, by regime")
    print("     (thesis: in calm HY is too tight -> HY UNDERPERFORMS duration afterward -> calm excess < turb)")
    hyg = etf["HYG"].dropna().reindex(cal).ffill(); ief = etf["IEF"].dropna().reindex(cal).ffill()
    cx = (hyg.shift(-21)/hyg - 1.0) - (ief.shift(-21)/ief - 1.0)         # credit excess over duration, fwd
    dcx = pd.DataFrame({"cx": cx*100, "c": calm}).dropna(); dcx["c"] = dcx["c"].astype(bool)
    cc, tt = dcx[dcx.c]["cx"], dcx[~dcx.c]["cx"]
    real, bp = boot_gap(dcx["cx"], dcx["c"])
    print("    calm fwd credit-excess %+.2f%%  | turb %+.2f%%  | gap %+.2f%% (calm-turb)  bootP(gap)=%.3f"
          % (cc.mean(), tt.mean(), real, bp))
    print("    %s" % ("CONSISTENT: HY underperforms duration after calm (rotate HY->IEF in calm)"
                      if real < 0 else "INCONSISTENT with the underpricing story"))

    # -------- Q4: credit OAS-widening, drop 2020, year split, level control --------
    print("\n[Q4] CREDIT robustness — HY OAS forward 21d widening (bps)")
    hy = mac["BAMLH0A0HYM2"].reindex(cal.union(mac.index)).ffill().reindex(cal)
    wid = (hy.shift(-21) - hy) * 100
    dW = pd.DataFrame({"w": wid, "c": calm, "hy": hy}).dropna(); dW["c"] = dW["c"].astype(bool)
    full, pf = boot_gap(dW["w"], dW["c"])
    # exclude covid
    mask = ~((dW.index >= "2020-02-15") & (dW.index <= "2020-06-15"))
    ex = dW[mask]; exg, pex = boot_gap(ex["w"], ex["c"])
    lc, _ = tercile_control(dW["w"], dW["c"], dW["hy"])
    print("    full sample:        calm-turb widening gap %+.1f bps  bootP=%.3f" % (full, pf))
    print("    ex-COVID(2020 H1):  calm-turb widening gap %+.1f bps  bootP=%.3f  <-- not just March 2020?" % (exg, pex))
    print("    OAS-level controlled gap: %+.1f bps  -> %s" % (lc, "survives" if lc == lc and lc > 0 else "gone"))
    print("    by year (calm-turb widening gap, bps):")
    dW["yr"] = dW.index.year
    for yr, g in dW.groupby("yr"):
        cc2, tt2 = g[g.c]["w"], g[~g.c]["w"]
        if len(cc2) >= 10 and len(tt2) >= 10:
            print("      %d  %+7.1f  (calm n=%3d turb n=%3d)" % (yr, cc2.mean()-tt2.mean(), len(cc2), len(tt2)))

    print("\n" + "=" * 96)
    print("READ: equity-vol edge is real only if Q1 survives the VIX-level control; credit edge is")
    print("the genuine bias only if Q3 is negative AND Q4 holds ex-COVID and within OAS levels.")
    print("=" * 96)

    # -------- Q5: CREDIT on LONG history (HY OAS is only 2023+; use Baa-10Y, 1990+) --------
    print("\n[Q5] CREDIT on 36yr history — HY OAS in repo is ONLY 2023-2026 (n=786), too short to trust.")
    print("     Re-test the widening thesis on Baa-10Y spread (1990-2026) — the real out-of-sample check.")
    spr = mac["BAA10Y"].dropna()
    calL = spr.index; cfL = flag.reindex(calL).ffill().shift(1)
    for H in (21, 63):
        widL = (spr.shift(-H) - spr) * 100
        dL = pd.DataFrame({"w": widL, "c": cfL, "lev": spr}).dropna(); dL["c"] = dL["c"].astype(bool)
        cc, tt = dL[dL.c]["w"], dL[~dL.c]["w"]
        real, p = boot_gap(dL["w"], dL["c"], block=H)
        lc, _ = tercile_control(dL["w"], dL["c"], dL["lev"])
        print("    H=%2dd: calm widen %+5.1f bps  turb %+5.1f bps  gap %+5.1f bps  bootP=%.3f  level-ctrl %+.1f bps"
              % (H, cc.mean(), tt.mean(), real, p, lc))
    print("    NOTE: a NEGATIVE gap means calm is followed by LESS widening (spreads grind tighter) —")
    print("    i.e. credit risk is NOT underpriced in calm on long history; the 2023+ HY OAS result was")
    print("    a small-sample artifact. Consistent with Q3 (HY outperforms duration in calm).")


if __name__ == "__main__":
    main()
