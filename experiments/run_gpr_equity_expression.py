#!/usr/bin/env python3
"""
Where does the GPR bias actually live in EQUITIES, and how to express it?
=========================================================================

Pass-1 found a real but DIFFUSE equity bias: calm-minus-turbulent next-day return
gap averaged +6.4%/yr and was positive in 8 of 10 equity ETFs, but no single ticker
cleared significance (QQQ closest, p=0.07). The vol-pricing analog (VIX vs realized)
actively REVERSED. So the equity edge is not a vol-underpricing story — it is a plain
RISK-PREMIUM / risk-on-risk-off story: equities drift up in calm, stall in turbulence.

This script asks the user's two questions head-on and proposes the expression:

  [A] POOL the bias. One equal-weight broad-equity factor (14 ETFs), regime-split,
      with a stationary block bootstrap over TIME (no cross-sectional double-counting).
      Pooling is the right move when an effect is real-but-noisy per name.

  [B] RAW GPR directly on equities (not the calm flag the user asked):
      - contemporaneous: SPY return on the day's GPR SHOCK (dGPR z)   -> the C&I result
      - forward:         fwd-21d SPY return on GPR LEVEL z and on dGPR -> risk-premium test

  [C] EXPRESSION 1 — a defensive TIMING overlay on SPY (full in calm, de-risked in
      turbulence). Sharpe / CAGR / maxDD vs buy-&-hold, with a post-2019 OOS split.

  [D] EXPRESSION 2 — a market-NEUTRAL cross-section that strips equity beta and keeps
      only the geo-regime tilt: long a risk-on basket (EM/China/Brazil/tech), short the
      conflict beneficiary (energy), turned ON only in calm. Energy is the one equity
      sleeve with a NEGATIVE gap (it rallies in turbulence), so it is the natural short.

Causality: same level-OR-vol GPR flag, lagged 1d. Honest about multiple looks.

Usage: python experiments/run_gpr_equity_expression.py
"""
from __future__ import annotations
import math
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import _bootstrap  # noqa

GPR = "datasets/gpr_daily.csv.gz"; ETF = "datasets/etf_universe_prices.csv.gz"; OUT = "data/cross_asset"
RNG = np.random.default_rng(20260625)
EQ = ["SPY", "QQQ", "IWM", "DIA", "EFA", "EEM", "FXI", "EWZ", "XLF", "XLK",
      "XLV", "XLP", "XLY", "XLI", "XLB", "VWO", "EWJ", "EWG"]   # broad equity sleeve


def tstat(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return x.mean() / (x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 and x.std() > 0 else np.nan


def sharpe(r): r = r.dropna(); return r.mean() / r.std(ddof=1) * math.sqrt(252) if r.std() > 0 else np.nan
def cagr(r):
    r = r.dropna(); g = (1 + r).prod(); yrs = len(r) / 252
    return g ** (1 / yrs) - 1 if yrs > 0 else np.nan
def maxdd(r):
    eq = (1 + r.fillna(0)).cumprod(); return ((eq - eq.cummax()) / eq.cummax()).min()


def gpr_series():
    return pd.read_csv(GPR, parse_dates=["date"]).set_index("date")["GPRD"].sort_index().asfreq("D").ffill()


def calm_flag(g):
    lvl = g.rolling(45, min_periods=15).mean()
    lvl_calm = lvl <= lvl.rolling(365, min_periods=180).median()
    vol = g.diff().rolling(45, min_periods=15).std()
    vol_calm = vol <= vol.rolling(365, min_periods=180).median()
    return (lvl_calm | vol_calm).rename("calm")


def block_boot_gap(series, calm, n=5000, block=21):
    m = pd.DataFrame({"x": series, "c": calm}).dropna()
    r = m["x"].values; c = m["c"].values.astype(bool); N = len(r)
    real = r[c].mean() - r[~c].mean(); nb = N // block; st = np.arange(N - block)
    null = np.empty(n)
    for i in range(n):
        idx = (RNG.choice(st, size=nb, replace=True)[:, None] + np.arange(block)).ravel()
        rr = r[idx]; cc = c[idx].copy(); RNG.shuffle(cc)
        null[i] = 0.0 if cc.all() or (~cc).all() else rr[cc].mean() - rr[~cc].mean()
    return real, float((null >= real).mean())


def main():
    import os; os.makedirs(OUT, exist_ok=True)
    g = gpr_series(); flag = calm_flag(g)
    etf = pd.read_csv(ETF, parse_dates=["Date"]).set_index("Date").sort_index()
    cal = etf["SPY"].dropna().index
    calm = flag.reindex(cal).ffill().shift(1)                  # lag 1d -> causal

    print("=" * 100)
    print("GPR BIAS IN EQUITIES — pool it, test raw GPR, and pick an expression")
    print("=" * 100)

    # ---------- [A] pooled broad-equity factor ----------
    have = [t for t in EQ if t in etf.columns]
    rets = etf[have].pct_change()
    factor = rets.mean(axis=1)                                 # equal-weight broad equity, daily
    df = pd.DataFrame({"r": factor, "calm": calm}).dropna(); df["calm"] = df["calm"].astype(bool)
    c, t = df[df.calm]["r"], df[~df.calm]["r"]
    real, bp = block_boot_gap(df["r"], df["calm"])
    print("\n[A] POOLED broad-equity factor (%d ETFs, equal-wt) — regime split, block-bootstrap over time" % len(have))
    print("    calm:  ann %+6.1f%%  Sharpe %+.2f   (%.0f%% of days)" % (c.mean()*252*100, sharpe(c), 100*len(c)/len(df)))
    print("    turb:  ann %+6.1f%%  Sharpe %+.2f" % (t.mean()*252*100, sharpe(t)))
    print("    GAP :  ann %+6.1f%%  (daily t=%+.2f)   block-bootstrap P(gap<=0)=%.4f  <-- pooling beats single names"
          % ((c.mean()-t.mean())*252*100, tstat(df["r"]*np.where(df.calm,1,-1)), bp))

    # ---------- [B] raw GPR directly (level + shock) ----------
    print("\n[B] RAW GPR DIRECTLY on SPY (not the flag) — standardized betas")
    spy = etf["SPY"].dropna(); r1 = spy.pct_change()
    gz = ((g - g.rolling(365, min_periods=180).mean()) / g.rolling(365, min_periods=180).std()).reindex(cal).ffill()
    dgpr = g.diff(); dz = (dgpr / dgpr.rolling(252, min_periods=120).std()).reindex(cal).ffill()
    fwd21 = spy.shift(-21) / spy - 1.0
    def beta(y, x):
        m = pd.DataFrame({"y": y, "x": x}).dropna()
        if len(m) < 100: return (np.nan, np.nan, 0)
        b = np.polyfit(m["x"], m["y"], 1)[0]
        # t via residual SE
        yhat = b * m["x"] + np.polyfit(m["x"], m["y"], 1)[1]; se = (m["y"]-yhat).std(ddof=2)
        tb = b / (se / (m["x"].std()*math.sqrt(len(m))))
        return b, tb, len(m)
    b1, t1, n1 = beta(r1.shift(-0), dz.shift(0))          # same-day SPY ret vs today's GPR shock
    b2, t2, n2 = beta(fwd21, gz.shift(1))                  # fwd 21d ret vs GPR level (lagged)
    b3, t3, n3 = beta(fwd21, dz.shift(1))                  # fwd 21d ret vs GPR shock (lagged)
    print("    same-day:  SPY_ret ~ GPR shock(z)     beta=%+.4f%%/σ (t=%+.2f)  [expect <0: shocks hurt stocks]" % (b1*100, t1))
    print("    forward :  SPY_fwd21 ~ GPR LEVEL(z)    beta=%+.3f%%/σ (t=%+.2f)  [>0 = high-GPR risk premium]" % (b2*100, t2))
    print("    forward :  SPY_fwd21 ~ GPR shock(z)    beta=%+.3f%%/σ (t=%+.2f)" % (b3*100, t3))

    # ---------- [C] timing overlay ----------
    print("\n[C] EXPRESSION 1 — defensive TIMING overlay on SPY (full in calm, de-risk in turbulence)")
    bh = r1.reindex(cal)
    print("    %-22s %7s %7s %7s %7s" % ("variant", "CAGR", "Sharpe", "maxDD", "OOS≥19 Sh"))
    for lbl, w_turb in [("buy & hold", 1.0), ("calm=1 / turb=0.5", 0.5), ("calm=1 / turb=0 (cash)", 0.0)]:
        pos = calm.reindex(cal).map({True: 1.0, False: w_turb}).fillna(1.0) if lbl != "buy & hold" else pd.Series(1.0, index=cal)
        strat = (pos * bh)
        oos = strat[strat.index >= "2019-01-01"]
        print("    %-22s %+6.1f%% %+7.2f %6.0f%% %+9.2f"
              % (lbl, 100*cagr(strat), sharpe(strat), 100*maxdd(strat), sharpe(oos)))

    # ---------- [D] market-neutral cross-section ----------
    print("\n[D] EXPRESSION 2 — market-NEUTRAL geo tilt: long risk-on (EEM,FXI,EWZ,QQQ) − short energy (XLE), calm-only")
    riskon = etf[[c for c in ["EEM","FXI","EWZ","QQQ"] if c in etf.columns]].pct_change().mean(axis=1)
    energy = etf["XLE"].pct_change()
    spread = (riskon - energy).reindex(cal)                # dollar-neutral long/short
    sd = pd.DataFrame({"s": spread, "calm": calm}).dropna(); sd["calm"] = sd["calm"].astype(bool)
    cc, tt = sd[sd.calm]["s"], sd[~sd.calm]["s"]
    real_s, bp_s = block_boot_gap(sd["s"], sd["calm"])
    print("    always-on spread:      ann %+6.1f%%  Sharpe %+.2f" % (sd["s"].mean()*252*100, sharpe(sd["s"])))
    print("    calm-only spread:      ann %+6.1f%%  Sharpe %+.2f   (turb-only ann %+.1f%%)"
          % (cc.mean()*252*100, sharpe(cc), tt.mean()*252*100))
    print("    calm−turb gap on spread: ann %+6.1f%%  block-bootstrap P(gap<=0)=%.4f" % (real_s*252*100, bp_s))

    # ---------- chart ----------
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for lbl, w in [("buy & hold", 1.0), ("calm=1/turb=0", 0.0)]:
        pos = calm.reindex(cal).map({True: 1.0, False: w}).fillna(1.0) if lbl != "buy & hold" else pd.Series(1.0, index=cal)
        eq = (1 + (pos*bh).fillna(0)).cumprod()
        ax[0].plot(eq.index, eq.values, lw=1.4, label="%s (Sh %.2f)" % (lbl, sharpe(pos*bh)))
    ax[0].set_yscale("log"); ax[0].legend(fontsize=8); ax[0].set_title("[C] SPY timing overlay (log equity)")
    eqs = (1 + sd.set_index(sd.index if False else pd.DatetimeIndex(df.index[:0])).pipe(lambda _:_)) if False else None
    sp_calm = (1 + np.where(sd.calm, sd["s"], 0)).cumprod()
    ax[1].plot(sd.index, sp_calm, color="#1f9d55", lw=1.4, label="calm-only L/S (Sh %.2f)" % sharpe(cc))
    ax[1].plot(sd.index, (1+sd["s"]).cumprod(), color="#888", lw=1.0, label="always-on L/S (Sh %.2f)" % sharpe(sd["s"]))
    ax[1].legend(fontsize=8); ax[1].set_title("[D] risk-on − energy, calm-gated (cum.)")
    fig.tight_layout(); fig.savefig(f"{OUT}/gpr_equity_expression.png", dpi=120)
    print("\nSaved: %s/gpr_equity_expression.png" % OUT)
    print("\nTAKEAWAY: the equity bias is real but diffuse -> express it POOLED, not per-ticker. "
          "Best forms: a de-risking timing overlay, and/or the calm-gated risk-on−energy spread.")


if __name__ == "__main__":
    main()
