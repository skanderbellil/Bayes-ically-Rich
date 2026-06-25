#!/usr/bin/env python3
"""
Can the GPR level-OR-vol gate be exploited through EQUITY VOLATILITY / VIX?
==========================================================================

The prediction-market bias is COMPLACENCY: in GPR-calm regimes the crowd under-
prices disruption. In a thin retail market that = cheap disruptive YES. To exploit
the SAME psychology through equity vol it would have to make protection CHEAP in
calm (VIX too low). The earlier disruption test found the opposite (VIX is RICH in
calm, realized tails no worse) — equity vol is too heavily hedged/arbitraged to be
left cheap by complacency. So we test the two things that CAN be true instead:

  [1] DOES GPR IMPROVE A VOL FORECAST?  Vol is forecastable; the real question is
      whether the GPR gate adds info BEYOND current realized vol and VIX. Regress
      forward 21d realized vol on (current RV, VIX, calm-flag). If the calm coef is
      negative & robust, GPR sharpens the forecast -> usable for vol-managed sizing.

  [2] IS THE VRP HARVEST BETTER (and the TAIL acceptable) IN CALM?  Selling vol earns
      VIX - realized. We split that P&L by regime and look at BOTH the mean AND the
      left tail (CVaR5, worst), because the whole risk of "ride the complacency" is a
      fatter crash when it breaks. Plus the VIX term-structure slope by regime (is the
      curve more complacently-contango in calm?).

Direction matters: [2] is FADING-WITH the hedgers (collect premium), NOT betting on
the bias. Betting on the bias (long convexity in calm) needs calm tails to be bigger
— which the data does not show. We report all of it and let the user choose.

Causality: same level-OR-vol GPR flag, lagged 1d. Block bootstrap for overlap.
VIX futures are NOT in the repo, so the VRP P&L is the variance-swap proxy
(VIX - realized), not a tradeable futures roll — flagged. Usage:
  python experiments/run_gpr_equity_vol.py
"""
from __future__ import annotations
import math
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import _bootstrap  # noqa

GPR = "datasets/gpr_daily.csv.gz"; ETF = "datasets/etf_universe_prices.csv.gz"
MAC = "datasets/fred_macro_plus.csv"; TS = "datasets/vix_termstructure.csv"; OUT = "data/cross_asset"
RNG = np.random.default_rng(20260625)


def gpr_calm_flag():
    g = pd.read_csv(GPR, parse_dates=["date"]).set_index("date")["GPRD"].sort_index().asfreq("D").ffill()
    lvl = g.rolling(45, min_periods=15).mean(); lc = lvl <= lvl.rolling(365, min_periods=180).median()
    vol = g.diff().rolling(45, min_periods=15).std(); vc = vol <= vol.rolling(365, min_periods=180).median()
    return (lc | vc).rename("calm")


def block_boot_mean_gap(x, calm, n=5000, block=21):
    m = pd.DataFrame({"x": x, "c": calm}).dropna(); r = m["x"].values; c = m["c"].values.astype(bool); N = len(r)
    real = r[c].mean() - r[~c].mean(); nb = N // block; st = np.arange(N - block); null = np.empty(n)
    for i in range(n):
        idx = (RNG.choice(st, size=nb, replace=True)[:, None] + np.arange(block)).ravel()
        rr = r[idx]; cc = c[idx].copy(); RNG.shuffle(cc)
        null[i] = 0.0 if cc.all() or (~cc).all() else rr[cc].mean() - rr[~cc].mean()
    return real, float((null >= real).mean()) if real >= 0 else float((null <= real).mean())


def block_boot_coef(y, X, idx_calm, n=2000, block=42):
    """Bootstrap the last regression coefficient (the calm flag) with time blocks."""
    d = pd.concat([y, X], axis=1).dropna(); yv = d.iloc[:, 0].values; Xv = d.iloc[:, 1:].values
    Xv = np.column_stack([np.ones(len(Xv)), Xv]); N = len(yv); k = Xv.shape[1]
    beta = np.linalg.lstsq(Xv, yv, rcond=None)[0]
    nb = N // block; st = np.arange(N - block); coefs = np.empty(n)
    for i in range(n):
        ix = (RNG.choice(st, size=nb, replace=True)[:, None] + np.arange(block)).ravel()
        b = np.linalg.lstsq(Xv[ix], yv[ix], rcond=None)[0]; coefs[i] = b[-1]
    return beta, coefs.std()


def main():
    import os; os.makedirs(OUT, exist_ok=True)
    flag = gpr_calm_flag()
    etf = pd.read_csv(ETF, parse_dates=["Date"]).set_index("Date").sort_index()
    mac = pd.read_csv(MAC, parse_dates=["date"]).set_index("date").sort_index()
    ts = pd.read_csv(TS, parse_dates=["Date"]).set_index("Date").sort_index()
    spy = etf["SPY"].dropna(); cal = spy.index
    calm = flag.reindex(cal).ffill().shift(1)
    logret = np.log(spy).diff()
    rv_cur = logret.rolling(21).std() * math.sqrt(252) * 100                 # trailing 21d realized vol (ann %)
    rv_fwd = (logret.rolling(21).std().shift(-20)) * math.sqrt(252) * 100    # forward 21d realized vol
    vix = mac["VIXCLS"].reindex(cal.union(mac.index)).ffill().reindex(cal)

    print("=" * 100)
    print("GPR GATE x EQUITY VOL — does the complacency bias pay through vol / VIX?")
    print("=" * 100)

    # ---------- [1] does GPR improve the vol forecast? ----------
    print("\n[1] VOL FORECAST: forward 21d realized vol ~ current RV + VIX + 1[calm]")
    Xdf = pd.DataFrame({"RV": rv_cur, "VIX": vix, "calm": calm.astype("float")})
    beta, se = block_boot_coef(rv_fwd.rename("y"), Xdf, calm)
    names = ["const", "RV", "VIX", "calm"]
    print("    %-6s %9s" % ("term", "coef"))
    for nm, b in zip(names, beta):
        extra = "  (block-boot SE %.3f, t=%+.2f)" % (se, b/se) if nm == "calm" else ""
        print("    %-6s %+9.3f%s" % (nm, b, extra))
    # plain regime means of forward vol (context)
    d1 = pd.DataFrame({"rvf": rv_fwd, "c": calm}).dropna(); d1["c"] = d1["c"].astype(bool)
    print("    context: mean fwd-vol  calm=%.1f%%  turb=%.1f%%  (calm coef<0 => GPR lowers vol forecast beyond RV/VIX)"
          % (d1[d1.c]["rvf"].mean(), d1[~d1.c]["rvf"].mean()))

    # ---------- [2] VRP harvest mean AND tail by regime ----------
    print("\n[2] SHORT-VOL P&L = VIX - realized21 (variance-swap proxy). Mean is the carry; tail is the risk.")
    vrp = (vix - rv_fwd).rename("vrp")                      # >0 = short vol profits
    d2 = pd.DataFrame({"vrp": vrp, "c": calm}).dropna(); d2["c"] = d2["c"].astype(bool)
    real, bp = block_boot_mean_gap(d2["vrp"], d2["c"])
    def desc(x):
        return (x.mean(), x.mean()/x.std()*math.sqrt(12), np.percentile(x, 5), x.min(), float((x < -5).mean()))
    print("    %-10s %7s %7s %7s %7s %9s" % ("regime", "mean", "IR(mo)", "CVaR5", "worst", "P(<-5pt)"))
    for lbl, x in [("calm", d2[d2.c]["vrp"]), ("turbulent", d2[~d2.c]["vrp"])]:
        m, ir, c5, w, pl = desc(x)
        print("    %-10s %+7.2f %+7.2f %+7.2f %+7.1f %8.0f%%" % (lbl, m, ir, c5, w, 100*pl))
    print("    calm-minus-turb mean VRP gap = %+.2f vol pts  (block-bootstrap P=%.3f)" % (real, bp))
    print("    -> if calm mean is HIGHER and tail (CVaR5/worst) NOT worse, short-vol-in-calm harvests more for no extra crash.")

    # ---------- term structure ----------
    print("\n[2b] VIX TERM STRUCTURE slope (VIX3M - VIX): is calm more complacently in contango?")
    slope = (ts["VIX3M"] - ts["VIX"]).reindex(cal).ffill()
    d3 = pd.DataFrame({"sl": slope, "c": calm}).dropna(); d3["c"] = d3["c"].astype(bool)
    print("    calm:  slope %+.2f  contango %.0f%% of days | turb: slope %+.2f  contango %.0f%%"
          % (d3[d3.c]["sl"].mean(), 100*(d3[d3.c]["sl"] > 0).mean(),
             d3[~d3.c]["sl"].mean(), 100*(d3[~d3.c]["sl"] > 0).mean()))

    # ---------- a regime-gated short-vol curve (proxy) ----------
    print("\n[3] REGIME-GATED short-vol carry (proxy, NOT futures): monthly non-overlapping VIX-RV")
    mvix = vix.resample("21D").first(); mrvf = rv_fwd.resample("21D").first(); mcalm = calm.resample("21D").first()
    pnl = (mvix - mrvf).dropna()                            # per-month short-vol P&L proxy (vol pts)
    mc = mcalm.reindex(pnl.index).astype("float")
    always = pnl; gated = pnl.where(mc.astype(bool), 0.0)   # gated: only short vol in calm, flat otherwise
    def ann(s): s = s.dropna(); return s.mean()*12, s.mean()/s.std()*math.sqrt(12), s.min(), (s.cumsum().cummax()-s.cumsum()).max()
    for lbl, s in [("always short vol", always), ("short vol in CALM only", gated)]:
        a, ir, w, mdd = ann(s)
        print("    %-24s ann %+6.1f pts  IR %+.2f  worst month %+.1f  max cum-drawdown %.1f pts" % (lbl, a, ir, w, mdd))

    # ---------- chart ----------
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    ax[0].hist(d2[d2.c]["vrp"], bins=40, alpha=.6, color="#1f9d55", density=True, label="calm")
    ax[0].hist(d2[~d2.c]["vrp"], bins=40, alpha=.6, color="#c0392b", density=True, label="turbulent")
    ax[0].axvline(0, color="k", lw=.8); ax[0].legend(fontsize=8)
    ax[0].set_title("Short-vol P&L (VIX-realized) by regime"); ax[0].set_xlabel("vol pts (>0 = short vol wins)")
    ax[1].plot(always.cumsum().index, always.cumsum().values, color="#888", lw=1.2, label="always short vol")
    ax[1].plot(gated.cumsum().index, gated.cumsum().values, color="#1f9d55", lw=1.4, label="short vol in calm only")
    ax[1].legend(fontsize=8); ax[1].set_title("Cumulative short-vol carry (proxy)"); ax[1].set_ylabel("cum vol pts")
    fig.tight_layout(); fig.savefig(f"{OUT}/gpr_equity_vol.png", dpi=120)
    print("\nSaved: %s/gpr_equity_vol.png" % OUT)


if __name__ == "__main__":
    main()
