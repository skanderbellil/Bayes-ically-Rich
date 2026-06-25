#!/usr/bin/env python3
"""
Is "disruption" underpriced in GPR-calm regimes? — faithful cross-asset analog
==============================================================================

The prediction-market edge is NOT "calm -> asset goes up". It is "in calm regimes
the market UNDERPRICES disruption" — cheap disruptive YES contracts settle true
more often than their price implies. To look for the SAME mechanism in real assets
we need each domain's own "price of disruption" and its realized "outcome":

  domain                price of disruption        realized outcome              underpricing M_t (calm should be high)
  --------------------  -------------------------  ----------------------------  --------------------------------------
  equities              implied vol  (VIX)         realized fwd vol of SPY       M = realized_fwd_vol - VIX   (VIX too low)
  credit                HY OAS spread (tight=cheap)forward spread WIDENING        M = OAS_{t+H} - OAS_t        (spread too tight)
  commodities / gold    (no clean implied)         forward return to the hedge    M = hedge fwd return         (hedge too cheap)

For each domain we build a daily underpricing series M_t (higher = disruption more
underpriced), attach the SAME causal level-OR-vol GPR calm flag (lagged 1d), and ask:

    is mean(M | calm)  >  mean(M | turbulent) ?   <-- the calm-underpricing claim

Significance: M_t over a forward window is heavily overlapping, so naive t-stats lie.
We use a stationary block bootstrap (block = the forward horizon) that resamples
joint (M, calm) blocks and reshuffles the calm label under the null of no gap.

CONFOUNDS WE NAME OUT LOUD:
  * credit/vol mean-revert: spreads/VIX that are LOW now mechanically have more room
    to rise. Calm correlates with low levels, so part of any credit/equity gap is
    level mean-reversion, not a GPR-specific edge. We therefore also report the gap
    CONTROLLING for the starting level (within-tercile of VIX / OAS), which is the
    honest test of whether GPR adds anything beyond "things are quiet right now".

Usage:
  python experiments/run_gpr_disruption_pricing.py
  python experiments/run_gpr_disruption_pricing.py --h 21 --h2 63
"""
from __future__ import annotations
import argparse, math
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import _bootstrap  # noqa

GPR = "datasets/gpr_daily.csv.gz"
ETF = "datasets/etf_universe_prices.csv.gz"
MAC = "datasets/fred_macro_plus.csv"
OUT = "data/cross_asset"
RNG = np.random.default_rng(20260625)


def tstat(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return x.mean() / (x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 and x.std() > 0 else np.nan


def gpr_calm_flag():
    g = pd.read_csv(GPR, parse_dates=["date"]).set_index("date")["GPRD"].sort_index().asfreq("D").ffill()
    lvl = g.rolling(45, min_periods=15).mean()
    lvl_calm = lvl <= lvl.rolling(365, min_periods=180).median()
    vol = g.diff().rolling(45, min_periods=15).std()
    vol_calm = vol <= vol.rolling(365, min_periods=180).median()
    return (lvl_calm | vol_calm).rename("calm")


def block_boot_gap(M, calm, n=5000, block=21):
    """One-sided block-bootstrap p for mean(M|calm) - mean(M|turb) being >0 by chance.
    Resamples `block`-length joint blocks, then permutes the calm label within the draw."""
    m = pd.DataFrame({"M": M, "calm": calm}).dropna()
    r = m["M"].values; c = m["calm"].values.astype(bool); N = len(r)
    if c.sum() < 30 or (~c).sum() < 30 or N // block < 5:
        return np.nan, np.nan
    real = r[c].mean() - r[~c].mean()
    nb = N // block; starts = np.arange(N - block)
    null = np.empty(n)
    for i in range(n):
        idx = (RNG.choice(starts, size=nb, replace=True)[:, None] + np.arange(block)).ravel()
        rr = r[idx]; cc = c[idx].copy(); RNG.shuffle(cc)
        null[i] = 0.0 if cc.all() or (~cc).all() else rr[cc].mean() - rr[~cc].mean()
    return real, float((null >= real).mean())


def level_controlled_gap(M, calm, level, q=3):
    """Mean calm-minus-turb gap of M, averaged WITHIN terciles of the starting level
    (VIX or OAS). Removes the 'low level mean-reverts' confound: does GPR-calm still
    add underpricing once we hold 'how quiet is it right now' fixed?"""
    d = pd.DataFrame({"M": M, "calm": calm.astype(float), "lev": level}).dropna()
    if len(d) < 100:
        return np.nan
    d["bin"] = pd.qcut(d["lev"], q, labels=False, duplicates="drop")
    gaps = []
    for _, g in d.groupby("bin"):
        cc = g[g.calm == 1]["M"]; tt = g[g.calm == 0]["M"]
        if len(cc) >= 15 and len(tt) >= 15:
            gaps.append(cc.mean() - tt.mean())
    return float(np.mean(gaps)) if gaps else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h", type=int, default=21, help="primary forward horizon (trading days)")
    ap.add_argument("--h2", type=int, default=63, help="secondary forward horizon (trading days)")
    args = ap.parse_args()
    import os; os.makedirs(OUT, exist_ok=True)
    H, H2 = args.h, args.h2

    flag = gpr_calm_flag()
    etf = pd.read_csv(ETF, parse_dates=["Date"]).set_index("Date").sort_index()
    mac = pd.read_csv(MAC, parse_dates=["date"]).set_index("date").sort_index()
    spy = etf["SPY"].dropna()
    cal = spy.index  # trading-day calendar
    calm_d = flag.reindex(cal).ffill().shift(1)  # lag 1d -> causal

    def mac_on_cal(col):
        return mac[col].reindex(cal.union(mac.index)).ffill().reindex(cal)

    # ---------------- EQUITIES: realized fwd vol  vs  implied (VIX) ----------------
    logret = np.log(spy).diff()
    rv_fwd = logret.rolling(H).std().shift(-(H - 1)) * math.sqrt(252) * 100   # ann. % realized over next H
    rv_fwd2 = logret.rolling(H2).std().shift(-(H2 - 1)) * math.sqrt(252) * 100
    vix = mac_on_cal("VIXCLS")
    M_eq = (rv_fwd - vix)                       # >0 : realized exceeded implied -> vol under-priced
    M_eq2 = (rv_fwd2 - vix)
    vrp = (vix - rv_fwd)                        # the variance risk premium itself

    # ---------------- CREDIT: forward HY-OAS widening ----------------
    hy = mac_on_cal("BAMLH0A0HYM2"); ig = mac_on_cal("BAMLC0A0CM")
    M_cr = hy.shift(-H) - hy                    # >0 : spreads widened after -> default risk under-priced
    M_cr2 = hy.shift(-H2) - hy
    M_ig = ig.shift(-H) - ig

    # ---------------- COMMODITIES / GOLD / HEDGES: forward return to the hedge ----------------
    def fwd_ret(sym):
        s = etf[sym].dropna().reindex(cal).ffill()
        return s.shift(-H) / s - 1.0
    hedges = {"Gold (GLD)": "GLD", "Gold miners (GDX)": "GDX", "Silver (SLV)": "SLV",
              "Long Tsy (TLT)": "TLT", "Energy (XLE)": "XLE", "Broad commod (DBC)": "DBC"}
    M_hedge = {k: fwd_ret(v) for k, v in hedges.items()}
    # long-vol P&L proxy: forward CHANGE in VIX (being long vol pays when VIX rises)
    M_longvol = vix.shift(-H) - vix

    print("=" * 110)
    print("IS DISRUPTION UNDERPRICED IN GPR-CALM? — faithful cross-asset mechanism test  (H=%dd, H2=%dd)" % (H, H2))
    print("  flag = same causal level-OR-vol GPR gate (lagged 1d). Claim: mean(M | calm) > mean(M | turbulent).")
    print("  M = how UNDERPRICED disruption was (realized minus implied / forward widening / hedge return).")
    print("=" * 110)

    def report(label, M, level=None, block=H, unit="", scale=1.0):
        c = calm_d.astype("float")
        d = pd.DataFrame({"M": M.reindex(cal) * scale, "calm": c}).dropna()
        d["calm"] = d["calm"].astype(bool)
        cc, tt = d[d.calm]["M"], d[~d.calm]["M"]
        real, bp = block_boot_gap(d["M"], d["calm"], block=block)
        lc = level_controlled_gap(d["M"], d["calm"], level.reindex(d.index)) if level is not None else np.nan
        star = "*" if (bp == bp and bp <= 0.10) else " "
        print("  %-30s calm=%+8.3f%s  turb=%+8.3f%s  | GAP=%+8.3f  bootP=%5.3f%s  level-ctrl GAP=%s"
              % (label, cc.mean(), unit, tt.mean(), unit, cc.mean() - tt.mean(), bp if bp == bp else float('nan'),
                 star, ("%+8.3f" % lc) if lc == lc else "   n/a "))
        return dict(label=label, calm=cc.mean(), turb=tt.mean(), gap=cc.mean() - tt.mean(),
                    bootP=bp, level_ctrl=lc, n=len(d))

    rows = []
    print("\n[EQUITIES] price of disruption = implied vol (VIX); outcome = realized fwd vol of SPY")
    print("  positive M = realized vol BEAT implied (VIX was too low) -> disruption under-priced")
    rows.append(report("realized%d - VIX (vol pts)" % H, M_eq, level=vix, block=H, unit=""))
    rows.append(report("realized%d - VIX (vol pts)" % H2, M_eq2, level=vix, block=H2, unit=""))
    rows.append(report("long-vol P&L: dVIX over %dd" % H, M_longvol, level=vix, block=H, unit=""))
    print("  (context) variance risk premium VIX-realized%d: calm=%+.2f turb=%+.2f  (smaller premium = vol cheaper in calm)"
          % (H, vrp.reindex(cal)[calm_d.fillna(False).astype(bool).reindex(cal).values].mean(),
             vrp.reindex(cal)[~calm_d.fillna(True).astype(bool).reindex(cal).values].mean()))
    # TAIL version the table asks for: is the realized CRASH worse after calm? (disruption under-anticipated)
    fwd_mdd = pd.Series(
        [(spy.iloc[i:i + H] / spy.iloc[i] - 1.0).min() if i + H <= len(spy) else np.nan for i in range(len(spy))],
        index=cal) * 100                                   # forward H-day max drawdown (%)
    fwd_minday = logret.rolling(H).min().shift(-(H - 1)) * 100   # worst single-day forward ret (%)
    print("  [TAIL] forward %dd drawdown / worst-day — claim: crashes BIGGER (more negative) after calm" % H)
    rows.append(report("fwd %dd maxDD (neg=worse)" % H, -fwd_mdd, level=vix, block=H, unit="", scale=1.0))
    rows.append(report("fwd %dd worst-day (neg=worse)" % H, -fwd_minday, level=vix, block=H, unit="", scale=1.0))
    print("       (M = -drawdown, so GAP>0 would mean deeper crashes in calm = tail under-priced)")

    print("\n[CREDIT] price of disruption = HY OAS (tight = cheap default risk); outcome = forward spread widening")
    print("  positive M = spreads WIDENED afterward (were too tight) -> default risk under-priced")
    rows.append(report("HY OAS widening over %dd (bps)" % H, M_cr, level=hy, block=H, unit="", scale=100))
    rows.append(report("HY OAS widening over %dd (bps)" % H2, M_cr2, level=hy, block=H2, unit="", scale=100))
    rows.append(report("IG OAS widening over %dd (bps)" % H, M_ig, level=ig, block=H, unit="", scale=100))

    print("\n[COMMODITIES / GOLD / HEDGES] no clean implied; outcome = forward %dd return to the hedge" % H)
    print("  positive M = hedge paid off forward -> hedge was cheap in calm (geopolitical insurance under-priced)")
    for k, M in M_hedge.items():
        rows.append(report("%s fwd %dd ret" % (k, H), M, level=None, block=H, unit="%", scale=100))

    # ---------------- verdict ----------------
    print("\n" + "=" * 110)
    print("VERDICT — does the calm-underpricing pattern (GAP>0) carry to real assets?")
    eq = [r for r in rows if "VIX" in r["label"] and "realized" in r["label"]]
    cr = [r for r in rows if "OAS widening" in r["label"]]
    hg = [r for r in rows if "fwd" in r["label"] and "ret" in r["label"]]
    def summ(name, g):
        pos = np.mean([r["gap"] > 0 for r in g]); sig = sum(r["bootP"] == r["bootP"] and r["bootP"] <= 0.10 for r in g)
        lc_ok = [r for r in g if r["level_ctrl"] == r["level_ctrl"]]
        lc_pos = np.mean([r["level_ctrl"] > 0 for r in lc_ok]) if lc_ok else float("nan")
        print("  %-26s GAP>0 in %.0f%% of %d tests | %d boot-sig | survives level-control in %.0f%%"
              % (name, 100 * pos, len(g), sig, 100 * lc_pos))
    summ("EQUITIES (VIX too low)", eq)
    summ("CREDIT (spreads too tight)", cr)
    summ("COMMODITIES/GOLD (cheap hedge)", hg)
    print("=" * 110)
    print("Read: a coherent edge shows GAP>0 AND survival of the level control (GPR adds something beyond 'it's quiet now').")

    # ---------------- chart ----------------
    fig, ax = plt.subplots(figsize=(10.5, 7))
    rs = sorted(rows, key=lambda r: r["gap"] / (abs(r["gap"]) + 1e-9) * (r["bootP"] if r["bootP"] == r["bootP"] else 1))
    rs = sorted(rows, key=lambda r: r["gap"])
    # normalize gaps to z-ish display by group is messy; instead show sign & boot significance
    y = np.arange(len(rs))
    colors = ["#2ca02c" if r["gap"] > 0 else "#c0392b" for r in rs]
    ax.barh(y, [r["gap"] for r in rs], color=colors,
            edgecolor=["black" if (r["bootP"] == r["bootP"] and r["bootP"] <= 0.10) else "none" for r in rs], linewidth=1.4)
    ax.set_yticks(y); ax.set_yticklabels([r["label"] for r in rs], fontsize=8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("calm − turbulent  underpricing gap  (vol pts / bps / %ret — mixed units, sign is the message)")
    ax.set_title("Disruption under-pricing in GPR-calm vs turbulent\n"
                 "green = under-priced more in calm (claim direction); black outline = block-bootstrap p<=0.10", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{OUT}/gpr_disruption_pricing.png", dpi=120)
    pd.DataFrame(rows).to_csv(f"{OUT}/gpr_disruption_pricing.csv", index=False)
    print("\nSaved: %s/gpr_disruption_pricing.png , %s/gpr_disruption_pricing.csv" % (OUT, OUT))


if __name__ == "__main__":
    main()
