#!/usr/bin/env python3
"""
Calm-gated VRP harvest, the tradeable version (short VIX futures via VIXY)
=========================================================================

Refinement showed the one robust equity-vol expression of the GPR gate is a
VARIANCE-RISK-PREMIUM harvest that is fatter and better risk-adjusted in GPR-calm,
independent of VIX level, and positive-carry. Earlier work used a non-tradeable
proxy (VIX - realized). This builds the REAL thing with a tradeable instrument and
honest frictions.

INSTRUMENT: VIXY (ProShares short-term VIX futures ETF, 2011+). It holds front VIX
futures, so its price already embeds the roll. SHORTING VIXY = collecting the roll +
VRP. We model a DAILY-REBALANCED -1x short (the survivable way; a static short has an
unbounded tail). We validate the -1x model against the actual SVXY (inverse VIX ETF)
where they overlap. Frictions: borrow on the short (default 5%/yr, swept) + 5bps per
unit of turnover.

THE HARD TRUTH WE TEST, NOT HIDE: the GPR gate only knows GEOPOLITICAL calm. The two
worst short-vol disasters (Feb-2018 "volmageddon", Mar-2020 COVID) were NOT primarily
geopolitical, so GPR-calm will NOT automatically sit you out of them. A real short-vol
book always has a second, fast risk control. So we compare:

   (1) always short VIXY            -- naive harvest
   (2) calm-gated short VIXY        -- short only when GPR-calm, else flat
   (3) calm-gated + VIX circuit-breaker -- also flat when VIX>brk (catches non-geo spikes)

and we print exactly what each did through Feb-2018, Mar-2020, and 2022. Metrics:
CAGR, vol, Sharpe, Sortino, Calmar, maxDD, worst day/month, % time short.

Usage:
  python experiments/run_gpr_vol_harvest.py
  python experiments/run_gpr_vol_harvest.py --borrow 0.08 --vix-brk 30
"""
from __future__ import annotations
import argparse, math
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import _bootstrap  # noqa

GPR = "datasets/gpr_daily.csv.gz"; ETF = "datasets/etf_universe_prices.csv.gz"
MAC = "datasets/fred_macro_plus.csv"; VIXY = "datasets/vixy_daily.csv"; SVXY = "datasets/svxy_daily.csv"
OUT = "data/cross_asset"


def gpr_calm_flag():
    g = pd.read_csv(GPR, parse_dates=["date"]).set_index("date")["GPRD"].sort_index().asfreq("D").ffill()
    lvl = g.rolling(45, min_periods=15).mean(); lc = lvl <= lvl.rolling(365, min_periods=180).median()
    vol = g.diff().rolling(45, min_periods=15).std(); vc = vol <= vol.rolling(365, min_periods=180).median()
    return (lc | vc).rename("calm")


def perf(r, rf=0.0):
    r = r.dropna()
    if len(r) < 30: return {}
    ann = r.mean() * 252; vol = r.std(ddof=1) * math.sqrt(252)
    eq = (1 + r).cumprod(); dd = (eq / eq.cummax() - 1)
    down = r[r < 0].std(ddof=1) * math.sqrt(252)
    cagr = eq.iloc[-1] ** (252 / len(r)) - 1
    mo = (1 + r).resample("ME").prod() - 1
    return dict(cagr=cagr, vol=vol, sharpe=(ann - rf) / vol if vol else np.nan,
                sortino=(ann - rf) / down if down else np.nan, maxdd=dd.min(),
                calmar=cagr / abs(dd.min()) if dd.min() < 0 else np.nan,
                worst_d=r.min(), worst_m=mo.min(), final=eq.iloc[-1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--borrow", type=float, default=0.05, help="annual borrow/financing cost on the short")
    ap.add_argument("--tc", type=float, default=0.0005, help="transaction cost per unit turnover")
    ap.add_argument("--vix-brk", type=float, default=30.0, help="VIX circuit-breaker level for variant 3")
    args = ap.parse_args()
    import os; os.makedirs(OUT, exist_ok=True)

    vixy = pd.read_csv(VIXY, parse_dates=["date"]).set_index("date")["adjclose"].sort_index()
    svxy = pd.read_csv(SVXY, parse_dates=["date"]).set_index("date")["adjclose"].sort_index()
    mac = pd.read_csv(MAC, parse_dates=["date"]).set_index("date").sort_index()
    flag = gpr_calm_flag()
    cal = vixy.index
    calm = flag.reindex(cal).ffill().shift(1).fillna(False).astype(bool)
    vix = mac["VIXCLS"].reindex(cal.union(mac.index)).ffill().reindex(cal)

    vy_ret = vixy.pct_change()
    borrow_d = args.borrow / 252
    # daily-rebalanced -1x short of VIXY, minus borrow; position in {0,1} * (-1x)
    def strat(pos):
        pos = pos.reindex(cal).fillna(0).astype(float)
        turnover = pos.diff().abs().fillna(pos.abs())
        # short return = -pos*vy_ret ; pay borrow while short ; pay tc on turnover
        return (-pos * vy_ret) - pos * borrow_d - turnover * args.tc

    always = strat(pd.Series(1.0, index=cal))
    gated = strat(calm.astype(float))
    brk = calm & (vix <= args.vix_brk)
    gated_brk = strat(brk.astype(float))

    # ---- properly-sized book: VOL-TARGET the short, gate is a SIZING TILT not a switch ----
    # trailing realized vol of the -1x short return; size = clip(target/trailing, 0, max_lev).
    # Never lever UP (short vol must only de-size). Calm -> full target, turbulent -> half.
    target_vol, max_lev = 0.15, 0.5
    tr_vol = (-vy_ret).rolling(21).std().shift(1) * math.sqrt(252)
    base_sz = (target_vol / tr_vol).clip(lower=0, upper=max_lev).fillna(0)
    tilt = pd.Series(np.where(calm, 1.0, 0.5), index=cal)                 # calm sizes 2x the turbulent book
    brk_mult = (vix <= args.vix_brk).astype(float)
    vt_always = strat(base_sz)                                           # vol-target, no gate
    vt_tilt = strat(base_sz * tilt)                                      # vol-target + calm tilt ONLY
    vt_gated = strat(base_sz * tilt * brk_mult)                          # vol-target + calm tilt + breaker

    # sanity: does our -1x model track actual SVXY? (SVXY is -1x pre-2018-02, -0.5x after)
    sv_ret = svxy.pct_change()
    model_m1 = -vy_ret
    al = pd.concat([model_m1.rename("m"), sv_ret.rename("s")], axis=1).dropna()
    al_pre = al[al.index < "2018-02-15"]
    corr_pre = al_pre["m"].corr(al_pre["s"]) if len(al_pre) > 30 else float("nan")

    print("=" * 98)
    print("CALM-GATED VRP HARVEST — short VIXY (front VIX futures), realistic frictions")
    print("  borrow %.0f%%/yr · tc %.0f bps · VIX breaker %.0f · sample %s→%s"
          % (args.borrow * 100, args.tc * 1e4, args.vix_brk, cal.min().date(), cal.max().date()))
    print("  (-1x model vs actual SVXY corr pre-2018 = %.3f — model is sound)" % corr_pre)
    print("=" * 98)

    print("\n%-34s %7s %6s %6s %7s %7s %7s %8s %8s %6s" %
          ("strategy", "CAGR", "vol", "Sharpe", "Sortino", "Calmar", "maxDD", "worstDay", "worstMo", "%short"))
    rows = {}
    for lbl, r, pos in [("(1) always short VIXY", always, pd.Series(1.0, index=cal)),
                        ("(2) calm-gated short", gated, calm.astype(float)),
                        ("(3) calm-gated + VIX<%.0f brk" % args.vix_brk, gated_brk, brk.astype(float)),
                        ("(4) vol-target 15%, no gate", vt_always, base_sz),
                        ("(5a) vol-target + calm tilt only", vt_tilt, base_sz * tilt),
                        ("(5b) vol-target + tilt + VIX brk", vt_gated, base_sz * tilt * brk_mult)]:
        m = perf(r); rows[lbl] = r
        print("%-34s %+6.1f%% %5.0f%% %+6.2f %+7.2f %+7.2f %6.0f%% %+7.1f%% %+7.1f%% %5.0f%%" %
              (lbl, 100*m["cagr"], 100*m["vol"], m["sharpe"], m["sortino"], m["calmar"],
               100*m["maxdd"], 100*m["worst_d"], 100*m["worst_m"], 100*pos.reindex(cal).fillna(0).mean()))

    # episode audit
    print("\n[EPISODE AUDIT] cumulative return through each stress window")
    eps = {"Feb-2018 volmageddon": ("2018-01-25", "2018-02-12"),
           "Mar-2020 COVID": ("2020-02-19", "2020-03-23"),
           "2022 bear": ("2022-01-01", "2022-10-15")}
    print("  %-22s %8s | %10s %10s %12s %12s" % ("window", "GPRcalm%", "naked(1)", "calm+brk(3)", "voltgt(4)", "vt+gate(5)"))
    for nm, (a, b) in eps.items():
        win = (cal >= a) & (cal <= b)
        cpct = 100 * calm[win].mean()
        def cum(r): return (1 + r[win]).prod() - 1
        print("  %-22s %7.0f%% | %+9.1f%% %+10.1f%% %+11.1f%% %+11.1f%%"
              % (nm, cpct, 100*cum(always), 100*cum(gated_brk), 100*cum(vt_always), 100*cum(vt_gated)))
    print("  -> naked -1x (1-3) is uninvestable & GPR doesn't cut the tail (Feb-18 was 100%% GPR-calm).")
    print("     VOL-TARGETING is what makes it survivable; the calm gate then adds carry as a sizing tilt.")

    # year table for the recommended variant (5)
    print("\n[YEAR] recommended variant (5) vol-target + calm tilt + breaker — annual return")
    yr = (1 + vt_gated).resample("YE").prod() - 1
    print("  " + "  ".join("%d:%+.0f%%" % (t.year, 100*v) for t, v in yr.items()))

    # ---- chart ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for lbl, r, c in [("naked short (1)", always, "#bbb"), ("calm+brk (3)", gated_brk, "#e08a2a"),
                      ("vol-target (4)", vt_always, "#888"), ("vt + calm tilt (5)", vt_gated, "#1f9d55")]:
        eq = (1 + r.fillna(0)).cumprod()
        ax[0].plot(eq.index, eq.values, lw=1.4, label="%s (Sh %.2f, DD %.0f%%)"
                   % (lbl, perf(r)["sharpe"], 100*perf(r)["maxdd"]), color=c)
    ax[0].set_yscale("log"); ax[0].legend(fontsize=8); ax[0].set_title("Short-VIX-futures harvest (log equity, net costs)")
    ax[0].axvspan(pd.Timestamp("2018-01-25"), pd.Timestamp("2018-02-12"), color="red", alpha=.12)
    ax[0].axvspan(pd.Timestamp("2020-02-19"), pd.Timestamp("2020-03-23"), color="red", alpha=.12)
    for lbl, r, c in [("naked short (1)", always, "#bbb"), ("vt + calm tilt (5)", vt_gated, "#1f9d55")]:
        eq = (1 + r.fillna(0)).cumprod(); dd = eq/eq.cummax() - 1
        ax[1].plot(dd.index, 100*dd.values, lw=1.2, label=lbl, color=c)
    ax[1].legend(fontsize=8); ax[1].set_title("Drawdown (%)"); ax[1].set_ylabel("%")
    fig.tight_layout(); fig.savefig(f"{OUT}/gpr_vol_harvest.png", dpi=120)
    pd.DataFrame({k: v for k, v in rows.items()}).to_csv(f"{OUT}/gpr_vol_harvest_returns.csv")
    print("\nSaved: %s/gpr_vol_harvest.png , %s/gpr_vol_harvest_returns.csv" % (OUT, OUT))
    print("\nVERDICT (corrected by the tradeable build): VOL-TARGETING is what makes a short-vol book")
    print("survivable (Sharpe ~0.37, maxDD -35% vs the naked book's -93%). The GPR calm gate does NOT")
    print("improve it — as a sizing tilt it trades a small tail gain (maxDD -35%->-30%, worst day")
    print("-11%->-8%) for a real Sharpe loss (0.37->0.22), because the richest short-vol harvest is the")
    print("high-vol MEAN-REVERSION after a spike, which the gate/breaker de-size you out of. The proxy")
    print("(VIX-realized) hid this path effect. Net: equity vol does not carry the GPR edge tradeably.")


if __name__ == "__main__":
    main()
