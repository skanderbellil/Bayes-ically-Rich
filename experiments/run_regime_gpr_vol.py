#!/usr/bin/env python3
"""
GPR volatility & GPR+VIX composite regimes for the geo-calm edge (no lookahead)
===============================================================================

The deployed gate is GPR *level* (trailing-45d mean ≤ trailing-1yr median). Two
ideas to test here, both fully causal:

  (A) GPR + VIX composite — does adding financial-stress sharpen the split?
  (B) GPR *volatility* instead of level — maybe mispricing tracks the *instability*
      of geopolitical risk, not its level. A stable-but-elevated GPR (a grinding
      war already priced) is "calm"; a jumpy GPR (shocks arriving) is turbulence.
      We also FORECAST GPR vol with a causal RiskMetrics EWMA (σ²_t = λσ²_{t-1} +
      (1-λ)r²_t, λ=0.94) and gate on the forecast.

All regime metrics are built from data dated ≤ the decision date (backward rolling /
ewm + merge_asof backward); CALM = the metric below its own trailing-365d median.
Evaluated on the geopolitics cheap-YES tail (price≤0.35) by calibration error
CE=outcome−price (n≈117 — more discriminating power than the 15-trade book), then
the deployable book (hold 2–45 d) is shown for the level baseline vs the best signal.

HONESTY: this compares ~8 regime definitions on one segment — a multiple-comparison
exercise. All are reported (not just the winner); treat a single best cell as
suggestive, the pattern across signals as the result.

Usage: python experiments/run_regime_gpr_vol.py
"""
from __future__ import annotations
import math
import numpy as np, pandas as pd
import _bootstrap  # noqa

PANEL = "data/polymarket_calibration_snapshot/regime_calibration_panel.csv"


def tstat(x):
    x = np.asarray(x, float)
    return x.mean() / (x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 and x.std() > 0 else np.nan


def gpr_daily():
    g = pd.read_csv("datasets/gpr_daily.csv.gz", parse_dates=["date"]).set_index("date")["GPRD"].sort_index()
    return g.asfreq("D").ffill()


def vix_daily():
    v = pd.read_csv("datasets/fred_macro_plus.csv", parse_dates=["date"])[["date", "VIXCLS"]].dropna()
    return v.set_index("date")["VIXCLS"].sort_index().asfreq("D").ffill()


def calm_from_metric(metric: pd.Series, ref_days=365):
    """Causal CALM flag: metric (higher=more turbulent) below its trailing median."""
    base = metric.rolling(ref_days, min_periods=ref_days // 2).median()
    return (metric <= base)


def build_signals():
    """Return dict name -> daily boolean CALM series (all causal)."""
    g, v = gpr_daily(), vix_daily()
    sig = {}

    # (baseline) GPR level: trailing-45d mean
    g_lvl = g.rolling(45, min_periods=15).mean()
    sig["gpr_level"] = calm_from_metric(g_lvl)

    # GPR realized volatility: trailing-45d std of daily GPR
    g_vol = g.diff().rolling(45, min_periods=15).std()
    sig["gpr_vol"] = calm_from_metric(g_vol)

    # GPR forecast vol: RiskMetrics EWMA (λ=0.94), causal 1-step-ahead
    r = g.diff()
    ewma_var = r.pow(2).ewm(alpha=1 - 0.94, adjust=False).mean()
    sig["gpr_vol_ewma"] = calm_from_metric(np.sqrt(ewma_var))

    # VIX level: trailing-45d mean
    v_lvl = v.rolling(45, min_periods=15).mean()
    sig["vix_level"] = calm_from_metric(v_lvl)

    # composites of the two level flags
    sig["gpr_AND_vix"] = sig["gpr_level"] & sig["vix_level"]
    sig["gpr_OR_vix"] = sig["gpr_level"] | sig["vix_level"]

    # z-score composite: causal expanding standardization, averaged, calm = below trailing median
    def zc(s):
        mu = s.expanding(min_periods=250).mean(); sd = s.expanding(min_periods=250).std()
        return (s - mu) / sd
    combo = (zc(g_lvl) + zc(v_lvl)) / 2
    sig["gpr+vix_z"] = calm_from_metric(combo)

    # GPR vol + VIX vol composite (both instability)
    v_vol = v.diff().rolling(45, min_periods=15).std()
    sig["gprvol_AND_vixvol"] = calm_from_metric(g_vol) & calm_from_metric(v_vol)

    return sig


def attach(P, calm_series, name):
    daily = pd.DataFrame({"Date": calm_series.index, name: calm_series.values}).dropna()
    return pd.merge_asof(P.sort_values("decision_date"), daily,
                         left_on="decision_date", right_on="Date", direction="backward").drop(columns="Date")


def evaluate(tail, name):
    sub = tail.dropna(subset=[name])
    calm, turb = sub[sub[name]], sub[~sub[name]]
    ce_c, ce_t = calm["CE"], turb["CE"]
    return dict(signal=name, n_calm=len(calm), calm_CE=ce_c.mean(), calm_t=tstat(ce_c),
                n_turb=len(turb), turb_CE=ce_t.mean(), turb_t=tstat(ce_t),
                gap=ce_c.mean() - ce_t.mean())


def main():
    P = pd.read_csv(PANEL, parse_dates=["decision_date", "res_date"])
    P["CE"] = P["outcome"] - P["price"]
    geo = P[(P.domain == "geopolitics") & (P.leg == "yes") & (P.price <= 0.35)].copy()

    sigs = build_signals()
    T = geo.copy()
    for name, s in sigs.items():
        T = attach(T, s, name)

    print("=" * 100)
    print("REGIME-SIGNAL COMPARISON — geopolitics cheap-YES tail (price<=0.35), CE=outcome-price")
    print("  CALM = signal below its trailing-1yr median (all causal). Hypothesis: calm_CE >> turb_CE, gap>0")
    print("=" * 100)
    print("  %-20s %5s %8s %6s | %5s %8s %6s | %7s" %
          ("signal", "nCalm", "calmCE", "t", "nTurb", "turbCE", "t", "GAP"))
    rows = [evaluate(T, n) for n in sigs]
    for r in sorted(rows, key=lambda x: -x["gap"]):
        print("  %-20s %5d %+8.4f %+6.2f | %5d %+8.4f %+6.2f | %+7.4f" %
              (r["signal"], r["n_calm"], r["calm_CE"], r["calm_t"], r["n_turb"],
               r["turb_CE"], r["turb_t"], r["gap"]))

    # agreement of each signal with the deployed level gate
    print("\n[AGREEMENT with deployed gpr_level gate]")
    base = T["gpr_level"]
    for n in sigs:
        if n == "gpr_level":
            continue
        ok = T.dropna(subset=[n, "gpr_level"])
        agree = (ok[n] == ok["gpr_level"]).mean()
        print("  %-20s agree %.0f%%" % (n, 100 * agree))

    # deployable book: level baseline vs gpr_vol vs forecast vol (net of 3c, hold 2-45d)
    print("\n[DEPLOYABLE BOOK] geo tail, hold 2-45d, net edge after 3c entry spread")
    hold = (T["res_date"] - T["decision_date"]).dt.days
    book = T[hold.between(2, 45)].copy()
    for n in ["gpr_level", "gpr_vol", "gpr_vol_ewma", "gpr_AND_vix", "gpr+vix_z"]:
        b = book.dropna(subset=[n]); cb = b[b[n]]
        net = cb["outcome"] - (cb["price"] + 0.03).clip(lower=0.01)
        print("  %-20s n_calm=%2d  net edge/$1=%+.4f (t=%+.2f)  win=%.2f" %
              (n, len(cb), net.mean() if len(cb) else float("nan"),
               tstat(cb["outcome"] - cb["price"]) if len(cb) > 1 else float("nan"),
               cb["outcome"].mean() if len(cb) else float("nan")))

    print("\nNote: ~8 signals on one segment — read the pattern, not one best cell. "
          "Level and vol largely agree (high GPR periods are also volatile), so the question is "
          "whether vol/forecast-vol cleans up the residual cases; see the gap column + agreement.")


if __name__ == "__main__":
    main()
