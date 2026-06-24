#!/usr/bin/env python3
"""
Exogenous-regime robustness for the calibration effect
======================================================

REGIME_CALIBRATION.md found disruptive-tail underpricing that fades after recent
shocks, but its regime measure (surprise intensity S) is built from the panel's
own longshot-YES resolutions — possibly circular. This re-tests the effect with a
**fully exogenous** regime indicator: market-wide risk (VIX) and credit stress
(BofA US HY OAS), both from the repo's local datasets and known at decision time.

Behavioural mapping: a *recent shock* = an elevated trailing-window VIX/credit
spike (availability bias high); *calm* = an extended low-VIX stretch. If the
underpricing is larger in exogenously-calm regimes too, the endogeneity caveat is
largely resolved.

Usage: python experiments/run_regime_exogenous.py
"""
from __future__ import annotations
import math
import numpy as np, pandas as pd
import _bootstrap  # noqa

PANEL = "data/polymarket_calibration_snapshot/regime_calibration_panel.csv"
DISRUPTIVE = {"geopolitics", "crypto/financial", "politics", "ai/tech"}

def tstat(x):
    x = np.asarray(x, float)
    return x.mean()/(x.std(ddof=1)/math.sqrt(len(x))) if len(x) > 1 and x.std() > 0 else np.nan

def main():
    P = pd.read_csv(PANEL, parse_dates=["decision_date", "res_date"]).sort_values("decision_date")
    P["CE"] = P["outcome"] - P["price"]

    # ---- exogenous regime features, known as-of decision_date (no lookahead) ----
    vix = pd.read_csv("datasets/vix_termstructure.csv", parse_dates=["Date"]).sort_values("Date")
    vix = vix.dropna(subset=["VIX"]).set_index("Date")["VIX"]
    vdaily = vix.resample("D").ffill()
    feat = pd.DataFrame({"vix": vdaily,
                         "vix_max45": vdaily.rolling(45, min_periods=10).max(),
                         "vix_mean45": vdaily.rolling(45, min_periods=10).mean()}).reset_index().rename(columns={"index": "Date", "Date": "Date"})
    feat.columns = ["Date", "vix", "vix_max45", "vix_mean45"]

    hy = pd.read_csv("datasets/fred_macro.csv", parse_dates=["date"]).sort_values("date")
    hy = hy[["date", "BAMLH0A0HYM2"]].dropna()
    hyd = hy.set_index("date")["BAMLH0A0HYM2"].resample("D").ffill()
    hyf = pd.DataFrame({"hy": hyd, "hy_max45": hyd.rolling(45, min_periods=10).max()}).reset_index()
    hyf.columns = ["Date", "hy", "hy_max45"]

    P = pd.merge_asof(P, feat, left_on="decision_date", right_on="Date", direction="backward").drop(columns="Date")
    P = pd.merge_asof(P, hyf, left_on="decision_date", right_on="Date", direction="backward").drop(columns="Date")
    cov = P["vix_max45"].notna().mean()
    print("VIX/credit coverage of decisions: %.0f%%  (panel %s → %s, VIX ends %s)"
          % (100*cov, P.decision_date.min().date(), P.decision_date.max().date(), vix.index.max().date()))

    D = P[P.domain.isin(DISRUPTIVE) & (P.price < 0.35) & P["vix_max45"].notna()].copy()
    # exogenous calm/turbulent: recent-shock = trailing-45d max VIX above its median
    thr = D["vix_max45"].median()
    D["calm_x"] = D["vix_max45"] <= thr
    hythr = D["hy_max45"].median()
    D["calm_hy"] = D["hy_max45"] <= hythr

    print("\n" + "="*84)
    print("EXOGENOUS-REGIME RE-TEST — disruptive tail (price<0.35, n=%d)" % len(D))
    print("="*84)

    print("\n[A] VIX regime (recent-shock = trailing-45d max VIX > median = %.1f)" % thr)
    for lbl, sub in [("CALM_vix", D[D.calm_x]), ("TURB_vix", D[~D.calm_x])]:
        print("  %-9s n=%3d  CE=%+.4f (t=%+.2f)  realized=%.3f  price=%.3f  vixmax=%.1f"
              % (lbl, len(sub), sub.CE.mean(), tstat(sub.CE), sub.outcome.mean(), sub.price.mean(), sub.vix_max45.mean()))
    print("  diff(calm-turb) = %+.4f   (hypothesis wants > 0)" % (D[D.calm_x].CE.mean()-D[~D.calm_x].CE.mean()))

    print("\n[B] Credit-stress regime (recent-shock = trailing-45d max HY OAS > median = %.2f)" % hythr)
    for lbl, sub in [("CALM_hy", D[D.calm_hy]), ("TURB_hy", D[~D.calm_hy])]:
        print("  %-8s n=%3d  CE=%+.4f (t=%+.2f)  realized=%.3f  price=%.3f"
              % (lbl, len(sub), sub.CE.mean(), tstat(sub.CE), sub.outcome.mean(), sub.price.mean()))
    print("  diff(calm-turb) = %+.4f" % (D[D.calm_hy].CE.mean()-D[~D.calm_hy].CE.mean()))

    print("\n[C] Continuous OLS: CE ~ vix_max45 (within-domain demeaned). Hypothesis: NEGATIVE")
    D["CEd"] = D.groupby("domain")["CE"].transform(lambda s: s-s.mean())
    D["xd"] = D.groupby("domain")["vix_max45"].transform(lambda s: s-s.mean())
    x, y = D["xd"].to_numpy(), D["CEd"].to_numpy()
    if x.std() > 0:
        b = np.cov(x, y, ddof=1)[0, 1]/x.var(ddof=1)
        se = np.sqrt(((y-b*x)**2).sum()/(len(x)-2))/np.sqrt(((x-x.mean())**2).sum())
        print("  dCE/d(vix_max45) = %+.5f per VIX-pt (t=%+.2f)" % (b, b/se))

    print("\n[D] Agreement: exogenous VIX-calm vs endogenous S-calm classification")
    D["calm_S"] = D.groupby("domain")["S"].transform(lambda s: s <= s.median())
    agree = (D.calm_x == D.calm_S).mean()
    print("  agree on %.0f%% of legs (corr of the two calm flags = %.2f)" % (100*agree, np.corrcoef(D.calm_x.astype(float), D.calm_S.astype(float))[0, 1]))

    print("\n[E] Per-domain CE under VIX regime")
    for dom, sub in D.groupby("domain"):
        c, t = sub[sub.calm_x], sub[~sub.calm_x]
        print("  %-17s calm n=%2d CE=%+.4f(t=%+.2f) | turb n=%2d CE=%+.4f(t=%+.2f)"
              % (dom, len(c), c.CE.mean(), tstat(c.CE), len(t), t.CE.mean(), tstat(t.CE)))

    print("\n[F] STRATEGY edge under exogenous-calm filter (buy cheap YES, calm_vix only)")
    Y = P[P.domain.isin(DISRUPTIVE) & (P.leg == "yes") & (P.price <= 0.35) & P["vix_max45"].notna()].copy()
    Y["calm_x"] = Y["vix_max45"] <= Y["vix_max45"].median()
    for lbl, sub in [("calm_vix (strategy)", Y[Y.calm_x]), ("turb_vix", Y[~Y.calm_x]), ("all", Y)]:
        ce = sub.outcome - sub.price
        print("  %-20s n=%3d  edge/$1=%+.4f (t=%+.2f)  win=%.2f" % (lbl, len(sub), ce.mean(), tstat(ce), sub.outcome.mean()))
    print("\n✓ done.")

if __name__ == "__main__":
    main()
