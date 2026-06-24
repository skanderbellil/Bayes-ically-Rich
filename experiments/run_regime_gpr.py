#!/usr/bin/env python3
"""
GPR-regime test for the calibration effect (domain-matched, no lookahead)
=========================================================================

REGIME_CALIBRATION.md found that cheap *disruptive* YES legs are underpriced in
calm regimes, an effect carried almost entirely by **geopolitics**. The two
regime measures used so far are imperfect for that domain: the surprise intensity
``S`` is endogenous (built from the panel's own resolutions), and VIX is a
*financial* risk index, near-blind to geopolitical calm. This script re-tests the
geopolitics legs against the **Caldara-Iacoviello Geopolitical Risk (GPR) index**
— an exogenous, daily, news-based measure built precisely for this regime — and
puts a number on the confidence question via a robustness grid + a placebo test.

Behavioural mapping: *recent shock* = elevated trailing-window GPR (geopolitical
availability bias high); *calm* = an extended low-GPR stretch. Hypothesis (same as
before): CE = outcome - price is larger (more underpricing) in GPR-calm regimes.

No-lookahead guarantees (every regime quantity uses only GPR dated <= decision):
  * The trailing feature is a backward rolling window of daily GPR ending on/before
    the decision date (pandas .rolling is causal: row t sees only rows <= t).
  * It is attached to each leg with merge_asof(direction="backward"), i.e. the most
    recent GPR observation at or before decision_date — never a future value.
  * The CALM/TURBULENT threshold is a **trailing 365-day median** of GPR computed
    from history strictly up to each decision date (causal, and relative to the
    *recent* normal rather than the 1985 baseline — GPR is structurally elevated in
    2023-26, so an all-history reference would label every recent day "turbulent").
    An in-sample-median split is also printed for comparability with the VIX
    analysis, clearly labelled as using a cross-sectional (not time-series) ref.

Usage:
  python experiments/run_regime_gpr.py
"""
from __future__ import annotations
import math
import numpy as np, pandas as pd
import _bootstrap  # noqa

PANEL = "data/polymarket_calibration_snapshot/regime_calibration_panel.csv"
GPR = "datasets/gpr_daily.csv.gz"
DISRUPTIVE = {"geopolitics", "crypto/financial", "politics", "ai/tech"}
RNG = np.random.default_rng(20260624)


def tstat(x):
    x = np.asarray(x, float)
    return x.mean() / (x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 and x.std() > 0 else np.nan


def gpr_daily() -> pd.DataFrame:
    """Daily GPR on a continuous calendar (ffill)."""
    g = pd.read_csv(GPR, parse_dates=["date"]).sort_values("date")
    g = g.dropna(subset=["GPRD"]).drop_duplicates("date").set_index("date")
    g = g.asfreq("D").ffill()  # carry last known reading over weekends/gaps
    return g


def attach_regime(P: pd.DataFrame, g: pd.DataFrame, window: int, ref_days: int = 365) -> pd.DataFrame:
    """Trailing-`window` GPR mean vs a trailing-`ref_days` median baseline, both causal
    (pandas .rolling sees only rows <= t) and attached with a backward merge_asof. Calm =
    recent GPR below its own running normal (a moving-average-crossover regime)."""
    trail = g["GPRD"].rolling(window, min_periods=max(5, window // 3)).mean()
    ref = g["GPRD"].rolling(ref_days, min_periods=ref_days // 2).median()
    daily = pd.DataFrame({"Date": g.index, "gpr_trail": trail.values,
                          "gpr_ref": ref.values}).dropna(subset=["gpr_trail", "gpr_ref"])
    out = pd.merge_asof(P.sort_values("decision_date"), daily,
                        left_on="decision_date", right_on="Date",
                        direction="backward").drop(columns="Date")
    return out


def ce_split(sub: pd.DataFrame, calm_mask: pd.Series):
    c, t = sub[calm_mask], sub[~calm_mask]
    return (len(c), c.CE.mean(), tstat(c.CE)), (len(t), t.CE.mean(), tstat(t.CE))


def main():
    P = pd.read_csv(PANEL, parse_dates=["decision_date", "res_date"]).sort_values("decision_date")
    P["CE"] = P["outcome"] - P["price"]
    g = gpr_daily()
    print("GPR coverage: %s -> %s (%d daily obs)" % (g.index.min().date(), g.index.max().date(), len(g)))

    # ---------- headline: geopolitics tail, baseline 45d mean vs trailing-1yr median ----------
    base = attach_regime(P, g, window=45, ref_days=365)
    geo = base[(base.domain == "geopolitics") & (base.price < 0.35) & base.gpr_trail.notna()].copy()
    cov = base["gpr_trail"].notna().mean()
    print("GPR feature coverage of panel decisions: %.0f%%   geopolitics tail n=%d\n" % (100 * cov, len(geo)))

    geo["calm_causal"] = geo["gpr_trail"] <= geo["gpr_ref"]   # causal trailing-1yr-median threshold
    geo["calm_insample"] = geo["gpr_trail"] <= geo["gpr_trail"].median()  # cross-sectional (labelled)

    print("=" * 74)
    print("HEADLINE — geopolitics tail (price<0.35), trailing-45d mean GPR")
    print("=" * 74)
    for name, col in [("CAUSAL trailing-1yr-median split (no lookahead)", "calm_causal"),
                      ("in-sample-median split (cross-sectional ref)", "calm_insample")]:
        (nc, cec, tc), (nt, cet, tt) = ce_split(geo, geo[col])
        print("\n[%s]" % name)
        print("  CALM      n=%3d  CE=%+.4f (t=%+.2f)" % (nc, cec, tc))
        print("  TURBULENT n=%3d  CE=%+.4f (t=%+.2f)" % (nt, cet, tt))
        print("  diff(calm-turb) = %+.4f   (hypothesis wants > 0)" % (cec - cet))

    # ---------- robustness grid ----------
    print("\n" + "=" * 74)
    print("ROBUSTNESS GRID — diff(calm-turb) CE on geopolitics tail, all causal")
    print("  trailing-mean window (rows) x trailing-median reference (cols)")
    print("  each cell: diff  (calm t / turb t, n_calm-n_turb)")
    print("=" * 74)
    refs = (365, 730)
    print("  %-8s %-28s %-28s" % ("window", "ref=365d", "ref=730d"))
    grid = {}
    for w in (30, 45, 60, 90):
        cells = []
        for rd in refs:
            b = attach_regime(P, g, window=w, ref_days=rd)
            sub = b[(b.domain == "geopolitics") & (b.price < 0.35) & b.gpr_trail.notna()].copy()
            calm = sub["gpr_trail"] <= sub["gpr_ref"]
            (nc, cec, tc), (nt, cet, tt) = ce_split(sub, calm)
            grid[(w, rd)] = (cec - cet, tc, tt)
            cells.append("%+.3f (%+.1f/%+.1f, %d-%d)" % (cec - cet, tc, tt, nc, nt))
        print("  %-8d %-28s %-28s" % (w, cells[0], cells[1]))
    diffs = [v[0] for v in grid.values()]
    print("\n  grid summary: %d/%d cells positive; diff range [%+.3f, %+.3f], median %+.3f"
          % (sum(d > 0 for d in diffs), len(diffs), min(diffs), max(diffs), float(np.median(diffs))))

    # ---------- placebo: shuffle the calm flag ----------
    print("\n" + "=" * 74)
    print("PLACEBO — random regime labels (geopolitics tail, 45d-mean, causal split)")
    print("=" * 74)
    real_calm = geo["calm_causal"].values
    ce = geo["CE"].values
    real_diff = ce[real_calm].mean() - ce[~real_calm].mean()
    k = int(real_calm.sum())
    n = len(ce)
    N = 5000
    null = np.empty(N)
    for i in range(N):
        idx = RNG.permutation(n)
        cm = np.zeros(n, bool); cm[idx[:k]] = True
        null[i] = ce[cm].mean() - ce[~cm].mean()
    p_two = float((np.abs(null) >= abs(real_diff)).mean())
    p_one = float((null >= real_diff).mean())
    print("  real diff(calm-turb) = %+.4f   (k=%d calm of n=%d)" % (real_diff, k, n))
    print("  placebo null: mean %+.4f  sd %.4f  95%%[%.3f, %.3f]"
          % (null.mean(), null.std(), np.quantile(null, .025), np.quantile(null, .975)))
    print("  p(placebo >= real, one-sided) = %.4f   p(|placebo| >= |real|) = %.4f" % (p_one, p_two))

    # ---------- continuous dose-response ----------
    print("\n" + "=" * 74)
    print("CONTINUOUS — OLS CE ~ trailing-45d-mean GPR (geopolitics tail). Hyp: NEGATIVE")
    print("=" * 74)
    x = geo["gpr_trail"].values.astype(float)
    x = x - x.mean()
    y = geo["CE"].values.astype(float)
    b1 = np.sum(x * y) / np.sum(x * x)
    resid = y - (y.mean() + b1 * x)
    se = math.sqrt((resid @ resid) / (n - 2) / np.sum(x * x))
    print("  dCE/dGPR = %+.5f per GPR-pt (t=%+.2f)   [negative => more GPR, less underpricing]"
          % (b1, b1 / se))

    # ---------- secondary: all-disruptive (GPR is a geopolitics measure; weaker fit) ----------
    alld = base[base.domain.isin(DISRUPTIVE) & (base.price < 0.35) & base.gpr_trail.notna()].copy()
    alld["calm"] = alld["gpr_trail"] <= alld["gpr_ref"]
    (nc, cec, tc), (nt, cet, tt) = ce_split(alld, alld["calm"])
    print("\n[secondary] all-disruptive tail under GPR-calm (GPR is geo-specific, expect weaker):")
    print("  CALM n=%d CE=%+.4f(t=%+.2f) | TURB n=%d CE=%+.4f(t=%+.2f) | diff %+.4f"
          % (nc, cec, tc, nt, cet, tt, cec - cet))


if __name__ == "__main__":
    main()
