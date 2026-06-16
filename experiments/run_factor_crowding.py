#!/usr/bin/env python3
"""
Factor crowding on the zoo — does the correlation structure warn the EW holder?

Idea (roadmap §4b unblock; completes the OpenAP triptych)
---------------------------------------------------------
Ideas 7 and 3 established the panel's verdict: hold ALL published factors
(EW, Sharpe ~1.3 gross), don't time them by age or momentum. That makes the
EW-zoo stream a strategic holding — and its known failure mode in the
literature is CROWDING: when published factors get arbitraged by the same
capital, their correlations tighten, and deleveraging events (Aug-2007
"quant quake" style) hit them all at once. Three causal monthly gauges on
the eligible (published) universe, each from a trailing 36m window:

  avg_corr   mean pairwise correlation of factor L/S returns
  pc1_share  share of variance on the first principal component
  disp       cross-factor dispersion of trailing 12m returns (low = crowded)

Tests, pre-declared
-------------------
1. Levels: do the gauges trend (is the zoo more crowded post-2010)?
2. Conditioning: EW-ALL forward 1m return / vol / drawdown by causal
   tercile of each gauge (terciles from EXPANDING history, no lookahead).
3. Rank IC of each gauge vs EW-ALL forward 1m and 12m returns.

Honesty box
-----------
• Gross academic L/S returns; crowding in live trading shows up through
  costs and capacity first, which this panel cannot see — a null here
  does not clear live crowding risk, but a positive would be a warning.
• 3 gauges × (1m, 12m) horizons on one stream = 6 trials.
• Same survivorship/hindsight-replication caveats as Ideas 7 and 3.
"""
import logging
import sys

import numpy as np
import pandas as pd
from tabulate import tabulate

import _bootstrap  # noqa: F401
from posterioralpha.data import load_openap_doc, load_openap_returns

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)
RESULTS_DIR = _bootstrap.ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MIN_ELIGIBLE = 30
WINDOW = 36                     # months for corr / PCA
MIN_OBS_TERC = 60               # expanding-tercile warm-up


def main():
    rets = load_openap_returns()
    doc = load_openap_doc().set_index("Acronym")
    meta = doc.loc[doc.index.intersection(rets.columns), ["Year"]].astype(float)
    rets = rets[meta.index]

    age_y = rets.index.year.values[:, None] - meta.Year.values
    published = pd.DataFrame(age_y >= 1, index=rets.index, columns=rets.columns)
    elig = published & rets.notna()
    live = elig.sum(axis=1)
    live = live[live >= MIN_ELIGIBLE].index
    ew = rets.where(elig).mean(axis=1).loc[live]
    logger.info(f"{rets.shape[1]} predictors; live (≥{MIN_ELIGIBLE} eligible): "
                f"{live[0].date()} → {live[-1].date()}")

    # ── causal crowding gauges ────────────────────────────────────────────
    g = {"avg_corr": pd.Series(np.nan, live), "pc1_share": pd.Series(np.nan, live),
         "disp": pd.Series(np.nan, live)}
    mom12 = rets.rolling(12).sum()
    for t in live:
        win = rets.loc[:t].tail(WINDOW)
        cols = elig.loc[t] & (win.notna().sum() == WINDOW)
        X = win.loc[:, cols]
        if X.shape[1] < MIN_ELIGIBLE:
            continue
        C = np.corrcoef(X.values.T)
        n = C.shape[0]
        g["avg_corr"][t] = (C.sum() - n) / (n * (n - 1))
        ev = np.linalg.eigvalsh(C)
        g["pc1_share"][t] = ev[-1] / ev.sum()
        g["disp"][t] = mom12.loc[t, cols].std()
    gauges = pd.DataFrame(g).dropna()

    print("\nGauge levels by half-decade (zoo getting more crowded?):")
    lv = gauges.groupby(gauges.index.year // 5 * 5).mean()
    print(tabulate([[f"{y}-{y+4}"] + [f"{r[c]:.3f}" for c in gauges.columns]
                    for y, r in lv.iterrows()],
                   headers=["period"] + list(gauges.columns), tablefmt="github"))

    # ── conditioning: EW forward stats by causal tercile ──────────────────
    fwd1 = ew.shift(-1)
    fwd12 = ew.rolling(12).sum().shift(-12)
    rows, raw = [], []
    for name, s in gauges.items():
        # expanding terciles: rank of today's value within its own history
        erank = s.expanding(MIN_OBS_TERC).apply(
            lambda x: (x[:-1] < x[-1]).mean(), raw=True)
        terc = pd.cut(erank, [0, 1 / 3, 2 / 3, 1.0001], labels=["lo", "mid", "hi"])
        for lab in ["lo", "mid", "hi"]:
            m = (terc == lab) & fwd1.notna()
            r = fwd1[m]
            rows.append([f"{name} {lab}", int(m.sum()), f"{r.mean()*12:+.2%}",
                         f"{r.mean()/ (r.std()+1e-12) * np.sqrt(12):.2f}",
                         f"{fwd12[terc == lab].mean():+.2%}"])
            raw.append({"gauge": name, "tercile": lab, "n": int(m.sum()),
                        "fwd1_ann": r.mean() * 12,
                        "fwd1_sharpe": r.mean() / (r.std() + 1e-12) * np.sqrt(12),
                        "fwd12": fwd12[terc == lab].mean()})
        ic1 = s.corr(fwd1, method="spearman")
        ic12 = s.corr(fwd12, method="spearman")
        n_eff1, n_eff12 = s.notna().sum(), (s.notna() & fwd12.notna()).sum()
        rows.append([f"{name} rank IC", "", f"1m {ic1:+.3f} "
                     f"(t {ic1*np.sqrt(n_eff1):+.1f})",
                     f"12m {ic12:+.3f}",
                     f"(t {ic12*np.sqrt(n_eff12/12):+.1f}, overlap-adj)"])
        raw.append({"gauge": name, "tercile": "IC", "n": int(n_eff1),
                    "fwd1_ann": ic1, "fwd1_sharpe": ic1 * np.sqrt(n_eff1),
                    "fwd12": ic12})
        # the gauges trend; within-sub-period IC separates signal from era
        sub = []
        for tag, mask in [("≤2004", s.index.year <= 2004),
                          (">2004", s.index.year > 2004)]:
            i1 = s[mask].corr(fwd1[mask], method="spearman")
            n1 = (s[mask].notna() & fwd1[mask].notna()).sum()
            sub.append(f"{tag} {i1:+.3f} (t {i1*np.sqrt(n1):+.1f})")
            raw.append({"gauge": name, "tercile": f"IC {tag}", "n": int(n1),
                        "fwd1_ann": i1, "fwd1_sharpe": i1 * np.sqrt(n1),
                        "fwd12": np.nan})
        rows.append([f"{name} IC by era", "", sub[0], sub[1], "1m fwd"])

    print("\nEW-ALL forward returns by causal crowding tercile "
          "(expanding ranks, no lookahead):")
    print(tabulate(rows, headers=["state", "n", "fwd 1m (ann) / IC",
                                  "fwd Sharpe", "fwd 12m"], tablefmt="github"))

    out = RESULTS_DIR / "factor_crowding.csv"
    pd.DataFrame(raw).to_csv(out, index=False)
    logger.info(f"\nSaved {out}")
    print("\n⚠ gross academic panel: live crowding acts through costs and "
          "capacity this data cannot see — a null here does NOT clear "
          "crowding risk for a real EW-zoo holder.")


if __name__ == "__main__":
    main()
