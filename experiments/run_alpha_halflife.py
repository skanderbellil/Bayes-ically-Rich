#!/usr/bin/env python3
"""
The alpha half-life trade — the factor zoo's age structure as a signal.

Idea (fresh, from the knowledge base)
-------------------------------------
McLean & Pontiff: anomaly returns decay ~58% after publication. The entire
literature treats this as a *discount to apply* to backtests. This study
treats it as an *allocation signal*: if decay is real and gradual, a
portfolio of YOUNG factors (recently published, early in their decay) should
beat OLD factors (long since arbitraged) — a momentum-of-ideas effect at
the level of the zoo itself. Publication year is causal information: at any
month t you know exactly which factors have been published and for how long.

Data: Chen-Zimmermann OpenAP — 212 predictors' monthly L/S returns
(1926-2024) + per-predictor publication year and original sample window.

Three parts
-----------
1. Replicate the decay fact on this data: mean annualized L/S return by
   phase — in-sample (original paper's window), post-sample-pre-publication,
   post-publication — plus the event-time profile around publication.
2. Tradable age buckets (causal, monthly EW of factor L/S returns):
   YOUNG (1-5y since publication), MID (6-15y), OLD (>15y), the
   YOUNG−OLD spread, and the all-published EW benchmark.
3. Sub-period stability (split at 2004, ≈ sample midpoint of the live era).

Honesty box
-----------
• Hindsight-replication bias: the zoo contains only factors good enough to
  be published AND replicated by C-Z. A real-time YOUNG portfolio would
  also hold factors that later failed replication; YOUNG is hit harder by
  this than OLD (less time to fail out). Positive YOUNG results are upper
  bounds.
• Returns are academic L/S constructions: no costs, monthly rebalanced,
  include hard-to-short names. Spreads between buckets partially net this
  out (same construction both legs); levels do not.
• Trials: 3 buckets + 1 spread + 1 benchmark on one panel = 5.
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

YOUNG_MAX, MID_MAX = 5, 15      # bucket edges, years since publication
MIN_LEG = 10                    # require this many factors per bucket
SPLIT_YEAR = 2004


def ann_stats(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 24:
        return {"ann_ret": np.nan, "ann_vol": np.nan, "sharpe": np.nan,
                "t_stat": np.nan, "max_dd": np.nan, "n_months": len(r)}
    mu, sd = r.mean() * 12, r.std() * np.sqrt(12)
    sharpe = mu / (sd + 1e-12)
    eq = (1 + r).cumprod()
    return {"ann_ret": mu, "ann_vol": sd, "sharpe": sharpe,
            "t_stat": sharpe * np.sqrt(len(r) / 12.0),
            "max_dd": float((eq / eq.cummax() - 1).min()),
            "n_months": len(r)}


def main():
    rets = load_openap_returns()
    doc = load_openap_doc().set_index("Acronym")
    meta = doc.loc[doc.index.intersection(rets.columns),
                   ["Year", "SampleStartYear", "SampleEndYear"]].astype(float)
    rets = rets[meta.index]
    logger.info(f"{rets.shape[1]} predictors, {rets.index[0].date()} → "
                f"{rets.index[-1].date()}, pub years "
                f"{int(meta.Year.min())}–{int(meta.Year.max())}")

    # ── 1. The decay fact on this data ────────────────────────────────────
    yrs = rets.index.year.values[:, None]
    in_sample = (yrs >= meta.SampleStartYear.values) & (yrs <= meta.SampleEndYear.values)
    post_sample = (yrs > meta.SampleEndYear.values) & (yrs <= meta.Year.values)
    post_pub = yrs > meta.Year.values

    phases = {}
    for name, mask in [("in-sample", in_sample),
                       ("post-sample, pre-pub", post_sample),
                       ("post-publication", post_pub)]:
        vals = rets.values.copy()
        vals[~mask] = np.nan
        phases[name] = float(np.nanmean(vals)) * 12
    decay = 1 - phases["post-publication"] / phases["in-sample"]

    print("\nPhase means (annualized, EW across predictors):")
    for k, v in phases.items():
        print(f"  {k:<22} {v:+.2%}")
    print(f"  → post-publication decay: {decay:.0%}  "
          f"(McLean-Pontiff report ~58%)")

    # event-time profile: years since publication, EW across predictors
    age_y = yrs - meta.Year.values
    prof = {}
    for lo, hi in [(-15, -10), (-10, -5), (-5, 0), (0, 5), (5, 10),
                   (10, 15), (15, 25)]:
        vals = rets.values.copy()
        vals[~((age_y >= lo) & (age_y < hi))] = np.nan
        prof[f"{lo:+d}..{hi:+d}y"] = float(np.nanmean(vals)) * 12
    print("\nEvent time (years relative to publication):")
    print(tabulate([[k, f"{v:+.2%}"] for k, v in prof.items()],
                   headers=["window", "ann ret"], tablefmt="github"))

    # ── 2. Tradable age buckets (causal) ──────────────────────────────────
    age = pd.DataFrame(age_y, index=rets.index, columns=rets.columns)
    published = age >= 1          # require a full year since publication
    young = rets.where(published & (age <= YOUNG_MAX)).mean(axis=1)
    mid   = rets.where(published & (age > YOUNG_MAX) & (age <= MID_MAX)).mean(axis=1)
    old   = rets.where(published & (age > MID_MAX)).mean(axis=1)
    bench = rets.where(published).mean(axis=1)

    counts = pd.DataFrame({
        "young": (published & (age <= YOUNG_MAX) & rets.notna()).sum(axis=1),
        "old":   (published & (age > MID_MAX) & rets.notna()).sum(axis=1),
    })
    live = counts[(counts["young"] >= MIN_LEG) & (counts["old"] >= MIN_LEG)].index
    start = live[0]
    logger.info(f"\nLive (≥{MIN_LEG} factors in YOUNG and OLD): "
                f"{start.date()} → {live[-1].date()}")

    ports = {
        f"YOUNG (1-{YOUNG_MAX}y)":      young.loc[start:],
        f"MID ({YOUNG_MAX+1}-{MID_MAX}y)": mid.loc[start:],
        f"OLD (>{MID_MAX}y)":           old.loc[start:],
        "YOUNG − OLD":                  (young - old).loc[start:],
        "ALL published (EW)":           bench.loc[start:],
    }

    rows, raw = [], []
    for name, r in ports.items():
        s = ann_stats(r)
        rows.append([name, f"{s['ann_ret']:+.2%}", f"{s['ann_vol']:.2%}",
                     f"{s['sharpe']:.2f}", f"{s['t_stat']:.2f}",
                     f"{s['max_dd']:.0%}", s["n_months"]])
        raw.append({"portfolio": name, **s})
        for tag, sub in [(f"≤{SPLIT_YEAR}", r[r.index.year <= SPLIT_YEAR]),
                         (f">{SPLIT_YEAR}", r[r.index.year > SPLIT_YEAR])]:
            ss = ann_stats(sub)
            raw.append({"portfolio": f"{name} {tag}", **ss})

    print("\nAge-bucket portfolios (EW of factor L/S returns, gross):")
    print(tabulate(rows, headers=["portfolio", "ann ret", "vol", "Sharpe",
                                  "t", "maxDD", "months"], tablefmt="github"))

    sub_rows = []
    for entry in raw:
        if f"≤{SPLIT_YEAR}" in entry["portfolio"] or f">{SPLIT_YEAR}" in entry["portfolio"]:
            sub_rows.append([entry["portfolio"], f"{entry['ann_ret']:+.2%}",
                             f"{entry['sharpe']:.2f}", f"{entry['t_stat']:.2f}"])
    print(f"\nSub-period stability (split {SPLIT_YEAR}):")
    print(tabulate(sub_rows, headers=["portfolio", "ann ret", "Sharpe", "t"],
                   tablefmt="github"))

    out = RESULTS_DIR / "alpha_halflife.csv"
    pd.DataFrame(raw).to_csv(out, index=False)
    logger.info(f"\nSaved {out}")
    print("\n⚠ gross academic L/S returns; hindsight-replication bias favours "
          "YOUNG — read positive YOUNG results as upper bounds.")


if __name__ == "__main__":
    main()
