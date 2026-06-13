#!/usr/bin/env python3
"""
Meta-factors and dynamic clusters on the zoo — is there structure beyond momentum?

Idea (user prompt, extends the OpenAP triptych)
-----------------------------------------------
Momentum is only one meta-factor. If the zoo behaves like an asset class,
the other canonical cross-sections should exist at the factor-of-factors
level too — and the panel's correlation structure may form CLUSTERS that
can be nowcast causally and allocated across. Two parts, all monthly,
signals lagged one month, against the EW-ALL benchmark (Sharpe ~1.3 gross):

A. Meta-factor cross-sections (top1/3 − bottom1/3 spreads, signed so the
   hypothesized premium is positive):
     LT reversal   long factors beaten down over months t-59..t-12
     low-vol       long low trailing-36m-vol factors
     seasonality   Keloharju: long factors strong in this calendar month
                   (same-month mean over past 10y, ≥5 obs)

B. Dynamic cluster nowcast: each month, average-linkage hierarchical
   clustering (López de Prado distance √(½(1−ρ)), trailing 36m, K=6
   pre-declared) on eligible factors. Diagnostics: month-to-month
   membership churn (adjusted Rand) and what the clusters mean
   (dominant Cat.Economic). Allocations:
     cluster-balanced   1/K per cluster, EW within (poor-man's HRP —
                        breaks the accounting-heavy tilt of raw EW)
     cluster rotation   hold clusters with positive trailing 12m return

Honesty box
-----------
• Trials: 3 spreads + 2 cluster allocations = 5 on one panel, on top of
  the triptych's ledger.
• Gross academic L/S returns; spreads are 2× leverage, cluster schemes
  rebalance monthly — costs flatter nothing here but hit spreads hardest.
• K=6 and the 36m window are pre-declared, not searched. Same
  survivorship/hindsight-replication caveats as Ideas 7/3/9.
"""
import logging
import sys

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score
from tabulate import tabulate

import _bootstrap  # noqa: F401
from posterioralpha.data import load_openap_doc, load_openap_returns

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)
RESULTS_DIR = _bootstrap.ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MIN_ELIGIBLE = 30
SPLIT_YEAR = 2004
CORR_WIN = 36
N_CLUSTERS = 6


def ann_stats(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 24:
        return {"ann_ret": np.nan, "ann_vol": np.nan, "sharpe": np.nan,
                "t_stat": np.nan, "max_dd": np.nan, "n_months": len(r)}
    mu, sd = r.mean() * 12, r.std() * np.sqrt(12)
    eq = (1 + r).cumprod()
    return {"ann_ret": mu, "ann_vol": sd, "sharpe": mu / (sd + 1e-12),
            "t_stat": mu / (sd + 1e-12) * np.sqrt(len(r) / 12.0),
            "max_dd": float((eq / eq.cummax() - 1).min()),
            "n_months": len(r)}


def alpha_vs(r: pd.Series, bench: pd.Series) -> tuple[float, float, float]:
    df = pd.concat([r, bench], axis=1, keys=["r", "b"], sort=False).dropna()
    X = np.column_stack([np.ones(len(df)), df["b"].values])
    beta, *_ = np.linalg.lstsq(X, df["r"].values, rcond=None)
    resid = df["r"].values - X @ beta
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * resid.var(ddof=2))
    return beta[0] * 12, beta[0] / se[0], beta[1]


def main():
    rets = load_openap_returns()
    doc = load_openap_doc().set_index("Acronym")
    meta = doc.loc[doc.index.intersection(rets.columns),
                   ["Year", "Cat.Economic"]]
    rets = rets[meta.index]
    pub_year = meta["Year"].astype(float)

    age_y = rets.index.year.values[:, None] - pub_year.values
    published = pd.DataFrame(age_y >= 1, index=rets.index, columns=rets.columns)
    elig = (published & rets.notna()).shift(1).fillna(False)
    live = elig.sum(axis=1)
    live = live[live >= MIN_ELIGIBLE].index
    rete = rets.where(elig)
    bench = rete.mean(axis=1).loc[live]
    logger.info(f"{rets.shape[1]} predictors; live (≥{MIN_ELIGIBLE} eligible): "
                f"{live[0].date()} → {live[-1].date()}")

    # ── A. meta-factor cross-sections ─────────────────────────────────────
    ltrev = -rets.rolling(48).sum().shift(13)            # t-60..t-13, long losers
    lowvol = -rets.rolling(CORR_WIN).std().shift(1)      # long low-vol
    seas = pd.DataFrame(np.nan, index=rets.index, columns=rets.columns)
    by_month = {m: rets[rets.index.month == m] for m in range(1, 13)}
    for m, block in by_month.items():
        # same-calendar-month mean over the prior 10 years, ≥5 observations
        s = block.shift(1).rolling(10, min_periods=5).mean()
        seas.loc[s.index] = s.values
    signals = {"LT reversal (t-60..t-13, long losers)": ltrev,
               "low-vol (36m, long low)": lowvol,
               "seasonality (same month, 10y)": seas}

    def xs_spread(sig):
        sig = sig.where(elig)
        hi = sig.ge(sig.quantile(2 / 3, axis=1), axis=0)
        lo = sig.le(sig.quantile(1 / 3, axis=1), axis=0)
        return rete.where(hi).mean(axis=1) - rete.where(lo).mean(axis=1)

    ports = {name: xs_spread(sig).loc[live] for name, sig in signals.items()}

    # ── B. dynamic clusters ───────────────────────────────────────────────
    labels_prev, ari, members = None, [], {}
    bal_ret, rot_ret = pd.Series(np.nan, live), pd.Series(np.nan, live)
    mom12 = rets.rolling(12).sum()
    for i, t in enumerate(live):
        win = rets.loc[:t].iloc[:-1].tail(CORR_WIN)      # data through t-1 only
        cols = rets.columns[elig.loc[t] & (win.notna().sum() == CORR_WIN).values]
        if len(cols) < MIN_ELIGIBLE:
            continue
        corr = np.corrcoef(win[cols].values.T)
        dist = np.sqrt(0.5 * np.clip(1 - corr, 0, 2))
        np.fill_diagonal(dist, 0.0)
        lab = fcluster(linkage(squareform(dist, checks=False), "average"),
                       N_CLUSTERS, criterion="maxclust")
        labels = pd.Series(lab, index=cols)
        if labels_prev is not None:
            common = labels.index.intersection(labels_prev.index)
            ari.append(adjusted_rand_score(labels_prev[common], labels[common]))
        labels_prev, members[t] = labels, labels

        r_t = rets.loc[t]
        cl_ret = {k: r_t[labels.index[labels == k]].mean()
                  for k in range(1, N_CLUSTERS + 1)}
        bal_ret[t] = np.nanmean(list(cl_ret.values()))
        cl_mom = {k: mom12.shift(1).loc[t, labels.index[labels == k]].mean()
                  for k in range(1, N_CLUSTERS + 1)}
        pos = [k for k, v in cl_mom.items() if v > 0]
        rot_ret[t] = np.nanmean([cl_ret[k] for k in pos]) if pos else 0.0

    print(f"\nCluster nowcast stability: median month-to-month adjusted Rand "
          f"= {np.median(ari):.2f} (1 = frozen, 0 = reshuffled)")
    last_t = max(members)
    lab = members[last_t]
    econ = meta["Cat.Economic"]
    rows = []
    for k in range(1, N_CLUSTERS + 1):
        mem = lab.index[lab == k]
        top = econ[mem].value_counts()
        rows.append([k, len(mem), f"{top.index[0]} ({top.iloc[0]}/{len(mem)})",
                     ", ".join(top.index[1:3])])
    print(f"\nWhat the clusters are ({last_t.date()}, dominant Cat.Economic):")
    print(tabulate(rows, headers=["cluster", "n", "dominant", "runners-up"],
                   tablefmt="github"))

    ports["cluster-balanced (1/K, EW within)"] = bal_ret
    ports["cluster rotation (12m>0 clusters)"] = rot_ret
    ports["EW ALL (benchmark)"] = bench

    # ── verdict table ─────────────────────────────────────────────────────
    rows, raw = [], []
    for name, r in ports.items():
        s = ann_stats(r)
        entry = {"portfolio": name, **s}
        if name != "EW ALL (benchmark)":
            a, t_a, b = alpha_vs(r, bench)
            entry.update({"alpha_ann": a, "alpha_t": t_a, "beta": b})
            extra = [f"{a:+.2%}", f"{t_a:.2f}", f"{b:.2f}"]
        else:
            extra = ["—", "—", "—"]
        rows.append([name, f"{s['ann_ret']:+.2%}", f"{s['ann_vol']:.2%}",
                     f"{s['sharpe']:.2f}", f"{s['t_stat']:.2f}",
                     f"{s['max_dd']:.0%}"] + extra)
        raw.append(entry)
        for tag in [f"≤{SPLIT_YEAR}", f">{SPLIT_YEAR}"]:
            sub = r[r.index.year <= SPLIT_YEAR] if tag.startswith("≤") \
                else r[r.index.year > SPLIT_YEAR]
            raw.append({"portfolio": f"{name} {tag}", **ann_stats(sub)})

    print("\nMeta-factors & cluster allocations vs EW ALL (gross):")
    print(tabulate(rows, headers=["portfolio", "ann ret", "vol", "Sharpe", "t",
                                  "maxDD", "α vs EW", "t(α)", "β"],
                   tablefmt="github"))

    sub_rows = [[e["portfolio"], f"{e['ann_ret']:+.2%}", f"{e['sharpe']:.2f}",
                 f"{e['t_stat']:.2f}"]
                for e in raw if str(SPLIT_YEAR) in e["portfolio"]]
    print(f"\nSub-period stability (split {SPLIT_YEAR}):")
    print(tabulate(sub_rows, headers=["portfolio", "ann ret", "Sharpe", "t"],
                   tablefmt="github"))

    # ── kill tests on the standouts ───────────────────────────────────────
    # 1) seasonality could be 12m factor momentum in disguise (same-month
    #    returns sit inside the 12m window): two-factor alpha + corr.
    # 2) both standouts could be early-era artifacts: sub-period alphas.
    sig12 = mom12.shift(1).where(elig)
    mom_spread = (rete.where(sig12.ge(sig12.quantile(2 / 3, axis=1), axis=0)).mean(axis=1)
                  - rete.where(sig12.le(sig12.quantile(1 / 3, axis=1), axis=0)).mean(axis=1)
                  ).loc[live]
    seas_sp = ports["seasonality (same month, 10y)"]
    df = pd.concat([seas_sp, bench, mom_spread], axis=1,
                   keys=["r", "ew", "mom"]).dropna()
    X = np.column_stack([np.ones(len(df)), df["ew"], df["mom"]])
    beta, *_ = np.linalg.lstsq(X, df["r"].values, rcond=None)
    resid = df["r"].values - X @ beta
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * resid.var(ddof=3))
    print("\nKill tests on the standouts:")
    print(f"  seasonality: corr with 12m XS-mom spread = "
          f"{seas_sp.corr(mom_spread):.2f}; two-factor alpha (EW + mom) = "
          f"{beta[0]*12:+.2%}/yr  t={beta[0]/se[0]:.2f}  "
          f"(β_ew {beta[1]:.2f}, β_mom {beta[2]:.2f})")
    for name in ["seasonality (same month, 10y)",
                 "cluster-balanced (1/K, EW within)"]:
        r = ports[name]
        parts = []
        for tag, sub in [(f"≤{SPLIT_YEAR}", r[r.index.year <= SPLIT_YEAR]),
                         (f">{SPLIT_YEAR}", r[r.index.year > SPLIT_YEAR])]:
            a, t_a, _ = alpha_vs(sub, bench)
            parts.append(f"{tag}: {a:+.2%} (t {t_a:.2f})")
        print(f"  {name} — alpha vs EW by era: " + "   ".join(parts))

    # the payoff question for the EW holder: 50/50 EW + vol-matched
    # seasonality (the surviving standout; +1 trial on the ledger)
    combo = 0.5 * bench + 0.5 * seas_sp * (bench.std() / seas_sp.std())
    print("\nDoes it improve the holding?  (50/50 EW + vol-matched seasonality)")
    for name, r in [("EW ALL", bench), ("combo", combo)]:
        s = ann_stats(r)
        sub = [f"{tag} Sharpe {ann_stats(x)['sharpe']:.2f}" for tag, x in
               [(f"≤{SPLIT_YEAR}", r[r.index.year <= SPLIT_YEAR]),
                (f">{SPLIT_YEAR}", r[r.index.year > SPLIT_YEAR])]]
        print(f"  {name:<8} ann {s['ann_ret']:+.2%}  Sharpe {s['sharpe']:.2f}"
              f"  maxDD {s['max_dd']:.0%}   " + "   ".join(sub))
        raw.append({"portfolio": f"combo check: {name}", **s})

    out = RESULTS_DIR / "zoo_metafactors.csv"
    pd.DataFrame(raw).to_csv(out, index=False)
    logger.info(f"\nSaved {out}")
    print("\n⚠ gross panel, 5 trials on top of the triptych's ledger; "
          "K=6/36m pre-declared, not searched.")


if __name__ == "__main__":
    main()
