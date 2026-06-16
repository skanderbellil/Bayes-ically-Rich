#!/usr/bin/env python3
"""
Do some factors LEAD others? Factor lead-lag network (user prompt 2026-06-13).

The idea
--------
Construction determines speed: market-priced factors (momentum, short-term
reversal, low-risk) absorb information fast; accounting-priced factors
(investment, accruals, profitability, value) update only on reporting
cycles. So a FAST factor's return this month might predict a SLOW factor's
return next month — a directed lead-lag network, not the symmetric
own-momentum already shown to be mere zoo exposure (Idea 3).

The overfitting trap and the discipline
---------------------------------------
A 13×13 theme lead-lag matrix is 156 directed tests; the 153-factor version
is ~23k — the Harvey problem at its worst. Two guards:
  1. Theme level (13×13), economically interpretable, not 153 individual.
  2. CROSS-REGION agreement is the kill test. The JKP panel gives usa /
     developed / emerging / world_ex_us. A lead-lag link real in the US that
     also holds out-of-region is structure; one that flips is mined noise.
  3. One PRE-DECLARED economic hypothesis (fast→slow) carries the
     confirmatory weight; the full network map is exploratory/descriptive.

Three layers
------------
A. Descriptive network: per region, lag-1 cross-correlation C[A,B] =
   corr(z(A_t), z(B_{t+1})); asymmetry S[A,B] = C[A,B] − C[B,A]; net
   "leadingness" of A = mean_B S[A,B]. Rank themes leader→laggard and test
   whether the ranking agrees across regions (Spearman).
B. Pre-declared fast→slow test: does the FAST basket lead the SLOW basket?
   corr(fast_t, slow_{t+1}) vs the symmetric control corr(slow_t, fast_{t+1}),
   per region.
C. Tradable, walk-forward: predict each theme's next return from the
   lagged returns of the OTHER themes (causal trailing cross-corr), form a
   L/S on predicted returns, net of cost — benchmarked against OWN-momentum
   (to prove cross-factor info adds beyond a theme predicting itself) and
   EW-ALL.

Honesty box: gross theme L/S returns; signs standardized per theme. The
predictive sleeve estimates 12 coefficients per laggard on a trailing
window → noisy; read the cross-region consistency, not any single t.
"""
import logging
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tabulate import tabulate

import _bootstrap  # noqa: F401
from posterioralpha.data import load_jkp_factors

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)
RESULTS_DIR = _bootstrap.ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

REGIONS = ["usa", "developed", "emerging", "world_ex_us"]
FAST = ["momentum", "short_term_reversal", "low_risk"]
SLOW = ["investment", "accruals", "profitability", "value",
        "profit_growth", "quality"]
TRAIN_MIN = 120          # months before walk-forward predictions begin
COST_BPS = 20


def zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Expanding causal z-score per column (no lookahead)."""
    mu = df.expanding(min_periods=36).mean()
    sd = df.expanding(min_periods=36).std()
    return ((df - mu) / sd).clip(-4, 4)


def ann_stats(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 24:
        return {"sharpe": np.nan, "ann_ret": np.nan, "t_stat": np.nan}
    mu, sd = r.mean() * 12, r.std() * np.sqrt(12)
    return {"ann_ret": mu, "sharpe": mu / (sd + 1e-12),
            "t_stat": mu / (sd + 1e-12) * np.sqrt(len(r) / 12.0)}


def leadingness(rets: pd.DataFrame) -> pd.Series:
    """Net leadingness per theme: mean_B(corr(A_t, B_{t+1}) − corr(B_t, A_{t+1}))."""
    z = rets.apply(lambda c: (c - c.mean()) / c.std())
    cols = z.columns
    C = pd.DataFrame(np.nan, index=cols, columns=cols)
    for a in cols:
        for b in cols:
            if a == b:
                continue
            x = z[a].iloc[:-1].values
            y = z[b].iloc[1:].values
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() > 36:
                C.loc[a, b] = np.corrcoef(x[m], y[m])[0, 1]
    asym = C - C.T
    return asym.mean(axis=1).sort_values(ascending=False), C


def main():
    # ── A. descriptive network + cross-region leadership stability ─────────
    lead_by_region = {}
    print("\n=== A. Net leadingness ranking per region (leader → laggard) ===")
    for region in REGIONS:
        rets = load_jkp_factors(region).dropna(how="all")
        rets = rets.loc[:, rets.notna().mean() > 0.8].dropna()
        lead, _ = leadingness(rets)
        lead_by_region[region] = lead
        top = ", ".join(f"{k}({v:+.3f})" for k, v in lead.head(3).items())
        bot = ", ".join(f"{k}({v:+.3f})" for k, v in lead.tail(3).items())
        print(f"  {region:<12} LEAD: {top}   |   LAG: {bot}")

    # cross-region agreement of the leadership ranking
    print("\nCross-region rank agreement of leadingness (Spearman):")
    allcols = sorted(set.intersection(*[set(v.index) for v in lead_by_region.values()]))
    rank_rows = []
    for i, r1 in enumerate(REGIONS):
        for r2 in REGIONS[i + 1:]:
            a = lead_by_region[r1].reindex(allcols)
            b = lead_by_region[r2].reindex(allcols)
            rho, _ = spearmanr(a, b)
            rank_rows.append([f"{r1} vs {r2}", f"{rho:+.2f}"])
    print(tabulate(rank_rows, headers=["pair", "rank corr"], tablefmt="github"))
    mean_lead = pd.concat(lead_by_region, axis=1).reindex(allcols).mean(axis=1).sort_values(ascending=False)
    print("\nAveraged leadingness across regions (robust leaders → laggards):")
    print(tabulate([[k, f"{v:+.3f}"] for k, v in mean_lead.items()],
                   headers=["theme", "net leadingness"], tablefmt="github"))

    # ── B. pre-declared fast → slow test ──────────────────────────────────
    print("\n=== B. Pre-declared FAST → SLOW lead-lag (corr, lag 1) ===")
    brows = []
    for region in REGIONS:
        rets = load_jkp_factors(region).dropna(how="all")
        fast = [c for c in FAST if c in rets.columns]
        slow = [c for c in SLOW if c in rets.columns]
        pair = pd.concat([rets[fast].mean(axis=1).rename("f"),
                          rets[slow].mean(axis=1).rename("s")], axis=1).dropna()
        fb = (pair["f"] - pair["f"].mean()) / pair["f"].std()
        sb = (pair["s"] - pair["s"].mean()) / pair["s"].std()
        c_fs = np.corrcoef(fb.iloc[:-1], sb.iloc[1:])[0, 1]     # fast leads slow
        c_sf = np.corrcoef(sb.iloc[:-1], fb.iloc[1:])[0, 1]     # slow leads fast
        n = len(fb) - 1
        brows.append([region, f"{c_fs:+.3f}", f"{c_fs*np.sqrt(n):+.2f}",
                      f"{c_sf:+.3f}", f"{c_sf*np.sqrt(n):+.2f}", n])
    print(tabulate(brows, headers=["region", "fast→slow", "t", "slow→fast",
                                   "t", "n"], tablefmt="github"))

    # ── C. tradable walk-forward cross-factor predictor ───────────────────
    print("\n=== C. Walk-forward cross-factor predictor vs own-mom vs EW ===")
    crows = []
    for region in REGIONS:
        rets = load_jkp_factors(region).dropna(how="all")
        rets = rets.loc[:, rets.notna().mean() > 0.8].dropna()
        z = zscore(rets).dropna()
        rets = rets.loc[z.index]
        cols = list(rets.columns)
        idx = rets.index

        Z = z.values                               # (T, K) causal z-scores
        T, K = Z.shape
        cross_arr = np.full((T, K), np.nan)
        own_arr = np.full((T, K), np.nan)
        for i in range(TRAIN_MIN, T - 1):
            x_lag = Z[:i - 1]                       # A_t,  t = 0..i-2
            y_now = Z[1:i]                          # B_{t+1}
            signal = Z[i]                           # this month's z → predict i+1
            # lag-1 cross-corr matrix C[a,b] on data through month i (causal)
            xc = x_lag - x_lag.mean(0)
            yc = y_now - y_now.mean(0)
            cov = xc.T @ yc / len(xc)
            denom = np.outer(x_lag.std(0), y_now.std(0)) + 1e-12
            C = cov / denom                         # C[a,b] = corr(A_t, B_{t+1})
            Coff = C.copy()
            np.fill_diagonal(Coff, 0.0)
            cross_arr[i + 1] = signal @ Coff        # sum_a≠b C[a,b]·signal[a]
            own_arr[i + 1] = np.diag(C) * signal
        cross_pred = pd.DataFrame(cross_arr, index=idx, columns=cols)
        own_pred = pd.DataFrame(own_arr, index=idx, columns=cols)

        def ls_from_pred(pred):
            sig = pred.reindex(idx)
            hi = sig.ge(sig.median(axis=1), axis=0)
            w = (hi.astype(float) - 0.5)
            w = w.div(w.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
            r = (w * rets).sum(axis=1)
            turn = w.diff().abs().sum(axis=1).mean()
            return r, turn

        r_cross, t_cross = ls_from_pred(cross_pred)
        r_own, _ = ls_from_pred(own_pred)
        ew = rets.mean(axis=1)
        live = cross_pred.dropna(how="all").index
        sc = ann_stats(r_cross.loc[live])
        sc_net = ann_stats((r_cross - COST_BPS / 1e4 * t_cross).loc[live])
        so = ann_stats(r_own.loc[live])
        se = ann_stats(ew.loc[live])
        crows.append([region, f"{sc['sharpe']:.2f}", f"{sc['t_stat']:.2f}",
                      f"{sc_net['sharpe']:.2f}", f"{so['sharpe']:.2f}",
                      f"{se['sharpe']:.2f}"])

    print(tabulate(crows, headers=["region", "cross SR", "cross t",
                                   f"cross net@{COST_BPS}bp", "own-mom SR",
                                   "EW SR"], tablefmt="github"))
    print("\nVerdict hangs on: (A) does the leadership ranking agree across "
          "regions? (B) is fast→slow stronger than slow→fast and consistent? "
          "(C) does the cross-factor sleeve beat own-momentum AND survive "
          "cost — cross-region, not in one lucky panel?")

    pd.concat(lead_by_region, axis=1).to_csv(RESULTS_DIR / "factor_leadlag_leadingness.csv")
    logger.info(f"\nSaved {RESULTS_DIR / 'factor_leadlag_leadingness.csv'}")


if __name__ == "__main__":
    main()
