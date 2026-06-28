#!/usr/bin/env python3
"""Why a t>6 PEAD anomaly doesn't yield a tradeable edge — and the best extraction of it.

Resolves the paradox: the strongly-significant PEAD IC is almost entirely the
ANNOUNCEMENT-DAY price reaction (efficient repricing when earnings drop), which is not
tradeable because you can't buy before the surprise is public. The genuine post-
announcement DRIFT (t+1 onward) — the only tradeable part — has a near-zero IC.

Part 1: decompose the IC into the untradeable jump (ret_t1) vs the tradeable drift.
Part 2: the most efficient harvest of the weak drift — a rank-weighted full-universe
        factor portfolio beats crude quintile buckets, but the tiny IC caps the IR
        (Fundamental Law: IR ~ IC x sqrt(breadth)).
Part 3: the ORACLE ceiling. ``sue`` is the *realized* earnings surprise — the
        driver variable Z. Correlating realized SUE with the drift is therefore
        the perfect-foresight (ρ₁ = 1) information coefficient, i.e. IC_ideal
        from the Sharpe decomposition (docs/knowledge/sharpe-decomposition-levers.md):
            Sharpe = √(R²) · IC_ideal · √(breadth) · √(decisions/yr)
        So Part 3 turns IC_ideal into the BEST Sharpe any PEAD model could reach
        even with a flawless surprise forecast, and contrasts the tradeable
        drift window against the untradeable jump — where IC_ideal is large but
        ρ₁ = 0 (you can't know the surprise before it is public).

Mega+Large, net of Alpaca costs. Requires signal_panel.csv + data/raw/pead/.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import _bootstrap  # noqa: F401
from posterioralpha.pead import paths
from posterioralpha.research import fundamental_law_sharpe
from experiments.pead_dynamic_exit import build_paths, _stats, COST


def _ic(p, col):
    ics = []
    for _, g in p.groupby("season"):
        if len(g) < 10:
            continue
        ics.append(spearmanr(g["sue"], g[col]).correlation)
    a = np.array(ics); a = a[~np.isnan(a)]
    return a.mean(), np.sqrt(len(a)) * a.mean() / a.std()


def part1():
    p = pd.read_csv(paths.RESULTS_DIR / "signal_panel.csv")
    p = p[p["market_cap"].isin(["Mega Cap", "Large Cap"])].dropna(subset=["sue", "ret_t1", "ret_t21"])
    p["season"] = pd.to_datetime(p["announcement_date"], utc=True, errors="coerce").dt.to_period("Q")
    p["drift"] = p["ret_t21"] - p["ret_t1"]
    print("PART 1 — IC decomposition (is the anomaly tradeable?):")
    for name, col in [("ret_t21  (pre-announce close → t+21: jump + drift)", "ret_t21"),
                      ("ret_t1   (announcement-day JUMP only — untradeable)", "ret_t1"),
                      ("drift    (t+1 → t+21 — the TRADEABLE part)", "drift")]:
        m, t = _ic(p, col)
        print(f"    {name:<52} IC {m:+.4f}  t {t:+.1f}")
    print("    → the strongly-significant 'anomaly' is mostly the untradeable jump.\n")


def part2():
    P, M = build_paths(); M = M.reset_index(drop=True)
    M["r"] = [(P[i, 1:64][~np.isnan(P[i, 1:64])][-1] if np.isfinite(P[i, 1:64]).any() else np.nan)
              for i in M.index]
    print("PART 2 — extracting the weak tradeable drift (rank-weighted vs quintile):")
    for lbl, yrs in [("OOS≥2014", range(2014, 2027)), ("FULL 2001-26", range(2001, 2027))]:
        q, rw, ic = [], [], []
        for _, g in M[M["year"].isin(yrs)].groupby("season"):
            g = g.dropna(subset=["sue", "r"])
            if len(g) < 15:
                continue
            sue, r = g["sue"].values, g["r"].values
            ic.append(spearmanr(sue, r).correlation)
            hi, lo = np.quantile(sue, 0.8), np.quantile(sue, 0.2)
            q.append(r[sue >= hi].mean() - r[sue <= lo].mean())
            z = pd.Series(sue).rank().values; z = z - z.mean(); w = z / np.abs(z).sum()
            rw.append(np.sum(w * r))
        a = np.array(ic); a = a[~np.isnan(a)]
        ql, rl = _stats(np.array(q) / 100 - COST), _stats(np.array(rw) / 100 - COST * 0.5)
        print(f"    [{lbl}] tradeable IC {a.mean():+.4f} (t {np.sqrt(len(a))*a.mean()/a.std():+.1f}) | "
              f"quintile Sharpe {ql[1]:+.2f} | rank-weighted Sharpe {rl[1]:+.2f}")
    print("    → rank-weighting harvests the signal better, but the tiny IC caps the IR.")


def part3_oracle():
    """IC_ideal ceiling: best Sharpe achievable with a perfect surprise forecast.

    SUE is the realized driver, so spearman(SUE, return) is the perfect-foresight
    IC_ideal. The Fundamental Law turns it into a ceiling Sharpe:
        SR_oracle = IC_ideal · √(bets/season · seasons/yr)      (TC = 1)
    Bets/season is the cross-section size each earnings season (PEAD events are
    largely idiosyncratic across names, so nominal ≈ effective breadth); there
    are ~4 seasons/yr. We compute it on the TRADEABLE drift (t+1 → t+21) and on
    the untradeable JUMP (ret_t1): the jump's IC_ideal is large but its ceiling
    is *unreachable* — there ρ₁ = 0, you cannot know the surprise before it is
    public. The tradeable-window IC_ideal is the binding lever.
    """
    p = pd.read_csv(paths.RESULTS_DIR / "signal_panel.csv")
    p = p[p["market_cap"].isin(["Mega Cap", "Large Cap"])].dropna(subset=["sue", "ret_t1", "ret_t21"])
    p["season"] = pd.to_datetime(p["announcement_date"], utc=True, errors="coerce").dt.to_period("Q")
    p["drift"] = p["ret_t21"] - p["ret_t1"]
    p["year"] = pd.to_datetime(p["announcement_date"], utc=True, errors="coerce").dt.year
    SEASONS_PER_YR = 4.0
    print("PART 3 — IC_ideal oracle ceiling (perfect-foresight SUE → Sharpe):")
    for lbl, yrs in [("OOS≥2014", range(2014, 2027)), ("FULL 2001-26", range(2001, 2027))]:
        ic_drift, ic_jump, sizes = [], [], []
        for _, g in p[p["year"].isin(yrs)].groupby("season"):
            if len(g) < 10:
                continue
            ic_drift.append(spearmanr(g["sue"], g["drift"]).correlation)
            ic_jump.append(spearmanr(g["sue"], g["ret_t1"]).correlation)
            sizes.append(len(g))
        d = np.array(ic_drift); d = d[~np.isnan(d)]
        j = np.array(ic_jump); j = j[~np.isnan(j)]
        if len(d) == 0:
            print(f"    [{lbl}] insufficient seasons"); continue
        breadth = float(np.mean(sizes))                      # bets per season
        ic_id = float(d.mean())
        sr_oracle = fundamental_law_sharpe(ic_id, breadth, SEASONS_PER_YR, tc=1.0)
        sr_jump = fundamental_law_sharpe(float(j.mean()), breadth, SEASONS_PER_YR, tc=1.0)
        print(f"    [{lbl}] IC_ideal(drift) {ic_id:+.4f} | breadth ~{breadth:.0f}/season"
              f" → ORACLE Sharpe ceiling {sr_oracle:+.2f}")
        print(f"           jump IC_ideal {j.mean():+.4f} → ceiling {sr_jump:+.2f}, "
              f"but UNREACHABLE (ρ₁=0 pre-announcement)")
    print("    → the tradeable-window IC_ideal — not your forecasting model — is the\n"
          "      binding lever; even a perfect SUE oracle caps the drift Sharpe low.")


def main():
    part1()
    part2()
    part3_oracle()


if __name__ == "__main__":
    main()
