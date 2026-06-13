#!/usr/bin/env python3
"""
Factor seasonality OOS at MATCHED BREADTH — the decisive test (Idea 10).

The open question from run_zoo_metafactors.py / run_seasonality_oos.py
-----------------------------------------------------------------------
Factor-level seasonality (Keloharju same-calendar-month) was the first
clean t>3 positive on the OpenAP US panel: α vs EW-ALL +9.5%/yr (t 3.6).
The cross-region OOS test confirmed *direction* but not *magnitude* — but
it was run on the 13 JKP THEMES (≈4 factors per leg), too thin to carry a
breadth effect. The roadmap flagged the real test as still open: rerun OOS
at matched breadth on the 153 individual JKP factors, not the 13 themes.

This study does exactly that. The JKP individual panel gives ~153
premium-oriented characteristic factors per region for four regions:
  • usa          — a construction-OOS of OpenAP (different vendor, defs,
                   weighting), at matched breadth → does the headline
                   replicate off the OpenAP panel?
  • developed / emerging / world_ex_us — genuine geographic OOS, now with
                   enough factors per leg to actually resolve t.

Construction (identical to run_zoo_metafactors.py)
--------------------------------------------------
  seas[t,f] = same-calendar-month mean of factor f over the prior 10y
              (≥5 obs), known at t (lagged one month)
  spread    = EW(top 1/3 by seas) − EW(bottom 1/3), monthly
  benchmark = EW of all eligible factors (the diversification carry)

Tests per region: spread Sharpe/t; α and t(α) vs EW-ALL; correlation with
the 12-month factor-momentum spread (is seasonality just momentum?);
sub-period stability (split 2004); and a turnover-based NET cost haircut
(the second open sub-question) — the seasonality sleeve turns over ~2×/yr
per leg, so cost realism decides promotability.

Honesty box
-----------
• JKP factors are gross academic L/S (no costs in the raw series); the
  explicit cost grid below is the friction model. Spreads net same-vendor
  construction bias across legs; levels do not.
• Matched breadth ≠ identical factors: JKP defs differ from OpenAP, which
  is the point (construction-OOS), but means usa is not a pure replication.
• Trials: 4 regions × (spread + α + combo) — disclosed; treat the
  multi-region agreement, not any single t, as the evidence.
"""
import logging
import sys

import numpy as np
import pandas as pd
from tabulate import tabulate

import _bootstrap  # noqa: F401
from posterioralpha.data import load_jkp_individual

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)
RESULTS_DIR = _bootstrap.ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

REGIONS = ["usa", "developed", "emerging", "world_ex_us"]
MIN_ELIGIBLE = 30          # factors needed before a region goes live
SPLIT_YEAR = 2004
SEAS_YEARS, SEAS_MINOBS = 10, 5
# one-way cost per unit of turnover, in bps (realistic factor-sleeve frictions)
COST_GRID_BPS = [0, 10, 20, 30, 50]


def ann_stats(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 24:
        return {"ann_ret": np.nan, "sharpe": np.nan, "t_stat": np.nan,
                "max_dd": np.nan, "n_months": len(r)}
    mu, sd = r.mean() * 12, r.std() * np.sqrt(12)
    eq = (1 + r).cumprod()
    return {"ann_ret": mu, "sharpe": mu / (sd + 1e-12),
            "t_stat": mu / (sd + 1e-12) * np.sqrt(len(r) / 12.0),
            "max_dd": float((eq / eq.cummax() - 1).min()), "n_months": len(r)}


def alpha_vs(r: pd.Series, bench: pd.Series):
    """Monthly OLS r = a + b·bench → (ann α, t(α), β)."""
    df = pd.concat([r, bench], axis=1, keys=["r", "b"]).dropna()
    if len(df) < 24:
        return np.nan, np.nan, np.nan
    X = np.column_stack([np.ones(len(df)), df["b"].values])
    beta, *_ = np.linalg.lstsq(X, df["r"].values, rcond=None)
    resid = df["r"].values - X @ beta
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * resid.var(ddof=2))
    return beta[0] * 12, beta[0] / se[0], beta[1]


def seasonality_signal(rets: pd.DataFrame) -> pd.DataFrame:
    """Same-calendar-month mean over prior SEAS_YEARS (≥SEAS_MINOBS), lagged."""
    seas = pd.DataFrame(np.nan, index=rets.index, columns=rets.columns)
    for m in range(1, 13):
        block = rets[rets.index.month == m]
        s = block.shift(1).rolling(SEAS_YEARS, min_periods=SEAS_MINOBS).mean()
        seas.loc[s.index] = s.values
    return seas


def tercile_spread_with_turnover(sig, rete, elig):
    """Return (spread series, per-leg avg monthly turnover)."""
    hi = sig.ge(sig.quantile(2 / 3, axis=1), axis=0) & elig
    lo = sig.le(sig.quantile(1 / 3, axis=1), axis=0) & elig
    spread = rete.where(hi).mean(axis=1) - rete.where(lo).mean(axis=1)
    # turnover: fraction of the long leg's membership that changes each month
    hi_f = hi.astype(float)
    w = hi_f.div(hi_f.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    turn = w.diff().abs().sum(axis=1) / 2.0     # one-way fraction
    return spread, float(turn.mean())


def main():
    panel = load_jkp_individual()
    logger.info(f"JKP individual: {panel['name'].nunique()} factors, "
                f"regions {sorted(panel['location'].unique())}, "
                f"{panel['date'].min().date()} → {panel['date'].max().date()}")

    summary, raw = [], []
    for region in REGIONS:
        rets = load_jkp_individual(region)
        elig = rets.notna() & rets.shift(1).notna()
        n_elig = elig.sum(axis=1)
        live = n_elig[n_elig >= MIN_ELIGIBLE].index
        if len(live) < 60:
            logger.info(f"{region}: too few live months, skipping")
            continue
        rets, elig = rets.loc[live], elig.loc[live]
        rete = rets.where(elig)

        seas = seasonality_signal(rets).loc[live]
        mom12 = rets.rolling(12).sum().shift(1).loc[live]

        sp, leg_turn = tercile_spread_with_turnover(seas, rete, elig)
        bench = rete.mean(axis=1)
        mom_sp, _ = tercile_spread_with_turnover(mom12, rete, elig)

        s = ann_stats(sp)
        a, t_a, b = alpha_vs(sp, bench)
        corr_mom = sp.corr(mom_sp)

        # NET cost grid: both legs turn over each month; monthly drag =
        # (bps/1e4) × (2 × one-way leg turnover). ann_turn for reporting.
        ann_turn = 2.0 * leg_turn * 12.0
        net = {f"net@{bps}bp":
               ann_stats(sp - (bps / 1e4) * 2.0 * leg_turn)["sharpe"]
               for bps in COST_GRID_BPS}

        summary.append([region, f"{s['ann_ret']:+.1%}", f"{s['sharpe']:.2f}",
                        f"{s['t_stat']:.2f}", f"{a:+.1%}", f"{t_a:.2f}",
                        f"{b:+.2f}", f"{corr_mom:+.2f}", f"{ann_turn:.1f}×",
                        s["n_months"]])
        raw.append({"region": region, **s, "alpha_ann": a, "t_alpha": t_a,
                    "beta_ew": b, "corr_mom": corr_mom, "ann_turnover": ann_turn,
                    **net})
        # sub-period stability
        for tag, sub in [(f"≤{SPLIT_YEAR}", sp[sp.index.year <= SPLIT_YEAR]),
                         (f">{SPLIT_YEAR}", sp[sp.index.year > SPLIT_YEAR])]:
            raw.append({"region": f"{region} {tag}", **ann_stats(sub)})

    print("\nSeasonality spread, matched breadth (JKP 153 individual factors):")
    print(tabulate(summary, headers=["region", "ann ret", "Sharpe", "t",
                                     "α vs EW", "t(α)", "β", "corr mom",
                                     "turn/yr", "months"], tablefmt="github"))

    print("\nNET Sharpe after cost haircut (bps per unit one-way turnover):")
    cost_rows = [[e["region"]] + [f"{e[f'net@{bps}bp']:.2f}"
                                  for bps in COST_GRID_BPS]
                 for e in raw if "alpha_ann" in e]
    print(tabulate(cost_rows, headers=["region"] + [f"{bps}bp"
                                                    for bps in COST_GRID_BPS],
                   tablefmt="github"))

    sub_rows = [[e["region"], f"{e['ann_ret']:+.1%}", f"{e['sharpe']:.2f}",
                 f"{e['t_stat']:.2f}"]
                for e in raw if str(SPLIT_YEAR) in e["region"]]
    print(f"\nSub-period stability (split {SPLIT_YEAR}):")
    print(tabulate(sub_rows, headers=["region", "ann ret", "Sharpe", "t"],
                   tablefmt="github"))

    out = RESULTS_DIR / "seasonality_matched_oos.csv"
    pd.DataFrame(raw).to_csv(out, index=False)
    logger.info(f"\nSaved {out}")
    print("\nRead: usa = construction-OOS of OpenAP's +9.5%/t3.6 headline; "
          "developed/emerging/world_ex_us = geographic OOS at last with "
          "enough breadth to resolve t. Agreement across regions + survival "
          "of the cost grid is the promotion bar.")


if __name__ == "__main__":
    main()
