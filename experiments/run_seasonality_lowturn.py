#!/usr/bin/env python3
"""
Can factor seasonality be traded? Low-turnover construction (Idea 10 follow-up).

The binding constraint
----------------------
run_seasonality_matched_oos.py confirmed the signal is real out of sample
(developed/emerging/world_ex_us α +2.3-3.4%/yr, t 3.1-4.3, momentum-
orthogonal) but the hard monthly tercile L/S turns over ~12×/yr, so the
break-even is ~25-30 bps/unit — institutional-only, dead at retail. The
open question: is that turnover STRUCTURAL (different factors peak in
different calendar months, so the book must churn) or REDUCIBLE (noise in
the breakpoint sort)? If reducible, seasonality becomes a tradable sleeve.

Three constructions of the same signal, same panel, all causal
--------------------------------------------------------------
  tercile   long top 1/3 / short bottom 1/3, fully reconstituted monthly
            (the baseline from the matched-breadth study)
  continuous dollar-neutral weights ∝ cross-sectional z-score of the
            seasonality signal (names near the median barely trade →
            structurally lower turnover than hard breakpoints)
  buffered  hysteresis tercile: a name enters the long leg at top-1/3 but
            is only dropped when it falls below the median; symmetric short
            leg. Reduces membership churn from boundary flicker.

For each: gross Sharpe, annual turnover, and NET Sharpe at 10/20/30 bps per
unit one-way turnover. The promotion question is whether any construction
holds a positive net Sharpe at retail-ish (20-30 bps) frictions while
keeping most of the gross signal.

Regions: usa (construction-OOS), developed + world_ex_us (cleanest
geographic OOS). JKP 153 individual factors, premium-oriented.
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

REGIONS = ["usa", "developed", "world_ex_us"]
MIN_ELIGIBLE = 30
SEAS_YEARS, SEAS_MINOBS = 10, 5
COST_GRID_BPS = [0, 10, 20, 30]


def ann_stats(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 24:
        return {"sharpe": np.nan, "ann_ret": np.nan, "t_stat": np.nan}
    mu, sd = r.mean() * 12, r.std() * np.sqrt(12)
    return {"ann_ret": mu, "sharpe": mu / (sd + 1e-12),
            "t_stat": mu / (sd + 1e-12) * np.sqrt(len(r) / 12.0)}


def seasonality_signal(rets: pd.DataFrame) -> pd.DataFrame:
    seas = pd.DataFrame(np.nan, index=rets.index, columns=rets.columns)
    for m in range(1, 13):
        block = rets[rets.index.month == m]
        s = block.shift(1).rolling(SEAS_YEARS, min_periods=SEAS_MINOBS).mean()
        seas.loc[s.index] = s.values
    return seas


def weights_to_returns(w: pd.DataFrame, rete: pd.DataFrame):
    """Portfolio return series + average monthly one-way turnover for a weight panel."""
    r = (w.shift(0) * rete).sum(axis=1)         # weights known at month start
    turn = w.diff().abs().sum(axis=1) / 2.0
    return r, float(turn.mean())


def tercile_weights(sig, elig):
    hi = sig.ge(sig.quantile(2 / 3, axis=1), axis=0) & elig
    lo = sig.le(sig.quantile(1 / 3, axis=1), axis=0) & elig
    wl = hi.astype(float).div(hi.sum(axis=1).replace(0, np.nan), axis=0)
    ws = lo.astype(float).div(lo.sum(axis=1).replace(0, np.nan), axis=0)
    return (wl - ws).fillna(0.0)


def continuous_weights(sig, elig):
    """Dollar-neutral weights ∝ cross-sectional z-score; gross leverage = 1."""
    s = sig.where(elig)
    z = s.sub(s.mean(axis=1), axis=0).div(s.std(axis=1).replace(0, np.nan), axis=0)
    z = z.fillna(0.0)
    gross = z.abs().sum(axis=1).replace(0, np.nan)
    return z.div(gross, axis=0).fillna(0.0)


def buffered_weights(sig, elig):
    """Hysteresis tercile: enter at top/bottom 1/3, exit only past the median."""
    enter_hi = sig.ge(sig.quantile(2 / 3, axis=1), axis=0) & elig
    keep_hi = sig.ge(sig.quantile(0.5, axis=1), axis=0) & elig
    enter_lo = sig.le(sig.quantile(1 / 3, axis=1), axis=0) & elig
    keep_lo = sig.le(sig.quantile(0.5, axis=1), axis=0) & elig

    idx, cols = sig.index, sig.columns
    long_mask = pd.DataFrame(False, index=idx, columns=cols)
    short_mask = pd.DataFrame(False, index=idx, columns=cols)
    prev_l = prev_s = pd.Series(False, index=cols)
    for t in idx:
        l = (prev_l & keep_hi.loc[t]) | enter_hi.loc[t]
        s = (prev_s & keep_lo.loc[t]) | enter_lo.loc[t]
        l = l & ~s                       # no name in both legs
        long_mask.loc[t], short_mask.loc[t] = l, s
        prev_l, prev_s = l, s
    wl = long_mask.astype(float).div(long_mask.sum(axis=1).replace(0, np.nan), axis=0)
    ws = short_mask.astype(float).div(short_mask.sum(axis=1).replace(0, np.nan), axis=0)
    return (wl - ws).fillna(0.0)


def main():
    rows = []
    for region in REGIONS:
        rets = load_jkp_individual(region)
        elig = rets.notna() & rets.shift(1).notna()
        live = elig.sum(axis=1)[elig.sum(axis=1) >= MIN_ELIGIBLE].index
        if len(live) < 60:
            continue
        rets, elig = rets.loc[live], elig.loc[live]
        rete = rets.where(elig)
        sig = seasonality_signal(rets).loc[live]

        for cname, wfn in [("tercile", tercile_weights),
                           ("continuous", continuous_weights),
                           ("buffered", buffered_weights)]:
            w = wfn(sig, elig)
            r, leg_turn = weights_to_returns(w, rete)
            s = ann_stats(r)
            ann_turn = 2.0 * leg_turn * 12.0
            nets = [ann_stats(r - (bps / 1e4) * 2.0 * leg_turn)["sharpe"]
                    for bps in COST_GRID_BPS]
            rows.append([region, cname, f"{s['ann_ret']:+.1%}",
                         f"{s['sharpe']:.2f}", f"{s['t_stat']:.2f}",
                         f"{ann_turn:.1f}×"] + [f"{n:.2f}" for n in nets])

    hdr = (["region", "construction", "gross ret", "gross SR", "t", "turn/yr"]
           + [f"net@{b}bp" for b in COST_GRID_BPS])
    print("\nLow-turnover seasonality constructions (JKP 153 individual factors):")
    print(tabulate(rows, headers=hdr, tablefmt="github"))

    out = RESULTS_DIR / "seasonality_lowturn.csv"
    pd.DataFrame(rows, columns=hdr).to_csv(out, index=False)
    logger.info(f"\nSaved {out}")
    print("\nPromotion question: does any construction keep a positive net "
          "Sharpe at 20-30 bps while retaining most of the gross signal? "
          "If turnover is structural, all three churn alike and die together.")


if __name__ == "__main__":
    main()
