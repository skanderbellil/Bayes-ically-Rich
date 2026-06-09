#!/usr/bin/env python3
"""
Levered liquidity vote — can the overlay beat SPY in ABSOLUTE terms?

The continuous vote earns Sharpe ~0.71-0.75 at only ~0.10 vol — better quality
than SPY but lower octane.  The classic fix: lever the higher-Sharpe stream
back up to market-level risk.  Exposure here is the product of two dials:

  exposure_t = vote_t × min(target_vol / realized_vol_t, cap)   ≤ cap (2x)

  vote          : fit-free liquidity × VIX-TS × credit strength in [0,1]
                  (research.overlay.liquidity_vote)
  vol targeting : 21-day realized vol scaled to a 16% target (SPY-like), so
                  risk is spent when the regime is calm and cut when wild.

Costs are charged honestly:
  financing  : max(exposure−1, 0) × (RF + 50bp) / 252 per day  (margin/futures)
  cash yield : max(1−exposure, 0) × RF / 252 per day
  trading    : 2bp × |Δexposure| per day  (index futures/ETF)

Underlyings: SPY and QQQ.  Rolling 3-year OOS win rates and per-year table.

    python experiments/run_liquidity_levered.py
"""
import logging
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tabulate import tabulate

import _bootstrap  # noqa: F401
from posterioralpha.data import load_net_liquidity, load_etf_universe_prices
from posterioralpha.research import liquidity_vote
from posterioralpha.validation import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)
logger = logging.getLogger(__name__)
RESULTS_DIR = _bootstrap.ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
VIX_CSV = _bootstrap.ROOT / "datasets" / "vix_term_structure.csv"

RF = 0.04
DAILY_RF = RF / 252
BORROW_SPREAD = 0.005          # 50bp over RF on levered notional
TC = 0.0002                    # 2bp per unit turnover (futures/liquid ETF)
TARGET_VOL = 0.16
LEV_CAP = 2.0
VOL_WIN = 21
ROLL = 756


def apply_exposure(under_ret: pd.Series, expo: pd.Series):
    """Exposure series → net returns with financing, cash and trading costs."""
    expo = expo.clip(0.0, LEV_CAP).fillna(0.0)
    gross = expo * under_ret
    financing = np.maximum(expo - 1.0, 0.0) * (RF + BORROW_SPREAD) / 252
    cash = np.maximum(1.0 - expo, 0.0) * DAILY_RF
    costs = TC * expo.diff().abs().fillna(0.0)
    return pd.Series(gross - financing + cash - costs, index=under_ret.index), expo


def levered_returns(under_ret: pd.Series, vote: pd.Series):
    """Vote × vol-target exposure (defensive composition — both dials cut)."""
    realized = under_ret.rolling(VOL_WIN).std() * np.sqrt(252)
    scalar = (TARGET_VOL / realized).clip(upper=LEV_CAP).shift(1)
    return apply_exposure(under_ret, vote * scalar)


def aggressive_returns(under_ret: pd.Series, vote: pd.Series):
    """exposure = 2×vote: neutral → 1x market, strong risk-on → 2x, off → 0.

    Fit-free symmetric mapping — the configuration that can actually beat the
    market in absolute terms (average exposure > 1) while still de-risking
    when the three signals turn."""
    return apply_exposure(under_ret, 2.0 * vote)


def braked_returns(under_ret: pd.Series, vote: pd.Series):
    """exposure = min(2×vote, vol brake): conviction leverage + crisis brake.

    The brake caps exposure at 2 × (expanding 'normal' vol / current 21d vol) —
    fully causal, no fitted numbers.  In calm regimes it never binds (cap ≥ 2);
    when vol runs above its own history (2020, 2022) leverage is cut
    proportionally.  Standard vol-control practice, not return-fitting."""
    rolling = under_ret.rolling(VOL_WIN).std() * np.sqrt(252)
    normal = under_ret.expanding(252).std() * np.sqrt(252)
    brake = (2.0 * normal / rolling).shift(1)
    return apply_exposure(under_ret, np.minimum(2.0 * vote, brake))


def convex_returns(under_ret: pd.Series, vote: pd.Series):
    """exposure = 2×vote²: leverage requires conviction.

    Convex a-priori mapping — neutral signals (0.5) → 0.5x (defensive),
    only strong agreement (→1) earns 2x.  Aims to keep the linear mapping's
    upside while refusing to stay levered through grinding bear markets
    where the vote hovers near neutral."""
    return apply_exposure(under_ret, 2.0 * vote ** 2)


def roll_sharpe(r: pd.Series) -> pd.Series:
    ex = r - DAILY_RF
    return (ex.rolling(ROLL).mean() / ex.rolling(ROLL).std()) * np.sqrt(252)


def main() -> None:
    nl = load_net_liquidity()["net_liquidity"]
    px = load_etf_universe_prices()
    vix = pd.read_csv(VIX_CSV, parse_dates=["Date"], index_col="Date").sort_index()
    idx = (nl.dropna().index.intersection(px[["SPY", "QQQ", "HYG", "IEF"]].dropna().index)
           .intersection(vix.dropna().index))
    nl, vix = nl.reindex(idx), vix.reindex(idx)

    vote = liquidity_vote(nl, vix["VIX"], vix["VIX3M"],
                          px["HYG"].reindex(idx), px["IEF"].reindex(idx))["exposure"]

    spy_ret = px["SPY"].reindex(idx).pct_change()
    qqq_ret = px["QQQ"].reindex(idx).pct_change()

    lev_spy, expo_spy = levered_returns(spy_ret, vote)
    lev_qqq, expo_qqq = levered_returns(qqq_ret, vote)
    agg_spy, aexpo_spy = aggressive_returns(spy_ret, vote)
    agg_qqq, aexpo_qqq = aggressive_returns(qqq_ret, vote)
    cvx_spy, cexpo_spy = convex_returns(spy_ret, vote)
    cvx_qqq, cexpo_qqq = convex_returns(qqq_ret, vote)
    brk_spy, bexpo_spy = braked_returns(spy_ret, vote)
    brk_qqq, bexpo_qqq = braked_returns(qqq_ret, vote)
    unlev_spy = vote.fillna(0) * spy_ret + (1 - vote.fillna(0)) * DAILY_RF

    rows = [
        ["SPY buy & hold", spy_ret],
        ["Unlevered vote on SPY", unlev_spy],
        ["Vote × vol-target on SPY (defensive)", lev_spy],
        ["2×vote on SPY (aggressive)", agg_spy],
        ["2×vote² on SPY (convex)", cvx_spy],
        ["2×vote + vol brake on SPY", brk_spy],
        ["QQQ buy & hold", qqq_ret],
        ["Vote × vol-target on QQQ (defensive)", lev_qqq],
        ["2×vote on QQQ (aggressive)", agg_qqq],
        ["2×vote² on QQQ (convex)", cvx_qqq],
        ["2×vote + vol brake on QQQ", brk_qqq],
    ]

    print("\n" + "=" * 80)
    print(f"  LEVERED LIQUIDITY VOTE  ·  {idx.min().date()} → {idx.max().date()}")
    print(f"  exposure = vote × vol-target({TARGET_VOL:.0%}), cap {LEV_CAP:.0f}x  ·  "
          f"financing RF+{BORROW_SPREAD:.1%}, TC {TC*1e4:.0f}bp")
    print("=" * 80)
    keys = ["CAGR", "Volatility", "Sharpe", "Sortino", "Max DD", "Calmar"]
    table = [[n] + [f"{compute_metrics(r.dropna(), rf=RF).get(k, float('nan')):.2f}" for k in keys]
             for n, r in rows]
    print(tabulate(table, headers=["strategy"] + keys, tablefmt="github"))
    print(f"\n  avg exposure — defensive SPY {float(expo_spy.mean()):.2f}x | "
          f"aggressive SPY {float(aexpo_spy.mean()):.2f}x "
          f"(max {float(aexpo_spy.max()):.2f}x) | "
          f"aggressive QQQ {float(aexpo_qqq.mean()):.2f}x")

    # ── Rolling OOS ───────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"  ROLLING OOS  ·  {ROLL}d (~3y) windows")
    print("=" * 80)
    for name, s, b in [("2×vote SPY vs SPY", agg_spy, spy_ret),
                       ("braked SPY vs SPY", brk_spy, spy_ret),
                       ("2×vote QQQ vs SPY", agg_qqq, spy_ret),
                       ("braked QQQ vs SPY", brk_qqq, spy_ret),
                       ("braked QQQ vs QQQ", brk_qqq, qqq_ret)]:
        ds = (roll_sharpe(s) - roll_sharpe(b)).dropna()
        # CAGR edge per window too — absolute-return comparison
        ws = (1 + s).rolling(ROLL).apply(np.prod, raw=True) ** (252 / ROLL) - 1
        wb = (1 + b).rolling(ROLL).apply(np.prod, raw=True) ** (252 / ROLL) - 1
        dc = (ws - wb).dropna()
        print(f"  {name:28s} Sharpe win {float((ds > 0).mean()):.0%} "
              f"(median {float(ds.median()):+.2f})   "
              f"CAGR win {float((dc > 0).mean()):.0%} "
              f"(median {float(dc.median()):+.1%})")

    # per-year
    yr = pd.DataFrame({"2x QQQ": agg_qqq, "braked QQQ": brk_qqq,
                       "braked SPY": brk_spy, "SPY": spy_ret}).dropna()
    ann = yr.groupby(yr.index.year).apply(lambda d: (1 + d).prod() - 1)
    ann["edge brkQQQ"] = ann["braked QQQ"] - ann["SPY"]
    print("\nPer-year returns:")
    print(tabulate((ann * 100).round(1),
                   headers=["year", "2x QQQ %", "braked QQQ %", "braked SPY %",
                            "SPY %", "edge brkQQQ %"], tablefmt="github"))

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    for n, r in rows:
        eq = (1 + r.reindex(idx).fillna(0)).cumprod()
        ax1.plot(eq.index, eq.values, lw=1.5 if "buy" in n else 1.2,
                 color="black" if n.startswith("SPY") else None, alpha=0.85, label=n)
    ax1.set_yscale("log"); ax1.set_ylabel("growth of $1 (log)")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    ax1.set_title("Levered fit-free liquidity vote vs buy & hold")
    ax2.plot(aexpo_spy.index, aexpo_spy.values, lw=0.8, color="tab:purple")
    ax2.axhline(1.0, color="grey", lw=0.6)
    ax2.set_ylabel("exposure (x)"); ax2.grid(alpha=0.3)
    fig.tight_layout()
    out = RESULTS_DIR / "liquidity_levered.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    logger.info("Saved: %s", out)
    print("\n✓  Done.")


if __name__ == "__main__":
    main()
