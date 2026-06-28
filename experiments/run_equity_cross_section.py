#!/usr/bin/env python3
"""
Exploratory cross-sectional study on the large US equity universe.

A big cross-section is what classic equity anomalies need.  This study runs a
textbook 12-1 cross-sectional **momentum** long/short on the ~500-name liquid
equity universe (financedatabase + yfinance), monthly rebalanced:

    formation : trailing 12 months, skipping the most recent month (reversal)
    long      : top-decile winners (equal-weighted)
    short     : bottom-decile losers (equal-weighted)
    WML       : winners − losers (dollar-neutral)

Reuses the shared ``research.wml_formation`` primitive.  Reports the long,
short, long/short, and equal-weight-universe legs vs an SPY benchmark, gross
and net of a simple turnover cost.  Runs offline on the bundled dataset.

    python experiments/run_equity_cross_section.py
"""
import logging
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tabulate import tabulate

import _bootstrap  # noqa: F401  (adds repo root to sys.path)
from posterioralpha.data import (
    load_equity_universe_returns,
    load_equity_universe_info,
    load_etf_universe_prices,
)
from posterioralpha.research import (
    wml_formation,
    average_pairwise_correlation,
    effective_breadth,
    transfer_coefficient,
)
from posterioralpha.validation import compute_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S", stream=sys.stdout,
)
logger = logging.getLogger(__name__)
RESULTS_DIR = _bootstrap.ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RF = 0.04
LOOKBACK = 252       # ~12 months formation
SKIP = 21            # skip most recent ~1 month (short-term reversal)
DECILE = 0.10        # top/bottom decile
TC = 0.0010          # 10 bps per unit one-way turnover


def run_xs_momentum(rets: pd.DataFrame) -> pd.DataFrame:
    """Monthly cross-sectional 12-1 momentum; returns daily leg returns + turnover."""
    month_ends = rets.resample("ME").last().index
    month_ends = month_ends[month_ends > rets.index[LOOKBACK + SKIP + 5]]

    legs = {k: {} for k in ("long", "short", "wml", "universe")}
    turn = {}                         # |Δ holdings| at each rebalance (long+short books)
    tc_reb = []                       # per-rebalance transfer coefficient (decile vs full rank)
    prev_long, prev_short = set(), set()

    for i, me in enumerate(month_ends[:-1]):
        hist = rets.loc[:me]
        n = hist.shape[1]
        k = max(1, int(round(n * DECILE)))
        winners, losers = wml_formation(hist, long_n=k, short_n=k,
                                         lookback=LOOKBACK, skip=SKIP)
        if not winners or not losers:
            continue

        # Transfer coefficient: the decile L/S book we actually trade vs the
        # unconstrained-ideal that rank-weights the FULL cross-section. Both
        # dollar-neutral and unit-gross, so TC = corr(w_actual, w_ideal) ∈ [-1,1].
        score = (1.0 + hist.iloc[-(LOOKBACK + SKIP):-SKIP]).prod() - 1.0
        score = score.dropna()
        if len(score) >= 2 * k + 1:
            ranks = score.rank()
            w_ideal = ranks - ranks.mean()
            w_ideal = w_ideal / w_ideal.abs().sum()           # dollar-neutral, unit gross
            w_actual = pd.Series(0.0, index=score.index)
            w_actual[winners] = 0.5 / k
            w_actual[losers] = -0.5 / k                        # unit gross, dollar-neutral
            tc_reb.append(transfer_coefficient(w_actual.to_numpy(), w_ideal.to_numpy()))

        nxt = month_ends[i + 1]
        hold = rets.index[(rets.index > me) & (rets.index <= nxt)]
        if len(hold) == 0:
            continue
        block = rets.loc[hold]

        long_r = block[winners].mean(axis=1)
        short_r = block[losers].mean(axis=1)
        for d in hold:
            legs["long"][d] = long_r.loc[d]
            legs["short"][d] = short_r.loc[d]
            legs["wml"][d] = long_r.loc[d] - short_r.loc[d]
            legs["universe"][d] = block.loc[d].mean()

        ws, ls = set(winners), set(losers)
        # one-way turnover of the two equal-weighted books (fraction of names rotated)
        turn[hold[0]] = (len(ws ^ prev_long) / (2 * k) + len(ls ^ prev_short) / (2 * k))
        prev_long, prev_short = ws, ls

    out = pd.DataFrame({k: pd.Series(v).sort_index() for k, v in legs.items()})
    out["turnover"] = pd.Series(turn).reindex(out.index).fillna(0.0)
    out.attrs["tc_weight"] = float(np.nanmean(tc_reb)) if tc_reb else float("nan")
    return out


def main() -> None:
    rets = load_equity_universe_returns().dropna(how="all")
    info = load_equity_universe_info()

    print("\n" + "=" * 74)
    print(f"  LARGE EQUITY UNIVERSE  ·  {rets.shape[1]} liquid US names  ·  "
          f"{rets.index.min().date()} → {rets.index.max().date()}")
    print("  source: financedatabase (universe + info) + yfinance (history)")
    print("=" * 74)
    if "sector" in info.columns:
        print("\nComposition by sector (financedatabase):")
        print(info["sector"].value_counts().to_string())

    logger.info("Running 12-1 cross-sectional momentum (monthly, decile L/S) …")
    bt = run_xs_momentum(rets)

    # Net-of-cost WML: charge TC on rebalance-day turnover.
    wml_net = bt["wml"].copy()
    wml_net.loc[bt["turnover"] > 0] -= TC * bt["turnover"][bt["turnover"] > 0]

    # SPY benchmark over the same window (from the ETF universe panel).
    try:
        spy = load_etf_universe_prices()["SPY"].pct_change().reindex(bt.index).dropna()
    except Exception:
        spy = None

    series = {
        "WML (L/S, gross)": bt["wml"],
        "WML (L/S, net)": wml_net,
        "Winners (long-only)": bt["long"],
        "Losers (short basket)": bt["short"],
        "Universe (equal-wt)": bt["universe"],
    }
    if spy is not None:
        series["SPY buy&hold"] = spy

    keys = ["CAGR", "Volatility", "Sharpe", "Sortino", "Max DD", "Calmar"]
    table = []
    for name, r in series.items():
        m = compute_metrics(r.dropna(), rf=RF)
        table.append([name] + [f"{m.get(k, float('nan')):.2f}" for k in keys])

    print("\n" + "=" * 74)
    print("  CROSS-SECTIONAL MOMENTUM  (12-1, monthly, top/bottom decile)")
    print("=" * 74)
    print(tabulate(table, headers=["leg"] + keys, tablefmt="github"))
    print(f"\n  avg one-way turnover/reb: {bt['turnover'][bt['turnover']>0].mean():.0%}"
          f"   (cost = {TC*1e4:.0f} bps)")

    # ----------------------------------------------------------------------
    # Fundamental-Law diagnostics: SR = TransferCoef · IC · √(effective breadth).
    # Two terms that silently inflate apparent Sharpe — is the breadth real,
    # and how much of the full ranking signal does the decile book transfer?
    # ----------------------------------------------------------------------
    n_uni = rets.shape[1]
    k = max(1, int(round(n_uni * DECILE)))
    nominal_book = 2 * k                                   # names in the L/S book
    avg_corr = average_pairwise_correlation(rets)
    _, br_eff, _ = effective_breadth(n=nominal_book, avg_corr=avg_corr)
    tc = bt.attrs.get("tc_weight", float("nan"))           # decile book vs full-rank ideal

    print("\n" + "=" * 74)
    print("  FUNDAMENTAL-LAW DIAGNOSTICS")
    print("=" * 74)
    print(f"  nominal book size (2 × decile)       : {nominal_book} names")
    print(f"  avg pairwise correlation (ρ̄)         : {avg_corr:+.3f}")
    print(f"  EFFECTIVE breadth  N/(1+(N-1)ρ̄)      : {br_eff:.1f} independent bets"
          f"   ({br_eff / nominal_book:.0%} of nominal)")
    print(f"  TRANSFER coefficient (decile vs rank): {tc:+.2f}"
          f"   (corr of traded weights vs full-rank ideal)")

    # Plot
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, r in series.items():
        eq = (1 + r.reindex(bt.index).fillna(0)).cumprod()
        ax.plot(eq.index, eq.values, label=name, lw=1.6 if "SPY" in name else 1.1,
                color="black" if "SPY" in name else None, alpha=0.9)
    ax.set_yscale("log")
    ax.set_title(f"Cross-sectional 12-1 momentum on {rets.shape[1]} US equities")
    ax.set_ylabel("growth of $1 (log)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = RESULTS_DIR / "equity_cross_section.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    logger.info("Saved: %s", out)
    print("\n✓  Done.")


if __name__ == "__main__":
    main()
