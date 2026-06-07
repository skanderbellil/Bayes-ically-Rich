#!/usr/bin/env python3
"""Honest, corrected PEAD backtest. Applies all five fixes and compares to REAL SPY.

Run: python3 research_pead_corrected.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.pead import corrected as C


# Alpaca-style long-only costs (bid-ask round-trip only; no commission, no borrow).
# One entry + one exit per quarter = 2 one-way spreads, 4x quarters/yr.
COST_ANNUAL = {
    "Mega Cap": 0.0024, "Large Cap": 0.0048, "Mid Cap": 0.0096,
    "Small Cap": 0.0192, "Micro Cap": 0.0360, "Nano Cap": 0.0720,
}


def equity(quarterly):
    return np.cumprod(1 + np.array(quarterly))


def main():
    panel = C.load_panel()
    spy = C.load_spy_quarterly()
    spy_q = spy.set_index("quarter")["spy_qret"]

    print("=" * 78)
    print("CORRECTED PEAD BACKTEST  (log->simple fixed, enter at t+1, real SPY)")
    print("=" * 78)

    # 1) Show the tradeability gap on small-cap top 25%, correct math, no costs.
    asb = C.backtest(panel, "Small Cap", "ret_t21", use_trailing=False, q=0.75)
    trd = C.backtest(panel, "Small Cap", "drift_t1_to_t21", use_trailing=False, q=0.75)
    print("\n[1] Small-cap top-25% SUE, correct return math, NO costs:")
    print(f"    As-backtested (incl. untradeable jump): {asb['annual_gross']*100:+.2f}%/yr  Sharpe {asb['sharpe']:.2f}  maxDD {asb['max_dd']*100:.1f}%")
    print(f"    TRADEABLE (enter t+1 close):            {trd['annual_gross']*100:+.2f}%/yr  Sharpe {trd['sharpe']:.2f}  maxDD {trd['max_dd']*100:.1f}%")

    # 2) Tradeable, trailing percentile (no in-quarter look-ahead), with Alpaca costs.
    print("\n[2] TRADEABLE + trailing percentile + Alpaca costs, by bucket:")
    print(f"    {'Bucket':12s} {'GrossAnn':>9s} {'NetAnn':>8s} {'Sharpe':>7s} {'maxDD':>7s} {'Qtrs':>5s} {'Pos':>5s}")
    results = {}
    for b in ["Mega Cap", "Large Cap", "Mid Cap", "Small Cap"]:
        r = C.backtest(panel, b, "drift_t1_to_t21", use_trailing=True, q=0.75,
                       cost_annual=COST_ANNUAL[b])
        results[b] = r
        print(f"    {b:12s} {r['annual_gross']*100:+8.2f}% {r['annual_net']*100:+7.2f}% "
              f"{r['sharpe']:7.2f} {r['max_dd']*100:6.1f}% {r['n_quarters']:5d} {r['avg_positions']:5.0f}")

    # Real SPY over the full span.
    spy_ann = (1 + spy_q.mean()) ** 4 - 1
    spy_sharpe = np.sqrt(4) * spy_q.mean() / spy_q.std()
    print(f"\n    {'SPY (real)':12s} {spy_ann*100:+8.2f}% {spy_ann*100:+7.2f}% {spy_sharpe:7.2f}    (buy & hold, 100% invested)")

    # 3) Honest verdict.
    best = max(results.items(), key=lambda kv: kv[1]["annual_net"])
    print("\n[3] HONEST VERDICT")
    print(f"    Best tradeable bucket: {best[0]} at {best[1]['annual_net']*100:+.2f}%/yr net, Sharpe {best[1]['sharpe']:.2f}")
    print(f"    Real SPY buy & hold:   {spy_ann*100:+.2f}%/yr, Sharpe {spy_sharpe:.2f}")
    gap = best[1]['annual_net'] - spy_ann
    print(f"    Net alpha vs SPY:      {gap*100:+.2f}%/yr  ->  " +
          ("BEATS buy & hold" if gap > 0 else "DOES NOT beat buy & hold"))
    print("    Caveat: strategy is only ~1 month/quarter invested (cash drag ignored above),")
    print("    and the live-only universe is survivorship-biased UPWARD (small-cap worst).")

    # Plot: tradeable small-cap vs as-backtested vs real SPY, same quarters where possible.
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(100_000 * equity(asb["quarterly"]), lw=2, color="#bbbbbb",
            label=f"As-backtested incl. jump (NOT tradeable): {asb['annual_gross']*100:+.1f}%/yr")
    sc_net = C.backtest(panel, "Small Cap", "drift_t1_to_t21", use_trailing=True, q=0.75,
                        cost_annual=COST_ANNUAL["Small Cap"])
    ax.plot(100_000 * equity(sc_net["quarterly"]), lw=2.5, color="#d62728",
            label=f"Tradeable small-cap, trailing+costs: {sc_net['annual_net']*100:+.1f}%/yr")
    ax.plot(100_000 * equity(spy_q.values), lw=2.5, color="#000000", ls="--",
            label=f"SPY buy & hold (real): {spy_ann*100:+.1f}%/yr")
    ax.set_yscale("log")
    ax.set_xlabel("Quarter (count; series span differ)", fontweight="bold")
    ax.set_ylabel("Growth of $100k (log)", fontweight="bold")
    ax.set_title("PEAD, honest accounting: the tradeable edge largely vanishes", fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    out = Path("results/pead/corrected_vs_spy.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
