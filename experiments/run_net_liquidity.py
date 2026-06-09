#!/usr/bin/env python3
"""
Net liquidity and the effect on equities.

Net liquidity (Fed assets − Treasury General Account − Reverse Repo) is a proxy
for the dollar liquidity sloshing into risk assets.  This study:

  1. Quantifies the relationship between net-liquidity changes and SPY returns
     (contemporaneous and lead-lag).
  2. Tests a simple liquidity-regime timing rule on SPY: hold equities while
     net liquidity is expanding, step to cash while it contracts.

Publication lag is respected — the FRED series are released with a delay, so
the signal is lagged a week before it drives any position (no lookahead).

    python experiments/run_net_liquidity.py            # offline (cached FRED)
    python experiments/run_net_liquidity.py --build     # refresh from FRED (network)
"""
import argparse
import logging
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tabulate import tabulate

import _bootstrap  # noqa: F401  (adds repo root to sys.path)
from posterioralpha.data import load_net_liquidity, load_etf_universe_prices
from posterioralpha.validation import compute_metrics

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S", stream=sys.stdout,
)
logger = logging.getLogger(__name__)
RESULTS_DIR = _bootstrap.ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RF = 0.04
PUB_LAG = 5          # business days: FRED release delay (avoid lookahead)
CHG_WIN = 65         # ~13-week change horizon for the liquidity signal
DAILY_RF = RF / 252


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true", help="refresh FRED data (network)")
    args = ap.parse_args()

    if args.build:
        from posterioralpha.data.macro import build_net_liquidity
        nl_df = build_net_liquidity()
    else:
        nl_df = load_net_liquidity()

    nl = nl_df["net_liquidity"]
    px = load_etf_universe_prices()
    spy, tlt = px["SPY"], px["TLT"]
    idx = (nl.dropna().index
           .intersection(spy.dropna().index)
           .intersection(tlt.dropna().index))
    nl, spy, tlt = nl.reindex(idx), spy.reindex(idx), tlt.reindex(idx)
    spy_ret, tlt_ret = spy.pct_change(), tlt.pct_change()

    print("\n" + "=" * 74)
    print(f"  NET LIQUIDITY vs EQUITIES  ·  {idx.min().date()} → {idx.max().date()}")
    print("  net liquidity = Fed assets (WALCL) − TGA (WTREGEN) − RRP (RRPONTSYD)")
    print("=" * 74)
    print(f"\n  net liquidity now: ${nl.iloc[-1]/1000:.2f}T   "
          f"({CHG_WIN}d change: {nl.diff(CHG_WIN).iloc[-1]:+,.0f}bn)")

    # ── 1. Lead-lag across horizons (which Δ-liquidity horizon leads SPY?) ────
    def corr(a, b):
        m = a.notna() & b.notna()
        return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 10 else float("nan")
    print("\nLead-lag: corr(net-liquidity change over h, SPY return over h)")
    hrows = []
    for h, lab in [(21, "~1m"), (42, "~2m"), (63, "~3m"), (126, "~6m"), (252, "~12m")]:
        dch = nl.diff(h)
        hrows.append([f"{lab} ({h}d)",
                      f"{corr(dch, spy.pct_change(h)):+.3f}",
                      f"{corr(dch, spy.pct_change(h).shift(-h)):+.3f}"])
    print(tabulate(hrows, headers=["horizon", "contemp", "next-period (lead)"],
                   tablefmt="github"))

    # ── 2. Strategies (all signals causal: publication lag + 1 day) ──────────
    d_nl = nl.diff(CHG_WIN)
    def causal(s):
        return s.shift(PUB_LAG).shift(1)
    z = ((d_nl - d_nl.rolling(756, min_periods=252).mean())
         / d_nl.rolling(756, min_periods=252).std())
    zc = causal(z)
    expanding = causal((d_nl > 0).astype(float)).fillna(0.0)

    # (a) Binary on/off: SPY when liquidity expanding, else cash.
    binary = pd.Series(np.where(expanding > 0, spy_ret, DAILY_RF),
                       index=idx, name="binary")

    # (b) Scaled exposure CENTRED AT 1.0 — stay invested, tilt 0–1.5x with the
    # liquidity-momentum z-score (the earlier 0.5-centred version under-invested).
    expo = (1.0 + 0.5 * zc).clip(0.0, 1.5).fillna(1.0)
    scaled = pd.Series(expo * spy_ret + (1 - expo) * DAILY_RF,
                       index=idx, name="scaled")

    # (c) Rotation: SPY when liquidity expanding, long-duration TLT when it
    # contracts (liquidity drains tend to favour duration over equities).
    rot = pd.Series(np.where(expanding > 0, spy_ret, tlt_ret),
                    index=idx, name="rotation")

    bh = spy_ret

    keys = ["CAGR", "Volatility", "Sharpe", "Sortino", "Max DD", "Calmar"]
    rows = [
        ["Binary on/off (SPY/cash)", binary],
        ["Scaled exposure (centred 1.0)", scaled],
        ["Rotation (SPY↔TLT)", rot],
        ["SPY buy & hold", bh],
    ]
    table = [[name] + [f"{compute_metrics(r.dropna(), rf=RF).get(k, float('nan')):.2f}" for k in keys]
             for name, r in rows]
    print("\n" + "=" * 74)
    print("  LIQUIDITY-DRIVEN STRATEGIES ON SPY  (signal lagged for publication delay)")
    print("=" * 74)
    print(tabulate(table, headers=["strategy"] + keys, tablefmt="github"))
    print(f"\n  binary: {float((expanding > 0).mean()):.0%} invested  |  "
          f"scaled: {float(expo.mean()):.2f}x avg exposure  |  "
          f"rotation: {float((expanding > 0).mean()):.0%} in SPY / rest in TLT   "
          f"(signal lagged {PUB_LAG}d)")
    strat, scaled = binary, scaled   # for the plot section below

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    eq_s = (1 + binary.fillna(0)).cumprod(); eq_z = (1 + scaled.fillna(0)).cumprod()
    eq_r = (1 + rot.fillna(0)).cumprod(); eq_b = (1 + bh.fillna(0)).cumprod()
    ax1.plot(eq_s.index, eq_s, label="Binary on/off", lw=1.1)
    ax1.plot(eq_z.index, eq_z, label="Scaled exposure (1.0)", lw=1.2)
    ax1.plot(eq_r.index, eq_r, label="Rotation SPY↔TLT", lw=1.4)
    ax1.plot(eq_b.index, eq_b, label="SPY buy & hold", lw=1.3, color="black", alpha=0.8)
    ax1.set_yscale("log"); ax1.set_ylabel("growth of $1 (log)")
    ax1.legend(); ax1.grid(alpha=0.3)
    ax1.set_title("Net-liquidity-driven strategies vs SPY buy & hold")
    ax2.plot(nl.index, nl / 1000, color="tab:green", lw=1.0)
    ax2.set_ylabel("net liquidity ($T)"); ax2.grid(alpha=0.3)
    fig.tight_layout()
    out = RESULTS_DIR / "net_liquidity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    logger.info("Saved: %s", out)
    print("\n✓  Done.")


if __name__ == "__main__":
    main()
