#!/usr/bin/env python3
"""
Polymarket — does a sharp UPSIDE vol move predict the actual outcome?
=====================================================================

An event study on resolved prediction markets, cross-fertilising the repo's
volatility / change-point machinery onto a different asset class. For every
sampled (market, day) we measure — causally, from prices up to that day — how
*sharp* and *upside-tilted* the market's recent log-odds churn has been, then
bucket those episodes and read off two things the brackets can't fake:

  * the **actual Yes-resolution rate** of the markets in each bracket, and
  * the **calibration residual** = yes_rate − mean price  (predictive content
    *beyond* the price level — a market at 0.9 resolves Yes ~90% anyway), and
  * the **forward log-odds drift** (does a sharp up-move keep going or revert?).

Metrics tested as the bracketing variable (``--by``):
  upside_vol   — RMS of the positive log-odds moves (upside semivol)   [default]
  sharp_up     — the single largest positive log-odds jump in the window
  vol_skew     — upside_vol − downside_vol (churn tilted up vs down)
  bocpd_cp     — BOCPD changepoint probability (repo's online detector)

Data is pulled live from Polymarket (Gamma metadata + settled outcome + CLOB
price history) and cached under ``data/raw/polymarket/``.

Usage
-----
  python experiments/run_polymarket_vol_outcome.py --refresh        # first run: pull labels
  python experiments/run_polymarket_vol_outcome.py --by sharp_up
  python experiments/run_polymarket_vol_outcome.py --n-markets 300 --vol-window 7
"""
import argparse
import logging

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate

import _bootstrap  # noqa: F401  (adds repo root to sys.path)
from posterioralpha.polymarket import (
    bracket_outcomes,
    build_event_table,
    build_price_panel,
    resolution_outcomes,
)
from posterioralpha.polymarket.paths import RESULTS_DIR, ensure_dirs

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-5s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

METRICS = ["upside_vol", "sharp_up", "vol_skew", "bocpd_cp"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-markets",  type=int,   default=250,    help="markets to pull")
    ap.add_argument("--min-volume", type=float, default=50_000, help="min lifetime USD volume")
    ap.add_argument("--by",         type=str,   default="upside_vol", choices=METRICS,
                    help="bracketing metric (default upside_vol)")
    ap.add_argument("--n-brackets", type=int,   default=5,      help="number of quantile brackets")
    ap.add_argument("--vol-window", type=int,   default=10,     help="vol/sharp-move window (days)")
    ap.add_argument("--fwd-horizon",type=int,   default=10,     help="forward drift horizon (days)")
    ap.add_argument("--sample-step",type=int,   default=3,      help="thin each market's days by N")
    ap.add_argument("--refresh",    action="store_true",        help="re-pull live data + labels")
    ap.add_argument("--no-plots",   action="store_true",        help="skip the chart")
    args = ap.parse_args()

    ensure_dirs()
    print("""
╔══════════════════════════════════════════════════════════╗
║   P O L Y M A R K E T   ·   sharp move → actual outcome    ║
║   are upside volatility spikes informative or overreaction? ║
╚══════════════════════════════════════════════════════════╝""")

    # ── Stage 1: data + resolution labels ─────────────────────────────────
    panel, meta = build_price_panel(
        n_markets=args.n_markets, min_volume=args.min_volume,
        closed=True, refresh=args.refresh,
    )
    outcomes = resolution_outcomes(meta, panel)
    logger.info("Panel %d days × %d markets · %d with a clean Yes/No label "
                "(base Yes-rate %.1f%%)",
                panel.shape[0], panel.shape[1], len(outcomes), 100 * outcomes.mean())

    # ── Stage 2+: event table ─────────────────────────────────────────────
    events = build_event_table(
        panel, outcomes,
        vol_window=args.vol_window, fwd_horizon=args.fwd_horizon,
        sample_step=args.sample_step,
    )
    if events.empty:
        logger.error("No episodes built — try --refresh or a larger --n-markets.")
        return

    # ── Headline bracket table on the chosen metric ───────────────────────
    tbl = bracket_outcomes(events, by=args.by, n_brackets=args.n_brackets)

    def _fmt(df: pd.DataFrame, metric: str) -> list:
        return [[
            r["bracket"], int(r["n"]), f"{r[metric]:>7.3f}",
            f"{r['mean_price']:>6.2f}", f"{r['yes_rate']:>7.1%}",
            f"{r['calib_resid']:>+7.1%}", f"{r['fwd_dlogodds']:>+7.3f}",
        ] for _, r in df.iterrows()]

    print("\n" + "═" * 80)
    print(f"  RESOLUTION OUTCOME BY {args.by.upper()} BRACKET  "
          f"(window={args.vol_window}d, fwd={args.fwd_horizon}d, {len(events)} episodes)")
    print("═" * 80)
    print(tabulate(_fmt(tbl, args.by),
                   headers=["bracket", "n", args.by, "price", "yes_rate",
                            "calib_resid", "fwd_Δz"],
                   tablefmt="rounded_grid"))
    print("  calib_resid = yes_rate − mean price  (predictive content beyond level)")
    print("  fwd_Δz      = mean forward log-odds drift  (+ continues up, − reverts)")

    # ── Compact view across all four metrics (Q-top vs Q-bottom spread) ───
    print("\n  Top-vs-bottom bracket spread, each metric:")
    summary = []
    for metric in METRICS:
        if metric not in events.columns:
            continue
        b = bracket_outcomes(events, by=metric, n_brackets=args.n_brackets)
        if b.empty:
            continue
        top, bot = b.iloc[-1], b.iloc[0]
        summary.append([
            metric,
            f"{bot['calib_resid']:>+6.1%}", f"{top['calib_resid']:>+6.1%}",
            f"{top['calib_resid'] - bot['calib_resid']:>+6.1%}",
            f"{bot['fwd_dlogodds']:>+6.3f}", f"{top['fwd_dlogodds']:>+6.3f}",
        ])
    print(tabulate(summary,
                   headers=["metric", "calib Q1", "calib Q5", "Q5−Q1",
                            "fwdΔz Q1", "fwdΔz Q5"],
                   tablefmt="rounded_grid"))

    # ── Plot ──────────────────────────────────────────────────────────────
    if not args.no_plots:
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
        x = np.arange(len(tbl))
        axL.bar(x - 0.2, tbl["mean_price"], width=0.4, label="mean price", color="tab:gray")
        axL.bar(x + 0.2, tbl["yes_rate"],   width=0.4, label="actual Yes-rate", color="tab:green")
        axL.set_xticks(x); axL.set_xticklabels(tbl["bracket"])
        axL.set_title(f"Yes-rate vs price by {args.by} bracket")
        axL.set_ylabel("probability"); axL.legend(); axL.grid(alpha=0.25, axis="y")

        axR.axhline(0, color="k", lw=0.6, alpha=0.5)
        axR.bar(x, tbl["fwd_dlogodds"], color="tab:blue")
        axR.set_xticks(x); axR.set_xticklabels(tbl["bracket"])
        axR.set_title(f"forward {args.fwd_horizon}d log-odds drift by {args.by} bracket")
        axR.set_ylabel("Δ log-odds  (+ momentum / − reversion)")
        axR.grid(alpha=0.25, axis="y")

        fig.tight_layout()
        out = RESULTS_DIR / f"vol_outcome_{args.by}.png"
        fig.savefig(out, dpi=130)
        logger.info("Saved chart → %s", out)

    print("\n✓  Sharp-move → outcome study complete.")


if __name__ == "__main__":
    main()
