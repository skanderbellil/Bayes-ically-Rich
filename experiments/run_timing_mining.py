#!/usr/bin/env python3
"""
Timing-dial mining on a single underlying (default QQQ).

The cross-sectional miner showed daily stock-picking nets out to nothing on
this data; the champion-stack thread showed the edge lives in *timing* a
high-beta underlying.  This experiment turns the automated miner on that
question: evolve exposure dials e_t ∈ [0, 1] built from the FRED macro panel
(curve, credit, VIX, dollar, claims, …), price trend/momentum, realized vol,
and BOCPD regime age — then validate the survivors on an untouched holdout
with the same randomized gauntlet (block bootstrap, circular permutation,
Deflated Sharpe charged for every candidate examined).

VALIDATED here requires beating buy & hold's holdout Sharpe with statistical
support — holding the underlying is the null strategy, not cash.

Usage
-----
  python experiments/run_timing_mining.py                     # QQQ, 1 seed
  python experiments/run_timing_mining.py --underlying SPY
  python experiments/run_timing_mining.py --seeds 5           # recurrence
  python experiments/run_timing_mining.py --quick             # smoke test
"""
import _bootstrap  # noqa: F401  (adds repo root to sys.path, loads .env)

import argparse
import logging

import pandas as pd

from posterioralpha.data import load_etf_universe_prices, load_fred_macro
from posterioralpha.mining.timing import TimingConfig, TimingMiner

logging.basicConfig(level=logging.INFO, format="%(message)s")

OUT_DIR = _bootstrap.ROOT / "results" / "alpha_mining"


def main():
    ap = argparse.ArgumentParser(description="Timing-dial mining")
    ap.add_argument("--underlying", default="QQQ")
    ap.add_argument("--population", type=int, default=40)
    ap.add_argument("--generations", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--tc-bps", type=float, default=2.0)
    ap.add_argument("--no-macro", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    cfg = TimingConfig(
        population=args.population,
        generations=args.generations,
        tc=args.tc_bps / 1e4,
        seed=args.seed,
    )
    if args.quick:
        cfg.population, cfg.generations = 12, 3
        cfg.n_boot, cfg.n_perm, cfg.n_finalists = 100, 50, 5

    etf = load_etf_universe_prices()
    if args.underlying not in etf.columns:
        raise SystemExit(f"{args.underlying} not in the ETF universe")
    prices = etf[args.underlying].dropna()

    macro = None
    if not args.no_macro:
        try:
            macro = load_fred_macro()
        except FileNotFoundError:
            print("Macro panel not found (run experiments/build_fred_macro.py) "
                  "— price-only dials")

    seeds = [args.seed + i for i in range(args.seeds)]
    print(f"Underlying: {args.underlying} — {len(prices)} days "
          f"({prices.index[0].date()} → {prices.index[-1].date()})")
    if macro is not None:
        print(f"Macro panel: {macro.shape[1]} FRED series — macro dials active")
    print(f"Mining: population={cfg.population}, generations={cfg.generations}, "
          f"seeds={seeds}, tc={cfg.tc * 1e4:.0f} bps\n")

    boards = []
    n_trials_total = 0
    for s in seeds:
        cfg.seed = s
        miner = TimingMiner(prices, cfg, macro=macro)
        lb, _ = miner.run()
        lb.insert(0, "seed", s)
        n_trials_total += lb.attrs.get("n_trials", 0)
        boards.append(lb)

    leaderboard = pd.concat(boards, ignore_index=True)
    bh = boards[0].attrs.get("bh_sharpe")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"timing_{args.underlying}"
    leaderboard.to_csv(OUT_DIR / f"leaderboard_{tag}.csv", index=False)

    print(f"\n{'=' * 110}\nTIMING HOLDOUT LEADERBOARD — {args.underlying} "
          f"(buy & hold holdout Sharpe: {bh:.2f}; {n_trials_total} candidates "
          f"examined across {len(seeds)} seed(s))\n" + "=" * 110)
    print(leaderboard.to_string(index=False))

    if len(seeds) > 1:
        top5 = leaderboard.sort_values("holdout_sharpe", ascending=False) \
                          .groupby("seed").head(5)
        rec = (top5.groupby("family")
                   .agg(seeds_in_top5=("seed", "nunique"),
                        median_holdout_sharpe=("holdout_sharpe", "median"))
                   .sort_values(["seeds_in_top5", "median_holdout_sharpe"],
                                ascending=False))
        print("\nFAMILY RECURRENCE (top-5 per seed)")
        print(rec.to_string())
        rec.to_csv(OUT_DIR / f"family_recurrence_{tag}.csv")

    print(f"\nSaved to {OUT_DIR}/: leaderboard_{tag}.csv")


if __name__ == "__main__":
    main()
