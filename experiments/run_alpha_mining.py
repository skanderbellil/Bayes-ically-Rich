#!/usr/bin/env python3
"""
Automated alpha mining — the research loop, automated.

The miner does what this experiments/ directory does by hand, in a loop:

  1. GENERATE   — sample candidate alphas from 9 parameterized signal
                  families (momentum, short-term & idiosyncratic reversal,
                  vol-scaled momentum, low-vol, channel breakout, MA cross,
                  skewness, BOCPD regime-gated momentum).
  2. ITERATE    — an evolutionary loop (tournament selection / crossover /
                  mutation / random immigrants) searches the joint space of
                  signal parameters + portfolio construction (rank vs
                  quantile weighting, smoothing) over several generations.
  3. RANDOMIZE  — fitness is scored on freshly randomized purged 1-year
                  windows redrawn every generation (mean window Sharpe minus
                  a dispersion penalty), so no candidate can overfit a fixed
                  split.
  4. VALIDATE   — the most recent 30% of history is an untouched holdout.
                  Finalists face a statistical gauntlet there:
                    • stationary block bootstrap → P(true Sharpe ≤ 0)
                    • circular permutation test  → P(no timing skill)
                    • Deflated Sharpe Ratio      → charged for EVERY unique
                      candidate the search examined (multiple testing).

Portfolios are dollar-neutral long-short, gross = 1, 1-day execution lag,
10 bps per unit one-way turnover. The expected outcome is that MOST mined
alphas are REJECTED — the system's value is that it says so honestly.

⚠️ the bundled equity universe is current-membership (survivorship-biased);
dollar-neutral L/S mutes but does not remove the bias.

Usage
-----
  python experiments/run_alpha_mining.py                     # 500 equities
  python experiments/run_alpha_mining.py --universe etf      # 250 ETFs
  python experiments/run_alpha_mining.py --universe sp500    # ~100 S&P names
  python experiments/run_alpha_mining.py --quick             # smoke test
  python experiments/run_alpha_mining.py --population 60 --generations 8
"""
import _bootstrap  # noqa: F401  (adds repo root to sys.path)

import argparse
import logging

import pandas as pd

from posterioralpha.data import (
    load_equity_universe_prices,
    load_etf_universe_prices,
    load_fred_macro,
    load_sp500_prices,
)
from posterioralpha.mining import AlphaMiner, MinerConfig

logging.basicConfig(level=logging.INFO, format="%(message)s")

OUT_DIR = _bootstrap.ROOT / "results" / "alpha_mining"

LOADERS = {
    "equity": load_equity_universe_prices,
    "etf":    load_etf_universe_prices,
    "sp500":  load_sp500_prices,
}


def load_prices(universe: str, min_coverage: float = 0.90) -> pd.DataFrame:
    prices = LOADERS[universe]()
    keep = prices.columns[prices.notna().mean() >= min_coverage]
    return prices[keep]


def write_report(leaderboard: pd.DataFrame, cfg: MinerConfig,
                 history: pd.DataFrame, prices: pd.DataFrame,
                 universe: str) -> str:
    ho_start = prices.index[int(len(prices) * (1 - cfg.holdout_frac))].date()
    lines = [
        "# Automated Alpha Mining Report",
        "",
        f"- Universe: `{universe}` — {prices.shape[1]} assets, "
        f"{prices.index[0].date()} → {prices.index[-1].date()}",
        f"- Search: population {cfg.population} × {cfg.generations} generations "
        f"({leaderboard.attrs.get('n_trials', '?')} unique candidates examined)",
        f"- Fitness: mean Sharpe − 0.5·std over {cfg.n_val_windows} randomized "
        f"purged {cfg.val_window_len}-day windows (redrawn each generation)",
        f"- Holdout (untouched during search): {ho_start} → "
        f"{prices.index[-1].date()}",
        f"- Costs: {cfg.tc * 1e4:.0f} bps per unit one-way turnover, "
        "1-day execution lag, dollar-neutral, gross = 1",
        "",
        "## Leaderboard (holdout)",
        "",
        leaderboard.to_markdown(index=False),
        "",
        "## Search progress",
        "",
        history.to_markdown(index=False),
        "",
        "Verdicts — VALIDATED: holdout Sharpe > 0, bootstrap p < 0.10, "
        "permutation p < 0.10 and DSR > 0.50. WEAK: positive holdout Sharpe "
        "with partial statistical support. REJECTED: everything else "
        "(the expected fate of most mined alphas).",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Automated alpha mining")
    ap.add_argument("--universe", choices=sorted(LOADERS), default="equity")
    ap.add_argument("--population", type=int, default=40)
    ap.add_argument("--generations", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--seeds", type=int, default=1,
                    help="run N seeds (seed, seed+1, …) and aggregate: "
                         "families that recur near the top across seeds are "
                         "evidence; one-seed wonders are noise")
    ap.add_argument("--tc-bps", type=float, default=10.0)
    ap.add_argument("--no-macro", action="store_true",
                    help="disable the FRED-panel macro-gated families")
    ap.add_argument("--quick", action="store_true",
                    help="small run for a fast smoke test")
    args = ap.parse_args()

    cfg = MinerConfig(
        population=args.population,
        generations=args.generations,
        tc=args.tc_bps / 1e4,
        seed=args.seed,
    )
    if args.quick:
        cfg.population, cfg.generations = 12, 3
        cfg.n_boot, cfg.n_perm, cfg.n_finalists = 100, 50, 5

    prices = load_prices(args.universe)
    print(f"Universe: {args.universe} — {prices.shape[1]} assets × "
          f"{len(prices)} days ({prices.index[0].date()} → "
          f"{prices.index[-1].date()})")

    macro = None
    if not args.no_macro:
        try:
            macro = load_fred_macro()
            print(f"Macro panel: {macro.shape[1]} FRED series "
                  f"(→ {macro.index[-1].date()}) — macro-gated families active")
        except FileNotFoundError:
            print("Macro panel not found (run experiments/build_fred_macro.py) "
                  "— price-only families")
    seeds = [args.seed + i for i in range(args.seeds)]
    print(f"Mining: population={cfg.population}, generations={cfg.generations}, "
          f"seeds={seeds}\n")

    boards, histories = [], []
    n_trials_total = 0
    for s in seeds:
        cfg.seed = s
        miner = AlphaMiner(prices, cfg, macro=macro)
        lb, _ = miner.run()
        lb.insert(0, "seed", s)
        n_trials_total += lb.attrs.get("n_trials", 0)
        boards.append(lb)
        h = pd.DataFrame(miner.history)
        h.insert(0, "seed", s)
        histories.append(h)

    leaderboard = pd.concat(boards, ignore_index=True)
    leaderboard.attrs["n_trials"] = n_trials_total
    history = pd.concat(histories, ignore_index=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(OUT_DIR / f"leaderboard_{args.universe}.csv", index=False)
    history.to_csv(OUT_DIR / f"search_history_{args.universe}.csv", index=False)
    report = write_report(leaderboard, cfg, history, prices, args.universe)
    (OUT_DIR / f"report_{args.universe}.md").write_text(report)

    print(f"\n{'=' * 100}\nHOLDOUT LEADERBOARD "
          f"({n_trials_total} candidates examined across {len(seeds)} seed(s); "
          "DSR is charged per seed)\n" + "=" * 100)
    print(leaderboard.to_string(index=False))

    if len(seeds) > 1:
        # recurrence: how often each family makes a seed's top 5 by holdout
        top5 = leaderboard.sort_values("holdout_sharpe", ascending=False) \
                          .groupby("seed").head(5)
        rec = (top5.groupby("family")
                   .agg(seeds_in_top5=("seed", "nunique"),
                        median_holdout_sharpe=("holdout_sharpe", "median"))
                   .sort_values(["seeds_in_top5", "median_holdout_sharpe"],
                                ascending=False))
        print(f"\nFAMILY RECURRENCE (top-5 by holdout Sharpe, per seed) — "
              f"recurring families are evidence, one-seed wonders are noise")
        print(rec.to_string())
        rec.to_csv(OUT_DIR / f"family_recurrence_{args.universe}.csv")

    print(f"\nSaved to {OUT_DIR}/: leaderboard_{args.universe}.csv, "
          f"search_history_{args.universe}.csv, report_{args.universe}.md")


if __name__ == "__main__":
    main()
