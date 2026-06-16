#!/usr/bin/env python3
"""
Retroactive DSR — paying the multiple-testing debt on the hand-built families.

Idea (roadmap Idea 4)
---------------------
The miner runs every candidate through the deflated-Sharpe gauntlet, but the
program's two flagship HAND-BUILT families never paid that tax: each version
was judged in its own experiment, one at a time. This study runs
`dsr_report` (Bailey & López de Prado: P(true SR > 0 | best of N trials))
retroactively over both:

  A. The 5-ETF allocation family (2016→2024, the v1→v4 arc): amr, amr_no_vt,
     inv_vol, bocpd_amr, bocpd_amr_v2/v3/v4, hmm3_amr, bayesian_hmm — re-run
     here under the exact published config (SHARED_AMR of run_bocpd_v*).
  B. The champion-stack ladder on QQQ (2011-10→, from
     results/champion_stack_returns.csv — run run_champion_stack.py once):
     the 6 ablation/leave-one-out variants + the retail QLD+UUP wrapper.

Each family's nominal trial count is what the dict contains; the honest
count is larger (sub-variants, parameter nudges, the governor/layer studies
that fed the ladder), so the headline of each part is a SENSITIVITY CURVE:
the best variant's DSR as the assumed trial count grows. The question is
not "is the Sharpe real at face value" but "how many historical trials does
it take to kill it".

Honesty box
-----------
• Sharpes are on EXCESS returns (rf 4%) — stricter than the miner's raw
  convention; the strategies hold cash/UUP slack, so total-return Sharpes
  would flatter them.
• Trial-Sharpe variance is estimated from the family members themselves;
  with ≤9 members it is noisy. The sensitivity curve holds it fixed while
  n grows — conservative if later trials were more dispersed.
• This study adds no new trials: it evaluates frozen variants. The two
  families' returns overlap with everything previously published.
"""
import logging
import sys

import numpy as np
import pandas as pd
from tabulate import tabulate

import _bootstrap  # noqa: F401
from posterioralpha.backtest.amr import AMR_STRATEGIES, run_amr_backtest
from posterioralpha.backtest.bayesian import run_backtest
from posterioralpha.data.loaders import load_portfolio_prices
from posterioralpha.mining.validation import deflated_sharpe_ratio
from posterioralpha.validation.robustness import dsr_report

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)
RESULTS_DIR = _bootstrap.ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
CHAMPION_CSV = RESULTS_DIR / "champion_stack_returns.csv"

RF = 0.04
DAILY_RF = RF / 252
BT_START, BT_END = "2016-01-01", "2024-12-31"
SHARED_AMR = dict(lookback=252, vol_window=21, target_vol=0.10,
                  leverage_cap=1.50, max_weight=0.40, l2_reg=0.001, tc=0.0005)
SHARED_BAY = dict(rebalance_freq="ME", min_history=252, recent_window=60,
                  ewma_halflife=30, rf=RF, sensitivity=1.5,
                  max_weight=0.40, tc=0.001)
TRIAL_GRID = (20, 40, 80)
CHAMPION_BENCHMARKS = ("SPY buy & hold", "QQQ buy & hold")


def report(name: str, excess: dict[str, pd.Series]
           ) -> tuple[pd.DataFrame, float, pd.Series]:
    rep = dsr_report(excess)
    print(f"\n{name} — DSR at nominal n_trials = {len(excess)} "
          f"(excess returns, rf {RF:.0%}):")
    print(tabulate(rep, headers=rep.columns, tablefmt="github",
                   floatfmt=".3f", showindex=False))

    daily_srs = [float(r.dropna().mean() / (r.dropna().std() + 1e-12))
                 for r in excess.values()]
    sr_var = float(np.var(daily_srs, ddof=1))
    best = rep.iloc[0]["strategy"]
    r = excess[best].dropna().values
    curve = [(n, deflated_sharpe_ratio(r, n, sr_var))
             for n in (len(excess), *TRIAL_GRID)]
    print(f"\n  best variant '{best}' — DSR vs assumed trial count "
          f"(trial-Sharpe var {sr_var:.2e} held fixed):")
    print("  " + "   ".join(f"n={n}: {d:.3f}" for n, d in curve))

    rep["family"] = name
    for n, d in curve:
        rep.loc[rep["strategy"] == best, f"dsr_n{n}"] = d
    return rep, sr_var, excess[best].dropna()


def main():
    # ── A. the 5-ETF allocation family ────────────────────────────────────
    returns = load_portfolio_prices().pct_change().dropna()
    logger.info(f"5-ETF panel: {list(returns.columns)}, "
                f"{returns.index[0].date()} → {returns.index[-1].date()}; "
                f"BT window {BT_START} → {BT_END}")
    fam_a = {}
    for strat in AMR_STRATEGIES:
        logger.info(f"  running {strat} …")
        res = run_amr_backtest(returns, strategy=strat, **SHARED_AMR)
        fam_a[strat] = res.returns.loc[BT_START:BT_END] - DAILY_RF
    logger.info("  running bayesian_hmm …")
    fam_a["bayesian_hmm"] = run_backtest(
        returns, strategy="bayesian_hmm",
        **SHARED_BAY).returns.loc[BT_START:BT_END] - DAILY_RF
    rep_a, var_a, _ = report("Family A: 5-ETF v1→v4 arc (2016-2024)", fam_a)

    # ── B. the champion-stack ladder ──────────────────────────────────────
    if not CHAMPION_CSV.exists():
        sys.exit("results/champion_stack_returns.csv not found — run "
                 "experiments/run_champion_stack.py first")
    champ = pd.read_csv(CHAMPION_CSV, parse_dates=[0], index_col=0)
    fam_b = {c: champ[c].dropna() - DAILY_RF for c in champ.columns
             if c not in CHAMPION_BENCHMARKS}
    rep_b, var_b, best_b = report("Family B: champion-stack ladder on QQQ "
                                  f"({champ.index[0].date()} →)", fam_b)

    # The ladder variants are near-duplicates (trial-Sharpe variance ~10×
    # smaller than family A's), which makes the expected-max benchmark SR₀
    # too easy. Cross-check: re-test B's winner under family A's dispersion.
    print(f"\nCross-check — B's winner under family A's trial dispersion "
          f"({var_a:.2e} vs its own {var_b:.2e}):")
    print("  " + "   ".join(
        f"n={n}: {deflated_sharpe_ratio(best_b.values, n, var_a):.3f}"
        for n in (len(fam_b), *TRIAL_GRID)))

    out = RESULTS_DIR / "retroactive_dsr.csv"
    pd.concat([rep_a, rep_b]).to_csv(out, index=False)
    logger.info(f"\nSaved {out}")
    print("\n⚠ DSR > 0.95 is the miner's pass bar. Nominal n_trials "
          "undercounts the true search (sub-variants, governor/layer "
          "studies); read the sensitivity row, not the headline.")


if __name__ == "__main__":
    main()
