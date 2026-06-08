"""
PEAD — Post-Earnings-Announcement-Drift strategy module.

A self-contained, event-driven cross-sectional equity-anomaly pipeline. It is a
different paradigm from the time-series portfolio strategies in
``posterioralpha.{research,backtest}`` (which optimise a fixed asset basket over
time), so it lives in its own subpackage — but its modules still follow the same
four-stage layering:

    1. data        universe.py   survivorship-aware US common-stock universe
                   fetch.py      checkpoint/resume yfinance earnings+price fetch
                   paths.py      repo-root-anchored runtime directories

    2. research    signals.py    analyst-surprise SUE + forward-return panel
                   momentum.py   20-day momentum signal + SUE/momentum ensemble
                   bayesvol.py   Bayesian online variance forecast (vol targeting)

    3. backtest    walk_forward.py   walk-forward season grouping (no lookahead)
                   backtest.py       quarterly long-short PEAD backtest
                   corrected.py      return calc fixed for the winners-only bug
                   costs.py          transaction-cost / capacity modelling

    4. validation  fama_macbeth.py   cross-sectional Spearman IC + t-tests
                   research_utils.py backtest metrics + equity-curve plots

Entry points live in ``experiments/``:
    download_finance_data.py  → fetch the FinanceDatabase universe (prerequisite)
    run_pead.py               → Fama-MacBeth IC test across cap buckets
    walk_forward_pead.py      → walk-forward long-short backtest

Findings (see docs/pead/) are deliberately self-critical: a prior +23.6%/yr
result was traced to a winners-only calculation bug and corrected to ~+11-15%/yr,
and the live-names-only universe makes long-short returns survivorship-biased
upward. Treat results as a research artefact, not a deployable edge.
"""

from posterioralpha.pead import (  # noqa: F401  (re-export submodules by stage)
    backtest,
    bayesvol,
    corrected,
    costs,
    fama_macbeth,
    fetch,
    momentum,
    paths,
    research_utils,
    signals,
    universe,
    walk_forward,
)

__all__ = [
    "universe", "fetch", "paths",          # stage 1
    "signals", "momentum", "bayesvol",     # stage 2
    "walk_forward", "backtest", "corrected", "costs",   # stage 3
    "fama_macbeth", "research_utils",      # stage 4
]
