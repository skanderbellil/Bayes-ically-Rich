"""
Automated alpha mining: generate → evolve → validate (randomized).

This subpackage automates the research loop the rest of the project runs by
hand, wiring together the existing pipeline stages:

  stage 2 (research)   — `signals`: parameterized alpha families, including a
                         BOCPD regime-gated family built on research.regimes
  stage 3 (backtest)   — `evaluation`: lagged, cost-aware, dollar-neutral
                         long-short portfolio construction
  stage 4 (validation) — `validation`: randomized statistical gauntlet
                         (random purged windows, stationary block bootstrap,
                         circular permutation tests, Deflated Sharpe Ratio)
  the loop             — `miner`: evolutionary search + holdout gauntlet
                         + leaderboard
"""
from posterioralpha.mining.miner import AlphaMiner, Candidate, MinerConfig

__all__ = ["AlphaMiner", "Candidate", "MinerConfig"]
