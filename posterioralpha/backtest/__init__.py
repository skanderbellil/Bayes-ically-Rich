"""Stage 3 — backtest engines: walk research primitives through history."""
from posterioralpha.backtest.amr import (
    AMR_STRATEGIES,
    AMRResult,
    run_all_amr,
    run_amr_backtest,
    sensitivity_sweep,
)
from posterioralpha.backtest.bayesian import (
    STRATEGIES,
    BacktestResult,
    run_all_strategies,
    run_backtest,
)

__all__ = [
    # Bayesian engine
    "STRATEGIES",
    "BacktestResult",
    "run_backtest",
    "run_all_strategies",
    # AMR engine
    "AMR_STRATEGIES",
    "AMRResult",
    "run_amr_backtest",
    "run_all_amr",
    "sensitivity_sweep",
]
