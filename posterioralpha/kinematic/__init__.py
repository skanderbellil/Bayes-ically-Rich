"""
Kinematic state-space signals from price alone (the "generative layer").

Modular stack, mirroring the rest of PosteriorAlpha (data → filter → signals →
regime → backtest → diagnostics):

  filter       KinematicKalman / LocalLevelKalman / PairsKalman + RTS smoother,
               MLE calibration. Strict causal/diagnostic split (read the
               CAUSALITY CONTRACT at the top of filter.py).
  signals      causal trend (velocity/accel) and reversion (innovation) tilts,
               plus the dynamic Kalman hedge spread.
  regime       2-state causal HMM on filtered velocity, gating trend vs revert.
  backtest     cost-aware walk-forward engine + EMA/MACD integral baselines and
               the (flagged) path-signature corner.
  diagnostics  noise-amplification study, IC/decay, PnL concentration, effective
               bets, Q/R sweep.

Data comes from ``posterioralpha.data.load_fresh_prices`` (SPY/QQQ/GLD).
"""
from posterioralpha.kinematic.filter import (
    FilterResult,
    KinematicKalman,
    LocalLevelKalman,
    PairsKalman,
)
from posterioralpha.kinematic.signals import (
    acceleration_signal,
    causal_zscore,
    innovation_signal,
    make_innovation_signal,
    pair_spread,
    spread_zscore,
    static_ols_spread,
    velocity_signal,
)
from posterioralpha.kinematic.regime import VelocityRegimeHMM
from posterioralpha.kinematic.backtest import (
    BacktestResult,
    ema_signal,
    macd_signal,
    run_signal_backtest,
    signature_trend_signal,
    walk_forward,
)
from posterioralpha.kinematic.ladder import (
    RegimeRungSelector,
    efficiency_ratio,
    ladder_rungs,
    leaky_integral,
)
from posterioralpha.kinematic import diagnostics

__all__ = [
    "FilterResult", "KinematicKalman", "LocalLevelKalman", "PairsKalman",
    "velocity_signal", "acceleration_signal", "innovation_signal",
    "make_innovation_signal", "causal_zscore", "pair_spread", "spread_zscore",
    "static_ols_spread",
    "VelocityRegimeHMM",
    "BacktestResult", "run_signal_backtest", "walk_forward",
    "ema_signal", "macd_signal", "signature_trend_signal",
    "leaky_integral", "efficiency_ratio", "ladder_rungs", "RegimeRungSelector",
    "diagnostics",
]
