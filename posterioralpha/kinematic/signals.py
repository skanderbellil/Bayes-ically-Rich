"""
Tradeable signals derived from the kinematic filters (section 2–4).

Everything here is CAUSAL by construction: a signal value dated t is built only
from filter outputs that condition on data ≤ t (filtered state, one-step
innovation), and from rolling statistics that look strictly backward. A signal
dated t is assumed known at the close of t and is matched against the return
realised over t → t+1 by the backtester. No smoother output ever enters here.

Signals
-------
  velocity_signal      filtered velocity → momentum tilt           (trend)
  acceleration_signal  filtered acceleration → momentum-of-momentum (trend)
  innovation_signal    −standardised local-level innovation        (reversion)
  pair_spread          alpha_t + beta_t·B vs A residual (Kalman hedge)
  spread_zscore        causal z-score of a spread → reversion signal
"""
from typing import Optional

import numpy as np
import pandas as pd

from posterioralpha.kinematic.filter import (
    FilterResult, LocalLevelKalman, PairsKalman,
)


# ─────────────────────────────────────────────────────────────────────────────
# Causal normalisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def causal_zscore(x: pd.Series, window: int = 252, min_periods: int = 60) -> pd.Series:
    """
    Rolling z-score using only the trailing `window` observations up to and
    including t (so the value at t is known at the close of t — causal). Used to
    turn a raw filter state into a unit-scale, sign-meaningful tilt.
    """
    mu = x.rolling(window, min_periods=min_periods).mean()
    sd = x.rolling(window, min_periods=min_periods).std()
    return (x - mu) / sd.replace(0.0, np.nan)


def _as_series(arr: np.ndarray, index: pd.Index) -> pd.Series:
    return pd.Series(np.asarray(arr, float), index=index)


# ─────────────────────────────────────────────────────────────────────────────
# Trend signals (sections 2 & 6 comparison): filtered velocity / acceleration
# ─────────────────────────────────────────────────────────────────────────────

def velocity_signal(
    res: FilterResult, index: pd.Index,
    zscore: bool = True, window: int = 252,
) -> pd.Series:
    """
    Momentum tilt from the filtered velocity (the latent drift of log price).
    Positive velocity → long. This is the kinematic-filter analogue of MACD;
    section 6 benchmarks the two head to head.
    """
    v = _as_series(res.velocity, index)
    return causal_zscore(v, window) if zscore else v


def acceleration_signal(
    res: FilterResult, index: pd.Index,
    zscore: bool = True, window: int = 252,
) -> pd.Series:
    """
    Momentum-of-momentum tilt from the filtered acceleration (curvature of log
    price). The honest expectation (see diagnostics) is that this is the
    noisiest state; we trade it anyway to measure whether the filter rescued it.
    """
    a = _as_series(res.acceleration, index)
    return causal_zscore(a, window) if zscore else a


# ─────────────────────────────────────────────────────────────────────────────
# Local-level innovation reversion signal (section 3)
# ─────────────────────────────────────────────────────────────────────────────

def innovation_signal(res: FilterResult, index: pd.Index) -> pd.Series:
    """
    Mean-reversion signal: the NEGATIVE standardised one-step innovation of the
    local-level filter. innovation_t = print_t − E[level_t | data < t]; a print
    above the predicted fair level (positive innovation) is faded (short),
    hence the minus sign. Already unit-scaled by sqrt(S_t), so no extra z-score.
    """
    return -_as_series(res.std_innovation, index)


def make_innovation_signal(
    log_price: pd.Series, kf: LocalLevelKalman,
) -> pd.Series:
    """Convenience: run a (pre-fit) local-level filter on a log-price series and
    return its reversion signal in one call."""
    res = kf.filter(log_price.values)
    return innovation_signal(res, log_price.index)


# ─────────────────────────────────────────────────────────────────────────────
# Pair signals (section 4): dynamic Kalman hedge ratio
# ─────────────────────────────────────────────────────────────────────────────

def pair_spread(
    log_a: pd.Series, log_b: pd.Series, pk: PairsKalman,
) -> pd.DataFrame:
    """
    Causal Kalman hedge for A on B. Returns a frame with the filtered alpha_t,
    beta_t and two spread definitions, all dated t and known at close t:

      resid_spread  = A_t − (alpha_t + beta_t·B_t)     uses a_{t|t}
      innov_spread  = A_t − (alpha_{t|t-1} + beta_{t|t-1}·B_t)  one-step innov

    `innov_spread` is the strictly-predictive residual (the filter's surprise);
    `resid_spread` is the contemporaneous fit. The backtester uses innov_spread
    for trading and resid_spread for diagnostics/plots.
    """
    res = pk.filter(log_a.values, log_b.values)
    idx = log_a.index
    alpha = res.state_filt[:, 0]
    beta = res.state_filt[:, 1]
    resid = log_a.values - (alpha + beta * log_b.values)
    return pd.DataFrame(
        {"alpha": alpha, "beta": beta,
         "resid_spread": resid, "innov_spread": res.innovation,
         "innov_std": res.std_innovation},
        index=idx,
    )


def static_ols_spread(
    log_a: pd.Series, log_b: pd.Series, beta: float, alpha: float,
) -> pd.Series:
    """Spread under a FIXED (train-estimated) OLS hedge — the static benchmark
    that the adaptive beta is meant to beat once the relationship drifts."""
    return log_a - (alpha + beta * log_b)


def spread_zscore(
    spread: pd.Series, window: int = 60, min_periods: int = 20,
) -> pd.Series:
    """
    Reversion signal from a spread: NEGATIVE causal z-score (fade the stretch).
    A short window (default 60d) makes the trade a genuine mean-reversion bet on
    the current relationship rather than a slow drift play.
    """
    return -causal_zscore(spread, window, min_periods)
