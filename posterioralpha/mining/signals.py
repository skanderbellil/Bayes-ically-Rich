"""
Alpha signal families — the search space for the miner.

Each family is a function (ctx, **params) -> DataFrame of raw scores (T × N).
Higher score = more attractive. Scores may contain NaN where there is not
enough history; the evaluator only allocates among valid assets.

All signals are strictly causal: the score at date t uses data up to and
including t, and the evaluator additionally lags positions by one day.
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from posterioralpha.research import precompute_bocpd


# ─────────────────────────────────────────────────────────────────────────────
# Shared, cached market context
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SignalContext:
    """Precomputed inputs shared by every candidate (computed once)."""
    prices:  pd.DataFrame          # adjusted close, T × N
    returns: pd.DataFrame          # daily arithmetic returns, T × N
    market:  pd.Series = field(init=False)   # equal-weight market return
    _erl: Optional[pd.Series] = field(default=None, init=False)

    def __post_init__(self):
        self.market = self.returns.mean(axis=1)

    @property
    def erl(self) -> pd.Series:
        """BOCPD expected run length of the market (cached — it's slow)."""
        if self._erl is None:
            m = self.market.dropna()
            _, erl = precompute_bocpd(m.values)
            self._erl = (pd.Series(erl, index=m.index)
                         .reindex(self.market.index)
                         .ffill().fillna(0.0))
        return self._erl


# ─────────────────────────────────────────────────────────────────────────────
# Signal families
# ─────────────────────────────────────────────────────────────────────────────

def momentum(ctx: SignalContext, lookback: int, skip: int) -> pd.DataFrame:
    """Classic cross-sectional momentum: return over [t-lookback, t-skip]."""
    p = ctx.prices
    return p.shift(skip) / p.shift(lookback) - 1.0


def mean_reversion(ctx: SignalContext, lookback: int) -> pd.DataFrame:
    """Short-term reversal: fade the trailing `lookback`-day return."""
    return -ctx.returns.rolling(lookback).sum()


def vol_scaled_momentum(
    ctx: SignalContext, lookback: int, skip: int, vol_window: int
) -> pd.DataFrame:
    """Momentum normalised by trailing volatility (a crude t-stat)."""
    mom = momentum(ctx, lookback, skip)
    vol = ctx.returns.rolling(vol_window).std() * np.sqrt(lookback)
    return mom / (vol + 1e-9)


def low_volatility(ctx: SignalContext, lookback: int) -> pd.DataFrame:
    """Low-vol anomaly: prefer the quietest names."""
    return -ctx.returns.rolling(lookback).std()


def channel_position(ctx: SignalContext, lookback: int) -> pd.DataFrame:
    """Position within the rolling high-low channel (1 = at the high)."""
    p = ctx.prices
    hi = p.rolling(lookback).max()
    lo = p.rolling(lookback).min()
    return (p - lo) / (hi - lo + 1e-9)


def ma_crossover(ctx: SignalContext, fast: int, slow: int) -> pd.DataFrame:
    """Trend: fast moving average relative to slow moving average."""
    fast = min(fast, slow - 1)
    p = ctx.prices
    return p.rolling(fast).mean() / p.rolling(slow).mean() - 1.0


def idio_reversal(ctx: SignalContext, lookback: int) -> pd.DataFrame:
    """Fade idiosyncratic moves: reversal on market-excess returns."""
    excess = ctx.returns.sub(ctx.market, axis=0)
    return -excess.rolling(lookback).sum()


def skewness(ctx: SignalContext, lookback: int) -> pd.DataFrame:
    """Skewness preference: lottery-like (high-skew) names underperform."""
    return -ctx.returns.rolling(lookback).skew()


def regime_gated_momentum(
    ctx: SignalContext, lookback: int, skip: int, erl_thresh: float
) -> pd.DataFrame:
    """
    Momentum gated by the BOCPD regime age of the market.

    When the regime is old/stable (expected run length above the threshold)
    trend-following gets full weight; right after a changepoint (young
    regime) the gate flips the signal toward reversion. Gate ∈ [-1, 1].
    """
    mom  = momentum(ctx, lookback, skip)
    gate = np.tanh((ctx.erl - erl_thresh) / (erl_thresh / 2.0 + 1e-9))
    return mom.mul(gate, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Parameter search space
# ─────────────────────────────────────────────────────────────────────────────
# Each parameter: (low, high, kind) with kind ∈ {"int", "float"}.
# Sampling is log-uniform for window lengths so short and long horizons are
# explored evenly.

ParamSpec = Dict[str, Tuple[float, float, str]]

FAMILIES: Dict[str, Tuple[Callable, ParamSpec]] = {
    "momentum": (momentum, {
        "lookback": (20, 252, "int"),
        "skip":     (0, 21, "int"),
    }),
    "mean_reversion": (mean_reversion, {
        "lookback": (2, 15, "int"),
    }),
    "vol_scaled_momentum": (vol_scaled_momentum, {
        "lookback":   (20, 252, "int"),
        "skip":       (0, 21, "int"),
        "vol_window": (10, 126, "int"),
    }),
    "low_volatility": (low_volatility, {
        "lookback": (20, 252, "int"),
    }),
    "channel_position": (channel_position, {
        "lookback": (20, 252, "int"),
    }),
    "ma_crossover": (ma_crossover, {
        "fast": (5, 60, "int"),
        "slow": (30, 252, "int"),
    }),
    "idio_reversal": (idio_reversal, {
        "lookback": (2, 15, "int"),
    }),
    "skewness": (skewness, {
        "lookback": (40, 252, "int"),
    }),
    "regime_gated_momentum": (regime_gated_momentum, {
        "lookback":   (20, 252, "int"),
        "skip":       (0, 21, "int"),
        "erl_thresh": (20, 150, "float"),
    }),
}


def sample_param(spec: Tuple[float, float, str], rng: np.random.Generator):
    lo, hi, kind = spec
    if kind == "int":
        # log-uniform over the integer range (favours neither end)
        v = int(round(np.exp(rng.uniform(np.log(lo + 1), np.log(hi + 1))) - 1))
        return int(np.clip(v, lo, hi))
    return float(rng.uniform(lo, hi))


def mutate_param(value, spec: Tuple[float, float, str], rng: np.random.Generator):
    """Jitter a parameter by a random multiplicative factor, clipped to range."""
    lo, hi, kind = spec
    factor = np.exp(rng.normal(0.0, 0.30))
    v = (value + 1) * factor - 1 if kind == "int" else value * factor
    if kind == "int":
        return int(np.clip(round(v), lo, hi))
    return float(np.clip(v, lo, hi))


def compute_scores(ctx: SignalContext, family: str, params: Dict) -> pd.DataFrame:
    func, _ = FAMILIES[family]
    return func(ctx, **params)
