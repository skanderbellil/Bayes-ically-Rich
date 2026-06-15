"""Cross-market momentum signals on Polymarket probability series.

Prediction-market prices are *probabilities* bounded in (0, 1), so raw price
changes are not comparable across markets: a 0.50→0.60 move and a 0.90→0.95 move
are very different events even though both are +0.05. We therefore work in
**log-odds** space::

    z = logit(p) = log( p / (1 - p) )

Log-odds is the natural, unbounded coordinate for a probability — a fixed step
in z is an equal multiplicative update to the odds regardless of level, which is
exactly the Bayesian belief-update scale. Momentum is the trailing change in z.

Two signal geometries:

  * **time-series** momentum — each market vs. its own past (sign of trailing z
    change). Trades "this market is drifting up/down".
  * **cross-sectional** momentum — each market vs. its *peers right now* (z-score
    of trailing change across the currently-active universe). Trades "this market
    is drifting up/down *more than the others*", and is naturally dollar-neutral.

Both are then **Bayesian-shrunk**: a market's momentum estimate is only as
trustworthy as it is large relative to its own noise, so we scale each raw score
by a signal-to-noise factor σ²/(σ² + s²) (a Normal-Normal posterior-mean shrink).
Quiet, illiquid markets get pulled toward zero; markets with a clean, large move
keep most of their signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-6
_CLIP = (0.02, 0.98)   # keep logit finite near resolution


# ---------------------------------------------------------------------------
# Probability <-> log-odds
# ---------------------------------------------------------------------------

def to_logodds(p: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """logit(p) with prices clipped away from 0/1 so the transform stays finite."""
    q = p.clip(*_CLIP)
    return np.log(q / (1.0 - q))


def from_logodds(z: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Inverse logit — log-odds back to a probability."""
    return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------

def logodds_momentum(panel: pd.DataFrame, lookback: int = 7) -> pd.DataFrame:
    """Trailing change in log-odds over ``lookback`` days: z_t - z_{t-lookback}.

    NaN until a market has at least ``lookback`` observations of history, so the
    signal at date *t* uses only data up to *t* (no lookahead).
    """
    z = to_logodds(panel)
    return z - z.shift(lookback)


def _rolling_logodds_vol(panel: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Trailing volatility of daily log-odds moves — the per-market noise scale."""
    z = to_logodds(panel)
    dz = z.diff()
    # std of daily steps over the lookback, scaled to the lookback horizon
    return dz.rolling(lookback, min_periods=max(3, lookback // 2)).std() * np.sqrt(lookback)


def bayesian_shrink(
    momentum: pd.DataFrame,
    panel: pd.DataFrame,
    lookback: int = 7,
    prior_scale: float = 1.0,
) -> pd.DataFrame:
    """Shrink each momentum estimate toward 0 by its own signal-to-noise ratio.

    Treat the observed trailing move ``m`` as a noisy read of a latent drift with
    Normal-Normal conjugacy: prior drift ~ N(0, τ²), observation noise s² (the
    market's trailing log-odds vol). The posterior mean is::

        m_hat = m · τ² / (τ² + s²)

    so a move that is small relative to the market's own churn is pulled to ~0,
    while a clean, large move survives. ``prior_scale`` sets τ as a multiple of
    the cross-sectional median noise (1.0 = neutral).
    """
    s = _rolling_logodds_vol(panel, lookback)
    s2 = s.pow(2)
    # τ²: a common prior width per date = (prior_scale × median noise)²
    tau = prior_scale * s.median(axis=1).clip(lower=_EPS)
    tau2 = (tau ** 2).values[:, None]
    shrink = tau2 / (tau2 + s2.values + _EPS)
    return momentum * pd.DataFrame(shrink, index=momentum.index, columns=momentum.columns)


def cross_sectional_score(signal: pd.DataFrame, min_names: int = 5) -> pd.DataFrame:
    """Per-date cross-sectional z-score of a signal across active markets.

    Demeans and standardises each row over the markets that are live (non-NaN)
    that day, so the score is "how this market's signal ranks against its peers
    *right now*" — scale-free and dollar-neutral by construction. Rows with fewer
    than ``min_names`` active markets are dropped (NaN).
    """
    valid = signal.notna().sum(axis=1) >= min_names
    mu = signal.mean(axis=1)
    sd = signal.std(axis=1).replace(0.0, np.nan)
    z = signal.sub(mu, axis=0).div(sd, axis=0)
    z[~valid] = np.nan
    return z
