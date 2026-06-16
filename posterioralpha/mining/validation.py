"""
Randomized statistical validation for mined alphas.

Defences against data mining, in the order they are applied:

1. random_windows        — fitness is scored on freshly randomized, purged
                           validation windows every generation, so no candidate
                           can overfit one fixed split.
2. block_bootstrap_pvalue — stationary block bootstrap of the daily returns:
                           P(true Sharpe ≤ 0) given autocorrelation.
3. permutation_pvalue    — circularly time-shift the weight matrix against the
                           real returns: P(observed Sharpe under "no skill",
                           same positions, same market).
4. deflated_sharpe_ratio — Bailey & López de Prado (2014): probability the
                           Sharpe is real after accounting for how many
                           candidates were tried during the search.
"""
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from posterioralpha.mining.evaluation import sharpe

_EULER = 0.5772156649015329


# ─────────────────────────────────────────────────────────────────────────────
# 1. Randomized purged validation windows
# ─────────────────────────────────────────────────────────────────────────────

def random_windows(
    n_obs: int,
    n_windows: int,
    window_len: int,
    rng: np.random.Generator,
    min_start: int = 0,
) -> List[Tuple[int, int]]:
    """
    Draw `n_windows` random contiguous [start, end) windows inside
    [min_start, n_obs). Windows may overlap across candidates/generations —
    the point is that no candidate ever sees a fixed, gameable split.
    """
    hi = n_obs - window_len
    if hi <= min_start:
        return [(min_start, n_obs)]
    starts = rng.integers(min_start, hi, size=n_windows)
    return [(int(s), int(s) + window_len) for s in starts]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Stationary block bootstrap
# ─────────────────────────────────────────────────────────────────────────────

def block_bootstrap_pvalue(
    returns: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = 500,
    avg_block: int = 20,
) -> Tuple[float, float, float]:
    """
    Stationary bootstrap (Politis & Romano): resample blocks of geometric
    length to preserve autocorrelation, recompute the Sharpe each time.

    Returns (p_value of Sharpe ≤ 0, ci_low, ci_high) for the annualised Sharpe.
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    T = len(r)
    if T < 60:
        return 1.0, 0.0, 0.0

    p_new = 1.0 / avg_block
    sharpes = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.empty(T, dtype=int)
        idx[0] = rng.integers(T)
        new_block = rng.random(T) < p_new
        for t in range(1, T):
            idx[t] = rng.integers(T) if new_block[t] else (idx[t - 1] + 1) % T
        sharpes[b] = sharpe(r[idx])

    p_val = float(np.mean(sharpes <= 0.0))
    lo, hi = np.percentile(sharpes, [2.5, 97.5])
    return p_val, float(lo), float(hi)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Circular-shift permutation test
# ─────────────────────────────────────────────────────────────────────────────

def permutation_pvalue(
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    rng: np.random.Generator,
    n_perm: int = 200,
    min_shift: int = 60,
    tc: float = 0.001,
) -> float:
    """
    Null hypothesis: the alpha's timing carries no information. Circularly
    shift the entire weight matrix in time by a random offset (≥ min_shift)
    and recompute net returns against the *real* asset returns. This keeps
    the positions' style, turnover and gross exposure identical — only the
    alignment with the market is destroyed.

    Returns the fraction of null Sharpes ≥ the observed Sharpe.
    """
    W = weights.values
    R = np.nan_to_num(asset_returns.values, nan=0.0)
    T = len(W)
    if T < 2 * min_shift:
        return 1.0

    def net_sharpe(Wm: np.ndarray) -> float:
        held = np.vstack([np.zeros((1, Wm.shape[1])), Wm[:-1]])
        gross = (held * R).sum(axis=1)
        turn = np.abs(np.diff(held, axis=0, prepend=np.zeros((1, Wm.shape[1])))).sum(axis=1)
        return sharpe(gross - tc * turn)

    observed = net_sharpe(W)
    shifts = rng.integers(min_shift, T - min_shift, size=n_perm)
    null = np.array([net_sharpe(np.roll(W, int(s), axis=0)) for s in shifts])
    return float((np.sum(null >= observed) + 1) / (n_perm + 1))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Deflated Sharpe Ratio
# ─────────────────────────────────────────────────────────────────────────────

def deflated_sharpe_ratio(
    returns: np.ndarray,
    n_trials: int,
    trial_sharpes_var: float,
) -> float:
    """
    DSR = P(true Sharpe > 0 | the best of `n_trials` searched candidates).

    The benchmark Sharpe SR₀ is the expected maximum of n_trials random
    Sharpes with variance `trial_sharpes_var` (both in daily units); the
    observed Sharpe is then tested against SR₀ with the Lo (2002) standard
    error adjusted for skew and kurtosis (Bailey & López de Prado 2014).
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    T = len(r)
    if T < 60 or r.std() == 0:
        return 0.0

    sr = float(r.mean() / r.std())                     # daily Sharpe
    skew = float(pd.Series(r).skew())
    kurt = float(pd.Series(r).kurt()) + 3.0            # raw kurtosis

    n = max(int(n_trials), 2)
    v = max(trial_sharpes_var, 1e-12)
    sr0 = np.sqrt(v) * (
        (1.0 - _EULER) * norm.ppf(1.0 - 1.0 / n)
        + _EULER * norm.ppf(1.0 - 1.0 / (n * np.e))
    )

    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
    if denom <= 0:
        return 0.0
    z = (sr - sr0) * np.sqrt(T - 1) / np.sqrt(denom)
    return float(norm.cdf(z))
