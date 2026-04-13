"""
Core Bayesian portfolio machinery.

Pipeline per rebalance date:
  1. estimate_moments()      → (μ_prior, Σ_prior) from full history via Ledoit-Wolf
  2. estimate_moments_ewma() → (μ_ewma,  Σ_ewma)  via exponential weighting (smooth recent)
  3. mahalanobis_distance()  → how far recent is from the long-run prior
  4. compute_lambda()        → map divergence to blending weight λ via sigmoid
  5. bayesian_posterior()    → (μ_post, Σ_post) = λ·recent + (1−λ)·prior
  6. max_sharpe_weights()    → portfolio weights (with optional TC penalty)
     min_variance_weights()  → fallback risk-off portfolio
"""
import logging
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)

_EPS = 1e-8   # ridge for numerical stability
_ANN = 252    # trading days per year


# ---------------------------------------------------------------------------
# Moment estimation
# ---------------------------------------------------------------------------

def estimate_moments(returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Annualised mean + covariance via Ledoit-Wolf shrinkage.

    Parameters
    ----------
    returns : (T, N) daily arithmetic returns

    Returns
    -------
    mu  : (N,) annualised expected returns
    cov : (N, N) annualised positive-definite covariance
    """
    mu = returns.mean(axis=0) * _ANN
    lw = LedoitWolf(assume_centered=False)
    lw.fit(returns)
    cov = lw.covariance_ * _ANN + _EPS * np.eye(returns.shape[1])
    return mu, cov


def estimate_moments_ewma(
    returns: np.ndarray,
    halflife: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exponentially weighted mean + covariance.

    Extension from the original idea: "rather than a hard window for 'recent',
    use EWMA so data decays smoothly."

    The most recent observation carries weight 1; weights halve every `halflife`
    days going back.  This replaces the hard 60-day window with a smooth signal.
    """
    T, N = returns.shape
    tau = halflife / np.log(2)
    t   = np.arange(T, dtype=float)
    w   = np.exp(-(T - 1 - t) / tau)   # most recent = highest
    w  /= w.sum()

    mu_w  = (returns * w[:, None]).sum(axis=0) * _ANN

    # Weighted covariance (outer-product form to preserve PD structure)
    r_bar = (returns * w[:, None]).sum(axis=0)
    r_c   = returns - r_bar
    cov_w = (r_c * w[:, None]).T @ r_c * _ANN
    cov_w += _EPS * np.eye(N)
    return mu_w, cov_w


# ---------------------------------------------------------------------------
# Divergence → λ
# ---------------------------------------------------------------------------

def mahalanobis_distance(
    mu_a: np.ndarray, mu_b: np.ndarray, cov: np.ndarray
) -> float:
    """
    Mahalanobis distance between two mean vectors under covariance `cov`.

    Measures how many 'regime standard-deviations' the recent window sits
    from the long-run historical prior.
    """
    diff = mu_a - mu_b
    try:
        L    = np.linalg.cholesky(cov)
        z    = np.linalg.solve(L, diff)
        dist = float(np.sqrt(np.dot(z, z)))
    except np.linalg.LinAlgError:
        dist = float(np.linalg.norm(diff))
    return max(dist, 0.0)


def compute_lambda(divergence: float, n_assets: int, sensitivity: float = 1.5) -> float:
    """
    Map Mahalanobis divergence to blending weight λ ∈ (0, 1).

    Under H₀ (stable regime) the distance is chi-distributed with n_assets df,
    so its expected value is √n_assets.  Normalising by √n_assets centres the
    sigmoid so λ ≈ 0.5 when divergence is 'typical'.

    λ → 1  :  large divergence → trust recent data
    λ → 0  :  small divergence → trust the long-run prior
    """
    normalised = divergence / (np.sqrt(n_assets) + _EPS)
    lam = 1.0 / (1.0 + np.exp(-sensitivity * (normalised - 1.0)))
    return float(np.clip(lam, 0.02, 0.98))


# ---------------------------------------------------------------------------
# Bayesian posterior blend
# ---------------------------------------------------------------------------

def bayesian_posterior(
    mu_prior:  np.ndarray, cov_prior:  np.ndarray,
    mu_recent: np.ndarray, cov_recent: np.ndarray,
    lam: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Shrinkage update:
        μ_post = λ · μ_recent + (1 − λ) · μ_prior
        Σ_post = λ · Σ_recent + (1 − λ) · Σ_prior
    """
    mu_post  = lam * mu_recent  + (1.0 - lam) * mu_prior
    cov_post = lam * cov_recent + (1.0 - lam) * cov_prior + _EPS * np.eye(len(mu_prior))
    return mu_post, cov_post


# ---------------------------------------------------------------------------
# Portfolio optimisers
# ---------------------------------------------------------------------------

def max_sharpe_weights(
    mu: np.ndarray,
    cov: np.ndarray,
    rf: float = 0.04,
    max_weight: float = 0.20,
    w_prev: Optional[np.ndarray] = None,
    tc: float = 0.0,
) -> np.ndarray:
    """
    Long-only maximum Sharpe portfolio (SLSQP).

    Extension: optional transaction-cost penalty
        objective += tc · Σ|w_i − w_prev_i|
    This prevents the portfolio from thrashing when λ oscillates,
    implementing the 'turnover penalty' from the idea description.

    Falls back to equal-weight on optimiser failure.
    """
    n  = len(mu)
    w0 = np.ones(n) / n if w_prev is None else w_prev.copy()

    def neg_sharpe(w: np.ndarray) -> float:
        excess = float(w @ mu) - rf
        vol    = float(np.sqrt(w @ cov @ w + _EPS))
        obj    = -excess / vol
        if w_prev is not None and tc > 0.0:
            obj += tc * float(np.sum(np.abs(w - w_prev)))
        return obj

    def neg_sharpe_grad(w: np.ndarray) -> np.ndarray:
        excess  = float(w @ mu) - rf
        vol_sq  = float(w @ cov @ w + _EPS)
        vol     = np.sqrt(vol_sq)
        grad    = -mu / vol + excess * (cov @ w) / (vol_sq * vol)
        if w_prev is not None and tc > 0.0:
            grad += tc * np.sign(w - w_prev)
        return grad

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds      = [(0.0, max_weight)] * n

    res = minimize(
        neg_sharpe, w0, jac=neg_sharpe_grad,
        method="SLSQP", bounds=bounds, constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 2000},
    )

    if res.success and np.isfinite(res.fun):
        w = np.clip(res.x, 0.0, max_weight)
        return w / w.sum()

    logger.debug("max_sharpe failed — equal weights")
    return np.ones(n) / n


def min_variance_weights(
    cov: np.ndarray,
    max_weight: float = 0.20,
) -> np.ndarray:
    """Long-only global minimum variance portfolio (SLSQP)."""
    n  = cov.shape[0]
    w0 = np.ones(n) / n

    res = minimize(
        lambda w: float(w @ cov @ w),
        w0,
        jac=lambda w: 2.0 * cov @ w,
        method="SLSQP",
        bounds=[(0.0, max_weight)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"ftol": 1e-10, "maxiter": 2000},
    )

    if res.success:
        w = np.clip(res.x, 0.0, max_weight)
        return w / w.sum()

    return np.ones(n) / n
