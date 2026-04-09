"""
Core Bayesian portfolio machinery.

Pipeline per rebalance date:
  1. estimate_moments()  → (μ_prior, Σ_prior) from full history
  2. estimate_moments()  → (μ_recent, Σ_recent) from rolling window
  3. mahalanobis_distance() → how far recent is from prior
  4. compute_lambda()    → map divergence to blending weight λ
  5. bayesian_posterior()→ (μ_post, Σ_post) = λ·recent + (1-λ)·prior
  6. max_sharpe_weights() / min_variance_weights() → portfolio weights
"""
import logging
from typing import Tuple

import numpy as np
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)

_EPS = 1e-8          # ridge for numerical stability
_ANN = 252           # trading days per year


# ---------------------------------------------------------------------------
# Moment estimation
# ---------------------------------------------------------------------------

def estimate_moments(returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate annualised mean and covariance via Ledoit-Wolf shrinkage.

    Parameters
    ----------
    returns : (T, N) daily arithmetic returns

    Returns
    -------
    mu  : (N,) annualised expected returns
    cov : (N, N) annualised covariance, positive-definite by construction
    """
    mu = returns.mean(axis=0) * _ANN

    lw = LedoitWolf(assume_centered=False)
    lw.fit(returns)
    cov = lw.covariance_ * _ANN

    # Tikhonov regularisation — keeps the matrix well-conditioned
    cov += _EPS * np.eye(len(mu))
    return mu, cov


# ---------------------------------------------------------------------------
# Divergence → λ
# ---------------------------------------------------------------------------

def mahalanobis_distance(
    mu_a: np.ndarray, mu_b: np.ndarray, cov: np.ndarray
) -> float:
    """
    Mahalanobis distance between two mean vectors under covariance `cov`.

    Measures how many 'regime standard-deviations' the recent window is
    from the long-run historical expectation.
    """
    diff = mu_a - mu_b
    try:
        L = np.linalg.cholesky(cov)           # cov = L L^T
        z = np.linalg.solve(L, diff)           # z = L^{-1} diff
        dist = float(np.sqrt(np.dot(z, z)))
    except np.linalg.LinAlgError:
        # Fallback: Euclidean (cov is not PD despite regularisation)
        dist = float(np.linalg.norm(diff))
    return max(dist, 0.0)


def compute_lambda(divergence: float, n_assets: int, sensitivity: float = 1.5) -> float:
    """
    Map Mahalanobis divergence to a blending weight λ ∈ (0, 1).

    Under H₀ (no regime shift) the Mahalanobis distance is chi-distributed
    with n_assets degrees of freedom, so its expected value is √n_assets.
    We normalise by √n_assets so λ ≈ 0.5 when divergence is "typical".

    λ → 1  :  large divergence → rely heavily on recent data
    λ → 0  :  small divergence → trust the long-run prior
    """
    normalised = divergence / (np.sqrt(n_assets) + _EPS)
    lam = 1.0 / (1.0 + np.exp(-sensitivity * (normalised - 1.0)))
    return float(np.clip(lam, 0.02, 0.98))


# ---------------------------------------------------------------------------
# Bayesian posterior
# ---------------------------------------------------------------------------

def bayesian_posterior(
    mu_prior: np.ndarray,  cov_prior: np.ndarray,
    mu_recent: np.ndarray, cov_recent: np.ndarray,
    lam: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bayesian shrinkage update:
        μ_post = λ · μ_recent + (1 − λ) · μ_prior
        Σ_post = λ · Σ_recent + (1 − λ) · Σ_prior

    λ is the posterior weight on recent evidence versus historical prior.
    """
    mu_post  = lam * mu_recent  + (1.0 - lam) * mu_prior
    cov_post = lam * cov_recent + (1.0 - lam) * cov_prior
    cov_post += _EPS * np.eye(len(mu_post))
    return mu_post, cov_post


# ---------------------------------------------------------------------------
# Portfolio optimisers
# ---------------------------------------------------------------------------

def max_sharpe_weights(
    mu: np.ndarray,
    cov: np.ndarray,
    rf: float = 0.04,
    max_weight: float = 0.20,
) -> np.ndarray:
    """
    Long-only maximum Sharpe ratio portfolio (SLSQP).

    Maximises  (wᵀμ − rf) / √(wᵀΣw)
    subject to  Σwᵢ = 1,  0 ≤ wᵢ ≤ max_weight.

    Falls back to equal-weight if the optimiser fails.
    """
    n = len(mu)

    def neg_sharpe(w: np.ndarray) -> float:
        excess = float(w @ mu) - rf
        vol = float(np.sqrt(w @ cov @ w + _EPS))
        return -excess / vol

    def neg_sharpe_grad(w: np.ndarray) -> np.ndarray:
        excess = float(w @ mu) - rf
        vol_sq = float(w @ cov @ w + _EPS)
        vol    = np.sqrt(vol_sq)
        # d(-S)/dw = −μ/vol + excess·(Σw) / vol³
        return -mu / vol + excess * (cov @ w) / (vol_sq * vol)

    w0          = np.ones(n) / n
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

    logger.debug("max_sharpe optimisation failed — using equal weights")
    return np.ones(n) / n


def min_variance_weights(
    cov: np.ndarray,
    max_weight: float = 0.20,
) -> np.ndarray:
    """Long-only global minimum variance portfolio (SLSQP)."""
    n = cov.shape[0]

    def port_var(w):      return float(w @ cov @ w)
    def port_var_grad(w): return 2.0 * cov @ w

    w0          = np.ones(n) / n
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds      = [(0.0, max_weight)] * n

    res = minimize(
        port_var, w0, jac=port_var_grad,
        method="SLSQP", bounds=bounds, constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 2000},
    )

    if res.success:
        w = np.clip(res.x, 0.0, max_weight)
        return w / w.sum()

    return np.ones(n) / n
