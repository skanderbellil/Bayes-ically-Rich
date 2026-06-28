"""
Fundamental-Law diagnostics (stage 2: research).

Primitives for decomposing a strategy's Sharpe through the Fundamental Law of
Active Management, ``SR = IC · √(breadth)`` per period (see
``docs/knowledge/sharpe-decomposition-levers.md``). Two quantities are
routinely *over*-stated and silently inflate a backtest's apparent Sharpe:

  - **Effective breadth** — the Law's ``breadth`` is the number of
    *independent* bets, not the nominal name count. Correlated names share
    information, so the true breadth is a fraction of the count. Buckle (2004)
    / Clarke–de Silva–Thorley (2002): for ``N`` names with average pairwise
    correlation ``ρ̄``, the effective breadth is ``N / (1 + (N − 1)·ρ̄)``.

  - **Transfer coefficient (TC)** — you can rarely act on a signal fully
    (long-only, position limits, costs). TC is the risk-adjusted correlation
    between the weights you *implemented* and the unconstrained-optimal
    weights, and it scales the achievable Sharpe directly:
    ``IR = TC · IC · √(breadth)``. Long-only books commonly run TC ≈ 0.3–0.5.

All functions are causal/offline and operate on plain returns frames.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def average_pairwise_correlation(returns: pd.DataFrame, min_overlap: int = 60) -> float:
    """
    Mean off-diagonal Pearson correlation across the columns of ``returns``.

    Pairs with fewer than ``min_overlap`` jointly-observed days are ignored so
    sparsely-overlapping names don't dominate the average. Returns ``nan`` if
    no pair clears the overlap threshold.
    """
    if returns.shape[1] < 2:
        return float("nan")
    corr = returns.corr(min_periods=min_overlap).to_numpy()
    iu = np.triu_indices_from(corr, k=1)
    off = corr[iu]
    off = off[np.isfinite(off)]
    return float(off.mean()) if off.size else float("nan")


def effective_breadth(
    returns: Optional[pd.DataFrame] = None,
    *,
    n: Optional[int] = None,
    avg_corr: Optional[float] = None,
    min_overlap: int = 60,
) -> Tuple[int, float, float]:
    """
    Effective number of independent bets via the equicorrelation haircut.

    ``BR_eff = N / (1 + (N − 1)·ρ̄)`` where ``ρ̄`` is the average pairwise
    correlation. With ``ρ̄ = 0`` this returns ``N`` (fully independent); as
    ``ρ̄ → 1`` it collapses toward 1 (one shared bet).

    Provide either a ``returns`` frame (``N`` and ``ρ̄`` are measured from it)
    or both ``n`` and ``avg_corr`` directly.

    Returns ``(n_nominal, breadth_effective, avg_corr)``.
    """
    if returns is not None:
        n = int(returns.shape[1])
        avg_corr = average_pairwise_correlation(returns, min_overlap=min_overlap)
    if n is None or avg_corr is None:
        raise ValueError("pass either `returns`, or both `n` and `avg_corr`")
    if n <= 1 or not np.isfinite(avg_corr):
        return int(n or 0), float(n or 0), float(avg_corr if avg_corr is not None else np.nan)
    rho = max(avg_corr, -1.0 / (n - 1) + 1e-9)  # keep denominator positive
    br = n / (1.0 + (n - 1) * rho)
    return int(n), float(br), float(avg_corr)


def participation_ratio(returns: pd.DataFrame, min_overlap: int = 60) -> float:
    """
    Effective independent dimensions from the correlation-matrix spectrum:
    ``PR = (Σλ)² / Σλ²`` over the eigenvalues ``λ`` of the correlation matrix.

    A spectral alternative to :func:`effective_breadth` — equals ``N`` when all
    names are uncorrelated and 1 when they move as one. Returns ``nan`` if the
    matrix can't be formed.
    """
    if returns.shape[1] < 2:
        return float("nan")
    corr = returns.corr(min_periods=min_overlap)
    corr = corr.dropna(how="all").dropna(how="all", axis=1)
    if corr.shape[0] < 2:
        return float("nan")
    c = np.array(corr.to_numpy(), dtype=float)
    c[~np.isfinite(c)] = 0.0
    np.fill_diagonal(c, 1.0)
    lam = np.linalg.eigvalsh(c)
    lam = lam[lam > 0]
    if lam.size == 0:
        return float("nan")
    return float(lam.sum() ** 2 / np.square(lam).sum())


def transfer_coefficient(
    w_actual: np.ndarray | pd.Series,
    w_ideal: np.ndarray | pd.Series,
    cov: Optional[np.ndarray] = None,
) -> float:
    """
    Risk-adjusted correlation between implemented and unconstrained-optimal
    active weights (Clarke–de Silva–Thorley 2002):

        TC = (wₐ' Σ wᵢ) / √(wₐ' Σ wₐ · wᵢ' Σ wᵢ)

    With ``cov=None`` the identity is used, reducing TC to the plain
    correlation of the two weight vectors. TC ∈ [−1, 1]; TC = 1 means the
    implemented book is a positive rescaling of the ideal book (no transfer
    loss), TC = 0 means the implementation is uninformative about the ideal.

    The two weight vectors must be aligned (same asset order/index).
    """
    wa = np.asarray(w_actual, dtype=float).ravel()
    wi = np.asarray(w_ideal, dtype=float).ravel()
    if wa.shape != wi.shape:
        raise ValueError("w_actual and w_ideal must have the same length")
    sigma = np.eye(wa.size) if cov is None else np.asarray(cov, dtype=float)
    num = wa @ sigma @ wi
    den = np.sqrt((wa @ sigma @ wa) * (wi @ sigma @ wi))
    return float(num / den) if den > 0 else float("nan")


def empirical_transfer_coefficient(sr_constrained: float, sr_ideal: float) -> float:
    """
    Realized-Sharpe proxy for the transfer coefficient: ``SR_constrained /
    SR_ideal``. Quick and model-free — run the *same* signal in its constrained
    form (e.g. long-only active) and its unconstrained form (dollar-neutral
    rank-weighted), and the ratio of achieved Sharpes estimates how much edge
    the constraint transfers. Returns ``nan`` if the ideal Sharpe is ~0.
    """
    return float(sr_constrained / sr_ideal) if abs(sr_ideal) > 1e-9 else float("nan")


def fundamental_law_sharpe(
    ic: float, breadth: float, periods_per_year: float = 1.0, tc: float = 1.0
) -> float:
    """
    Annualized Sharpe implied by the Fundamental Law:
    ``SR = TC · IC · √(breadth · periods_per_year)``.

    ``breadth`` is the per-period effective breadth (use
    :func:`effective_breadth`), ``periods_per_year`` the number of independent
    decision periods, and ``tc`` the transfer coefficient (default 1 = full
    implementation).
    """
    return float(tc * ic * np.sqrt(max(breadth, 0.0) * max(periods_per_year, 0.0)))


def cross_sectional_ic(
    forecast: pd.Series, realized: pd.Series, method: str = "spearman"
) -> float:
    """
    One-period cross-sectional information coefficient — the correlation
    between a ``forecast`` cross-section and the ``realized`` returns over the
    matching names. ``method`` is ``"spearman"`` (rank IC, default) or
    ``"pearson"``. Returns ``nan`` if fewer than 3 names overlap.
    """
    df = pd.concat([forecast.rename("f"), realized.rename("r")], axis=1).dropna()
    if len(df) < 3:
        return float("nan")
    return float(df["f"].corr(df["r"], method=method))
