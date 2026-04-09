"""
Backtesting engine for PosteriorAlpha.

Supported strategies
--------------------
bayesian        — Bayesian adaptive Sharpe (our main strategy)
equal_weight    — 1/N rebalanced monthly
min_variance    — long-only global minimum variance
hist_max_sharpe — max Sharpe on full historical prior only (no adaptation)
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.bayesian import (
    bayesian_posterior,
    compute_lambda,
    estimate_moments,
    mahalanobis_distance,
    max_sharpe_weights,
    min_variance_weights,
)

logger = logging.getLogger(__name__)

STRATEGIES = ("bayesian", "equal_weight", "min_variance", "hist_max_sharpe")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """All outputs of a single backtest run."""
    strategy: str
    returns:  pd.Series          # daily arithmetic portfolio returns
    weights:  pd.DataFrame       # weight vector at each rebalance date
    lambdas:  Optional[pd.Series] = None   # λ values (Bayesian only)

    @property
    def cumulative(self) -> pd.Series:
        """Cumulative wealth index starting at 1."""
        return (1.0 + self.returns).cumprod()

    @property
    def n_days(self) -> int:
        return len(self.returns)


# ---------------------------------------------------------------------------
# Single-strategy backtest
# ---------------------------------------------------------------------------

def run_backtest(
    returns: pd.DataFrame,
    strategy: str = "bayesian",
    rebalance_freq: str = "ME",
    min_history: int = 252,
    recent_window: int = 60,
    rf: float = 0.04,
    sensitivity: float = 1.5,
    max_weight: float = 0.20,
) -> BacktestResult:
    """
    Run a monthly-rebalanced backtest.

    Parameters
    ----------
    returns        : daily arithmetic returns, (T, N) DataFrame
    strategy       : one of STRATEGIES
    rebalance_freq : pandas offset alias for rebalance grid (default 'ME' = month-end)
    min_history    : minimum look-back rows needed before first rebalance
    recent_window  : rows used for the 'recent' likelihood window
    rf             : annual risk-free rate used in Sharpe maximisation
    sensitivity    : slope of the sigmoid that maps divergence → λ
    max_weight     : per-asset weight cap
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from {STRATEGIES}.")

    tickers  = returns.columns.tolist()
    n_assets = len(tickers)

    # Build rebalance calendar — use the last trading day of each month
    try:
        rebal_grid = returns.resample(rebalance_freq).last().index
    except ValueError:
        rebal_grid = returns.resample("M").last().index  # older pandas fallback

    # Only rebalance when enough history is available
    rebal_dates = rebal_grid[rebal_grid > returns.index[min_history]]

    port_rets:   List[float]   = []
    ret_dates:   List         = []
    weight_rows: List          = []
    weight_dates: List         = []
    lambda_vals: List[float]  = []
    lambda_dates: List         = []

    for i, rebal_t in enumerate(tqdm(rebal_dates, desc=f"  {strategy:<20}", leave=False)):
        hist = returns.loc[:rebal_t]
        if len(hist) < min_history + recent_window:
            continue

        prior_arr  = hist.values
        recent_arr = hist.values[-recent_window:]

        # ── Weight computation ────────────────────────────────────────────
        lam: Optional[float] = None

        if strategy == "equal_weight":
            weights = np.ones(n_assets) / n_assets

        elif strategy == "min_variance":
            _, cov = estimate_moments(prior_arr)
            weights = min_variance_weights(cov, max_weight)

        elif strategy == "hist_max_sharpe":
            mu, cov = estimate_moments(prior_arr)
            weights = max_sharpe_weights(mu, cov, rf=rf, max_weight=max_weight)

        elif strategy == "bayesian":
            mu_prior,  cov_prior  = estimate_moments(prior_arr)
            mu_recent, cov_recent = estimate_moments(recent_arr)

            divergence = mahalanobis_distance(mu_recent, mu_prior, cov_prior)
            lam        = compute_lambda(divergence, n_assets, sensitivity)

            mu_post, cov_post = bayesian_posterior(
                mu_prior, cov_prior, mu_recent, cov_recent, lam
            )
            weights = max_sharpe_weights(mu_post, cov_post, rf=rf, max_weight=max_weight)

        # ── Apply weights over the holding period ─────────────────────────
        next_t  = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else returns.index[-1]
        period  = returns.loc[rebal_t:next_t].iloc[1:]   # exclude rebalance day itself

        if len(period) == 0:
            continue

        pf_rets = period.values @ weights   # (T_period,) portfolio returns

        port_rets.extend(pf_rets.tolist())
        ret_dates.extend(period.index.tolist())

        weight_rows.append(weights)
        weight_dates.append(rebal_t)

        if lam is not None:
            lambda_vals.append(lam)
            lambda_dates.append(rebal_t)

    returns_series = pd.Series(port_rets, index=ret_dates, name=strategy)
    weights_df     = pd.DataFrame(weight_rows, index=weight_dates, columns=tickers)
    lambdas_series = (
        pd.Series(lambda_vals, index=lambda_dates, name="lambda")
        if lambda_vals else None
    )

    return BacktestResult(
        strategy=strategy,
        returns=returns_series,
        weights=weights_df,
        lambdas=lambdas_series,
    )


# ---------------------------------------------------------------------------
# Convenience: run all strategies at once
# ---------------------------------------------------------------------------

def run_all_strategies(
    returns: pd.DataFrame,
    **kwargs,
) -> Dict[str, BacktestResult]:
    """Run every strategy on the same returns DataFrame. Extra kwargs forwarded to run_backtest."""
    results = {}
    for strat in STRATEGIES:
        try:
            results[strat] = run_backtest(returns, strategy=strat, **kwargs)
        except Exception as exc:
            logger.error(f"Strategy '{strat}' failed: {exc}")
    return results
