"""
Turn raw alpha scores into realistic portfolio returns.

Pipeline: scores → cross-sectional weights (dollar-neutral long-short)
→ optional smoothing → 1-day execution lag → returns net of turnover costs.
"""
from typing import Tuple

import numpy as np
import pandas as pd

_ANN = 252


def scores_to_weights(
    scores: pd.DataFrame,
    weighting: str = "rank",
    top_frac: float = 0.2,
    smooth: int = 1,
) -> pd.DataFrame:
    """
    Convert scores into dollar-neutral weights with gross exposure 1.

    weighting
      "rank"     — linear in the demeaned cross-sectional rank (uses all names)
      "quantile" — equal-weight long top `top_frac`, short bottom `top_frac`
    """
    ranks = scores.rank(axis=1)                      # NaN scores stay NaN
    n_valid = ranks.notna().sum(axis=1)

    if weighting == "quantile":
        k = (n_valid * top_frac).clip(lower=1)
        long_mask  = ranks.gt(n_valid.sub(k), axis=0)
        short_mask = ranks.le(k, axis=0)
        w = long_mask.astype(float).div(k, axis=0) \
            - short_mask.astype(float).div(k, axis=0)
        w = w * 0.5                                  # gross = 1
    else:
        centered = ranks.sub(ranks.mean(axis=1), axis=0)
        gross = centered.abs().sum(axis=1)
        w = centered.div(gross.replace(0.0, np.nan), axis=0)

    w = w.fillna(0.0)

    if smooth > 1:
        # EMA of target weights: cuts turnover, keeps weights causal
        w = w.ewm(span=smooth, adjust=False).mean()
        gross = w.abs().sum(axis=1)
        w = w.div(gross.replace(0.0, np.nan), axis=0).fillna(0.0)

    return w


def portfolio_returns(
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    tc: float = 0.001,
) -> Tuple[pd.Series, pd.Series]:
    """
    Daily net returns of the weighted portfolio with a 1-day execution lag.

    Weights decided at t are held over t+1; costs charge `tc` per unit of
    one-way turnover. Returns (net_returns, daily_turnover).
    """
    w_held = weights.shift(1).fillna(0.0)
    rets = asset_returns.fillna(0.0)

    gross = (w_held * rets).sum(axis=1)
    turnover = w_held.diff().abs().sum(axis=1).fillna(0.0)
    net = gross - tc * turnover
    return net, turnover


def sharpe(returns: np.ndarray) -> float:
    """Annualised Sharpe of daily returns (rf = 0 for long-short alphas)."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 20 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(_ANN))
