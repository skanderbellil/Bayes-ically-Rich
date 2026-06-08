"""Bayesian dynamic volatility forecasting on second moments.

Returns are ~unforecastable (PEAD tradeable IC was 0.03). Variance is NOT: it clusters.
This models the daily-return precision phi = 1/sigma^2 with a Gamma posterior, updated
online with power-discounting (West & Harrison) so old data is forgotten. The discount
ADAPTS: when an observation is a big surprise under the current posterior predictive
(Student-t) -- a BOCPD-style regime signal -- it forgets faster to jump to the new vol
level. Forecasts are strictly causal (use only past data).

No price/return direction is predicted anywhere; only the second moment.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


def bayes_vol_forecast(returns: np.ndarray, delta: float = 0.96,
                       adapt: bool = True, surprise_p: float = 0.02,
                       delta_fast: float = 0.80, a0: float = 1.0, b0: float = 1e-4):
    """One-step-ahead daily variance forecasts (causal).

    Gamma(a,b) on precision; InvGamma on variance with mean b/(a-1).
    Predict step discounts (a,b) by delta (or delta_fast after a surprise), inflating
    uncertainty; update step adds 1/2 and r^2/2. Returns array of forecast daily
    variances v_t made from data through t-1 (v[0] uses prior only).
    """
    n = len(returns)
    vfore = np.empty(n)
    a, b = a0, b0
    for t in range(n):
        # choose discount: faster forgetting right after a surprise
        d = delta
        if adapt and t > 0:
            # posterior predictive for r_{t-1} was Student-t(df=2a) scale sqrt(b/a)
            df = max(2 * a, 2.1)
            scale = np.sqrt(b / a)
            # tail prob of the just-seen return
            p = 2 * stats.t.sf(abs(returns[t - 1]) / scale, df)
            if p < surprise_p:
                d = delta_fast
        # predict (discount)
        a, b = d * a, d * b
        # forecast variance for today's return (mean of InvGamma)
        vfore[t] = b / (a - 1) if a > 1 else b / a
        # update with today's observation (so it informs t+1)
        a += 0.5
        b += 0.5 * returns[t] ** 2
    return vfore


def trailing_vol_forecast(returns: np.ndarray, window: int = 20) -> np.ndarray:
    """Naive baseline: forecast next-day variance = trailing-window realized variance."""
    s = pd.Series(returns)
    return (s.rolling(window).var().shift(1)).to_numpy()


def vol_managed_returns(returns: np.ndarray, var_forecast: np.ndarray,
                        target_annual_vol: float = 0.12, w_cap: float = 1.0,
                        margin_annual: float = 0.06, cost_bps: float = 1.0):
    """Scale exposure to target vol using a causal variance forecast.

    w_t = clip(target_daily_sigma / sqrt(v_t), 0, w_cap). Strategy return = w_t * r_t,
    minus trading cost on |dw| and margin cost on any leverage (w>1).
    """
    tgt_daily = target_annual_vol / np.sqrt(252)
    v = np.where(np.isfinite(var_forecast) & (var_forecast > 0), var_forecast, np.nan)
    w = np.clip(tgt_daily / np.sqrt(v), 0, w_cap)
    w = pd.Series(w).ffill().fillna(0).to_numpy()
    gross = w * returns
    turn = np.abs(np.diff(w, prepend=w[0]))
    tc = turn * (cost_bps / 1e4)
    lev_cost = np.maximum(w - 1, 0) * (margin_annual / 252)
    net = gross - tc - lev_cost
    return net, w


def metrics(r: np.ndarray) -> dict:
    r = pd.Series(r).dropna().to_numpy()
    if len(r) == 0 or r.std() == 0:
        return dict(ann=0, sharpe=0, maxdd=0, vol=0)
    eq = np.cumprod(1 + r)
    dd = (eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)
    return dict(ann=(1 + r.mean()) ** 252 - 1, sharpe=np.sqrt(252) * r.mean() / r.std(),
                maxdd=float(dd.min()), vol=r.std() * np.sqrt(252))
