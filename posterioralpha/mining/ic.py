"""
IC lab — pre-backtest signal evaluation (Grinold & Kahn).

The miner's gauntlet (`mining.validation`) answers "is this PORTFOLIO's
Sharpe real?"; this module answers the cheaper, earlier question "does this
SCORE contain information at all?" — the cross-sectional information
coefficient between today's scores and forward returns. The rule from
Active Portfolio Management: a signal with mean monthly rank IC consistently
above ~0.02 and an IC t-stat above 2 has earned a backtest; anything else
should die here, before it can consume a trial in the gauntlet.

Usage: build score panels (e.g. via `mining.signals` families on a
SignalContext), then `evaluate_signals(prices, {"name": panel, ...})`.
"""
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_ANN = 252


def forward_returns(prices: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Forward total return over the next `horizon` trading days."""
    return prices.shift(-horizon) / prices - 1.0


def _month_end_trade_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each calendar month present in `index`."""
    dates = []
    for me in pd.Series(index=index, dtype=float).resample("ME").last().index:
        sub = index[index <= me]
        if len(sub):
            dates.append(sub[-1])
    return pd.DatetimeIndex(sorted(set(dates)))


def monthly_ic(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    horizons: Tuple[int, ...] = (21, 63, 126),
    min_names: int = 30,
) -> Dict[int, pd.Series]:
    """
    Spearman rank IC at each month-end, per horizon.

    Returns {horizon: Series of ICs indexed by trade date}.
    """
    trade_dates = _month_end_trade_dates(signal.index)
    out: Dict[int, pd.Series] = {}
    for h in horizons:
        fwd = forward_returns(prices, h)
        ics, dates = [], []
        for t in trade_dates:
            if t not in signal.index or t not in fwd.index:
                continue
            s, f = signal.loc[t], fwd.loc[t]
            mask = s.notna() & f.notna()
            if mask.sum() < min_names:
                continue
            ic, _ = spearmanr(s[mask], f[mask])
            if np.isfinite(ic):
                ics.append(float(ic))
                dates.append(t)
        out[h] = pd.Series(ics, index=dates, name=f"IC_{h}d")
    return out


def ic_summary(ic_series: pd.Series, horizon_days: int) -> Dict[str, float]:
    """
    Mean IC, IC vol, t-stat, hit rate.

    For horizons > 21 days the monthly observations overlap, inflating the
    naive t-stat; shrink the effective sample size by the overlap factor
    (conservative Newey-West-style fix).
    """
    n = len(ic_series)
    if n < 12:
        return {"mean_ic": np.nan, "ic_vol": np.nan, "t_stat": np.nan,
                "hit_rate": np.nan, "n_obs": n}
    n_eff = n / max(1.0, horizon_days / 21.0)
    mean_ic = float(ic_series.mean())
    ic_vol = float(ic_series.std())
    return {
        "mean_ic": mean_ic,
        "ic_vol": ic_vol,
        "t_stat": float(mean_ic / (ic_vol + 1e-12) * np.sqrt(n_eff)),
        "hit_rate": float((ic_series > 0).mean()),
        "n_obs": n,
    }


def quintile_spread(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    horizon: int = 21,
    min_names: int = 30,
) -> Dict[str, float]:
    """
    Equal-weighted Q5 − Q1 spread (gross, no costs) as a monotonicity check:
    an IC that doesn't show up as an ordered quintile pattern is fragile.
    """
    fwd = forward_returns(prices, horizon)
    spreads, q_means = [], []
    for t in _month_end_trade_dates(signal.index):
        if t not in fwd.index:
            continue
        s, f = signal.loc[t], fwd.loc[t]
        mask = s.notna() & f.notna()
        if mask.sum() < min_names:
            continue
        q = pd.qcut(s[mask].rank(method="first"), 5, labels=False)
        means = f[mask].groupby(q).mean()
        if len(means) == 5:
            spreads.append(float(means[4] - means[0]))
            q_means.append(means.values)
    if len(spreads) < 12:
        return {"ann_spread": np.nan, "spread_sharpe": np.nan, "monotonic": np.nan}
    spreads = np.array(spreads)
    ppy = _ANN / horizon
    q_avg = np.mean(np.stack(q_means), axis=0)
    return {
        "ann_spread": float(spreads.mean() * ppy),
        "spread_sharpe": float(spreads.mean() / (spreads.std() + 1e-12)
                               * np.sqrt(ppy / max(1.0, horizon / 21.0))),
        "monotonic": float(np.all(np.diff(q_avg) > 0)),
    }


def evaluate_signals(
    prices: pd.DataFrame,
    signals: Dict[str, pd.DataFrame],
    horizons: Tuple[int, ...] = (21, 63, 126),
) -> pd.DataFrame:
    """
    Full IC report: one row per (signal, horizon) with IC stats; the 21-day
    rows also carry the quintile-spread sanity check.
    """
    rows: List[dict] = []
    for name, sig in signals.items():
        ics = monthly_ic(sig, prices, horizons=horizons)
        qs = quintile_spread(sig, prices, horizon=21)
        for h in horizons:
            row = {"signal": name, "horizon_d": h, **ic_summary(ics[h], h)}
            if h == 21:
                row.update(qs)
            rows.append(row)
    return pd.DataFrame(rows)
