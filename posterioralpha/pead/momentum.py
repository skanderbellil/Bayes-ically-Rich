"""Momentum signal: recent price performance predicts forward returns.

Momentum is one of the strongest empirical anomalies (Jegadeesh & Titman 1993).
Intuition: Winners keep winning (behavioral: trend-chasing, slow information diffusion).

We'll compute 20-day momentum (from t-20 to t-0, where t=announcement date)
and ensemble it with PEAD for a more robust signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf


def compute_momentum(ticker: str, end_date: pd.Timestamp, lookback_days: int = 20) -> float:
    """Compute momentum: (price_today - price_20d_ago) / price_20d_ago.

    Args:
      ticker: Stock ticker
      end_date: End date for momentum calculation (e.g., announcement date)
      lookback_days: How far back to look (default 20 days)

    Returns:
      Momentum as decimal (e.g., 0.05 = 5% gain over past 20 days)
    """
    try:
        # Fetch price data around the announcement date
        start = end_date - pd.Timedelta(days=lookback_days + 10)  # Extra buffer for trading days
        hist = yf.Ticker(ticker).history(start=start, end=end_date, auto_adjust=True)

        if len(hist) < lookback_days:
            return np.nan

        # Get prices
        prices = hist["Close"].values
        if len(prices) < 2:
            return np.nan

        price_today = prices[-1]
        price_lookback = prices[-(lookback_days + 1)] if len(prices) > lookback_days else prices[0]

        momentum = (price_today - price_lookback) / price_lookback
        return momentum

    except Exception:
        return np.nan


def add_momentum_to_panel(panel: pd.DataFrame, momentum_lookback: int = 20) -> pd.DataFrame:
    """Add momentum column to signal panel.

    Args:
      panel: Signal panel with [ticker, announcement_date, sue, ret_t21, ...]
      momentum_lookback: Number of days for momentum calculation

    Returns:
      Panel with new 'momentum' column
    """
    panel = panel.copy()
    panel["announcement_date"] = pd.to_datetime(panel["announcement_date"], utc=True, errors="coerce")

    momentum_values = []
    for _, row in panel.iterrows():
        ticker = row["ticker"]
        ann_date = row["announcement_date"]

        mom = compute_momentum(ticker, ann_date, lookback_days=momentum_lookback)
        momentum_values.append(mom)

    panel["momentum"] = momentum_values
    return panel


def ensemble_sue_momentum(
    sue: float,
    momentum: float,
    sue_weight: float = 0.5,
    momentum_weight: float = 0.5,
) -> float:
    """Combine SUE and momentum into a single score.

    Ensemble approach: weighted average of z-scores.
    Assumes both signals have been standardized (z-score normalized).

    Args:
      sue: SUE signal (e.g., 50 = 50% earnings beat)
      momentum: Momentum (e.g., 0.05 = 5% price gain)
      sue_weight: Weight for SUE (default 0.5)
      momentum_weight: Weight for momentum (default 0.5)

    Returns:
      Ensemble score
    """
    # Simple weighted average (assumes pre-normalized signals)
    ensemble = sue_weight * sue + momentum_weight * momentum
    return ensemble


def standardize_signal(series: pd.Series) -> pd.Series:
    """Standardize a signal to z-scores (mean 0, std 1).

    Args:
      series: Raw signal values

    Returns:
      Z-score standardized series
    """
    mean = series.mean()
    std = series.std()
    if std == 0:
        return series * 0  # All zeros if no variance
    return (series - mean) / std
