"""Simple backtest harness for PEAD long-short strategy.

Strategy:
  Each earnings season, go long on SUE > median, short on SUE <= median.
  Hold for 21 days (one month), then rebalance.

Metrics:
  - Cumulative return
  - Sharpe ratio
  - Max drawdown
  - Win rate
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd


class BacktestResult(NamedTuple):
    """Backtest summary."""

    cum_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    equity_curve: pd.Series


def backtest_pead_simple(
    panel: pd.DataFrame,
    initial_capital: float = 100_000,
    lookback_days: int = 21,
    transaction_cost_bps: float = 20,
) -> BacktestResult:
    """Run a simple PEAD long-short backtest.

    Args:
      panel: signal panel with columns [announcement_date, ticker, sue, ret_t21]
      initial_capital: starting capital
      lookback_days: hold period (21 = one month)
      transaction_cost_bps: round-trip cost in basis points (20 = 0.2%)

    Returns:
      BacktestResult with equity curve and metrics.
    """
    if panel.empty:
        raise ValueError("Panel is empty")

    # Group by quarter (same as FM test) and apply the signal.
    panel = panel.copy()
    panel["announcement_date"] = pd.to_datetime(panel["announcement_date"], utc=True, errors="coerce")
    panel = panel.dropna(subset=["announcement_date", "ret_t21"])
    panel = panel.sort_values("announcement_date")

    # Group by quarter to match Fama-MacBeth methodology
    panel["season"] = panel["announcement_date"].dt.to_period("Q")

    # Compute returns on holdings.
    portfolio_returns = []
    dates = []

    for season, group in panel.groupby("season"):
        if len(group) < 5:  # need sufficient observations
            continue

        median_sue = group["sue"].median()
        long = group[group["sue"] > median_sue]
        short = group[group["sue"] <= median_sue]

        if len(long) == 0 or len(short) == 0:
            continue

        # Average return for long and short legs (in %).
        long_ret = long["ret_t21"].mean() / 100  # convert from % to decimal
        short_ret = short["ret_t21"].mean() / 100

        # Long-short return (equal-weight long and short).
        ls_return = long_ret - short_ret

        # Subtract transaction cost (round-trip: open + close).
        tc = transaction_cost_bps / 10_000
        net_return = ls_return - tc

        portfolio_returns.append(net_return)
        dates.append(season.end_time)

    if not portfolio_returns:
        raise ValueError("No valid trading periods")

    # Build equity curve.
    returns = np.array(portfolio_returns)
    cumulative_rets = np.cumprod(1 + returns) - 1
    equity = initial_capital * (1 + cumulative_rets)

    # Create series.
    eq_series = pd.Series(equity, index=dates, name="equity")

    # Metrics.
    total_return = cumulative_rets[-1]
    years = (dates[-1] - dates[0]).days / 365.25
    annual_return = (1 + total_return) ** (1 / years) - 1

    # Sharpe (assuming ~252 trading days/year).
    daily_returns = returns
    sharpe = np.sqrt(252) * (daily_returns.mean() / daily_returns.std()) if daily_returns.std() > 0 else 0

    # Max drawdown.
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    max_dd = drawdown.min()

    # Win rate.
    win_rate = (returns > 0).sum() / len(returns)

    return BacktestResult(
        cum_return=total_return,
        annual_return=annual_return,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        win_rate=win_rate,
        equity_curve=eq_series,
    )


def fetch_spy_benchmark(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.Series:
    """Fetch SPY returns for benchmark.

    Returns a series of daily SPY closes from start_date to end_date.
    """
    import yfinance as yf

    spy = yf.Ticker("SPY")
    hist = spy.history(start=start_date, end=end_date, auto_adjust=True)
    if hist.empty:
        raise ValueError(f"No SPY data between {start_date} and {end_date}")
    return hist["Close"]


def compute_benchmark_equity(
    spy_prices: pd.Series, initial_capital: float = 100_000
) -> pd.Series:
    """Convert SPY price series to equity curve."""
    returns = spy_prices.pct_change().fillna(0)
    cumulative_rets = np.cumprod(1 + returns) - 1
    equity = initial_capital * (1 + cumulative_rets)
    return pd.Series(equity, index=spy_prices.index, name="SPY_equity")
