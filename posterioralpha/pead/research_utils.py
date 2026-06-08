"""Shared utilities for PEAD research scripts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def compute_backtest_metrics(quarterly_returns: list[float]) -> dict:
    """Compute standardized backtest metrics from quarterly returns.

    Args:
      quarterly_returns: List of quarterly returns (decimal, e.g., 0.02 = 2%)

    Returns:
      Dict with annual_return, sharpe, max_dd, win_rate, n_seasons
    """
    if not quarterly_returns:
        return {
            "annual_return": 0.0,
            "sharpe": 0.0,
            "max_dd": 0.0,
            "win_rate": 0.0,
            "sortino": 0.0,
            "n_seasons": 0,
        }

    rets = np.array(quarterly_returns)
    mean_ret = np.mean(rets)
    std_ret = np.std(rets)
    annual_ret = (1 + mean_ret) ** 4 - 1
    sharpe = np.sqrt(4) * (mean_ret / std_ret) if std_ret > 0 else 0.0

    # Max drawdown: compute cumulative then find max trough
    cumulative = np.cumprod(1 + rets) - 1
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative - running_max
    max_dd = np.min(drawdowns) if len(drawdowns) > 0 else 0.0

    # Sortino: use downside vol only
    downside_rets = rets[rets < 0]
    downside_vol = np.std(downside_rets) if len(downside_rets) > 0 else 0.0
    sortino = np.sqrt(4) * (mean_ret / downside_vol) if downside_vol > 0 else 0.0

    win_rate = np.sum(rets > 0) / len(rets)

    return {
        "annual_return": annual_ret,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "win_rate": win_rate,
        "sortino": sortino,
        "n_seasons": len(rets),
    }


def backtest_long_only(panel: pd.DataFrame, filter_func, return_col: str = "ret_t21") -> dict:
    """Run long-only quarterly backtest with filter function.

    Args:
      panel: Signal panel with [announcement_date, ret_t21, season, ...]
      filter_func: Function that takes a group and returns filtered group
      return_col: Column name for returns (default "ret_t21")

    Returns:
      Dict with metrics and quarterly_returns
    """
    quarterly_returns = []
    n_positions = []

    for season, group in panel.groupby("season"):
        filtered = filter_func(group)

        if len(filtered) == 0:
            continue

        avg_ret = filtered[return_col].mean() / 100  # Convert % to decimal
        quarterly_returns.append(avg_ret)
        n_positions.append(len(filtered))

    metrics = compute_backtest_metrics(quarterly_returns)
    metrics["quarterly_returns"] = quarterly_returns
    metrics["n_positions"] = n_positions
    metrics["avg_positions"] = np.mean(n_positions) if n_positions else 0

    return metrics


def plot_equity_curves(curves_dict: dict[str, list], title: str, output_path: Path, figsize=(15, 8)):
    """Plot multiple equity curves with legend.

    Args:
      curves_dict: Dict of {label: quarterly_returns_list}
      title: Plot title
      output_path: Save path
      figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for idx, (label, returns) in enumerate(curves_dict.items()):
        if not returns:
            continue

        cumulative = np.cumprod(1 + np.array(returns)) - 1
        equity = 100_000 * (1 + cumulative)

        ax.plot(equity, linewidth=2.5, label=label, color=colors[idx % len(colors)])

    ax.set_xlabel("Quarter", fontweight="bold", fontsize=11)
    ax.set_ylabel("Portfolio Value ($)", fontweight="bold", fontsize=11)
    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x/1e3:.0f}k"))

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_metrics_bars(metrics_dict: dict[str, dict], metric_keys: list[str], title: str, output_path: Path, figsize=(12, 6)):
    """Plot bar charts for multiple metrics across strategies.

    Args:
      metrics_dict: Dict of {strategy: {metric: value}}
      metric_keys: List of metrics to plot (e.g., ["annual_return", "sharpe", "win_rate"])
      title: Plot title
      output_path: Save path
      figsize: Figure size
    """
    fig, axes = plt.subplots(1, len(metric_keys), figsize=figsize)
    if len(metric_keys) == 1:
        axes = [axes]

    strategies = list(metrics_dict.keys())
    x = np.arange(len(strategies))
    width = 0.6

    colors_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for ax, metric in zip(axes, metric_keys):
        values = [metrics_dict[s].get(metric, 0) for s in strategies]

        bars = ax.bar(x, values, width, color=colors_list[:len(strategies)])
        ax.set_xlabel("Strategy", fontweight="bold")
        ax.set_ylabel(metric.replace("_", " ").title(), fontweight="bold")
        ax.set_title(metric.replace("_", " ").title(), fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=45, ha="right")
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height, f"{height:.2f}",
                   ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def add_quarterly_season(panel: pd.DataFrame, date_col: str = "announcement_date") -> pd.DataFrame:
    """Add quarterly season column (handles timezone issues gracefully).

    Args:
      panel: DataFrame with date column
      date_col: Name of datetime column

    Returns:
      Panel with new 'season' column
    """
    import warnings
    warnings.filterwarnings("ignore")

    panel = panel.copy()
    panel[date_col] = pd.to_datetime(panel[date_col], utc=True, errors="coerce")
    panel["season"] = panel[date_col].dt.to_period("Q")

    return panel
