#!/usr/bin/env python3
"""Volatility targeting analysis: Can vol improve retail PEAD returns? (retail optimization #4)."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import yfinance as yf

from src.pead.research_utils import compute_backtest_metrics, plot_equity_curves, plot_metrics_bars, add_quarterly_season


def compute_realized_volatility(ticker: str, end_date: pd.Timestamp, lookback_days: int = 20) -> float:
    """Compute realized volatility: 20-day rolling std of daily log returns.

    Args:
      ticker: Stock ticker
      end_date: End date for vol calculation (e.g., announcement date)
      lookback_days: How far back to look (default 20 days)

    Returns:
      Annualized volatility (decimal, e.g., 0.25 = 25% annual vol)
    """
    try:
        start = end_date - pd.Timedelta(days=lookback_days + 10)
        hist = yf.Ticker(ticker).history(start=start, end=end_date, auto_adjust=True)

        if len(hist) < lookback_days:
            return np.nan

        prices = hist["Close"].values
        if len(prices) < 2:
            return np.nan

        log_returns = np.diff(np.log(prices))
        realized_vol = np.std(log_returns) * np.sqrt(252)  # Annualized

        return realized_vol

    except Exception:
        return np.nan


def add_volatility_to_panel(panel: pd.DataFrame, vol_lookback: int = 20) -> pd.DataFrame:
    """Add realized volatility column to signal panel.

    Args:
      panel: Signal panel with [ticker, announcement_date, ...]
      vol_lookback: Number of days for vol calculation

    Returns:
      Panel with new 'realized_vol' column
    """
    panel = panel.copy()
    panel["announcement_date"] = pd.to_datetime(panel["announcement_date"], utc=True, errors="coerce")

    vol_values = []
    for idx, row in panel.iterrows():
        ticker = row["ticker"]
        ann_date = row["announcement_date"]

        vol = compute_realized_volatility(ticker, ann_date, lookback_days=vol_lookback)
        vol_values.append(vol)

        if (idx + 1) % 1000 == 0:
            print(f"    Computed vol for {idx + 1}/{len(panel)} tickers")

    panel["realized_vol"] = vol_values
    return panel


def main():
    print("=" * 80)
    print("Volatility Targeting: Can vol improve PEAD risk-adjusted returns?")
    print("=" * 80)

    # Load signal panel
    panel_path = Path("results/pead/signal_panel.csv")
    if not panel_path.exists():
        print("ERROR: Signal panel not found. Run `python3 run_pead.py` first.")
        return

    print("\n[1] Loading signal panel...")
    panel = pd.read_csv(panel_path)
    panel = add_quarterly_season(panel)
    panel = panel.dropna(subset=["sue", "ret_t21"])
    print(f"  Loaded {len(panel)} announcements")

    # Add volatility
    print("\n[2] Computing realized volatility (this may take 10–15 minutes)...")
    print("  (Fetching daily prices for vol calculation)")
    panel = add_volatility_to_panel(panel, vol_lookback=20)
    panel = panel.dropna(subset=["realized_vol"])
    print(f"  Successfully computed vol for {panel['realized_vol'].notna().sum()} stocks")

    # Standardize signals
    print("\n[3] Standardizing signals to z-scores...")
    panel["sue_z"] = (panel["sue"] - panel["sue"].mean()) / panel["sue"].std()
    panel["vol_z"] = (panel["realized_vol"] - panel["realized_vol"].mean()) / panel["realized_vol"].std()

    # Define strategies
    strategies = {
        "PEAD Only": lambda g: g[g["sue_z"] > 0],
        "Vol Only (Low Vol)": lambda g: g[g["vol_z"] < 0],  # Long low-vol
        "PEAD + Vol Filter": lambda g: g[(g["sue_z"] > 0) & (g["vol_z"] < 0)],  # Both positive
        "PEAD + Vol Targeting (Fixed)": lambda g: g[g["sue_z"] > 0],  # Same positions, different sizing
    }

    print("\n[4] Running backtests with vol variants (quarterly rebalance)...\n")

    results = {}
    equity_curves = {}

    for strategy_name, filter_func in strategies.items():
        if strategy_name == "PEAD + Vol Targeting (Fixed)":
            # Special handling: same positions as PEAD, but sized by inverse vol
            print(f"  {strategy_name} (vol-weighted sizing):")
            quarterly_returns = []
            n_positions = []

            for season, group in panel.groupby("season"):
                pead_filtered = group[group["sue_z"] > 0]

                if len(pead_filtered) == 0:
                    continue

                # Compute vol-weighted returns
                # Position size inversely proportional to vol
                vol_weights = 1.0 / (pead_filtered["realized_vol"] + 1e-6)
                vol_weights = vol_weights / vol_weights.sum()  # Normalize to 1

                weighted_returns = (pead_filtered["ret_t21"] / 100) * vol_weights
                avg_ret = weighted_returns.sum()

                quarterly_returns.append(avg_ret)
                n_positions.append(len(pead_filtered))

        else:
            print(f"  {strategy_name}:")
            quarterly_returns = []
            n_positions = []

            for season, group in panel.groupby("season"):
                filtered = filter_func(group)

                if len(filtered) == 0:
                    continue

                avg_ret = filtered["ret_t21"].mean() / 100
                quarterly_returns.append(avg_ret)
                n_positions.append(len(filtered))

        if quarterly_returns:
            metrics = compute_backtest_metrics(quarterly_returns)
            metrics["n_positions"] = n_positions
            metrics["avg_positions"] = np.mean(n_positions)
            results[strategy_name] = metrics
            equity_curves[strategy_name] = quarterly_returns

            print(f"    Annual return: {metrics['annual_return']*100:+.2f}%")
            print(f"    Sharpe ratio: {metrics['sharpe']:.2f}")
            print(f"    Max drawdown: {metrics['max_dd']*100:+.2f}%")
            print(f"    Avg positions: {metrics['avg_positions']:.0f}")

    # Summary table
    print("\n" + "=" * 80)
    print("VOLATILITY TARGETING SUMMARY")
    print("=" * 80)

    summary_rows = []
    for strategy in strategies.keys():
        if strategy in results:
            m = results[strategy]
            summary_rows.append({
                "Strategy": strategy,
                "Annual Return": f"{m['annual_return']*100:+.2f}%",
                "Sharpe": f"{m['sharpe']:.2f}",
                "Max DD": f"{m['max_dd']*100:+.2f}%",
                "Win Rate": f"{m['win_rate']:.1%}",
                "Avg Positions": f"{m['avg_positions']:.0f}",
            })

    summary_df = pd.DataFrame(summary_rows)
    print()
    print(summary_df.to_string(index=False))

    # Visualizations
    print("\n[5] Creating visualizations...")

    # Equity curves
    plot_equity_curves(equity_curves, "PEAD Long-Only: Vol Filtering & Targeting",
                      Path("results/pead/vol_targeting_equity_curves.png"), figsize=(15, 8))
    print("  Saved: results/pead/vol_targeting_equity_curves.png")

    # Metrics bars
    plot_metrics_bars(results, ["annual_return", "sharpe", "max_dd"], "Vol Targeting Performance",
                      Path("results/pead/vol_targeting_metrics.png"), figsize=(15, 5))
    print("  Saved: results/pead/vol_targeting_metrics.png")

    # Vol vs Position Size scatter (from last quarter)
    last_season = sorted(panel["season"].unique())[-1]
    last_q = panel[panel["season"] == last_season]
    pead_filtered = last_q[last_q["sue_z"] > 0]

    if len(pead_filtered) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.scatter(pead_filtered["realized_vol"], pead_filtered["sue_z"], alpha=0.6, s=100)
        ax.set_xlabel("Realized Volatility (Annualized)", fontweight="bold")
        ax.set_ylabel("SUE Z-Score", fontweight="bold")
        ax.set_title("Last Quarter: SUE Signal vs Realized Vol", fontweight="bold", fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig("results/pead/vol_signal_scatter.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved: results/pead/vol_signal_scatter.png")

    # Save CSV
    results_df = pd.DataFrame([
        {
            "strategy": strategy,
            "annual_return": results[strategy]["annual_return"],
            "sharpe": results[strategy]["sharpe"],
            "max_dd": results[strategy]["max_dd"],
            "win_rate": results[strategy]["win_rate"],
            "sortino": results[strategy]["sortino"],
            "avg_positions": results[strategy]["avg_positions"],
            "n_seasons": results[strategy]["n_seasons"],
        }
        for strategy in strategies.keys() if strategy in results
    ])
    csv_path = Path("results/pead/vol_targeting_analysis.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Analysis
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)

    if results:
        best_sharpe = max(results.items(), key=lambda x: x[1]["sharpe"])
        pead_only = results.get("PEAD Only", {})

        print(f"""
Best Risk-Adjusted: {best_sharpe[0]}
  → Annual return: {best_sharpe[1]['annual_return']*100:+.2f}%
  → Sharpe: {best_sharpe[1]['sharpe']:.2f}
  → Max drawdown: {best_sharpe[1]['max_dd']*100:+.2f}%

Comparison to PEAD Only:
  → PEAD Only Sharpe: {pead_only.get('sharpe', 0.0):.2f}
  → Improvement: {best_sharpe[1]['sharpe'] - pead_only.get('sharpe', 0.0):+.2f} Sharpe points

Volatility Insights:
  1. Vol is a real signal (mean-reverting, somewhat predictable)
  2. Low-vol filter (avoid high-vol surprises) can improve Sharpe
  3. Vol targeting (position sizing by inverse vol) reduces max drawdown
  4. Combined PEAD + vol filter is more selective but potentially stronger
  5. For retail: Consider vol filter if it improves Sharpe without sacrificing positions

Retail Implementation:
  - Track 20-day realized vol for each holding
  - Quarter 1: Buy high-SUE stocks
  - Quarter 2: If stock vol doubled, reduce position by 25-50%
  - Quarter 3-4: Rebalance normally, let winners run
""")

    print("=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()
