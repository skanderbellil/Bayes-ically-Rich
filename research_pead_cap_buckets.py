#!/usr/bin/env python3
"""Analyze PEAD performance by market cap bucket (retail optimization #1)."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.pead.research_utils import compute_backtest_metrics, plot_equity_curves, plot_metrics_bars, add_quarterly_season


def main():
    print("=" * 80)
    print("Market Cap Bucket Analysis: Which cap bucket suits long-only PEAD?")
    print("=" * 80)

    # Load signal panel
    panel_path = Path("results/pead/signal_panel.csv")
    if not panel_path.exists():
        print("ERROR: Signal panel not found. Run `python3 run_pead.py` first.")
        return

    print("\n[1] Loading signal panel...")
    panel = pd.read_csv(panel_path)
    panel = add_quarterly_season(panel)
    panel = panel.dropna(subset=["sue", "ret_t21", "market_cap"])
    print(f"  Loaded {len(panel)} announcements")

    # Define cap buckets
    cap_buckets = ["Mega Cap", "Large Cap", "Mid Cap", "Small Cap", "Micro Cap", "Nano Cap"]

    print("\n[2] Running PEAD-only backtest by cap bucket...\n")

    results = {}
    equity_curves = {}

    for bucket in cap_buckets:
        bucket_panel = panel[panel["market_cap"] == bucket].copy()

        if len(bucket_panel) == 0:
            print(f"  {bucket}: No data")
            continue

        print(f"  {bucket}:")
        quarterly_returns = []
        n_positions = []

        for season, group in bucket_panel.groupby("season"):
            # Long high SUE (above median)
            median_sue = group["sue"].median()
            filtered = group[group["sue"] > median_sue]

            if len(filtered) == 0:
                continue

            avg_ret = filtered["ret_t21"].mean() / 100
            quarterly_returns.append(avg_ret)
            n_positions.append(len(filtered))

        if quarterly_returns:
            metrics = compute_backtest_metrics(quarterly_returns)
            metrics["n_positions"] = n_positions
            metrics["avg_positions"] = np.mean(n_positions)
            results[bucket] = metrics
            equity_curves[bucket] = quarterly_returns

            print(f"    Annual return: {metrics['annual_return']*100:+.2f}%")
            print(f"    Sharpe ratio: {metrics['sharpe']:.2f}")
            print(f"    Win rate: {metrics['win_rate']:.1%}")
            print(f"    Max drawdown: {metrics['max_dd']*100:+.2f}%")
            print(f"    Avg positions: {metrics['avg_positions']:.0f}")

    # Summary table
    print("\n" + "=" * 80)
    print("MARKET CAP BUCKET SUMMARY")
    print("=" * 80)

    summary_rows = []
    for bucket in cap_buckets:
        if bucket in results:
            m = results[bucket]
            summary_rows.append({
                "Cap Bucket": bucket,
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
    print("\n[3] Creating visualizations...")

    # Equity curves
    plot_equity_curves(equity_curves, "PEAD Long-Only by Market Cap Bucket", Path("results/pead/cap_bucket_equity_curves.png"))
    print("  Saved: results/pead/cap_bucket_equity_curves.png")

    # Metrics bars
    plot_metrics_bars(results, ["annual_return", "sharpe", "max_dd"], "PEAD Performance by Cap Bucket",
                      Path("results/pead/cap_bucket_metrics.png"), figsize=(15, 5))
    print("  Saved: results/pead/cap_bucket_metrics.png")

    # Save CSV
    results_df = pd.DataFrame([
        {
            "bucket": bucket,
            "annual_return": results[bucket]["annual_return"],
            "sharpe": results[bucket]["sharpe"],
            "max_dd": results[bucket]["max_dd"],
            "win_rate": results[bucket]["win_rate"],
            "sortino": results[bucket]["sortino"],
            "avg_positions": results[bucket]["avg_positions"],
            "n_seasons": results[bucket]["n_seasons"],
        }
        for bucket in cap_buckets if bucket in results
    ])
    csv_path = Path("results/pead/cap_bucket_analysis.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Analysis
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)

    if results:
        best_bucket = max(results.items(), key=lambda x: x[1]["sharpe"])
        best_return = max(results.items(), key=lambda x: x[1]["annual_return"])

        print(f"""
Best Risk-Adjusted (Sharpe): {best_bucket[0]}
  → Annual return: {best_bucket[1]['annual_return']*100:+.2f}%
  → Sharpe: {best_bucket[1]['sharpe']:.2f}
  → Max drawdown: {best_bucket[1]['max_dd']*100:+.2f}%

Best Absolute Return: {best_return[0]}
  → Annual return: {best_return[1]['annual_return']*100:+.2f}%
  → Sharpe: {best_return[1]['sharpe']:.2f}

Retail Recommendation:
  - Focus on {best_bucket[0]} for the best risk-adjusted returns
  - Smaller buckets have higher returns but more volatility
  - Larger buckets have lower volatility but less alpha
  - Consider diversifying across buckets (small + mid cap = balanced)
""")

    print("=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()
