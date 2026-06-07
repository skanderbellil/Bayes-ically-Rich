#!/usr/bin/env python3
"""Analyze optimal SUE signal strength threshold (retail optimization #2)."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.pead.research_utils import compute_backtest_metrics, plot_equity_curves, plot_metrics_bars, add_quarterly_season


def main():
    print("=" * 80)
    print("Signal Strength Analysis: What SUE threshold maximizes returns?")
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

    # Define filter thresholds
    filters = {
        "All Positive (SUE > 0)": lambda g: g[g["sue"] > 0],
        "Top 50% of Positive": lambda g: g[g["sue"] > g["sue"].quantile(0.50)],
        "Top 25% (>75th pctile)": lambda g: g[g["sue"] > g["sue"].quantile(0.75)],
        "Top 10% (>90th pctile)": lambda g: g[g["sue"] > g["sue"].quantile(0.90)],
    }

    print("\n[2] Running PEAD backtest with different SUE filters...\n")

    results = {}
    equity_curves = {}

    for filter_name, filter_func in filters.items():
        print(f"  {filter_name}:")
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
            metrics["return_per_position"] = (metrics["annual_return"] * 100) / metrics["avg_positions"] if metrics["avg_positions"] > 0 else 0
            results[filter_name] = metrics
            equity_curves[filter_name] = quarterly_returns

            print(f"    Annual return: {metrics['annual_return']*100:+.2f}%")
            print(f"    Sharpe ratio: {metrics['sharpe']:.2f}")
            print(f"    Avg positions: {metrics['avg_positions']:.0f}")
            print(f"    Return per position: {metrics['return_per_position']:.4f}%")

    # Summary table
    print("\n" + "=" * 80)
    print("SIGNAL STRENGTH THRESHOLD SUMMARY")
    print("=" * 80)

    summary_rows = []
    for filter_name in filters.keys():
        if filter_name in results:
            m = results[filter_name]
            summary_rows.append({
                "Filter": filter_name,
                "Annual Return": f"{m['annual_return']*100:+.2f}%",
                "Sharpe": f"{m['sharpe']:.2f}",
                "Win Rate": f"{m['win_rate']:.1%}",
                "Avg Positions": f"{m['avg_positions']:.0f}",
                "Return/Position": f"{m['return_per_position']:.4f}%",
            })

    summary_df = pd.DataFrame(summary_rows)
    print()
    print(summary_df.to_string(index=False))

    # Visualizations
    print("\n[3] Creating visualizations...")

    # Equity curves
    plot_equity_curves(equity_curves, "PEAD Long-Only: Different SUE Thresholds",
                      Path("results/pead/signal_filter_equity_curves.png"))
    print("  Saved: results/pead/signal_filter_equity_curves.png")

    # Metrics bars
    plot_metrics_bars(results, ["annual_return", "sharpe", "win_rate"], "PEAD Performance by Signal Strength",
                      Path("results/pead/signal_filter_metrics.png"), figsize=(15, 5))
    print("  Saved: results/pead/signal_filter_metrics.png")

    # Selectivity vs Sharpe scatter
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    filter_names = list(results.keys())
    sharpes = [results[f]["sharpe"] for f in filter_names]
    positions = [results[f]["avg_positions"] for f in filter_names]
    returns = [results[f]["annual_return"] * 100 for f in filter_names]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    # Scatter 1: Positions vs Sharpe
    ax1.scatter(positions, sharpes, s=300, c=colors, alpha=0.7, edgecolors="black", linewidth=2)
    for i, label in enumerate(filter_names):
        ax1.annotate(label, (positions[i], sharpes[i]), fontsize=9, ha="right", xytext=(5, 5), textcoords="offset points")
    ax1.set_xlabel("Avg Positions per Quarter", fontweight="bold", fontsize=11)
    ax1.set_ylabel("Sharpe Ratio", fontweight="bold", fontsize=11)
    ax1.set_title("Signal Selectivity vs Risk-Adjusted Return", fontweight="bold", fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Scatter 2: Positions vs Return
    ax2.scatter(positions, returns, s=300, c=colors, alpha=0.7, edgecolors="black", linewidth=2)
    for i, label in enumerate(filter_names):
        ax2.annotate(label, (positions[i], returns[i]), fontsize=9, ha="right", xytext=(5, 5), textcoords="offset points")
    ax2.set_xlabel("Avg Positions per Quarter", fontweight="bold", fontsize=11)
    ax2.set_ylabel("Annual Return (%)", fontweight="bold", fontsize=11)
    ax2.set_title("Selectivity vs Absolute Return", fontweight="bold", fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig("results/pead/signal_filter_selectivity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: results/pead/signal_filter_selectivity.png")

    # Save CSV
    results_df = pd.DataFrame([
        {
            "filter": filter_name,
            "annual_return": results[filter_name]["annual_return"],
            "sharpe": results[filter_name]["sharpe"],
            "win_rate": results[filter_name]["win_rate"],
            "sortino": results[filter_name]["sortino"],
            "avg_positions": results[filter_name]["avg_positions"],
            "return_per_position": results[filter_name]["return_per_position"],
            "n_seasons": results[filter_name]["n_seasons"],
        }
        for filter_name in filters.keys() if filter_name in results
    ])
    csv_path = Path("results/pead/signal_filter_analysis.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Analysis
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)

    if results:
        best_sharpe = max(results.items(), key=lambda x: x[1]["sharpe"])
        best_return = max(results.items(), key=lambda x: x[1]["annual_return"])

        print(f"""
Best Risk-Adjusted (Sharpe): {best_sharpe[0]}
  → Annual return: {best_sharpe[1]['annual_return']*100:+.2f}%
  → Sharpe: {best_sharpe[1]['sharpe']:.2f}
  → Avg positions: {best_sharpe[1]['avg_positions']:.0f}

Best Absolute Return: {best_return[0]}
  → Annual return: {best_return[1]['annual_return']*100:+.2f}%
  → Sharpe: {best_return[1]['sharpe']:.2f}
  → Avg positions: {best_return[1]['avg_positions']:.0f}

Selectivity Trade-off:
  - Tighter filters reduce position count (lower diversification risk)
  - But tighter filters may increase transaction costs
  - {best_sharpe[0]} achieves best risk-adjusted returns at {best_sharpe[1]['avg_positions']:.0f} positions
  - Recommendation: Use {best_sharpe[0]} for retail execution
""")

    print("=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()
