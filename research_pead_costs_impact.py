#!/usr/bin/env python3
"""Cost-adjusted analysis: Do returns survive realistic trading costs? (retail optimization #3)."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.pead import costs
from src.pead.research_utils import compute_backtest_metrics, plot_equity_curves, plot_metrics_bars, add_quarterly_season


def apply_quarterly_cost(quarterly_return: float, cost_model: costs.CostModel) -> float:
    """Apply quarterly trading cost to a quarterly return.

    Args:
      quarterly_return: Quarterly return (decimal, e.g., 0.02 = 2%)
      cost_model: Cost model for the bucket

    Returns:
      Net quarterly return after costs
    """
    quarterly_cost_bps = costs.compute_quarterly_cost(cost_model, quarterly_return * 100)
    quarterly_cost_decimal = quarterly_cost_bps / 10_000
    return quarterly_return - quarterly_cost_decimal


def main():
    print("=" * 80)
    print("Cost Impact Analysis: Do PEAD returns survive realistic friction?")
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

    cap_buckets = ["Mega Cap", "Large Cap", "Mid Cap", "Small Cap", "Micro Cap", "Nano Cap"]
    cost_models = costs.get_cost_models()

    print("\n[2] Analyzing gross vs net returns by cap bucket...\n")

    results = {}

    for bucket in cap_buckets:
        bucket_panel = panel[panel["market_cap"] == bucket].copy()

        if len(bucket_panel) == 0:
            continue

        # Compute gross returns
        quarterly_returns_gross = []
        quarterly_returns_net = []
        n_positions = []

        cost_model = cost_models[bucket]

        for season, group in bucket_panel.groupby("season"):
            median_sue = group["sue"].median()
            filtered = group[group["sue"] > median_sue]

            if len(filtered) == 0:
                continue

            avg_ret_gross = filtered["ret_t21"].mean() / 100
            avg_ret_net = apply_quarterly_cost(avg_ret_gross, cost_model)

            quarterly_returns_gross.append(avg_ret_gross)
            quarterly_returns_net.append(avg_ret_net)
            n_positions.append(len(filtered))

        if quarterly_returns_gross:
            metrics_gross = compute_backtest_metrics(quarterly_returns_gross)
            metrics_net = compute_backtest_metrics(quarterly_returns_net)

            # Compute average quarterly cost
            quarterly_cost_bps = costs.compute_quarterly_cost(cost_model, np.mean(quarterly_returns_gross) * 100)
            annual_cost_bps = quarterly_cost_bps * 4
            annual_cost_pct = annual_cost_bps / 100

            results[bucket] = {
                "gross_annual": metrics_gross["annual_return"],
                "net_annual": metrics_net["annual_return"],
                "gross_sharpe": metrics_gross["sharpe"],
                "net_sharpe": metrics_net["sharpe"],
                "cost_bps": annual_cost_bps,
                "cost_pct": annual_cost_pct,
                "survival": "✓" if metrics_net["annual_return"] > 0 else "✗",
                "avg_positions": np.mean(n_positions),
            }

            print(f"  {bucket}:")
            print(f"    Gross annual: {metrics_gross['annual_return']*100:+.2f}% (Sharpe: {metrics_gross['sharpe']:.2f})")
            print(f"    Annual costs: {annual_cost_bps:.0f} bps ({annual_cost_pct:.2f}%)")
            print(f"    Net annual:   {metrics_net['annual_return']*100:+.2f}% (Sharpe: {metrics_net['sharpe']:.2f})")
            print(f"    Viable: {results[bucket]['survival']}")

    # Summary table
    print("\n" + "=" * 80)
    print("COST-ADJUSTED SUMMARY")
    print("=" * 80)

    summary_rows = []
    for bucket in cap_buckets:
        if bucket in results:
            r = results[bucket]
            summary_rows.append({
                "Cap Bucket": bucket,
                "Gross Return": f"{r['gross_annual']*100:+.2f}%",
                "Annual Costs": f"{r['cost_bps']:.0f} bps",
                "Net Return": f"{r['net_annual']*100:+.2f}%",
                "Gross Sharpe": f"{r['gross_sharpe']:.2f}",
                "Net Sharpe": f"{r['net_sharpe']:.2f}",
                "Viable": r['survival'],
            })

    summary_df = pd.DataFrame(summary_rows)
    print()
    print(summary_df.to_string(index=False))

    # Visualizations
    print("\n[3] Creating visualizations...")

    # Gross vs Net Returns Bar Chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    buckets = list(results.keys())
    gross_returns = [results[b]["gross_annual"] * 100 for b in buckets]
    net_returns = [results[b]["net_annual"] * 100 for b in buckets]
    costs_bps = [results[b]["cost_bps"] for b in buckets]

    x = np.arange(len(buckets))
    width = 0.35

    # Chart 1: Gross vs Net
    ax = axes[0]
    bars1 = ax.bar(x - width / 2, gross_returns, width, label="Gross", color="#2ca02c", alpha=0.8)
    bars2 = ax.bar(x + width / 2, net_returns, width, label="Net (after costs)", color="#d62728", alpha=0.8)

    ax.set_xlabel("Market Cap Bucket", fontweight="bold")
    ax.set_ylabel("Annual Return (%)", fontweight="bold")
    ax.set_title("PEAD Returns: Gross vs Net", fontweight="bold", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(buckets, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(0, color="black", linestyle="-", linewidth=0.5)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height, f"{height:.1f}%",
               ha="center", va="bottom" if height >= 0 else "top", fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height, f"{height:.1f}%",
               ha="center", va="bottom" if height >= 0 else "top", fontsize=8)

    # Chart 2: Cost breakdown
    ax = axes[1]
    bars = ax.bar(buckets, costs_bps, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"][:len(buckets)], alpha=0.8)
    ax.set_ylabel("Annual Cost (basis points)", fontweight="bold")
    ax.set_title("Annual Trading Costs by Cap Bucket", fontweight="bold", fontsize=12)
    ax.set_xticklabels(buckets, rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height, f"{height:.0f} bps",
               ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig.savefig("results/pead/cost_impact_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: results/pead/cost_impact_analysis.png")

    # Save CSV
    results_df = pd.DataFrame([
        {
            "bucket": bucket,
            "gross_annual": results[bucket]["gross_annual"],
            "net_annual": results[bucket]["net_annual"],
            "annual_cost_bps": results[bucket]["cost_bps"],
            "annual_cost_pct": results[bucket]["cost_pct"],
            "gross_sharpe": results[bucket]["gross_sharpe"],
            "net_sharpe": results[bucket]["net_sharpe"],
            "viable": results[bucket]["survival"],
        }
        for bucket in cap_buckets if bucket in results
    ])
    csv_path = Path("results/pead/cost_impact_analysis.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Analysis
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)

    viable = [b for b in cap_buckets if b in results and results[b]["survival"] == "✓"]
    nonviable = [b for b in cap_buckets if b in results and results[b]["survival"] == "✗"]

    print(f"""
Viable Buckets (Net Return > 0):
  {', '.join(viable) if viable else 'None'}

Non-Viable Buckets (Net Return < 0):
  {', '.join(nonviable) if nonviable else 'None'}

Cost Impact:
  - Larger costs in smaller-cap (higher friction)
  - Large-cap/Mega-cap: Low costs, moderate returns
  - Mid-cap: Best balance (moderate costs, decent returns)
  - Small-cap and below: High costs eat into returns

Retail Recommendation:
  - Focus on {viable[0] if viable else 'Large Cap'} for net-positive alpha
  - Avoid {nonviable[0] if nonviable else 'Micro Cap'} and below (costs prohibitive)
  - Consider equal-weight blend of viable buckets
""")

    print("=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()
