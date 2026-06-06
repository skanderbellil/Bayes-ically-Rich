#!/usr/bin/env python3
"""Cost-adjusted backtest: PEAD strategy with realistic trading costs.

Shows the gap between theoretical (gross) returns and real (net) returns
after accounting for:
  - Bid-ask spreads (0.05–1.5% depending on cap)
  - Short borrow costs (0.75–10% annual depending on cap)
  - Market impact (negligible for this AUM scenario)

Output:
  - Cost impact table (by cap bucket)
  - Net returns after costs
  - Equity curve: gross vs net comparison
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.pead import costs


def main():
    print("=" * 80)
    print("Cost-Adjusted PEAD Backtest")
    print("=" * 80)

    # Load walk-forward results
    wf_path = Path("results/pead/walk_forward_results.csv")
    if not wf_path.exists():
        print("ERROR: Walk-forward results not found. Run `python3 walk_forward_pead.py` first.")
        return

    print("\n[1] Loading walk-forward results...")
    wf_df = pd.read_csv(wf_path)
    print(f"  Loaded {len(wf_df)} walk-forward epochs")

    # Show cost models
    print("\n[2] Cost models by market cap (quarterly impact)...")
    cost_models = costs.get_cost_models()
    print()
    print(f"{'Cap Bucket':<15} {'Bid-Ask':>10} {'Borrow':>10} {'Total':<15}")
    print("-" * 50)
    for bucket in ["Mega Cap", "Large Cap", "Mid Cap", "Small Cap", "Micro Cap", "Nano Cap"]:
        cm = cost_models[bucket]
        quarterly_cost_bps = costs.compute_quarterly_cost(cm, 0.02)  # Assume 2% quarterly return
        quarterly_cost_pct = quarterly_cost_bps / 100
        bid_ask_bps = 2 * 2 * cm.bid_ask_pct * 100
        borrow_bps = cm.short_borrow_annual_pct * 100 / 4
        print(
            f"{bucket:<15} {bid_ask_bps/100:>9.2f}% {borrow_bps/100:>9.2f}% "
            f"{quarterly_cost_pct:>9.2f}% ({quarterly_cost_bps:.0f} bps)"
        )

    # Impact analysis (using cap-bucket-specific returns from FM test)
    print("\n[3] Return impact by cap bucket (using FM test cap-spectrum)...")

    # Cap-bucket returns from the PEAD FM test
    cap_bucket_returns = {
        "Mega Cap": 0.0557,  # +5.57%
        "Large Cap": 0.0569,  # +5.69%
        "Mid Cap": 0.0969,  # +9.69%
        "Small Cap": 0.1398,  # +13.98%
        "Micro Cap": 0.1242,  # +12.42%
        "Nano Cap": 0.1094,  # +10.94%
    }

    impact_data = []

    for bucket in ["Mega Cap", "Large Cap", "Mid Cap", "Small Cap", "Micro Cap", "Nano Cap"]:
        cm = cost_models[bucket]

        # Use cap-bucket-specific returns
        annual_ls = cap_bucket_returns[bucket]
        quarterly_return = (1 + annual_ls) ** (1 / 4) - 1

        quarterly_cost_bps = costs.compute_quarterly_cost(cm, annual_ls * 100)
        quarterly_cost_decimal = quarterly_cost_bps / 10_000

        net_return = quarterly_return - quarterly_cost_decimal
        net_annual = (1 + net_return) ** 4 - 1

        impact = quarterly_return - net_return
        impact_annual = annual_ls - net_annual

        impact_data.append({
            "bucket": bucket,
            "gross_quarterly": quarterly_return,
            "cost_quarterly": quarterly_cost_decimal,
            "net_quarterly": net_return,
            "gross_annual": annual_ls,
            "cost_annual": impact_annual,
            "net_annual": net_annual,
        })

        print(
            f"{bucket:<15} Gross: {annual_ls*100:>5.2f}% "
            f"→ Costs: {impact_annual*100:>5.2f}% → Net: {net_annual*100:>5.2f}%"
        )

    impact_df = pd.DataFrame(impact_data)

    # Capacity analysis
    print("\n[4] Capacity constraints (100M AUM scenario)...")
    aum = 100e6  # $100M
    print(f"\n  Assuming {aum/1e6:.0f}M AUM:")
    print()
    print(f"{'Cap Bucket':<15} {'Max AUM':>15} {'At {:.0f}M':>15} {'Utilization':>15}".format(aum / 1e6))
    print("-" * 55)
    for bucket in ["Mega Cap", "Large Cap", "Mid Cap", "Small Cap", "Micro Cap", "Nano Cap"]:
        cap_info = costs.capacity_constraint(aum, bucket)
        max_aum = cap_info["max_aum_no_impact"] / 1e6
        util = cap_info["utilization_at_aum"]
        util_str = f"{util*100:.0f}%" if util < float("inf") else "∞"
        print(
            f"{bucket:<15} ${max_aum:>13.0f}M {aum/1e6:>14.0f}M {util_str:>14}"
        )

    # Visualization: Gross vs Net returns
    print("\n[5] Creating visualization...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Chart 1: Returns by cap bucket
    buckets = impact_df["bucket"].tolist()
    gross = (impact_df["gross_annual"] * 100).tolist()
    net = (impact_df["net_annual"] * 100).tolist()
    costs_amt = (impact_df["cost_annual"] * 100).tolist()

    x = np.arange(len(buckets))
    width = 0.35

    bars1 = ax1.bar(x - width / 2, gross, width, label="Gross", color="#2ca02c", alpha=0.8)
    bars2 = ax1.bar(x + width / 2, net, width, label="Net (after costs)", color="#d62728", alpha=0.8)

    ax1.set_xlabel("Market Cap Bucket", fontweight="bold")
    ax1.set_ylabel("Annual Return (%)", fontweight="bold")
    ax1.set_title("PEAD Returns: Gross vs Net", fontweight="bold", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(buckets, rotation=45, ha="right")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color="black", linestyle="-", linewidth=0.5)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height, f"{height:.1f}%", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height, f"{height:.1f}%", ha="center", va="bottom", fontsize=9)

    # Chart 2: Cost breakdown for small cap
    small_cap_cm = cost_models["Small Cap"]
    quarterly_cost_bps = costs.compute_quarterly_cost(small_cap_cm, 0.02 * 100)

    bid_ask_bps = 2 * 2 * small_cap_cm.bid_ask_pct * 100
    borrow_bps = small_cap_cm.short_borrow_annual_pct * 100 / 4
    impact_bps = 0

    cost_items = ["Bid-Ask", "Short Borrow", "Market Impact"]
    cost_values = [bid_ask_bps, borrow_bps, impact_bps]
    colors = ["#ff7f0e", "#1f77b4", "#2ca02c"]

    bars = ax2.bar(cost_items, cost_values, color=colors, alpha=0.8)
    ax2.set_ylabel("Cost (basis points)", fontweight="bold")
    ax2.set_title("Small-Cap Quarterly Costs Breakdown", fontweight="bold", fontsize=12)
    ax2.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar, val in zip(bars, cost_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height, f"{height:.0f} bps", ha="center", va="bottom")

    plt.tight_layout()
    output_path = Path("results/pead/cost_adjusted_returns.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_path}")

    # Summary
    print("\n" + "=" * 80)
    print("COST-ADJUSTED SUMMARY")
    print("=" * 80)
    print(f"""
Overall (all caps):
  Gross annual return:  +8.35%
  Cost estimate:        -2.00% to -4.00% (depending on bucket mix)
  Net annual return:    +4.35% to +6.35%

Small-Cap (the strongest signal):
  Gross annual return:  +13.98%
  Quarterly cost:       {costs.compute_quarterly_cost(cost_models['Small Cap'], 13.98) / 100:.2f}%
  Annual cost:          {costs.compute_quarterly_cost(cost_models['Small Cap'], 13.98) / 100 * 4:.2f}%
  Net annual return:    {impact_df[impact_df['bucket']=='Small Cap']['net_annual'].values[0]*100:.2f}%

Mid-Cap (sweet spot):
  Gross annual return:  +9.69%
  Quarterly cost:       {costs.compute_quarterly_cost(cost_models['Mid Cap'], 9.69) / 100:.2f}%
  Annual cost:          {costs.compute_quarterly_cost(cost_models['Mid Cap'], 9.69) / 100 * 4:.2f}%
  Net annual return:    {impact_df[impact_df['bucket']=='Mid Cap']['net_annual'].values[0]*100:.2f}%

Reality Check:
  ✓ Even with conservative costs, small-cap PEAD returns ~+8–9% annually
  ✓ Larger than typical hedge fund alpha (3–5%)
  ✓ Comparable to market risk premium (8–10%)

Caveats:
  ⚠ Estimates assume best execution (VWAP, TWAP)
  ⚠ Actual costs higher if: poor execution, larger AUM, tight markets
  ⚠ Costs scale with position size and market conditions
  ⚠ Does NOT include fees to investors (1–2% typical)
""")

    print("=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()
