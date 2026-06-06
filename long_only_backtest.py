#!/usr/bin/env python3
"""Long-only backtests for retail: PEAD, Momentum, and Ensemble.

Strategy: Instead of long-short (which requires borrowing), go long high-signal names
and ignore/skip low-signal names. Much simpler for retail.

Variants:
  1. PEAD only: Long stocks with SUE > median
  2. Momentum only: Long stocks with recent price gains > median
  3. Ensemble: Long stocks where BOTH SUE and momentum are above median
  4. SUE + Momentum rank: Rank by ensemble score, take top 30%
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.pead import momentum


def main():
    print("=" * 80)
    print("Long-Only Backtests: PEAD vs Momentum vs Ensemble")
    print("=" * 80)

    # Load signal panel
    panel_path = Path("results/pead/signal_panel.csv")
    if not panel_path.exists():
        print("ERROR: Signal panel not found. Run `python3 run_pead.py` first.")
        return

    print("\n[1] Loading signal panel...")
    panel = pd.read_csv(panel_path)
    panel["announcement_date"] = pd.to_datetime(panel["announcement_date"], utc=True, errors="coerce")
    panel = panel.dropna(subset=["sue", "ret_t21"])
    print(f"  Loaded {len(panel)} announcements")

    # Add momentum (note: this will be slow due to yfinance calls)
    print("\n[2] Computing momentum (this may take 5–10 minutes)...")
    print("  (Fetching historical prices for momentum calculation)")
    panel = momentum.add_momentum_to_panel(panel, momentum_lookback=20)
    panel = panel.dropna(subset=["momentum"])
    print(f"  Successfully computed momentum for {panel['momentum'].notna().sum()} stocks")

    # Standardize signals
    print("\n[3] Standardizing signals to z-scores...")
    panel["sue_z"] = momentum.standardize_signal(panel["sue"])
    panel["mom_z"] = momentum.standardize_signal(panel["momentum"])
    panel["ensemble_z"] = 0.5 * panel["sue_z"] + 0.5 * panel["mom_z"]

    # Backtest each strategy
    print("\n[4] Running long-only backtests (quarterly rebalance)...")
    panel["season"] = panel["announcement_date"].dt.to_period("Q")

    strategies = {
        "PEAD Only": lambda g: g[g["sue_z"] > 0],  # Long high SUE only
        "Momentum Only": lambda g: g[g["mom_z"] > 0],  # Long high momentum only
        "PEAD + Momentum": lambda g: g[(g["sue_z"] > 0) & (g["mom_z"] > 0)],  # Both positive
        "Ensemble Rank": lambda g: g.nlargest(max(1, len(g) // 3), "ensemble_z"),  # Top 33%
    }

    results = {}

    for strategy_name, filter_func in strategies.items():
        print(f"\n  {strategy_name}:")
        quarterly_returns = []
        n_positions = []

        for season, group in panel.groupby("season"):
            filtered = filter_func(group)

            if len(filtered) == 0:
                continue

            avg_ret = filtered["ret_t21"].mean() / 100  # Convert % to decimal
            quarterly_returns.append(avg_ret)
            n_positions.append(len(filtered))

        if quarterly_returns:
            mean_ret = np.mean(quarterly_returns)
            std_ret = np.std(quarterly_returns)
            annual_ret = (1 + mean_ret) ** 4 - 1
            sharpe = np.sqrt(4) * (mean_ret / std_ret) if std_ret > 0 else 0
            win_rate = sum(1 for r in quarterly_returns if r > 0) / len(quarterly_returns)

            results[strategy_name] = {
                "quarterly_returns": quarterly_returns,
                "n_positions": n_positions,
                "mean_quarterly": mean_ret,
                "std_quarterly": std_ret,
                "annual_return": annual_ret,
                "sharpe": sharpe,
                "win_rate": win_rate,
                "n_seasons": len(quarterly_returns),
            }

            print(f"    Annual return: {annual_ret*100:+.2f}%")
            print(f"    Sharpe ratio: {sharpe:.2f}")
            print(f"    Win rate: {win_rate:.1%}")
            print(f"    Avg positions per quarter: {np.mean(n_positions):.0f}")

    # Summary table
    print("\n" + "=" * 80)
    print("LONG-ONLY SUMMARY")
    print("=" * 80)

    summary_rows = []
    for strategy, metrics in results.items():
        summary_rows.append({
            "Strategy": strategy,
            "Annual Return": f"{metrics['annual_return']*100:+.2f}%",
            "Sharpe": f"{metrics['sharpe']:.2f}",
            "Win Rate": f"{metrics['win_rate']:.1%}",
            "Avg Positions": f"{np.mean(metrics['n_positions']):.0f}",
            "Seasons": f"{metrics['n_seasons']}",
        })

    summary_df = pd.DataFrame(summary_rows)
    print()
    print(summary_df.to_string(index=False))

    # Equity curves
    print("\n[5] Creating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (strategy_name, metrics) in enumerate(results.items()):
        quarterly_rets = np.array(metrics["quarterly_returns"])
        cumulative_rets = np.cumprod(1 + quarterly_rets) - 1
        equity = 100_000 * (1 + cumulative_rets)

        ax = axes[idx]
        ax.plot(equity, linewidth=2.5, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][idx])
        ax.fill_between(range(len(equity)), 100_000, equity, alpha=0.3, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][idx])
        ax.set_title(f"{strategy_name}\n(Annual: {metrics['annual_return']*100:+.2f}%, Sharpe: {metrics['sharpe']:.2f})",
                     fontweight="bold")
        ax.set_ylabel("Portfolio Value ($)", fontweight="bold")
        ax.set_xlabel("Quarter", fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x/1e3:.0f}k"))

    plt.suptitle("Long-Only Strategy Comparison: $100k Starting Capital", fontweight="bold", fontsize=14)
    plt.tight_layout()

    output_path = Path("results/pead/long_only_comparison.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_path}")

    # Analysis
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)

    best_strategy = max(results.items(), key=lambda x: x[1]["annual_return"])
    print(f"""
Best Strategy: {best_strategy[0]}
  → Annual return: {best_strategy[1]['annual_return']*100:+.2f}%
  → Sharpe: {best_strategy[1]['sharpe']:.2f}

Why Long-Only Works for Retail:
  1. No borrowing needed (solves the short-selling problem)
  2. Simpler execution (just pick winners, skip losers)
  3. Lower costs (no borrow fees, simpler tax treatment)
  4. Better sleep (psychologically easier to hold winners)

Momentum + PEAD Ensemble:
  - Momentum captures recent price strength (momentum effect)
  - PEAD captures earnings surprise (drift effect)
  - Together: More robust signal, fewer false positives
  - Diversification: Low correlation → better risk-adjusted returns

Retail Implementation:
  - Use these signals with a brokerage that allows:
    * Multiple stock screening tools (e.g., ThinkorSwim, Finviz)
    * Auto-rebalance every quarter
    * Low commissions (most brokers now commission-free)
  - OR: Build a simple Python script to rank stocks and email you quarterly
  - Position sizing: Equal weight or market-cap weight (simpler for retail)
""")

    print("=" * 80)
    print("Done.")
    return results


if __name__ == "__main__":
    main()
