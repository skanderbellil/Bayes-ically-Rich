#!/usr/bin/env python3
"""Compare optimized PEAD strategies vs buy & hold market benchmark."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

from src.pead.research_utils import add_quarterly_season


def compute_realized_volatility(ticker: str, end_date: pd.Timestamp, lookback_days: int = 20) -> float:
    """Compute realized volatility: 20-day rolling std of daily log returns."""
    try:
        start = end_date - pd.Timedelta(days=lookback_days + 10)
        hist = yf.Ticker(ticker).history(start=start, end=end_date, auto_adjust=True)

        if len(hist) < lookback_days:
            return np.nan

        prices = hist["Close"].values
        if len(prices) < 2:
            return np.nan

        log_returns = np.diff(np.log(prices))
        realized_vol = np.std(log_returns) * np.sqrt(252)

        return realized_vol
    except Exception:
        return np.nan


def main():
    print("=" * 80)
    print("Strategy Comparison: Optimized PEAD vs Buy & Hold")
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
    print("\n[2] Computing realized volatility...")
    vol_values = []
    for idx, row in panel.iterrows():
        ticker = row["ticker"]
        ann_date = pd.to_datetime(row["announcement_date"], utc=True, errors="coerce")
        vol = compute_realized_volatility(ticker, ann_date, lookback_days=20)
        vol_values.append(vol)
        if (idx + 1) % 5000 == 0:
            print(f"    {idx + 1}/{len(panel)}")

    panel["realized_vol"] = vol_values
    panel = panel.dropna(subset=["realized_vol"])
    print(f"  Successfully computed vol for {len(panel)} entries")

    # Standardize signals
    print("\n[3] Standardizing signals...")
    panel["sue_z"] = (panel["sue"] - panel["sue"].mean()) / panel["sue"].std()
    panel["vol_z"] = (panel["realized_vol"] - panel["realized_vol"].mean()) / panel["realized_vol"].std()

    # Define strategies
    print("\n[4] Running strategy backtests...\n")

    strategies = {
        "PEAD Only (Basic)": {
            "filter": lambda g: g[g["sue"] > g["sue"].median()],
            "color": "#1f77b4",
        },
        "PEAD + Top 25% SUE": {
            "filter": lambda g: g[g["sue"] > g["sue"].quantile(0.75)],
            "color": "#ff7f0e",
        },
        "PEAD + Vol Filter": {
            "filter": lambda g: g[(g["sue"] > g["sue"].median()) & (g["realized_vol"] < g["realized_vol"].median())],
            "color": "#2ca02c",
        },
        "PEAD + Top 25% + Vol": {
            "filter": lambda g: g[(g["sue"] > g["sue"].quantile(0.75)) & (g["realized_vol"] < g["realized_vol"].median())],
            "color": "#d62728",
        },
    }

    results = {}

    for strategy_name, strategy_config in strategies.items():
        print(f"  {strategy_name}:")
        filter_func = strategy_config["filter"]
        quarterly_returns = []
        n_positions_list = []

        for season, group in panel.groupby("season"):
            filtered = filter_func(group)

            if len(filtered) == 0:
                continue

            avg_ret = filtered["ret_t21"].mean() / 100
            quarterly_returns.append(avg_ret)
            n_positions_list.append(len(filtered))

        if quarterly_returns:
            # Compute equity curve
            cumulative = np.cumprod(1 + np.array(quarterly_returns)) - 1
            equity = 100_000 * (1 + cumulative)

            annual_ret = (1 + np.mean(quarterly_returns)) ** 4 - 1
            sharpe = np.sqrt(4) * (np.mean(quarterly_returns) / np.std(quarterly_returns)) if np.std(quarterly_returns) > 0 else 0

            results[strategy_name] = {
                "equity": equity,
                "quarterly_returns": quarterly_returns,
                "annual_return": annual_ret,
                "sharpe": sharpe,
                "color": strategy_config["color"],
                "avg_positions": np.mean(n_positions_list),
            }

            print(f"    Annual return: {annual_ret*100:+.2f}%")
            print(f"    Sharpe: {sharpe:.2f}")
            print(f"    Avg positions: {np.mean(n_positions_list):.0f}")

    # Fetch SPY benchmark
    print("\n[5] Fetching SPY benchmark...")
    earliest_date = pd.to_datetime(panel["announcement_date"].min(), utc=True)
    latest_date = pd.to_datetime(panel["announcement_date"].max(), utc=True)

    spy = yf.download("SPY", start=earliest_date, end=latest_date, progress=False)
    spy_daily = spy["Close"].pct_change().dropna()

    # Convert to quarterly returns to match backtest
    spy_daily.index = pd.to_datetime(spy_daily.index)
    spy_quarterly = spy_daily.resample("Q").apply(lambda x: (1 + x).prod() - 1)

    spy_cumulative = np.cumprod(1 + spy_quarterly.values) - 1
    spy_equity = 100_000 * (1 + spy_cumulative)

    spy_annual = (1 + np.mean(spy_quarterly.values)) ** 4 - 1
    spy_sharpe = np.sqrt(4) * (np.mean(spy_quarterly.values) / np.std(spy_quarterly.values)) if np.std(spy_quarterly.values) > 0 else 0

    results["Buy & Hold (SPY)"] = {
        "equity": spy_equity,
        "quarterly_returns": spy_quarterly.values,
        "annual_return": spy_annual,
        "sharpe": spy_sharpe,
        "color": "#000000",
        "avg_positions": np.nan,
    }

    print(f"  SPY Annual return: {spy_annual*100:+.2f}%")
    print(f"  SPY Sharpe: {spy_sharpe:.2f}")

    # Visualization
    print("\n[6] Creating visualizations...")

    # Main comparison chart
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

    # Chart 1: Equity curves (log scale)
    for strategy_name, metrics in results.items():
        if strategy_name != "Buy & Hold (SPY)":
            ax1.semilogy(metrics["equity"], linewidth=2.5, label=strategy_name, color=metrics["color"], alpha=0.8)

    ax1.semilogy(results["Buy & Hold (SPY)"]["equity"], linewidth=3, label="Buy & Hold (SPY)",
                 color=results["Buy & Hold (SPY)"]["color"], linestyle="--", alpha=0.8)

    ax1.set_ylabel("Portfolio Value ($, log scale)", fontweight="bold", fontsize=12)
    ax1.set_title("PEAD Strategy Comparison: Equity Curves (All Strategies vs Market)", fontweight="bold", fontsize=14)
    ax1.legend(fontsize=11, loc="best")
    ax1.grid(True, alpha=0.3, which="both")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x/1e3:.0f}k"))

    # Chart 2: Performance metrics comparison
    strategy_names = list(results.keys())
    annual_returns = [results[s]["annual_return"] * 100 for s in strategy_names]
    sharpes = [results[s]["sharpe"] for s in strategy_names]

    x = np.arange(len(strategy_names))
    width = 0.35

    bars1 = ax2.bar(x - width / 2, annual_returns, width, label="Annual Return", color="#2ca02c", alpha=0.8)
    ax2_twin = ax2.twinx()
    bars2 = ax2_twin.bar(x + width / 2, sharpes, width, label="Sharpe Ratio", color="#d62728", alpha=0.8)

    ax2.set_xlabel("Strategy", fontweight="bold", fontsize=12)
    ax2.set_ylabel("Annual Return (%)", fontweight="bold", fontsize=11, color="#2ca02c")
    ax2_twin.set_ylabel("Sharpe Ratio", fontweight="bold", fontsize=11, color="#d62728")
    ax2.set_title("Performance Metrics by Strategy", fontweight="bold", fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategy_names, rotation=45, ha="right")
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.axhline(0, color="black", linestyle="-", linewidth=0.5)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height, f"{height:.1f}%",
                ha="center", va="bottom" if height >= 0 else "top", fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        ax2_twin.text(bar.get_x() + bar.get_width() / 2., height, f"{height:.2f}",
                     ha="center", va="bottom" if height >= 0 else "top", fontsize=9)

    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="upper left")

    plt.tight_layout()
    output_path = Path("results/pead/strategy_comparison_vs_spy.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")

    # Separate detailed comparison chart
    fig, ax = plt.subplots(figsize=(16, 8))

    for strategy_name, metrics in results.items():
        if strategy_name == "Buy & Hold (SPY)":
            ax.plot(metrics["equity"], linewidth=3.5, label=strategy_name, color=metrics["color"],
                   linestyle="--", alpha=0.9, zorder=10)
        else:
            ax.plot(metrics["equity"], linewidth=2.5, label=strategy_name, color=metrics["color"], alpha=0.8)

    ax.set_xlabel("Quarter", fontweight="bold", fontsize=12)
    ax.set_ylabel("Portfolio Value ($)", fontweight="bold", fontsize=12)
    ax.set_title("PEAD Strategy Comparison: Linear Scale (Better for Final Values)", fontweight="bold", fontsize=14)
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x/1e3:.0f}k"))
    ax.fill_between(range(len(results["Buy & Hold (SPY)"]["equity"])), 0, 1e10, alpha=0.02, color="gray")

    plt.tight_layout()
    output_path = Path("results/pead/strategy_comparison_linear.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")

    # Summary table
    print("\n" + "=" * 80)
    print("STRATEGY PERFORMANCE SUMMARY")
    print("=" * 80)

    summary_rows = []
    for strategy_name in strategy_names:
        metrics = results[strategy_name]
        final_value = metrics["equity"][-1]
        total_return = (final_value - 100_000) / 100_000
        summary_rows.append({
            "Strategy": strategy_name,
            "Final Value": f"${final_value:,.0f}",
            "Total Return": f"{total_return*100:+.1f}%",
            "Annual Return": f"{metrics['annual_return']*100:+.2f}%",
            "Sharpe Ratio": f"{metrics['sharpe']:.2f}",
            "Avg Positions": f"{metrics['avg_positions']:.0f}" if not np.isnan(metrics['avg_positions']) else "N/A",
        })

    summary_df = pd.DataFrame(summary_rows)
    print()
    print(summary_df.to_string(index=False))

    # Outperformance analysis
    print("\n" + "=" * 80)
    print("OUTPERFORMANCE vs BUY & HOLD")
    print("=" * 80)

    spy_annual = results["Buy & Hold (SPY)"]["annual_return"]
    spy_sharpe = results["Buy & Hold (SPY)"]["sharpe"]

    print()
    for strategy_name in strategy_names:
        if strategy_name != "Buy & Hold (SPY)":
            metrics = results[strategy_name]
            ret_outperformance = metrics["annual_return"] - spy_annual
            sharpe_outperformance = metrics["sharpe"] - spy_sharpe

            print(f"{strategy_name}:")
            print(f"  Return outperformance: {ret_outperformance*100:+.2f}% annual")
            print(f"  Sharpe outperformance: {sharpe_outperformance:+.2f}")
            print()

    # Save CSV summary
    csv_path = Path("results/pead/strategy_comparison.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    print("=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()
