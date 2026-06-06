#!/usr/bin/env python3
"""Backtest PEAD strategy vs SPY buy-and-hold.

Run the long-short PEAD strategy from 1999–2026 and compare to SPY benchmark.
Outputs equity curves and performance metrics.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from src.pead import backtest


def main():
    print("=" * 80)
    print("PEAD Strategy Backtest vs SPY Buy-and-Hold")
    print("=" * 80)

    # --- Load signal panel ---
    panel_path = Path("results/pead/signal_panel.csv")
    if not panel_path.exists():
        print("ERROR: Signal panel not found. Run `python3 run_pead.py` first.")
        return

    print("\n[1] Loading signal panel...")
    panel = pd.read_csv(panel_path)
    panel["announcement_date"] = pd.to_datetime(panel["announcement_date"], utc=True, errors="coerce")
    panel = panel.dropna(subset=["sue", "ret_t21", "announcement_date"])
    print(f"  Loaded {len(panel)} announcements from {panel['announcement_date'].min().date()} "
          f"to {panel['announcement_date'].max().date()}")

    # --- Run PEAD backtest ---
    print("\n[2] Running PEAD long-short backtest (20 bps transaction cost)...")
    try:
        result = backtest.backtest_pead_simple(panel, transaction_cost_bps=20)
        print(f"  Cumulative return: {result.cum_return:+.2%}")
        print(f"  Annual return: {result.annual_return:+.2%}")
        print(f"  Sharpe ratio: {result.sharpe_ratio:.2f}")
        print(f"  Max drawdown: {result.max_drawdown:.2%}")
        print(f"  Win rate: {result.win_rate:.1%}")
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    # --- Fetch SPY benchmark ---
    print("\n[3] Fetching SPY benchmark...")
    spy_start = panel["announcement_date"].min()
    spy_end = panel["announcement_date"].max()
    try:
        spy_prices = backtest.fetch_spy_benchmark(spy_start, spy_end)
        spy_equity = backtest.compute_benchmark_equity(spy_prices, initial_capital=100_000)

        # Remove timezone info
        spy_equity.index = spy_equity.index.tz_localize(None) if spy_equity.index.tz is not None else spy_equity.index
        print(f"  SPY data: {len(spy_prices)} days")

        # Metrics for SPY.
        spy_return = (spy_prices.iloc[-1] / spy_prices.iloc[0]) - 1
        spy_annual = (1 + spy_return) ** (1 / ((spy_end - spy_start).days / 365.25)) - 1
        print(f"  SPY cumulative return: {spy_return:+.2%}")
        print(f"  SPY annual return: {spy_annual:+.2%}")
    except Exception as e:
        print(f"  ERROR fetching SPY: {e}")
        return

    # --- Align dates for plotting ---
    print("\n[4] Preparing visualization...")
    # Resample PEAD curve to daily (forward-fill) for smoother comparison.
    pead_curve = result.equity_curve
    pead_daily = pead_curve.resample("D").ffill()

    # Remove timezone info for comparison
    pead_daily.index = pead_daily.index.tz_localize(None)

    # Align to common date range.
    common_start = max(pead_daily.index.min(), spy_equity.index.min())
    common_end = min(pead_daily.index.max(), spy_equity.index.max())

    pead_aligned = pead_daily[common_start:common_end]
    spy_aligned = spy_equity[common_start:common_end]

    print(f"  Common period: {common_start.date()} to {common_end.date()}")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(14, 8))

    ax.plot(pead_aligned.index, pead_aligned.values, label="PEAD Long-Short", linewidth=2.5, color="#1f77b4")
    ax.plot(spy_aligned.index, spy_aligned.values, label="SPY Buy-and-Hold", linewidth=2.5, color="#ff7f0e")

    ax.set_xlabel("Date", fontsize=12, fontweight="bold")
    ax.set_ylabel("Portfolio Value ($)", fontsize=12, fontweight="bold")
    ax.set_title("PEAD Strategy vs SPY Buy-and-Hold (1999–2026)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x/1e3:.0f}k"))

    # Add shaded regions for major market events.
    events = [
        ("2000–02 Dot-com", "2000-01-01", "2002-10-01"),
        ("2008 GFC", "2007-10-01", "2009-03-01"),
        ("2020 COVID", "2020-02-01", "2020-04-01"),
    ]
    for label, start, end in events:
        ax.axvspan(pd.to_datetime(start), pd.to_datetime(end), alpha=0.1, color="red")

    plt.tight_layout()
    output_path = Path("results/pead/backtest_equity_curve.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_path}")

    # --- Summary table ---
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)

    pead_final = pead_aligned.iloc[-1]
    spy_final = spy_aligned.iloc[-1]

    summary_data = {
        "Metric": [
            "Cumulative Return",
            "Annual Return",
            "Sharpe Ratio",
            "Max Drawdown",
            "Win Rate",
            "Final Value",
        ],
        "PEAD Long-Short": [
            f"{result.cum_return:+.2%}",
            f"{result.annual_return:+.2%}",
            f"{result.sharpe_ratio:.2f}",
            f"{result.max_drawdown:.2%}",
            f"{result.win_rate:.1%}",
            f"${pead_final:,.0f}",
        ],
        "SPY Buy-Hold": [
            f"{spy_return:+.2%}",
            f"{spy_annual:+.2%}",
            f"N/A",
            f"N/A",
            f"N/A",
            f"${spy_final:,.0f}",
        ],
    }

    summary_df = pd.DataFrame(summary_data)
    print()
    print(summary_df.to_string(index=False))

    # --- Excess return ---
    print()
    excess_return = result.cum_return - spy_return
    excess_annual = result.annual_return - spy_annual
    print(f"\nExcess Return (PEAD - SPY):")
    print(f"  Cumulative: {excess_return:+.2%}")
    print(f"  Annual: {excess_annual:+.2%}")

    # --- Caveats ---
    print("\n" + "=" * 80)
    print("IMPORTANT CAVEATS")
    print("=" * 80)
    print("""
1. SURVIVORSHIP BIAS: Panel includes only live names; dead/bankrupt names missing
   → long-short return is UPWARD-BIASED

2. TRANSACTION COSTS: This backtest assumes 20 bps round-trip, but reality includes:
   - Small-cap bid-ask spreads (0.2–0.5%)
   - Short-borrow costs (1.5–3% annual)
   - Market impact (depends on capital)
   → Net returns are likely 2–4% lower

3. IN-SAMPLE RESULTS: Walk-forward validation not yet complete

4. CAPACITY LIMITS: Small-cap strategy faces liquidity constraints; scaling to billions
   would require more sophisticated execution

5. REBALANCING: Simple strategy rebalances monthly; sophisticated implementations use
   gradual entry/exit to reduce market impact
""")

    print("=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()
