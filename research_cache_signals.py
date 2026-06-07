#!/usr/bin/env python3
"""Cache computed signals (momentum, volatility) to signal panel to avoid recalculation."""

from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")


def compute_momentum(ticker: str, end_date: pd.Timestamp, lookback_days: int = 20) -> float:
    """20-day momentum: (price_today - price_20d_ago) / price_20d_ago."""
    try:
        start = end_date - pd.Timedelta(days=lookback_days + 10)
        hist = yf.Ticker(ticker).history(start=start, end=end_date, auto_adjust=True)

        if len(hist) < lookback_days:
            return np.nan

        prices = hist["Close"].values
        if len(prices) < 2:
            return np.nan

        price_today = prices[-1]
        price_lookback = prices[-(lookback_days + 1)] if len(prices) > lookback_days else prices[0]
        momentum = (price_today - price_lookback) / price_lookback
        return momentum

    except Exception:
        return np.nan


def compute_realized_volatility(ticker: str, end_date: pd.Timestamp, lookback_days: int = 20) -> float:
    """20-day realized volatility: annualized std of daily log returns."""
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
    print("Cache Computed Signals: Add momentum & vol to signal panel")
    print("=" * 80)

    panel_path = Path("results/pead/signal_panel.csv")
    if not panel_path.exists():
        print("ERROR: Signal panel not found.")
        return

    print("\n[1] Loading signal panel...")
    panel = pd.read_csv(panel_path)
    panel["announcement_date"] = pd.to_datetime(panel["announcement_date"], utc=True, errors="coerce")
    print(f"  Loaded {len(panel)} announcements")

    # Check if already cached
    if "momentum" in panel.columns and "realized_vol" in panel.columns:
        existing_momentum = panel["momentum"].notna().sum()
        existing_vol = panel["realized_vol"].notna().sum()
        print(f"\n  Already cached:")
        print(f"    Momentum: {existing_momentum}/{len(panel)}")
        print(f"    Realized Vol: {existing_vol}/{len(panel)}")

        if existing_momentum > 0.9 * len(panel) and existing_vol > 0.9 * len(panel):
            print(f"  Cache is >90% complete. Skipping recomputation.")
            return

    # Add momentum if not present or incomplete
    print("\n[2] Computing momentum (20-day price return)...")
    if "momentum" not in panel.columns:
        panel["momentum"] = np.nan

    missing_momentum = panel["momentum"].isna().sum()
    if missing_momentum > 0:
        print(f"  Computing {missing_momentum} missing momentum values...")
        for idx, row in panel.iterrows():
            if pd.isna(row["momentum"]):
                ticker = row["ticker"]
                ann_date = row["announcement_date"]
                mom = compute_momentum(ticker, ann_date, lookback_days=20)
                panel.at[idx, "momentum"] = mom

                if (idx + 1) % 2000 == 0:
                    print(f"    {idx + 1}/{len(panel)}")

    # Add realized vol if not present or incomplete
    print("\n[3] Computing realized volatility (20-day rolling std)...")
    if "realized_vol" not in panel.columns:
        panel["realized_vol"] = np.nan

    missing_vol = panel["realized_vol"].isna().sum()
    if missing_vol > 0:
        print(f"  Computing {missing_vol} missing vol values...")
        for idx, row in panel.iterrows():
            if pd.isna(row["realized_vol"]):
                ticker = row["ticker"]
                ann_date = row["announcement_date"]
                vol = compute_realized_volatility(ticker, ann_date, lookback_days=20)
                panel.at[idx, "realized_vol"] = vol

                if (idx + 1) % 2000 == 0:
                    print(f"    {idx + 1}/{len(panel)}")

    # Summary
    print("\n[4] Cache summary...")
    print(f"  Momentum values: {panel['momentum'].notna().sum()}/{len(panel)} ({100*panel['momentum'].notna().sum()/len(panel):.1f}%)")
    print(f"  Realized vol values: {panel['realized_vol'].notna().sum()}/{len(panel)} ({100*panel['realized_vol'].notna().sum()/len(panel):.1f}%)")

    # Save back to CSV
    print("\n[5] Saving enhanced signal panel...")
    panel.to_csv(panel_path, index=False)
    print(f"  Saved: {panel_path}")

    print("\n[6] Columns in panel:")
    print(f"  {', '.join(panel.columns.tolist())}")

    print("\n" + "=" * 80)
    print("Done. Signals cached. Future runs will use these cached values.")
    print("=" * 80)


if __name__ == "__main__":
    main()
