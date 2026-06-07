#!/usr/bin/env python3
"""Diagnose return calculation discrepancy between Bayesian and Dynamic Hold analyses."""

import pandas as pd
import numpy as np

def load_data():
    """Load PEAD signal panel, filter to mega-cap high-signal."""
    df = pd.read_csv("results/pead/signal_panel.csv", parse_dates=["announcement_date"])
    mega = df[df["market_cap"] == "Mega Cap"].dropna(
        subset=['sue', 'ret_t1', 'ret_t21', 'ret_t63']
    ).copy()

    sue_75 = mega['sue'].quantile(0.75)
    signal = mega[mega['sue'] >= sue_75].copy()
    signal['drift_21d'] = signal['ret_t21'] - signal['ret_t1']
    signal['drift_63d'] = signal['ret_t63'] - signal['ret_t1']
    signal['win_21d'] = (signal['drift_21d'] > 0).astype(int)
    signal['win_63d'] = (signal['drift_63d'] > 0).astype(int)
    signal['announce_quarter'] = signal['announcement_date'].dt.to_period('Q')

    return signal.sort_values('announcement_date')

def method1_bayesian_style(signal):
    """Calculate returns using previous Bayesian method (only winners)."""
    quarters = sorted(signal['announce_quarter'].unique())
    returns = []

    for qtr in quarters:
        trades_qtr = signal[signal['announce_quarter'] == qtr]
        if len(trades_qtr) < 3:
            continue

        winners = trades_qtr['win_21d'].sum()
        ret_winners = trades_qtr[trades_qtr['win_21d'] == 1]['drift_63d'].mean()

        if np.isnan(ret_winners):
            ret_winners = 0

        # Previous method: only count winners
        ret = (winners * ret_winners) / len(trades_qtr)
        returns.append(ret)

    return np.array(returns)

def method2_correct_exit_losers(signal):
    """Calculate returns using correct exit losers method (winners 63d + losers 21d)."""
    quarters = sorted(signal['announce_quarter'].unique())
    returns = []

    for qtr in quarters:
        trades_qtr = signal[signal['announce_quarter'] == qtr]
        if len(trades_qtr) < 3:
            continue

        winners = trades_qtr['win_21d'].sum()
        losers = len(trades_qtr) - winners

        ret_winners = trades_qtr[trades_qtr['win_21d'] == 1]['drift_63d'].mean()
        ret_losers = trades_qtr[trades_qtr['win_21d'] == 0]['drift_21d'].mean()

        if np.isnan(ret_winners):
            ret_winners = 0
        if np.isnan(ret_losers):
            ret_losers = 0

        # Correct method: include loser losses
        ret = (winners * ret_winners + losers * ret_losers) / len(trades_qtr)
        returns.append(ret)

    return np.array(returns)

def method3_rigid_63d(signal):
    """Calculate returns using rigid hold all 63d."""
    quarters = sorted(signal['announce_quarter'].unique())
    returns = []

    for qtr in quarters:
        trades_qtr = signal[signal['announce_quarter'] == qtr]
        if len(trades_qtr) < 3:
            continue

        ret = trades_qtr['drift_63d'].mean()
        returns.append(ret)

    return np.array(returns)

def compute_metrics(returns_pct, name=""):
    """Compute annualized return, Sharpe, max drawdown."""
    ret = returns_pct.dropna() / 100 if isinstance(returns_pct, pd.Series) else returns_pct / 100

    if len(ret) < 4:
        return None

    cum = np.prod(1 + ret)
    ann = cum ** (4 / len(ret)) - 1
    sharpe = np.mean(ret) / np.std(ret) * np.sqrt(4) if np.std(ret) > 0 else 0
    cummax = np.cumprod(1 + ret)
    dd = ((cummax - np.maximum.accumulate(cummax)) / np.maximum.accumulate(cummax)).min()

    return {
        'name': name,
        'cum': cum,
        'ann': ann,
        'sharpe': sharpe,
        'dd': dd,
        'n': len(ret),
    }

def main():
    signal = load_data()

    print("=" * 80)
    print("RETURN CALCULATION DIAGNOSTIC")
    print("=" * 80)
    print(f"\nData: {len(signal)} mega-cap earnings over {signal['announce_quarter'].nunique()} quarters\n")

    # Method 1: Previous Bayesian (only winners)
    ret1 = method1_bayesian_style(signal)
    m1 = compute_metrics(ret1, "Previous Bayesian (winners only)")

    print("METHOD 1: Previous Bayesian Calculation (only winners, ignore loser losses)")
    print(f"  Quarters: {m1['n']}")
    print(f"  Annual return: {m1['ann']:+.1%}")
    print(f"  Sharpe: {m1['sharpe']:.2f}")
    print(f"  Max DD: {m1['dd']:.0%}")
    print(f"  Cumulative: {m1['cum']:.0%}")

    # Method 2: Correct exit losers (winners 63d + losers 21d)
    ret2 = method2_correct_exit_losers(signal)
    m2 = compute_metrics(ret2, "Correct Exit Losers")

    print(f"\nMETHOD 2: Correct Exit Losers (winners 63d, losers 21d)")
    print(f"  Quarters: {m2['n']}")
    print(f"  Annual return: {m2['ann']:+.1%}")
    print(f"  Sharpe: {m2['sharpe']:.2f}")
    print(f"  Max DD: {m2['dd']:.0%}")
    print(f"  Cumulative: {m2['cum']:.0%}")

    # Method 3: Rigid 63d hold
    ret3 = method3_rigid_63d(signal)
    m3 = compute_metrics(ret3, "Rigid 63d Hold")

    print(f"\nMETHOD 3: Rigid Hold 63d (all positions to 63d)")
    print(f"  Quarters: {m3['n']}")
    print(f"  Annual return: {m3['ann']:+.1%}")
    print(f"  Sharpe: {m3['sharpe']:.2f}")
    print(f"  Max DD: {m3['dd']:.0%}")
    print(f"  Cumulative: {m3['cum']:.0%}")

    # Analysis
    print(f"\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    print(f"""
Previous Bayesian analysis showed +23.6% annually, which came from METHOD 1
(only counting winner returns, ignoring loser losses). This inflates returns
by ignoring portfolio losses from positions that didn't trigger entry signal.

The CORRECT calculation is METHOD 2 (exit losers at 21d), which accounts for:
  - Winners: hold to 63d, capture full drift_63d
  - Losers: exit at 21d, realize loss at drift_21d

This shows {m2['ann']:+.1%} annual return (vs {m1['ann']:+.1%} in previous analysis).

The difference between rigid 63d hold ({m3['ann']:+.1%}) and exit losers ({m2['ann']:+.1%})
is:
  - Return edge: {(m2['ann'] - m3['ann'])*100:+.2f}%
  - Risk reduction: Sharpe {m2['sharpe'] - m3['sharpe']:+.2f}

Next: Dynamic hold using logistic regression (Phase 3) should improve on
method 2 by better timing exit decisions.
""")

    # Quarter-by-quarter comparison
    print(f"=" * 80)
    print("QUARTER-BY-QUARTER BREAKDOWN (First 10 quarters)")
    print("=" * 80)
    print(f"\n{'Quarter':<10} {'Rigid 63d':>12} {'Exit Losers':>12} {'Difference':>12}")
    print("-" * 50)

    quarters = sorted(signal['announce_quarter'].unique())[:10]
    for qtr in quarters:
        trades_qtr = signal[signal['announce_quarter'] == qtr]
        if len(trades_qtr) < 3:
            continue

        ret_63d = trades_qtr['drift_63d'].mean()

        winners = trades_qtr['win_21d'].sum()
        losers = len(trades_qtr) - winners
        ret_winners = trades_qtr[trades_qtr['win_21d'] == 1]['drift_63d'].mean()
        ret_losers = trades_qtr[trades_qtr['win_21d'] == 0]['drift_21d'].mean()

        if np.isnan(ret_winners):
            ret_winners = 0
        if np.isnan(ret_losers):
            ret_losers = 0

        ret_exit_losers = (winners * ret_winners + losers * ret_losers) / len(trades_qtr)

        diff = ret_exit_losers - ret_63d

        print(f"{str(qtr):<10} {ret_63d:>11.2f}% {ret_exit_losers:>11.2f}% {diff:>11.2f}%")

if __name__ == "__main__":
    main()
