"""Walk-forward out-of-sample validation for PEAD signal.

Methodology:
  For each test window (e.g., 2017–2026):
    1. Train on all preceding data (e.g., 1999–2016)
    2. Compute FM IC on training data
    3. Apply signal to test data
    4. Measure OOS returns
    5. Compare to in-sample returns

A real signal should show:
  - OOS IC remains significantly positive (p < 0.05)
  - OOS returns don't collapse vs in-sample
  - Cap-spectrum pattern holds OOS
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from . import fama_macbeth


class WalkForwardResult(NamedTuple):
    """Results for one train/test split."""

    train_period: str
    test_period: str
    train_fm: fama_macbeth.FamaMacbethResult
    test_fm: fama_macbeth.FamaMacbethResult
    train_ic_t1: float
    test_ic_t1: float
    train_ls_annual: float
    test_ls_annual: float
    deterioration_ic: float  # train IC - test IC (positive = degradation)
    deterioration_ls: float  # train LS - test LS


def split_by_date(
    panel: pd.DataFrame,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split panel into train/test by date ranges."""
    panel = panel.copy()
    panel["announcement_date"] = pd.to_datetime(panel["announcement_date"], utc=True, errors="coerce")

    # Convert split dates to UTC-aware
    train_start_dt = pd.to_datetime(train_start, utc=True)
    train_end_dt = pd.to_datetime(train_end, utc=True)
    test_start_dt = pd.to_datetime(test_start, utc=True)
    test_end_dt = pd.to_datetime(test_end, utc=True)

    train = panel[
        (panel["announcement_date"] >= train_start_dt)
        & (panel["announcement_date"] <= train_end_dt)
    ]
    test = panel[
        (panel["announcement_date"] >= test_start_dt)
        & (panel["announcement_date"] <= test_end_dt)
    ]

    return train, test


def run_walk_forward(
    panel: pd.DataFrame,
    splits: list[dict],
) -> list[WalkForwardResult]:
    """Run walk-forward validation across multiple splits.

    Args:
      panel: signal panel with [announcement_date, sue, ret_t21, market_cap]
      splits: list of dicts with keys:
        - train_start, train_end (e.g., "1999-01-01", "2016-12-31")
        - test_start, test_end (e.g., "2017-01-01", "2026-12-31")

    Returns:
      List of WalkForwardResult objects.
    """
    results = []

    for split in splits:
        print(
            f"\n  Train: {split['train_start']} to {split['train_end']} | "
            f"Test: {split['test_start']} to {split['test_end']}"
        )

        train, test = split_by_date(
            panel,
            split["train_start"],
            split["train_end"],
            split["test_start"],
            split["test_end"],
        )

        if len(train) < 100 or len(test) < 100:
            print(f"    Skipped: insufficient data (train={len(train)}, test={len(test)})")
            continue

        try:
            # FM test on both periods.
            train_fm = fama_macbeth.fama_macbeth_ic(train)
            test_fm = fama_macbeth.fama_macbeth_ic(test)

            deterioration_ic = train_fm.mean_ic_t1 - test_fm.mean_ic_t1
            deterioration_ls = (train_fm.long_short_annual_return or 0) - (test_fm.long_short_annual_return or 0)

            result = WalkForwardResult(
                train_period=f"{split['train_start'][:4]}–{split['train_end'][:4]}",
                test_period=f"{split['test_start'][:4]}–{split['test_end'][:4]}",
                train_fm=train_fm,
                test_fm=test_fm,
                train_ic_t1=train_fm.mean_ic_t1,
                test_ic_t1=test_fm.mean_ic_t1,
                train_ls_annual=train_fm.long_short_annual_return or 0,
                test_ls_annual=test_fm.long_short_annual_return or 0,
                deterioration_ic=deterioration_ic,
                deterioration_ls=deterioration_ls,
            )

            results.append(result)

            print(
                f"    Train: IC={result.train_ic_t1:+.4f}, LS={result.train_ls_annual:+.2f}% | "
                f"Test: IC={result.test_ic_t1:+.4f}, LS={result.test_ls_annual:+.2f}%"
            )
            print(
                f"    Deterioration: IC {result.deterioration_ic:+.4f} "
                f"({result.deterioration_ic/result.train_ic_t1*100:+.1f}%), "
                f"LS {result.deterioration_ls:+.2f}%"
            )

        except Exception as e:
            print(f"    ERROR: {e}")
            continue

    return results


def summarize_walk_forward(results: list[WalkForwardResult]) -> pd.DataFrame:
    """Create summary table of walk-forward results."""
    rows = []
    for r in results:
        rows.append({
            "train_period": r.train_period,
            "test_period": r.test_period,
            "train_ic_t1": f"{r.train_ic_t1:+.4f}",
            "test_ic_t1": f"{r.test_ic_t1:+.4f}",
            "ic_deterioration": f"{r.deterioration_ic:+.4f}",
            "train_ls_annual": f"{r.train_ls_annual:+.2f}%",
            "test_ls_annual": f"{r.test_ls_annual:+.2f}%",
            "ls_deterioration": f"{r.deterioration_ls:+.2f}%",
        })
    return pd.DataFrame(rows)
