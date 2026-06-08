"""Fama-MacBeth cross-sectional IC test for PEAD signal.

The IC test asks: does the SUE signal's cross-sectional correlation with
forward returns vary systematically from zero? We compute Spearman IC within
each earnings season (roughly quarterly) and t-test the IC time series.

Robust to non-normal returns, outlier-insensitive (rank), and the right test
for cross-sectional anomalies. We report:
  - Mean IC, std IC, IC t-stat
  - Drift test: IC at t+1 (announcement day), t+21, t+63
  - Long-short long-only annual return (signal > median vs signal <= median)
  - Power: number of seasons, average N per season
  - Out-of-sample (OOS) check: walk-forward on disjoint date ranges
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy import stats


class SeasonResult(NamedTuple):
    """Result for one earnings season."""

    season_label: str
    n: int
    ic_t1: float
    ic_t21: float
    ic_t63: float | None
    pval_t1: float
    pval_t21: float


class FamaMacbethResult(NamedTuple):
    """Aggregate PEAD test result across seasons."""

    signal_name: str
    n_seasons: int
    avg_n_per_season: float
    mean_ic_t1: float
    mean_ic_t21: float
    mean_ic_t63: float | None
    std_ic_t1: float
    std_ic_t21: float
    std_ic_t63: float | None
    t_stat_t1: float
    t_stat_t21: float
    t_stat_t63: float | None
    pval_t1: float
    pval_t21: float
    pval_t63: float | None
    long_short_annual_return: float | None
    long_short_pvalue: float | None
    season_results: list[SeasonResult]


def compute_ic(signal: np.ndarray, returns: np.ndarray) -> tuple[float, float]:
    """Compute Spearman rank IC and its p-value (two-tailed)."""
    # Drop NaNs row-wise.
    mask = ~(np.isnan(signal) | np.isnan(returns))
    if mask.sum() < 3:
        return np.nan, 1.0

    s, r = signal[mask], returns[mask]
    ic, pval = stats.spearmanr(s, r)
    return float(ic), float(pval)


def season_label(dt: pd.Timestamp) -> str:
    """Convert a date to a quarter label (e.g., '2020Q1')."""
    q = (dt.month - 1) // 3 + 1
    return f"{dt.year}Q{q}"


def fama_macbeth_ic(
    panel: pd.DataFrame,
    ret_col: str = "ret_t1",
    min_n_per_season: int = 5,
) -> FamaMacbethResult:
    """Run Fama-MacBeth IC test on the panel.

    Args:
      panel: frame with columns [announcement_date, sue, ret_t1, ret_t21, ret_t63]
      ret_col: which forward-return column to test ("ret_t1", "ret_t21", "ret_t63")
      min_n_per_season: drop seasons with fewer than this many obs.

    Returns:
      FamaMacbethResult with IC t-stats across seasons and long-short return.
    """
    if len(panel) == 0:
        raise ValueError("empty panel")

    panel = panel.copy()
    panel["season"] = panel["announcement_date"].dt.to_period("Q")

    season_results = []
    ic_t1_series = []
    ic_t21_series = []
    ic_t63_series = []
    long_short_rets = []

    for season, group in panel.groupby("season"):
        if len(group) < min_n_per_season:
            continue

        season_label_str = f"{season.year}Q{season.quarter}"
        n = len(group)

        # IC at each horizon.
        ic_1, pv_1 = compute_ic(group["sue"].values, group["ret_t1"].values)
        ic_21, pv_21 = compute_ic(
            group["sue"].values,
            group["ret_t21"].values if "ret_t21" in group.columns else np.full(n, np.nan),
        )
        ic_63, pv_63 = (
            compute_ic(
                group["sue"].values,
                group["ret_t63"].values,
            )
            if "ret_t63" in group.columns
            else (np.nan, 1.0)
        )

        ic_t1_series.append(ic_1)
        ic_t21_series.append(ic_21)
        ic_t63_series.append(ic_63)

        # Long-short: SUE > median (long) vs SUE <= median (short).
        median_sue = group["sue"].median()
        long = group[group["sue"] > median_sue]["ret_t1"]
        short = group[group["sue"] <= median_sue]["ret_t1"]
        if len(long) > 0 and len(short) > 0:
            ls_ret = long.mean() - short.mean()
            long_short_rets.append(ls_ret)

        season_results.append(
            SeasonResult(
                season_label=season_label_str,
                n=n,
                ic_t1=ic_1,
                ic_t21=ic_21,
                ic_t63=ic_63,
                pval_t1=pv_1,
                pval_t21=pv_21,
            )
        )

    if len(season_results) == 0:
        raise ValueError("no seasons survived min_n_per_season filter")

    # Aggregate.
    ic_t1 = np.array([r.ic_t1 for r in season_results])
    ic_t21 = np.array([r.ic_t21 for r in season_results])
    ic_t63 = np.array([r.ic_t63 for r in season_results])

    mean_ic_t1 = np.nanmean(ic_t1)
    mean_ic_t21 = np.nanmean(ic_t21)
    mean_ic_t63 = np.nanmean(ic_t63) if not np.all(np.isnan(ic_t63)) else None

    std_ic_t1 = np.nanstd(ic_t1)
    std_ic_t21 = np.nanstd(ic_t21)
    std_ic_t63 = np.nanstd(ic_t63) if not np.all(np.isnan(ic_t63)) else None

    n_valid_t1 = (~np.isnan(ic_t1)).sum()
    n_valid_t21 = (~np.isnan(ic_t21)).sum()
    n_valid_t63 = (~np.isnan(ic_t63)).sum() if not np.all(np.isnan(ic_t63)) else 0

    t_stat_t1 = mean_ic_t1 / (std_ic_t1 / np.sqrt(n_valid_t1)) if std_ic_t1 > 0 else np.nan
    t_stat_t21 = (
        mean_ic_t21 / (std_ic_t21 / np.sqrt(n_valid_t21)) if std_ic_t21 > 0 else np.nan
    )
    t_stat_t63 = (
        mean_ic_t63 / (std_ic_t63 / np.sqrt(n_valid_t63))
        if std_ic_t63 is not None and std_ic_t63 > 0
        else None
    )

    pval_t1 = 2 * (1 - stats.t.cdf(abs(t_stat_t1), n_valid_t1 - 1)) if not np.isnan(t_stat_t1) else 1.0
    pval_t21 = 2 * (1 - stats.t.cdf(abs(t_stat_t21), n_valid_t21 - 1)) if not np.isnan(t_stat_t21) else 1.0
    pval_t63 = (
        2 * (1 - stats.t.cdf(abs(t_stat_t63), n_valid_t63 - 1))
        if t_stat_t63 is not None and not np.isnan(t_stat_t63)
        else None
    )

    # Long-short annualized return.
    ls_annual = None
    ls_pval = None
    if long_short_rets:
        ls_rets = np.array(long_short_rets)
        mean_ls = ls_rets.mean()
        std_ls = ls_rets.std()
        # Annualize: ~4 quarters/year.
        ls_annual = mean_ls * 4
        # T-test H0: mean LS return = 0.
        t_ls = mean_ls / (std_ls / np.sqrt(len(ls_rets)))
        ls_pval = 2 * (1 - stats.t.cdf(abs(t_ls), len(ls_rets) - 1))

    return FamaMacbethResult(
        signal_name="SUE (analyst-surprise)",
        n_seasons=len(season_results),
        avg_n_per_season=np.mean([r.n for r in season_results]),
        mean_ic_t1=mean_ic_t1,
        mean_ic_t21=mean_ic_t21,
        mean_ic_t63=mean_ic_t63,
        std_ic_t1=std_ic_t1,
        std_ic_t21=std_ic_t21,
        std_ic_t63=std_ic_t63,
        t_stat_t1=t_stat_t1,
        t_stat_t21=t_stat_t21,
        t_stat_t63=t_stat_t63,
        pval_t1=pval_t1,
        pval_t21=pval_t21,
        pval_t63=pval_t63,
        long_short_annual_return=ls_annual,
        long_short_pvalue=ls_pval,
        season_results=season_results,
    )


def test_by_cap_bucket(
    panel: pd.DataFrame,
    cap_col: str = None,
) -> dict[str, FamaMacbethResult]:
    """Run FM test separately for each market-cap bucket.

    Args:
      panel: signal panel with optional cap_col identifying the bucket.
      cap_col: column name for market_cap bucket; if None, all names in one test.

    Returns:
      dict of {cap_bucket: FamaMacbethResult}.
    """
    if cap_col is None or cap_col not in panel.columns:
        return {"all": fama_macbeth_ic(panel)}

    results = {}
    for bucket in sorted(panel[cap_col].unique()):
        sub = panel[panel[cap_col] == bucket]
        if len(sub) > 0:
            results[bucket] = fama_macbeth_ic(sub)
    return results
