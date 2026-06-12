"""
Robustness toolkit — statistical hygiene from the knowledge base.

1. Deflated Sharpe Ratio (Bailey & López de Prado 2014)
   Corrects an observed Sharpe for the number of strategy variants tried,
   the variance of Sharpe across those trials, sample length, skewness and
   kurtosis. Output is the probability that the true Sharpe exceeds zero
   after accounting for selection bias. The BOCPD-AMR v1→v4 family is
   exactly the multiple-trial situation this statistic exists for.

2. Rebalance-offset dispersion (Newfound "rebalance timing luck")
   Re-run a weekly strategy anchored on each weekday and report the
   dispersion of terminal wealth / Sharpe. If strategy rankings change
   with the anchor, the measured edge is timing luck, not signal.
"""
import logging
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

logger = logging.getLogger(__name__)

_ANN = 252
_EULER_GAMMA = 0.5772156649015329


# ---------------------------------------------------------------------------
# Deflated Sharpe Ratio
# ---------------------------------------------------------------------------

def expected_max_sharpe(sr_var_trials: float, n_trials: int) -> float:
    """
    E[max SR] under H0 (true SR = 0) across n_trials, given the variance of
    estimated Sharpe across trials. Bailey & López de Prado (2014), eq. for
    SR0 using the Euler–Mascheroni approximation to the expected maximum of
    n standard normals.

    Inputs are in PER-PERIOD units (e.g. daily Sharpe and its variance).
    """
    if n_trials <= 1 or sr_var_trials <= 0:
        return 0.0
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(sr_var_trials) * ((1.0 - _EULER_GAMMA) * z1 + _EULER_GAMMA * z2))


def probabilistic_sharpe(sr_obs: float, sr_benchmark: float, n_obs: int,
                         skew: float, kurt: float) -> float:
    """
    PSR(SR*) = P(true SR > SR*), with non-normality adjustment.

    sr_obs / sr_benchmark in per-period units; kurt is the FULL kurtosis
    (normal = 3), not excess.
    """
    denom = np.sqrt(max(1e-12,
                        1.0 - skew * sr_obs + (kurt - 1.0) / 4.0 * sr_obs ** 2))
    z = (sr_obs - sr_benchmark) * np.sqrt(max(n_obs - 1, 1)) / denom
    return float(norm.cdf(z))


def deflated_sharpe_ratio(
    returns: pd.Series,
    trial_sharpes_daily: Iterable[float],
) -> Dict[str, float]:
    """
    DSR for one strategy given the per-period (daily) Sharpe ratios of ALL
    variants tried during the research process (including this one).

    Returns dict with observed annualized Sharpe, the deflation benchmark
    SR0 (annualized), and DSR = P(true SR > 0 | n_trials).
    DSR > 0.95 is the conventional pass threshold.
    """
    r = returns.dropna()
    n = len(r)
    if n < 60:
        return {"sharpe_ann": np.nan, "sr0_ann": np.nan, "dsr": np.nan}

    sr_daily = float(r.mean() / (r.std() + 1e-12))
    skew = float(r.skew())
    kurt = float(r.kurtosis()) + 3.0   # pandas gives excess kurtosis

    trials = np.asarray(list(trial_sharpes_daily), dtype=float)
    n_trials = len(trials)
    sr_var = float(np.var(trials, ddof=1)) if n_trials > 1 else 0.0
    sr0 = expected_max_sharpe(sr_var, n_trials)

    dsr = probabilistic_sharpe(sr_daily, sr0, n, skew, kurt)
    return {
        "sharpe_ann": sr_daily * np.sqrt(_ANN),
        "sr0_ann": sr0 * np.sqrt(_ANN),
        "n_trials": n_trials,
        "dsr": dsr,
    }


def dsr_report(returns_by_strategy: Dict[str, pd.Series]) -> pd.DataFrame:
    """
    Treat every strategy in the dict as one trial of the same research
    process and compute each one's DSR against the family.
    """
    daily_srs = {k: float(v.dropna().mean() / (v.dropna().std() + 1e-12))
                 for k, v in returns_by_strategy.items()}
    rows = []
    for name, rets in returns_by_strategy.items():
        stats = deflated_sharpe_ratio(rets, daily_srs.values())
        rows.append({"strategy": name, **stats})
    return pd.DataFrame(rows).sort_values("dsr", ascending=False)


# ---------------------------------------------------------------------------
# Rebalance-offset dispersion (timing luck)
# ---------------------------------------------------------------------------

WEEK_ANCHORS = ("W-MON", "W-TUE", "W-WED", "W-THU", "W-FRI")


def rebalance_offset_dispersion(
    backtest_fn: Callable[..., pd.Series],
    anchors: Iterable[str] = WEEK_ANCHORS,
    **kwargs,
) -> pd.DataFrame:
    """
    Run `backtest_fn(rebal_anchor=a, **kwargs)` for each weekly anchor and
    summarize the dispersion. backtest_fn must return a daily return Series.

    Report: per-anchor CAGR / Sharpe / MaxDD plus the cross-anchor range.
    A CAGR range of several % per year on the same signal = timing luck
    dominates; tranche the strategy across anchors before trusting it.
    """
    from src.metrics import compute_metrics

    rows = []
    for a in anchors:
        try:
            rets = backtest_fn(rebal_anchor=a, **kwargs)
            m = compute_metrics(rets)
            rows.append({"anchor": a, "CAGR": m["CAGR"],
                         "Sharpe": m["Sharpe"], "MaxDD": m["Max DD"]})
        except Exception as exc:
            logger.error(f"anchor {a} failed: {exc}")
    df = pd.DataFrame(rows)
    if len(df) > 1:
        spread = {"anchor": "RANGE",
                  "CAGR": df["CAGR"].max() - df["CAGR"].min(),
                  "Sharpe": df["Sharpe"].max() - df["Sharpe"].min(),
                  "MaxDD": df["MaxDD"].max() - df["MaxDD"].min()}
        df = pd.concat([df, pd.DataFrame([spread])], ignore_index=True)
    return df
