"""
Stage 4 — robustness checks beyond point metrics.

1. Rebalance-offset dispersion (Newfound "rebalance timing luck"):
   re-run a weekly strategy anchored on each weekday and report the
   dispersion of CAGR/Sharpe/MaxDD. Two identical strategies rebalanced on
   different days can diverge by several % per year; if rankings change
   with the anchor, the measured edge is timing luck — tranche across
   anchors before trusting it.

2. Family-level Deflated Sharpe report: treat every strategy variant tried
   on the same sample as one trial of the same research process (the
   BOCPD-AMR v1→v4 situation) and score each variant's DSR against the
   family, via `mining.validation.deflated_sharpe_ratio`.
"""
import logging
from typing import Callable, Dict, Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

WEEK_ANCHORS = ("W-MON", "W-TUE", "W-WED", "W-THU", "W-FRI")


def rebalance_offset_dispersion(
    backtest_fn: Callable[..., pd.Series],
    anchors: Iterable[str] = WEEK_ANCHORS,
    **kwargs,
) -> pd.DataFrame:
    """
    Run `backtest_fn(rebal_anchor=a, **kwargs)` per weekly anchor; the
    function must return a daily return Series. Appends a RANGE row with
    the cross-anchor max−min of each metric.
    """
    from posterioralpha.validation.metrics import compute_metrics

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
        df = pd.concat([df, pd.DataFrame([{
            "anchor": "RANGE",
            "CAGR": df["CAGR"].max() - df["CAGR"].min(),
            "Sharpe": df["Sharpe"].max() - df["Sharpe"].min(),
            "MaxDD": df["MaxDD"].max() - df["MaxDD"].min(),
        }])], ignore_index=True)
    return df


def dsr_report(returns_by_strategy: Dict[str, pd.Series]) -> pd.DataFrame:
    """
    DSR for each variant in a research family, where every entry in the
    dict counts as one trial. DSR > 0.95 is the conventional pass bar.
    """
    from posterioralpha.mining.validation import deflated_sharpe_ratio

    daily_srs = []
    for v in returns_by_strategy.values():
        r = v.dropna()
        daily_srs.append(float(r.mean() / (r.std() + 1e-12)))
    n_trials = len(daily_srs)
    sr_var = float(np.var(daily_srs, ddof=1)) if n_trials > 1 else 0.0

    rows = []
    for name, rets in returns_by_strategy.items():
        r = rets.dropna()
        rows.append({
            "strategy": name,
            "sharpe_ann": float(r.mean() / (r.std() + 1e-12)) * np.sqrt(252),
            "n_trials": n_trials,
            "dsr": deflated_sharpe_ratio(r.values, n_trials, sr_var),
        })
    return pd.DataFrame(rows).sort_values("dsr", ascending=False)
