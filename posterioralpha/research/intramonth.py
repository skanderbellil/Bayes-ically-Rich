"""
Intramonth momentum primitives (stage 2: research).

Implements the building blocks for the Nathan, Suominen & Tasa (2026)
hypothesis that US momentum returns concentrate in a six-trading-day window
each month (T-9 to T-4 relative to month-end), attributed to an institutional
"dash for cash."

Two causal primitives:
  - ``intramonth_window_mask`` labels which days of a holding period fall in
    the T-start … T-end window (counted back from the last trading day).
  - ``wml_formation`` ranks a cross-section by 12-1 momentum into winners and
    losers using only data available at formation.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def intramonth_window_mask(
    holding_days: pd.DatetimeIndex, start: int = 9, end: int = 4
) -> np.ndarray:
    """
    Boolean mask, True on days T-``start`` … T-``end`` (inclusive), counted
    back from the last trading day of ``holding_days``.

    Mapping: T-k corresponds to ``holding_days[-(k+1)]``. With the defaults
    (start=9, end=4) this flags the six-day intramonth momentum window.
    """
    n = len(holding_days)
    mask = np.zeros(n, dtype=bool)
    for k in range(end, start + 1):
        pos = -(k + 1)
        if abs(pos) <= n:
            mask[pos] = True
    return mask


def wml_formation(
    history: pd.DataFrame,
    long_n: int = 2,
    short_n: int = 2,
    lookback: int = 252,
    skip: int = 21,
) -> Tuple[List[str], List[str]]:
    """
    Cross-sectional 12-1 momentum formation on a returns ``history`` frame
    (rows = days up to and including the formation date, columns = assets).

    Ranks assets by cumulative return over ``[-(lookback+skip) : -skip]`` —
    i.e. the trailing ``lookback`` days, skipping the most recent ``skip`` days
    to avoid short-term reversal. Returns ``(winners, losers)``: the top
    ``long_n`` and bottom ``short_n`` tickers. Empty lists if history is short.
    """
    if len(history) < lookback + skip:
        return [], []
    formation = history.iloc[-(lookback + skip):-skip]
    cum_ret = ((1.0 + formation).prod() - 1.0).dropna()
    if len(cum_ret) < long_n + short_n + 1:
        return [], []
    ranked = cum_ret.sort_values()
    losers = ranked.index[:short_n].tolist()
    winners = ranked.index[-long_n:].tolist()
    return winners, losers
