"""Volatility & sharp-move features on Polymarket probability series.

All features are computed in **log-odds** space (see ``signals.to_logodds``) so a
move's magnitude is comparable across price levels, and all are **causal** —
the value at date *t* uses only data up to *t*.

The motivating question: when a market's implied probability takes a *sharp
upside move*, is that move informative (the market is right, it keeps going / it
resolves Yes) or is it overreaction (it reverts)? To answer it we need to
separate the *size and direction* of recent moves from the *level*:

  realized_vol  — RMS of daily log-odds moves (total churn)
  upside_vol    — RMS of the positive part only (upside semivol)
  downside_vol  — RMS of the negative part only
  vol_skew      — upside_vol − downside_vol  (is the churn tilted up or down?)
  sharp_up      — largest single-day positive log-odds jump in the window
                  (the literal "sharp vol return")
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .signals import to_logodds


def logodds_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """Daily change in log-odds — the per-market 'return' we measure vol on."""
    return to_logodds(panel).diff()


def _semivol(dz: pd.DataFrame, window: int, *, side: str) -> pd.DataFrame:
    """Rolling RMS of the positive ('up') or negative ('down') part of moves."""
    part = dz.clip(lower=0.0) if side == "up" else dz.clip(upper=0.0)
    mp = max(3, window // 2)
    return np.sqrt(part.pow(2).rolling(window, min_periods=mp).mean())


def realized_vol(panel: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Rolling RMS of daily log-odds moves (total volatility)."""
    dz = logodds_returns(panel)
    mp = max(3, window // 2)
    return np.sqrt(dz.pow(2).rolling(window, min_periods=mp).mean())


def upside_vol(panel: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Rolling upside semivol — RMS of the positive log-odds moves only."""
    return _semivol(logodds_returns(panel), window, side="up")


def downside_vol(panel: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Rolling downside semivol — RMS of the negative log-odds moves only."""
    return _semivol(logodds_returns(panel), window, side="down")


def vol_skew(panel: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """upside_vol − downside_vol: positive ⇒ recent churn tilts upward."""
    dz = logodds_returns(panel)
    return _semivol(dz, window, side="up") - _semivol(dz, window, side="down")


def sharp_up(panel: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Largest single-day positive log-odds jump within the trailing window."""
    dz = logodds_returns(panel)
    mp = max(2, window // 2)
    return dz.clip(lower=0.0).rolling(window, min_periods=mp).max()
