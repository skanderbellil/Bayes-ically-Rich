"""Signals, execution models and portfolio construction for the same-day dip buy.

All functions are pure (DataFrame in, DataFrame/Series out). Parameters are frozen in docs/dipbuy/DESIGN.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COST = 0.001              # retail round trip used in the first pass
THR_FLOOR, THR_SD = 0.03, 3.0
IS_END, OOS_START = "2021-12-31", "2022-01-01"


def eligible(p: pd.DataFrame) -> pd.Series:
    return (p.year >= 2015) & p.Q.notna() & (p.prev_close >= 5) & p.sd60.notna() & p.open.notna() & p.close.notna() \
        & (p.gap.abs() < 0.5) & ~p.split & ~p.exdiv_any


def signals(p: pd.DataFrame, side: str = "down", gap_col: str = "agap") -> pd.Series:
    thr = np.maximum(THR_FLOOR, THR_SD * p.sd60)
    s = p[gap_col] <= -thr if side == "down" else p[gap_col] >= thr
    return s & eligible(p)


def trade_returns(t: pd.DataFrame, side: str = "down", cost: float = COST) -> pd.DataFrame:
    """Per-trade net returns under each execution model. Long for gap-downs, short for gap-ups.
    E2 is NaN when the strict limit would not have filled."""
    sgn = 1.0 if side == "down" else -1.0
    r = pd.DataFrame(index=t.index)
    r["E1_auction"] = sgn * (t.close / t.open - 1) - cost
    filled = (t.cons_lo < t.open * (1 - 1e-6)) if side == "down" else (t.cons_hi > t.open * (1 + 1e-6))
    r["E2_strict_limit"] = r.E1_auction.where(filled)
    r["E3_other_venue_entry"] = sgn * (t.close / t.alt_open_med - 1) - cost
    for s in (0.005, 0.01):
        r[f"E5_slip_{s:.1%}"] = sgn * (t.close / (t.open * (1 + sgn * s)) - 1) - cost
    r["gross_E1"] = sgn * (t.close / t.open - 1)
    return r


def period(d: pd.Series) -> pd.Series:
    return np.where(d <= IS_END, "IS 2015-21", "OOS 2022-26")


def overlay(trades: pd.DataFrame, ret_col: str, spy: pd.DataFrame, days: pd.DatetimeIndex,
            per_name: float = 0.02, max_gross: float = 0.20) -> pd.Series:
    """100% SPY core; on signal days sell up to ``max_gross`` of SPY at the open (``per_name`` each),
    buy the dips, sell at the close and rebuy SPY. Unfilled trades (NaN) leave that slice in SPY."""
    t = trades[["date", ret_col]].copy()
    t["filled"] = t[ret_col].notna()
    n = t.groupby("date").size()
    w = (max_gross / n).clip(upper=per_name)
    t["w"] = t.date.map(w)
    spy_i = spy.intra.reindex(days)
    t["ex"] = (t[ret_col] - t.date.map(spy_i)).where(t.filled, 0.0) * t.w
    excess = t.groupby("date").ex.sum().reindex(days).fillna(0.0)
    return spy.ret.reindex(days).fillna(0.0) + excess


def sleeve(trades: pd.DataFrame, ret_col: str, days: pd.DatetimeIndex, per_name: float = 0.10) -> pd.Series:
    """Stand-alone dip sleeve: each name 1/max(N, 1/per_name) of the sleeve; cash otherwise (0%)."""
    t = trades[["date", ret_col]].dropna()
    g = t.groupby("date")[ret_col]
    return (g.sum() / g.size().clip(lower=int(round(1 / per_name)))).reindex(days).fillna(0.0)


def fixed_ticket_pnl(trades: pd.DataFrame, ret_col: str, ticket: float, cap_col: str = "cap_usd",
                     max_names: int = 10) -> pd.DataFrame:
    """Daily $ P&L of a fixed-$ ticket per signal (capped at ``cap_col`` $ = participation limit),
    at most ``max_names`` per day (largest prior $ volume first). Returns date-indexed frame."""
    t = trades.dropna(subset=[ret_col]).copy()
    t = t.sort_values(["date", "prior_dvol"], ascending=[True, False])
    t = t.groupby("date").head(max_names)
    t["size"] = np.minimum(ticket, t[cap_col])
    t["pnl"] = t["size"] * t[ret_col]
    return t.groupby("date").agg(pnl=("pnl", "sum"), deployed=("size", "sum"), n=("size", "size"),
                                 capped=("size", lambda s: (s < ticket).mean()))
