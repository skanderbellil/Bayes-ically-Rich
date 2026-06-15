"""Cross-market momentum backtest engine for Polymarket.

A market position is a holding in the Yes token, whose price is the implied
probability p ∈ (0, 1). We size positions as a fraction of $1 gross notional and
mark them to market daily, so the per-day PnL of a position with weight w in
market i is::

    pnl_i,t = w_i,t · (p_i,t+1 - p_i,t)

i.e. the dollar change in the Yes price (a long captures up-moves, a short — a
Yes weight of the opposite sign, equivalent to buying No — captures down-moves).
Holding to resolution realises the full move to 0 or 1, which the price history
already encodes. This "$1-of-max-loss" convention avoids the 1/p blow-up of a
naive percentage return on cheap longshots and keeps cross-market sizing honest.

Each day the engine:
  1. reads the signal computed *only from prices up to t* (no lookahead),
  2. forms target weights over the markets active that day,
  3. charges a cost on the turnover from yesterday's book,
  4. earns w · Δp into t+1.

Strategies (selected by ``PMParams.strategy``):
  xs_momentum  — long top / short bottom by cross-sectional log-odds momentum
  xs_reversal  — the contrarian leg (longshot mean-reversion hypothesis)
  ts_momentum  — per-market time-series momentum (sign of own trailing move)
  long_all     — equal-weight long every active market (drift baseline)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .signals import (
    bayesian_shrink,
    cross_sectional_score,
    logodds_momentum,
)

_EPS = 1e-9


@dataclass
class PMParams:
    """Configuration for a Polymarket momentum backtest."""
    strategy: str = "xs_momentum"     # xs_momentum | xs_reversal | ts_momentum | long_all
    lookback: int = 7                 # momentum formation window (days)
    holding: int = 1                  # rebalance every N days
    top_frac: float = 0.2             # fraction of universe in each leg (xs strategies)
    gross: float = 1.0                # gross exposure ($ notional)
    min_names: int = 6                # minimum active markets to trade a date
    min_price: float = 0.05           # skip near-resolved markets (illiquid edges)
    max_price: float = 0.95
    bayesian: bool = True             # apply signal-to-noise shrinkage
    prior_scale: float = 1.0          # Bayesian prior width (× median noise)
    cost: float = 0.005              # per-unit turnover cost (50 bps of notional traded)


@dataclass
class PMResult:
    """Output of a Polymarket backtest."""
    returns: pd.Series                       # daily portfolio PnL on $1 gross, net of cost
    gross_returns: pd.Series                  # same, before turnover cost
    weights: pd.DataFrame                     # date × market target weights
    gross_series: pd.Series                   # realised gross exposure per day
    n_active: pd.Series                       # tradable markets per day
    turnover: pd.Series                       # per-day one-way turnover
    params: PMParams = field(default_factory=PMParams)


# ---------------------------------------------------------------------------
# Weight construction
# ---------------------------------------------------------------------------

def _xs_weights(score_row: pd.Series, top_frac: float, gross: float,
                contrarian: bool) -> pd.Series:
    """Dollar-neutral long-top / short-bottom weights from a cross-sectional score."""
    s = score_row.dropna()
    n = len(s)
    w = pd.Series(0.0, index=score_row.index)
    k = max(1, int(round(n * top_frac)))
    if n < 2 * k:
        return w
    order = s.sort_values()
    bottom = order.index[:k]
    top = order.index[-k:]
    sign = -1.0 if contrarian else 1.0
    leg = gross / 2.0 / k
    w[top] = sign * leg
    w[bottom] = -sign * leg
    return w


def _ts_weights(mom_row: pd.Series, gross: float) -> pd.Series:
    """Per-market time-series momentum: long up-drifters, short down-drifters."""
    s = mom_row.dropna()
    n = len(s)
    w = pd.Series(0.0, index=mom_row.index)
    if n == 0:
        return w
    w[s.index] = np.sign(s.values) * (gross / n)
    return w


def _long_all_weights(active: pd.Index, all_cols: pd.Index, gross: float) -> pd.Series:
    """Equal-weight long every active market (passive drift baseline)."""
    w = pd.Series(0.0, index=all_cols)
    if len(active):
        w[active] = gross / len(active)
    return w


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def run_polymarket_momentum(panel: pd.DataFrame, params: PMParams = PMParams()) -> PMResult:
    """Walk a daily Yes-price panel through history, no lookahead.

    The signal at date *t* is built from prices up to *t*; the resulting book is
    held into *t+1* and earns w·Δp, net of turnover cost charged at *t*.
    """
    p = panel.sort_index()
    dates = p.index

    # ── Signal (causal: uses only past prices at each row) ────────────────
    mom = logodds_momentum(p, params.lookback)
    if params.bayesian:
        mom = bayesian_shrink(mom, p, params.lookback, params.prior_scale)

    if params.strategy in ("xs_momentum", "xs_reversal"):
        score = cross_sectional_score(mom, min_names=params.min_names)
    else:
        score = mom

    # tradability mask: market must be live and priced inside [min,max] at t
    tradable = p.notna() & (p >= params.min_price) & (p <= params.max_price)

    weights = pd.DataFrame(0.0, index=dates, columns=p.columns)
    prev_w = pd.Series(0.0, index=p.columns)
    rets, gross_rets, turn, gross_real, n_act = [], [], [], [], []

    for i, t in enumerate(dates):
        active = p.columns[tradable.loc[t].values]

        # rebalance only every `holding` days; otherwise carry the book
        if i % params.holding == 0:
            if params.strategy == "long_all":
                w = _long_all_weights(active, p.columns, params.gross)
            elif params.strategy == "ts_momentum":
                w = _ts_weights(mom.loc[t].reindex(active), params.gross)
                w = w.reindex(p.columns, fill_value=0.0)
            else:  # xs_momentum / xs_reversal
                row = score.loc[t].reindex(active)
                w = _xs_weights(row, params.top_frac, params.gross,
                                contrarian=(params.strategy == "xs_reversal"))
                w = w.reindex(p.columns, fill_value=0.0)
        else:
            w = prev_w.copy()
            w[~p.columns.isin(active)] = 0.0   # drop names that left the universe

        weights.loc[t] = w

        # turnover cost charged on the change from yesterday's book
        to = float((w - prev_w).abs().sum())
        cost = params.cost * to

        # PnL realised t -> t+1 from the price change of held names
        if i + 1 < len(dates):
            dp = (p.iloc[i + 1] - p.iloc[i]).reindex(p.columns).fillna(0.0)
            pnl = float((w * dp).sum())
        else:
            pnl = 0.0

        rets.append(pnl - cost)
        gross_rets.append(pnl)
        turn.append(to)
        gross_real.append(float(w.abs().sum()))
        n_act.append(len(active))
        prev_w = w

    idx = dates
    return PMResult(
        returns=pd.Series(rets, index=idx, name=params.strategy),
        gross_returns=pd.Series(gross_rets, index=idx, name=params.strategy + "_gross"),
        weights=weights,
        gross_series=pd.Series(gross_real, index=idx, name="gross"),
        n_active=pd.Series(n_act, index=idx, name="n_active"),
        turnover=pd.Series(turn, index=idx, name="turnover"),
        params=params,
    )
