"""Event-driven trade simulator — can the sharp-move edge be *harvested*?

The vol→outcome study (``events.py`` / ``SHARP_MOVE_OUTCOME.md``) found two
opposite-signed effects after a sharp upside move: the market **underprices Yes**
(it resolves Yes more often than its price implies — a positive *calibration*
edge), yet the price **overreacts short-term** (negative forward drift). Those
two facts call for two different exit policies, which is exactly what this
simulator makes testable.

A trade is opened when a market's chosen vol metric (upside_vol / sharp_up /
vol_skew) crosses into its top quantile, with deliberate **reality checks**:

  * the trigger is causal (metric at *t* uses prices ≤ *t*);
  * the fill happens *after* the move — at *t + entry_lag* — so we pay the already
    elevated price, never the pre-spike one;
  * crossing the book costs ``slippage`` on entry, and again on an early exit
    (settlement at resolution has no exit slippage — the contract just pays 0/1);
  * one position per market at a time, with a re-entry ``cooldown``.

Exit policies (``TradeParams.exit``):
  resolution  — hold to settlement; PnL = side·(outcome − entry).   [Idea 1]
  take_profit — bank a favourable drift of ``take_profit``, stop at ``stop_loss``,
                else settle at the end.                              [Idea 2]
  horizon     — mark out after ``max_hold`` days (pure drift capture).

All PnL is per Yes share on $1 notional, in probability units. A No position
(side = −1) is bookkept in Yes-price space: PnL = −(exit − entry), and settles to
``1 − outcome`` automatically since side·(outcome − entry) handles it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .volatility import sharp_up, upside_vol, vol_skew

logger = logging.getLogger(__name__)

_METRICS = {"upside_vol": upside_vol, "sharp_up": sharp_up, "vol_skew": vol_skew}


@dataclass
class TradeParams:
    metric: str = "upside_vol"        # trigger metric (top-quantile crossing)
    entry_quantile: float = 0.8       # 0.8 → top 20% (Q5); 0.6 → top 40% (Q4+Q5)
    side: str = "yes_only"            # "yes_only" | "no_only" | "directional" (vol_skew sign)
    exit: str = "resolution"          # "resolution" | "take_profit" | "horizon"
    take_profit: float = 0.10         # favourable move banked (prob units)
    stop_loss: float = 0.25           # adverse move stopped (prob units)
    max_hold: int = 60                # cap on holding days
    vol_window: int = 10
    entry_lag: int = 1                # fill this many bars after the trigger
    slippage: float = 0.01            # half-spread paid per book crossing
    cost: float = 0.0                 # extra per-trade fee
    min_price: float = 0.05           # tradable entry band (skip dead longshots / near-settled)
    max_price: float = 0.95
    cooldown: int = 10                # bars before re-entering the same market


@dataclass
class TradeResult:
    trades: pd.DataFrame              # one row per closed trade
    summary: dict                     # aggregate stats
    params: TradeParams = field(default_factory=TradeParams)


def _metric_panel(panel: pd.DataFrame, params: TradeParams) -> pd.DataFrame:
    fn = _METRICS[params.metric]
    return fn(panel, params.vol_window)


def simulate_event_trades(
    panel: pd.DataFrame,
    outcomes: pd.Series,
    params: TradeParams = TradeParams(),
) -> TradeResult:
    """Run the event-trigger / exit-policy simulation over all labelled markets."""
    metric = _metric_panel(panel, params)
    skew = vol_skew(panel, params.vol_window)

    # entry threshold: top-quantile of the metric over plausibly-tradable cells
    band = panel.where((panel >= params.min_price) & (panel <= params.max_price))
    mvals = metric.where(band.notna()).values.ravel()
    mvals = mvals[np.isfinite(mvals)]
    if mvals.size == 0:
        return TradeResult(pd.DataFrame(), {"n_trades": 0}, params)
    thresh = float(np.quantile(mvals, params.entry_quantile))

    rows = []
    for m in panel.columns:
        if m not in outcomes.index:
            continue
        y = float(outcomes.loc[m])
        s = panel[m].dropna()
        if len(s) < params.vol_window + params.entry_lag + 2:
            continue
        prices = s.values
        dates = s.index
        mser = metric[m].reindex(dates).values
        sk = skew[m].reindex(dates).values
        n = len(prices)
        last = n - 1

        i = params.vol_window
        while i < n - params.entry_lag:
            mi = mser[i]
            if not np.isfinite(mi) or mi < thresh:
                i += 1
                continue
            ei = i + params.entry_lag                       # fill bar (post-move)
            entry = prices[ei]
            if not (params.min_price <= entry <= params.max_price):
                i += 1
                continue

            if params.side == "yes_only":
                sd = 1.0
            elif params.side == "no_only":
                sd = -1.0
            else:  # directional: let the vol skew pick the side
                sd = 1.0 if (not np.isfinite(sk[i]) or sk[i] >= 0) else -1.0

            # ── resolve the exit ──────────────────────────────────────────
            early = False
            if params.exit == "resolution":
                exit_val = y
                exit_idx = last
            elif params.exit == "horizon":
                exit_idx = min(ei + params.max_hold, last)
                early = exit_idx < last
                exit_val = prices[exit_idx] if early else y
            else:  # take_profit
                exit_idx, exit_val, early = last, y, False
                cap = min(ei + params.max_hold, last)
                for k in range(ei + 1, cap + 1):
                    move = sd * (prices[k] - entry)         # favourable move in our direction
                    if move >= params.take_profit:
                        exit_idx, exit_val, early = k, prices[k], True
                        break
                    if move <= -params.stop_loss:
                        exit_idx, exit_val, early = k, prices[k], True
                        break
                else:
                    # never triggered within max_hold; settle/mark at the cap
                    exit_idx = cap
                    exit_val = y if cap == last else prices[cap]
                    early = cap < last

            slip = params.slippage * (1.0 + (1.0 if early else 0.0))
            pnl = sd * (exit_val - entry) - slip - params.cost
            rows.append({
                "market": m, "entry_date": dates[ei], "exit_date": dates[exit_idx],
                "side": sd, "entry": entry, "exit_val": exit_val,
                "hold_days": int(exit_idx - ei), "outcome": y,
                "early_exit": early, "pnl": pnl,
            })
            i = exit_idx + params.cooldown                  # no overlapping positions

    trades = pd.DataFrame(rows)
    return TradeResult(trades, summarize_trades(trades), params)


def summarize_trades(trades: pd.DataFrame) -> dict:
    """Aggregate stats for a set of trades (per-trade, not time-annualised)."""
    if trades.empty:
        return {"n_trades": 0}
    pnl = trades["pnl"]
    return {
        "n_trades":    int(len(trades)),
        "hit_rate":    float((pnl > 0).mean()),
        "avg_pnl":     float(pnl.mean()),
        "median_pnl":  float(pnl.median()),
        "total_pnl":   float(pnl.sum()),
        "pnl_sharpe":  float(pnl.mean() / (pnl.std() + 1e-9)),   # per-trade info ratio
        "avg_entry":   float(trades["entry"].mean()),
        "yes_rate":    float(trades["outcome"].mean()),
        "avg_hold":    float(trades["hold_days"].mean()),
        "yes_share":   float((trades["side"] > 0).mean()),
    }
