"""
Multi-asset council: the macro dial allocated across an ETF basket.

Architecture (all fit-free, monthly):

  1. RISK DIAL  — the validated 6-seat macro council (rates, credit, vol,
                  dollar, liquidity, trend) reads blinded contexts built on
                  the broad market and votes exposure e ∈ [0, 1]. Identical
                  machinery to the single-asset engine, mirror audit included.
  2. RISKY SLEEVE (weight e) — spread across the basket (broad + sector
                  ETFs) by inverse volatility, with a bounded relative-
                  strength tilt: w_i ∝ invvol_i · (1 + κ·tanh(rs_z_i)),
                  κ = 0.5 fixed. The tilt is capped at ±50% of the base
                  weight by construction — it can lean, never concentrate.
  3. SLACK SLEEVE (weight 1−e) — cash by default; optionally a defensive
                  basket (TLT+GLD, or UUP per the dollar-slack finding).

Costs are the Alpaca retail model (commission-free; ~1 bp one-way ETF
half-spread + 0.3 bp regulatory fee on sells), charged on actual turnover —
this is replicable in a retail account as-is.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from posterioralpha.backtest.alpaca_costs import AlpacaCosts, HALF_SPREAD_BPS
from posterioralpha.council.council import CouncilBacktest, CouncilConfig

RISKY_BASKET = ["SPY", "QQQ", "XLK", "XLF", "XLE", "XLV",
                "XLY", "XLP", "XLI", "XLU", "XLB"]

SLACK_BASKETS: Dict[str, List[str]] = {
    "cash":    [],
    "tlt-gld": ["TLT", "GLD"],
    "uup":     ["UUP"],
}

_TILT_KAPPA = 0.5      # max relative-strength lean: ±50% of base weight
_VOL_WINDOW = 63
_RS_WINDOW = 126


@dataclass
class MultiAssetResult:
    returns:  pd.Series           # daily net portfolio returns
    weights:  pd.DataFrame        # daily held weights (incl. slack assets)
    exposure: pd.Series           # the council's risk dial
    audit:    pd.DataFrame        # mirror audit of the macro seats
    minutes:  pd.DataFrame        # council minutes (per-period stances)
    cost_drag_annual: float       # realized annualized cost drag


class MultiAssetCouncil:
    def __init__(
        self,
        prices: pd.DataFrame,                 # basket + slack + dial proxy columns
        macro: Optional[pd.DataFrame] = None,
        liquidity: Optional[pd.Series] = None,
        risky: Optional[List[str]] = None,
        slack: str = "cash",
        tilt: bool = True,
        dial_proxy: str = "SPY",
        seed: int = 7,
    ):
        self.risky = [t for t in (risky or RISKY_BASKET) if t in prices.columns]
        self.slack_assets = [t for t in SLACK_BASKETS[slack] if t in prices.columns]
        self.tilt = tilt
        cols = sorted(set(self.risky + self.slack_assets + [dial_proxy]))
        self.prices = prices[cols].dropna(how="all").ffill()
        self.returns = self.prices.pct_change()
        self.costs = AlpacaCosts()

        # the macro dial runs on a broad proxy with the full blinded machinery
        self.dial = CouncilBacktest(
            self.prices[dial_proxy], macro=macro, liquidity=liquidity,
            config=CouncilConfig(seed=seed),
        )

    # ── Risky-sleeve weights at one rebalance date ─────────────────────────

    def _sleeve_weights(self, date: pd.Timestamp) -> pd.Series:
        loc = self.prices.index.get_loc(date)
        if loc < _RS_WINDOW + 1:
            return pd.Series(1.0 / len(self.risky), index=self.risky)

        rets = self.returns[self.risky].iloc[: loc + 1]
        vol = rets.iloc[-_VOL_WINDOW:].std()
        inv = (1.0 / (vol + 1e-9))
        base = inv / inv.sum()

        if self.tilt:
            rs = (self.prices[self.risky].iloc[loc]
                  / self.prices[self.risky].iloc[loc - _RS_WINDOW] - 1.0)
            z = (rs - rs.mean()) / (rs.std() + 1e-9)      # cross-sectional
            w = base * (1.0 + _TILT_KAPPA * np.tanh(z))
        else:
            w = base
        return w / w.sum()

    # ── Run ────────────────────────────────────────────────────────────────

    def run(self) -> MultiAssetResult:
        dial_res = self.dial.run()
        # period exposure decided at each council date
        e_periods = dial_res.minutes["exposure"]

        rows, dates = [], []
        for date, e in e_periods.items():
            sleeve = self._sleeve_weights(date)
            w = pd.Series(0.0, index=self.prices.columns)
            w[self.risky] = float(e) * sleeve
            if self.slack_assets:
                w[self.slack_assets] = (1.0 - float(e)) / len(self.slack_assets)
            # remainder (if slack == cash) is implicitly cash at 0%
            rows.append(w)
            dates.append(date)

        target = pd.DataFrame(rows, index=dates)
        daily = target.reindex(self.prices.index).ffill().fillna(0.0)
        held = daily.shift(1).fillna(0.0)

        gross = (held * self.returns.fillna(0.0)).sum(axis=1)

        # Alpaca costs: buys pay the half-spread; sells pay it + reg fee
        dw = held.diff().fillna(0.0)
        buy_turn = dw.clip(lower=0.0).sum(axis=1)
        sell_turn = (-dw.clip(upper=0.0)).sum(axis=1)
        bps = HALF_SPREAD_BPS["ETF"] / 1e4
        fee = self.costs.reg_sell_fee_bps / 1e4
        cost = buy_turn * bps + sell_turn * (bps + fee)
        net = gross - cost

        start = dates[0]
        net = net.loc[start:]
        n_years = max(len(net) / 252.0, 1e-9)
        return MultiAssetResult(
            returns=net,
            weights=held.loc[start:],
            exposure=dial_res.exposure,
            audit=dial_res.audit,
            minutes=dial_res.minutes,
            cost_drag_annual=float(cost.loc[start:].sum() / n_years),
        )
