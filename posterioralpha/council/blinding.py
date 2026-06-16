"""
Blinded period contexts: what a council member is allowed to see.

A PeriodContext is the entire information set for one decision. It is built
from data up to (and including) the decision date only, then anonymized:

  - no dates: observations are indexed t-N … t0 (trading days back)
  - no tickers / series names: domains expose generic indicator labels
  - no levels: every series is a rolling z-score or a normalized return path
  - randomized presentation: per-context jitter in rounding and sampling
    stride, so an LLM cannot match byte patterns across runs or periods

`mirror()` returns the sign-flipped copy used by the leakage audit: a pure
analyst's stance must invert on the mirrored context.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _zscore(s: pd.Series, window: int = 252) -> pd.Series:
    mu = s.rolling(window, min_periods=window // 2).mean()
    sd = s.rolling(window, min_periods=window // 2).std()
    return (s - mu) / (sd + 1e-9)


@dataclass
class DomainView:
    """One specialist's slice of a period context."""
    domain: str
    # generic label -> recent z-score path (oldest → newest), no dates/names
    indicators: Dict[str, List[float]]

    def mirror(self) -> "DomainView":
        return DomainView(
            domain=self.domain,
            indicators={k: [-x for x in v] for k, v in self.indicators.items()},
        )

    def latest(self, label: str) -> float:
        v = self.indicators.get(label, [])
        return v[-1] if v else 0.0


@dataclass
class PeriodContext:
    """Everything the council may see for one decision (blinded)."""
    period_id: int                       # opaque — NOT a date
    views: Dict[str, DomainView]
    asset_path: List[float]              # normalized cumulative return path
    asset_vol_z: float                   # realized-vol z of the underlying
    is_mirrored: bool = False

    def mirror(self) -> "PeriodContext":
        return PeriodContext(
            period_id=self.period_id,
            views={d: v.mirror() for d, v in self.views.items()},
            asset_path=[-x for x in self.asset_path],
            asset_vol_z=self.asset_vol_z,   # vol is sign-invariant
            is_mirrored=not self.is_mirrored,
        )


# Domain → (panel column → generic blinded label). The specialist never sees
# the FRED ids, only the right-hand labels.
DOMAIN_SERIES: Dict[str, Dict[str, str]] = {
    "rates": {
        "DGS2":   "short_rate_z",
        "DGS10":  "long_rate_z",
        "T10Y2Y": "curve_slope_z",
        "DFII10": "real_yield_z",
    },
    "credit": {
        "BAA10Y": "credit_spread_z",
        "AAA10Y": "quality_spread_z",
    },
    "volatility": {
        "VIXCLS": "implied_vol_z",
    },
    "dollar": {
        "DTWEXBGS": "fx_strength_z",
    },
    "liquidity": {
        "net_liquidity": "system_liquidity_z",
    },
    # "trend" is built from the asset itself, not the macro panel
}

_Z_WINDOW = 126        # rolling z window (~6 months)
_PATH_DAYS = 126       # context lookback for paths


class ContextBuilder:
    """Precomputes z-scores once, then cuts blinded contexts per period."""

    def __init__(
        self,
        prices: pd.Series,
        macro: Optional[pd.DataFrame] = None,
        liquidity: Optional[pd.Series] = None,
        seed: int = 0,
    ):
        self.prices = prices
        self.returns = prices.pct_change()
        self.rng = np.random.default_rng(seed)

        # assemble all domain source series, aligned to the price calendar
        sources: Dict[str, pd.Series] = {}
        if macro is not None:
            for col in macro.columns:
                sources[col] = macro[col].reindex(prices.index).ffill()
        if liquidity is not None:
            sources["net_liquidity"] = liquidity.reindex(prices.index).ffill()

        self._z: Dict[str, Dict[str, pd.Series]] = {}
        for domain, mapping in DOMAIN_SERIES.items():
            self._z[domain] = {}
            for col, label in mapping.items():
                if col in sources:
                    self._z[domain][label] = _zscore(sources[col], _Z_WINDOW)

        # trend domain from the asset itself
        ma = prices.rolling(_Z_WINDOW).mean()
        self._z["trend"] = {
            "trend_strength_z": _zscore(prices / ma - 1.0, _Z_WINDOW),
            "momentum_z":       _zscore(prices.pct_change(63), _Z_WINDOW),
        }

        vol = self.returns.rolling(21).std()
        self._vol_z = _zscore(vol, 252)

    @property
    def domains(self) -> List[str]:
        return [d for d, m in self._z.items() if m]

    def build(self, date: pd.Timestamp, period_id: int) -> Optional[PeriodContext]:
        """Blinded context using data ≤ `date`. None if history insufficient."""
        loc = self.prices.index.get_loc(date)
        if loc < _PATH_DAYS + _Z_WINDOW:
            return None

        # randomized presentation: stride 2-4 days, 2-3 decimals
        stride = int(self.rng.integers(2, 5))
        decimals = int(self.rng.integers(2, 4))

        def cut(s: pd.Series) -> List[float]:
            w = s.iloc[loc - _PATH_DAYS: loc + 1].dropna()
            vals = w.values[::-1][::stride][::-1]      # keep newest point
            return [round(float(x), decimals) for x in vals]

        views = {}
        for domain, mapping in self._z.items():
            indicators = {lbl: cut(z) for lbl, z in mapping.items()}
            indicators = {k: v for k, v in indicators.items() if len(v) >= 5}
            if indicators:
                views[domain] = DomainView(domain=domain, indicators=indicators)

        path = self.returns.iloc[loc - _PATH_DAYS: loc + 1].fillna(0.0)
        cum = (1.0 + path).cumprod()
        norm = cum / cum.iloc[0] - 1.0
        sd = norm.std() + 1e-9
        asset_path = [round(float(x / sd), decimals)
                      for x in norm.values[::-1][::stride][::-1]]

        return PeriodContext(
            period_id=period_id,
            views=views,
            asset_path=asset_path,
            asset_vol_z=round(float(self._vol_z.iloc[loc]), 2)
            if not np.isnan(self._vol_z.iloc[loc]) else 0.0,
        )
