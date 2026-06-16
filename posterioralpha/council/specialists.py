"""
Council specialists: one domain each, stance from the blinded context only.

A specialist maps a DomainView (plus the asset's blinded path) to a Verdict:
stance ∈ [-1, +1] (risk-off … risk-on), confidence ∈ [0, 1], and a one-line
rationale. The statistical backend is deterministic and fit-free — fixed
mappings, no tuned parameters — so the walk-forward backtest contains zero
in-sample optimization. It is also the calibration baseline for the mirror
audit: every statistical stance is an odd function of its inputs, so its
anti-symmetry score is exactly 1.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Protocol

import numpy as np

from posterioralpha.council.blinding import PeriodContext


@dataclass
class Verdict:
    stance:     float   # -1 (fully risk-off) … +1 (fully risk-on)
    confidence: float   # 0 … 1
    rationale:  str = ""


class Specialist(Protocol):
    domain: str
    def assess(self, ctx: PeriodContext) -> Verdict: ...


def _conf(z: float) -> float:
    """Confidence grows with signal extremity, capped at 1."""
    return float(np.clip(abs(z) / 2.0, 0.1, 1.0))


class StatisticalSpecialist:
    """
    Deterministic domain analyst. The sign conventions encode only textbook
    direction (tight credit good, high vol bad, rising liquidity good …) —
    magnitudes come from the data, nothing is fitted.
    """

    def __init__(self, domain: str):
        self.domain = domain

    def assess(self, ctx: PeriodContext) -> Verdict:
        view = ctx.views.get(self.domain)
        if view is None:
            return Verdict(0.0, 0.0, "no data")

        d = self.domain
        if d == "rates":
            # rising short/real rates = tightening headwind
            z = 0.5 * view.latest("short_rate_z") + 0.5 * view.latest("real_yield_z")
            stance = -np.tanh(z)
            why = f"rate pressure z={z:+.2f}"
        elif d == "credit":
            z = view.latest("credit_spread_z")
            stance = -np.tanh(z)
            why = f"credit spread z={z:+.2f}"
        elif d == "volatility":
            z = view.latest("implied_vol_z")
            stance = -np.tanh(z)
            why = f"implied vol z={z:+.2f}"
        elif d == "dollar":
            z = view.latest("fx_strength_z")
            stance = -np.tanh(z)
            why = f"dollar strength z={z:+.2f}"
        elif d == "liquidity":
            z = view.latest("system_liquidity_z")
            stance = np.tanh(z)
            why = f"system liquidity z={z:+.2f}"
        elif d == "trend":
            z = 0.5 * view.latest("trend_strength_z") + 0.5 * view.latest("momentum_z")
            stance = np.tanh(z)
            why = f"trend z={z:+.2f}"
        else:
            return Verdict(0.0, 0.0, "unknown domain")

        z_for_conf = float(np.arctanh(np.clip(stance, -0.999, 0.999)))
        return Verdict(float(stance), _conf(z_for_conf), why)


DOMAINS = ("rates", "credit", "volatility", "dollar", "liquidity", "trend")


def statistical_council(domains=DOMAINS) -> Dict[str, Specialist]:
    return {d: StatisticalSpecialist(d) for d in domains}
