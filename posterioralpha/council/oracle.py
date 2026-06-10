"""
Oracle specialists: simulated lookahead for auditing the audit.

An OracleSpecialist mimics the failure mode the council is designed to
catch — an analyst (e.g. an LLM) that recognizes WHICH historical episode
it is looking at and recalls what happened next, instead of analyzing the
numbers. It blends an honest statistical stance with a "memory" stance:

    stance = (1 - leak) · honest(ctx)  +  leak · memory(period)

The memory term is keyed on the period's identity, NOT on the context data,
so it is identical for the original and the mirrored context — exactly the
signature the mirror audit looks for. `leak` is the dose: 0 = honest
analyst, 1 = pure oracle.

This lets the harness be falsified before any LLM is involved: inject a
known quantity of lookahead and check that (a) the mirror audit flags it,
(b) performance inflates the way lookahead should, and (c) dropping flagged
seats repairs the backtest.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from posterioralpha.council.blinding import PeriodContext
from posterioralpha.council.council import CouncilBacktest
from posterioralpha.council.specialists import StatisticalSpecialist, Verdict


def build_answer_key(bt: CouncilBacktest) -> Dict[int, float]:
    """
    {period_id: forward-return z} — the "history book" an oracle remembers.

    Forward return = underlying's return over the period the decision is
    held for (decision date → next decision date), standardized across
    periods so the oracle's memory is scale-free.
    """
    dates = [d for d, _ in bt.build_contexts()]
    ids = [c.period_id for _, c in bt.build_contexts()]
    fwd = []
    for i, d in enumerate(dates):
        end = dates[i + 1] if i + 1 < len(dates) else bt.prices.index[-1]
        seg = bt.returns.loc[d:end].iloc[1:]
        fwd.append(float((1.0 + seg).prod() - 1.0) if len(seg) else 0.0)
    z = (np.array(fwd) - np.mean(fwd)) / (np.std(fwd) + 1e-9)
    return dict(zip(ids, z))


class OracleSpecialist:
    """
    Honest analyst with a tunable amount of episode memory.

    `recall_noise` makes the memory imperfect (an LLM remembers "the market
    recovered", not the exact return) — drawn once per period so original
    and mirrored assessments still agree, as real recognition would.
    """

    def __init__(self, domain: str, answer_key: Dict[int, float],
                 leak: float, recall_noise: float = 0.5, seed: int = 0):
        self.domain = domain
        self.leak = float(leak)
        self._honest = StatisticalSpecialist(domain)
        rng = np.random.default_rng(seed + hash(domain) % 10_000)
        self._memory = {
            pid: float(np.tanh(z + recall_noise * rng.normal()))
            for pid, z in answer_key.items()
        }

    def assess(self, ctx: PeriodContext) -> Verdict:
        honest = self._honest.assess(ctx)
        memory = self._memory.get(ctx.period_id, 0.0)   # ignores mirroring!
        stance = (1.0 - self.leak) * honest.stance + self.leak * memory
        conf = max(honest.confidence, self.leak * 0.9)
        return Verdict(float(np.clip(stance, -1, 1)), float(conf),
                       f"leak={self.leak:.2f}")
