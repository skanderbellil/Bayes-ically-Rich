"""
Council backtest engine.

For each period (default: month-end), the engine builds a blinded context
from trailing data only, asks every specialist for a stance, aggregates the
council into an exposure e ∈ [0, 1], and holds it (1-day execution lag,
turnover costs) until the next period. Alongside the council it tracks each
specialist's SOLO dial, so the report shows who actually earned their seat.

Mirror audit
------------
Every specialist also assesses the sign-flipped copy of each context. For an
honest analyst the stance is an odd function of the data, so

    anti_symmetry = -corr(stance(original), stance(mirrored)) ≈ 1.

A score well below 1 for an LLM specialist means its stances are driven by
something other than the supplied numbers — episode recognition (lookahead
through memory) being the prime suspect. The statistical backend scores an
exact 1 by construction, calibrating the metric. This measures residual
leakage rather than assuming the blinding worked.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from posterioralpha.council.blinding import ContextBuilder, PeriodContext
from posterioralpha.council.specialists import (
    DOMAINS,
    Specialist,
    Verdict,
    statistical_council,
)
from posterioralpha.mining.evaluation import sharpe

logger = logging.getLogger(__name__)


@dataclass
class CouncilConfig:
    rebalance: str = "ME"          # month-end decisions
    tc: float = 0.0002             # 2 bps per unit one-way turnover
    min_history: int = 260
    seed: int = 7
    domains: Tuple[str, ...] = DOMAINS


@dataclass
class CouncilResult:
    exposure: pd.Series            # daily exposure actually held
    returns: pd.Series             # daily net returns
    minutes: pd.DataFrame          # per-period stances/confidences/exposure
    solo_returns: Dict[str, pd.Series]
    audit: pd.DataFrame            # mirror-test anti-symmetry per specialist


class CouncilBacktest:
    def __init__(
        self,
        prices: pd.Series,
        macro: Optional[pd.DataFrame] = None,
        liquidity: Optional[pd.Series] = None,
        config: Optional[CouncilConfig] = None,
    ):
        self.cfg = config or CouncilConfig()
        self.prices = prices.dropna()
        self.returns = self.prices.pct_change()
        self.builder = ContextBuilder(self.prices, macro=macro,
                                      liquidity=liquidity, seed=self.cfg.seed)
        self.domains = [d for d in self.cfg.domains if d in self.builder.domains]

    # ── Period grid & contexts ─────────────────────────────────────────────

    def period_dates(self) -> List[pd.Timestamp]:
        grid = self.prices.resample(self.cfg.rebalance).last().index
        dates = []
        for g in grid:
            idx = self.prices.index[self.prices.index <= g]
            if len(idx) > self.cfg.min_history:
                dates.append(idx[-1])
        return sorted(set(dates))

    def build_contexts(self) -> List[Tuple[pd.Timestamp, PeriodContext]]:
        out = []
        for i, date in enumerate(self.period_dates()):
            ctx = self.builder.build(date, period_id=i)
            if ctx is not None:
                out.append((date, ctx))
        return out

    # ── Aggregation ────────────────────────────────────────────────────────

    @staticmethod
    def aggregate(verdicts: Dict[str, Verdict]) -> float:
        """Confidence-weighted mean stance, mapped to exposure ∈ [0, 1]."""
        num = sum(v.stance * v.confidence for v in verdicts.values())
        den = sum(v.confidence for v in verdicts.values())
        stance = num / den if den > 0 else 0.0
        return float(np.clip(0.5 + 0.5 * stance, 0.0, 1.0))

    # ── Run ────────────────────────────────────────────────────────────────

    def run(
        self,
        council: Optional[Dict[str, Specialist]] = None,
        verdicts_cache: Optional[Dict[Tuple[int, str, bool], Verdict]] = None,
        period_ids: Optional[set] = None,
    ) -> CouncilResult:
        """
        `council`        — {domain: Specialist}; defaults to the statistical one.
        `verdicts_cache` — pre-computed verdicts keyed (period_id, domain,
                           is_mirrored), e.g. from llm.run_llm_council_batch;
                           specialists are then never called directly.
        `period_ids`     — restrict the backtest to these periods (e.g. the
                           ones actually paid for in an LLM batch).
        """
        if council is None and verdicts_cache is None:
            council = statistical_council(self.domains)

        contexts = self.build_contexts()
        if period_ids is not None:
            contexts = [(d, c) for d, c in contexts if c.period_id in period_ids]
        rows = []
        stance_track: Dict[str, List[float]] = {d: [] for d in self.domains}
        mirror_track: Dict[str, List[float]] = {d: [] for d in self.domains}

        for date, ctx in tqdm(contexts, desc="  council periods", leave=False):
            verdicts: Dict[str, Verdict] = {}
            mirrored: Dict[str, Verdict] = {}
            mctx = ctx.mirror()
            for d in self.domains:
                if verdicts_cache is not None:
                    verdicts[d] = verdicts_cache.get(
                        (ctx.period_id, d, False), Verdict(0.0, 0.0, "missing"))
                    mirrored[d] = verdicts_cache.get(
                        (ctx.period_id, d, True), Verdict(0.0, 0.0, "missing"))
                else:
                    verdicts[d] = council[d].assess(ctx)
                    mirrored[d] = council[d].assess(mctx)
                stance_track[d].append(verdicts[d].stance)
                mirror_track[d].append(mirrored[d].stance)

            row = {"date": date, "exposure": self.aggregate(verdicts)}
            for d in self.domains:
                row[f"{d}_stance"] = round(verdicts[d].stance, 3)
                row[f"{d}_conf"] = round(verdicts[d].confidence, 3)
                row[f"{d}_rationale"] = verdicts[d].rationale
            rows.append(row)

        minutes = pd.DataFrame(rows).set_index("date")

        # daily exposure: decided at period end, held (lagged 1 day) until next
        result_returns, exposure = self._exposure_to_returns(minutes["exposure"])

        solo = {}
        for d in self.domains:
            e = pd.Series(
                np.clip(0.5 + 0.5 * np.array(stance_track[d]), 0, 1),
                index=minutes.index,
            )
            solo[d], _ = self._exposure_to_returns(e)

        audit = self._mirror_audit(stance_track, mirror_track)
        return CouncilResult(exposure=exposure, returns=result_returns,
                             minutes=minutes, solo_returns=solo, audit=audit)

    def _exposure_to_returns(self, period_exposure: pd.Series):
        e_daily = period_exposure.reindex(self.prices.index).ffill().fillna(0.0)
        held = e_daily.shift(1).fillna(0.0)
        turnover = held.diff().abs().fillna(0.0)
        net = held * self.returns.fillna(0.0) - self.cfg.tc * turnover
        start = period_exposure.index[0]
        return net.loc[start:], e_daily.loc[start:]

    @staticmethod
    def _mirror_audit(stances: Dict[str, List[float]],
                      mirrored: Dict[str, List[float]]) -> pd.DataFrame:
        """
        Two statistics per specialist:

        anti_symmetry — -corr(stance, mirrored stance); ≈1 for an honest
                        analyst. Blunt at low leak levels (the oracle test
                        showed a 25% leak keeps it ≈0.84).
        leak_ratio    — std(orig + mirrored) / (std(orig) + std(mirrored)).
                        For an honest (odd) analyst the sum is identically
                        zero, so ANY symmetric residual is direct evidence
                        of episode memory; ≈0 honest, ≈1 pure oracle, and
                        ≈0.33 already at a 25% leak.

        SUSPECT if anti_symmetry < 0.8 or leak_ratio > 0.25.
        """
        rows = []
        for d in stances:
            a = np.array(stances[d])
            b = np.array(mirrored[d])
            ok = (np.std(a) > 1e-9) and (np.std(b) > 1e-9)
            anti = float(-np.corrcoef(a, b)[0, 1]) if ok else np.nan
            leak = float(np.std(a + b) / (np.std(a) + np.std(b) + 1e-9))
            suspect = (anti == anti and anti < 0.8) or leak > 0.25
            rows.append({
                "specialist": d,
                "anti_symmetry": round(anti, 3) if anti == anti else np.nan,
                "leak_ratio": round(leak, 3),
                "verdict": ("SUSPECT" if suspect
                            else "CLEAN" if anti == anti else "n/a"),
            })
        return pd.DataFrame(rows).set_index("specialist")

    # ── Reporting ──────────────────────────────────────────────────────────

    def summary(self, result: CouncilResult) -> pd.DataFrame:
        rows = []

        def add(name, r):
            r = r.dropna()
            ann = 252
            cagr = (1 + r).prod() ** (ann / max(len(r), 1)) - 1
            cum = (1 + r).cumprod()
            dd = float(((cum - cum.cummax()) / cum.cummax()).min())
            rows.append({"strategy": name, "sharpe": round(sharpe(r.values), 3),
                         "cagr": round(float(cagr), 4), "max_dd": round(dd, 4)})

        add("council", result.returns)
        for d, r in result.solo_returns.items():
            add(f"solo:{d}", r)
        bh = self.returns.loc[result.returns.index[0]:]
        add("buy_and_hold", bh)
        return pd.DataFrame(rows).set_index("strategy")
