"""
Quant pod simulator: walk-forward alpha mining with a live sleeve book.

This is the loop the user described — what a pod does with an idea:

  RESEARCH  — at each quarterly review, run the timing miner on data up to
              that date ONLY (its internal randomized-window fitness and
              recent-holdout gauntlet all live inside the slice).
  HIRE      — candidates that pass the gate (recent-holdout Sharpe above
              buy & hold with bootstrap and permutation support) are
              deployed into one of a fixed number of book seats.
  MONITOR   — from the hire date on, each sleeve runs live: its dial
              parameters are frozen; only new data flows through them.
  RETIRE    — three rules, mirroring how desks actually cut a signal:
                1. RISK STOP   (monthly): live drawdown exceeds 1.5× the
                   worst drawdown the signal showed in research — the edge
                   is behaving unlike anything in its history.
                2. DECAY       (at review): trailing-1y live Sharpe < 0 —
                   the alpha has been harvested / crowded out.
                3. RE-VALIDATION (at review): the sleeve fails to beat
                   buy & hold over the latest holdout window twice in a
                   row — it no longer passes the gate it was hired on.

Every decision uses only data available at decision time; book performance
is measured exclusively on capital deployed by past decisions. The honest
headline statistic is the IS → LIVE SHARPE HAIRCUT per sleeve: research
Sharpe at hire vs realized Sharpe while deployed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from posterioralpha.mining.evaluation import sharpe
from posterioralpha.mining.timing import (
    DIAL_FAMILIES,
    DialCandidate,
    TimingConfig,
    TimingContext,
    TimingMiner,
)

logger = logging.getLogger(__name__)

# erl_dial excluded by default: BOCPD re-runs are the only slow piece per refit
POD_FAMILIES = ["macro_dial", "trend_dial", "mom_dial", "vol_dial",
                "macro_vote2", "macro_trend_vote"]


@dataclass
class PodConfig:
    seats:          int = 3        # max concurrent sleeves; idle seats in cash
    review_months:  int = 3        # research review cadence (quarterly)
    min_train_days: int = 1500     # ~6y of history before the first review
    holdout_days:   int = 378      # last ~18m of each slice = hiring gate
    population:     int = 20       # miner size per review (small, repeated)
    generations:    int = 4
    gate_perm_p:    float = 0.25
    gate_boot_p:    float = 0.25
    dd_stop_mult:   float = 1.5    # rule 1: live DD vs research max-DD
    decay_window:   int = 252      # rule 2: trailing live Sharpe window
    reval_strikes:  int = 2        # rule 3: consecutive gate failures
    tc:             float = 0.00012  # Alpaca ETF: ~1bp buy / 1.3bp sell, avg
    seed:           int = 7


@dataclass
class Sleeve:
    candidate:  DialCandidate
    hired_at:   pd.Timestamp
    is_sharpe:  float              # holdout Sharpe at hire (research claim)
    is_maxdd:   float              # research max drawdown (for the risk stop)
    net:        pd.Series          # full-history net returns of the frozen dial
    fired_at:   Optional[pd.Timestamp] = None
    fire_reason: str = ""
    strikes:    int = 0

    @property
    def active(self) -> bool:
        return self.fired_at is None

    def live(self, upto: Optional[pd.Timestamp] = None) -> pd.Series:
        end = self.fired_at or upto
        r = self.net.loc[self.hired_at:end].iloc[1:]
        return r if upto is None else r.loc[:upto]


class QuantPod:
    def __init__(self, prices: pd.Series, macro: Optional[pd.DataFrame] = None,
                 config: Optional[PodConfig] = None,
                 families: Optional[List[str]] = None):
        self.cfg = config or PodConfig()
        self.prices = prices.dropna()
        self.returns = self.prices.pct_change()
        self.macro = macro
        self.families = families or POD_FAMILIES
        self.full_ctx = TimingContext(prices=self.prices, macro=macro)
        self.journal: List[Dict] = []
        self.sleeves: List[Sleeve] = []

    # ── Dial evaluation on the full (causal) history ───────────────────────

    def _net_returns(self, c: DialCandidate) -> Tuple[pd.Series, pd.Series]:
        func, _ = DIAL_FAMILIES[c.family]
        e = func(self.full_ctx, **c.params).clip(0.0, 1.0)
        if c.smooth > 1:
            e = e.ewm(span=c.smooth, adjust=False).mean()
        held = e.shift(1).fillna(0.0)
        turnover = held.diff().abs().fillna(0.0)
        net = held * self.returns.fillna(0.0) - self.cfg.tc * turnover
        return net, e

    # ── Review schedule ────────────────────────────────────────────────────

    def _monthly_dates(self) -> List[pd.Timestamp]:
        grid = self.prices.resample("ME").last().index
        out = []
        for g in grid:
            idx = self.prices.index[self.prices.index <= g]
            if len(idx) > self.cfg.min_train_days + self.cfg.holdout_days:
                out.append(idx[-1])
        return sorted(set(out))

    # ── Research review: mine the trailing slice, return hireable rows ─────

    def _research(self, t: pd.Timestamp, review_i: int):
        slice_prices = self.prices.loc[:t]
        hf = self.cfg.holdout_days / len(slice_prices)
        cfg = TimingConfig(
            population=self.cfg.population, generations=self.cfg.generations,
            holdout_frac=hf, n_finalists=6, max_per_family=2,
            n_boot=120, n_perm=80, tc=self.cfg.tc,
            seed=self.cfg.seed + review_i,          # fresh randomization
        )
        miner = TimingMiner(slice_prices, cfg, macro=self.macro,
                            families=self.families)
        leaderboard, finalists = miner.run()
        by_key = {c.key: c for c in finalists}

        hires = []
        for _, row in leaderboard.iterrows():
            gate = (row["holdout_sharpe"] > row["bh_sharpe"]
                    and row["perm_p"] <= self.cfg.gate_perm_p
                    and row["boot_p"] <= self.cfg.gate_boot_p)
            if gate and row["dial"] in by_key:
                hires.append((by_key[row["dial"]], float(row["holdout_sharpe"])))
        return hires

    # ── Retirement rules ───────────────────────────────────────────────────

    def _monitor(self, t: pd.Timestamp, is_review: bool):
        bh = self.returns.loc[:t]
        for s in self.sleeves:
            if not s.active:
                continue
            live = s.live(upto=t)
            if len(live) < 21:
                continue

            # rule 1 — risk stop (every month)
            cum = (1 + live).cumprod()
            live_dd = float(((cum - cum.cummax()) / cum.cummax()).min())
            if live_dd < self.cfg.dd_stop_mult * s.is_maxdd:
                self._fire(s, t, f"risk stop: live DD {live_dd:.1%} vs "
                                 f"research {s.is_maxdd:.1%}")
                continue

            if not is_review:
                continue

            # rule 2 — decay (at review)
            if len(live) >= self.cfg.decay_window:
                tr = live.iloc[-self.cfg.decay_window:]
                if sharpe(tr.values) < 0:
                    self._fire(s, t, "decay: trailing-1y live Sharpe < 0")
                    continue

            # rule 3 — re-validation (at review): beat B&H on the latest
            # holdout window, else strike; two strikes → out
            w = self.cfg.holdout_days
            sl = s.net.loc[:t].iloc[-w:]
            bw = bh.iloc[-w:]
            if sharpe(sl.values) <= sharpe(bw.values):
                s.strikes += 1
                if s.strikes >= self.cfg.reval_strikes:
                    self._fire(s, t, f"re-validation: {s.strikes} consecutive "
                                     "gate failures")
            else:
                s.strikes = 0

    def _fire(self, s: Sleeve, t: pd.Timestamp, reason: str):
        s.fired_at = t
        s.fire_reason = reason
        live = s.live()
        self.journal.append({
            "date": t, "action": "FIRE", "sleeve": s.candidate.key,
            "reason": reason, "is_sharpe": round(s.is_sharpe, 2),
            "live_sharpe": round(sharpe(live.values), 2),
            "live_days": len(live),
        })

    # ── Main loop ──────────────────────────────────────────────────────────

    def run(self) -> Dict:
        months = self._monthly_dates()
        active_keys = lambda: {s.candidate.key for s in self.sleeves if s.active}

        for i, t in enumerate(tqdm(months, desc="  pod reviews", leave=False)):
            is_review = i % self.cfg.review_months == 0
            self._monitor(t, is_review)
            if not is_review:
                continue

            free = self.cfg.seats - len(active_keys())
            hires = self._research(t, review_i=i) if free > 0 else []
            for cand, is_sh in hires:
                if free <= 0:
                    break
                if cand.key in active_keys():
                    continue
                net, _ = self._net_returns(cand)
                research = net.loc[:t]
                cum = (1 + research).cumprod()
                is_dd = float(((cum - cum.cummax()) / cum.cummax()).min())
                self.sleeves.append(Sleeve(
                    candidate=cand, hired_at=t, is_sharpe=is_sh,
                    is_maxdd=min(is_dd, -0.02), net=net,
                ))
                self.journal.append({
                    "date": t, "action": "HIRE", "sleeve": cand.key,
                    "reason": f"gate passed (holdout Sharpe {is_sh:.2f})",
                    "is_sharpe": round(is_sh, 2),
                    "live_sharpe": np.nan, "live_days": 0,
                })
                free -= 1
            if is_review and not hires and free == self.cfg.seats:
                self.journal.append({
                    "date": t, "action": "PASS", "sleeve": "",
                    "reason": "no candidate passed the gate; book in cash",
                    "is_sharpe": np.nan, "live_sharpe": np.nan, "live_days": 0,
                })

        # assemble the book: each seat is 1/seats of capital; idle seats cash
        start = months[0]
        idx = self.prices.loc[start:].index
        book = pd.Series(0.0, index=idx)
        n_active = pd.Series(0, index=idx)
        for s in self.sleeves:
            live = s.live()
            book = book.add(live.reindex(idx).fillna(0.0) / self.cfg.seats,
                            fill_value=0.0)
            n_active = n_active.add(
                pd.Series(1, index=live.index).reindex(idx).fillna(0),
                fill_value=0)

        haircut = pd.DataFrame([{
            "sleeve": s.candidate.key[:60],
            "hired": s.hired_at.date(), "fired": s.fired_at.date()
            if s.fired_at is not None else "active",
            "research_sharpe": round(s.is_sharpe, 2),
            "live_sharpe": round(sharpe(s.live().values), 2),
            "live_days": len(s.live()),
            "fire_reason": s.fire_reason,
        } for s in self.sleeves])

        return {
            "book": book,
            "n_active": n_active,
            "journal": pd.DataFrame(self.journal),
            "haircut": haircut,
            "sleeves": self.sleeves,
        }
