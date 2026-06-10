"""
Evolutionary alpha miner.

The loop automates the research cycle:

  generate candidates → backtest in-sample → score fitness on RANDOMIZED
  purged windows → select / crossover / mutate → repeat → final gauntlet
  on an untouched holdout (bootstrap + permutation + deflated Sharpe).

The holdout (the most recent `holdout_frac` of history, separated by a purge
gap) is never touched during evolution; only the surviving finalists are
ever evaluated on it, and the Deflated Sharpe Ratio is charged for every
unique candidate the search examined along the way.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from posterioralpha.mining.evaluation import portfolio_returns, scores_to_weights, sharpe
from posterioralpha.mining.signals import (
    FAMILIES,
    SignalContext,
    compute_scores,
    mutate_param,
    sample_param,
)
from posterioralpha.mining.validation import (
    block_bootstrap_pvalue,
    deflated_sharpe_ratio,
    permutation_pvalue,
    random_windows,
)

logger = logging.getLogger(__name__)

_WEIGHTINGS = ("rank", "quantile")


# ─────────────────────────────────────────────────────────────────────────────
# Candidate genome
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Candidate:
    family:    str
    params:    Dict
    weighting: str   = "rank"
    top_frac:  float = 0.2
    smooth:    int   = 1

    # filled by evaluation
    fitness:        float = -np.inf
    window_sharpes: List[float] = field(default_factory=list)
    is_sharpe:      float = 0.0    # full in-sample annualised Sharpe

    @property
    def key(self) -> str:
        p = ",".join(f"{k}={round(v, 3)}" for k, v in sorted(self.params.items()))
        # top_frac only matters for quantile weighting — keep keys canonical
        tf = f"|tf={self.top_frac:.2f}" if self.weighting == "quantile" else ""
        return f"{self.family}({p})|{self.weighting}{tf}|sm={self.smooth}"

    def describe(self) -> str:
        return self.key


def random_candidate(rng: np.random.Generator) -> Candidate:
    family = str(rng.choice(list(FAMILIES.keys())))
    _, space = FAMILIES[family]
    params = {name: sample_param(spec, rng) for name, spec in space.items()}
    return Candidate(
        family=family,
        params=params,
        weighting=str(rng.choice(_WEIGHTINGS)),
        top_frac=float(rng.uniform(0.10, 0.40)),
        smooth=int(rng.integers(1, 11)),
    )


def mutate(c: Candidate, rng: np.random.Generator) -> Candidate:
    _, space = FAMILIES[c.family]
    params = dict(c.params)
    for name, spec in space.items():
        if rng.random() < 0.5:
            params[name] = mutate_param(params[name], spec, rng)
    weighting = c.weighting if rng.random() > 0.2 else str(rng.choice(_WEIGHTINGS))
    top_frac  = float(np.clip(c.top_frac * np.exp(rng.normal(0, 0.2)), 0.10, 0.40))
    smooth    = int(np.clip(round(c.smooth * np.exp(rng.normal(0, 0.3))), 1, 10))
    return Candidate(c.family, params, weighting, top_frac, smooth)


def crossover(a: Candidate, b: Candidate, rng: np.random.Generator) -> Candidate:
    """Parameter-wise mix of two parents from the same family."""
    if a.family != b.family:
        return mutate(a if rng.random() < 0.5 else b, rng)
    params = {k: (a.params[k] if rng.random() < 0.5 else b.params[k])
              for k in a.params}
    pick = lambda x, y: x if rng.random() < 0.5 else y
    return Candidate(a.family, params, pick(a.weighting, b.weighting),
                     pick(a.top_frac, b.top_frac), pick(a.smooth, b.smooth))


# ─────────────────────────────────────────────────────────────────────────────
# Miner
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MinerConfig:
    population:     int   = 40
    generations:    int   = 6
    elite_frac:     float = 0.15   # carried over unchanged
    immigrant_frac: float = 0.20   # fresh random blood each generation
    n_val_windows:  int   = 8      # randomized purged windows per generation
    val_window_len: int   = 252    # 1 year each
    holdout_frac:   float = 0.30   # most recent 30% — untouched until the end
    purge_gap:      int   = 21     # days between in-sample and holdout
    tc:             float = 0.001  # 10 bps per unit one-way turnover
    n_finalists:    int   = 10
    n_boot:         int   = 400
    n_perm:         int   = 200
    seed:           int   = 7


class AlphaMiner:
    def __init__(self, prices: pd.DataFrame, config: Optional[MinerConfig] = None):
        self.cfg = config or MinerConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

        returns = prices.pct_change(fill_method=None)
        self.ctx = SignalContext(prices=prices, returns=returns)

        T = len(prices)
        self.holdout_start = int(T * (1.0 - self.cfg.holdout_frac))
        self.is_end = self.holdout_start - self.cfg.purge_gap

        # registry of every unique candidate examined (for the DSR penalty)
        self.trials: Dict[str, float] = {}      # key → daily in-sample Sharpe
        self._cache: Dict[str, Tuple[pd.DataFrame, pd.Series]] = {}
        self.history: List[Dict] = []

    # ── Evaluation ─────────────────────────────────────────────────────────

    def _weights_and_returns(self, c: Candidate) -> Tuple[pd.DataFrame, pd.Series]:
        """Full-period weights and net daily returns (cached per genome)."""
        if c.key not in self._cache:
            scores = compute_scores(self.ctx, c.family, c.params)
            weights = scores_to_weights(scores, c.weighting, c.top_frac, c.smooth)
            net, _ = portfolio_returns(weights, self.ctx.returns, tc=self.cfg.tc)
            if len(self._cache) > 600:           # bound memory
                self._cache.pop(next(iter(self._cache)))
            self._cache[c.key] = (weights, net)
        return self._cache[c.key]

    def _evaluate(self, c: Candidate, windows: List[Tuple[int, int]]) -> Candidate:
        weights, net = self._weights_and_returns(c)
        is_net = net.iloc[: self.is_end]

        # disqualify degenerate (mostly inactive) portfolios so that a
        # do-nothing alpha can't outrank noisy-but-real candidates
        mean_gross = float(weights.iloc[260: self.is_end].abs().sum(axis=1).mean())
        if mean_gross < 0.5:
            c.fitness, c.is_sharpe, c.window_sharpes = -10.0, 0.0, []
            return c

        c.window_sharpes = [sharpe(is_net.iloc[s:e].values) for s, e in windows]
        ws = np.array(c.window_sharpes)
        # robustness-first fitness: penalise dispersion across random windows
        c.fitness = float(ws.mean() - 0.5 * ws.std())
        c.is_sharpe = sharpe(is_net.values)

        # register the trial (daily Sharpe) for the DSR multiple-testing charge
        r = is_net.values
        r = r[~np.isnan(r)]
        if c.key not in self.trials and len(r) > 60 and r.std() > 0:
            self.trials[c.key] = float(r.mean() / r.std())
        return c

    # ── Evolution loop ─────────────────────────────────────────────────────

    def evolve(self) -> List[Candidate]:
        cfg = self.cfg
        pop = [random_candidate(self.rng) for _ in range(cfg.population)]
        n_elite = max(1, int(cfg.population * cfg.elite_frac))
        n_immig = max(1, int(cfg.population * cfg.immigrant_frac))

        best: Dict[str, Candidate] = {}

        for gen in range(cfg.generations):
            # fresh randomized validation windows every generation
            windows = random_windows(
                self.is_end, cfg.n_val_windows, cfg.val_window_len, self.rng,
                min_start=260,   # leave room for the longest lookbacks
            )

            for c in tqdm(pop, desc=f"  gen {gen + 1}/{cfg.generations}", leave=False):
                self._evaluate(c, windows)
                if c.key not in best or c.fitness > best[c.key].fitness:
                    best[c.key] = c

            pop.sort(key=lambda c: c.fitness, reverse=True)
            top = pop[0]
            self.history.append({
                "generation": gen + 1,
                "best_fitness": top.fitness,
                "best_is_sharpe": top.is_sharpe,
                "best_key": top.key,
                "mean_fitness": float(np.mean([c.fitness for c in pop])),
                "n_trials_total": len(self.trials),
            })
            logger.info(
                "gen %d | best fitness %.2f | IS Sharpe %.2f | %s",
                gen + 1, top.fitness, top.is_sharpe, top.key,
            )

            if gen == cfg.generations - 1:
                break

            # next generation: elites + offspring + immigrants
            elites = pop[:n_elite]
            children: List[Candidate] = []
            seen = {e.key for e in elites}
            while len(children) < cfg.population - n_elite - n_immig:
                a, b = self._tournament(pop), self._tournament(pop)
                child = crossover(a, b, self.rng)
                if self.rng.random() < 0.8:
                    child = mutate(child, self.rng)
                if child.key not in seen:
                    seen.add(child.key)
                    children.append(child)
            immigrants = [random_candidate(self.rng) for _ in range(n_immig)]
            pop = elites + children + immigrants

        ranked = sorted(best.values(), key=lambda c: c.fitness, reverse=True)
        return ranked[: cfg.n_finalists]

    def _tournament(self, pop: List[Candidate], k: int = 3) -> Candidate:
        picks = [pop[int(i)] for i in self.rng.integers(0, len(pop), size=k)]
        return max(picks, key=lambda c: c.fitness)

    # ── Holdout gauntlet ───────────────────────────────────────────────────

    def gauntlet(self, finalists: List[Candidate]) -> pd.DataFrame:
        """Evaluate finalists on the untouched holdout with randomized tests."""
        n_trials = max(len(self.trials), 1)
        trial_var = float(np.var(list(self.trials.values()))) if self.trials else 1e-6

        rows = []
        for c in tqdm(finalists, desc="  holdout gauntlet", leave=False):
            weights, net = self._weights_and_returns(c)
            ho_net = net.iloc[self.holdout_start:]
            ho_w   = weights.iloc[self.holdout_start:]
            ho_ret = self.ctx.returns.iloc[self.holdout_start:]

            ho_sharpe = sharpe(ho_net.values)
            boot_p, ci_lo, ci_hi = block_bootstrap_pvalue(
                ho_net.values, self.rng, n_boot=self.cfg.n_boot)
            perm_p = permutation_pvalue(
                ho_w, ho_ret, self.rng, n_perm=self.cfg.n_perm, tc=self.cfg.tc)
            dsr = deflated_sharpe_ratio(ho_net.values, n_trials, trial_var)

            validated = (ho_sharpe > 0 and boot_p < 0.10
                         and perm_p < 0.10 and dsr > 0.50)
            weak = ho_sharpe > 0 and (boot_p < 0.25 or perm_p < 0.25)

            rows.append({
                "alpha":        c.describe(),
                "family":       c.family,
                "fitness":      round(c.fitness, 3),
                "is_sharpe":    round(c.is_sharpe, 3),
                "holdout_sharpe": round(ho_sharpe, 3),
                "boot_p":       round(boot_p, 4),
                "boot_ci":      f"[{ci_lo:.2f}, {ci_hi:.2f}]",
                "perm_p":       round(perm_p, 4),
                "dsr":          round(dsr, 4),
                "verdict":      ("VALIDATED" if validated
                                 else "WEAK" if weak else "REJECTED"),
            })

        df = pd.DataFrame(rows).sort_values("holdout_sharpe", ascending=False)
        df.attrs["n_trials"] = n_trials
        return df.reset_index(drop=True)

    # ── One-call entry point ───────────────────────────────────────────────

    def run(self) -> Tuple[pd.DataFrame, List[Candidate]]:
        finalists = self.evolve()
        leaderboard = self.gauntlet(finalists)
        return leaderboard, finalists
