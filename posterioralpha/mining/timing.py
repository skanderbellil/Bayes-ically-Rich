"""
Timing-dial mining: evolve exposure dials on a single underlying.

The cross-sectional miner asks "which stock?"; this one asks the question the
champion-stack thread showed actually pays on this data: "how much of the
underlying (QQQ/SPY) should I hold today?".  A candidate is a parameterized
DIAL mapping causal information (FRED macro panel, price trend, realized vol,
BOCPD regime age) to exposure e_t ∈ [0, 1], evaluated net of turnover costs
with a 1-day execution lag.

Search and validation mirror `miner.py`: evolutionary loop, fitness on
randomized purged windows redrawn each generation, untouched holdout, then
block bootstrap + circular permutation + Deflated Sharpe charged for every
candidate examined.  Buy & hold is reported alongside — a dial only matters
if it beats simply holding the underlying.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from posterioralpha.mining.evaluation import sharpe
from posterioralpha.mining.validation import (
    block_bootstrap_pvalue,
    deflated_sharpe_ratio,
    random_windows,
)
from posterioralpha.research import precompute_bocpd

logger = logging.getLogger(__name__)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


# ─────────────────────────────────────────────────────────────────────────────
# Context
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TimingContext:
    prices:  pd.Series                       # underlying adjusted close
    macro:   Optional[pd.DataFrame] = None   # FRED panel (publication-lagged)
    returns: pd.Series = field(init=False)
    _erl: Optional[pd.Series] = field(default=None, init=False)

    def __post_init__(self):
        self.returns = self.prices.pct_change()
        # macro series usable as dials: needs decent coverage over this sample
        self.macro_cols: List[str] = []
        if self.macro is not None:
            aligned = self.macro.reindex(self.prices.index)
            self.macro_cols = [c for c in aligned.columns
                               if aligned[c].notna().mean() >= 0.80]

    @property
    def erl(self) -> pd.Series:
        if self._erl is None:
            r = self.returns.dropna()
            _, erl = precompute_bocpd(r.values)
            self._erl = (pd.Series(erl, index=r.index)
                         .reindex(self.prices.index).ffill().fillna(0.0))
        return self._erl

    def macro_z(self, column: str, window: int) -> pd.Series:
        m = self.macro[column].reindex(self.prices.index).ffill()
        mu = m.rolling(window, min_periods=window // 2).mean()
        sd = m.rolling(window, min_periods=window // 2).std()
        return (m - mu) / (sd + 1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# Dial families  (each → exposure series in [0, 1])
# ─────────────────────────────────────────────────────────────────────────────

def macro_dial(ctx, series: str, z_window: int, direction: int, steep: float):
    """Sigmoid of a macro z-score; `direction` decides which tail is risk-on."""
    return _sigmoid(direction * steep * ctx.macro_z(series, z_window))


def trend_dial(ctx, window: int, steep: float):
    """Price vs its moving average, soft-thresholded (continuous Gayed-style)."""
    ma = ctx.prices.rolling(window).mean()
    return _sigmoid(steep * (ctx.prices / ma - 1.0))


def mom_dial(ctx, lookback: int, steep: float):
    """Trailing return of the underlying, soft-signed."""
    return _sigmoid(steep * (ctx.prices / ctx.prices.shift(lookback) - 1.0))


def vol_dial(ctx, window: int, z_window: int, direction: int, steep: float):
    """Realized-vol z-score dial (direction −1 = de-risk when vol is high)."""
    vol = ctx.returns.rolling(window).std()
    mu = vol.rolling(z_window, min_periods=z_window // 2).mean()
    sd = vol.rolling(z_window, min_periods=z_window // 2).std()
    return _sigmoid(direction * steep * (vol - mu) / (sd + 1e-9))


def erl_dial(ctx, thresh: float):
    """BOCPD regime age: young regime (recent changepoint) → low exposure."""
    return _sigmoid((ctx.erl - thresh) / (thresh / 4.0 + 1e-9))


def macro_vote2(ctx, series_a: str, series_b: str, z_window: int,
                dir_a: int, dir_b: int, steep: float):
    """
    Two-layer macro vote (champion-stack architecture): the average of two
    independent sigmoid dials.  Diversified layers, continuous mapping —
    the construction the liquidity thread found robust.
    """
    ea = _sigmoid(dir_a * steep * ctx.macro_z(series_a, z_window))
    eb = _sigmoid(dir_b * steep * ctx.macro_z(series_b, z_window))
    return 0.5 * (ea + eb)


def macro_trend_vote(ctx, series: str, z_window: int, direction: int,
                     steep: float, trend_window: int, trend_steep: float):
    """Macro layer × price-trend layer, averaged — liquidity × trend vote."""
    em = _sigmoid(direction * steep * ctx.macro_z(series, z_window))
    ma = ctx.prices.rolling(trend_window).mean()
    et = _sigmoid(trend_steep * (ctx.prices / ma - 1.0))
    return 0.5 * (em + et)


# spec kinds: ("int"/"float", lo, hi) handled as in signals; ("cat", choices)
DIAL_FAMILIES: Dict[str, Tuple] = {
    "macro_dial": (macro_dial, {
        "series":    ("cat", "MACRO_COLS"),     # resolved against the context
        "z_window":  (40, 252, "int"),
        "direction": ("cat", [-1, 1]),
        "steep":     (0.5, 3.0, "float"),
    }),
    "trend_dial": (trend_dial, {
        "window": (20, 252, "int"),
        "steep":  (5.0, 60.0, "float"),
    }),
    "mom_dial": (mom_dial, {
        "lookback": (20, 252, "int"),
        "steep":    (5.0, 60.0, "float"),
    }),
    "vol_dial": (vol_dial, {
        "window":    (10, 126, "int"),
        "z_window":  (40, 252, "int"),
        "direction": ("cat", [-1, 1]),
        "steep":     (0.5, 3.0, "float"),
    }),
    "erl_dial": (erl_dial, {
        "thresh": (20.0, 150.0, "float"),
    }),
    "macro_vote2": (macro_vote2, {
        "series_a":  ("cat", "MACRO_COLS"),
        "series_b":  ("cat", "MACRO_COLS"),
        "z_window":  (40, 252, "int"),
        "dir_a":     ("cat", [-1, 1]),
        "dir_b":     ("cat", [-1, 1]),
        "steep":     (0.5, 3.0, "float"),
    }),
    "macro_trend_vote": (macro_trend_vote, {
        "series":       ("cat", "MACRO_COLS"),
        "z_window":     (40, 252, "int"),
        "direction":    ("cat", [-1, 1]),
        "steep":        (0.5, 3.0, "float"),
        "trend_window": (20, 252, "int"),
        "trend_steep":  (5.0, 60.0, "float"),
    }),
}

MACRO_DIAL_FAMILIES = frozenset({"macro_dial", "macro_vote2", "macro_trend_vote"})


def _sample(spec, ctx: TimingContext, rng: np.random.Generator):
    if spec[0] == "cat":
        choices = ctx.macro_cols if spec[1] == "MACRO_COLS" else spec[1]
        return choices[int(rng.integers(len(choices)))]
    lo, hi, kind = spec
    if kind == "int":
        v = int(round(np.exp(rng.uniform(np.log(lo + 1), np.log(hi + 1))) - 1))
        return int(np.clip(v, lo, hi))
    return float(rng.uniform(lo, hi))


def _mutate_value(value, spec, ctx: TimingContext, rng: np.random.Generator):
    if spec[0] == "cat":
        return _sample(spec, ctx, rng)
    lo, hi, kind = spec
    v = (value + 1) * np.exp(rng.normal(0, 0.3)) - 1 if kind == "int" \
        else value * np.exp(rng.normal(0, 0.3))
    return int(np.clip(round(v), lo, hi)) if kind == "int" \
        else float(np.clip(v, lo, hi))


# ─────────────────────────────────────────────────────────────────────────────
# Candidate genome
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DialCandidate:
    family: str
    params: Dict
    smooth: int = 1

    fitness:   float = -np.inf
    is_sharpe: float = 0.0

    @property
    def key(self) -> str:
        p = ",".join(f"{k}={round(v, 3) if isinstance(v, float) else v}"
                     for k, v in sorted(self.params.items()))
        return f"{self.family}({p})|sm={self.smooth}"

    def describe(self) -> str:
        return self.key


def _random_dial(ctx: TimingContext, rng: np.random.Generator,
                 families: List[str]) -> DialCandidate:
    family = str(rng.choice(families))
    _, space = DIAL_FAMILIES[family]
    params = {k: _sample(spec, ctx, rng) for k, spec in space.items()}
    return DialCandidate(family, params, smooth=int(rng.integers(1, 11)))


def _mutate_dial(c: DialCandidate, ctx: TimingContext,
                 rng: np.random.Generator) -> DialCandidate:
    _, space = DIAL_FAMILIES[c.family]
    params = {k: (_mutate_value(v, space[k], ctx, rng) if rng.random() < 0.5 else v)
              for k, v in c.params.items()}
    smooth = int(np.clip(round(c.smooth * np.exp(rng.normal(0, 0.3))), 1, 10))
    return DialCandidate(c.family, params, smooth)


def _crossover_dial(a: DialCandidate, b: DialCandidate, ctx: TimingContext,
                    rng: np.random.Generator) -> DialCandidate:
    if a.family != b.family:
        return _mutate_dial(a if rng.random() < 0.5 else b, ctx, rng)
    params = {k: (a.params[k] if rng.random() < 0.5 else b.params[k])
              for k in a.params}
    return DialCandidate(a.family, params,
                         a.smooth if rng.random() < 0.5 else b.smooth)


# ─────────────────────────────────────────────────────────────────────────────
# Miner
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TimingConfig:
    population:     int   = 40
    generations:    int   = 6
    elite_frac:     float = 0.15
    immigrant_frac: float = 0.20
    n_val_windows:  int   = 8
    val_window_len: int   = 504    # 2 years: timing edges need longer windows
    holdout_frac:   float = 0.30
    purge_gap:      int   = 21
    tc:             float = 0.0002  # 2 bps — liquid index ETF, matches thread
    n_finalists:    int   = 10
    max_per_family: int   = 3
    n_boot:         int   = 400
    n_perm:         int   = 200
    seed:           int   = 7


class TimingMiner:
    def __init__(self, prices: pd.Series, config: Optional[TimingConfig] = None,
                 macro: Optional[pd.DataFrame] = None,
                 families: Optional[List[str]] = None):
        self.cfg = config or TimingConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.ctx = TimingContext(prices=prices, macro=macro)
        allowed = [f for f in DIAL_FAMILIES
                   if self.ctx.macro_cols or f not in MACRO_DIAL_FAMILIES]
        self.families = [f for f in (families or allowed) if f in allowed]

        T = len(prices)
        self.holdout_start = int(T * (1.0 - self.cfg.holdout_frac))
        self.is_end = self.holdout_start - self.cfg.purge_gap

        self.trials: Dict[str, float] = {}
        self._cache: Dict[str, Tuple[pd.Series, pd.Series]] = {}
        self.history: List[Dict] = []

    # ── Evaluation ─────────────────────────────────────────────────────────

    def _exposure_and_returns(self, c: DialCandidate) -> Tuple[pd.Series, pd.Series]:
        if c.key not in self._cache:
            func, _ = DIAL_FAMILIES[c.family]
            e = func(self.ctx, **c.params).clip(0.0, 1.0)
            if c.smooth > 1:
                e = e.ewm(span=c.smooth, adjust=False).mean()
            held = e.shift(1).fillna(0.0)
            turnover = held.diff().abs().fillna(0.0)
            net = held * self.ctx.returns.fillna(0.0) - self.cfg.tc * turnover
            if len(self._cache) > 600:
                self._cache.pop(next(iter(self._cache)))
            self._cache[c.key] = (e, net)
        return self._cache[c.key]

    def _evaluate(self, c: DialCandidate, windows) -> DialCandidate:
        e, net = self._exposure_and_returns(c)
        is_net = net.iloc[: self.is_end]

        # a dial that never moves is just (scaled) buy & hold — disqualify
        if float(e.iloc[260: self.is_end].std()) < 0.005:
            c.fitness, c.is_sharpe = -10.0, 0.0
            return c

        ws = np.array([sharpe(is_net.iloc[s:end].values) for s, end in windows])
        c.fitness = float(ws.mean() - 0.5 * ws.std())
        c.is_sharpe = sharpe(is_net.values)

        r = is_net.values
        r = r[~np.isnan(r)]
        if c.key not in self.trials and len(r) > 60 and r.std() > 0:
            self.trials[c.key] = float(r.mean() / r.std())
        return c

    # ── Evolution ──────────────────────────────────────────────────────────

    def evolve(self) -> List[DialCandidate]:
        cfg = self.cfg
        pop = [_random_dial(self.ctx, self.rng, self.families)
               for _ in range(cfg.population)]
        n_elite = max(1, int(cfg.population * cfg.elite_frac))
        n_immig = max(1, int(cfg.population * cfg.immigrant_frac))
        best: Dict[str, DialCandidate] = {}

        for gen in range(cfg.generations):
            windows = random_windows(self.is_end, cfg.n_val_windows,
                                     cfg.val_window_len, self.rng, min_start=260)
            for c in tqdm(pop, desc=f"  gen {gen + 1}/{cfg.generations}", leave=False):
                self._evaluate(c, windows)
                if c.key not in best or c.fitness > best[c.key].fitness:
                    best[c.key] = c

            pop.sort(key=lambda c: c.fitness, reverse=True)
            self.history.append({
                "generation": gen + 1,
                "best_fitness": pop[0].fitness,
                "best_is_sharpe": pop[0].is_sharpe,
                "best_key": pop[0].key,
                "n_trials_total": len(self.trials),
            })
            logger.info("gen %d | best fitness %.2f | IS Sharpe %.2f | %s",
                        gen + 1, pop[0].fitness, pop[0].is_sharpe, pop[0].key)
            if gen == cfg.generations - 1:
                break

            elites = pop[:n_elite]
            children, seen = [], {e.key for e in elites}
            while len(children) < cfg.population - n_elite - n_immig:
                a = max((pop[int(i)] for i in self.rng.integers(0, len(pop), 3)),
                        key=lambda c: c.fitness)
                b = max((pop[int(i)] for i in self.rng.integers(0, len(pop), 3)),
                        key=lambda c: c.fitness)
                child = _crossover_dial(a, b, self.ctx, self.rng)
                if self.rng.random() < 0.8:
                    child = _mutate_dial(child, self.ctx, self.rng)
                if child.key not in seen:
                    seen.add(child.key)
                    children.append(child)
            immigrants = [_random_dial(self.ctx, self.rng, self.families)
                          for _ in range(n_immig)]
            pop = elites + children + immigrants

        ranked = sorted(best.values(), key=lambda c: c.fitness, reverse=True)
        finalists, per_family = [], {}
        for c in ranked:
            if per_family.get(c.family, 0) >= cfg.max_per_family:
                continue
            per_family[c.family] = per_family.get(c.family, 0) + 1
            finalists.append(c)
            if len(finalists) >= cfg.n_finalists:
                break
        return finalists

    # ── Holdout gauntlet ───────────────────────────────────────────────────

    def _permutation_pvalue(self, exposure: pd.Series, n_perm: int,
                            min_shift: int = 60) -> float:
        """Circularly shift the exposure against the real underlying returns."""
        e = exposure.values
        r = np.nan_to_num(self.ctx.returns.iloc[self.holdout_start:].values, nan=0.0)
        T = len(e)
        if T < 2 * min_shift:
            return 1.0

        def net_sharpe(ev):
            held = np.concatenate([[0.0], ev[:-1]])
            turn = np.abs(np.diff(held, prepend=0.0))
            return sharpe(held * r - self.cfg.tc * turn)

        observed = net_sharpe(e)
        shifts = self.rng.integers(min_shift, T - min_shift, size=n_perm)
        null = np.array([net_sharpe(np.roll(e, int(s))) for s in shifts])
        return float((np.sum(null >= observed) + 1) / (n_perm + 1))

    def gauntlet(self, finalists: List[DialCandidate]) -> pd.DataFrame:
        n_trials = max(len(self.trials), 1)
        trial_var = float(np.var(list(self.trials.values()))) if self.trials else 1e-6
        bh_sharpe = sharpe(self.ctx.returns.iloc[self.holdout_start:].values)

        rows = []
        for c in tqdm(finalists, desc="  holdout gauntlet", leave=False):
            e, net = self._exposure_and_returns(c)
            ho_net = net.iloc[self.holdout_start:]
            ho_sharpe = sharpe(ho_net.values)

            boot_p, ci_lo, ci_hi = block_bootstrap_pvalue(
                ho_net.values, self.rng, n_boot=self.cfg.n_boot)
            perm_p = self._permutation_pvalue(
                e.iloc[self.holdout_start:], self.cfg.n_perm)
            dsr = deflated_sharpe_ratio(ho_net.values, n_trials, trial_var)

            validated = (ho_sharpe > bh_sharpe and boot_p < 0.10
                         and perm_p < 0.10 and dsr > 0.50)
            weak = (ho_sharpe > 0
                    and (boot_p < 0.25 or perm_p < 0.25))

            rows.append({
                "dial":          c.describe(),
                "family":        c.family,
                "fitness":       round(c.fitness, 3),
                "is_sharpe":     round(c.is_sharpe, 3),
                "holdout_sharpe": round(ho_sharpe, 3),
                "bh_sharpe":     round(bh_sharpe, 3),
                "boot_p":        round(boot_p, 4),
                "boot_ci":       f"[{ci_lo:.2f}, {ci_hi:.2f}]",
                "perm_p":        round(perm_p, 4),
                "dsr":           round(dsr, 4),
                "verdict":       ("VALIDATED" if validated
                                  else "WEAK" if weak else "REJECTED"),
            })

        df = pd.DataFrame(rows).sort_values("holdout_sharpe", ascending=False)
        df.attrs["n_trials"] = n_trials
        df.attrs["bh_sharpe"] = bh_sharpe
        return df.reset_index(drop=True)

    # ── Fresh-sample gate ──────────────────────────────────────────────────

    def fresh_gauntlet(self, finalists: List[DialCandidate],
                       fresh_prices: pd.Series,
                       leaderboard: pd.DataFrame) -> pd.DataFrame:
        """
        The gate the within-sample gauntlet cannot provide: re-run each
        finalist's dial on price history from BEFORE the search sample
        (data no part of the selection ever saw) and compare against buy &
        hold there.  The first QQQ run showed why this is mandatory — five
        seeds converged on "falling rates = risk-on" dials that validated
        on 2010→ data and then lost to B&H across 1999–2009, because pre-QE
        the Fed eased *into* recessions.

        Final verdicts:
          PROMOTED    — VALIDATED on the holdout AND beats B&H fresh
          ARTIFACT    — VALIDATED on the holdout, loses to B&H fresh
          VALIDATED*  — VALIDATED, but the dial's inputs don't exist in the
                        fresh sample (insufficient coverage) — unresolved
          (others unchanged)
        """
        fctx = TimingContext(prices=fresh_prices, macro=self.ctx.macro)

        # dial Sharpe and B&H Sharpe are compared over the SAME span: from the
        # dial's first valid value — otherwise a dial whose inputs start late
        # (e.g. DFII10 from 2003) silently skips an early crash and gets
        # credit for sitting in cash through it
        fresh_sr: Dict[str, Tuple[float, float]] = {}
        for c in finalists:
            func, _ = DIAL_FAMILIES[c.family]
            try:
                e = func(fctx, **c.params).clip(0.0, 1.0)
            except Exception:
                continue
            valid = e.dropna()
            if len(valid) < 500:
                continue                      # inputs not available pre-sample
            start = valid.index[0]
            e = e.loc[start:]
            if c.smooth > 1:
                e = e.ewm(span=c.smooth, adjust=False).mean()
            rets = fctx.returns.loc[start:]
            held = e.shift(1).fillna(0.0)
            turnover = held.diff().abs().fillna(0.0)
            net = held * rets.fillna(0.0) - self.cfg.tc * turnover
            fresh_sr[c.key] = (sharpe(net.values), sharpe(rets.values))

        lb = leaderboard.copy()
        lb["fresh_sharpe"] = lb["dial"].map(
            lambda k: round(fresh_sr[k][0], 3) if k in fresh_sr else np.nan)
        lb["fresh_bh"] = lb["dial"].map(
            lambda k: round(fresh_sr[k][1], 3) if k in fresh_sr else np.nan)

        def final(row):
            if row["verdict"] != "VALIDATED":
                return row["verdict"]
            if pd.isna(row["fresh_sharpe"]):
                return "VALIDATED*"
            return ("PROMOTED" if row["fresh_sharpe"] > row["fresh_bh"]
                    else "ARTIFACT")

        lb["final"] = lb.apply(final, axis=1)
        lb.attrs.update(leaderboard.attrs)
        lb.attrs["fresh_span"] = (str(fresh_prices.index[0].date()),
                                  str(fresh_prices.index[-1].date()))
        return lb

    def run(self) -> Tuple[pd.DataFrame, List[DialCandidate]]:
        finalists = self.evolve()
        return self.gauntlet(finalists), finalists
