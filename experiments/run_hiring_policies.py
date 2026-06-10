#!/usr/bin/env python3
"""
Pod hiring policies — does HOW you pick among gate-passers matter?

The first pod run (run_quant_pod.py) measured a *negative* correlation
between research Sharpe at hire and realized live Sharpe (winner's curse:
the harder a candidate won the gate, the more of its win was selection
luck). This experiment asks whether the pod can exploit that, by changing
only the ORDER in which gate-passers fill free seats at each review:

  raw     — highest holdout (research) Sharpe first. The status quo; bitwise
            identical to run_quant_pod.py's behaviour.
  dsr     — highest Deflated Sharpe Ratio first: theory-driven shrinkage
            that charges each candidate for every trial the search examined.
            No meta-fitting.
  shrunk  — at each review, regress live-vs-research Sharpe on the pod's
            OWN completed sleeves so far, then order by *predicted* live
            Sharpe. If the learned slope is negative the ranking flips —
            learned, not assumed. Falls back to raw until enough sleeves
            have completed (>= 6).
  median  — anti-winner's-curse heuristic: fill seats starting from the
            MEDIAN-ranked passer, expanding outward (lower side first),
            so the top-ranked candidate is hired last.
  inverse — LOWEST research Sharpe first: the pure exploit of the negative
            research->live slope. ⚠️ hard-codes a fact learned from this
            sample; the walk-forward-honest versions are shrunk/veto.
  veto    — shrunk with teeth: same learned predicted-live ranking, but a
            passer with predicted live Sharpe <= 0 is NOT hired at all
            (an idle seat earns cash = 0). Unlike pure reordering, the
            veto can act even when a review has a single passer.

Everything else — the gate (holdout Sharpe > B&H + bootstrap + permutation
support), the seats, the three retirement rules, the costs — is unchanged.
The miner is deterministic per review (seed = pod seed + review index) and
its output does not depend on the book, so research results are cached and
shared across policies: four policies cost about one pod run.

Definitions used in the summary table
-------------------------------------
  mean_live_sharpe — mean realized Sharpe while deployed, across hired
                     sleeves with >= 60 live days (completed or active;
                     a 1-quarter-old hire's Sharpe is meaningless noise).
  haircut_corr     — Pearson corr(research Sharpe at hire, live Sharpe)
                     across the same sleeves: the winner's-curse measure.

Usage
-----
  python experiments/run_hiring_policies.py                # QQQ, ~30-45 min
  python experiments/run_hiring_policies.py --underlying SPY --seed 11
  python experiments/run_hiring_policies.py --n-boot 0     # skip bootstrap
"""
import _bootstrap  # noqa: F401  (adds repo root to sys.path, loads .env)

import argparse
import logging
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from posterioralpha.data import load_etf_universe_prices, load_fred_macro
from posterioralpha.mining.evaluation import sharpe
from posterioralpha.mining.pod import PodConfig, QuantPod
from posterioralpha.mining.timing import TimingConfig, TimingMiner

logging.basicConfig(level=logging.WARNING, format="%(message)s")

OUT_DIR = _bootstrap.ROOT / "results" / "pod_policies"

POLICIES = ["raw", "dsr", "shrunk", "median", "inverse", "veto"]
MIN_LIVE_DAYS = 21      # floor for the 'shrunk' fit (completed sleeves only)
ANALYSIS_LIVE_DAYS = 60  # floor for REPORTED sleeve stats: younger live
                         # records are annualized noise (47 days -> "5.28")
MIN_FIT_SLEEVES = 6     # 'shrunk' needs this many completed sleeves to fit


# ─────────────────────────────────────────────────────────────────────────────
# Pod with a pluggable hire-ordering policy + shared research cache
# ─────────────────────────────────────────────────────────────────────────────

class PolicyPod(QuantPod):
    """QuantPod whose only difference is the ORDER gate-passers are hired in.

    `research_cache` maps review index -> list of gate-passer dicts
    ({cand, holdout_sharpe, dsr}). The miner is deterministic given
    (slice end date, seed + review index) and independent of the book, so
    the cache is exactly equivalent to re-running it — and shareable
    across policies.
    """

    def __init__(self, prices, macro=None, config=None, families=None,
                 policy: str = "raw",
                 research_cache: Optional[Dict[int, List[dict]]] = None):
        super().__init__(prices, macro=macro, config=config, families=families)
        assert policy in POLICIES, policy
        self.policy = policy
        self.research_cache = research_cache if research_cache is not None else {}

    # ── gate-passers per review (cached, policy-independent) ───────────────

    def _passers(self, t: pd.Timestamp, review_i: int) -> List[dict]:
        if review_i in self.research_cache:
            return self.research_cache[review_i]

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

        passers = []
        for _, row in leaderboard.iterrows():
            gate = (row["holdout_sharpe"] > row["bh_sharpe"]
                    and row["perm_p"] <= self.cfg.gate_perm_p
                    and row["boot_p"] <= self.cfg.gate_boot_p)
            if gate and row["dial"] in by_key:
                passers.append({
                    "cand": by_key[row["dial"]],
                    "holdout_sharpe": float(row["holdout_sharpe"]),
                    "dsr": float(row["dsr"]),
                })
        self.research_cache[review_i] = passers
        return passers

    # ── policy orderings ────────────────────────────────────────────────────

    def _completed_pairs(self) -> Tuple[np.ndarray, np.ndarray]:
        """(research, live) Sharpe of this pod's completed sleeves so far."""
        pairs = [(s.is_sharpe, sharpe(s.live().values)) for s in self.sleeves
                 if not s.active and len(s.live()) >= MIN_LIVE_DAYS]
        if not pairs:
            return np.array([]), np.array([])
        a = np.array(pairs)
        return a[:, 0], a[:, 1]

    def _order(self, passers: List[dict]) -> List[dict]:
        # veto filters before the trivial-length shortcut: it must be able
        # to reject even a lone passer (reordering never can)
        if self.policy == "veto":
            res, live = self._completed_pairs()
            if len(res) >= MIN_FIT_SLEEVES and res.std() > 1e-9:
                b, a = np.polyfit(res, live, 1)
                keep = [p for p in passers
                        if a + b * p["holdout_sharpe"] > 0.0]
                return sorted(keep,
                              key=lambda p: -(a + b * p["holdout_sharpe"]))
            return sorted(passers, key=lambda p: -p["holdout_sharpe"])

        if len(passers) <= 1:
            return passers

        if self.policy == "raw":
            return sorted(passers, key=lambda p: -p["holdout_sharpe"])

        if self.policy == "dsr":
            return sorted(passers, key=lambda p: -p["dsr"])

        if self.policy == "inverse":
            return sorted(passers, key=lambda p: p["holdout_sharpe"])

        if self.policy == "shrunk":
            res, live = self._completed_pairs()
            if len(res) >= MIN_FIT_SLEEVES and res.std() > 1e-9:
                b, a = np.polyfit(res, live, 1)
                return sorted(passers,
                              key=lambda p: -(a + b * p["holdout_sharpe"]))
            return sorted(passers, key=lambda p: -p["holdout_sharpe"])

        # median: start at the lower-median rank, expand outward, lower
        # side first — the top-ranked passer is hired last
        srt = sorted(passers, key=lambda p: p["holdout_sharpe"])  # ascending
        m = (len(srt) - 1) // 2
        order = []
        for off in range(len(srt)):
            for i in ((m - off, m + off) if off else (m,)):
                if 0 <= i < len(srt) and srt[i] not in order:
                    order.append(srt[i])
        return order

    # ── plug into the base run loop ─────────────────────────────────────────

    def _research(self, t: pd.Timestamp, review_i: int):
        ordered = self._order(self._passers(t, review_i))
        return [(p["cand"], p["holdout_sharpe"]) for p in ordered]


# ─────────────────────────────────────────────────────────────────────────────
# Analysis helpers
# ─────────────────────────────────────────────────────────────────────────────

def metrics(r: pd.Series) -> dict:
    r = r.dropna()
    cum = (1 + r).cumprod()
    dd = float(((cum - cum.cummax()) / cum.cummax()).min())
    cagr = float(cum.iloc[-1] ** (252 / max(len(r), 1)) - 1)
    return {"sharpe": round(sharpe(r.values), 2), "cagr": round(cagr, 4),
            "max_dd": round(dd, 4)}


def sleeve_table(res: dict) -> pd.DataFrame:
    """Hired sleeves with a meaningful live record (>= ANALYSIS_LIVE_DAYS)."""
    h = res["haircut"]
    return h[h.live_days >= ANALYSIS_LIVE_DAYS].copy()


def winner_curse_regression(done: pd.DataFrame) -> None:
    """The core empirical finding, on the raw policy's sleeves."""
    n = len(done)
    x, y = done.research_sharpe.values, done.live_sharpe.values
    lr = stats.linregress(x, y)
    print(f"\nWINNER'S CURSE — live vs research Sharpe, n={n} sleeves "
          f"(>= {ANALYSIS_LIVE_DAYS} live days)")
    print("=" * 70)
    print(f"  live = {lr.intercept:.2f} {lr.slope:+.2f} x research   "
          f"(r = {lr.rvalue:.2f}, p = {lr.pvalue:.3f}, "
          f"slope SE = {lr.stderr:.2f})")

    half = done.sort_values("hired")
    for name, part in (("first half", half.iloc[:n // 2]),
                       ("second half", half.iloc[n // 2:])):
        if len(part) >= 4 and part.research_sharpe.std() > 0:
            h = stats.linregress(part.research_sharpe, part.live_sharpe)
            print(f"  {name:<12} ({part.hired.min()} → {part.hired.max()}): "
                  f"slope {h.slope:+.2f} (r = {h.rvalue:.2f}, "
                  f"p = {h.pvalue:.2f}, n = {len(part)})")

    top_raw = done.nlargest(n // 3, 'research_sharpe').live_sharpe.mean()
    top_pred = done.nsmallest(n // 3, 'research_sharpe').live_sharpe.mean()
    print(f"  mean live Sharpe, top-third by RAW research Sharpe:       "
          f"{top_raw:.2f}")
    print(f"  mean live Sharpe, top-third by PREDICTED live Sharpe:     "
          f"{top_pred:.2f}  (= lowest research Sharpe; slope < 0)")


def raw_vs_dsr_diagnostic(cache: Dict[int, List[dict]],
                          journals: Dict[str, pd.DataFrame]) -> None:
    """Why raw and dsr coincide: ordering only matters when there are more
    passers than free seats, and dsr rarely reorders the same passers."""
    multi = {i: p for i, p in cache.items() if len(p) >= 2}
    n_top_diff = n_order_diff = 0
    for p in multi.values():
        raw_o = [d["cand"].key for d in
                 sorted(p, key=lambda d: -d["holdout_sharpe"])]
        dsr_o = [d["cand"].key for d in sorted(p, key=lambda d: -d["dsr"])]
        n_top_diff += raw_o[0] != dsr_o[0]
        n_order_diff += raw_o != dsr_o

    hires = {k: set(map(tuple, j[j.action == "HIRE"][["date", "sleeve"]]
                        .itertuples(index=False))) for k, j in journals.items()}
    print(f"\nWHY raw == dsr")
    print("=" * 70)
    print(f"  reviews mined: {len(cache)} | with >= 2 gate-passers: "
          f"{len(multi)} (ordering can only matter here)")
    print(f"  ... where dsr changes the TOP pick: {n_top_diff} | changes any "
          f"order: {n_order_diff}")
    same = hires["raw"] == hires["dsr"]
    print(f"  hired (date, sleeve) sets identical: {same} "
          f"({len(hires['raw'])} vs {len(hires['dsr'])} hires)")
    print("  DSR is monotone in holdout Sharpe within a review (same sample "
          "length, same\n  trial count/variance), so re-ranking the same "
          "passers almost never reorders them.")


def stationary_bootstrap(rets: pd.DataFrame, n_boot: int, rng,
                         mean_block: int = 63) -> pd.DataFrame:
    """Joint stationary block bootstrap (same blocks for every column), so
    policy Sharpes stay comparable draw by draw."""
    X = rets.values
    T = len(X)
    out = np.empty((n_boot, X.shape[1]))
    for b in range(n_boot):
        idx = np.empty(T, dtype=int)
        t = 0
        while t < T:
            start = rng.integers(0, T)
            for j in range(rng.geometric(1.0 / mean_block)):
                if t >= T:
                    break
                idx[t] = (start + j) % T
                t += 1
        r = X[idx]
        mu, sd = r.mean(axis=0), r.std(axis=0)
        out[b] = np.where(sd > 0, mu / sd * np.sqrt(252), 0.0)
    return pd.DataFrame(out, columns=rets.columns)


def jackknife_sleeves(res: dict, seats: int) -> pd.Series:
    """Leave-one-sleeve-out book Sharpe: does the result hinge on one hire?"""
    book = res["book"]
    out = {}
    for s in res["sleeves"]:
        live = s.live().reindex(book.index).fillna(0.0)
        label = f"{s.hired_at.date()} {s.candidate.key[:40]}"
        out[label] = sharpe((book - live / seats).values)
    return pd.Series(out).sort_values()


def plot_curves(books: Dict[str, pd.Series], bench_half: pd.Series,
                underlying: str) -> str:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    series = dict(books)
    series[f"{underlying} half-exposure"] = bench_half
    for name, r in series.items():
        cum = (1 + r.fillna(0)).cumprod()
        style = dict(lw=1.2, ls="--", color="0.4") \
            if name.endswith("half-exposure") else dict(lw=1.4)
        ax1.plot(cum.index, cum.values, label=name, **style)
        ax2.plot(cum.index, (cum / cum.cummax() - 1).values, **style)
    ax1.set_yscale("log")
    ax1.set_ylabel("growth of $1 (log)")
    ax1.set_title(f"Pod hiring policies — {underlying}, walk-forward, "
                  "Alpaca ETF costs")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.3)
    ax2.set_ylabel("drawdown")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    path = OUT_DIR / f"equity_curves_{underlying}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Pod hiring-policy comparison")
    ap.add_argument("--underlying", default="QQQ")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--seats", type=int, default=3)
    ap.add_argument("--n-boot", type=int, default=2000,
                    help="bootstrap draws for the robustness section (0 = skip)")
    args = ap.parse_args()

    prices = load_etf_universe_prices()[args.underlying].dropna()
    macro = load_fred_macro()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Underlying: {args.underlying} | seats: {args.seats} | seed: "
          f"{args.seed} | policies: {', '.join(POLICIES)}")
    print("Research results are cached per review and shared across policies "
          "(deterministic\nminer, seed = pod seed + review index) — four "
          "policies ≈ one pod run.\n")

    cache: Dict[int, List[dict]] = {}
    results, books, journals = {}, {}, {}
    for policy in POLICIES:
        print(f"── policy: {policy}")
        pod = PolicyPod(prices, macro=macro,
                        config=PodConfig(seed=args.seed, seats=args.seats),
                        policy=policy, research_cache=cache)
        res = pod.run()
        results[policy], books[policy] = res, res["book"]
        journals[policy] = res["journal"]
        res["haircut"].to_csv(
            OUT_DIR / f"haircut_{policy}_{args.underlying}.csv", index=False)

    bench = prices.pct_change().loc[books["raw"].index]
    half = bench / 2

    # ── summary table ───────────────────────────────────────────────────────
    rows = []
    for policy in POLICIES:
        res = results[policy]
        done = sleeve_table(res)
        corr = (float(done.research_sharpe.corr(done.live_sharpe))
                if len(done) >= 3 else np.nan)
        rows.append({
            "policy": policy, **metrics(books[policy]),
            "hires": int((res["journal"].action == "HIRE").sum()),
            "mean_live_sharpe": round(float(done.live_sharpe.mean()), 2),
            "haircut_corr": round(corr, 2),
        })
    rows.append({"policy": f"{args.underlying} half-exposure (benchmark)",
                 **metrics(half), "hires": np.nan,
                 "mean_live_sharpe": np.nan, "haircut_corr": np.nan})
    summary = pd.DataFrame(rows)

    print("\n" + "=" * 70)
    print(f"HIRING POLICIES ({books['raw'].index[0].date()} → "
          f"{books['raw'].index[-1].date()} — every decision walk-forward)")
    print("=" * 70)
    print(summary.to_string(index=False))
    print(f"\nmean_live_sharpe / haircut_corr: across hired sleeves with "
          f">= {ANALYSIS_LIVE_DAYS} live days;\nhaircut_corr = corr(research "
          "Sharpe at hire, realized live Sharpe).")
    summary.to_csv(OUT_DIR / f"summary_{args.underlying}.csv", index=False)

    # ── the winner's-curse regression (raw policy's sleeves) ────────────────
    winner_curse_regression(sleeve_table(results["raw"]))

    # ── why raw == dsr ──────────────────────────────────────────────────────
    raw_vs_dsr_diagnostic(cache, journals)

    # ── shrunk vs median: where they actually diverged ──────────────────────
    print("\nHIRE DIFFERENCES vs raw")
    print("=" * 70)
    raw_hires = set(map(tuple, journals["raw"][journals["raw"].action == "HIRE"]
                        [["date", "sleeve"]].itertuples(index=False)))
    for policy in POLICIES[1:]:
        h = set(map(tuple, journals[policy][journals[policy].action == "HIRE"]
                    [["date", "sleeve"]].itertuples(index=False)))
        print(f"  {policy:<7} hires: {len(h)} | shared with raw: "
              f"{len(h & raw_hires)} | different: {len(h ^ raw_hires)}")

    # ── plots ───────────────────────────────────────────────────────────────
    path = plot_curves(books, half, args.underlying)
    print(f"\nEquity + drawdown plot: {path}")

    daily = pd.DataFrame({**{p: books[p] for p in POLICIES},
                          f"{args.underlying}_half": half}).fillna(0.0)
    daily.to_csv(OUT_DIR / f"daily_returns_{args.underlying}.csv")

    # ── robustness ──────────────────────────────────────────────────────────
    if args.n_boot > 0:
        rng = np.random.default_rng(args.seed)
        boot = stationary_bootstrap(daily, args.n_boot, rng)
        boot.to_csv(OUT_DIR / f"bootstrap_sharpes_{args.underlying}.csv",
                    index=False)
        print(f"\nROBUSTNESS — stationary block bootstrap of daily returns "
              f"({args.n_boot} draws,\nmean block 63d, SAME blocks for every "
              "policy)")
        print("=" * 70)
        q = boot.quantile([0.05, 0.50, 0.95]).T
        q.columns = ["p05", "p50", "p95"]
        print(q.round(2).to_string())
        bench_col = f"{args.underlying}_half"
        for p in POLICIES[1:]:
            print(f"  {p:<8} P(> raw): "
                  f"{float((boot[p] > boot['raw']).mean()):.2f} | "
                  f"P(> {args.underlying} half): "
                  f"{float((boot[p] > boot[bench_col]).mean()):.2f}")

        print("\nROBUSTNESS — leave-one-sleeve-out book Sharpe")
        print("=" * 70)
        for policy in ("raw", "median", "inverse", "veto"):
            jk = jackknife_sleeves(results[policy], args.seats)
            print(f"  {policy:<7} full {metrics(books[policy])['sharpe']:.2f} | "
                  f"drop-1 range [{jk.min():.2f}, {jk.max():.2f}] | "
                  f"most relied-on sleeve: {jk.index[0]}")

    print(f"\nSaved to {OUT_DIR}/: summary_*, haircut_<policy>_*, "
          "daily_returns_*, bootstrap_sharpes_*, equity_curves_*.png")


if __name__ == "__main__":
    main()
