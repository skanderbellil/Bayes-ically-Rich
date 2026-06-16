#!/usr/bin/env python3
"""
The zoo as a market-neutral overlay — does orthogonality expand the frontier,
and at what cost does the benefit die?

Idea (the arc's synthesis, made decision-relevant)
--------------------------------------------------
run_zoo_tracking + run_reachable_frontier proved the levered EW zoo is the
orthogonal complement of a long-only book (ρ to equities ≈ −0.4) — so you
can't REBUILD it from beta, but you can ADD it. This study asks the only
question left: if you access a market-neutral version of the zoo (via real
shorts / a liquid-alts sleeve), how much does a small overlay improve an
equity core, what's the optimal dose, and — since the zoo is GROSS
academic — at what annual cost drag does the benefit vanish?

Setup
-----
Core     : FF total market (Mkt-RF + RF), monthly — ≈ a broad equity index,
           full 1998-2024 overlap with the zoo (vs the ETF panel's 2011).
Overlay  : levered EW zoo @10% vol (the Idea-12 construction), GROSS, then
           net of an annual cost-drag grid {0,2,4,6,8}%/yr — the band the
           anomaly-cost literature puts on hard-to-trade academic spreads.
Blends   : core + w·overlay, both held long (overlay is internally
           market-neutral), w ∈ {0,.1,.2,.3,.5}; plus the ex-ante optimal
           static w and a walk-forward 36m-tangency w (capped [0,.5], no
           future info). Report break-even cost where d(Sharpe)/dw|0 = 0.

Honesty box
-----------
• The overlay still REQUIRES market-neutral access — shorts or an alts
  fund. This is the "if you can hold it" study; run_reachable_frontier is
  the "what you can buy long-only" study. Together they bracket reality.
• The cost grid is the load-bearing honesty: the gross zoo's edge is real
  only net of the frictions the prior studies flagged. Read the break-even,
  not the gross lift.
• Core is FF Mkt (total), ~SPY but broader/older. Trials: 5 static doses +
  1 optimal + 1 walk-forward × (gross + 5 cost levels) — descriptive
  frontier mapping, not a search for a winning w.
"""
import logging
import sys

import numpy as np
import pandas as pd
from tabulate import tabulate

import _bootstrap  # noqa: F401
from posterioralpha.data import load_openap_doc, load_openap_returns

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)
RESULTS_DIR = _bootstrap.ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
DATA = _bootstrap.ROOT / "datasets"

MIN_ELIGIBLE = 30
TARGET_VOL, LEV_CAP = 0.10, 3.0
COST_GRID = [0.0, 0.02, 0.04, 0.06, 0.08]   # annual drag on the overlay
DOSES = [0.0, 0.1, 0.2, 0.3, 0.5]
WF_WIN, WF_MIN, WF_CAP = 36, 24, 0.5


def sharpe(r: pd.Series, rf: pd.Series) -> float:
    x = (r - rf.reindex(r.index).fillna(0.0)).dropna()
    return float(x.mean() / (x.std() + 1e-12) * np.sqrt(12))


def ann(r):
    return float(r.dropna().mean() * 12)


def maxdd(r):
    eq = (1 + r.dropna()).cumprod()
    return float((eq / eq.cummax() - 1).min())


def build_zoo():
    rets = load_openap_returns()
    doc = load_openap_doc().set_index("Acronym")
    meta = doc.loc[doc.index.intersection(rets.columns), ["Year"]].astype(float)
    rets = rets[meta.index]
    age = rets.index.year.values[:, None] - meta.Year.values
    pub = pd.DataFrame(age >= 1, index=rets.index, columns=rets.columns)
    elig = (pub & rets.notna()).shift(1).fillna(False)
    n = elig.sum(axis=1) >= MIN_ELIGIBLE
    ew = rets.where(elig).mean(axis=1)[n]
    lev = (TARGET_VOL / (ew.rolling(12).std() * np.sqrt(12))).clip(upper=LEV_CAP).shift(1)
    return (ew * lev).dropna()


def main():
    ff = pd.read_csv(DATA / "ff_factors_monthly.csv", index_col=0, parse_dates=True)
    zoo = build_zoo()
    idx = ff.index.intersection(zoo.index)
    core = (ff["Mkt-RF"] + ff["RF"]).reindex(idx)
    rf = ff["RF"].reindex(idx)
    zoo = zoo.reindex(idx)
    rho = core.corr(zoo)
    logger.info(f"core = FF total market; overlay = levered EW zoo. "
                f"{idx[0].date()} → {idx[-1].date()}, n={len(idx)}, "
                f"corr(core, overlay) = {rho:+.2f}")
    logger.info(f"core Sharpe {sharpe(core, rf):.2f}  |  overlay Sharpe "
                f"{sharpe(zoo, rf):.2f} (gross)")

    # ── break-even cost: closed form from the two Sharpes + ρ ─────────────
    # d/dw Sharpe(core + w·overlay)|w=0 > 0  ⇔  S_o > ρ·S_c  (overlay helps
    # at the margin). Solve the annual drag c that makes net S_o = ρ·S_c.
    Sc = sharpe(core, rf)
    vo = zoo.std() * np.sqrt(12)
    So_gross = sharpe(zoo, rf)
    breakeven = (So_gross - rho * Sc) * vo      # cost (ann) that zeros margin
    print(f"\nMarginal-benefit condition: overlay helps while its net Sharpe "
          f"> ρ·S_core = {rho * Sc:+.2f}.")
    print(f"Break-even annual cost on the overlay ≈ {breakeven:.1%} "
          f"(gross edge {So_gross:.2f} − hurdle {rho*Sc:+.2f}, × vol {vo:.0%}).")

    # ── frontier: Sharpe(core + w·overlay) over dose × cost ───────────────
    rows = []
    opt_by_cost = {}
    for c in COST_GRID:
        net = zoo - c / 12
        line = []
        # fine grid for the optimum
        grid = np.linspace(0, 1.0, 101)
        s_grid = [sharpe(core + w * net, rf) for w in grid]
        w_opt = grid[int(np.argmax(s_grid))]
        opt_by_cost[c] = (w_opt, max(s_grid))
        for w in DOSES:
            line.append(sharpe(core + w * net, rf))
        rows.append([f"{c:.0%}"] + [f"{s:.2f}" for s in line]
                    + [f"{w_opt:.0%}", f"{max(s_grid):.2f}"])
    print("\nSharpe of (core + w·overlay) by overlay dose w and annual cost:")
    print(tabulate(rows, headers=["cost"] + [f"w={w:.0%}" for w in DOSES]
                   + ["w*", "Sharpe*"], tablefmt="github"))
    print(f"  core-only Sharpe = {Sc:.2f} (the w=0 column); w* = "
          f"unconstrained-optimal static dose at that cost")

    # ── the implementable test: walk-forward dose, no future info ─────────
    print("\nWalk-forward overlay (36m tangency dose, capped [0,50%], "
          "monthly), NET at each cost:")
    wf_rows = []
    for c in COST_GRID:
        net = zoo - c / 12
        w_t = pd.Series(0.0, index=idx)
        for i in range(WF_MIN, len(idx)):
            lo = max(0, i - WF_WIN)
            cc, oo = core.iloc[lo:i], net.iloc[lo:i]
            # 2-asset tangency weight on the overlay (vs core), long-only cap
            mu = np.array([cc.mean(), oo.mean()])
            cov = np.cov(np.vstack([cc.values, oo.values]))
            try:
                raw = np.linalg.solve(cov, mu)
                w_o = raw[1] / raw.sum() if raw.sum() > 0 else 0.0
            except np.linalg.LinAlgError:
                w_o = 0.0
            w_t.iloc[i] = float(np.clip(w_o, 0.0, WF_CAP))
        blend = core + w_t.shift(1).fillna(0.0) * net
        live = blend.iloc[WF_MIN:]
        wf_rows.append([f"{c:.0%}", f"{w_t.iloc[WF_MIN:].mean():.0%}",
                        f"{sharpe(live, rf):.2f}",
                        f"{ann(live):+.2%}", f"{maxdd(live):.0%}"])
    core_live = core.iloc[WF_MIN:]
    print(tabulate(wf_rows, headers=["cost", "avg w", "blend Sharpe",
                                     "blend ann", "blend maxDD"],
                   tablefmt="github"))
    print(f"  core-only over the same window: Sharpe {sharpe(core_live, rf):.2f}"
          f"  ann {ann(core_live):+.2%}  maxDD {maxdd(core_live):.0%}")

    out = RESULTS_DIR / "zoo_overlay.csv"
    pd.DataFrame([{"cost": c, "w_opt": opt_by_cost[c][0],
                   "sharpe_opt": opt_by_cost[c][1]} for c in COST_GRID]
                 ).to_csv(out, index=False)
    logger.info(f"\nSaved {out}")
    print("\n⚠ overlay needs market-neutral access (shorts/alts), and the "
          "gross zoo's edge survives only below the break-even cost — read "
          "the cost row that matches your real frictions.")


if __name__ == "__main__":
    main()
