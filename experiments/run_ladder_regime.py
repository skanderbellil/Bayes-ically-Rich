#!/usr/bin/env python3
"""
Adaptive rung selection up the integral ladder (user 2026-06-30).

Hypothesis: WHICH rung of the kinematic ladder carries the signal changes over
time. In trending regimes one rung leads; in choppy regimes another. A causal
trend/chop classifier (Kaufman efficiency ratio) gates which rung we trade, with
the rung→regime map LEARNED ON TRAIN only.

Honest premise (already checked): the informative rung does flip by regime, but
not in the naive direction — dip-buying (deviation-reversion) leads in high-ER
trends, the slow integral (absement) leads in chop, velocity loses everywhere.
So we let the data assign rung→regime rather than hand-labelling.

Rigor: strict causality (ER, rungs, regime thresholds all backward-looking;
positions lagged), expanding walk-forward (thresholds + rung choice refit every
fold), net of cost with turnover, and the key conditional diagnostic — does the
switch beat the best FIXED rung and a static blend, or is it just picking the
better rung on average?
"""
import sys
import warnings

import numpy as np
import pandas as pd
from tabulate import tabulate

import _bootstrap  # noqa: F401
from posterioralpha.data import load_fresh_prices
from posterioralpha.kinematic import run_signal_backtest, walk_forward
from posterioralpha.kinematic.ladder import (
    ladder_rungs, efficiency_ratio, RegimeRungSelector,
)
from posterioralpha.kinematic.diagnostics import (
    information_coefficient, pnl_concentration, effective_bets,
)
from posterioralpha.mining.validation import deflated_sharpe_ratio

warnings.filterwarnings("ignore")
ANN = 252
COST_BPS, COST_BPS_HI = 1.0, 5.0
ER_WINDOW = 30
OUT = _bootstrap.ROOT / "data" / "kinematic"
OUT.mkdir(parents=True, exist_ok=True)


def _tab(df, **kw):
    print(tabulate(df, headers="keys", tablefmt="github", floatfmt="+.3f",
                   showindex=kw.get("showindex", False)))


def bt_row(name, sig, ret, te):
    s = sig.reindex(te)
    fwd = ret.shift(-1).reindex(te)
    b1 = run_signal_backtest(s, ret, cost_bps=COST_BPS)
    b5 = run_signal_backtest(s, ret, cost_bps=COST_BPS_HI)
    return {"strategy": name, "IC": information_coefficient(s, fwd),
            "gross": b1.stats["sharpe_gross"], "net@1bp": b1.stats["sharpe_net"],
            "net@5bp": b5.stats["sharpe_net"], "turn": b1.stats["ann_turnover"]}, b1.net


def make_signal_fn():
    """fit_signal_fn for walk_forward: fit the selector on the train slice,
    emit the gated signal over the full index (causal)."""
    def fn(train_lp: pd.Series, full_lp: pd.Series) -> pd.Series:
        rungs_tr = ladder_rungs(train_lp)
        ER_tr = efficiency_ratio(np.exp(train_lp), ER_WINDOW)
        fwd_tr = np.exp(train_lp).pct_change().shift(-1)
        sel = RegimeRungSelector(n_regimes=3).fit(rungs_tr, ER_tr, fwd_tr)
        rungs_full = ladder_rungs(full_lp)
        ER_full = efficiency_ratio(np.exp(full_lp), ER_WINDOW)
        return sel.transform(rungs_full, ER_full)
    return fn


def main():
    p = np.log(load_fresh_prices("SPY").dropna()["SPY"])
    price = np.exp(p)
    ret = price.pct_change()
    n = len(p); ntr = n // 2; te = p.index[ntr:]
    print(f"SPY {p.index[0].date()} → {p.index[-1].date()} ({n} days); "
          f"OOS from {te[0].date()}\n")

    rungs = ladder_rungs(p)
    ER = efficiency_ratio(price, ER_WINDOW)
    fwd_tr = ret.shift(-1).iloc[:ntr]

    # ── learned rung→regime map (fit on train) ───────────────────────────────
    sel = RegimeRungSelector(n_regimes=3).fit(rungs.iloc[:ntr], ER.iloc[:ntr], fwd_tr)
    print("Learned rung→regime map (TRAIN efficiency-ratio terciles, "
          "0=chop … 2=trend):")
    _tab(sel.summary())
    gated = sel.transform(rungs, ER)

    # ── head-to-head, single split OOS ───────────────────────────────────────
    print("\nOOS head-to-head (single 50/50 split), net of cost:")
    rows, curves = [], {}
    for name in rungs.columns:
        r, net = bt_row(name, rungs[name], ret, te); rows.append(r); curves[name] = net
    static_blend = rungs.mean(axis=1)
    r, net = bt_row("static blend (P+I+D)/3", static_blend, ret, te); rows.append(r); curves[r["strategy"]] = net
    r, net = bt_row("REGIME-GATED rung", gated, ret, te); rows.append(r); curves[r["strategy"]] = net
    _tab(pd.DataFrame(rows))

    # ── the decisive test: walk-forward (refit map every fold) ───────────────
    print("\nExpanding walk-forward (rung→regime map refit each of 5 folds):")
    wf = walk_forward(p, make_signal_fn(), n_folds=5)
    wf_rows = []
    for name, sig in [("REGIME-GATED (walk-fwd)", wf),
                      ("best fixed rung: absement-trend", rungs["absement-trend"]),
                      ("best fixed rung: deviation-rev", rungs["deviation-reversion"])]:
        r, net = bt_row(name, sig, ret, p.index[ntr:]); wf_rows.append(r)
        if "GATED" in name:
            gated_wf_net = net
    _tab(pd.DataFrame(wf_rows))

    # ── honest diagnostics on the walk-forward gated strategy ────────────────
    net = gated_wf_net.dropna()
    conc = pnl_concentration(net)
    print(f"\nWalk-forward gated: top10 days = {conc['top10_share']:.0%} of net PnL, "
          f"Gini|pnl|={conc['gini_abs']:.2f}, effective bets N_eff="
          f"{effective_bets(net):.0f}/{len(net)}")
    yr = net.groupby(net.index.year).sum()
    print("PnL by year (walk-forward gated, net):")
    _tab(yr.round(4).to_frame("net_pnl"), showindex=True)

    # DSR across the rung variants tried
    trials = {**{k: v for k, v in curves.items()}, "walk-fwd gated": gated_wf_net.dropna()}
    srs = [float(v.dropna().mean()/(v.dropna().std()+1e-12)) for v in trials.values()]
    sr_var = float(np.var(srs, ddof=1))
    print(f"\nDeflated Sharpe (n_trials={len(trials)}):")
    drows = [{"variant": k,
              "Sharpe": float(v.dropna().mean()/(v.dropna().std()+1e-12))*np.sqrt(ANN),
              "DSR": deflated_sharpe_ratio(v.dropna().values, len(trials), sr_var)}
             for k, v in trials.items()]
    _tab(pd.DataFrame(drows).sort_values("DSR", ascending=False))

    _verdict(rows, wf_rows)
    yr.to_csv(OUT / "ladder_regime_pnl_by_year.csv")


def _verdict(rows, wf_rows):
    split = {r["strategy"]: r for r in rows}
    gated = split["REGIME-GATED rung"]["net@1bp"]
    best_fixed = max(split[k]["net@1bp"] for k in
                     ("deviation-reversion", "absement-trend", "velocity-trend"))
    blend = split["static blend (P+I+D)/3"]["net@1bp"]
    wf_gated = next(r["net@1bp"] for r in wf_rows if "GATED" in r["strategy"])
    print("\n" + "═" * 70)
    print("VERDICT")
    print("═" * 70)
    print(f"  single-split: gated net {gated:+.2f}  vs best fixed rung {best_fixed:+.2f}  "
          f"vs static blend {blend:+.2f}")
    print(f"  walk-forward: gated net {wf_gated:+.2f}  (the real OOS number)")
    if wf_gated > best_fixed and wf_gated > blend:
        print("  → switching ADDS value over the best fixed rung and the blend.")
    else:
        print("  → switching does NOT clearly beat the best fixed rung / blend on "
              "this single asset — the conditional edge is real but too small to "
              "monetise here. Cross-section is the next test.")


if __name__ == "__main__":
    main()
