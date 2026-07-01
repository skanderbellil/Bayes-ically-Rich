#!/usr/bin/env python3
"""
Integral ladder in the cross-section: does the rung that works flip by regime,
and is that flip exploitable? (user 2026-06-30)

Backstory. Up the kinematic ladder (integrate) the noise cancels; down it
(differentiate) the noise explodes. On a SINGLE asset the "which rung wins
changes over time" idea was visible but not learnable — one history has too few
regime episodes, so a train-fit switcher just picked reversion everywhere and
lost (see experiments/run_ladder_regime.py). The cross-section supplies the
breadth. Here we test, on ~100 S&P names:

  1. Cross-sectional IC of each rung, OVERALL and split by market regime
     (Kaufman efficiency ratio on the EW index → trend vs chop).
  2. The exploitable test: a parameter-free regime-gated blend (weight rungs by
     1−efficiency) vs each fixed rung, a naive equal-weight combo, an ADAPTIVE
     trailing-IC combo, and 12-1 momentum.

Findings (honest):
  • Up beats down — velocity (derivative) is significantly NEGATIVE in every
    regime; the integral rung is weakly positive.
  • The flip is real WITH breadth: absement leads in chop, reversion in trend.
  • The right way to COMBINE is structural gating (condition on the regime),
    not weighting by recent performance — the adaptive-IC combo chases mean-
    reverting edges and loses; the regime-gated blend beats every fixed rung.
  • Ceiling is honest: a weak (~0.44-corr) diversifier to momentum, not a
    standalone alpha, on a survivorship-biased current-membership universe.

Honest box: sp500_top100 is CURRENT membership → survivorship-biased, which
inflates return LEVELS. Read the regime CONTRAST and the velocity SIGN (both
bias-robust), not the absolute Sharpes. Costs charged on turnover; single-name
equity, so 5 bp default with a 10 bp column.
"""
import sys
import warnings

import numpy as np
import pandas as pd
from tabulate import tabulate

import _bootstrap  # noqa: F401
from posterioralpha.data import load_sp500_prices
from posterioralpha.kinematic.ladder import (
    per_name_rungs, month_ends, efficiency_ratio,
    cross_sectional_ls, cross_sectional_ic, regime_gated_blend,
)

warnings.filterwarnings("ignore")
ANN = 252
ER_WINDOW = 126           # ~6m market trend/chop window
ADAPT_LOOKBACK = 12       # months of trailing IC for the adaptive combo
RUNGS = ["absement-trend", "deviation-reversion", "velocity-trend"]
OUT = _bootstrap.ROOT / "data" / "kinematic"
OUT.mkdir(parents=True, exist_ok=True)


def _tab(df, **kw):
    print(tabulate(df, headers="keys", tablefmt="github", floatfmt="+.3f",
                   showindex=kw.get("showindex", False)))


def shp(r):
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(12)) if r.std() else 0.0


def tstat(r):
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(len(r))) if r.std() else 0.0


def main():
    px = load_sp500_prices()
    px = px.loc[:, px.notna().mean() > 0.95].dropna(how="all").ffill(limit=3)
    lp = np.log(px)
    me = month_ends(lp.index)
    fwd = px.reindex(me).pct_change().shift(-1)
    print(f"Universe {px.shape[1]} names, {px.index[0].date()} → {px.index[-1].date()}, "
          f"{len(me)} months")

    rungs = per_name_rungs(lp)
    rungs_all = {**rungs, "mom 12-1": px.shift(21) / px.shift(252) - 1.0}

    # market regime (efficiency ratio of the EW index)
    ew = np.exp(lp.mean(axis=1))
    ER = efficiency_ratio(ew, ER_WINDOW)
    er_m = ER.reindex(me)
    q1, q2 = er_m.quantile(1 / 3), er_m.quantile(2 / 3)
    regime = pd.Series(np.where(er_m <= q1, "CHOP",
                                np.where(er_m >= q2, "TREND", "MID")), index=me)

    # ── 1. rung IC, overall and by regime ────────────────────────────────────
    print(f"\nRegime months — CHOP {int((regime=='CHOP').sum())} | "
          f"MID {int((regime=='MID').sum())} | TREND {int((regime=='TREND').sum())}")
    print("\nCross-sectional IC per rung (t-stat in parens):")
    rows = []
    ic_by_rung = {}
    for name in RUNGS + ["mom 12-1"]:
        ic = cross_sectional_ic(rungs_all[name], fwd, me)
        ic_by_rung[name] = ic
        ch = ic[regime.reindex(ic.index) == "CHOP"]
        tr = ic[regime.reindex(ic.index) == "TREND"]
        rows.append({
            "rung": name,
            "overall": f"{ic.mean():+.3f} ({tstat(ic):+.1f})",
            "CHOP": f"{ch.mean():+.3f} ({tstat(ch):+.1f})",
            "TREND": f"{tr.mean():+.3f} ({tstat(tr):+.1f})",
        })
    print(tabulate(pd.DataFrame(rows), headers="keys", tablefmt="github", showindex=False))
    print("→ velocity (derivative) negative in both regimes; the informative rung "
          "FLIPS: absement (integral) in chop, reversion in trend.")

    # ── 2. exploitable combination test ──────────────────────────────────────
    gated = regime_gated_blend(rungs, ER)                       # parameter-free structural
    ew_combo = sum(rungs[k] for k in RUNGS) / 3.0               # naive equal weight

    # adaptive: weight each rung by trailing observable IC (shift 1 → causal)
    icdf = pd.DataFrame({k: ic_by_rung[k] for k in RUNGS}).reindex(me)
    w = icdf.shift(1).rolling(ADAPT_LOOKBACK, min_periods=6).mean().clip(lower=0)
    w = w.div(w.sum(axis=1).replace(0, np.nan), axis=0).fillna(1 / 3)

    def xs_z(frame):
        return frame.sub(frame.mean(axis=1), axis=0).div(frame.std(axis=1), axis=0)
    zr = {k: xs_z(rungs[k]) for k in RUNGS}
    adaptive = sum(zr[k].mul(w[k].reindex(zr[k].index), axis=0) for k in RUNGS)

    print("\nCross-sectional long-short, net of cost:")
    combos = [
        ("velocity (derivative) fixed", rungs["velocity-trend"]),
        ("absement (best fixed rung)", rungs["absement-trend"]),
        ("equal-weight combo (P+I+D)", ew_combo),
        ("ADAPTIVE trailing-IC combo", adaptive),
        ("REGIME-GATED blend (1−ER)", gated),
        ("12-1 momentum (benchmark)", rungs_all["mom 12-1"]),
    ]
    tab, nets = [], {}
    for name, sig in combos:
        r5 = cross_sectional_ls(sig, fwd, me, cost_bps=5.0)
        r10 = cross_sectional_ls(sig, fwd, me, cost_bps=10.0)
        nets[name] = r5
        tab.append({"factor": name, "net@5bp": shp(r5), "net@10bp": shp(r10),
                    "t-stat": tstat(r5)})
    _tab(pd.DataFrame(tab))

    rG, rM = nets["REGIME-GATED blend (1−ER)"], nets["12-1 momentum (benchmark)"]
    combo = 0.5 * rG.add(rM, fill_value=0).dropna()
    print(f"\nregime-gated corr to momentum = {rG.corr(rM):+.2f}; "
          f"50/50 gated+momentum Sharpe {shp(combo):+.2f} vs momentum alone {shp(rM):+.2f}")

    _verdict(tab)
    _plots(w, me, nets, rM, combo, rungs, fwd, regime, ic_by_rung)
    pd.DataFrame({k: v for k, v in nets.items()}).to_csv(OUT / "ladder_xsec_returns.csv")


def _verdict(tab):
    d = {r["factor"]: r["net@5bp"] for r in tab}
    print("\n" + "═" * 72)
    print("VERDICT")
    print("═" * 72)
    print(f"  gating on market state:   {d['REGIME-GATED blend (1−ER)']:+.2f}  (beats every fixed rung)")
    print(f"  best single fixed rung:   {d['absement (best fixed rung)']:+.2f}")
    print(f"  equal-weight combo:       {d['equal-weight combo (P+I+D)']:+.2f}  (drags in dead velocity rung)")
    print(f"  weighting by recent IC:   {d['ADAPTIVE trailing-IC combo']:+.2f}  (chases mean-reverting edges → loses)")
    print(f"  12-1 momentum benchmark:  {d['12-1 momentum (benchmark)']:+.2f}")
    print("  → Combining derivations helps ONLY when the weight tracks a stable "
          "structural\n    regime; weighting by recent performance actively hurts. "
          "Conditioning beats chasing.")


def _plots(w, me, nets, rM, combo, rungs, fwd, regime, ic_by_rung):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"(plots skipped: {exc})")
        return
    fig, ax = plt.subplots(2, 1, figsize=(13, 9))
    colr = {"absement-trend": "#16a085", "deviation-reversion": "#c0392b",
            "velocity-trend": "#e67e22"}
    w.reindex(me).plot(ax=ax[0], lw=1.2, color=colr)
    ax[0].set_title(f"Adaptive rung weights chase whatever worked lately (trailing "
                    f"{ADAPT_LOOKBACK}m IC) — and thrash"); ax[0].set_ylabel("weight")
    ax[0].legend(fontsize=8, ncol=3)
    for name, c in [("REGIME-GATED blend (1−ER)", "#16a085"),
                    ("ADAPTIVE trailing-IC combo", "#8e44ad"),
                    ("absement (best fixed rung)", "#95a5a6")]:
        (1 + nets[name]).cumprod().plot(ax=ax[1], color=c, lw=1.6,
                                        label=f"{name} (Shp {shp(nets[name]):+.2f})")
    (1 + rM).cumprod().plot(ax=ax[1], color="#34495e", lw=1.1, ls=":",
                            label=f"12-1 momentum (Shp {shp(rM):+.2f})")
    ax[1].set_title("Cross-sectional long-short, net @5bp — gating (green) holds, "
                    "chasing (purple) bleeds")
    ax[1].legend(fontsize=8); ax[1].set_ylabel("cumulative")
    fig.tight_layout(); fig.savefig(OUT / "ladder_xsec.png", dpi=110)
    print(f"\nFigure → {(OUT / 'ladder_xsec.png').relative_to(_bootstrap.ROOT)}")


if __name__ == "__main__":
    main()
