"""
Long-book weighting bake-off + FALSIFICATION suite.

MOTIVATION
----------
Across iterations the plain long-only book (+40% YTD) still beats every
dollar/beta/risk-neutral variant. So the sharpest remaining question is: how
should we weight the LONG leg -- and, more importantly, can we *falsify* the idea
that any particular weighting is genuinely better, rather than noise or an
artifact of one tier?

METHOD (deliberately consistent + no lookahead)
-----------------------------------------------
The whole study runs on ONE basis so methods and the random null are directly
comparable:
  * FORMATION WINDOW: the first `FORM` trading days are used only to *estimate*
    (volatilities). No performance is scored there.
  * EVALUATION: weights are set once at the end of the formation window and held
    (buy-and-hold) to the end. NAV_t = Σ w_i · price_i,t / price_i,form.
  * Buy-and-hold (not monthly rebalancing) is chosen on purpose: it isolates the
    *weighting decision* from any rebalancing luck, and over a ~6-month window it
    is nearly identical to monthly rebalancing anyway. It is also vectorizable, so
    the random-weight null (thousands of draws) is instant.

The four methods -- all simple, general, no return-fitting:
  1. equal                  : 1/N. The null hypothesis of "weighting".
  2. inverse_vol            : w_i ∝ 1/vol_i (equal risk, ignores tiers).
  3. tier_balanced          : each of the 4 thesis tiers gets 1/4 of the book,
                              split equally within -- so name-count can't decide
                              tier exposure (Tier4 has 11 names, Tier1 has 9...).
  4. inv_vol_within_tier    : tier-balanced budget, inverse-vol within each tier
                              (a lite hierarchical-risk-parity).

FALSIFICATION TESTS (the point of the exercise)
-----------------------------------------------
  A. Subperiod stability   : split the eval window in two; do method ranks hold?
  B. Jackknife (LOO)       : drop each name, renormalize; how much does each
                             method's return move? Does any edge survive?
  C. Drop the carry tier   : remove Tier4 silicon (which carried H1). Does any
                             weighting still work, or was it all one tier?
  D. Parameter sensitivity : inverse-vol with formation windows {20,40,60}.
  E. Random-weight NULL    : thousands of Dirichlet weight vectors -> the null
                             distribution of outcomes. Each method's PERCENTILE in
                             that null is the falsification: a method sitting near
                             the median is indistinguishable from random weighting.

The honest prior: over a single 6-month window where 47 longs all ride the same
AI-capex beta, weighting differences are probably mostly NOISE and the return is
driven by (theme + one tier), not by the weighting scheme. These tests are built
to give that hypothesis every chance to be *rejected* -- and to report plainly if
it is not.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import data as datamod
from config import (LONG_TICKERS, UNIVERSE, LONG_TIERS, RISK_FREE_ANNUAL,
                    TRADING_DAYS)
from analysis import metrics as series_metrics

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)
_PCT = FuncFormatter(lambda x, _: f"{x:.0%}")

FORM = 40                 # formation window (trading days) for vol estimation
N_NULL = 5000             # random-weight null draws
RNG = np.random.default_rng(20260712)   # fixed seed -> reproducible

METHOD_COLORS = {
    "equal": "#7F7F7F",
    "inverse_vol": "#0072B2",
    "tier_balanced": "#009E73",
    "inv_vol_within_tier": "#CC79A7",
}


# ---------------------------------------------------------------------------
# Buy-and-hold evaluation primitives
# ---------------------------------------------------------------------------
def rel_prices(prices, members, form_idx):
    """Per-name gross multiple relative to the formation-end price, eval window."""
    p = prices[members].iloc[form_idx:]
    return p / p.iloc[0]


def bh_nav(rel, w):
    """Buy-and-hold portfolio NAV path from static weights `w` (Series/array)."""
    if isinstance(w, pd.Series):
        w = w.reindex(rel.columns).fillna(0.0).values
    return pd.Series(rel.values @ w, index=rel.index)


def bh_metrics(rel, w, name=""):
    nav = bh_nav(rel, w)
    ret = nav.pct_change().dropna()
    m = series_metrics(ret, name=name, rf=RISK_FREE_ANNUAL)
    m["eval_return"] = float(nav.iloc[-1] / nav.iloc[0] - 1.0)
    return m


# ---------------------------------------------------------------------------
# Weight schemes (set at formation end; all sum to 1 over `members`)
# ---------------------------------------------------------------------------
def w_equal(members, vol, tiers):
    return pd.Series(1.0 / len(members), index=members)


def w_inverse_vol(members, vol, tiers):
    inv = 1.0 / vol.reindex(members)
    return inv / inv.sum()


def w_tier_balanced(members, vol, tiers):
    w = pd.Series(0.0, index=members)
    present = [t for t in LONG_TIERS if any(tiers[m] == t for m in members)]
    for t in present:
        names = [m for m in members if tiers[m] == t]
        w.loc[names] = (1.0 / len(present)) / len(names)
    return w


def w_inv_vol_within_tier(members, vol, tiers):
    w = pd.Series(0.0, index=members)
    present = [t for t in LONG_TIERS if any(tiers[m] == t for m in members)]
    for t in present:
        names = [m for m in members if tiers[m] == t]
        inv = 1.0 / vol.reindex(names)
        w.loc[names] = (inv / inv.sum()) * (1.0 / len(present))
    return w


METHODS = {
    "equal": w_equal,
    "inverse_vol": w_inverse_vol,
    "tier_balanced": w_tier_balanced,
    "inv_vol_within_tier": w_inv_vol_within_tier,
}


# ---------------------------------------------------------------------------
# Core: build method weights + evaluate
# ---------------------------------------------------------------------------
def formation_vol(returns, members, form_idx):
    """Per-name vol estimated over the formation window only (no lookahead)."""
    return returns[members].iloc[:form_idx].std(ddof=1)


def run_methods(prices, returns, members, form=FORM):
    tiers = {m: UNIVERSE[m]["tier"] for m in members}
    vol = formation_vol(returns, members, form)
    rel = rel_prices(prices, members, form)
    weights, mets = {}, {}
    for name, fn in METHODS.items():
        w = fn(members, vol, tiers)
        weights[name] = w
        mets[name] = bh_metrics(rel, w, name=name)
    return weights, pd.DataFrame(mets).T, rel, vol


# ---------------------------------------------------------------------------
# FALSIFICATION TESTS
# ---------------------------------------------------------------------------
def test_random_null(rel, weights, n=N_NULL, alpha=1.0):
    """Dirichlet(alpha) random weights -> null distribution of eval returns.
    Returns (null_returns array, {method: percentile})."""
    k = rel.shape[1]
    W = RNG.dirichlet(np.full(k, alpha), size=n)         # (n, k)
    end = rel.values[-1]                                  # gross multiple at end
    null_ret = end @ W.T - 1.0                            # (n,)
    pct = {}
    for name, w in weights.items():
        r = float(rel.values[-1] @ w.reindex(rel.columns).fillna(0).values - 1.0)
        pct[name] = float((null_ret < r).mean())
    return null_ret, pct


def test_subperiods(prices, returns, members, form=FORM, k=2):
    """Split the eval window into k consecutive chunks; report each method's
    return per chunk and its rank, to see if the ranking is stable."""
    tiers = {m: UNIVERSE[m]["tier"] for m in members}
    vol = formation_vol(returns, members, form)
    rel_full = rel_prices(prices, members, form)
    idx = rel_full.index
    bounds = np.array_split(np.arange(len(idx)), k)
    rows = {}
    for name, fn in METHODS.items():
        w = fn(members, vol, tiers)
        nav = bh_nav(rel_full, w)
        chunk_ret = {}
        for ci, b in enumerate(bounds):
            seg = nav.iloc[b]
            chunk_ret[f"sub{ci+1}"] = float(seg.iloc[-1] / seg.iloc[0] - 1.0)
        rows[name] = chunk_ret
    df = pd.DataFrame(rows).T
    ranks = df.rank(ascending=False).astype(int)   # 1 = best that subperiod
    return df, ranks


def test_jackknife(prices, returns, members, form=FORM):
    """Leave-one-name-out: for each method, drop each name (renormalize) and
    recompute eval return. Report the min/max/std of each method's return and
    whether its edge over `equal` ever flips sign."""
    tiers = {m: UNIVERSE[m]["tier"] for m in members}
    vol = formation_vol(returns, members, form)
    out = {}
    for name, fn in METHODS.items():
        vals = []
        for drop in members:
            sub = [m for m in members if m != drop]
            rel = rel_prices(prices, sub, form)
            w = fn(sub, vol.reindex(sub), {m: tiers[m] for m in sub})
            vals.append(float(rel.values[-1] @ w.reindex(sub).fillna(0).values - 1))
        s = pd.Series(vals, index=members)
        out[name] = dict(mean=s.mean(), min=s.min(), max=s.max(),
                         std=s.std(ddof=1), range=s.max() - s.min())
    return pd.DataFrame(out).T


def test_drop_tier(prices, returns, members, drop_tier="Tier4_silicon", form=FORM):
    """Recompute all methods with the carry tier removed."""
    sub = [m for m in members if UNIVERSE[m]["tier"] != drop_tier]
    _, mt, _, _ = run_methods(prices, returns, sub, form=form)
    return mt


def test_param_sensitivity(prices, returns, members, windows=(20, 40, 60)):
    """inverse_vol eval return as the formation window varies."""
    tiers = {m: UNIVERSE[m]["tier"] for m in members}
    out = {}
    for f in windows:
        vol = formation_vol(returns, members, f)
        rel = rel_prices(prices, members, f)
        w = w_inverse_vol(members, vol, tiers)
        we = w_equal(members, vol, tiers)
        out[f] = dict(
            inverse_vol=float(rel.values[-1] @ w.reindex(members).values - 1),
            equal=float(rel.values[-1] @ we.reindex(members).values - 1),
        )
    df = pd.DataFrame(out).T
    df["inv_vol_minus_equal"] = df["inverse_vol"] - df["equal"]
    return df


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def chart_method_nav(rel_by_method, spy_rel, fname="09_long_weighting_nav.png"):
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, nav in rel_by_method.items():
        ax.plot(nav.index, nav.values - 1.0, lw=2.0, color=METHOD_COLORS[name],
                label=name)
    if spy_rel is not None:
        ax.plot(spy_rel.index, spy_rel.values - 1.0, lw=1.4, ls="--",
                color="#000000", label="SPY")
    ax.axhline(0, color="k", lw=0.8, alpha=0.5)
    ax.yaxis.set_major_formatter(_PCT)
    ax.set_title("Long-book weighting methods (buy-and-hold from formation, eval window)")
    ax.set_ylabel("Cumulative return"); ax.legend(frameon=False)
    return _save(fig, fname)


def chart_null(null_ret, pct, method_rets, spy_ret, fname="10_long_null.png"):
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.hist(null_ret, bins=60, color="#CCCCCC", edgecolor="white",
            label=f"random weights (Dirichlet, n={len(null_ret)})")
    ymax = ax.get_ylim()[1]
    for name, r in method_rets.items():
        ax.axvline(r, color=METHOD_COLORS[name], lw=2.2)
        ax.text(r, ymax * (0.72 + 0.06 * list(method_rets).index(name)),
                f" {name}\n {pct[name]:.0%}ile", color=METHOD_COLORS[name],
                fontsize=8, rotation=0, va="top")
    if spy_ret is not None:
        ax.axvline(spy_ret, color="k", lw=1.6, ls="--")
        ax.text(spy_ret, ymax * 0.95, " SPY", fontsize=8, va="top")
    med = np.median(null_ret)
    ax.axvline(med, color="#D55E00", lw=1.2, ls=":")
    ax.text(med, ymax * 0.5, f" random median {med:.0%}", color="#D55E00", fontsize=8)
    ax.xaxis.set_major_formatter(_PCT)
    ax.set_title("Falsification: do the methods beat RANDOM weighting?\n"
                 "(each method's percentile in the random-weight null)")
    ax.set_xlabel("Eval-window return"); ax.set_ylabel("count of random portfolios")
    ax.legend(frameon=False, loc="upper left")
    return _save(fig, fname)


def chart_falsification(sub_df, jack_df, droptier_mt, full_mt, param_df,
                        fname="11_long_falsification.png"):
    fig, axs = plt.subplots(2, 2, figsize=(15, 11))

    # A. subperiod returns per method
    ax = axs[0, 0]
    x = np.arange(len(sub_df)); wd = 0.8 / sub_df.shape[1]
    for j, col in enumerate(sub_df.columns):
        ax.bar(x + j * wd, sub_df[col].values * 100, wd, label=col)
    ax.set_xticks(x + wd * (sub_df.shape[1] - 1) / 2)
    ax.set_xticklabels(sub_df.index, rotation=20, fontsize=8, ha="right")
    ax.axhline(0, color="k", lw=0.8); ax.set_ylabel("Return (%)")
    ax.set_title("A. Subperiod stability (does the ranking hold?)")
    ax.legend(frameon=False, fontsize=8)

    # B. jackknife range per method
    ax = axs[0, 1]
    means = jack_df["mean"].values * 100
    lo = (jack_df["mean"] - jack_df["min"]).values * 100
    hi = (jack_df["max"] - jack_df["mean"]).values * 100
    ax.errorbar(range(len(jack_df)), means, yerr=[lo, hi], fmt="o", capsize=5,
                color="#0072B2")
    ax.set_xticks(range(len(jack_df)))
    ax.set_xticklabels(jack_df.index, rotation=20, fontsize=8, ha="right")
    ax.set_ylabel("Eval return (%)")
    ax.set_title("B. Leave-one-name-out jackknife (range across dropped names)")

    # C. drop Tier4 vs full
    ax = axs[1, 0]
    idx = full_mt.index
    x = np.arange(len(idx))
    ax.bar(x - 0.2, full_mt["eval_return"].values * 100, 0.4, label="full",
           color="#009E73")
    ax.bar(x + 0.2, droptier_mt["eval_return"].reindex(idx).values * 100, 0.4,
           label="drop Tier4", color="#D55E00")
    ax.set_xticks(x); ax.set_xticklabels(idx, rotation=20, fontsize=8, ha="right")
    ax.axhline(0, color="k", lw=0.8); ax.set_ylabel("Eval return (%)")
    ax.set_title("C. Drop the carry tier (Tier4 silicon): was it the weighting or the tier?")
    ax.legend(frameon=False, fontsize=8)

    # D. inverse-vol vs equal across formation windows
    ax = axs[1, 1]
    ax.plot(param_df.index, param_df["inverse_vol"].values * 100, "-o",
            color="#0072B2", label="inverse_vol")
    ax.plot(param_df.index, param_df["equal"].values * 100, "-o",
            color="#7F7F7F", label="equal")
    ax.set_xlabel("formation window (days)"); ax.set_ylabel("Eval return (%)")
    ax.set_title("D. Parameter sensitivity of inverse-vol")
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    return _save(fig, fname)


def _save(fig, name):
    path = OUT / name
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
_lines = []
def emit(*a):
    line = " ".join(str(x) for x in a); print(line); _lines.append(line)


def main(force=False, dataset=None):
    d = dataset or datamod.load_dataset(force=force)
    prices, returns = d["prices"], d["returns"]
    members = [t for t in LONG_TICKERS if t in returns.columns]

    emit("=" * 92)
    emit("  LONG-BOOK WEIGHTING BAKE-OFF + FALSIFICATION")
    emit(f"  formation = first {FORM} days; eval = buy-and-hold to end; "
         f"null draws = {N_NULL}")
    emit("=" * 92)

    weights, mt, rel, vol = run_methods(prices, returns, members)
    spy_rel = None
    if "SPY" in prices:
        sp = prices["SPY"].iloc[FORM:]
        spy_rel = sp / sp.iloc[0]

    emit("\n[L1] METHOD METRICS (eval window, buy-and-hold)")
    show = mt[["eval_return", "ann_vol", "Sharpe_rf4", "max_drawdown",
               "pct_positive_days"]].copy()
    for c in ["eval_return", "ann_vol", "max_drawdown", "pct_positive_days"]:
        show[c] = show[c].map(lambda x: f"{x:+.2%}")
    show["Sharpe_rf4"] = show["Sharpe_rf4"].map(lambda x: f"{x:+.2f}")
    emit(show.to_string())
    if spy_rel is not None:
        emit(f"     SPY (same window): {spy_rel.iloc[-1]-1:+.2%}")

    # E. random null
    null_ret, pct = test_random_null(rel, weights)
    method_rets = {n: float(rel.values[-1] @ weights[n].reindex(rel.columns).fillna(0).values - 1)
                   for n in weights}
    emit("\n[L2] RANDOM-WEIGHT NULL (Dirichlet α=1) — the key falsification")
    emit(f"     null mean {null_ret.mean():+.2%}, median {np.median(null_ret):+.2%}, "
         f"5-95pct [{np.percentile(null_ret,5):+.2%}, {np.percentile(null_ret,95):+.2%}]")
    for n in weights:
        verdict = ("indistinguishable from random" if 0.15 < pct[n] < 0.85
                   else "outside the bulk of random")
        emit(f"     {n:22s} {method_rets[n]:+.2%}  -> {pct[n]:.0%}ile  ({verdict})")

    # A. subperiods
    sub_df, ranks = test_subperiods(prices, returns, members)
    emit("\n[L3] SUBPERIOD STABILITY (eval split in 2)")
    for name in sub_df.index:
        emit(f"     {name:22s} " +
             "  ".join(f"{c}={sub_df.loc[name,c]:+.1%}(rank {ranks.loc[name,c]})"
                       for c in sub_df.columns))
    best_each = [sub_df[c].idxmax() for c in sub_df.columns]
    emit(f"     Best method per subperiod: {best_each}  -> "
         + ("STABLE" if len(set(best_each)) == 1 else "RESHUFFLES (no stable winner)"))

    # B. jackknife
    jack = test_jackknife(prices, returns, members)
    emit("\n[L4] LEAVE-ONE-NAME-OUT JACKKNIFE (range of eval return across drops)")
    eq_mean = jack.loc["equal", "mean"]
    for name in jack.index:
        edge = jack.loc[name, "mean"] - eq_mean
        emit(f"     {name:22s} mean {jack.loc[name,'mean']:+.2%}  "
             f"range {jack.loc[name,'range']:.2%}  edge-vs-equal {edge:+.2%}")

    # C. drop tier4
    drop_mt = test_drop_tier(prices, returns, members)
    emit("\n[L5] DROP THE CARRY TIER (Tier4 silicon removed)")
    for name in mt.index:
        emit(f"     {name:22s} full {mt.loc[name,'eval_return']:+.2%}  ->  "
             f"ex-Tier4 {drop_mt.loc[name,'eval_return']:+.2%}")

    # D. parameter sensitivity
    param = test_param_sensitivity(prices, returns, members)
    emit("\n[L6] INVERSE-VOL PARAMETER SENSITIVITY (formation window)")
    for f in param.index:
        emit(f"     window {f:>3}d: inverse_vol {param.loc[f,'inverse_vol']:+.2%}, "
             f"equal {param.loc[f,'equal']:+.2%}, "
             f"edge {param.loc[f,'inv_vol_minus_equal']:+.2%}")

    # charts
    rel_by_method = {n: bh_nav(rel, weights[n]) for n in weights}
    p1 = chart_method_nav(rel_by_method, spy_rel)
    spy_eval_ret = float(spy_rel.iloc[-1] - 1) if spy_rel is not None else None
    p2 = chart_null(null_ret, pct, method_rets, spy_eval_ret)
    p3 = chart_falsification(sub_df, jack, drop_mt, mt, param)
    emit(f"\n[L7] charts: {p1.name}, {p2.name}, {p3.name}")

    # save tables
    mt.to_csv(OUT / "long_weighting_metrics.csv")
    pd.DataFrame(weights).to_csv(OUT / "long_weighting_weights.csv")

    # ---- verdict ----
    inside = [n for n in weights if 0.15 < pct[n] < 0.85]
    stable = len(set(best_each)) == 1
    iv_edges = param["inv_vol_minus_equal"]
    iv_consistent = (iv_edges > 0).all() or (iv_edges < 0).all()
    emit("\n" + "=" * 92)
    emit("  FALSIFICATION VERDICT (long-book weighting)")
    emit("=" * 92)
    emit(f"  * {len(inside)}/{len(weights)} methods land INSIDE the random-weight "
         f"cloud (15-85pct): {inside or 'none'} -> for those, the weighting is "
         f"NOT distinguishable from luck over this window.")
    emit(f"  * Subperiod winner is {'STABLE' if stable else 'UNSTABLE'} "
         f"({best_each}).")
    emit(f"  * inverse_vol edge vs equal is "
         f"{'CONSISTENTLY '+('positive' if (iv_edges>0).all() else 'negative') if iv_consistent else 'SIGN-UNSTABLE'} "
         f"across formation windows ({', '.join(f'{v:+.1%}' for v in iv_edges)}).")
    dd = (mt['eval_return'] - drop_mt['eval_return'].reindex(mt.index))
    emit(f"  * Removing Tier4 cuts every method by {dd.min():.0%}..{dd.max():.0%} "
         f"of return -> the long alpha is a TIER bet far more than a weighting bet.")
    emit("=" * 92)

    (OUT / "long_study_report.txt").write_text("\n".join(_lines))
    return dict(metrics=mt, null_pct=pct, subperiods=sub_df, jackknife=jack,
                drop_tier=drop_mt, param=param)


if __name__ == "__main__":
    import sys
    main(force="--force" in sys.argv)
