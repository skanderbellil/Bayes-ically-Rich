#!/usr/bin/env python3
"""
Regime-weighted liquidity vote — cashing in the interaction finding.

run_liquidity_interaction.py showed liquidity's lead over forward equity
returns is concentrated in FRAGILE states (corr +0.34 in the high-VIX tercile)
and near-zero/negative when calm (−0.11).  Its conclusion: the actionable use
is regime-dependent signal *weighting* inside the vote, not a standalone
switch.  This experiment builds exactly that.

The fragility state is the causal VIX rolling-3y percentile rank f_t ∈ [0,1]
(same variable as the interaction study, lagged one day).  The liquidity
layer's weight inside the 4-layer vote becomes a function of f_t — the other
three layers (VIX-TS, credit, dollar) keep weight 1:

  equal     : w_liq = 1          (current champion, the baseline)
  calm-mute : w_liq = f_t        (liquidity fades to zero when calm, never
                                  exceeds equal weight)
  centered  : w_liq = 2·f_t      (median fragility → equal weight; calm → 0,
                                  fragile → double weight)

Both mappings are a-priori (identity / doubled identity on the rank — no
constants fitted to returns), keeping the thread's fit-free discipline.
Compared on the champion configuration: 2×vote on QQQ with honest
financing / trading costs, rolling 3y OOS win rates vs SPY, per-year table.

    python experiments/run_liquidity_regime_weighted.py
"""
import logging
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tabulate import tabulate

import _bootstrap  # noqa: F401
from posterioralpha.data import load_net_liquidity, load_etf_universe_prices
from posterioralpha.research import liquidity_vote, soft_sign
from posterioralpha.validation import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)
logger = logging.getLogger(__name__)
RESULTS_DIR = _bootstrap.ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
VIX_CSV = _bootstrap.ROOT / "datasets" / "vix_term_structure.csv"

RF = 0.04
DAILY_RF = RF / 252
BORROW_SPREAD = 0.005
TC = 0.0002
LEV_CAP = 2.0
ROLL = 756


def apply_2xvote(under_ret: pd.Series, vote: pd.Series):
    expo = (2.0 * vote).clip(0.0, LEV_CAP).fillna(0.0)
    gross = expo * under_ret
    financing = np.maximum(expo - 1.0, 0.0) * (RF + BORROW_SPREAD) / 252
    cash = np.maximum(1.0 - expo, 0.0) * DAILY_RF
    costs = TC * expo.diff().abs().fillna(0.0)
    return pd.Series(gross - financing + cash - costs, index=under_ret.index), expo


def roll_sharpe(r):
    ex = r - DAILY_RF
    return (ex.rolling(ROLL).mean() / ex.rolling(ROLL).std()) * np.sqrt(252)


def roll_cagr(r):
    return (1 + r).rolling(ROLL).apply(np.prod, raw=True) ** (252 / ROLL) - 1


def weighted_vote(strengths: pd.DataFrame, w_liq: pd.Series) -> pd.Series:
    """Mean of the four strengths with a time-varying liquidity weight.

    NaN strengths drop out of both numerator and denominator (same behaviour
    as the equal-weight ``mean(axis=1)`` of the earlier scripts)."""
    w = pd.DataFrame(1.0, index=strengths.index, columns=strengths.columns)
    w["liq"] = w_liq
    w = w.where(strengths.notna())
    return (strengths * w).sum(axis=1) / w.sum(axis=1)


def main() -> None:
    nl = load_net_liquidity()["net_liquidity"]
    px = load_etf_universe_prices()
    vix = pd.read_csv(VIX_CSV, parse_dates=["Date"], index_col="Date").sort_index()
    idx = (nl.dropna().index
           .intersection(px[["SPY", "QQQ", "HYG", "IEF", "UUP"]].dropna().index)
           .intersection(vix.dropna().index))
    nl, vix = nl.reindex(idx), vix.reindex(idx)
    qqq_ret = px["QQQ"].reindex(idx).pct_change()
    spy_ret = px["SPY"].reindex(idx).pct_change()

    # 4 layer strengths (causal: publication lag + 1d shift inside/below)
    base = liquidity_vote(nl, vix["VIX"], vix["VIX3M"],
                          px["HYG"].reindex(idx), px["IEF"].reindex(idx))
    dollar = soft_sign(-px["UUP"].reindex(idx).pct_change(63)).shift(1)
    s4 = pd.concat([base[["liq", "vix_ts", "credit"]], dollar.rename("dollar")], axis=1)

    # Fragility state: causal VIX rolling-3y percentile rank (interaction study)
    frag = vix["VIX"].rolling(756, min_periods=252).apply(
        lambda x: (x < x[-1]).mean(), raw=True).shift(1)

    votes = {
        "4-layer equal":              s4.mean(axis=1),
        "calm-mute  (w_liq = f)":     weighted_vote(s4, frag),
        "centered   (w_liq = 2f)":    weighted_vote(s4, 2.0 * frag),
    }

    rets = {"SPY buy & hold": spy_ret, "QQQ buy & hold": qqq_ret}
    expos = {}
    for name, v in votes.items():
        r, e = apply_2xvote(qqq_ret, v)
        rets[f"2×{name} on QQQ"] = r
        expos[name] = e

    # ── Metrics ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 88)
    print(f"  REGIME-WEIGHTED LIQUIDITY VOTE (2× on QQQ)  ·  "
          f"{idx.min().date()} → {idx.max().date()}")
    print("  liquidity layer weight = causal VIX-percentile fragility rank")
    print("=" * 88)
    keys = ["CAGR", "Volatility", "Sharpe", "Sortino", "Max DD", "Calmar"]
    table = []
    for name, r in rets.items():
        m = compute_metrics(r.dropna(), rf=RF)
        ret22 = (1 + r[r.index.year == 2022]).prod() - 1
        table.append([name] + [f"{m.get(k, float('nan')):.2f}" for k in keys]
                     + [f"{ret22:+.0%}"])
    print(tabulate(table, headers=["strategy"] + keys + ["2022"], tablefmt="github"))

    # ── Rolling OOS vs SPY ───────────────────────────────────────────────────
    print("\n" + "=" * 88)
    print(f"  ROLLING OOS vs SPY  ·  {ROLL}d (~3y) windows")
    print("=" * 88)
    for name in votes:
        r = rets[f"2×{name} on QQQ"]
        ds = (roll_sharpe(r) - roll_sharpe(spy_ret)).dropna()
        dc = (roll_cagr(r) - roll_cagr(spy_ret)).dropna()
        print(f"  {name:26s} CAGR win {float((dc > 0).mean()):.0%} "
              f"(median {float(dc.median()):+.1%})   "
              f"Sharpe win {float((ds > 0).mean()):.0%} "
              f"(median {float(ds.median()):+.2f})")

    # ── Per-year ─────────────────────────────────────────────────────────────
    yr = pd.DataFrame({n: rets[f"2×{n} on QQQ"] for n in votes} | {"SPY": spy_ret}).dropna()
    ann = yr.groupby(yr.index.year).apply(lambda d: (1 + d).prod() - 1)
    print("\nPer-year returns (%):")
    print(tabulate((ann * 100).round(1), headers=["year"] + list(ann.columns),
                   tablefmt="github"))

    # ── Graphs ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    ax = axes[0]
    for name, r in rets.items():
        eqc = (1 + r.reindex(idx).fillna(0)).cumprod()
        ax.plot(eqc.index, eqc.values, lw=1.6 if "buy" in name else 1.3,
                color="black" if name.startswith("SPY") else
                ("grey" if name.startswith("QQQ") else None),
                alpha=0.9, label=name)
    ax.set_yscale("log"); ax.set_ylabel("growth of $1 (log)")
    ax.set_title("Regime-weighted liquidity vote (2× on QQQ) vs equal-weight champion")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    for name in votes:
        dc = roll_cagr(rets[f"2×{name} on QQQ"]) - roll_cagr(spy_ret)
        ax.plot(dc.index, dc.values * 100, lw=1.2, label=name)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_ylabel("rolling 3y CAGR edge\nvs SPY (pp)"); ax.legend(fontsize=8)
    ax.set_title("Rolling 3-year CAGR edge over SPY"); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(frag.index, frag.values, lw=0.7, color="tab:red", alpha=0.8,
            label="fragility f (VIX 3y pct rank)")
    dexpo = expos["centered   (w_liq = 2f)"] - expos["4-layer equal"]
    ax.plot(dexpo.index, dexpo.values, lw=0.8, color="tab:purple",
            label="exposure diff: centered − equal (x)")
    ax.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31"),
               color="red", alpha=0.08, label="2022 grind")
    ax.axhline(0, color="grey", lw=0.6)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("Fragility state and where the re-weighting changes exposure")
    fig.tight_layout()
    out = RESULTS_DIR / "liquidity_regime_weighted.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    logger.info("Saved: %s", out)
    print("\n✓  Done.")


if __name__ == "__main__":
    main()
