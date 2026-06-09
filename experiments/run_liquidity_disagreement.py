#!/usr/bin/env python3
"""
Layer DISAGREEMENT as a risk gauge — and as an exposure throttle.

The 4-layer vote averages its strengths, so a 0.5 vote can mean "all four
layers neutral" (benign) or "two screaming risk-on, two screaming risk-off"
(fragile).  The cross-layer dispersion is information the mean throws away.

  disagreement d_t = cross-sectional std of the four causal strengths
                     (liq, vix_ts, credit, dollar), each in [0,1]

Part 1 — is d a *distinct* risk gauge?
  • correlation with VIX (should be low if it's not just vol in disguise)
  • by causal tercile of d (rolling 3y percentile rank, lagged): forward
    realized vol, forward-return left tail, crash frequency
  • the novelty control: within the MIDDLE VIX tercile only, does high
    disagreement still degrade forward returns?  If yes, d adds information
    beyond VIX.

Part 2 — throttle the champion with it.
  confidence c_t = 1 − rank(d_t)   (causal 3y percentile rank)
  throttled exposure = 2 × vote × c_t        on QQQ, honest costs
  Renormalized variant: rescale c_t by its own trailing mean (causal) so the
  long-run risk budget matches the unthrottled champion — the same average
  exposure redistributed toward high-confidence days:
  renorm exposure = 2 × vote × c_t / trailing_mean(c_t), capped at 2x.

All mappings are a-priori (ranks, trailing means — nothing fitted to
returns), keeping the thread's fit-free discipline.

    python experiments/run_liquidity_disagreement.py
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
RANK_WIN = 756
FWD = 126            # 6-month forward window for return diagnostics
CRASH = -0.15        # "crash" = forward 6m return below −15%


def causal_rank(s: pd.Series, win: int = RANK_WIN) -> pd.Series:
    return s.rolling(win, min_periods=252).apply(
        lambda x: (x < x[-1]).mean(), raw=True).shift(1)


def apply_exposure(under_ret: pd.Series, expo: pd.Series):
    expo = expo.clip(0.0, LEV_CAP).fillna(0.0)
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
    qqq = px["QQQ"].reindex(idx)

    base = liquidity_vote(nl, vix["VIX"], vix["VIX3M"],
                          px["HYG"].reindex(idx), px["IEF"].reindex(idx))
    dollar = soft_sign(-px["UUP"].reindex(idx).pct_change(63)).shift(1)
    s4 = pd.concat([base[["liq", "vix_ts", "credit"]], dollar.rename("dollar")], axis=1)
    vote = s4.mean(axis=1)

    disagree = s4.std(axis=1)               # cross-layer dispersion (causal already)
    d_rank = causal_rank(disagree)
    lo = d_rank <= 0.33
    hi = d_rank >= 0.67
    mid = ~lo & ~hi & d_rank.notna()

    # ── Part 1: distinct risk gauge? ─────────────────────────────────────────
    m = disagree.notna() & vix["VIX"].notna()
    corr_vix = float(np.corrcoef(disagree[m], vix["VIX"][m])[0, 1])

    fwd_ret = qqq.pct_change(FWD).shift(-FWD)                     # next-6m QQQ return
    fwd_vol = (qqq_ret.rolling(21).std() * np.sqrt(252)).shift(-21)  # next-21d realized vol

    print("\n" + "=" * 88)
    print(f"  LAYER DISAGREEMENT AS RISK GAUGE  ·  {idx.min().date()} → {idx.max().date()}")
    print(f"  d = cross-layer std of the 4 strengths   ·   corr(d, VIX) = {corr_vix:+.2f}")
    print("=" * 88)
    rows = []
    for lab, mask in [("Low  disagreement", lo), ("Mid", mid), ("High disagreement", hi)]:
        mm = mask & fwd_ret.notna()
        rows.append([lab,
                     f"{float(fwd_vol[mask & fwd_vol.notna()].mean()):.1%}",
                     f"{float(fwd_ret[mm].mean()):+.1%}",
                     f"{float(fwd_ret[mm].quantile(0.05)):+.1%}",
                     f"{float((fwd_ret[mm] < CRASH).mean()):.0%}",
                     int(mask.sum())])
    print(tabulate(rows, headers=["d tercile (causal rank)", "fwd 21d vol",
                                  "fwd 6m ret", "5th pct", f"P(6m<{CRASH:.0%})", "days"],
                   tablefmt="github"))

    # Novelty control: hold VIX fixed (middle tercile), split by disagreement
    v_rank = causal_rank(vix["VIX"])
    v_mid = (v_rank > 0.33) & (v_rank < 0.67)
    print("\n  Control — middle VIX tercile only (vol held ~fixed):")
    rows = []
    for lab, mask in [("low d", lo & v_mid), ("high d", hi & v_mid)]:
        mm = mask & fwd_ret.notna()
        rows.append([lab, f"{float(fwd_ret[mm].mean()):+.1%}",
                     f"{float(fwd_ret[mm].quantile(0.05)):+.1%}",
                     f"{float((fwd_ret[mm] < CRASH).mean()):.0%}", int(mm.sum())])
    print(tabulate(rows, headers=["state", "fwd 6m ret", "5th pct",
                                  f"P(6m<{CRASH:.0%})", "days"], tablefmt="github"))
    print("  (overlapping 6m windows — descriptive, not t-tested)")

    # ── Part 2: throttle the champion ────────────────────────────────────────
    conf = (1.0 - d_rank)                                   # causal confidence
    # renormalize: same trailing-average risk budget, redistributed (causal)
    conf_norm = conf / conf.rolling(RANK_WIN, min_periods=252).mean().shift(1)

    # All variants evaluated from the first date every exposure is live —
    # otherwise the champion banks 2010-2013 while the ranked variants sit in
    # cash during their causal warm-up, biasing the comparison.
    live = conf_norm.first_valid_index()
    expos, rets = {}, {"SPY buy & hold": spy_ret.loc[live:],
                       "QQQ buy & hold": qqq_ret.loc[live:]}
    for name, e in [
        ("2×vote (champion)", 2.0 * vote),
        ("throttled (×conf)", 2.0 * vote * conf),
        ("renormalized (×conf/mean)", 2.0 * vote * conf_norm),
    ]:
        r, ee = apply_exposure(qqq_ret, e)
        rets[f"{name} on QQQ"] = r.loc[live:]
        expos[name] = ee.loc[live:]
    spy_ret = spy_ret.loc[live:]

    print("\n" + "=" * 88)
    print(f"  DISAGREEMENT THROTTLE ON THE 4-LAYER 2×VOTE (QQQ, honest costs)")
    print(f"  common live window {live.date()} → {idx.max().date()}")
    print("=" * 88)
    keys = ["CAGR", "Volatility", "Sharpe", "Sortino", "Max DD", "Calmar"]
    table = []
    for name, r in rets.items():
        mtr = compute_metrics(r.dropna(), rf=RF)
        ret22 = (1 + r[r.index.year == 2022]).prod() - 1
        avg_e = f"{float(expos[name.replace(' on QQQ', '')].mean()):.2f}x" \
            if name.endswith("on QQQ") else "1.00x"
        table.append([name] + [f"{mtr.get(k, float('nan')):.2f}" for k in keys]
                     + [f"{ret22:+.0%}", avg_e])
    print(tabulate(table, headers=["strategy"] + keys + ["2022", "avg expo"],
                   tablefmt="github"))

    print("\n" + "=" * 88)
    print(f"  ROLLING OOS vs SPY  ·  {ROLL}d (~3y) windows")
    print("=" * 88)
    for name in expos:
        r = rets[f"{name} on QQQ"]
        ds = (roll_sharpe(r) - roll_sharpe(spy_ret)).dropna()
        dc = (roll_cagr(r) - roll_cagr(spy_ret)).dropna()
        print(f"  {name:28s} CAGR win {float((dc > 0).mean()):.0%} "
              f"(median {float(dc.median()):+.1%})   "
              f"Sharpe win {float((ds > 0).mean()):.0%} "
              f"(median {float(ds.median()):+.2f})")

    yr = pd.DataFrame({n: rets[f"{n} on QQQ"] for n in expos} | {"SPY": spy_ret}).dropna()
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
    ax.set_title("Disagreement-throttled vote vs champion / buy & hold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(disagree.index, disagree.values, lw=0.6, color="tab:orange", alpha=0.5,
            label="disagreement d (cross-layer std)")
    ax.plot(d_rank.index, d_rank.values * disagree.max(), lw=0.8, color="tab:red",
            alpha=0.7, label="causal 3y rank of d (scaled)")
    ax.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31"),
               color="red", alpha=0.08, label="2022 grind")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("Cross-layer disagreement")

    ax = axes[2]
    for name in ["2×vote (champion)", "renormalized (×conf/mean)"]:
        ax.plot(expos[name].index, expos[name].values, lw=0.7, alpha=0.85, label=name)
    ax.axhline(1.0, color="grey", lw=0.6)
    ax.set_ylabel("exposure (x)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("Exposure — same long-run budget, redistributed toward agreement")
    fig.tight_layout()
    out = RESULTS_DIR / "liquidity_disagreement.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    logger.info("Saved: %s", out)
    print("\n✓  Done.")


if __name__ == "__main__":
    main()
