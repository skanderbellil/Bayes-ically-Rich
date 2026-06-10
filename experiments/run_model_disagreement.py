#!/usr/bin/env python3
"""
Cross-MODEL disagreement — mining the strategy graveyard for a regime gauge.

The repo has accumulated a dozen models (BOCPD, HMM, Hurst/trend, intramonth,
PEAD, market-neutral, liquidity).  Most "failed" to beat SPY standalone and
sit in a drawer.  But nobody asked: does their DISAGREEMENT predict regime?
When the trend sleeve and the mean-reversion sleeve violently disagree,
that's a transition tell.  Cross-model dispersion of our own model-implied
exposures is a free, proprietary "model-implied uncertainty" gauge.

Five graveyard models emit a causal daily risk stance in [0,1] on QQQ:

  liquidity : the 4-layer macro vote (liq, VIX-TS, credit, dollar)
  trend     : soft-signed 126-day return            (the Hurst-thread trend leg)
  meanrev   : soft-signed NEGATIVE 21-day return    (fade the short-term move)
  bocpd     : causal 3y rank of the BOCPD expected run length (regime age —
              long stable regime = risk-on, fresh changepoint = risk-off)
  hmm       : 3-state Gaussian HMM p(bull) + ½·p(sideways), refit every 21d
              on a trailing 3y window, forward-filtered (walk-forward, causal)

(PEAD and market-neutral are event-driven / cross-sectional — they have no
daily aggregate market stance, so they stay in the drawer.)

  disagreement D_t = cross-model std of the five stances
  pair tell        = |trend − meanrev|   (the user-named suspect pair)

Part 1 — is D a distinct risk gauge?  corr with VIX, forward vol / return /
         left tail / crash frequency by causal tercile, and the novelty
         control: does high D still degrade forward returns when VIX is held
         in its middle tercile?
Part 2 — the transition tell: P(QQQ 200-day-MA trend state flips within the
         next 21 days) by D tercile, and the same for the trend/meanrev pair.
Part 3 — throttle the champion (2×4-layer vote on QQQ, honest costs) by
         model agreement: conf = 1 − rank(D), plus the causally renormalized
         version (conf divided by its trailing 3y mean, capped at 2×) that
         redistributes the same risk budget toward high-agreement days.

All mappings are a-priori (soft signs, ranks, trailing means — nothing fitted
to returns).

    python experiments/run_model_disagreement.py
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
from posterioralpha.research import HMM3, liquidity_vote, precompute_bocpd, soft_sign
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
HMM_WIN = 756
HMM_REFIT = 21
FWD = 126
CRASH = -0.15
FLIP_H = 21          # "transition within the next 21 days"


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


def hmm_stance(ret: pd.Series) -> pd.Series:
    """Walk-forward HMM bull-stance: refit every HMM_REFIT days on a trailing
    HMM_WIN window, forward-filtered p(bull) + ½·p(sideways) at the fit date,
    held until the next refit.  Only past data enters each fit."""
    vals = ret.values.reshape(-1, 1)
    out = pd.Series(np.nan, index=ret.index)
    for t in range(HMM_WIN, len(ret), HMM_REFIT):
        window = vals[t - HMM_WIN:t + 1]
        model = HMM3().fit(window)
        p_bull, p_side, _ = model.regime_probs(window)
        out.iloc[t] = p_bull + 0.5 * p_side
    return out.ffill()


def main() -> None:
    nl = load_net_liquidity()["net_liquidity"]
    px = load_etf_universe_prices()
    vix = pd.read_csv(VIX_CSV, parse_dates=["Date"], index_col="Date").sort_index()
    idx = (nl.dropna().index
           .intersection(px[["SPY", "QQQ", "HYG", "IEF", "UUP"]].dropna().index)
           .intersection(vix.dropna().index))
    nl, vix = nl.reindex(idx), vix.reindex(idx)
    qqq = px["QQQ"].reindex(idx)
    qqq_ret = qqq.pct_change()
    spy_ret = px["SPY"].reindex(idx).pct_change()

    # ── The five graveyard stances ───────────────────────────────────────────
    base = liquidity_vote(nl, vix["VIX"], vix["VIX3M"],
                          px["HYG"].reindex(idx), px["IEF"].reindex(idx))
    dollar = soft_sign(-px["UUP"].reindex(idx).pct_change(63)).shift(1)
    s4 = pd.concat([base[["liq", "vix_ts", "credit"]], dollar.rename("dollar")], axis=1)
    vote = s4.mean(axis=1)                                  # the champion's dial

    logger.info("Computing BOCPD expected run length …")
    _, erl = precompute_bocpd(qqq_ret.fillna(0.0).values)
    logger.info("Walk-forward HMM (refit every %dd) …", HMM_REFIT)
    stances = pd.DataFrame({
        "liquidity": vote,
        "trend":     soft_sign(qqq.pct_change(126)).shift(1),
        "meanrev":   soft_sign(-qqq.pct_change(21)).shift(1),
        "bocpd":     causal_rank(pd.Series(erl, index=idx)),
        "hmm":       hmm_stance(qqq_ret).shift(1),
    })

    full = stances.notna().all(axis=1)
    disagree = stances.std(axis=1).where(full)
    pair = (stances["trend"] - stances["meanrev"]).abs().where(full)
    conviction = (stances - 0.5).abs().mean(axis=1).where(full)
    d_rank = causal_rank(disagree)
    p_rank = causal_rank(pair)
    c_rank = causal_rank(conviction)
    lo, hi = d_rank <= 0.33, d_rank >= 0.67
    mid = ~lo & ~hi & d_rank.notna()

    corr_m = stances.loc[full].corr().round(2)
    print("\nCross-model stance correlations (are these really different models?):")
    print(tabulate(corr_m, headers=corr_m.columns, tablefmt="github"))

    # ── Part 1: distinct risk gauge? ─────────────────────────────────────────
    m = disagree.notna() & vix["VIX"].notna()
    corr_vix = float(np.corrcoef(disagree[m], vix["VIX"][m])[0, 1])
    fwd_ret = qqq.pct_change(FWD).shift(-FWD)
    fwd_vol = (qqq_ret.rolling(21).std() * np.sqrt(252)).shift(-21)

    print("\n" + "=" * 88)
    print(f"  CROSS-MODEL DISAGREEMENT AS RISK GAUGE  ·  "
          f"{disagree.dropna().index.min().date()} → {idx.max().date()}")
    print(f"  D = std of 5 model stances   ·   corr(D, VIX) = {corr_vix:+.2f}")
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
    print(tabulate(rows, headers=["D tercile (causal rank)", "fwd 21d vol",
                                  "fwd 6m ret", "5th pct", f"P(6m<{CRASH:.0%})", "days"],
                   tablefmt="github"))

    v_rank = causal_rank(vix["VIX"])
    v_mid = (v_rank > 0.33) & (v_rank < 0.67)
    print("\n  Control — middle VIX tercile only (vol held ~fixed):")
    rows = []
    for lab, mask in [("low D", lo & v_mid), ("high D", hi & v_mid)]:
        mm = mask & fwd_ret.notna()
        rows.append([lab, f"{float(fwd_ret[mm].mean()):+.1%}",
                     f"{float(fwd_ret[mm].quantile(0.05)):+.1%}",
                     f"{float((fwd_ret[mm] < CRASH).mean()):.0%}", int(mm.sum())])
    print(tabulate(rows, headers=["state", "fwd 6m ret", "5th pct",
                                  f"P(6m<{CRASH:.0%})", "days"], tablefmt="github"))
    print("  (overlapping 6m windows — descriptive, not t-tested)")

    # ── Part 2: the transition tell ──────────────────────────────────────────
    state = (qqq > qqq.rolling(200).mean()).astype(int)
    flip = state.diff().abs() == 1
    flip_ahead = (flip.shift(-1).rolling(FLIP_H).max()           # any flip in (t, t+21]
                  .shift(-(FLIP_H - 1)).fillna(0).astype(bool))
    print("\n" + "=" * 88)
    print(f"  TRANSITION TELL  ·  P(QQQ 200d-MA trend state flips within {FLIP_H}d)")
    print("=" * 88)
    rows = []
    base_rate = float(flip_ahead[d_rank.notna()].mean())
    for lab, rk in [("model disagreement D", d_rank), ("|trend − meanrev| pair", p_rank),
                    ("conviction mean|s−½| (post-hoc)", c_rank)]:
        cells = [lab]
        for name, mask in [("low", rk <= 0.33), ("mid", (rk > 0.33) & (rk < 0.67)),
                           ("high", rk >= 0.67)]:
            cells.append(f"{float(flip_ahead[mask].mean()):.1%}")
        rows.append(cells)
    print(tabulate(rows, headers=["gauge tercile", "low", "mid", "high"],
                   tablefmt="github"))
    print(f"  unconditional flip rate: {base_rate:.1%}")

    # ── Part 3: throttle the champion ────────────────────────────────────────
    conf = 1.0 - d_rank
    conf_norm = conf / conf.rolling(RANK_WIN, min_periods=252).mean().shift(1)

    live = conf_norm.first_valid_index()
    expos, rets = {}, {"SPY buy & hold": spy_ret.loc[live:],
                       "QQQ buy & hold": qqq_ret.loc[live:]}
    for name, e in [
        ("2×vote (champion)", 2.0 * vote),
        ("throttled (×conf)", 2.0 * vote * conf),
        ("renormalized (×conf/mean)", 2.0 * vote * conf_norm),
        # Sign chosen AFTER seeing Part 1/2 invert the hypothesis — post-hoc,
        # reported for completeness, not a validated strategy.
        ("inverted (×rank D, post-hoc)", 2.0 * vote * d_rank),
    ]:
        r, ee = apply_exposure(qqq_ret, e)
        rets[f"{name} on QQQ"] = r.loc[live:]
        expos[name] = ee.loc[live:]
    spy_live = spy_ret.loc[live:]

    print("\n" + "=" * 88)
    print("  MODEL-AGREEMENT THROTTLE ON THE 4-LAYER 2×VOTE (QQQ, honest costs)")
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
        ds = (roll_sharpe(r) - roll_sharpe(spy_live)).dropna()
        dc = (roll_cagr(r) - roll_cagr(spy_live)).dropna()
        print(f"  {name:28s} CAGR win {float((dc > 0).mean()):.0%} "
              f"(median {float(dc.median()):+.1%})   "
              f"Sharpe win {float((ds > 0).mean()):.0%} "
              f"(median {float(ds.median()):+.2f})")

    yr = pd.DataFrame({n: rets[f"{n} on QQQ"] for n in expos} | {"SPY": spy_live}).dropna()
    ann = yr.groupby(yr.index.year).apply(lambda d: (1 + d).prod() - 1)
    print("\nPer-year returns (%):")
    print(tabulate((ann * 100).round(1), headers=["year"] + list(ann.columns),
                   tablefmt="github"))

    # ── Graphs ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    ax = axes[0]
    for name, r in rets.items():
        eqc = (1 + r.fillna(0)).cumprod()
        ax.plot(eqc.index, eqc.values, lw=1.6 if "buy" in name else 1.3,
                color="black" if name.startswith("SPY") else
                ("grey" if name.startswith("QQQ") else None),
                alpha=0.9, label=name)
    ax.set_yscale("log"); ax.set_ylabel("growth of $1 (log)")
    ax.set_title("Model-agreement throttled vote vs champion / buy & hold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    for c in stances.columns:
        ax.plot(stances.index, stances[c].values, lw=0.6, alpha=0.6, label=c)
    ax.legend(fontsize=7, ncol=5); ax.grid(alpha=0.3)
    ax.set_ylabel("stance [0,1]")
    ax.set_title("The five graveyard-model stances")

    ax = axes[2]
    ax.plot(disagree.index, disagree.values, lw=0.7, color="tab:orange",
            alpha=0.8, label="cross-model disagreement D")
    flips = state.index[flip & d_rank.notna()]
    for i, f in enumerate(flips):
        ax.axvline(f, color="tab:red", lw=0.5, alpha=0.4,
                   label="200d-MA trend flip" if i == 0 else None)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("Disagreement vs trend-state flips — does D spike into transitions?")
    fig.tight_layout()
    out = RESULTS_DIR / "model_disagreement.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    logger.info("Saved: %s", out)
    print("\n✓  Done.")


if __name__ == "__main__":
    main()
