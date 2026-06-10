#!/usr/bin/env python3
"""
Flip-risk as a conviction modulator — the right use of a validated tell.

run_disagreement_fresh settled the model-dispersion question: as a RETURN
signal it's dead (the inverted throttle degrades Sharpe in every fresh
window), but as a TRANSITION signal it's robust — P(200d-MA flip within 21d)
is monotone-decreasing in dispersion across 30 years and two assets.
Indecision (all models drifting neutral together) precedes regime flips.

A flip probability is direction-agnostic: it says "the current state won't
last", not "returns will be bad".  The throttle that died shrank exposure
toward CASH — a directional bet.  The direction-agnostic use is to shrink
the exposure toward NEUTRAL (1× market):

  flip_risk f = 1 − causal rank of D     (low dispersion = high flip risk)

  neutral-shrink :  e' = 1 + (e − 1)·(1 − f)      e = 2×vote, the champion
                    f high → hold the market;  f low → let the dial work
  slow-down      :  e' = 2×[f·MA63(vote) + (1−f)·vote]
                    f high → trade the smoothed vote (transitions are choppy;
                    damp the whipsaw), f low → trade the crisp vote
  dead throttle  :  e' = e·(1 − f)   (the killed version, shown as the
                    reference point this must NOT collapse into)

D uses the same 4 price-only stances as the fresh validation (trend,
meanrev, BOCPD ERL rank, walk-forward HMM), computed on QQQ; the vote is
the 5-layer champion; GFC-inclusive stress-panel sample, honest costs.
Pass bar: tail/turnover improvement at preserved Sharpe, broad across years.

    python experiments/run_flip_risk_modulator.py
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
from posterioralpha.data import load_net_liquidity
from posterioralpha.research import HMM3, precompute_bocpd, soft_sign
from posterioralpha.validation import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)
logger = logging.getLogger(__name__)
RESULTS_DIR = _bootstrap.ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
PANEL_CSV = _bootstrap.ROOT / "datasets" / "overnight_panel.csv.gz"
STRESS_CSV = _bootstrap.ROOT / "datasets" / "stress_panel.csv.gz"

RF = 0.04
DAILY_RF = RF / 252
BORROW_SPREAD = 0.005
TC = 0.0002
LEV_CAP = 2.0
ROLL = 756
H_LIQ, PUB_LAG, CREDIT_WIN = 126, 5, 21
RANK_WIN = 756
HMM_WIN, HMM_REFIT = 756, 21


def causal_rank(s: pd.Series, win: int = RANK_WIN) -> pd.Series:
    return s.rolling(win, min_periods=252).apply(
        lambda x: (x < x[-1]).mean(), raw=True).shift(1)


def hmm_stance(ret: pd.Series) -> pd.Series:
    vals = ret.values.reshape(-1, 1)
    out = pd.Series(np.nan, index=ret.index)
    for t in range(HMM_WIN, len(ret), HMM_REFIT):
        window = vals[t - HMM_WIN:t + 1]
        model = HMM3().fit(window)
        p_bull, p_side, _ = model.regime_probs(window)
        out.iloc[t] = p_bull + 0.5 * p_side
    return out.ffill()


def roll_sharpe(r):
    ex = r - DAILY_RF
    return (ex.rolling(ROLL).mean() / ex.rolling(ROLL).std()) * np.sqrt(252)


def roll_cagr(r):
    return (1 + r).rolling(ROLL).apply(np.prod, raw=True) ** (252 / ROLL) - 1


def main() -> None:
    nl = load_net_liquidity()["net_liquidity"]
    p = pd.read_csv(STRESS_CSV, parse_dates=["Date"], index_col="Date").sort_index()
    panel = pd.read_csv(PANEL_CSV, parse_dates=["Date"], index_col="Date").sort_index()
    idx = (nl.dropna().index
           .intersection(p[["SPY", "QQQ", "HYG", "IEF", "UUP", "VIX", "VIX3M"]]
                         .dropna().index)
           .intersection(panel.dropna().index))
    nl, p = nl.reindex(idx), p.reindex(idx)
    qqq = p["QQQ"]
    r_cc = qqq.pct_change()
    spy_ret = p["SPY"].pct_change()

    # ── 5-layer champion vote ────────────────────────────────────────────────
    liq = soft_sign(nl.diff(H_LIQ).shift(PUB_LAG)).shift(1)
    vix_ts = soft_sign(p["VIX3M"] / p["VIX"] - 1.0).shift(1)
    credit = soft_sign(p["HYG"].pct_change(CREDIT_WIN)
                       - p["IEF"].pct_change(CREDIT_WIN)).shift(1)
    dollar = soft_sign(-p["UUP"].pct_change(63)).shift(1)
    q_o, q_c = panel["QQQ_Open"].reindex(idx), panel["QQQ_Close"].reindex(idx)
    r_on, r_id = q_o / q_c.shift(1) - 1.0, q_c / q_o - 1.0
    tug = soft_sign(((1 + r_on).rolling(21).apply(np.prod, raw=True)
                     - (1 + r_id).rolling(21).apply(np.prod, raw=True))).shift(1)
    vote = pd.DataFrame({"liq": liq, "vix_ts": vix_ts, "credit": credit,
                         "dollar": dollar, "tug": tug}).mean(axis=1)

    # ── Flip risk from the validated dispersion gauge ────────────────────────
    logger.info("BOCPD ERL …")
    _, erl = precompute_bocpd(r_cc.fillna(0.0).values)
    logger.info("Walk-forward HMM …")
    stances = pd.DataFrame({
        "trend":   soft_sign(qqq.pct_change(126)).shift(1),
        "meanrev": soft_sign(-qqq.pct_change(21)).shift(1),
        "bocpd":   causal_rank(pd.Series(erl, index=idx)),
        "hmm":     hmm_stance(r_cc).shift(1),
    })
    D = stances.std(axis=1).where(stances.notna().all(axis=1))
    f = 1.0 - causal_rank(D)          # high f = high flip risk (indecision)

    # sanity: the tell on this sample
    state = (qqq > qqq.rolling(200).mean()).astype(int)
    flip = state.diff().abs() == 1
    flip_ahead = (flip.shift(-1).rolling(21).max()
                  .shift(-20).fillna(0).astype(bool))
    rows = []
    for lab, m in [("low f", f <= 0.33), ("mid", (f > 0.33) & (f < 0.67)),
                   ("high f", f >= 0.67)]:
        rows.append([lab, f"{float(flip_ahead[m].mean()):.1%}", int(m.sum())])
    print("\n" + "=" * 96)
    print(f"  FLIP-RISK CONVICTION MODULATOR  ·  QQQ  ·  "
          f"{f.dropna().index[0].date()} → {idx[-1].date()}")
    print("=" * 96)
    print("\n  Sanity — P(200d-MA flip within 21d) by flip-risk tercile "
          f"(uncond {float(flip_ahead[f.notna()].mean()):.1%}):")
    print(tabulate(rows, headers=["f tercile", "flip prob", "days"],
                   tablefmt="github"))

    # ── Variants ─────────────────────────────────────────────────────────────
    vote_ma = vote.rolling(63).mean()
    e_champ = 2.0 * vote
    variants = {
        "champion (2×vote)": e_champ,
        "neutral-shrink: 1+(e−1)(1−f)": 1.0 + (e_champ - 1.0) * (1.0 - f),
        "slow-down: 2×[f·MA63 + (1−f)·vote]": 2.0 * (f * vote_ma + (1.0 - f) * vote),
        "dead throttle: e×(1−f) (reference)": e_champ * (1.0 - f),
    }

    def run(expo):
        expo = expo.clip(0.0, LEV_CAP).fillna(0.0)
        gross = expo * r_cc
        financing = np.maximum(expo - 1.0, 0.0) * (RF + BORROW_SPREAD) / 252
        cash = np.maximum(1.0 - expo, 0.0) * DAILY_RF
        costs = TC * expo.diff().abs().fillna(0.0)
        return gross - financing + cash - costs, expo

    live = max([v.first_valid_index() for v in variants.values()])
    strat, turns = {"SPY buy & hold": spy_ret.loc[live:],
                    "QQQ buy & hold": r_cc.loc[live:]}, {}
    for name, e in variants.items():
        r, ee = run(e)
        strat[name] = r.loc[live:]
        turns[name] = float(ee.loc[live:].diff().abs().mean()) * 252
    spy_live = spy_ret.loc[live:]

    print(f"\n  Head-to-head  ·  honest costs  ·  common live {live.date()}:")
    keys = ["CAGR", "Volatility", "Sharpe", "Sortino", "Max DD", "Calmar"]
    table = []
    for name, r in strat.items():
        m = compute_metrics(r.dropna(), rf=RF)
        gfc = (1 + r.loc["2008-09-01":"2009-06-30"]).prod() - 1
        ret22 = (1 + r[r.index.year == 2022]).prod() - 1
        dc = (roll_cagr(r) - roll_cagr(spy_live)).dropna()
        ds = (roll_sharpe(r) - roll_sharpe(spy_live)).dropna()
        tn = f"{turns[name]:.1f}x" if name in turns else "—"
        table.append([name] + [f"{m.get(k, float('nan')):.2f}" for k in keys]
                     + [f"{gfc:+.0%}", f"{ret22:+.0%}",
                        f"{float((dc > 0).mean()):.0%}",
                        f"{float((ds > 0).mean()):.0%}", tn])
    print(tabulate(table, headers=["strategy"] + keys
                   + ["GFC", "2022", "CAGRwin", "SRwin", "turn/yr"],
                   tablefmt="github"))
    print("  win rates vs SPY, rolling 3y windows  ·  turn/yr = mean |Δexposure|×252")

    yr = pd.DataFrame({n: strat[n] for n in variants} | {"QQQ": r_cc.loc[live:]}).dropna()
    ann = yr.groupby(yr.index.year).apply(lambda d: (1 + d).prod() - 1)
    print("\nPer-year returns (%):")
    print(tabulate((ann * 100).round(1),
                   headers=["year"] + [c[:18] for c in ann.columns],
                   tablefmt="github"))

    # ── Graph ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    ax = axes[0]
    for name, r in strat.items():
        eq = (1 + r.fillna(0)).cumprod()
        ax.plot(eq.index, eq.values, lw=1.6 if "neutral" in name else 1.1,
                color="black" if name.startswith("SPY") else
                ("grey" if name.startswith("QQQ") else None), label=name)
    ax.set_yscale("log"); ax.set_ylabel("growth of $1 (log)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("Direction-agnostic flip-risk modulation vs the champion")

    ax = axes[1]
    ax.plot(f.index, f.rolling(21).mean(), lw=0.9, color="tab:purple",
            label="flip risk f (21d MA)")
    flips = qqq.index[flip & f.notna()]
    for i, d in enumerate(flips):
        ax.axvline(d, color="tab:red", lw=0.5, alpha=0.35,
                   label="200d-MA flip" if i == 0 else None)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("Flip risk (model indecision) vs realized trend flips")

    ax = axes[2]
    for name in ["champion (2×vote)", "neutral-shrink: 1+(e−1)(1−f)"]:
        eq = (1 + strat[name].fillna(0)).cumprod()
        dd = eq / eq.cummax() - 1
        ax.plot(dd.index, dd.values, lw=1.1, label=name)
    ax.set_ylabel("drawdown"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("Drawdowns")
    fig.tight_layout()
    out = RESULTS_DIR / "flip_risk_modulator.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    logger.info("Saved: %s", out)
    print("\n✓  Done.")


if __name__ == "__main__":
    main()
