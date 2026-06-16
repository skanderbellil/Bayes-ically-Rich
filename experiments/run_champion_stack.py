#!/usr/bin/env python3
"""
The champion stack — freeze the spec, verify the pieces stack jointly.

This session validated three upgrades to the 5-layer champion, each tested
against it SEPARATELY: the debt-ceiling ×½ (mechanism-justified exposure
halving), the dollar slack (risk-off cash → UUP, promoted through three
gates), and the flip-risk slow-down (trade the smoothed vote when the
models go indecisive — same returns, 41% less turnover).  Interactions
could break any of them: the ceiling grows the slack exactly when the
slow-down is damping the vote.  This study runs the ablation ladder and
the leave-one-out matrix to verify each piece earns its place JOINTLY,
then freezes the reference configuration.

  FROZEN SPEC (all parameters inherited, nothing re-fitted here)
  --------------------------------------------------------------
  vote        equal-weight mean of 5 soft-signed layers:
              liq    Δ126d net liquidity, pub-lag 5
              vix_ts VIX3M/VIX − 1
              credit HYG−IEF 21d relative return
              dollar −UUP 63d return
              tug    21d overnight−intraday compounded spread (per asset)
  slow-down   f = 1 − causal-rank(std of 4 price stances);
              vote_eff = f·MA63(vote) + (1−f)·vote
  exposure    e = 2×vote_eff, capped [0, 2]
  ceiling     e ×½ for 60bd after each debt-ceiling resolution (list
              hardcoded through Jun-2023 — APPEND NEW DEALS)
  slack       max(1−e, 0) held in UUP (not cash)
  costs       TC 2bp on |Δe| + |Δw_UUP|, financing RF+50bp on e>1

The slow-down's warm-up (HMM + dispersion rank) pushes the joint ladder's
common live date to ~2011-10; the GFC behaviour of the un-modulated stack
is documented in run_dollar_slack (GFC −3%).

    python experiments/run_champion_stack.py
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
LEV_CSV = _bootstrap.ROOT / "datasets" / "levered_etfs.csv.gz"

RF = 0.04
DAILY_RF = RF / 252
BORROW_SPREAD = 0.005
TC = 0.0002
LEV_CAP = 2.0
ROLL = 756
H_LIQ, PUB_LAG, CREDIT_WIN = 126, 5, 21
RANK_WIN, HMM_WIN, HMM_REFIT = 756, 756, 21

DEBT_CEILING = ["2011-08-02", "2013-10-17", "2014-02-15", "2015-11-02",
                "2017-09-08", "2018-02-09", "2019-08-02", "2021-10-14",
                "2021-12-16", "2023-06-03"]   # complete through Jun-2023


def causal_rank(s, win=RANK_WIN):
    return s.rolling(win, min_periods=252).apply(
        lambda x: (x < x[-1]).mean(), raw=True).shift(1)


def hmm_stance(ret):
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
    lev = pd.read_csv(LEV_CSV, parse_dates=["Date"], index_col="Date").sort_index()
    idx = (nl.dropna().index
           .intersection(p[["SPY", "QQQ", "HYG", "IEF", "UUP", "VIX", "VIX3M"]]
                         .dropna().index)
           .intersection(panel.dropna().index))
    nl, p = nl.reindex(idx), p.reindex(idx)
    qqq = p["QQQ"]
    r_cc = qqq.pct_change()
    spy_ret = p["SPY"].pct_change()
    r_uup = p["UUP"].pct_change()
    r_qld = lev["QLD"].reindex(idx).pct_change()

    # ── vote ─────────────────────────────────────────────────────────────────
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

    # ── flip risk ────────────────────────────────────────────────────────────
    logger.info("BOCPD + walk-forward HMM for the flip-risk gauge …")
    _, erl = precompute_bocpd(r_cc.fillna(0.0).values)
    stances = pd.DataFrame({
        "trend":   soft_sign(qqq.pct_change(126)).shift(1),
        "meanrev": soft_sign(-qqq.pct_change(21)).shift(1),
        "bocpd":   causal_rank(pd.Series(erl, index=idx)),
        "hmm":     hmm_stance(r_cc).shift(1),
    })
    D = stances.std(axis=1).where(stances.notna().all(axis=1))
    f = 1.0 - causal_rank(D)
    vote_ma = vote.rolling(63).mean()
    vote_slow = f * vote_ma + (1.0 - f) * vote

    # ── ceiling mask ─────────────────────────────────────────────────────────
    dc_mask = pd.Series(False, index=idx)
    for d in pd.to_datetime(DEBT_CEILING):
        i = idx.searchsorted(d)
        if 0 < i < len(idx):
            dc_mask.iloc[i:min(i + 61, len(idx))] = True
    dc_mask = dc_mask.shift(1).fillna(False)

    # ── engine ───────────────────────────────────────────────────────────────
    def run(v, ceiling, uup):
        e = (2.0 * v).clip(0.0, LEV_CAP)
        if ceiling:
            e = e * np.where(dc_mask, 0.5, 1.0)
        e = pd.Series(e, index=idx).fillna(0.0)
        slack = np.maximum(1.0 - e, 0.0)
        w_uup = slack * (1.0 if uup else 0.0)
        gross = e * r_cc + w_uup * r_uup + (slack - w_uup) * DAILY_RF
        financing = np.maximum(e - 1.0, 0.0) * (RF + BORROW_SPREAD) / 252
        turn = e.diff().abs().fillna(0.0) + w_uup.diff().abs().fillna(0.0)
        return gross - financing - TC * turn, float(turn.mean()) * 252

    ladder = {
        "champion (2×vote, cash slack)":      (vote, False, False),
        "+ ceiling ×½":                       (vote, True, False),
        "+ UUP slack":                        (vote, True, True),
        "+ slow-down  (FULL STACK)":          (vote_slow, True, True),
        "full − ceiling":                     (vote_slow, False, True),
        "full − UUP":                         (vote_slow, True, False),
    }
    live = vote_slow.first_valid_index()
    strat = {"SPY buy & hold": (spy_ret, np.nan),
             "QQQ buy & hold": (r_cc, np.nan)}
    for name, (v, c, u) in ladder.items():
        strat[name] = run(v, c, u)

    # implementable QLD wrapper of the full stack (3 tickers, real prints)
    e_full = (2.0 * vote_slow).clip(0.0, LEV_CAP) * np.where(dc_mask, 0.5, 1.0)
    w = (pd.Series(e_full, index=idx) / 2.0).fillna(0.0)
    w_uup = 1.0 - w
    r_retail = (w * r_qld + w_uup * r_uup
                - TC * (w.diff().abs().fillna(0.0) + w_uup.diff().abs().fillna(0.0)))
    strat["FULL STACK in QLD+UUP (retail)"] = (
        r_retail, float((w.diff().abs() + w_uup.diff().abs()).mean()) * 252)

    # persist daily returns for downstream studies (factor attribution etc.)
    rets_out = pd.DataFrame({n: r for n, (r, _) in strat.items()}).loc[live:]
    rets_csv = RESULTS_DIR / "champion_stack_returns.csv"
    rets_out.to_csv(rets_csv)
    logger.info("Saved: %s", rets_csv)

    print("\n" + "=" * 100)
    print(f"  THE CHAMPION STACK  ·  QQQ, honest costs  ·  common live "
          f"{live.date()} → {idx[-1].date()}")
    print("=" * 100)
    keys = ["CAGR", "Volatility", "Sharpe", "Sortino", "Max DD", "Calmar"]
    spy_live = spy_ret.loc[live:]
    table = []
    for name, (r, tn) in strat.items():
        r = r.loc[live:]
        m = compute_metrics(r.dropna(), rf=RF)
        ret22 = (1 + r[r.index.year == 2022]).prod() - 1
        dc = (roll_cagr(r) - roll_cagr(spy_live)).dropna()
        ds = (roll_sharpe(r) - roll_sharpe(spy_live)).dropna()
        table.append([name] + [f"{m.get(k, float('nan')):.2f}" for k in keys]
                     + [f"{ret22:+.0%}", f"{float((dc > 0).mean()):.0%}",
                        f"{float((ds > 0).mean()):.0%}",
                        f"{tn:.1f}x" if np.isfinite(tn) else "—"])
    print(tabulate(table, headers=["strategy"] + keys
                   + ["2022", "CAGRwin", "SRwin", "turn/yr"], tablefmt="github"))
    print("  win rates vs SPY, rolling 3y windows  ·  ladder = additive; "
          "'full −' rows = leave-one-out")

    yr = pd.DataFrame({"full stack": strat["+ slow-down  (FULL STACK)"][0],
                       "champion": strat["champion (2×vote, cash slack)"][0],
                       "retail QLD+UUP": strat["FULL STACK in QLD+UUP (retail)"][0],
                       "QQQ": r_cc}).loc[live:].dropna()
    ann = yr.groupby(yr.index.year).apply(lambda d: (1 + d).prod() - 1)
    print("\nPer-year returns (%):")
    print(tabulate((ann * 100).round(1), headers=["year"] + list(ann.columns),
                   tablefmt="github"))

    # ── Graph ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    show = [("+ slow-down  (FULL STACK)", "#B71C1C", 2.0),
            ("FULL STACK in QLD+UUP (retail)", "#1A237E", 1.6),
            ("champion (2×vote, cash slack)", "#00695C", 1.3),
            ("QQQ buy & hold", "grey", 1.2),
            ("SPY buy & hold", "black", 1.2)]
    ax = axes[0]
    for name, c, lw in show:
        eq = (1 + strat[name][0].loc[live:].fillna(0)).cumprod()
        ax.plot(eq.index, eq.values, color=c, lw=lw,
                label=f"{name}  (${eq.iloc[-1]:,.0f})")
    ax.set_yscale("log"); ax.set_ylabel("growth of $1 (log)")
    ax.set_title("The frozen reference configuration vs its ancestor",
                 fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax = axes[1]
    for name, c, lw in show:
        eq = (1 + strat[name][0].loc[live:].fillna(0)).cumprod()
        dd = eq / eq.cummax() - 1
        ax.plot(dd.index, dd.values, color=c, lw=lw)
    ax.set_ylabel("drawdown"); ax.grid(alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    fig.tight_layout()
    out = RESULTS_DIR / "champion_stack.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    logger.info("Saved: %s", out)
    print("\n✓  Done.")


if __name__ == "__main__":
    main()
