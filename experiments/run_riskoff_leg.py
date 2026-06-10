#!/usr/bin/env python3
"""
The risk-off leg — is cash the right asset when the vote says retreat?

Every champion variant in the thread dials QQQ exposure with the 5-layer
vote and parks the slack in cash at the risk-free rate.  That risk-off leg
has never been examined: the bundled 250-ETF universe contains a whole menu
of defensive assets (duration, credit, TIPS, gold, commodities, dollar,
staples/utilities, REITs) and the question "what should the strategy HOLD
when it isn't holding QQQ?" is untested.  Risk-off as rotation, not retreat.

  Tests
  -----
  1. CONDITIONAL MENU   Ann. return / Sharpe of 12 defensive candidates and
                        cash, conditional on the vote tercile, plus their
                        correlation with QQQ on low-vote days.  Does anything
                        reliably beat cash exactly when the vote is low?
  2. HEAD-TO-HEAD       Champion (2×vote on QQQ) with the slack max(1−e,0)
                        in: cash (baseline), each static defensive, and a
                        CAUSAL walk-forward pick — every 21d choose the
                        candidate (cash included) with the best trailing-3y
                        Sharpe measured on low-vote days only.  Honest costs
                        on both legs.  Static winners are cherry-picked by
                        construction; the walk-forward column is the honest
                        test.  2022 is the acid test: duration crashed WITH
                        equities — a real selector must dodge TLT there.

Universe starts 2010 → the GFC is out of reach; 2011, 2015-16, 2018, 2020,
2022 are in sample (common live pushed to ~2014 by the selector warm-up).

    python experiments/run_riskoff_leg.py
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
from posterioralpha.research import soft_sign
from posterioralpha.validation import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)
logger = logging.getLogger(__name__)
RESULTS_DIR = _bootstrap.ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
PANEL_CSV = _bootstrap.ROOT / "datasets" / "overnight_panel.csv.gz"
STRESS_CSV = _bootstrap.ROOT / "datasets" / "stress_panel.csv.gz"
ETF_CSV = _bootstrap.ROOT / "datasets" / "etf_universe_prices.csv.gz"

RF = 0.04
DAILY_RF = RF / 252
BORROW_SPREAD = 0.005
TC = 0.0002
LEV_CAP = 2.0
ROLL = 756
H_LIQ, PUB_LAG, CREDIT_WIN = 126, 5, 21
SEL_WIN, SEL_REFIT, SEL_MIN_LOW = 756, 21, 60

DEFENSIVES = ["TLT", "IEF", "SHY", "BND", "LQD", "TIP",
              "GLD", "DBC", "UUP", "XLP", "XLU", "VNQ"]


def roll_sharpe(r):
    ex = r - DAILY_RF
    return (ex.rolling(ROLL).mean() / ex.rolling(ROLL).std()) * np.sqrt(252)


def roll_cagr(r):
    return (1 + r).rolling(ROLL).apply(np.prod, raw=True) ** (252 / ROLL) - 1


def main() -> None:
    nl = load_net_liquidity()["net_liquidity"]
    p = pd.read_csv(STRESS_CSV, parse_dates=["Date"], index_col="Date").sort_index()
    panel = pd.read_csv(PANEL_CSV, parse_dates=["Date"], index_col="Date").sort_index()
    etf = pd.read_csv(ETF_CSV, parse_dates=["Date"], index_col="Date").sort_index()
    # idx2: the full stress-panel sample (2007→) for the vote + the UUP
    # out-of-selection check; idx: ∩ ETF universe (2010→) for the menu study
    idx2 = (nl.dropna().index
            .intersection(p[["SPY", "QQQ", "HYG", "IEF", "UUP", "VIX", "VIX3M"]]
                          .dropna().index)
            .intersection(panel.dropna().index))
    idx = idx2.intersection(etf[DEFENSIVES].dropna().index)
    nl2, p2 = nl.reindex(idx2), p.reindex(idx2)
    r_cc2 = p2["QQQ"].pct_change()
    p = p.reindex(idx)
    r_cc = p["QQQ"].pct_change()
    spy_ret = p["SPY"].pct_change()
    r_def = etf[DEFENSIVES].reindex(idx).pct_change()

    # ── The 5-layer vote (house construction, on the full 2007→ sample) ─────
    liq = soft_sign(nl2.diff(H_LIQ).shift(PUB_LAG)).shift(1)
    vix_ts = soft_sign(p2["VIX3M"] / p2["VIX"] - 1.0).shift(1)
    credit = soft_sign(p2["HYG"].pct_change(CREDIT_WIN)
                       - p2["IEF"].pct_change(CREDIT_WIN)).shift(1)
    dollar = soft_sign(-p2["UUP"].pct_change(63)).shift(1)
    q_o, q_c = panel["QQQ_Open"].reindex(idx2), panel["QQQ_Close"].reindex(idx2)
    r_on, r_id = q_o / q_c.shift(1) - 1.0, q_c / q_o - 1.0
    tug = soft_sign(((1 + r_on).rolling(21).apply(np.prod, raw=True)
                     - (1 + r_id).rolling(21).apply(np.prod, raw=True))).shift(1)
    vote2 = pd.DataFrame({"liq": liq, "vix_ts": vix_ts, "credit": credit,
                          "dollar": dollar, "tug": tug}).mean(axis=1)
    vote = vote2.reindex(idx)
    vstart = vote.first_valid_index()

    # ── 1. The conditional menu ──────────────────────────────────────────────
    lo, hi = vote <= 1 / 3, vote >= 2 / 3
    mid = ~lo & ~hi & vote.notna()
    print("\n" + "=" * 96)
    print(f"  THE RISK-OFF LEG  ·  {vstart.date()} → {idx[-1].date()}  "
          f"(5-layer vote, natural thirds: low {int(lo.sum())}d / mid "
          f"{int(mid.sum())}d / high {int(hi.sum())}d)")
    print("=" * 96)
    print("\n  Defensive menu by vote state — ann. return (Sharpe), and "
          "corr with QQQ on low-vote days:")
    rows = []
    cands = {**{c: r_def[c] for c in DEFENSIVES},
             "cash (RF)": pd.Series(DAILY_RF, index=idx)}
    for name, r in cands.items():
        row = [name]
        for m in (lo, mid, hi):
            mm = m & r.notna()
            mu, sd = float(r[mm].mean()) * 252, float(r[mm].std()) * np.sqrt(252)
            sh = (mu - RF) / sd if sd > 0 else float("nan")
            row.append(f"{mu:+.1%} ({sh:+.2f})" if sd > 0 else f"{mu:+.1%}")
        mm = lo & r.notna() & r_cc.notna()
        c = float(np.corrcoef(r[mm], r_cc[mm])[0, 1]) if name != "cash (RF)" else 0.0
        row.append(f"{c:+.2f}")
        rows.append(row)
    print(tabulate(rows, headers=["asset", "vote low", "vote mid", "vote high",
                                  "corr QQQ | low"], tablefmt="github"))

    # ── 2. Walk-forward defensive selection (causal) ─────────────────────────
    # every SEL_REFIT days: trailing SEL_WIN window, low-vote days only,
    # pick the candidate with the best conditional Sharpe (cash included).
    cand_df = pd.DataFrame({c: r_def[c] for c in DEFENSIVES})
    cand_df["cash"] = DAILY_RF
    low_mask = (vote < 0.5)
    pick = pd.Series(index=idx, dtype=object)
    cur = "cash"
    v_ok = vote.notna().values
    for i in range(len(idx)):
        if i % SEL_REFIT == 0:
            lo_w = max(0, i - SEL_WIN)
            m = low_mask.values[lo_w:i] & v_ok[lo_w:i]
            if m.sum() >= SEL_MIN_LOW:
                w = cand_df.iloc[lo_w:i][m]
                ex = w - DAILY_RF
                sh = ex.mean() / ex.std()
                if sh.notna().any():
                    cur = sh.idxmax()
        pick.iloc[i] = cur
    pick = pick.shift(1)

    # ── Champion variants: slack in cash / static defensive / WF pick ────────
    expo = (2.0 * vote).clip(0.0, LEV_CAP)

    def run_with_slack(slack_ret, slack_w_change_extra=None):
        e = expo.fillna(0.0)
        slack = np.maximum(1.0 - e, 0.0)
        gross = e * r_cc + slack * slack_ret
        financing = np.maximum(e - 1.0, 0.0) * (RF + BORROW_SPREAD) / 252
        turn = e.diff().abs().fillna(0.0) + slack.diff().abs().fillna(0.0)
        if slack_w_change_extra is not None:   # full rebuild on a switch day
            turn = turn + 2.0 * slack * slack_w_change_extra
        return gross - financing - TC * turn

    strat = {"SPY buy & hold": spy_ret, "QQQ buy & hold": r_cc,
             "slack → cash (champion)": run_with_slack(pd.Series(DAILY_RF, index=idx))}
    for c in DEFENSIVES:
        strat[f"slack → {c} (static)"] = run_with_slack(r_def[c])
    wf_ret = pd.Series([cand_df[pick.iloc[i]].iloc[i] if isinstance(pick.iloc[i], str)
                        else np.nan for i in range(len(idx))], index=idx)
    switch = (pick != pick.shift(1)) & pick.notna() & pick.shift(1).notna()
    strat["slack → walk-forward pick"] = run_with_slack(
        wf_ret, switch.astype(float))

    live = max(vstart, pick.dropna().index[0] if pick.notna().any() else vstart)
    # selector live = first day it could have made an informed choice
    first_informed = next((idx[i] for i in range(len(idx))
                           if i > 0 and isinstance(pick.iloc[i], str)
                           and (low_mask.values[max(0, i - SEL_WIN):i]
                                & v_ok[max(0, i - SEL_WIN):i]).sum() >= SEL_MIN_LOW),
                          vstart)
    live = max(live, first_informed)
    strat = {k: v.loc[live:] for k, v in strat.items()}
    spy_live = spy_ret.loc[live:]

    print(f"\n  Champion head-to-head  ·  QQQ + slack asset, honest costs  ·  "
          f"common live {live.date()}:")
    keys = ["CAGR", "Volatility", "Sharpe", "Sortino", "Max DD", "Calmar"]
    table = []
    for name, r in strat.items():
        m = compute_metrics(r.dropna(), rf=RF)
        ret22 = (1 + r[r.index.year == 2022]).prod() - 1
        dc = (roll_cagr(r) - roll_cagr(spy_live)).dropna()
        ds = (roll_sharpe(r) - roll_sharpe(spy_live)).dropna()
        table.append([name] + [f"{m.get(k, float('nan')):.2f}" for k in keys]
                     + [f"{ret22:+.0%}", f"{float((dc > 0).mean()):.0%}",
                        f"{float((ds > 0).mean()):.0%}"])
    print(tabulate(table, headers=["strategy"] + keys + ["2022", "CAGRwin", "SRwin"],
                   tablefmt="github"))
    print("  win rates vs SPY, rolling 3y windows  ·  static rows are "
          "cherry-picked by construction — judge the walk-forward row")

    sw = pick.loc[live:]
    shares = sw.value_counts(normalize=True).sort_values(ascending=False)
    n_switch = int(switch.loc[live:].sum())
    print(f"\n  Walk-forward selector: {n_switch} switches  ·  time in pick: "
          + "  ".join(f"{k} {v:.0%}" for k, v in shares.items()))
    p22 = pick.loc["2022"].value_counts(normalize=True)
    print("  2022 allocation of the slack: "
          + "  ".join(f"{k} {v:.0%}" for k, v in p22.items()))

    # ── 3. UUP promotion check — out-of-selection 2008-2012 window ──────────
    # The dollar is the menu's standout AND the only candidate the selection
    # period never saw in a true crisis: stress-panel UUP runs from 2007, so
    # 2008-2012 (GFC + euro crisis) is data the static-row choice never used.
    expo2 = (2.0 * vote2).clip(0.0, LEV_CAP).fillna(0.0)
    slack2 = np.maximum(1.0 - expo2, 0.0)
    r_uup2 = p2["UUP"].pct_change()
    fin2 = np.maximum(expo2 - 1.0, 0.0) * (RF + BORROW_SPREAD) / 252

    def run2(slack_ret):
        turn = expo2.diff().abs().fillna(0.0) + slack2.diff().abs().fillna(0.0)
        return expo2 * r_cc2 + slack2 * slack_ret - fin2 - TC * turn

    live2 = vote2.first_valid_index()
    chk = {"slack → cash": run2(pd.Series(DAILY_RF, index=idx2)).loc[live2:],
           "slack → UUP": run2(r_uup2).loc[live2:]}
    print(f"\n  UUP promotion check  ·  GFC-inclusive sample, live {live2.date()}  "
          "(2008-2012 = out-of-selection):")
    rows = []
    for name, r in chk.items():
        m = compute_metrics(r.dropna(), rf=RF)
        oos = (1 + r.loc["2008":"2012"]).prod() - 1
        gfc = (1 + r.loc["2008-09-01":"2009-06-30"]).prod() - 1
        rows.append([name, f"{m['Sharpe']:.2f}", f"{m['Max DD']:.2f}",
                     f"{m['Calmar']:.2f}", f"{oos:+.0%}", f"{gfc:+.0%}"])
    print(tabulate(rows, headers=["slack asset", "Sharpe (full)", "Max DD",
                                  "Calmar", "2008-2012", "GFC"],
                   tablefmt="github"))

    # ── Graph ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 13))
    ax = axes[0]
    show = ["SPY buy & hold", "QQQ buy & hold", "slack → cash (champion)",
            "slack → TLT (static)", "slack → GLD (static)",
            "slack → walk-forward pick"]
    for name in show:
        r = strat[name]
        eq = (1 + r.fillna(0)).cumprod()
        ax.plot(eq.index, eq.values, lw=1.8 if "walk" in name else 1.1,
                color="black" if name.startswith("SPY") else
                ("grey" if name.startswith("QQQ") else None), label=name)
    ax.set_yscale("log"); ax.set_ylabel("growth of $1 (log)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("Risk-off as rotation: what the champion holds when it isn't holding QQQ")

    ax = axes[1]
    names = sorted(set(sw.dropna()))
    code = {n: i for i, n in enumerate(names)}
    ax.scatter(sw.index, [code[v] for v in sw], s=2, c="tab:blue")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title("Walk-forward defensive pick over time (selected on low-vote-day Sharpe)")

    ax = axes[2]
    for name in ["slack → cash (champion)", "slack → TLT (static)",
                 "slack → walk-forward pick"]:
        eq = (1 + strat[name].fillna(0)).cumprod()
        dd = eq / eq.cummax() - 1
        ax.plot(dd.index, dd.values, lw=1.1, label=name)
    ax.set_ylabel("drawdown"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("Drawdowns — 2022 is the acid test (duration crashed with equities)")
    fig.tight_layout()
    out = RESULTS_DIR / "riskoff_leg.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    logger.info("Saved: %s", out)
    print("\n✓  Done.")


if __name__ == "__main__":
    main()
