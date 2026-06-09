#!/usr/bin/env python3
"""
Exploration lab — four cheap, novel hypotheses fired at the champion at once.

Full exploratory mode: each idea is allowed to fail; failures are findings and
go in the table like everything else.  Common harness = the 4-layer 2×vote on
QQQ with honest costs (the champion), common live window, rolling 3y OOS.

  A. TUG-OF-WAR   The cached OHLC panel as a *signal*, not a holding window:
                  trailing 21d (overnight − intraday) cumulative return spread,
                  soft-signed.  Lou-Polk-Skouras-style clientele tension —
                  when the overnight crowd and the cash-session crowd pull
                  apart, who is right?  Tested standalone and as a 5th layer.

  B. HYSTERESIS   Every exposure mapping so far is memoryless (exposure is a
                  function of today's vote only).  A Schmitt trigger adds
                  state: flip risk-on only when vote > 0.5+band, flip off only
                  when vote < 0.5−band, otherwise HOLD the previous stance.
                  Swept over band ∈ {0.05, 0.10, 0.15} to expose fragility
                  rather than fit it.

  C. VELOCITY     Trade the vote's direction of travel, not its level:
                  soft-signed Δ21d vote.  An improving 0.4 may beat a
                  deteriorating 0.6.  Standalone and as a 5th layer.

  D. LIQ DRAWDOWN Replace the Δ6m liquidity layer with an asymmetric gauge:
                  net liquidity % below its trailing 1y peak (publication-
                  lagged, soft-signed).  Only sees contractions — by
                  construction silent in expansions.

    python experiments/run_exploration_lab.py
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
PANEL_CSV = _bootstrap.ROOT / "datasets" / "overnight_panel.csv.gz"

RF = 0.04
DAILY_RF = RF / 252
BORROW_SPREAD = 0.005
TC = 0.0002
LEV_CAP = 2.0
ROLL = 756
PUB_LAG = 5


def run(expo, r_cc):
    expo = expo.clip(0.0, LEV_CAP).fillna(0.0)
    gross = expo * r_cc
    financing = np.maximum(expo - 1.0, 0.0) * (RF + BORROW_SPREAD) / 252
    cash = np.maximum(1.0 - expo, 0.0) * DAILY_RF
    costs = TC * expo.diff().abs().fillna(0.0)
    return pd.Series(gross - financing + cash - costs, index=r_cc.index)


def roll_sharpe(r):
    ex = r - DAILY_RF
    return (ex.rolling(ROLL).mean() / ex.rolling(ROLL).std()) * np.sqrt(252)


def roll_cagr(r):
    return (1 + r).rolling(ROLL).apply(np.prod, raw=True) ** (252 / ROLL) - 1


def schmitt(vote: pd.Series, band: float) -> pd.Series:
    """Stateful trigger: on above 0.5+band, off below 0.5−band, else hold."""
    v = vote.values
    out = np.full(len(v), np.nan)
    state = np.nan
    for i in range(len(v)):
        if np.isnan(v[i]):
            out[i] = state
            continue
        if v[i] > 0.5 + band:
            state = 1.0
        elif v[i] < 0.5 - band:
            state = 0.0
        elif np.isnan(state):
            state = 0.5            # no history yet: start neutral
        out[i] = state
    return pd.Series(out, index=vote.index)


def main() -> None:
    nl = load_net_liquidity()["net_liquidity"]
    px = load_etf_universe_prices()
    vix = pd.read_csv(VIX_CSV, parse_dates=["Date"], index_col="Date").sort_index()
    panel = pd.read_csv(PANEL_CSV, parse_dates=["Date"], index_col="Date").sort_index()
    idx = (nl.dropna().index
           .intersection(px[["SPY", "QQQ", "HYG", "IEF", "UUP"]].dropna().index)
           .intersection(vix.dropna().index)
           .intersection(panel.dropna().index))
    nl, vix = nl.reindex(idx), vix.reindex(idx)
    r_cc = px["QQQ"].reindex(idx).pct_change()
    spy_ret = px["SPY"].reindex(idx).pct_change()

    base = liquidity_vote(nl, vix["VIX"], vix["VIX3M"],
                          px["HYG"].reindex(idx), px["IEF"].reindex(idx))
    dollar = soft_sign(-px["UUP"].reindex(idx).pct_change(63)).shift(1)
    s4 = pd.concat([base[["liq", "vix_ts", "credit"]], dollar.rename("dollar")], axis=1)
    vote = s4.mean(axis=1)

    # ── A. tug-of-war ────────────────────────────────────────────────────────
    q_o = panel["QQQ_Open"].reindex(idx)
    q_c = panel["QQQ_Close"].reindex(idx)
    r_on = q_o / q_c.shift(1) - 1.0
    r_id = q_c / q_o - 1.0
    tug_raw = ((1 + r_on).rolling(21).apply(np.prod, raw=True)
               - (1 + r_id).rolling(21).apply(np.prod, raw=True))
    tug = soft_sign(tug_raw).shift(1)

    # ── C. velocity ──────────────────────────────────────────────────────────
    velo = soft_sign(vote.diff(21)).shift(1)

    # ── D. liquidity drawdown layer ──────────────────────────────────────────
    liq_dd_raw = (nl / nl.rolling(252).max() - 1.0).shift(PUB_LAG)   # ≤ 0
    liq_dd = soft_sign(liq_dd_raw).shift(1)
    s4_dd = pd.concat([liq_dd.rename("liq_dd"), s4[["vix_ts", "credit", "dollar"]]],
                      axis=1)

    votes5 = {
        "champion (4-layer)":   vote,
        "A. +tug-of-war layer": pd.concat([s4, tug.rename("tug")], axis=1).mean(axis=1),
        "C. +velocity layer":   pd.concat([s4, velo.rename("v")], axis=1).mean(axis=1),
        "D. liq-drawdown swap": s4_dd.mean(axis=1),
    }

    rets, descr = {}, []
    live = max(v.first_valid_index() for v in
               [vote, tug, velo, liq_dd, schmitt(vote, 0.1)])

    rets["QQQ buy & hold"] = r_cc
    rets["SPY buy & hold"] = spy_ret
    for name, v in votes5.items():
        rets[f"2×{name}"] = run(2.0 * v, r_cc)
    # standalone dials (same 2× mapping so they're comparable)
    rets["2×A. tug-of-war alone"] = run(2.0 * tug, r_cc)
    rets["2×C. velocity alone"] = run(2.0 * velo, r_cc)
    # B: hysteresis mappings of the champion vote (0/1/2× exposure states)
    for band in (0.05, 0.10, 0.15):
        rets[f"2×B. Schmitt band ±{band:.2f}"] = run(2.0 * schmitt(vote, band), r_cc)

    rets = {k: v.loc[live:] for k, v in rets.items()}
    spy_live = spy_ret.loc[live:]

    print("\n" + "=" * 96)
    print(f"  EXPLORATION LAB  ·  QQQ, honest costs  ·  {live.date()} → {idx[-1].date()}")
    print("=" * 96)
    keys = ["CAGR", "Volatility", "Sharpe", "Sortino", "Max DD", "Calmar"]
    table = []
    for name, r in rets.items():
        m = compute_metrics(r.dropna(), rf=RF)
        ret22 = (1 + r[r.index.year == 2022]).prod() - 1
        dc = (roll_cagr(r) - roll_cagr(spy_live)).dropna()
        ds = (roll_sharpe(r) - roll_sharpe(spy_live)).dropna()
        table.append([name] + [f"{m.get(k, float('nan')):.2f}" for k in keys]
                     + [f"{ret22:+.0%}",
                        f"{float((dc > 0).mean()):.0%}",
                        f"{float((ds > 0).mean()):.0%}"])
    print(tabulate(table, headers=["strategy"] + keys
                   + ["2022", "CAGRwin", "SRwin"], tablefmt="github"))
    print("  CAGRwin/SRwin = share of rolling 3y windows beating SPY")

    # quick diagnostics for the two signal candidates
    for lab, s in [("tug-of-war", tug), ("velocity", velo)]:
        rk = s.rank(pct=True)
        hi = float(r_cc[rk >= 0.67].mean()) * 252
        lo = float(r_cc[rk <= 0.33].mean()) * 252
        print(f"  {lab:12s} next-day QQQ by signal tercile: "
              f"low {lo:+.1%}  high {hi:+.1%}  spread {hi - lo:+.1%}")

    # ── Round 2: interrogate the winner (tug-of-war) ─────────────────────────
    print("\n" + "=" * 96)
    print("  ROUND 2  ·  is the tug-of-war layer real?")
    print("=" * 96)
    corr5 = pd.concat([s4, tug.rename("tug")], axis=1).corr()["tug"].drop("tug")
    print("  corr(tug, existing layers): "
          + "  ".join(f"{k} {v:+.2f}" for k, v in corr5.items()))

    rows = []
    for win in (10, 21, 42):
        t_raw = ((1 + r_on).rolling(win).apply(np.prod, raw=True)
                 - (1 + r_id).rolling(win).apply(np.prod, raw=True))
        t_w = soft_sign(t_raw).shift(1)
        v5 = pd.concat([s4, t_w.rename("tug")], axis=1).mean(axis=1)
        r = run(2.0 * v5, r_cc).loc[live:]
        m = compute_metrics(r.dropna(), rf=RF)
        dc = (roll_cagr(r) - roll_cagr(spy_live)).dropna()
        rows.append([f"tug window {win}d (QQQ)", f"{m['CAGR']:.2f}",
                     f"{m['Sharpe']:.2f}", f"{m['Max DD']:.2f}",
                     f"{float((dc > 0).mean()):.0%}"])
    # same construction on SPY's own overnight/intraday legs, applied to SPY
    s_on = panel["SPY_Open"].reindex(idx) / panel["SPY_Close"].reindex(idx).shift(1) - 1
    s_id = panel["SPY_Close"].reindex(idx) / panel["SPY_Open"].reindex(idx) - 1
    t_spy = soft_sign(((1 + s_on).rolling(21).apply(np.prod, raw=True)
                       - (1 + s_id).rolling(21).apply(np.prod, raw=True))).shift(1)
    v5_spy = pd.concat([s4, t_spy.rename("tug")], axis=1).mean(axis=1)
    r = run(2.0 * v5_spy, spy_ret).loc[live:]
    m = compute_metrics(r.dropna(), rf=RF)
    dc = (roll_cagr(r) - roll_cagr(spy_live)).dropna()
    rows.append(["tug 21d, SPY signal+underlying", f"{m['CAGR']:.2f}",
                 f"{m['Sharpe']:.2f}", f"{m['Max DD']:.2f}",
                 f"{float((dc > 0).mean()):.0%}"])
    print(tabulate(rows, headers=["variant", "CAGR", "Sharpe", "MaxDD",
                                  "CAGRwin"], tablefmt="github"))

    # split-sample check of champion vs 5-layer
    v5_21 = pd.concat([s4, tug.rename("tug")], axis=1).mean(axis=1)
    r5 = run(2.0 * v5_21, r_cc).loc[live:]
    rc = rets["2×champion (4-layer)"]
    print("\n  Split sample (Sharpe):")
    for lab, lo_, hi_ in [("2014-2019", "2014", "2019"), ("2020-2026", "2020", "2026")]:
        s_c = compute_metrics(rc.loc[lo_:hi_].dropna(), rf=RF)["Sharpe"]
        s_5 = compute_metrics(r5.loc[lo_:hi_].dropna(), rf=RF)["Sharpe"]
        print(f"    {lab}:  champion {s_c:.2f}  →  +tug 5-layer {s_5:.2f}")

    # stack the two validated upgrades: tug layer + debt-ceiling calendar halving
    DEBT_CEILING = ["2011-08-02", "2013-10-17", "2014-02-15", "2015-11-02",
                    "2017-09-08", "2018-02-09", "2019-08-02", "2021-10-14",
                    "2021-12-16", "2023-06-03"]   # complete through Jun-2023
    dc_mask = pd.Series(False, index=idx)
    for d in pd.to_datetime(DEBT_CEILING):
        i = idx.searchsorted(d)
        if 0 < i < len(idx):
            dc_mask.iloc[i:min(i + 61, len(idx))] = True
    dc_mask = dc_mask.shift(1).fillna(False)
    r_stack = run(2.0 * v5_21 * np.where(dc_mask, 0.5, 1.0), r_cc).loc[live:]

    print("\n  The stack — tug layer + ×½ in debt-ceiling windows:")
    rows = []
    for name, r in [("2×champion (4-layer)", rc),
                    ("2×5-layer (+tug)", r5),
                    ("2×5-layer + debt-ceiling ×½", r_stack)]:
        m = compute_metrics(r.dropna(), rf=RF)
        ret22 = (1 + r[r.index.year == 2022]).prod() - 1
        dc_ = (roll_cagr(r) - roll_cagr(spy_live)).dropna()
        ds_ = (roll_sharpe(r) - roll_sharpe(spy_live)).dropna()
        rows.append([name, f"{m['CAGR']:.2f}", f"{m['Sharpe']:.2f}",
                     f"{m['Max DD']:.2f}", f"{m['Calmar']:.2f}", f"{ret22:+.0%}",
                     f"{float((dc_ > 0).mean()):.0%}", f"{float((ds_ > 0).mean()):.0%}"])
    print(tabulate(rows, headers=["strategy", "CAGR", "Sharpe", "MaxDD", "Calmar",
                                  "2022", "CAGRwin", "SRwin"], tablefmt="github"))
    rets["2×5-layer + debt-ceiling ×½"] = r_stack

    # ── Round 3: fresh-sample validation of the tug signal ───────────────────
    # The lab's live window starts 2014, but SPY OHLC reaches back to 1993 and
    # QQQ to 1999 — data the layer-selection never saw.  If the tug spread is
    # real it should exist there too; if it only appears post-2014/2020 it is
    # a recent-clientele artifact and must not be promoted.
    print("\n" + "=" * 96)
    print("  ROUND 3  ·  fresh-sample check: next-day return by causal tug tercile")
    print("=" * 96)
    rows = []
    for t in ["SPY", "QQQ"]:
        o = panel[f"{t}_Open"].dropna()
        c = panel[f"{t}_Close"].dropna()
        on = o / c.shift(1) - 1.0
        iday = c / o - 1.0
        ret1 = c.pct_change()
        t_raw = ((1 + on).rolling(21).apply(np.prod, raw=True)
                 - (1 + iday).rolling(21).apply(np.prod, raw=True))
        sig = soft_sign(t_raw).shift(1)
        rk = sig.rolling(756, min_periods=252).apply(
            lambda x: (x < x[-1]).mean(), raw=True).shift(1)
        for lab, a, b in [("1994-2003", "1994", "2003"), ("2004-2013", "2004", "2013"),
                          ("2014-2026", "2014", "2026")]:
            r_, k_ = ret1.loc[a:b], rk.loc[a:b]
            if k_.notna().sum() < 252:
                continue
            hi_ = float(r_[k_ >= 0.67].mean()) * 252
            lo_ = float(r_[k_ <= 0.33].mean()) * 252
            rows.append([t, lab, f"{lo_:+.1%}", f"{hi_:+.1%}", f"{hi_ - lo_:+.1%}",
                         int(k_.notna().sum())])
    print(tabulate(rows, headers=["ticker", "period", "low tercile",
                                  "high tercile", "spread", "days"], tablefmt="github"))

    # ── Graph ─────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))
    show = ["QQQ buy & hold", "2×champion (4-layer)", "2×A. +tug-of-war layer",
            "2×C. +velocity layer", "2×D. liq-drawdown swap",
            "2×B. Schmitt band ±0.10"]
    for name in show:
        eq = (1 + rets[name].fillna(0)).cumprod()
        ax1.plot(eq.index, eq.values, lw=1.6 if "champion" in name else 1.1,
                 color="grey" if "buy" in name else None, label=name)
    ax1.set_yscale("log"); ax1.set_ylabel("growth of $1 (log)")
    ax1.set_title("Exploration lab — four novel dials vs the champion")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    for name in ["2×champion (4-layer)", "2×B. Schmitt band ±0.10",
                 "2×D. liq-drawdown swap"]:
        eq = (1 + rets[name].fillna(0)).cumprod()
        dd = eq / eq.cummax() - 1
        ax2.plot(dd.index, dd.values, lw=1.0, label=name)
    ax2.set_ylabel("drawdown"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    ax2.set_title("Drawdown of the most interesting variants")
    fig.tight_layout()
    out = RESULTS_DIR / "exploration_lab.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    logger.info("Saved: %s", out)
    print("\n✓  Done.")


if __name__ == "__main__":
    main()
