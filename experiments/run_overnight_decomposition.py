#!/usr/bin/env python3
"""
Does the liquidity edge live OVERNIGHT?  —  split the trading day in half.

Every backtest in this project used close-to-close returns and never asked *when
inside the day* the return happens.  Two facts motivate splitting it:

  1. The overnight anomaly: US equities historically earn ~all of their drift
     overnight (close→open) and ~zero intraday (open→close), and the overnight
     stream has far smaller drawdowns than the 24h stream.
  2. Net liquidity is a MACRO/FLOW signal — Fed plumbing, Treasury settlement,
     global risk repricing hit in futures *overnight*, before the cash open.

Hypothesis: our liquidity vote predicts the OVERNIGHT component far better than
the intraday one.  If true, the "worth paying for" version isn't a better signal
— it's the same signal in a better holding window: hold overnight when the vote
says go, sit in cash during the cash session.  That decouples leverage from
holding period, the exact wall the close-to-close strategies kept hitting.

  overnight_t = Open_t / Close_{t-1} − 1      (gap; where macro flow lands)
  intraday_t  = Close_t / Open_t − 1          (cash session)
  (1+overnight)(1+intraday) = 1 + close-to-close          ← sanity-checked

Honest costs: the overnight-only leg trades a round-trip EVERY day (in at close,
out at open), so it pays a daily futures round-trip (1bp) on deployed notional
on top of the 2bp rebalencing cost — reported gross AND net.

    python experiments/run_overnight_decomposition.py
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
from posterioralpha.research import liquidity_vote, soft_sign
from posterioralpha.validation import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)
logger = logging.getLogger(__name__)
RESULTS_DIR = _bootstrap.ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
PANEL_CSV = _bootstrap.ROOT / "datasets" / "stress_panel.csv.gz"
OHLC_CSV = _bootstrap.ROOT / "datasets" / "overnight_panel.csv.gz"

RF = 0.04
DAILY_RF = RF / 252
BORROW_SPREAD = 0.005
TC = 0.0002             # 2bp rebalancing
TC_NIGHT = 0.0001       # 1bp/day futures round-trip for the daily overnight in/out
LEV_CAP = 2.0
ROLL = 756
VIX_PCT_WIN = 756
UNDERS = ["SPY", "QQQ"]


def load_ohlc() -> pd.DataFrame:
    if OHLC_CSV.exists():
        return pd.read_csv(OHLC_CSV, parse_dates=["Date"], index_col="Date").sort_index()
    import yfinance as yf
    raw = yf.download(UNDERS, start="2007-01-01", auto_adjust=True, progress=False)
    out = pd.DataFrame(index=raw.index)
    for u in UNDERS:
        out[f"{u}_Open"] = raw["Open"][u]
        out[f"{u}_Close"] = raw["Close"][u]
    out = out.dropna(how="all")
    out.to_csv(OHLC_CSV, index_label="Date")
    logger.info("cached OHLC panel → %s", OHLC_CSV.name)
    return out


def leg(component_ret, expo, overnight=False):
    """Net daily return of a position of size `expo` earning `component_ret`.

    Cash on uninvested capital and financing on the levered part are modelled
    identically for every leg, so the comparison is apples-to-apples; the
    overnight leg additionally pays a daily futures round-trip (it flattens and
    re-enters every day)."""
    expo = expo.clip(0.0, LEV_CAP).fillna(0.0)
    gross = expo * component_ret
    fin = np.maximum(expo - 1.0, 0.0) * (RF + BORROW_SPREAD) / 252
    cash = np.maximum(1.0 - expo, 0.0) * DAILY_RF
    costs = TC * expo.diff().abs().fillna(0.0)
    if overnight:
        costs = costs + TC_NIGHT * expo            # daily in/out on deployed notional
    return pd.Series(gross - fin + cash - costs, index=component_ret.index)


def roll_cagr(r):
    return (1 + r).rolling(ROLL).apply(np.prod, raw=True) ** (252 / ROLL) - 1


def main() -> None:
    nl = load_net_liquidity()["net_liquidity"]
    p = pd.read_csv(PANEL_CSV, parse_dates=["Date"], index_col="Date").sort_index()
    o = load_ohlc()
    idx = (nl.dropna().index
           .intersection(p.dropna(subset=["HYG", "IEF", "UUP", "VIX", "VIX3M"]).index)
           .intersection(o.dropna(subset=[f"{u}_Open" for u in UNDERS]
                                  + [f"{u}_Close" for u in UNDERS]).index))
    nl, p, o = nl.reindex(idx), p.reindex(idx), o.reindex(idx)

    # ── Decompose each underlying's day ───────────────────────────────────────
    comp = {}
    for u in UNDERS:
        cc = o[f"{u}_Close"].pct_change()
        overnight = o[f"{u}_Open"] / o[f"{u}_Close"].shift(1) - 1.0
        intraday = o[f"{u}_Close"] / o[f"{u}_Open"] - 1.0
        comp[u] = dict(cc=cc, on=overnight, id=intraday)
    # sanity: (1+on)(1+id) == 1+cc
    chk = ((1 + comp["SPY"]["on"]) * (1 + comp["SPY"]["id"]) - 1 - comp["SPY"]["cc"]).abs().max()
    logger.info("decomposition identity max error: %.2e", chk)

    # ── The anomaly in OUR window: where does buy&hold's return come from? ────
    print("\n" + "=" * 88)
    print(f"  WHERE DOES BUY & HOLD'S RETURN COME FROM?  ·  {idx.min().date()} → {idx.max().date()}")
    print("=" * 88)
    rows = []
    for u in UNDERS:
        for lab, r in [("close→close (24h)", comp[u]["cc"]),
                       ("overnight (C→O)", comp[u]["on"]),
                       ("intraday (O→C)", comp[u]["id"])]:
            m = compute_metrics(r.dropna(), rf=0.0)
            dd = ((1 + r.fillna(0)).cumprod() / (1 + r.fillna(0)).cumprod().cummax() - 1).min()
            rows.append([f"{u} {lab}", f"{m['CAGR']:+.1%}", f"{m['Volatility']:.1%}",
                         f"{m['Sharpe']:.2f}", f"{dd:.0%}"])
    print(tabulate(rows, headers=["stream", "CAGR", "Vol", "Sharpe(rf0)", "MaxDD"],
                   tablefmt="github"))

    # ── Causal vote + fragility weighting (our best mechanism) ────────────────
    base = liquidity_vote(nl, p["VIX"], p["VIX3M"], p["HYG"], p["IEF"])
    dollar = soft_sign(-p["UUP"].pct_change(63)).shift(1)
    liq = base["liq"]
    market = pd.concat([base["vix_ts"], base["credit"], dollar.rename("d")],
                       axis=1).mean(axis=1)
    vix_pct = p["VIX"].rolling(VIX_PCT_WIN, min_periods=252).apply(
        lambda x: (x < x[-1]).mean(), raw=True)
    frag = (0.5 * vix_pct + 0.5 * (1.0 - base["vix_ts"])).shift(1).clip(0, 1)
    vote = ((1 - frag) * market + frag * liq)        # fragility-weighted vote
    live = pd.concat([vote, frag], axis=1).dropna().index.min()
    ev = idx[idx >= live]

    # ── Does the vote's edge concentrate overnight? (conditional means) ───────
    print("\n" + "=" * 88)
    print("  DOES THE VOTE'S EDGE CONCENTRATE OVERNIGHT?  (mean next-day component by vote)")
    print("=" * 88)
    hi = (vote > 0.6).reindex(idx).fillna(False)     # risk-on days
    lo = (vote < 0.4).reindex(idx).fillna(False)     # risk-off days
    rows = []
    for u in UNDERS:
        for lab, r in [("overnight", comp[u]["on"]), ("intraday", comp[u]["id"])]:
            mu_hi = r[hi].mean() * 252
            mu_lo = r[lo].mean() * 252
            rows.append([f"{u} {lab}", f"{mu_hi:+.1%}", f"{mu_lo:+.1%}",
                         f"{(mu_hi - mu_lo):+.1%}"])
    print(tabulate(rows, headers=["component", "ann. ret | vote>0.6",
                                  "ann. ret | vote<0.4", "spread (on-edge)"],
                   tablefmt="github"))
    print("  (if the liquidity edge lives overnight, the overnight spread >> the intraday spread)")

    # ── Strategies: same vote exposure, three holding windows ─────────────────
    expo = (2.0 * vote)                              # same 2x mapping as champion
    rets = {"SPY buy & hold": comp["SPY"]["cc"].reindex(ev),
            "QQQ buy & hold": comp["QQQ"]["cc"].reindex(ev)}
    for u in UNDERS:
        rets[f"2×vote {u} · 24h (close-close)"] = leg(comp[u]["cc"], expo).reindex(ev)
        rets[f"2×vote {u} · OVERNIGHT only"] = leg(comp[u]["on"], expo, overnight=True).reindex(ev)
        rets[f"2×vote {u} · intraday only"] = leg(comp[u]["id"], expo, overnight=True).reindex(ev)

    print("\n" + "=" * 100)
    print(f"  SAME VOTE, THREE HOLDING WINDOWS  ·  {live.date()} → {ev.max().date()}  (net of costs)")
    print("=" * 100)
    keys = ["CAGR", "Volatility", "Sharpe", "Sortino", "Max DD", "Calmar"]
    table = []
    for n, r in rets.items():
        m = compute_metrics(r.dropna(), rf=RF)
        gfc = (1 + r[(r.index >= "2008-09-01") & (r.index <= "2009-06-30")]).prod() - 1
        y22 = (1 + r[r.index.year == 2022]).prod() - 1
        table.append([n] + [f"{m.get(k, float('nan')):.2f}" for k in keys]
                     + [f"{gfc:+.0%}", f"{y22:+.0%}"])
    print(tabulate(table, headers=["strategy"] + keys + ["GFC", "2022"], tablefmt="github"))

    # ── A fair-vol comparison: lever overnight-only up to the 24h leg's vol ───
    print("\n" + "=" * 100)
    print("  VOL-MATCHED:  lever OVERNIGHT-only to the 24h leg's volatility (decouple lever↔holding)")
    print("=" * 100)
    rows = []
    for u in UNDERS:
        full = rets[f"2×vote {u} · 24h (close-close)"].dropna()
        night = leg(comp[u]["on"], expo, overnight=True).reindex(ev).dropna()
        k = full.std() / night.std()                 # scale overnight to 24h vol
        night_lev = (night * k)                      # (financing on the extra notional ignored ⇒ optimistic; flagged)
        for nm, r in [(f"{u} 24h 2×vote", full), (f"{u} overnight ×{k:.1f} (vol-matched)", night_lev)]:
            m = compute_metrics(r, rf=RF)
            dd = ((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min()
            y22 = (1 + r[r.index.year == 2022]).prod() - 1
            rows.append([nm, f"{m['CAGR']:.2f}", f"{m['Volatility']:.2f}",
                         f"{m['Sharpe']:.2f}", f"{dd:.0%}", f"{y22:+.0%}"])
    print(tabulate(rows, headers=["strategy", "CAGR", "Vol", "Sharpe", "MaxDD", "2022"],
                   tablefmt="github"))
    print("  ⚠ vol-matched overnight ignores the financing cost of the extra leverage → optimistic upper bound.")

    # ── Graphs ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 13))
    ax = axes[0]
    for u in UNDERS:
        ax.plot((1 + comp[u]["on"].reindex(ev).fillna(0)).cumprod(), lw=1.3,
                label=f"{u} overnight (C→O)")
        ax.plot((1 + comp[u]["id"].reindex(ev).fillna(0)).cumprod(), lw=1.0, ls="--",
                label=f"{u} intraday (O→C)")
    ax.set_yscale("log"); ax.set_ylabel("growth of $1 (log)")
    ax.set_title("Buy & hold, decomposed: overnight carries the drift, intraday is flat/negative")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    for n in [f"2×vote QQQ · 24h (close-close)", f"2×vote QQQ · OVERNIGHT only",
              "SPY buy & hold"]:
        ax.plot((1 + rets[n].fillna(0)).cumprod(), lw=1.4 if "buy" in n else 1.2,
                color="black" if "SPY" in n else None, label=n)
    ax.set_yscale("log"); ax.set_ylabel("growth of $1 (log)")
    ax.set_title("Same vote, QQQ: 24h vs overnight-only vs SPY"); ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    for n in [f"2×vote QQQ · 24h (close-close)", f"2×vote QQQ · OVERNIGHT only",
              "SPY buy & hold"]:
        eqc = (1 + rets[n].fillna(0)).cumprod(); dd = eqc / eqc.cummax() - 1
        ax.plot(dd.index, dd.values * 100, lw=1.1,
                color="black" if "SPY" in n else None, label=n)
    ax.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31"),
               color="red", alpha=0.08, label="2022 grind")
    ax.set_ylabel("drawdown (%)"); ax.legend(fontsize=8)
    ax.set_title("Drawdowns — does holding overnight-only tame the tail?"); ax.grid(alpha=0.3)
    fig.tight_layout()
    out = RESULTS_DIR / "overnight_decomposition.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    logger.info("Saved: %s", out)
    print("\n✓  Done.")


if __name__ == "__main__":
    main()
