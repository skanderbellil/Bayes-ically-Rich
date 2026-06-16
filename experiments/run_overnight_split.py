#!/usr/bin/env python3
"""
Split the day in half — does the liquidity signal live in the OVERNIGHT leg?

Every backtest in this repo uses close-to-close returns; we never asked WHEN
inside the day the return happens.  The decades-old anomaly: US equities earn
essentially all of their return overnight (close→open) and ~zero intraday
(open→close), and overnight-only holding has a fraction of the 24h drawdown.

Why that matters here: net liquidity is a macro/flow signal — Fed plumbing,
Treasury settlement, global risk repricing hit overnight, in futures, before
the cash open.  Sharp hypothesis: the liquidity vote predicts the overnight
component far better than the intraday component.  If true, the play isn't
"lever QQQ 2×" — it's "hold QQQ overnight when liquidity says go, flatten
during the day."  Same drift, a fraction of the drawdown — decoupling the
leverage↔drawdown dial that every refinement so far has slammed into.

  r_on = Open_t / Close_{t-1} − 1      (overnight, includes div ex-dates)
  r_id = Close_t / Open_t − 1          (intraday, cash session)

Part 1 — the anomaly itself on SPY (1993→) and QQQ (1999→), gross.
Part 2 — signal loading: corr(causal Δ6m net liquidity, next-21d overnight
         vs intraday cumulative return) + next-day components by causal vote
         tercile.
Part 3 — strategies on QQQ with honest costs.  An overnight-only position
         trades TWICE a day (buy close, sell open): turnover = 2×expo/day,
         i.e. ~10%/yr drag at the house 2bp — charged in full, plus a 1bp
         sensitivity row (MOC/MOO on QQQ).  Overnight legs earn half-day cash
         credit on idle capital and pay half-day financing above 1×.

Data: datasets/overnight_panel.csv.gz (yfinance adjusted OHLC; rebuild with
--refresh).  Vote inputs as in the rest of the thread.

    python experiments/run_overnight_split.py
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
TC = 0.0002                 # house standard: 2bp per unit turnover
TC_LOW = 0.0001             # sensitivity: 1bp (MOO/MOC auctions on QQQ)
LEV_CAP = 2.0
ROLL = 756
H_LIQ = 126
PUB_LAG = 5


def refresh_panel() -> None:
    import yfinance as yf
    out = {}
    for t in ["QQQ", "SPY"]:
        h = yf.Ticker(t).history(period="max", auto_adjust=True)
        h.index = h.index.tz_localize(None)
        out[f"{t}_Open"], out[f"{t}_Close"] = h["Open"], h["Close"]
    df = pd.DataFrame(out)
    df.index.name = "Date"
    df.to_csv(PANEL_CSV)
    logger.info("Rebuilt %s %s", PANEL_CSV, df.shape)


def split_returns(panel: pd.DataFrame, t: str):
    """(overnight, intraday, close-to-close) simple returns for ticker t."""
    o, c = panel[f"{t}_Open"], panel[f"{t}_Close"]
    r_on = o / c.shift(1) - 1.0
    r_id = c / o - 1.0
    r_cc = c.pct_change()
    return r_on, r_id, r_cc


def causal_rank(s: pd.Series, win: int = 756) -> pd.Series:
    return s.rolling(win, min_periods=252).apply(
        lambda x: (x < x[-1]).mean(), raw=True).shift(1)


def overnight_strategy(r_on: pd.Series, expo: pd.Series, tc: float):
    """Hold `expo` overnight only, flat during the cash session.

    Turnover = 2×expo per day (enter at close, exit at open).  Idle capital
    earns DAILY_RF scaled by the idle fraction of the day (intraday half is
    always idle); financing above 1× is charged for the overnight half only."""
    expo = expo.clip(0.0, LEV_CAP).fillna(0.0)
    gross = expo * r_on
    cash = (np.maximum(1.0 - expo, 0.0) * 0.5 + 0.5) * DAILY_RF
    financing = np.maximum(expo - 1.0, 0.0) * (RF + BORROW_SPREAD) / 252 * 0.5
    costs = tc * 2.0 * expo
    return pd.Series(gross + cash - financing - costs, index=r_on.index)


def daily_strategy(r_cc: pd.Series, expo: pd.Series, tc: float = TC):
    """House 24h exposure model (identical to the rest of the thread)."""
    expo = expo.clip(0.0, LEV_CAP).fillna(0.0)
    gross = expo * r_cc
    financing = np.maximum(expo - 1.0, 0.0) * (RF + BORROW_SPREAD) / 252
    cash = np.maximum(1.0 - expo, 0.0) * DAILY_RF
    costs = tc * expo.diff().abs().fillna(0.0)
    return pd.Series(gross - financing + cash - costs, index=r_cc.index)


def roll_cagr(r):
    return (1 + r).rolling(ROLL).apply(np.prod, raw=True) ** (252 / ROLL) - 1


def roll_sharpe(r):
    ex = r - DAILY_RF
    return (ex.rolling(ROLL).mean() / ex.rolling(ROLL).std()) * np.sqrt(252)


def main() -> None:
    if "--refresh" in sys.argv:
        refresh_panel()
    panel = pd.read_csv(PANEL_CSV, parse_dates=["Date"], index_col="Date").sort_index()

    # ── Part 1: the anomaly, gross, full history ─────────────────────────────
    print("\n" + "=" * 88)
    print("  THE OVERNIGHT/INTRADAY DECOMPOSITION  ·  gross, no costs")
    print("=" * 88)
    keys = ["CAGR", "Volatility", "Sharpe", "Max DD"]
    rows = []
    for t in ["SPY", "QQQ"]:
        r_on, r_id, r_cc = split_returns(panel, t)
        for leg, r in [("24h close-to-close", r_cc), ("overnight only", r_on),
                       ("intraday only", r_id)]:
            m = compute_metrics(r.dropna(), rf=RF)
            rows.append([f"{t} {leg}", f"{r.dropna().index[0].date()}"]
                        + [f"{m.get(k, float('nan')):.2f}" for k in keys])
    print(tabulate(rows, headers=["leg", "from"] + keys, tablefmt="github"))

    # ── Vote and aligned data (liquidity thread window) ──────────────────────
    nl = load_net_liquidity()["net_liquidity"]
    px = load_etf_universe_prices()
    vix = pd.read_csv(VIX_CSV, parse_dates=["Date"], index_col="Date").sort_index()
    idx = (nl.dropna().index
           .intersection(px[["SPY", "QQQ", "HYG", "IEF", "UUP"]].dropna().index)
           .intersection(vix.dropna().index)
           .intersection(panel.dropna().index))
    nl, vix = nl.reindex(idx), vix.reindex(idx)

    base = liquidity_vote(nl, vix["VIX"], vix["VIX3M"],
                          px["HYG"].reindex(idx), px["IEF"].reindex(idx))
    dollar = soft_sign(-px["UUP"].reindex(idx).pct_change(63)).shift(1)
    vote = pd.concat([base[["liq", "vix_ts", "credit"]], dollar.rename("dollar")],
                     axis=1).mean(axis=1)

    r_on, r_id, r_cc = (s.reindex(idx) for s in split_returns(panel, "QQQ"))

    # ── Part 2: where does the signal load? ──────────────────────────────────
    d_nl = nl.diff(H_LIQ).shift(PUB_LAG)
    fwd = {"overnight": (1 + r_on).rolling(21).apply(np.prod, raw=True).shift(-21) - 1,
           "intraday":  (1 + r_id).rolling(21).apply(np.prod, raw=True).shift(-21) - 1,
           "24h":       (1 + r_cc).rolling(21).apply(np.prod, raw=True).shift(-21) - 1}

    print("\n" + "=" * 88)
    print(f"  SIGNAL LOADING  ·  QQQ  ·  {idx.min().date()} → {idx.max().date()}")
    print("=" * 88)
    rows = []
    for leg, f in fwd.items():
        m = d_nl.notna() & f.notna()
        c = float(np.corrcoef(d_nl[m], f[m])[0, 1])
        rows.append([f"corr(Δ6m net-liq, next-21d {leg})", f"{c:+.3f}"])
    print(tabulate(rows, tablefmt="github", headers=["", "corr"]))

    v_rank = causal_rank(vote)
    print("\n  Next-day return components by causal vote tercile (annualized):")
    rows = []
    for lab, mask in [("vote low", v_rank <= 0.33),
                      ("vote mid", (v_rank > 0.33) & (v_rank < 0.67)),
                      ("vote high", v_rank >= 0.67)]:
        rows.append([lab,
                     f"{float(r_on[mask].mean()) * 252:+.1%}",
                     f"{float(r_id[mask].mean()) * 252:+.1%}",
                     f"{float(r_cc[mask].mean()) * 252:+.1%}", int(mask.sum())])
    rows.append(["ALL", f"{float(r_on[v_rank.notna()].mean()) * 252:+.1%}",
                 f"{float(r_id[v_rank.notna()].mean()) * 252:+.1%}",
                 f"{float(r_cc[v_rank.notna()].mean()) * 252:+.1%}",
                 int(v_rank.notna().sum())])
    print(tabulate(rows, headers=["vote tercile", "overnight", "intraday",
                                  "24h", "days"], tablefmt="github"))

    # ── Part 3: strategies (QQQ, honest costs) ───────────────────────────────
    live = v_rank.first_valid_index()
    strat = {
        "QQQ buy & hold":                  r_cc,
        "2×vote 24h (champion)":           daily_strategy(r_cc, 2.0 * vote),
        "overnight always (2bp)":          overnight_strategy(r_on, pd.Series(1.0, idx), TC),
        "overnight always (1bp)":          overnight_strategy(r_on, pd.Series(1.0, idx), TC_LOW),
        "2×vote overnight (2bp)":          overnight_strategy(r_on, 2.0 * vote, TC),
        "2×vote overnight (1bp)":          overnight_strategy(r_on, 2.0 * vote, TC_LOW),
    }
    strat = {k: v.loc[live:] for k, v in strat.items()}
    spy_live = px["SPY"].reindex(idx).pct_change().loc[live:]

    print("\n" + "=" * 88)
    print(f"  STRATEGIES ON QQQ  ·  common live window {live.date()} → {idx.max().date()}")
    print("  overnight legs charged 2 trades/day; 2bp ≈ 10%/yr drag at 1×, 1bp ≈ 5%/yr")
    print("=" * 88)
    keys = ["CAGR", "Volatility", "Sharpe", "Sortino", "Max DD", "Calmar"]
    table = []
    for name, r in strat.items():
        m = compute_metrics(r.dropna(), rf=RF)
        ret22 = (1 + r[r.index.year == 2022]).prod() - 1
        table.append([name] + [f"{m.get(k, float('nan')):.2f}" for k in keys]
                     + [f"{ret22:+.0%}"])
    print(tabulate(table, headers=["strategy"] + keys + ["2022"], tablefmt="github"))

    print("\n" + "=" * 88)
    print(f"  ROLLING OOS vs SPY  ·  {ROLL}d (~3y) windows")
    print("=" * 88)
    for name, r in strat.items():
        if "buy" in name:
            continue
        ds = (roll_sharpe(r) - roll_sharpe(spy_live)).dropna()
        dc = (roll_cagr(r) - roll_cagr(spy_live)).dropna()
        print(f"  {name:28s} CAGR win {float((dc > 0).mean()):.0%} "
              f"(median {float(dc.median()):+.1%})   "
              f"Sharpe win {float((ds > 0).mean()):.0%} "
              f"(median {float(ds.median()):+.2f})")

    yr = pd.DataFrame(strat | {"SPY": spy_live}).dropna()
    ann = yr.groupby(yr.index.year).apply(lambda d: (1 + d).prod() - 1)
    print("\nPer-year returns (%):")
    print(tabulate((ann * 100).round(1), headers=["year"] + list(ann.columns),
                   tablefmt="github"))

    # ── Graphs ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    ax = axes[0]
    for t, col in [("SPY", "black"), ("QQQ", "grey")]:
        r_on_f, r_id_f, _ = split_returns(panel, t)
        for leg, r, ls in [("overnight", r_on_f, "-"), ("intraday", r_id_f, "--")]:
            eq = (1 + r.fillna(0)).cumprod()
            ax.plot(eq.index, eq.values, lw=1.2, ls=ls, color=col,
                    label=f"{t} {leg} only (gross)")
    ax.set_yscale("log"); ax.set_ylabel("growth of $1 (log)")
    ax.set_title("The anomaly: overnight vs intraday legs, full history, gross")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    for name in ["QQQ buy & hold", "2×vote 24h (champion)",
                 "2×vote overnight (2bp)", "2×vote overnight (1bp)"]:
        eq = (1 + strat[name].fillna(0)).cumprod()
        ax.plot(eq.index, eq.values, lw=1.3,
                color="grey" if "buy" in name else None, label=name)
    ax.set_yscale("log"); ax.set_ylabel("growth of $1 (log)")
    ax.set_title("Vote-gated overnight holding vs the 24h champion (honest costs)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2]
    for name in ["QQQ buy & hold", "2×vote 24h (champion)", "2×vote overnight (2bp)"]:
        eq = (1 + strat[name].fillna(0)).cumprod()
        dd = eq / eq.cummax() - 1
        ax.plot(dd.index, dd.values, lw=1.0,
                color="grey" if "buy" in name else None, label=name)
    ax.set_ylabel("drawdown"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("Does the overnight leg break the leverage↔drawdown link?")
    fig.tight_layout()
    out = RESULTS_DIR / "overnight_split.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    logger.info("Saved: %s", out)
    print("\n✓  Done.")


if __name__ == "__main__":
    main()
