#!/usr/bin/env python3
"""
Trade the liquidity CALENDAR, not the level — scheduled, forward-dated drains.

Every liquidity signal so far is a lagging transform of the net-liquidity
level.  But the big liquidity shocks are mechanical, scheduled and known in
advance — the opposite of lagging:

  tax dates     : Jan/Apr/Jun/Sep 15 (next business day) — receipts settle
                  into the TGA over the following days, draining reserves
  quarter-end   : last business day of Mar/Jun/Sep/Dec — money funds window-
                  dress into the RRP, draining liquidity into the turn and
                  releasing it right after
  QRA           : quarterly refunding announcement, first Wednesday of
                  Feb/May/Aug/Nov — coupon-supply news for the quarter
  debt-ceiling  : after each resolution the Treasury rebuilds the TGA from
                  near zero — a brutal, pre-announced multi-month drain
                  (hardcoded public-record list, see DEBT_CEILING below)

Part 1 verifies the plumbing in event time straight from the FRED components
(ΔTGA, ΔRRP, Δnet-liquidity around each event type).  Part 2 asks whether
equities (QQQ, close-to-close and the overnight leg) systematically suffer in
the drain windows.  Part 3 tests one a-priori overlay on the champion: halve
the 2×vote exposure during mechanical drain windows (tax +0..+10, QRA +0..+5,
debt-ceiling +0..+60 trading days) — windows fixed by the mechanism BEFORE
looking at returns, so Part 2/3 are honest even where they fail.

Event counts are small (debt-ceiling n=10) — descriptive, not t-tested.

    python experiments/run_liquidity_calendar.py
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
NL_CSV = _bootstrap.ROOT / "datasets" / "net_liquidity.csv"
PANEL_CSV = _bootstrap.ROOT / "datasets" / "overnight_panel.csv.gz"

RF = 0.04
DAILY_RF = RF / 252
BORROW_SPREAD = 0.005
TC = 0.0002
LEV_CAP = 2.0
ROLL = 756

# Debt-ceiling suspensions/raises that triggered a TGA rebuild (public record).
# NOTE: list is complete through the Jun-2023 Fiscal Responsibility Act; any
# later deals are missing and should be appended when they happen.
DEBT_CEILING = ["2011-08-02", "2013-10-17", "2014-02-15", "2015-11-02",
                "2017-09-08", "2018-02-09", "2019-08-02", "2021-10-14",
                "2021-12-16", "2023-06-03"]

# Mechanical drain windows in TRADING days from the event (a-priori)
DRAIN = {"tax": (0, 10), "qra": (0, 5), "debt_ceiling": (0, 60)}
QE_WIN = (-3, 5)          # quarter-end: drain into the turn, release after


def to_trading(dates, index) -> list:
    """Snap calendar dates to the first trading day on/after each date."""
    out = []
    for d in pd.to_datetime(dates):
        i = index.searchsorted(d)
        if 0 < i < len(index):
            out.append(index[i])
    return sorted(set(out))


def event_dates(index) -> dict:
    years = range(index[0].year, index[-1].year + 1)
    tax = [f"{y}-{m:02d}-15" for y in years for m in (1, 4, 6, 9)]
    qra = []
    for y in years:
        for m in (2, 5, 8, 11):
            d = pd.Timestamp(y, m, 1)
            qra.append(d + pd.offsets.Week(weekday=2) - pd.offsets.Week()
                       if d.weekday() == 2 else d + pd.offsets.Week(weekday=2))
    qe = [pd.Timestamp(y, m, 1) + pd.offsets.QuarterEnd() for y in years
          for m in (3, 6, 9, 12)]
    return {"tax": to_trading(tax, index),
            "qra": to_trading(qra, index),
            "quarter_end": [index[index.searchsorted(d, side="right") - 1]
                            for d in qe if index[0] <= d <= index[-1]],
            "debt_ceiling": to_trading(DEBT_CEILING, index)}


def event_paths(series: pd.Series, events: list, lo: int, hi: int) -> pd.DataFrame:
    """Matrix of `series` aligned in event time (trading-day offsets lo..hi)."""
    idx = series.index
    cols = {}
    for e in events:
        i = idx.get_loc(e)
        if i + lo < 0 or i + hi >= len(idx):
            continue
        cols[e] = series.values[i + lo:i + hi + 1]
    return pd.DataFrame(cols, index=range(lo, hi + 1))


def main() -> None:
    raw = pd.read_csv(NL_CSV, parse_dates=["date"], index_col="date").sort_index()
    panel = pd.read_csv(PANEL_CSV, parse_dates=["Date"], index_col="Date").sort_index()
    px = load_etf_universe_prices()
    vix = pd.read_csv(VIX_CSV, parse_dates=["Date"], index_col="Date").sort_index()

    idx = (raw.dropna().index
           .intersection(panel.dropna().index)
           .intersection(px[["SPY", "QQQ", "HYG", "IEF"]].dropna().index))
    raw = raw.reindex(idx).ffill()
    qqq_c = panel["QQQ_Close"].reindex(idx)
    qqq_o = panel["QQQ_Open"].reindex(idx)
    r_cc = qqq_c.pct_change()
    r_on = qqq_o / qqq_c.shift(1) - 1.0
    spy_ret = px["SPY"].reindex(idx).pct_change()

    ev = event_dates(idx)

    # ── Part 1: verify the plumbing ──────────────────────────────────────────
    print("\n" + "=" * 92)
    print(f"  THE LIQUIDITY CALENDAR  ·  {idx[0].date()} → {idx[-1].date()}")
    print("  mean change in FRED components over each event window ($bn)")
    print("=" * 92)
    rows = []
    spec = [("tax", "WTREGEN", "ΔTGA", DRAIN["tax"]),
            ("qra", "net_liquidity", "Δnet-liq", DRAIN["qra"]),
            ("quarter_end", "RRPONTSYD", "ΔRRP into turn", (QE_WIN[0], 0)),
            ("quarter_end", "RRPONTSYD", "ΔRRP after turn", (1, QE_WIN[1])),
            ("debt_ceiling", "WTREGEN", "ΔTGA", DRAIN["debt_ceiling"]),
            ("debt_ceiling", "net_liquidity", "Δnet-liq", DRAIN["debt_ceiling"])]
    for name, comp, lab, (lo, hi) in spec:
        p = event_paths(raw[comp], ev[name], lo, hi)
        chg = p.iloc[-1] - p.iloc[0]
        rows.append([f"{name} [{lo:+d},{hi:+d}]", lab, f"{float(chg.mean()):+.0f}",
                     f"{float(chg.median()):+.0f}",
                     f"{float((chg > 0).mean()):.0%}", p.shape[1]])
    print(tabulate(rows, headers=["event window (td)", "component", "mean",
                                  "median", "% positive", "n"], tablefmt="github"))

    # ── Part 2: do equities suffer in the drain windows? ─────────────────────
    print("\n" + "=" * 92)
    print("  QQQ IN EVENT TIME  ·  annualized mean daily return inside each window")
    print("=" * 92)
    base_cc = float(r_cc.mean()) * 252
    base_on = float(r_on.mean()) * 252
    rows = []
    for name, (lo, hi) in list(DRAIN.items()) + [("quarter_end", QE_WIN)]:
        mask = pd.Series(False, index=idx)
        for e in ev[name]:
            i = idx.get_loc(e)
            mask.iloc[max(i + lo, 0):min(i + hi + 1, len(idx))] = True
        cc = float(r_cc[mask].mean()) * 252
        on = float(r_on[mask].mean()) * 252
        rows.append([f"{name} [{lo:+d},{hi:+d}]", f"{cc:+.1%}", f"{on:+.1%}",
                     f"{float(mask.mean()):.0%}", int(mask.sum())])
    rows.append(["ALL days (baseline)", f"{base_cc:+.1%}", f"{base_on:+.1%}",
                 "100%", len(idx)])
    print(tabulate(rows, headers=["drain window", "24h ann.", "overnight ann.",
                                  "share of days", "days"], tablefmt="github"))
    print("  (overlapping windows possible; descriptive, not t-tested)")

    # ── Part 3: a-priori de-risk overlay on the champion ────────────────────
    nl = load_net_liquidity()["net_liquidity"]
    vix_i = vix.reindex(idx)
    base = liquidity_vote(nl.reindex(idx), vix_i["VIX"], vix_i["VIX3M"],
                          px["HYG"].reindex(idx), px["IEF"].reindex(idx))
    dollar = soft_sign(-px["UUP"].reindex(idx).pct_change(63)).shift(1)
    vote = pd.concat([base[["liq", "vix_ts", "credit"]], dollar.rename("dollar")],
                     axis=1).mean(axis=1)

    drain_mask = pd.Series(False, index=idx)
    for name, (lo, hi) in DRAIN.items():
        for e in ev[name]:
            i = idx.get_loc(e)
            drain_mask.iloc[max(i + lo, 0):min(i + hi + 1, len(idx))] = True
    # the calendar is known in advance — but still trade with a 1-day lag for
    # consistency with the rest of the thread
    drain_mask = drain_mask.shift(1).fillna(False)

    def run(expo):
        expo = expo.clip(0.0, LEV_CAP).fillna(0.0)
        gross = expo * r_cc
        financing = np.maximum(expo - 1.0, 0.0) * (RF + BORROW_SPREAD) / 252
        cash = np.maximum(1.0 - expo, 0.0) * DAILY_RF
        costs = TC * expo.diff().abs().fillna(0.0)
        return pd.Series(gross - financing + cash - costs, index=idx), expo

    # Debt-ceiling-only variant: Part 1 shows it is the only event type whose
    # window drains NET liquidity (tax/QRA drains are offset by RRP/Fed flows)
    # — conditioning on the plumbing, not on equity returns.
    dc_mask = pd.Series(False, index=idx)
    lo, hi = DRAIN["debt_ceiling"]
    for e in ev["debt_ceiling"]:
        i = idx.get_loc(e)
        dc_mask.iloc[max(i + lo, 0):min(i + hi + 1, len(idx))] = True
    dc_mask = dc_mask.shift(1).fillna(False)

    champ_r, champ_e = run(2.0 * vote)
    cal_r, cal_e = run(2.0 * vote * np.where(drain_mask, 0.5, 1.0))
    dc_r, dc_e = run(2.0 * vote * np.where(dc_mask, 0.5, 1.0))

    live = vote.first_valid_index()
    strat = {"SPY buy & hold": spy_ret.loc[live:],
             "QQQ buy & hold": r_cc.loc[live:],
             "2×vote (champion)": champ_r.loc[live:],
             "2×vote, ×½ in drain windows": cal_r.loc[live:],
             "2×vote, ×½ debt-ceiling only": dc_r.loc[live:]}
    spy_live = spy_ret.loc[live:]

    print("\n" + "=" * 92)
    print(f"  CALENDAR DE-RISK OVERLAY  ·  common live window {live.date()} → {idx[-1].date()}")
    print(f"  drain windows cover {float(drain_mask.loc[live:].mean()):.0%} of days "
          f"(tax/QRA/debt-ceiling, fixed a-priori)")
    print("=" * 92)
    keys = ["CAGR", "Volatility", "Sharpe", "Sortino", "Max DD", "Calmar"]
    table = []
    for name, r in strat.items():
        m = compute_metrics(r.dropna(), rf=RF)
        ret22 = (1 + r[r.index.year == 2022]).prod() - 1
        table.append([name] + [f"{m.get(k, float('nan')):.2f}" for k in keys]
                     + [f"{ret22:+.0%}"])
    print(tabulate(table, headers=["strategy"] + keys + ["2022"], tablefmt="github"))

    def roll_sharpe(r):
        ex = r - DAILY_RF
        return (ex.rolling(ROLL).mean() / ex.rolling(ROLL).std()) * np.sqrt(252)

    def roll_cagr(r):
        return (1 + r).rolling(ROLL).apply(np.prod, raw=True) ** (252 / ROLL) - 1

    print(f"  debt-ceiling-only windows cover "
          f"{float(dc_mask.loc[live:].mean()):.0%} of days")
    print("\n  Rolling 3y OOS vs SPY:")
    for name in ["2×vote (champion)", "2×vote, ×½ in drain windows",
                 "2×vote, ×½ debt-ceiling only"]:
        r = strat[name]
        ds = (roll_sharpe(r) - roll_sharpe(spy_live)).dropna()
        dc = (roll_cagr(r) - roll_cagr(spy_live)).dropna()
        print(f"  {name:30s} CAGR win {float((dc > 0).mean()):.0%} "
              f"(median {float(dc.median()):+.1%})   "
              f"Sharpe win {float((ds > 0).mean()):.0%} "
              f"(median {float(ds.median()):+.2f})")

    # ── Graphs ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 13))
    ax = axes[0]
    for name, comp, lab, color in [("debt_ceiling", "WTREGEN", "TGA", "tab:red"),
                                   ("debt_ceiling", "net_liquidity", "net liquidity",
                                    "tab:blue")]:
        p = event_paths(raw[comp], ev[name], -10, 90)
        path = (p - p.loc[0]).mean(axis=1)
        ax.plot(path.index, path.values, color=color, lw=1.6,
                label=f"{lab} (mean, n={p.shape[1]})")
        ax.fill_between(path.index, (p - p.loc[0]).quantile(0.25, axis=1),
                        (p - p.loc[0]).quantile(0.75, axis=1), color=color, alpha=0.15)
    ax.axvline(0, color="grey", lw=0.8); ax.axhline(0, color="grey", lw=0.6)
    ax.set_xlabel("trading days from debt-ceiling resolution")
    ax.set_ylabel("$bn vs day 0")
    ax.set_title("The pre-announced drain: TGA rebuild after debt-ceiling deals")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    for name, color in [("tax", "tab:red"), ("qra", "tab:purple"),
                        ("quarter_end", "tab:green"), ("debt_ceiling", "tab:brown")]:
        p = event_paths(qqq_c, ev[name], -5, 21)
        path = (p / p.loc[0] - 1).mean(axis=1)
        ax.plot(path.index, path.values * 100, color=color, lw=1.5,
                label=f"{name} (n={p.shape[1]})")
    ax.axvline(0, color="grey", lw=0.8); ax.axhline(0, color="grey", lw=0.6)
    ax.set_xlabel("trading days from event"); ax.set_ylabel("mean QQQ cum ret (%)")
    ax.set_title("QQQ in event time around the four scheduled liquidity events")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2]
    for name, r in strat.items():
        eq = (1 + r.fillna(0)).cumprod()
        ax.plot(eq.index, eq.values, lw=1.3,
                color="black" if name.startswith("SPY") else
                ("grey" if name.startswith("QQQ") else None), label=name)
    ax.set_yscale("log"); ax.set_ylabel("growth of $1 (log)")
    ax.set_title("Calendar de-risk overlay on the champion")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    out = RESULTS_DIR / "liquidity_calendar.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    logger.info("Saved: %s", out)
    print("\n✓  Done.")


if __name__ == "__main__":
    main()
