#!/usr/bin/env python3
"""
Bankroll simulation — what a real $1,000 account would actually have done.
=========================================================================

The paper-trade ledgers report cumulative PnL as a *sum of independent
fraction-of-bankroll bets* with no cash constraint — so when a signal wants 150
concurrent positions the headline implies ~15× leverage that no real account
could deploy. This script instead runs an honest, **capital-constrained** cash
simulation:

  * start with ``--capital`` (default $1,000),
  * each new position stakes ``--stake`` of *current equity* (default 10%),
  * a trade is only taken if there is enough **free cash** — otherwise it is
    skipped (and counted), which is exactly the capacity limit a real account
    hits,
  * cash is locked in the position until it resolves, then the payout
    (shares × outcome, or shares × exit price for stop/flip) returns to cash,
  * positions still open at the end are marked at their last ``current_price``.

Events are processed in date order (resolutions before new entries on the same
day, so freed cash can be redeployed). Outputs the real ending equity, total
return, max drawdown, how many trades were actually affordable, and the
fair-pricing edge test (mean ``outcome − price_paid`` with its t-stat).

Usage
-----
  python experiments/run_bankroll_sim.py
  python experiments/run_bankroll_sim.py --capital 1000 --stake 0.10
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from posterioralpha.polymarket.paths import REPO_ROOT
from tabulate import tabulate

DATA = REPO_ROOT / "data" / "paper_trade"

# (label, file, entry-price column)
STRATEGIES = [
    ("Smart Flow",        "smart_flow_positions.csv",      "entry_ask"),
    ("Smart Flow (ROI)",  "smart_flow_roi_positions.csv",  "entry_ask"),
    ("Mid-priced YES",    "midprice_yes_positions.csv",    "entry_ask"),
    ("Dip-Confirm YES",   "dip_confirm_positions.csv",     "entry_ask"),
    ("Macro (Fed cuts)",  "macro_positions.csv",           "entry_price"),
]


def _events(df: pd.DataFrame, entry_col: str) -> list[tuple]:
    """Flatten a ledger into (date, kind, payload) events, sorted causally.

    kind="entry" carries the entry price; kind="exit" carries the realised
    per-share value (1/0 on resolution, current_price for stop/flip). Sorting
    puts exits before entries on the same day so freed cash can be reused.
    """
    ev = []
    for i, r in df.iterrows():
        entry_px = pd.to_numeric(r.get(entry_col), errors="coerce")
        if not (entry_px and entry_px > 0):
            continue
        edate = str(r.get("entry_date") or "")
        if not edate:
            continue
        ev.append((edate, 1, {"id": i, "price": float(entry_px)}))  # 1=entry (after exits)

        status = str(r.get("status") or "")
        if status in ("won", "lost"):
            per_share = pd.to_numeric(r.get("outcome"), errors="coerce")
        elif status in ("stopped", "flipped"):
            per_share = pd.to_numeric(r.get("current_price"), errors="coerce")
        else:
            per_share = None  # still open
        if per_share is not None and not math.isnan(per_share):
            xdate = str(r.get("exit_date") or edate)
            ev.append((xdate, 0, {"id": i, "value": float(per_share)}))  # 0=exit (first)
    ev.sort(key=lambda e: (e[0], e[1]))
    return ev


def simulate(df: pd.DataFrame, entry_col: str, capital: float, stake_frac: float) -> dict:
    cash = capital
    open_pos: dict[int, dict] = {}        # id -> {shares, cost}
    taken = skipped = 0
    realized = 0.0                         # $ PnL banked from closed positions
    eq_curve = []

    last_price = {i: pd.to_numeric(r.get("current_price"), errors="coerce")
                  for i, r in df.iterrows()}

    for _date, kind, p in _events(df, entry_col):
        if kind == 0:                      # exit: return payout to cash, bank realized PnL
            pos = open_pos.pop(p["id"], None)
            if pos:
                payout = pos["shares"] * p["value"]
                cash += payout
                realized += payout - pos["cost"]
        else:                              # entry: stake 10% of equity if affordable
            equity = cash + sum(v["cost"] for v in open_pos.values())
            stake = stake_frac * equity
            if stake > 0 and cash >= stake:
                open_pos[p["id"]] = {"shares": stake / p["price"], "cost": stake}
                cash -= stake
                taken += 1
            else:
                skipped += 1
        eq_curve.append(cash + sum(v["cost"] for v in open_pos.values()))

    # mark any still-open positions at last current_price → unrealized PnL
    open_val = sum(v["shares"] * float(last_price.get(i) or 0.0)
                   for i, v in open_pos.items())
    open_cost = sum(v["cost"] for v in open_pos.values())
    unrealized = open_val - open_cost
    final_equity = cash + open_val

    eq = pd.Series(eq_curve) if eq_curve else pd.Series([capital])
    peak = eq.cummax()
    max_dd = float(((eq - peak) / peak).min()) if len(eq) else 0.0

    return {
        "final_equity": final_equity,
        "total_return": final_equity / capital - 1.0,
        "realized": realized,
        "unrealized": unrealized,
        "max_dd": max_dd,
        "taken": taken,
        "skipped": skipped,
        "still_open": len(open_pos),
    }


def scale_analysis(df: pd.DataFrame, entry_col: str, stake_frac: float) -> dict:
    """Compare sizing conventions on the same trades (answers 'is it leveraged?').

      additive   = Σ stake·r_i        — the headline (sum of fraction-of-bankroll bets)
      equal_wt   = mean r_i           — de-levered: weights rescaled to sum to 100%
      peak_lev   = max simultaneous gross exposure under the additive convention

    The identity ``additive = equal_wt × (N · stake)`` shows the headline is just
    the de-levered return times the nominal summed exposure — so when many trades
    overlap, the headline is a leverage artifact, not extra revenue.
    """
    cl = df[df["status"].isin(["won", "lost"])].copy()
    cl["px"] = pd.to_numeric(cl[entry_col], errors="coerce")
    cl["oc"] = pd.to_numeric(cl["outcome"], errors="coerce")
    cl = cl.dropna(subset=["px", "oc"])
    n = len(cl)
    if n == 0:
        return {"additive": float("nan"), "equal_wt": float("nan"), "peak_lev": float("nan")}
    r = cl["oc"] / cl["px"] - 1.0
    ev = []
    for _, x in df.iterrows():
        if str(x.get("entry_date") or ""):
            ev.append((str(x["entry_date"]), +stake_frac))
        if str(x.get("exit_date") or ""):
            ev.append((str(x["exit_date"]), -stake_frac))
    ev.sort()
    cur = peak = 0.0
    for _, d in ev:
        cur += d
        peak = max(peak, cur)
    return {"additive": float(stake_frac * r.sum()), "equal_wt": float(r.mean()), "peak_lev": peak}


def edge_stat(df: pd.DataFrame, entry_col: str) -> tuple[float, float, int]:
    """Fair-pricing test on resolved positions: mean(outcome − price), t-stat, n."""
    cl = df[df["status"].isin(["won", "lost"])].copy()
    cl["px"] = pd.to_numeric(cl[entry_col], errors="coerce")
    cl["oc"] = pd.to_numeric(cl["outcome"], errors="coerce")
    cl = cl.dropna(subset=["px", "oc"])
    if len(cl) < 2:
        return float("nan"), float("nan"), len(cl)
    edge = cl["oc"] - cl["px"]
    m = edge.mean()
    se = edge.std(ddof=1) / math.sqrt(len(cl))
    return float(m), float(m / se if se else float("nan")), len(cl)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capital", type=float, default=1000.0, help="starting bankroll ($)")
    ap.add_argument("--stake",   type=float, default=0.10,   help="fraction of equity per trade")
    args = ap.parse_args()

    rows, out = [], []
    g_start = g_real = g_unreal = 0.0      # global equal-weight aggregates ($)
    for label, fname, entry_col in STRATEGIES:
        path = DATA / fname
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=str)
        if df.empty or "status" not in df.columns:
            continue
        sim = simulate(df, entry_col, args.capital, args.stake)
        m, t, n = edge_stat(df, entry_col)
        cap = args.capital
        mtm = sim["realized"] + sim["unrealized"]
        rows.append([
            label,
            f"${cap:,.0f}",
            f"${sim['final_equity']:,.0f}",
            f"{mtm/cap*100:+.1f}%",
            f"{sim['realized']/cap*100:+.1f}%",
            f"{sim['unrealized']/cap*100:+.1f}%",
            f"{sim['taken']}/{sim['taken']+sim['skipped']}",
            f"{t:+.2f}" if not math.isnan(t) else "—",
        ])
        g_start += cap
        g_real  += sim["realized"]
        g_unreal += sim["unrealized"]
        out.append({
            "strategy": label, "file": fname,
            "capital": cap, "stake_frac": args.stake,
            "final_equity": round(sim["final_equity"], 2),
            "total_return": round(sim["total_return"], 4),
            "realized": round(sim["realized"], 2),
            "unrealized": round(sim["unrealized"], 2),
            "max_dd": round(sim["max_dd"], 4),
            "trades_taken": sim["taken"], "trades_skipped": sim["skipped"],
            "still_open": sim["still_open"],
            "edge_tstat": round(t, 4) if not math.isnan(t) else "",
            "n_resolved": n,
        })

    g_mtm = g_real + g_unreal
    g_equity = g_start + g_mtm

    print(f"\n{'='*104}")
    print(f"  BANKROLL SIMULATION  ·  ${args.capital:,.0f} per strategy  ·  {args.stake:.0%} of equity per trade  ·  "
          f"capital-constrained")
    print(f"{'='*104}")
    print(tabulate(rows, headers=[
        "strategy", "start", "now", "MTM", "realized", "unrealized", "taken/seen", "edge\nt"],
        tablefmt="rounded_grid"))
    print(f"\n  GLOBAL — equal-weighted portfolio of all strategies ({len(rows)} × ${args.capital:,.0f} = ${g_start:,.0f})")
    print(f"    Equity now : ${g_equity:,.0f}   ({g_mtm/g_start*100:+.1f}% MTM)")
    print(f"    Realized   : ${g_real:+,.0f}   ({g_real/g_start*100:+.1f}%)")
    print(f"    Unrealized : ${g_unreal:+,.0f}   ({g_unreal/g_start*100:+.1f}%)")

    out.append({
        "strategy": "GLOBAL (equal-weight)", "file": "", "capital": g_start, "stake_frac": args.stake,
        "final_equity": round(g_equity, 2), "total_return": round(g_mtm / g_start, 4),
        "realized": round(g_real, 2), "unrealized": round(g_unreal, 2),
        "max_dd": "", "trades_taken": "", "trades_skipped": "", "still_open": "",
        "edge_tstat": "", "n_resolved": "",
    })
    summary_path = DATA / "bankroll_summary.csv"
    pd.DataFrame(out).to_csv(summary_path, index=False)
    print(f"\n  Summary → {summary_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
