#!/usr/bin/env python3
"""
Geo-calm sleeve — breakeven model: how many wins pay for the losses, by price
=============================================================================

A complement to the sizing study (``run_regime_sizing_sensitivity.py``). It
answers, in closed form and then on the live book: *how many winning trades does
it take to compensate the losers, and at what price does the sleeve break even?*

The model (binary YES, bought and held to settlement)
-----------------------------------------------------
Buy 1 share of a YES contract at effective entry price ``ep = price + cost``.
Per $1 staked:
    win  -> you hold $1 of settlement, profit = (1 - ep)/ep  ≡  b   (the payoff multiple)
    loss -> the contract settles $0,    profit = -1

So one winner pays for ``b = (1 - ep)/ep`` losing $1 bets, and conversely it
takes ``1/b = ep/(1 - ep)`` winners to cover a single $1 loss. Setting expected
profit to zero for equal-$ bets gives the breakeven win rate:

    p* · b - (1 - p*) = 0   =>   p* = ep = price + cost

i.e. **the breakeven win rate is just the entry price** (plus cost). Buy cheap,
need few wins; buy dear, need many. Edge = (realised win rate) - (entry price).

For the whole book the prices differ, so we don't average them — we count
dollars directly: total per-$1 P&L = Σ[win_i·b_i - loss_i] / N, and the book's
breakeven price is the single price at which its realised win rate would just
break even (= realised win rate).

Usage:
  python experiments/run_regime_breakeven_model.py
  python experiments/run_regime_breakeven_model.py --base-cost 0.03
"""
from __future__ import annotations
import argparse, math
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import _bootstrap  # noqa
from run_regime_sizing_sensitivity import gpr_calm_book

PANEL = "data/polymarket_calibration_snapshot/regime_calibration_panel.csv"
OUT = "data/polymarket_calibration_snapshot"


def wilson(k, n, z=1.96):
    """Wilson 95% CI for a win rate (small-n honest)."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-price", type=float, default=0.35)
    ap.add_argument("--base-cost", type=float, default=0.03)
    ap.add_argument("--gate", choices=["level", "level_or_vol"], default="level_or_vol")
    args = ap.parse_args()
    c = args.base_cost

    P = pd.read_csv(PANEL, parse_dates=["decision_date", "res_date"])
    book = gpr_calm_book(P, max_price=args.max_price, gate=args.gate)
    book["ep"] = (book["price"] + c).clip(0.01, 0.99)

    print("=" * 90)
    print("GEO-CALM SLEEVE — BREAKEVEN MODEL (how many wins pay for the losses, by price)")
    print("  geopolitics · YES · price<=%.2f · GPR-calm[%s] · hold 2-45d · entry spread %.0f¢"
          % (args.max_price, args.gate, c * 100))
    print("  n=%d  wins=%d  losses=%d  realised win rate=%.1f%%  avg price=%.3f (median %.3f)"
          % (len(book), int(book.outcome.sum()), int((book.outcome == 0).sum()),
             100 * book.outcome.mean(), book.price.mean(), book.price.median()))
    print("=" * 90)

    # ---------------- closed-form price grid ----------------
    print("\n[MODEL] one winner pays for b=(1-ep)/ep losers; breakeven win rate p* = ep = price+cost")
    print("  %-7s %-7s %9s %12s %12s %14s" %
          ("price", "ep", "payoff b", "breakeven p*", "wins / 1 loss", "losses / 1 win"))
    for pr in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]:
        ep = min(pr + c, 0.99)
        b = (1 - ep) / ep
        print("  %-7.2f %-7.3f %9.2f %11.1f%% %12.2f %14.2f"
              % (pr, ep, b, 100 * ep, ep / (1 - ep), b))
    print("  reading: at price 0.10 (ep %.2f) one win covers ~%.1f losses; you only need to be"
          % (0.10 + c, (1 - (0.10 + c)) / (0.10 + c)))
    print("           right %.0f%% of the time to break even." % (100 * (0.10 + c)))

    # ---------------- empirical: realised win rate vs breakeven, by price bucket ----------------
    print("\n[EMPIRICAL] realised win rate vs breakeven (= avg ep) by price bucket")
    print("  %-14s %4s %5s %9s %11s %9s %12s" %
          ("price bucket", "n", "W-L", "win rate", "breakeven", "margin", "wins/1 loss*"))
    buckets = [(0.0, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, args.max_price + 1e-9)]
    for lo, hi in buckets:
        s = book[(book.price >= lo) & (book.price < hi)]
        if not len(s):
            continue
        k, n = int(s.outcome.sum()), len(s)
        wr, be = s.outcome.mean(), s.ep.mean()
        b = (1 - be) / be
        wl, wh = wilson(k, n)
        print("  [%.2f,%.2f)%4s %4d %3d-%-2d %7.0f%% %10.0f%% %+8.0f%% %11.2f"
              % (lo, hi, "", n, k, n - k, 100 * wr, 100 * be, 100 * (wr - be), b))
    print("  *wins/1 loss = payoff multiple b at the bucket's avg entry price (what 1 winner offsets)")

    # ---------------- aggregate compensation: dollars, not averages ----------------
    b_i = (1 - book["ep"]) / book["ep"]                  # payoff multiple per trade
    win = book["outcome"].values == 1
    win_pnl = b_i[win].sum()                             # $ gained per $1/bet from winners
    loss_pnl = (~win).sum()                              # $ lost ($1 each) from losers
    net = win_pnl - loss_pnl
    nwin, nloss = int(win.sum()), int((~win).sum())
    avg_b_win = b_i[win].mean()
    # how many *additional* average-priced losers the realised winners could still absorb
    extra_losers = win_pnl - nloss
    book_be_price = book.outcome.mean()                  # breakeven price for the book = realised win rate
    print("\n[AGGREGATE] equal-$ book, per $1 staked per trade")
    print("  winners: %d trades returning total +$%.2f  (avg payoff %.2fx on a win)"
          % (nwin, win_pnl, avg_b_win))
    print("  losers : %d trades costing  total -$%.2f" % (nloss, loss_pnl))
    print("  NET    : %+.2f per $1-unit  ->  %+.0f%% on total $%d staked" % (net, 100 * net / len(book), len(book)))
    print("  the %d winners alone pay for ~%.0f $1-losers; the book had %d -> %.1fx loss coverage"
          % (nwin, win_pnl, nloss, win_pnl / max(nloss, 1)))
    print("  loss budget: could absorb ~%.0f MORE average losers (%.0f%% more losses) and still break even"
          % (extra_losers, 100 * extra_losers / max(nloss, 1)))
    print("  book breakeven PRICE = realised win rate = %.3f; sleeve buys at avg %.3f -> margin %+.3f"
          % (book_be_price, book.price.mean(), book_be_price - book.price.mean()))

    # ---------------- worst-case: consecutive-loss tolerance from the bankroll ----------------
    # at flat 2% stake, a run of L losses costs 1-(0.98)^L of equity; how deep before -25%/-50%?
    print("\n[DRAWDOWN TOLERANCE] flat 2% stake, pure losing streak")
    for dd in (0.25, 0.50):
        L = math.log(1 - dd) / math.log(1 - 0.02)
        print("  a %.0f%% drawdown needs ~%.0f straight losers (p≈%.1e at %.0f%% loss rate)"
              % (100 * dd, L, (1 - book.outcome.mean()) ** L, 100 * (1 - book.outcome.mean())))

    # ---------------- plot ----------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    pr = np.linspace(0.02, args.max_price, 200)
    ep = pr + c
    ax[0].plot(pr, (1 - ep) / ep, color="#1f9d55", lw=2, label="losses one win covers  b=(1-ep)/ep")
    ax[0].plot(pr, ep / (1 - ep), color="#c0392b", lw=2, label="wins to cover one loss  ep/(1-ep)")
    ax[0].axhline(1, color="grey", ls="--", lw=.8)
    ax[0].set_xlabel("entry price"); ax[0].set_ylabel("count (per $1 bet)")
    ax[0].set_title("Win/loss compensation vs price (%.0f¢ cost)" % (c * 100)); ax[0].legend(fontsize=8)
    # breakeven line (p*=ep) vs realised win rate by bucket
    ax[1].plot([0, args.max_price + c], [0, args.max_price + c], color="grey", ls="--", lw=1, label="breakeven: win rate = ep")
    for lo, hi in buckets:
        s = book[(book.price >= lo) & (book.price < hi)]
        if not len(s):
            continue
        k, n = int(s.outcome.sum()), len(s)
        wl, wh = wilson(k, n)
        x = s.ep.mean()
        ax[1].errorbar(x, s.outcome.mean(), yerr=[[s.outcome.mean() - wl], [wh - s.outcome.mean()]],
                       fmt="o", color="#1f9d55", capsize=3, ms=7)
        ax[1].annotate("n=%d" % n, (x, s.outcome.mean()), textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax[1].set_xlabel("effective entry price ep"); ax[1].set_ylabel("realised win rate")
    ax[1].set_title("Realised win rate vs breakeven (above line = edge)"); ax[1].legend(fontsize=8)
    ax[1].set_xlim(0, args.max_price + c + 0.02); ax[1].set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(f"{OUT}/regime_breakeven_model.png", dpi=120)
    print("\nSaved: regime_breakeven_model.png")


if __name__ == "__main__":
    main()
