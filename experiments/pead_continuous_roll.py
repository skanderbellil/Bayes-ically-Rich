#!/usr/bin/env python3
"""Continuous-roll PEAD market-neutral backtest (realistic turnover accounting).

The batch backtest (pead_dynamic_exit.py) treats each quarter as an independent book
and charges a full round-trip on the whole book every quarter — even for a name that
stays top-quintile and is really just *held*. That over-charges turnover.

This version simulates the deployed process: at each name's earnings it re-ranks and
ROLLS (no trade) if the side is unchanged, FLIPS or EXITS only on a real change, and
STOPS out at -stop% from entry (then flat until the name's next earnings). Cost is
charged only on actual trades. Gross returns are identical to the batch version; only
the cost differs — so this should come out slightly *better*, confirming the batch
number is conservative. Reports live turnover and holding-period stats.

Mega+Large, t+1 entry, 6% stop-from-entry, hold-to-next-earnings, net of Alpaca costs.
Requires data/raw/pead/ (run experiments/run_pead.py first).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from posterioralpha.backtest.alpaca_costs import AlpacaCosts
from experiments.pead_dynamic_exit import build_paths, _stats, COST

STOP, HOLD = 6.0, 63
AC = AlpacaCosts()
ONEWAY = AC.oneway_cost("Large Cap")          # half-spread, one fill
ROUNDTRIP = AC.roundtrip_cost("Large Cap")    # entry + exit


def position_outcome(path, sign):
    """Return (signal-dir return %, holding days, stopped?) under a -STOP%-from-entry stop."""
    c = sign * path[1:HOLD + 1]; c = c[~np.isnan(c)]
    if len(c) == 0:
        return np.nan, 0, False
    hit = np.where(c <= -STOP)[0]
    if len(hit):
        return -STOP, int(hit[0] + 1), True
    return float(c[-1]), len(c), False


def main():
    P, M = build_paths(); M = M.reset_index(drop=True)

    # target side per announcement: +1 top-quintile SUE, -1 bottom, 0 otherwise (per season)
    M["target"] = 0
    for _, g in M.groupby("season"):
        if len(g) < 10:
            continue
        hi, lo = g["sue"].quantile(0.8), g["sue"].quantile(0.2)
        M.loc[g.index[g["sue"] >= hi], "target"] = 1
        M.loc[g.index[g["sue"] <= lo], "target"] = -1

    # per-position outcome
    out = [position_outcome(P[i], M.at[i, "target"]) if M.at[i, "target"] != 0 else (np.nan, 0, False)
           for i in M.index]
    M["ret"], M["hold"], M["stopped"] = [o[0] for o in out], [o[1] for o in out], [o[2] for o in out]

    seasons = sorted(M["season"].unique())
    gross, batch_net, cont_net = {}, {}, {}
    prev_side = {}          # ticker -> side held coming into this season
    roll = flip = entry = exitc = stopc = 0
    holds = []

    for s in seasons:
        gq = M[M["season"] == s]
        book = gq[gq["target"] != 0]
        if len(book) < 8:
            prev_side = {}    # reset chain if a season is too thin
            continue
        longs = book[book["target"] == 1]["ret"].dropna()
        shorts = book[book["target"] == -1]["ret"].dropna()
        if len(longs) < 2 or len(shorts) < 2:
            continue
        g_ls = (longs.mean() + shorts.mean()) / 100      # signal-dir, gross
        gross[s] = g_ls
        batch_net[s] = g_ls - COST                       # batch: whole book round-trips

        # continuous turnover: one fill per actual side change; rolls are free
        n_long, n_short = max(len(longs), 1), max(len(shorts), 1)
        cost = 0.0
        cur_side = {}
        for _, r in book.iterrows():
            t, side = r["ticker"], int(r["target"])
            cur_side[t] = side
            w = 1.0 / (n_long if side == 1 else n_short)
            p = prev_side.get(t, 0)
            if p == side:
                roll += 1                                # no trade
            elif p == 0:
                cost += w * ONEWAY; entry += 1           # open
            else:
                cost += w * 2 * ONEWAY; flip += 1        # close + open
            if r["stopped"]:                             # stop closes early → extra exit fill
                cost += w * ONEWAY; stopc += 1
            holds.append(r["hold"])
        # names that were in the book last quarter but dropped out → exit fills
        for t, p in prev_side.items():
            if p != 0 and t not in cur_side:
                cost += (1.0 / (n_long if p == 1 else n_short)) * ONEWAY; exitc += 1
        cont_net[s] = g_ls - cost
        # carry forward: stopped names are flat into next season
        prev_side = {t: (0 if M[(M["season"] == s) & (M["ticker"] == t)]["stopped"].any() else sd)
                     for t, sd in cur_side.items()}

    gS, bS, cS = pd.Series(gross), pd.Series(batch_net), pd.Series(cont_net)
    common = bS.index.intersection(cS.index)
    oos = [s for s in common if s.year >= 2014]

    def stat(series, idx):
        return _stats(series.loc[idx].values)

    print("Continuous-roll vs batch backtest (Mega+Large, 6% stop, net of Alpaca cost):\n")
    print(f"  {'Version':<26}{'Full Ann':>9}{'Shrp':>6} | {'OOS Ann':>8}{'Shrp':>6}{'MaxDD':>7}")
    for lbl, ser in [("Batch (charges full RT/qtr)", bS), ("Continuous roll (real turnover)", cS)]:
        a, b = stat(ser, common), stat(ser, oos)
        print(f"  {lbl:<26}{a[0]:>+9.1%}{a[1]:>6.2f} | {b[0]:>+8.1%}{b[1]:>6.2f}{b[2]:>+7.1%}")

    tot = roll + flip + entry + exitc
    print(f"\nLive turnover / holding stats:")
    print(f"  position-quarter decisions: roll {roll/tot:.0%}, new-entry {entry/tot:.0%}, "
          f"flip {flip/tot:.0%}, exit {exitc/tot:.0%}")
    print(f"  stop-outs: {stopc} positions ({stopc/(roll+flip+entry):.0%} of opened)")
    print(f"  median holding {np.median(holds):.0f} trading days (IQR "
          f"{np.percentile(holds,25):.0f}-{np.percentile(holds,75):.0f})")
    rolled_frac = roll / tot
    print(f"  → {rolled_frac:.0%} of position-quarters are rolls (no trade) — the batch model "
          f"needlessly pays\n    a round-trip on those, which is why continuous comes out ahead.")


if __name__ == "__main__":
    main()
