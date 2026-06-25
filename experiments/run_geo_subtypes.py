#!/usr/bin/env python3
"""
What *types* of geopolitical markets carry the calm-underpricing edge?
=====================================================================

"Geopolitics" is too coarse. This buckets the cheap geopolitical YES tail by the
*kind* of event and asks (a) where the calibration edge concentrates, and (b)
whether tilting the live strategy toward the high-edge buckets actually helps.

Two findings:
  * The edge is in **deadline-bound, named-actor, discrete-event** markets
    ("Will <actor> do <action> by <date>?"), and it is SYMMETRIC across direction:
    both ESCALATION (strike/nuclear/closure) and DE-ESCALATION (ceasefire/peace)
    are underpriced — peace deals most of all. That points to status-quo /
    continuity bias (the crowd over-weights "nothing decisive happens by the
    deadline"), not one-sided dread of war. Vague no-catalyst "other" markets are
    if anything slightly OVER-priced.
  * A sub-type tilt does NOT sharpen the deployable book: the regime + cheap-price
    + 2-45d horizon filters already screen out the bad "other" markets (only ~1
    survives into the live book), and cherry-picking the winning buckets just
    shrinks n and lowers the Sharpe (overfitting penalty). So we DON'T add a
    sub-type filter to the live strategy — the result is a reassuring negative.

Usage: python experiments/run_geo_subtypes.py
"""
from __future__ import annotations
import re
import numpy as np, pandas as pd
import _bootstrap  # noqa
from run_validated_regime_backtest import gpr_calm_flag, sim, trade_metrics, tstat, boot_ci

PANEL = "data/polymarket_calibration_snapshot/regime_calibration_panel.csv"
_PATS = [
    ("strike/kinetic", r"strike|military action|target|attack|missile|forces enter|enters |win on"),
    ("ceasefire/peace", r"ceasefire|peace deal|truce|withdraw|extension"),
    ("nuclear", r"nuclear|enrich|uranium"),
    ("closure", r"airspace|shipping|hormuz|island"),
    ("leadership", r"supreme leader|president|out as|parliament|dissolved|khamenei"),
    ("sanctions", r"sanction|mineral deal|tariff|transit fee"),
]


def subtype(q: str) -> str:
    ql = str(q).lower()
    for b, p in _PATS:
        if re.search(p, ql):
            return b
    return "other"


def main():
    P = pd.read_csv(PANEL, parse_dates=["decision_date", "res_date"])
    P["CE"] = P["outcome"] - P["price"]
    geo = P[(P.domain == "geopolitics") & (P.leg == "yes") & (P.price <= 0.35)].copy()
    geo["styp"] = geo["question"].apply(subtype)
    geo["deadline"] = geo["question"].str.lower().str.contains(r"\bby \w|\bbefore |on 20|on \d", regex=True)

    print("=" * 92)
    print("SUB-TYPE BREAKDOWN — cheap geopolitical YES tail (price<=0.35, n=%d), CE=outcome-price" % len(geo))
    print("=" * 92)
    print("  %-18s %4s %6s %7s %9s %6s" % ("sub-type", "n", "price", "realiz", "CE", "t"))
    for b, s in sorted(geo.groupby("styp"), key=lambda kv: -kv[1].CE.mean()):
        print("  %-18s %4d %6.3f %7.3f %+9.4f %+6.2f" % (b, len(s), s.price.mean(), s.outcome.mean(), s.CE.mean(), tstat(s.CE)))

    esc = geo[geo.styp.isin(["strike/kinetic", "nuclear", "closure"])]
    dee = geo[geo.styp == "ceasefire/peace"]
    print("\n[DIRECTION] is underpricing symmetric? (status-quo bias) vs one-sided (dread)")
    for lbl, s in [("escalation (strike/nuclear/closure)", esc), ("de-escalation (ceasefire/peace)", dee)]:
        print("  %-38s n=%3d realiz=%.3f CE=%+.4f (t=%+.2f)" % (lbl, len(s), s.outcome.mean(), s.CE.mean(), tstat(s.CE)))
    print("\n[STRUCTURE] deadline-bound 'by <date>' vs open-ended")
    for lbl, s in [("deadline-bound", geo[geo.deadline]), ("open-ended", geo[~geo.deadline])]:
        print("  %-14s n=%3d realiz=%.3f CE=%+.4f (t=%+.2f)" % (lbl, len(s), s.outcome.mean(), s.CE.mean(), tstat(s.CE)))

    # ---- does a sub-type tilt help the DEPLOYABLE book? ----
    Pc = gpr_calm_flag(P, gate="level_or_vol")
    hold = (Pc["res_date"] - Pc["decision_date"]).dt.days
    book = Pc[(Pc.domain == "geopolitics") & (Pc.leg == "yes") & (Pc.price <= 0.35)
              & Pc["calm"] & hold.between(2, 45)].copy()
    book["styp"] = book["question"].apply(subtype)

    def line(name, b):
        tm = trade_metrics(b, 0.03); _, m = sim(b, stake=0.02, cost=0.03); lo, hi, _ = boot_ci(b, 0.03)
        print("  %-28s n=%2d net=%+.4f (t=%+.2f) PF=%4.1f Sharpe=%.2f maxDD=%3.0f%% CI[%+.2f,%+.2f]"
              % (name, tm["n"], tm["net_edge"], tm["t"], tm["profit_factor"],
                 m.get("sharpe", float("nan")), 100 * m.get("maxdd", float("nan")), lo, hi))

    print("\n[TILT TEST] deployable OR-gate book (hold 2-45d, net of 3c, 2% stake)")
    line("FULL (all sub-types)", book)
    line("exclude 'other' (defensible)", book[book.styp != "other"])
    line("exclude other+closure", book[~book.styp.isin(["other", "closure"])])
    line("ONLY peace+strike (overfit)", book[book.styp.isin(["ceasefire/peace", "strike/kinetic"])])
    print("\n  book sub-type counts:", book["styp"].value_counts().to_dict())
    print("  -> tilt is redundant: regime+price+horizon filters already drop the bad 'other' bucket;")
    print("     cherry-picking winners shrinks n and lowers Sharpe. Keep the full book.")


if __name__ == "__main__":
    main()
