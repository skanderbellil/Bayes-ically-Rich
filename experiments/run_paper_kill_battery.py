#!/usr/bin/env python3
"""
Kill battery for the forward paper-trade sleeves
=================================================

The same "we are trying to kill it" standard `run_regime_domain_validation.py`
applies to the regime study, applied to every live paper ledger — so sleeves get
retired on pre-registered evidence instead of by eyeballing the dashboard.

Why this exists: as of 2026-08, every sleeve's 95% CI on edge/$1 spans zero, the
largest |t| across the whole book is under 2, and picking the best of ~14
candidates needs |t| > 2.9 just to beat selection luck. Deleting the "losers" and
keeping the "winners" at that point is selecting on noise — the biggest apparent
winner (YES [20-40%], +214% at Kelly) has a NEGATIVE mean edge and got there on
five days of leveraged coin-flips. This script encodes the bar instead.

EVENT CLUSTERING is applied throughout: one observation per settlement date, not
per trade. These books hold many positions that settle off the same real-world
event, so nominal trade counts wildly overstate independent evidence — the
sleeves below carry 2,000+ trades against as few as 23 independent events.

The battery (all computed on resolved trades only):
  [n]     >= MIN_EVENTS independent settlement days
  [t]     event-clustered t on edge/$1 >= T_MIN
  [mc]    survives the multiple-comparison bar for K simultaneous candidates
  [oos]   edge > 0 in the most recent OOS_FRAC of settlement days
  [cost]  edge still > 0 after an incremental slippage haircut
  [boot]  day-clustered bootstrap P(edge <= 0) <= BOOT_P
  [conc]  top-1 settlement day is < TOP1_MAX of gross profit

VALIDATED only if every check passes. Anything else is KILLED with the binding
reasons listed. A KILLED sleeve is NOT deleted — it keeps trading on paper and
keeps accruing the forward data that is the only thing which can ever promote it.
Only its claim on capital is withheld.

Writes data/paper_trade/sleeve_validation.json.

Usage:
  python experiments/run_paper_kill_battery.py
  python experiments/run_paper_kill_battery.py --min-events 30 --t-min 2.0
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import _bootstrap  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_dashboard import REGISTRY, DERIVED, DATA, load_ledger  # noqa: E402

OUT = DATA / "sleeve_validation.json"
RNG = np.random.default_rng(20260809)

MIN_EVENTS = 30      # independent settlement days before a verdict is even possible
T_MIN = 2.0          # event-clustered t on edge/$1
OOS_FRAC = 0.30      # most recent share of settlement days held out
# INCREMENTAL cost only. The ledgers already enter at the ask (observed
# entry_ask - entry_mid = 0.005 = a full half-spread, median spread 0.005), so
# the touch is paid for in the recorded entry price. This models what a real
# stake pays BEYOND the touch — walking the book — calibrated to one further
# half-spread. A 3c haircut, the figure the regime study uses against mid-priced
# fills, would be ~6x the entire observed spread here and would kill every
# sleeve by assumption rather than by evidence. STRESS_HAIRCUT is reported
# alongside as a pessimistic check but is not part of the verdict.
HAIRCUT = 0.005
STRESS_HAIRCUT = 0.02
BOOT_P = 0.10        # max bootstrap P(edge <= 0)
TOP1_MAX = 0.60      # max share of gross profit from a single settlement day


def event_edges(resolved):
    """One mean edge/$1 per settlement date — the independent unit of evidence."""
    by: dict[str, list] = {}
    for t in resolved:
        by.setdefault(t["xd"] or t["ed"], []).append(t["outcome"] - t["entry"])
    days = sorted(by)
    return days, np.array([float(np.mean(by[d])) for d in days])


def tstat(x):
    x = np.asarray(x, float)
    if len(x) < 2 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))


def mc_threshold(k, sims=200_000):
    """|t| a best-of-k pick must clear so the winner isn't just selection luck:
    the 95th percentile of max|t| over k independent standard normals."""
    return float(np.percentile(np.abs(RNG.standard_normal((sims, max(k, 1)))).max(axis=1), 95))


def boot_p(ev, n=10_000):
    if len(ev) < 5:
        return float("nan")
    idx = RNG.integers(0, len(ev), size=(n, len(ev)))
    return float((ev[idx].mean(axis=1) <= 0).mean())


def top1_share(resolved, hc):
    g = np.array(sorted((t["outcome"] / (t["entry"] + hc) - 1.0 for t in resolved), reverse=True))
    gross = g[g > 0].sum()
    return float(g[0] / gross) if gross > 0 and len(g) else float("nan")


def assess(label, trades, mc_bar, args):
    resolved = [t for t in trades if t["resolved"] and (t["xd"] or t["ed"])]
    r = {"sleeve": label, "trades": len(resolved)}
    if len(resolved) < 5:
        r.update(events=0, verdict="KILLED", reasons=["no resolved history"])
        return r

    days, ev = event_edges(resolved)
    n = len(ev)
    edge = float(ev.mean())
    t = tstat(ev)
    se = float(ev.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    cut = max(1, int(n * (1 - args.oos_frac)))
    oos = ev[cut:]
    net = float(np.mean([t_["outcome"] - (t_["entry"] + args.haircut) for t_ in resolved]))
    net_stress = float(np.mean([t_["outcome"] - (t_["entry"] + STRESS_HAIRCUT) for t_ in resolved]))
    bp = boot_p(ev)
    t1 = top1_share(resolved, args.haircut)

    r.update(events=n, trades_per_event=round(len(resolved) / n, 1),
             edge=round(edge, 4), t=round(t, 2) if t == t else None,
             ci95=[round(edge - 1.96 * se, 4), round(edge + 1.96 * se, 4)] if se == se else None,
             mc_bar=round(mc_bar, 2),
             oos_n=len(oos), oos_edge=round(float(oos.mean()), 4) if len(oos) else None,
             net_cost=round(net, 4), net_stress=round(net_stress, 4),
             boot_p=round(bp, 4) if bp == bp else None,
             top1=round(t1, 3) if t1 == t1 else None)

    checks = {
        f"events>={args.min_events}": n >= args.min_events,
        f"t>={args.t_min}": t == t and t >= args.t_min,
        f"t>mc_bar({mc_bar:.2f})": t == t and t > mc_bar,
        "oos_edge>0": len(oos) > 0 and float(oos.mean()) > 0,
        f"net@{args.haircut:.3f}>0": net > 0,
        f"boot_p<={args.boot_p}": bp == bp and bp <= args.boot_p,
        f"top1<{args.top1_max}": t1 == t1 and t1 < args.top1_max,
    }
    failed = [k for k, ok in checks.items() if not ok]
    r["verdict"] = "VALIDATED" if not failed else "KILLED"
    r["reasons"] = ["survives every kill test"] if not failed else failed
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-events", type=int, default=MIN_EVENTS)
    ap.add_argument("--t-min", type=float, default=T_MIN)
    ap.add_argument("--oos-frac", type=float, default=OOS_FRAC)
    ap.add_argument("--haircut", type=float, default=HAIRCUT)
    ap.add_argument("--boot-p", type=float, default=BOOT_P)
    ap.add_argument("--top1-max", type=float, default=TOP1_MAX)
    ap.add_argument("--dry-run", action="store_true", help="print but do not write the JSON")
    args = ap.parse_args()

    sleeves = [(f, e, q, lab, None) for f, e, q, lab in REGISTRY] + \
              [(f, "entry_ask", "question", lab, filt) for f, _sid, lab, filt in DERIVED]
    mc_bar = mc_threshold(len(sleeves))

    results = []
    for fname, ecol, qcol, label, filt in sleeves:
        trades = load_ledger(DATA / fname, ecol, qcol, row_filter=filt)
        results.append(assess(label, trades, mc_bar, args))

    print("=" * 108)
    print("PAPER-SLEEVE KILL BATTERY — event-clustered, multiple-comparison aware. "
          "Trying to kill every sleeve.")
    print("=" * 108)
    def cell(r, key, fmt, width):
        v = r.get(key)
        return f"{format(v, fmt) if v is not None else '—':>{width}}"

    print(f"{'sleeve':30s} {'trd':>5} {'evt':>5} {'trd/evt':>8} {'edge':>9} {'t':>6} "
          f"{'oos':>8} {'net':>8} {'boot':>6} {'top1':>6}  verdict")
    for r in sorted(results, key=lambda x: -(x.get("t") or -99)):
        print(f"{r['sleeve'][:30]:30s} {r['trades']:5d} {r.get('events', 0):5d} "
              f"{str(r.get('trades_per_event', '—')):>8} "
              f"{cell(r, 'edge', '+.4f', 9)} {cell(r, 't', '+.2f', 6)} "
              f"{cell(r, 'oos_edge', '+.4f', 8)} {cell(r, 'net_cost', '+.4f', 8)} "
              f"{cell(r, 'boot_p', '.3f', 6)} {cell(r, 'top1', '.2f', 6)}  {r['verdict']}")

    validated = [r["sleeve"] for r in results if r["verdict"] == "VALIDATED"]
    print("\n" + "=" * 108)
    print("VALIDATED: %s" % (validated or "(none — no sleeve has yet cleared the bar)"))
    print("selection-luck bar: a best-of-%d pick needs |t| > %.2f" % (len(sleeves), mc_bar))
    print("KILLED sleeves keep trading on paper — they are not deleted; only their "
          "capital allocation is withheld.")
    print("=" * 108)
    for r in results:
        if r["verdict"] == "KILLED" and r.get("events"):
            print(f"  {r['sleeve'][:30]:30s} fails: {', '.join(r['reasons'])}")

    payload = {
        "generated": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC"),
        "standard": {"min_events": args.min_events, "t_min": args.t_min,
                     "oos_frac": args.oos_frac, "haircut": args.haircut,
                     "boot_p": args.boot_p, "top1_max": args.top1_max,
                     "mc_bar": round(mc_bar, 2), "n_candidates": len(sleeves)},
        "validated": validated,
        "per_sleeve": {r["sleeve"]: r for r in results},
    }
    if args.dry_run:
        return 0
    # Only rewrite when a verdict or statistic actually moved — otherwise the
    # hourly cron churns a timestamp-only diff into every commit.
    try:
        old = json.loads(OUT.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        old = None
    new_cmp = json.loads(json.dumps({k: v for k, v in payload.items() if k != "generated"},
                                    default=str))
    if old is not None and {k: v for k, v in old.items() if k != "generated"} == new_cmp:
        print("\nVerdicts unchanged since %s — keeping %s as-is" % (old.get("generated", "?"), OUT))
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    print("\nWrote %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
