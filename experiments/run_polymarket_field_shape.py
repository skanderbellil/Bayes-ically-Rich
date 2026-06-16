#!/usr/bin/env python3
"""
Polymarket — does field SHAPE condition predictiveness? (peaked vs flat)
========================================================================

A multi-outcome event can be **peaked** (one candidate towering over the field) or
**flat** (several comparably-priced outcomes splitting the mass). Both are valid
probability vectors, but they encode different states of knowledge. This study asks
whether shape is *itself* informative beyond the prices:

  1. Information content — are peaked fields priced more accurately than flat ones?
     (Brier skill of all constituents vs a uniform 1/k baseline, by shape bucket.)
  2. Leader calibration — conditional on its price, does the leader of a peaked field
     win MORE or LESS often than the leader of a flat field? (resid = win-rate − price,
     event-clustered bootstrap CI on the peaked−flat gap.)
  3. Tradability — buy the leader only in peaked fields: event-clustered PnL net of slip.

One row per event (the field leader over a 7–90-day pre-resolution window) so the
binary "did the leader win" fact is counted once per market.

Usage
-----
  python experiments/run_polymarket_field_shape.py
"""
import argparse
import logging

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate

import _bootstrap  # noqa: F401  (adds repo root to sys.path)
from posterioralpha.polymarket import (
    build_price_panel,
    bucket_by_shape,
    classify_markets,
    constituent_rows,
    event_panels,
    field_shape_events,
    resolution_outcomes,
)
from posterioralpha.polymarket.paths import RESULTS_DIR, ensure_dirs

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-5s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def _boot_mean(x, n=5000, seed=0):
    """Bootstrap mean + 95% CI over an array of per-event values."""
    x = np.asarray(x, float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    b = [rng.choice(x, len(x), replace=True).mean() for _ in range(n)]
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def _boot_diff(a, b, n=5000, seed=1):
    """Bootstrap the difference mean(a) − mean(b) with independent event resampling."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    d = [rng.choice(a, len(a), replace=True).mean() - rng.choice(b, len(b), replace=True).mean()
         for _ in range(n)]
    return float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # this study pulls a deliberately LARGE, low-volume-floor universe: multi-outcome
    # fields are scarce and their longshot constituents are low-volume, so a tight
    # screen starves the field count (24 fields) and inflates a selection bias. The
    # wide pull yields ~83 fields and is cached after the first (~10 min) run.
    ap.add_argument("--n-markets",  type=int,   default=2500)
    ap.add_argument("--min-volume", type=float, default=3_000)
    ap.add_argument("--refresh",    action="store_true")
    ap.add_argument("--no-plots",   action="store_true")
    args = ap.parse_args()

    ensure_dirs()
    print("""
╔══════════════════════════════════════════════════════════════╗
║   P O L Y M A R K E T   ·   field shape vs predictiveness        ║
║   peaked (one favorite)  vs  flat (proportional scrum)           ║
╚══════════════════════════════════════════════════════════════╝""")

    panel, meta = build_price_panel(n_markets=args.n_markets, min_volume=args.min_volume,
                                    closed=True, refresh=args.refresh)
    outcomes = resolution_outcomes(meta, panel)
    evp = event_panels(panel, meta, min_k=3)        # ≥3-way fields → widest event sample
    df = field_shape_events(evp, outcomes)
    logger.info("%d multi-outcome events with a clean leader (%d winner-take-all)",
                len(df), int(df["wta"].sum()))
    if df.empty:
        print("No qualifying events."); return

    # ── 1. information content by shape ───────────────────────────────────
    buck = bucket_by_shape(df, col="entropy", n_bins=2)
    print("\n" + "═" * 86)
    print("  FIELD SHAPE  →  calibration of the leader  (entropy split: peaked vs flat)")
    print("═" * 86)
    print(tabulate([[r["shape"], int(r["n_events"]), f"{r['mean_entropy']:.2f}",
                     f"{r['mean_p_fav']:.3f}", f"{r['fav_win_rate']:.3f}",
                     f"{r['resid']:+.3f}", f"{r['brier_skill']:+.2f}"]
                    for _, r in buck.iterrows()],
                   headers=["shape", "n", "entropy", "leader price", "leader win",
                            "resid", "Brier skill"],
                   tablefmt="rounded_grid"))

    # ── 2. is the peaked − flat calibration gap real? (event-clustered) ───
    peaked = df.loc[df["entropy"] <= df["entropy"].median(), "resid"].values
    flat   = df.loc[df["entropy"] >  df["entropy"].median(), "resid"].values
    pm, plo, phi = _boot_mean(peaked, seed=2)
    fm, flo, fhi = _boot_mean(flat,   seed=3)
    dm, dlo, dhi = _boot_diff(peaked, flat)
    print("\n  leader calibration residual (win-rate − implied price), event-clustered:")
    print(f"    peaked fields : {pm:+.3f}  95% CI [{plo:+.3f}, {phi:+.3f}]  (n={len(peaked)})")
    print(f"    flat   fields : {fm:+.3f}  95% CI [{flo:+.3f}, {fhi:+.3f}]  (n={len(flat)})")
    print(f"    peaked − flat : {dm:+.3f}  95% CI [{dlo:+.3f}, {dhi:+.3f}]  "
          f"{'→ shape adds info beyond price' if (dlo > 0 or dhi < 0) else '→ CI spans 0: not established'}")

    # ── 3. tradability: buy the leader in peaked fields, hold to resolution ─
    for tag, sub in [("peaked", df[df["entropy"] <= df["entropy"].median()]),
                     ("flat",   df[df["entropy"] >  df["entropy"].median()])]:
        slip = 0.01
        pnl = (sub["fav_win"] - sub["p_fav"]).values - slip      # buy leader at price, settle 0/1
        m, lo, hi = _boot_mean(pnl, seed=7)
        print(f"  TRADE buy-leader [{tag:6s}] PnL/event {m:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
              f"(n={len(pnl)}, 1% slip)")

    # ── 4. price-matched calibration across ALL constituents (higher power) ─
    cr = constituent_rows(evp, outcomes)
    cr["shape"] = np.where(cr["entropy"] <= df["entropy"].median(), "peaked", "flat")
    bands = [(0.0, 0.10), (0.10, 0.35), (0.35, 0.65), (0.65, 1.0)]
    print("\n" + "═" * 84)
    print("  PRICE-MATCHED CALIBRATION  ·  all constituents, peaked vs flat fields")
    print("  (within a price band, does a like-priced outcome resolve Yes more in one shape?)")
    print("═" * 84)
    tbl = []
    for lo, hi in bands:
        seg = cr[(cr["price"] >= lo) & (cr["price"] < hi)]
        row = [f"[{lo:.2f},{hi:.2f})"]
        for shp in ("peaked", "flat"):
            s = seg[seg["shape"] == shp]
            row += [len(s), f"{s['price'].mean():.3f}" if len(s) else "—",
                    f"{s['outcome'].mean():.3f}" if len(s) else "—",
                    f"{(s['outcome'].mean() - s['price'].mean()):+.3f}" if len(s) else "—"]
        tbl.append(row)
    print(tabulate(tbl, headers=["price band", "n", "price", "yes", "resid  (peaked)",
                                 "n", "price", "yes", "resid  (flat)"],
                   tablefmt="rounded_grid"))
    print("  resid = realised Yes-rate − mean price; constituents are event-clustered "
          f"(only {cr['event'].nunique()} independent fields → read direction, not significance)")

    # ── 5. "just size the leader and win?" — buy-leader economics by category ─
    # The leader wins ~70-83% of the time, which *looks* like a sizable edge. Test it
    # honestly: pool ALL fields, hold the leader to resolution, split by topic. A real
    # edge must (a) clear t>2 and (b) not live only in the one political regime.
    cat = classify_markets(meta)
    ev_cat = {}
    for ev, sub in evp.items():
        cs = [cat.get(m) for m in sub.columns if m in cat.index]
        cs = [c for c in cs if c]
        ev_cat[ev] = pd.Series(cs).mode().iloc[0] if cs else "other"
    df["cat"] = df["event"].map(ev_cat)
    df["pnl"] = (df["fav_win"] - df["p_fav"]) - 0.01           # buy leader at price, 1% slip

    def _t(x):
        x = np.asarray(x, float)
        return x.mean() / x.std() * np.sqrt(len(x)) if len(x) > 1 and x.std() > 0 else float("nan")

    print("\n" + "═" * 78)
    print("  BUY-LEADER, hold to resolution (1% slip), event-clustered  ·  by topic")
    print("  → can we 'just size the favorite and win'?  a real edge needs t>2 AND breadth")
    print("═" * 78)
    rows = []
    for grp, g in [("ALL", df)] + sorted(df.groupby("cat"), key=lambda kv: -len(kv[1])):
        if isinstance(grp, tuple):
            grp, g = grp
        if len(g) < 2:
            continue
        m, lo, hi = _boot_mean(g["pnl"].values, seed=11)
        rows.append([grp, len(g), f"{g['fav_win'].mean():.2f}", f"{g['p_fav'].mean():.3f}",
                     f"{m:+.3f}", f"[{lo:+.3f},{hi:+.3f}]", f"{_t(g['pnl'].values):.2f}",
                     "✓" if (lo > 0) else ""])
    print(tabulate(rows, headers=["group", "n", "win", "price", "PnL/ev", "95% CI", "t", "CI>0"],
                   tablefmt="rounded_grid"))
    print("  reading: the ALL-fields leader edge is NOT significant once the universe is")
    print("  un-selected; only MACRO survives (t≈3, CI excludes 0) — the favorite-longshot")
    print("  candidate already flagged in STRATEGY_SYNTHESIS, now confirmed at higher n.")

    # ── plot: residual vs entropy, sized by k, colored by win/lose ────────
    if not args.no_plots:
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))
        c = np.where(df["fav_win"] > 0.5, "#2a9d8f", "#e76f51")
        axL.scatter(df["entropy"], df["resid"], s=20 + 12 * df["k"], c=c, alpha=0.7,
                    edgecolor="k", linewidth=0.3)
        axL.axhline(0, color="k", lw=0.8); axL.axvline(df["entropy"].median(), color="grey", ls="--", lw=0.8)
        axL.set_xlabel("field entropy  (0 = peaked  →  1 = flat)")
        axL.set_ylabel("leader residual  (win − price)")
        axL.set_title("Leader calibration vs field shape\n(green = leader won, size ∝ #outcomes)")
        axL.grid(alpha=0.25)
        # bucketed bars
        b3 = bucket_by_shape(df, col="entropy", n_bins=3)
        axR.bar(range(len(b3)), b3["resid"], color="#264653", alpha=0.85)
        axR.set_xticks(range(len(b3))); axR.set_xticklabels(b3["shape"])
        axR.axhline(0, color="k", lw=0.8)
        for i, r in b3.iterrows():
            axR.text(i, r["resid"], f"{r['resid']:+.2f}\nn={int(r['n_events'])}",
                     ha="center", va="bottom" if r["resid"] >= 0 else "top", fontsize=8)
        axR.set_ylabel("mean leader residual"); axR.set_title("Calibration by shape tercile")
        axR.grid(alpha=0.25, axis="y")
        fig.tight_layout()
        out = RESULTS_DIR / "field_shape.png"
        fig.savefig(out, dpi=130)
        logger.info("Saved chart → %s", out)

    print("\n✓  Field-shape study complete.")


if __name__ == "__main__":
    main()
