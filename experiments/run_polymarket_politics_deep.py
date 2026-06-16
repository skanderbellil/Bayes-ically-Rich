#!/usr/bin/env python3
"""
Polymarket — is the political mispricing exploitable, or one correlated era?
============================================================================

The universe-robustness study (`UNIVERSE_ROBUSTNESS.md`) localised the
favorite–longshot underpricing to **politics**. The obvious follow-up (and a fair
challenge): if it spans many different political markets, why not exploit it? This
study stress-tests exactly that.

It (1) plots the full political calibration curve, (2) clusters by parent event so
correlated sub-markets count once and bootstraps a CI over events, (3) splits by
year, and (4) runs the actual tradable strategy it implies — long under-priced /
short over-priced political markets, held to resolution, with realistic slippage
and PnL **aggregated per event** (not per market) so within-event correlation can't
inflate it — against a non-politics control.

The verdict it reaches: the effect is real, diversified across ~25 events and
significant treating events as independent — BUT every event sits in one 2023–2026
political era, so the events share a latent regime (this period's elections broke
toward outsiders / against high-priced favorites). That is a directional regime bet,
not a proven free lunch.

Usage
-----
  python experiments/run_polymarket_politics_deep.py            # cached 514-market universe
  python experiments/run_polymarket_politics_deep.py --refresh --n-markets 600
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
    build_event_table,
    build_price_panel,
    classify_markets,
    resolution_outcomes,
)
from posterioralpha.polymarket.paths import RESULTS_DIR, ensure_dirs
from posterioralpha.polymarket.trades import TradeParams, simulate_event_trades

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-5s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

CURVE_BANDS = [(0.02, 0.10), (0.10, 0.20), (0.20, 0.35), (0.35, 0.50),
               (0.50, 0.65), (0.65, 0.80), (0.80, 0.98)]


def _market_eq_resid(df: pd.DataFrame) -> tuple[float, int]:
    """One residual per market (equal weight), avoiding episode-tenure weighting."""
    mk = df.groupby("market").agg(p=("price", "mean"), o=("outcome", "first"))
    return float((mk["o"] - mk["p"]).mean()), int(len(mk))


def _event_clustered_pnl(trades: pd.DataFrame, n_boot: int = 5000, seed: int = 0) -> dict:
    """Mean PnL with each parent event as one observation + bootstrap CI over events."""
    if trades.empty:
        return {"n_trades": 0}
    ev = trades.groupby("event")["pnl"].mean()
    rng = np.random.default_rng(seed)
    boot = [rng.choice(ev.values, len(ev), replace=True).mean() for _ in range(n_boot)]
    return {
        "n_trades": int(len(trades)), "n_events": int(len(ev)),
        "per_trade": float(trades["pnl"].mean()),
        "per_event": float(ev.mean()),
        "ci_lo": float(np.percentile(boot, 2.5)),
        "ci_hi": float(np.percentile(boot, 97.5)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-markets",  type=int,   default=600)
    ap.add_argument("--min-volume", type=float, default=20_000)
    ap.add_argument("--slippage",   type=float, default=0.01)
    ap.add_argument("--refresh",    action="store_true")
    ap.add_argument("--no-plots",   action="store_true")
    args = ap.parse_args()

    ensure_dirs()
    print("""
╔══════════════════════════════════════════════════════════╗
║   P O L Y M A R K E T   ·   is the political edge real?     ║
║   event-clustered calibration + a tradable stress test      ║
╚══════════════════════════════════════════════════════════╝""")

    panel, meta = build_price_panel(n_markets=args.n_markets, min_volume=args.min_volume,
                                    closed=True, refresh=args.refresh)
    cat = classify_markets(meta)
    outcomes = resolution_outcomes(meta, panel)
    events = build_event_table(panel, outcomes)
    events["category"] = events["market"].map(cat.to_dict())
    events["event"] = events["market"].map(meta["event_ticker"].to_dict())
    events["year"] = pd.to_datetime(events["date"]).dt.year
    pol = events[events["category"] == "politics"]
    nonpol = events[events["category"] != "politics"]

    # ── 1. political calibration curve (market-equal-weight) ──────────────
    rows = []
    for lo, hi in CURVE_BANDS:
        s = pol[(pol["price"] >= lo) & (pol["price"] < hi)]
        if s["market"].nunique() < 5:
            continue
        resid, n = _market_eq_resid(s)
        rows.append([f"[{lo:.2f},{hi:.2f})", n, f"{s['price'].mean():.2f}",
                     f"{s['outcome'].mean():.1%}", f"{resid:+.1%}"])
    print("\n" + "═" * 64)
    print("  POLITICAL CALIBRATION CURVE  (one obs per market, equal weight)")
    print("═" * 64)
    print(tabulate(rows, headers=["price band", "n_mkt", "price", "yes", "residual"],
                   tablefmt="rounded_grid"))
    print("  underpriced where residual > 0 (low/mid) · overpriced where < 0 (favorites)")

    # ── 2. event-clustered significance + year split (mid band) ───────────
    band = pol[(pol["price"] >= 0.15) & (pol["price"] < 0.50)]
    mk = band.groupby("market").agg(p=("price", "mean"), o=("outcome", "first"),
                                    e=("event", "first"))
    ev_res = mk.groupby("e").apply(lambda d: (d["o"] - d["p"]).mean(), include_groups=False)
    rng = np.random.default_rng(0)
    boot = [rng.choice(ev_res.values, len(ev_res), replace=True).mean() for _ in range(5000)]
    print(f"\n  [0.15,0.50) event-clustered residual: {ev_res.mean():+.1%}  "
          f"95% CI [{np.percentile(boot,2.5):+.1%}, {np.percentile(boot,97.5):+.1%}]  "
          f"({len(ev_res)} events, {len(mk)} markets)")
    yr_rows = []
    for y in sorted(band["year"].unique()):
        r, n = _market_eq_resid(band[band["year"] == y])
        yr_rows.append([int(y), n, f"{r:+.1%}"])
    print("  by year:  " + "   ".join(f"{y}: {r} (n={n})" for y, n, r in yr_rows))

    # ── 3. tradable stress test: long under- / short over-priced, by event ─
    def leg(sub, side, lo, hi):
        tr = simulate_event_trades(panel[[c for c in sub if c in panel.columns]], outcomes,
                                   TradeParams(entry_quantile=0.0, side=side, exit="resolution",
                                               slippage=args.slippage, min_price=lo, max_price=hi)).trades
        if not tr.empty:
            tr = tr.copy(); tr["event"] = tr["market"].map(meta["event_ticker"].to_dict())
        return tr

    pol_ids = cat.index[cat == "politics"]
    non_ids = cat.index[cat != "politics"]
    long_pol  = leg(pol_ids, "yes_only", 0.10, 0.50)        # buy under-priced
    short_pol = leg(pol_ids, "no_only",  0.65, 0.95)        # fade over-priced favorites
    combo_pol = pd.concat([long_pol, short_pol], ignore_index=True)
    long_non  = leg(non_ids, "yes_only", 0.10, 0.50)        # control

    strat_rows = []
    for name, tr in [("politics: long [0.10,0.50] Yes", long_pol),
                     ("politics: short [0.65,0.95] (No)", short_pol),
                     ("politics: long+short combo", combo_pol),
                     ("non-politics: long [0.10,0.50] Yes (control)", long_non)]:
        s = _event_clustered_pnl(tr)
        if s.get("n_trades", 0) == 0:
            strat_rows.append([name, 0, 0, "—", "—", "—"]); continue
        strat_rows.append([name, s["n_trades"], s["n_events"],
                           f"{s['per_trade']:+.3f}", f"{s['per_event']:+.3f}",
                           f"[{s['ci_lo']:+.3f}, {s['ci_hi']:+.3f}]"])
    print("\n" + "═" * 92)
    print(f"  TRADABLE STRESS TEST  ·  hold to resolution, slippage {args.slippage:.0%}, "
          f"PnL clustered per event")
    print("═" * 92)
    print(tabulate(strat_rows, headers=["strategy", "n_trd", "n_evt", "PnL/trade",
                                        "PnL/event", "95% CI (per-event)"],
                   tablefmt="rounded_grid"))
    print("  per-event PnL + bootstrap CI over events = the de-correlated, honest read.")

    # ── plot: calibration curves politics vs non-politics ─────────────────
    if not args.no_plots:
        fig, ax = plt.subplots(figsize=(10, 6))
        for label, src, color in [("politics", pol, "tab:red"),
                                   ("non-politics", nonpol, "tab:blue")]:
            xs, ys = [], []
            for lo, hi in CURVE_BANDS:
                s = src[(src["price"] >= lo) & (src["price"] < hi)]
                if s["market"].nunique() < 5:
                    continue
                mk2 = s.groupby("market").agg(p=("price", "mean"), o=("outcome", "first"))
                xs.append(mk2["p"].mean()); ys.append(mk2["o"].mean())
            ax.plot(xs, ys, "o-", color=color, label=label, lw=1.8)
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="perfect calibration")
        ax.set_xlabel("market price (implied prob)"); ax.set_ylabel("actual Yes-rate")
        ax.set_title("Calibration: politics is mis-calibrated, non-politics is not")
        ax.legend(); ax.grid(alpha=0.25); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        fig.tight_layout()
        out = RESULTS_DIR / "politics_calibration.png"
        fig.savefig(out, dpi=130)
        logger.info("Saved chart → %s", out)

    print("\n✓  Political-edge stress test complete.")


if __name__ == "__main__":
    main()
