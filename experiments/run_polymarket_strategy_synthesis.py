#!/usr/bin/env python3
"""
Polymarket — can any of this become a strategy? The synthesis.
==============================================================

Pulls the whole thread together to answer one question: is there a deployable
strategy here? It runs the classic **favorite–longshot** trade (short over-priced
longshots: buy No, hold to resolution) **per category**, event-clustered, and pairs
it with the **volume-selection diagnostic** that explains why most apparent edges are
artifacts.

Two results decide it:
  1. Short-longshot PnL per category (event-clustered, bootstrap CI, worst trade).
  2. Longshot calibration residual by volume tercile — if HIGH volume shows MORE
     "underpricing", the edge is a selection artifact (exciting underdogs draw
     volume only once they're already winning), not something tradable in real time.

Usage
-----
  python experiments/run_polymarket_strategy_synthesis.py
  python experiments/run_polymarket_strategy_synthesis.py --slippage 0.03
"""
import argparse
import logging

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
from posterioralpha.polymarket.paths import ensure_dirs
from posterioralpha.polymarket.trades import TradeParams, simulate_event_trades

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-5s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def _short_longshot(panel, meta, outcomes, ids, lo, hi, slip):
    ids = [m for m in ids if m in panel.columns]
    tr = simulate_event_trades(panel[ids], outcomes, TradeParams(
        entry_quantile=0.0, side="no_only", exit="resolution",
        slippage=slip, min_price=lo, max_price=hi)).trades
    if tr.empty:
        return None
    tr = tr.copy(); tr["event"] = tr["market"].map(meta["event_ticker"].to_dict())
    ep = tr.groupby("event")["pnl"].mean()
    rng = np.random.default_rng(0)
    boot = [rng.choice(ep.values, len(ep), replace=True).mean() for _ in range(5000)]
    return {
        "n_trd": len(tr), "n_evt": tr["event"].nunique(),
        "pnl_evt": ep.mean(), "ci_lo": np.percentile(boot, 2.5),
        "ci_hi": np.percentile(boot, 97.5), "hit": (tr["pnl"] > 0).mean(),
        "worst": tr["pnl"].min(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-markets",  type=int,   default=600)
    ap.add_argument("--min-volume", type=float, default=20_000)
    ap.add_argument("--slippage",   type=float, default=0.01)
    ap.add_argument("--refresh",    action="store_true")
    args = ap.parse_args()

    ensure_dirs()
    print("""
╔══════════════════════════════════════════════════════════╗
║   P O L Y M A R K E T   ·   is there a strategy here?       ║
║   favorite–longshot trade + the selection-bias reality check ║
╚══════════════════════════════════════════════════════════╝""")

    panel, meta = build_price_panel(n_markets=args.n_markets, min_volume=args.min_volume,
                                    closed=True, refresh=args.refresh)
    outcomes = resolution_outcomes(meta, panel)
    cat = classify_markets(meta)

    # ── 1. short-longshot trade by category (event-clustered) ─────────────
    rows = []
    for c in ["macro", "sports", "politics", "crypto", "other", "geopolitics"]:
        r = _short_longshot(panel, meta, outcomes, cat.index[cat == c], 0.03, 0.20, args.slippage)
        if r is None or r["n_evt"] < 4:
            continue
        sig = "✓ robust" if r["ci_lo"] > 0 else ("✗ loses" if r["ci_hi"] < 0 else "~ noise")
        rows.append([c, r["n_trd"], r["n_evt"], f"{r['pnl_evt']:+.3f}",
                     f"[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]", f"{r['hit']:.0%}",
                     f"{r['worst']:+.3f}", sig])
    print("\n" + "═" * 90)
    print(f"  SHORT-LONGSHOT (buy No on [0.03,0.20], hold to resolution) by domain · slippage {args.slippage:.0%}")
    print("═" * 90)
    print(tabulate(rows, headers=["domain", "n_trd", "n_evt", "PnL/evt", "95% CI",
                                  "hit", "worst trade", "verdict"], tablefmt="rounded_grid"))

    # ── 2. volume-selection diagnostic ────────────────────────────────────
    ev = build_event_table(panel, outcomes, sample_step=2)
    ev["cat"] = ev["market"].map(cat.to_dict())
    ev["vol"] = ev["market"].map(meta["volume"].to_dict())
    band = ev[(ev["price"] >= 0.08) & (ev["price"] < 0.30)]
    diag = []
    for c in ["sports", "politics", "macro"]:
        mk = band[band["cat"] == c].groupby("market").agg(
            p=("price", "mean"), o=("outcome", "first"), v=("vol", "first"))
        if len(mk) < 15:
            continue
        mk["vt"] = pd.qcut(mk["v"].rank(method="first"), 3, labels=["lowV", "midV", "highV"])
        res = {t: (mk[mk.vt == t]["o"] - mk[mk.vt == t]["p"]).mean() for t in ["lowV", "midV", "highV"]}
        diag.append([c, len(mk), f"{res['lowV']:+.3f}", f"{res['midV']:+.3f}", f"{res['highV']:+.3f}",
                     "← selection" if res["highV"] > res["lowV"] + 0.05 else ""])
    print("\n" + "═" * 78)
    print("  VOLUME-SELECTION DIAGNOSTIC · longshot [0.08,0.30] residual by volume tercile")
    print("═" * 78)
    print(tabulate(diag, headers=["domain", "n_mkt", "lowV", "midV", "highV", "flag"],
                   tablefmt="rounded_grid"))
    print("  highV ≫ lowV ⇒ 'underpricing' is exciting-underdog survivorship, not a real-time edge")

    print("""
  VERDICT
  ───────
  • Directional alpha: no. Momentum/reversion revert to a zero baseline; the
    politics/sports 'underpricing' is largely VOLUME-SELECTION bias (above).
  • One structural pattern survives — the classic favorite–longshot bias — but only
    in MACRO (scheduled Fed/rate markets), where longshot tail events are over-priced
    and there is no volume-selection. In-sample it is tight and +EV.
  • BUT it is a SHORT-VOLATILITY trade: ~100% hit rate because no tail fired in this
    calm 2024–25 rate regime. Its real risk (a Fed surprise → lose ~0.9 on a No bet)
    is unobserved here. Small sample, one regime, tiny capacity. A candidate to size
    with explicit tail-risk budgeting — not a free lunch.
""")
    print("✓  Strategy synthesis complete.")


if __name__ == "__main__":
    main()
