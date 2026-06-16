#!/usr/bin/env python3
"""
Polymarket order flow — does smart-crowd breadth / imbalance predict the cross-section?
======================================================================================

Beyond *following* individual traders, does the **aggregate flow** of the
candidate pool predict which markets move next? Two signals over a trailing
window, both built from the (non-MM) pool's fills:

  breadth   — # distinct wallets net-buying a token minus # net-selling it
              (a consensus signal one whale's size can't fake)
  imbalance — net signed dollars (buys − sells): classic order-flow pressure

The trap on this universe is a pervasive **favorite-drift beta** (a naive
long-all-tokens book is already ~0.5 Sharpe). So we trade each signal
**cross-sectionally and dollar-neutral** — long the top fraction, short the
bottom by the day's cross-sectional z-score — which cancels the common drift and
pays out only the signal's *ranking* power. Each signal is run both as a
follow book and as its contrarian mirror (the fade), and we print the
mean next-day Δp by signal quintile (the calibration behind the Sharpe).

Validated on a deliberately **large** universe (4 leaderboard windows + volume
leaders, hundreds of tokens).

Usage
-----
  python experiments/run_polymarket_flow.py
  python experiments/run_polymarket_flow.py --refresh
  python experiments/run_polymarket_flow.py --lookback 5 --top-frac 0.25
"""
import argparse
import logging

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

import _bootstrap  # noqa: F401
from posterioralpha.polymarket import (
    FlowParams,
    build_token_panel,
    candidate_universe,
    fetch_many_traders,
    fetch_volume_wallets,
    flow_signal_panels,
    run_flow_book,
    screen_market_makers,
    select_universe,
)
from posterioralpha.polymarket.paths import RESULTS_DIR, ensure_dirs
from posterioralpha.validation import compute_metrics

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-5s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def cum_pnl_dd(r: pd.Series) -> tuple[float, float]:
    eq = r.fillna(0.0).cumsum()
    return float(eq.iloc[-1]), float((eq - eq.cummax()).min())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-window", type=int, default=120)
    ap.add_argument("--top-tokens", type=int, default=600)
    ap.add_argument("--max-trades", type=int, default=3000)
    ap.add_argument("--min-obs",    type=int, default=10)
    ap.add_argument("--lookback",   type=int, default=7,  help="flow window (days)")
    ap.add_argument("--top-frac",   type=float, default=0.2, help="fraction per long/short leg")
    ap.add_argument("--cost",       type=float, default=0.005)
    ap.add_argument("--refresh",    action="store_true")
    ap.add_argument("--no-plots",   action="store_true")
    args = ap.parse_args()

    ensure_dirs()
    print("""
╔══════════════════════════════════════════════════════════╗
║   P O L Y M A R K E T   ·   order-flow signals            ║
║   smart-crowd breadth & imbalance, cross-sectional/neutral ║
╚══════════════════════════════════════════════════════════╝""")

    # ── DATA ─────────────────────────────────────────────────────────────────
    pool = candidate_universe(("1d", "7d", "30d", "all"), per_window=args.per_window)
    wallets = list(dict.fromkeys(pool["wallet"].tolist()
                                 + sorted(fetch_volume_wallets("all", 150))))
    traders = fetch_many_traders(wallets, max_trades=args.max_trades, refresh=args.refresh)
    if not traders:
        raise SystemExit("No trader histories returned.")
    universe = select_universe(traders, top_k=args.top_tokens)
    panel = build_token_panel(universe, min_obs=args.min_obs, use_cache=not args.refresh)
    universe = [t for t in universe if t in panel.columns]
    logger.info("Panel: %d days × %d tokens", panel.shape[0], panel.shape[1])
    is_mm = screen_market_makers(traders, panel, set(fetch_volume_wallets("all", 150)))
    logger.info("MM screen: %d/%d flagged", sum(is_mm.values()), len(traders))

    sigs = flow_signal_panels(traders, panel, is_mm, lookback=args.lookback)

    # ── Books: follow & fade for each signal ─────────────────────────────────
    books = {}
    for sig in ("breadth", "imbalance"):
        for contra in (False, True):
            name = f"{sig}{'_fade' if contra else ''}"
            params = FlowParams(signal=sig, lookback=args.lookback, top_frac=args.top_frac,
                                cost=args.cost, contrarian=contra)
            books[name] = run_flow_book(panel, sigs[sig], params)

    # ── VALIDATION ───────────────────────────────────────────────────────────
    rows = []
    for name, res in books.items():
        g = compute_metrics(res.gross_returns, rf=0.0).get("Sharpe", 0.0)
        n = compute_metrics(res.returns, rf=0.0).get("Sharpe", 0.0)
        g_pnl, _ = cum_pnl_dd(res.gross_returns)
        n_pnl, n_dd = cum_pnl_dd(res.returns)
        rows.append([name, f"{g:>6.2f}", f"{g_pnl:>+6.2f}", f"{n:>6.2f}",
                     f"{n_pnl:>+6.2f}", f"{n_dd:>6.2f}", f"{res.turnover.mean():>6.3f}"])

    print("\n" + "═" * 76)
    print(f"  POLYMARKET ORDER FLOW  ·  $1 gross dollar-neutral, lookback={args.lookback}d, "
          f"top/bottom {args.top_frac:.0%}, cost={args.cost:.1%}/turn")
    print(f"  universe={panel.shape[1]} tokens, pool={len(traders)} ({sum(is_mm.values())} MM)")
    print("  (long-short, so the favorite-drift beta is removed — this is pure ranking power)")
    print("═" * 76)
    print(tabulate(rows, headers=["book", "gross\nSharpe", "gross\nPnL", "net\nSharpe",
                                  "net\nPnL", "net\nmaxDD", "avg\nturn"],
                   tablefmt="rounded_grid"))

    # calibration: next-day drift by signal quintile
    print("\n  Next-day Δp by signal quintile (pooled, Q1=lowest signal → Q5=highest):")
    for sig in ("breadth", "imbalance"):
        qd = books[sig].quantile_drift
        if not qd.empty:
            cells = "  ".join(f"{k} {v:+.4f}" for k, v in qd.items())
            print(f"    {sig:10s}: {cells}")

    if not args.no_plots:
        fig, ax = plt.subplots(figsize=(11, 6))
        for name, res in books.items():
            eq = res.returns.fillna(0.0).cumsum()
            ax.plot(eq.index, eq.values, label=name, lw=1.5)
        ax.axhline(0.0, color="k", lw=0.6, alpha=0.4)
        ax.set_title("Polymarket order-flow — cumulative PnL (net, $1 gross, dollar-neutral)")
        ax.set_ylabel("cumulative PnL ($/$1 gross)")
        ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=0.25)
        fig.tight_layout()
        out = RESULTS_DIR / "polymarket_flow.png"
        fig.savefig(out, dpi=130)
        logger.info("Saved dashboard → %s", out)

    print("\n✓  Polymarket order-flow study complete.")


if __name__ == "__main__":
    main()
