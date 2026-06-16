#!/usr/bin/env python3
"""
Polymarket smart money — do recent winners' trades predict, net of frictions?
=============================================================================

A different cut at prediction markets from the cross-market *price* momentum
elsewhere in this subpackage: here we trade **trader flow**. The hypothesis is
*trader momentum* — wallets that have been winning keep making good trades, so a
book that mirrors the recent winners' net positions (after screening out market
makers) should earn their edge.

The honesty hinges on two things, both built in:

  * **No survivorship in timing.** The end-of-sample leaderboard only seeds the
    candidate *pool*. Who we actually follow on any date is chosen
    **point-in-time**, by the PnL each wallet had realised *strictly before* that
    date — reconstructed from its own fills marked to the CLOB panel.
  * **No market makers.** Each wallet is fingerprinted (two-sidedness, turnover,
    edge-per-dollar, trade count); MM-like accounts are flagged and excluded.

Four books make the claim falsifiable:

  smart_money — top-N non-MM wallets by point-in-time PnL          (the hypothesis)
  all_leaders — every non-MM wallet, inventory-weighted           (does selection help?)
  anti_smart  — bottom-N non-MM wallets                           (negative control)
  with_mm     — top-N by PnL *including* market makers            (does the screen matter?)

Mirror-and-exit: the cohort's target in each outcome token is the selected
leaders' aggregate net *dollar* inventory, normalised to $1 gross and marked to
the CLOB mid daily (a long captures up-moves; "shorting the Yes" is just holding
the No token). Costs charged on turnover. PnL accumulates additively on $1 gross,
exactly like ``run_polymarket_momentum.py``.

Usage
-----
  python experiments/run_polymarket_smart_money.py
  python experiments/run_polymarket_smart_money.py --refresh          # re-pull live data
  python experiments/run_polymarket_smart_money.py --n-follow 30 --rebalance 5
  python experiments/run_polymarket_smart_money.py --no-plots
"""
import argparse
import logging

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

import _bootstrap  # noqa: F401  (adds repo root to sys.path)
from posterioralpha.polymarket import (
    FollowParams,
    MMScreen,
    build_token_panel,
    candidate_universe,
    fetch_many_traders,
    fetch_volume_wallets,
    mm_fingerprint,
    run_smart_money_follow,
    select_universe,
)
from posterioralpha.polymarket.paths import RESULTS_DIR, ensure_dirs
from posterioralpha.validation import compute_metrics

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-5s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

BOOKS = ["smart_money", "all_leaders", "anti_smart", "with_mm"]


def cum_pnl_dd(r: pd.Series) -> tuple[float, float]:
    """Cumulative additive PnL on $1 gross and its max drawdown."""
    eq = r.fillna(0.0).cumsum()
    dd = float((eq - eq.cummax()).min())
    return float(eq.iloc[-1]), dd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-window", type=int, default=80, help="leaderboard wallets per window")
    ap.add_argument("--top-tokens", type=int, default=150, help="copyable token universe size")
    ap.add_argument("--max-trades", type=int, default=4000, help="fills pulled per wallet")
    ap.add_argument("--n-follow",   type=int, default=20,  help="cohort size (top-N by PnL)")
    ap.add_argument("--rebalance",  type=int, default=7,   help="re-select cohort every N days")
    ap.add_argument("--burn-in",    type=int, default=45,  help="history days before first selection")
    ap.add_argument("--cost",       type=float, default=0.005, help="turnover cost per unit notional")
    ap.add_argument("--refresh",    action="store_true", help="re-pull live data (ignore cache)")
    ap.add_argument("--no-plots",   action="store_true", help="skip the dashboard")
    args = ap.parse_args()

    ensure_dirs()

    print("""
╔══════════════════════════════════════════════════════════╗
║   P O L Y M A R K E T   ·   smart-money follow            ║
║   point-in-time copy-trading, market makers screened out   ║
╚══════════════════════════════════════════════════════════╝""")

    # ── Stage 1: DATA — candidate pool, their fills, the copyable universe ────
    logger.info("Pulling candidate winner pool (profit leaderboard) …")
    pool = candidate_universe(profit_windows=("7d", "30d", "all"), per_window=args.per_window)
    logger.info("Candidate pool: %d unique wallets", len(pool))

    logger.info("Pulling per-wallet trade history (cached) …")
    traders = fetch_many_traders(pool["wallet"].tolist(), max_trades=args.max_trades,
                                 refresh=args.refresh)
    if not traders:
        raise SystemExit("No trader histories returned — check network / leaderboard.")

    universe = select_universe(traders, top_k=args.top_tokens)
    logger.info("Copyable universe: %d outcome tokens (top by cohort notional)", len(universe))

    logger.info("Building token price panel (CLOB history, cached) …")
    panel = build_token_panel(universe, use_cache=not args.refresh)
    universe = [t for t in universe if t in panel.columns]
    logger.info("Panel: %d days (%s → %s) × %d tokens",
                panel.shape[0], panel.index.min().date(), panel.index.max().date(),
                panel.shape[1])

    # ── MM screen (fingerprint on the full history; flagged wallets excluded) ─
    mm_watch = fetch_volume_wallets("all", limit=150)
    fp_rows, is_mm = [], {}
    for w, t in traders.items():
        fp = mm_fingerprint(t, panel, asof=None, screen=MMScreen())
        # a wallet sitting in the top-volume list AND flagged behaviourally is an MM
        flagged = fp["is_mm"] or (w in mm_watch and fp["n_trades"] >= 200
                                  and fp["edge_per_dollar"] <= 0.01)
        is_mm[w] = flagged
        fp_rows.append((w[:10], fp["n_trades"], fp["two_sided"], fp["turnover"],
                        fp["edge_per_dollar"], "MM" if flagged else ""))
    n_mm = sum(is_mm.values())
    logger.info("MM screen: %d/%d wallets flagged as market-maker-like", n_mm, len(traders))

    # ── Stages 2+3: walk-forward follow, four books ──────────────────────────
    results = {}
    for book in BOOKS:
        params = FollowParams(book=book, n_follow=args.n_follow, rebalance=args.rebalance,
                              burn_in=args.burn_in, cost=args.cost)
        results[book] = run_smart_money_follow(traders, panel, is_mm, params)

    # ── Stage 4: VALIDATION — gross vs net, additive $1-gross PnL ────────────
    rows = []
    for book in BOOKS:
        res = results[book]
        g_sharpe = compute_metrics(res.gross_returns, rf=0.0).get("Sharpe", 0.0)
        n_sharpe = compute_metrics(res.returns, rf=0.0).get("Sharpe", 0.0)
        g_pnl, _    = cum_pnl_dd(res.gross_returns)
        n_pnl, n_dd = cum_pnl_dd(res.returns)
        rows.append([
            book, f"{g_sharpe:>6.2f}", f"{g_pnl:>+7.2f}",
            f"{n_sharpe:>6.2f}", f"{n_pnl:>+7.2f}", f"{n_dd:>7.2f}",
            f"{res.turnover.mean():>6.3f}", f"{res.n_cohort.mean():>5.1f}",
        ])

    print("\n" + "═" * 84)
    print(f"  POLYMARKET SMART MONEY  ·  $1 gross, follow top-{args.n_follow} "
          f"every {args.rebalance}d, cost={args.cost:.1%}/turn")
    print(f"  pool={len(traders)} wallets ({n_mm} MM-flagged), "
          f"universe={panel.shape[1]} tokens, burn-in={args.burn_in}d")
    print("  (Return/DD = cumulative PnL per $1 gross notional, additive)")
    print("═" * 84)
    print(tabulate(rows,
                   headers=["book", "gross\nSharpe", "gross\nPnL", "net\nSharpe",
                            "net\nPnL", "net\nmaxDD", "avg\nturn", "avg\n#foll"],
                   tablefmt="rounded_grid"))
    print("  Reads: smart_money > all_leaders ⇒ point-in-time selection adds value;")
    print("         smart_money > anti_smart  ⇒ winners' flow beats losers' (a real sign);")
    print("         smart_money ≈ with_mm     ⇒ the MM screen is/ isn't doing work.")

    # ── Dashboard ────────────────────────────────────────────────────────────
    if not args.no_plots:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8),
                                       gridspec_kw={"height_ratios": [3, 1]})
        for book in BOOKS:
            eq = results[book].returns.fillna(0.0).cumsum()
            ax1.plot(eq.index, eq.values, label=book, lw=1.6)
        ax1.axhline(0.0, color="k", lw=0.6, alpha=0.4)
        ax1.set_title("Polymarket smart-money follow — cumulative PnL (net, $1 gross)")
        ax1.set_ylabel("cumulative PnL ($/$1 gross)")
        ax1.legend(loc="upper left", fontsize=9)
        ax1.grid(alpha=0.25)

        ax2.plot(results["smart_money"].n_cohort.index,
                 results["smart_money"].n_cohort.values, color="tab:gray", lw=1.2)
        ax2.set_title("smart_money — traders followed per day")
        ax2.set_ylabel("# followed")
        ax2.grid(alpha=0.25)

        fig.tight_layout()
        out = RESULTS_DIR / "polymarket_smart_money.png"
        fig.savefig(out, dpi=130)
        logger.info("Saved dashboard → %s", out)

    print("\n✓  Polymarket smart-money study complete.")


if __name__ == "__main__":
    main()
