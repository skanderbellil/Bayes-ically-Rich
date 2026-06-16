#!/usr/bin/env python3
"""
Polymarket domain specialists — does following traders *in their domain* help?
=============================================================================

The plain smart-money follow (``run_polymarket_smart_money.py``) failed: ranking
wallets by total PnL and mirroring the winners lost to mirroring the losers. One
likely culprit — the efficiency and field-shape studies showed forecasting skill
is **domain-specific** (macro ≫ sports/crypto). Ranking a wallet *globally*
blends its real politics edge with its sports noise, so the "top" cohort is a
muddle.

This study conditions skill on topic. At each rebalance we tag every outcome
token (politics / geopolitics / crypto / macro / sports / culture / other),
reconstruct each wallet's **point-in-time PnL within each domain**, and follow
the per-domain specialists *only on that domain's markets* — still mirror-and-
exit on $1 gross, costs on turnover, no lookahead.

Four books:
  specialist   — top-N per domain by in-domain point-in-time PnL  (the hypothesis)
  global       — top-K by *total* PnL, domain-blind  (the original smart-money baseline)
  anti_special — bottom-N per domain  (negative control)
  all_leaders  — every non-MM wallet, inventory-weighted  (selection-free baseline)

Validated on a deliberately **large** universe (4 leaderboard windows + the
volume leaders, hundreds of tokens) so the read isn't a single-event artefact.

Usage
-----
  python experiments/run_polymarket_specialists.py
  python experiments/run_polymarket_specialists.py --refresh
  python experiments/run_polymarket_specialists.py --per-window 120 --top-tokens 600 --n-per-domain 5
"""
import argparse
import logging

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

import _bootstrap  # noqa: F401
from posterioralpha.polymarket import (
    SpecialistParams,
    build_token_panel,
    candidate_universe,
    fetch_many_traders,
    fetch_volume_wallets,
    run_specialist_follow,
    screen_market_makers,
    select_universe,
    token_domains,
)
from posterioralpha.polymarket.paths import RESULTS_DIR, ensure_dirs
from posterioralpha.validation import compute_metrics

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-5s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

BOOKS = ["specialist", "global", "anti_special", "all_leaders"]


def cum_pnl_dd(r: pd.Series) -> tuple[float, float]:
    eq = r.fillna(0.0).cumsum()
    return float(eq.iloc[-1]), float((eq - eq.cummax()).min())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-window",   type=int, default=120, help="leaderboard wallets per window")
    ap.add_argument("--top-tokens",   type=int, default=600, help="copyable token universe size")
    ap.add_argument("--max-trades",   type=int, default=3000, help="fills pulled per wallet")
    ap.add_argument("--min-obs",      type=int, default=10,  help="min daily price points per token")
    ap.add_argument("--n-per-domain", type=int, default=5,   help="specialists followed per domain")
    ap.add_argument("--rebalance",    type=int, default=7,   help="re-select cohort every N days")
    ap.add_argument("--burn-in",      type=int, default=45,  help="history days before first selection")
    ap.add_argument("--cost",         type=float, default=0.005, help="turnover cost per unit notional")
    ap.add_argument("--refresh",      action="store_true", help="re-pull live data")
    ap.add_argument("--no-plots",     action="store_true", help="skip the dashboard")
    args = ap.parse_args()

    ensure_dirs()
    print("""
╔══════════════════════════════════════════════════════════╗
║   P O L Y M A R K E T   ·   domain specialists            ║
║   follow traders in the topic they're actually good at     ║
╚══════════════════════════════════════════════════════════╝""")

    # ── DATA: wide pool (winners across 4 windows + volume leaders) ──────────
    pool = candidate_universe(("1d", "7d", "30d", "all"), per_window=args.per_window)
    wallets = list(dict.fromkeys(pool["wallet"].tolist()
                                 + sorted(fetch_volume_wallets("all", 150))))
    logger.info("Candidate pool: %d wallets", len(wallets))
    traders = fetch_many_traders(wallets, max_trades=args.max_trades, refresh=args.refresh)
    if not traders:
        raise SystemExit("No trader histories returned.")

    universe = select_universe(traders, top_k=args.top_tokens)
    panel = build_token_panel(universe, min_obs=args.min_obs, use_cache=not args.refresh)
    universe = [t for t in universe if t in panel.columns]
    logger.info("Panel: %d days (%s → %s) × %d tokens",
                panel.shape[0], panel.index.min().date(), panel.index.max().date(), panel.shape[1])

    domain = token_domains(traders, universe)
    comp = domain.value_counts()
    logger.info("Universe domain mix: %s", dict(comp))

    is_mm = screen_market_makers(traders, panel, set(fetch_volume_wallets("all", 150)))
    logger.info("MM screen: %d/%d wallets flagged", sum(is_mm.values()), len(traders))

    # ── Walk-forward, four books ─────────────────────────────────────────────
    results = {}
    for book in BOOKS:
        params = SpecialistParams(book=book, n_per_domain=args.n_per_domain,
                                  rebalance=args.rebalance, burn_in=args.burn_in, cost=args.cost)
        results[book] = run_specialist_follow(traders, panel, domain, is_mm, params)

    # ── VALIDATION ───────────────────────────────────────────────────────────
    rows = []
    for book in BOOKS:
        res = results[book]
        g_sharpe = compute_metrics(res.gross_returns, rf=0.0).get("Sharpe", 0.0)
        n_sharpe = compute_metrics(res.returns, rf=0.0).get("Sharpe", 0.0)
        g_pnl, _    = cum_pnl_dd(res.gross_returns)
        n_pnl, n_dd = cum_pnl_dd(res.returns)
        rows.append([book, f"{g_sharpe:>6.2f}", f"{g_pnl:>+7.2f}", f"{n_sharpe:>6.2f}",
                     f"{n_pnl:>+7.2f}", f"{n_dd:>7.2f}", f"{res.turnover.mean():>6.3f}",
                     f"{res.n_cohort.mean():>5.1f}"])

    dommix = ", ".join(f"{d}:{int(n)}" for d, n in comp.items())
    print("\n" + "═" * 86)
    print(f"  POLYMARKET DOMAIN SPECIALISTS  ·  $1 gross, top-{args.n_per_domain}/domain "
          f"every {args.rebalance}d, cost={args.cost:.1%}/turn")
    print(f"  pool={len(traders)} wallets ({sum(is_mm.values())} MM), "
          f"universe={panel.shape[1]} tokens  [{dommix}]")
    print("═" * 86)
    print(tabulate(rows, headers=["book", "gross\nSharpe", "gross\nPnL", "net\nSharpe",
                                  "net\nPnL", "net\nmaxDD", "avg\nturn", "avg\n#foll"],
                   tablefmt="rounded_grid"))
    print("  Read: specialist > global ⇒ domain-conditioning rescues trader-following;")
    print("        specialist > anti_special ⇒ in-domain winners' flow predicts.")

    if not args.no_plots:
        fig, ax = plt.subplots(figsize=(11, 6))
        for book in BOOKS:
            eq = results[book].returns.fillna(0.0).cumsum()
            ax.plot(eq.index, eq.values, label=book, lw=1.6)
        ax.axhline(0.0, color="k", lw=0.6, alpha=0.4)
        ax.set_title("Polymarket domain specialists — cumulative PnL (net, $1 gross)")
        ax.set_ylabel("cumulative PnL ($/$1 gross)")
        ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=0.25)
        fig.tight_layout()
        out = RESULTS_DIR / "polymarket_specialists.png"
        fig.savefig(out, dpi=130)
        logger.info("Saved dashboard → %s", out)

    print("\n✓  Polymarket domain-specialist study complete.")


if __name__ == "__main__":
    main()
