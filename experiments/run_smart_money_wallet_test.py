#!/usr/bin/env python3
"""
Smart-money wallet test — does leaderboard-winner *selection* actually hold?
===========================================================================

A focused re-run of the ``run_polymarket_smart_money.py`` four-book panel whose
only job is to **falsify (or not) the wallet-selection premise** behind the
smart-flow consensus signal, and to dump small, tracked artifacts so the run
survives the ephemeral container.

The smart-flow paper tracker (``smartflow_papertrade.py``) follows
profit-leaderboard wallets minus the volume crowd. This script asks the prior
question those entries assume an answer to: *do the recent winners' trades
carry any edge over following everyone, or even over following the losers?*

Four books, identical plumbing, only the cohort selection differs:
  smart_money — top-N non-MM wallets by point-in-time PnL   (the hypothesis)
  all_leaders — every non-MM wallet, inventory-weighted     (does selection help?)
  anti_smart  — bottom-N non-MM wallets                     (negative control)
  with_mm     — top-N by PnL *including* market makers       (does the screen help?)

Decision rule (net of cost):
  smart_money > all_leaders  ⇒ point-in-time selection adds value
  smart_money > anti_smart   ⇒ winners' flow beats losers' (the idea holds)
  smart_money ≈ with_mm      ⇒ the MM screen does no work here

Artifacts written to ``data/polymarket_smart_money_snapshot/`` (git-tracked):
  wallet_test_summary.csv   — the four-book metric table
  wallet_test_equity.csv    — daily cumulative net/gross PnL per book
  wallet_test_cohorts.json  — rebalance-date → followed wallets, per book
  wallet_test_meta.json     — run parameters + pool/universe shape

Usage
-----
  python experiments/run_smart_money_wallet_test.py
  python experiments/run_smart_money_wallet_test.py --n-follow 30 --rebalance 5
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

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
from posterioralpha.polymarket.paths import REPO_ROOT
from posterioralpha.validation import compute_metrics

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-5s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

BOOKS = ["smart_money", "all_leaders", "anti_smart", "with_mm"]
OUT_DIR = REPO_ROOT / "data" / "polymarket_smart_money_snapshot"


def cum_pnl_dd(r: pd.Series) -> tuple[float, float]:
    """Cumulative additive PnL on $1 gross and its max drawdown."""
    eq = r.fillna(0.0).cumsum()
    dd = float((eq - eq.cummax()).min())
    return float(eq.iloc[-1]), dd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-window", type=int, default=60)
    ap.add_argument("--top-tokens", type=int, default=150)
    ap.add_argument("--max-trades", type=int, default=2000)
    ap.add_argument("--n-follow",   type=int, default=20)
    ap.add_argument("--rebalance",  type=int, default=7)
    ap.add_argument("--burn-in",    type=int, default=45)
    ap.add_argument("--cost",       type=float, default=0.005)
    ap.add_argument("--refresh",    action="store_true", help="re-pull live data (ignore cache)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── DATA (cached) ─────────────────────────────────────────────────────────
    pool = candidate_universe(profit_windows=("7d", "30d", "all"), per_window=args.per_window)
    traders = fetch_many_traders(pool["wallet"].tolist(), max_trades=args.max_trades,
                                 refresh=args.refresh)
    if not traders:
        raise SystemExit("No trader histories returned — check network / leaderboard.")

    universe = select_universe(traders, top_k=args.top_tokens)
    panel = build_token_panel(universe, use_cache=not args.refresh)
    universe = [t for t in universe if t in panel.columns]

    # ── MM screen ─────────────────────────────────────────────────────────────
    mm_watch = fetch_volume_wallets("all", limit=150)
    is_mm: dict[str, bool] = {}
    for w, t in traders.items():
        fp = mm_fingerprint(t, panel, asof=None, screen=MMScreen())
        is_mm[w] = bool(fp["is_mm"] or (w in mm_watch and fp["n_trades"] >= 200
                                        and fp["edge_per_dollar"] <= 0.01))
    n_mm = sum(is_mm.values())

    # ── Four books ────────────────────────────────────────────────────────────
    results, eq_cols, summary_rows, cohorts = {}, {}, [], {}
    for book in BOOKS:
        params = FollowParams(book=book, n_follow=args.n_follow, rebalance=args.rebalance,
                              burn_in=args.burn_in, cost=args.cost)
        res = run_smart_money_follow(traders, panel, is_mm, params)
        results[book] = res

        g_sharpe = compute_metrics(res.gross_returns, rf=0.0).get("Sharpe", 0.0)
        n_sharpe = compute_metrics(res.returns, rf=0.0).get("Sharpe", 0.0)
        g_pnl, _    = cum_pnl_dd(res.gross_returns)
        n_pnl, n_dd = cum_pnl_dd(res.returns)
        summary_rows.append({
            "book": book, "gross_sharpe": round(g_sharpe, 4), "gross_pnl": round(g_pnl, 4),
            "net_sharpe": round(n_sharpe, 4), "net_pnl": round(n_pnl, 4),
            "net_maxdd": round(n_dd, 4), "avg_turnover": round(float(res.turnover.mean()), 4),
            "avg_n_followed": round(float(res.n_cohort.mean()), 2),
        })
        eq_cols[f"{book}_net"]   = res.returns.fillna(0.0).cumsum()
        eq_cols[f"{book}_gross"] = res.gross_returns.fillna(0.0).cumsum()
        cohorts[book] = {str(d.date()): wl for d, wl in res.cohort_log.items()}

    # ── Persist artifacts ─────────────────────────────────────────────────────
    sfx = f"_n{args.n_follow}"  # keep artifacts from different cohort sizes side by side
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / f"wallet_test_summary{sfx}.csv", index=False)
    pd.DataFrame(eq_cols).to_csv(OUT_DIR / f"wallet_test_equity{sfx}.csv")
    (OUT_DIR / f"wallet_test_cohorts{sfx}.json").write_text(json.dumps(cohorts, indent=2))
    (OUT_DIR / f"wallet_test_meta{sfx}.json").write_text(json.dumps({
        "run_date": pd.Timestamp.utcnow().isoformat(),
        "params": vars(args),
        "pool_wallets": len(traders),
        "mm_flagged": n_mm,
        "universe_tokens": int(panel.shape[1]),
        "panel_days": int(panel.shape[0]),
        "panel_start": str(panel.index.min().date()),
        "panel_end": str(panel.index.max().date()),
    }, indent=2))

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n" + "═" * 84)
    print(f"  SMART-MONEY WALLET TEST  ·  $1 gross, follow top-{args.n_follow} "
          f"every {args.rebalance}d, cost={args.cost:.1%}/turn")
    print(f"  pool={len(traders)} wallets ({n_mm} MM-flagged), "
          f"universe={panel.shape[1]} tokens, burn-in={args.burn_in}d")
    print("═" * 84)
    print(tabulate(summary, headers="keys", tablefmt="rounded_grid", showindex=False))
    print(f"\n  Artifacts → {OUT_DIR.relative_to(REPO_ROOT)}/")
    print("✓  Wallet test complete.")


if __name__ == "__main__":
    main()
