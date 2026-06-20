#!/usr/bin/env python3
"""
Polymarket — insider pre-positioning event study
=================================================

For each price-flip event (large upward move in a short window), identify
which wallets were buying in the hours BEFORE the flip. Wallets that
systematically appear pre-flip across many events are candidate insiders —
they act before the information is public.

Methodology
-----------
1. Fetch hourly price series for a set of markets (CLOB prices-history).
2. Detect "flip events": price rises >= threshold within <= window_hours.
3. For each flip, pull trades in [flip_time - lead_hours, flip_time] via
   data-api (proxyWallet, side, price, timestamp).
4. Build a wallet x event matrix: did wallet W buy pre-event E?
5. Score wallets: appearance_rate, avg_lead_hours, avg_entry_discount
   (resolution_price - entry_price → larger = better informed).

Usage
-----
  # Analyse top 80 resolved markets (historical)
  python experiments/run_insider_event_study.py --n-markets 80

  # Analyse markets from current paper-trade ledger (open + resolved)
  python experiments/run_insider_event_study.py --from-ledger

  # Stricter flip threshold, wider lead window
  python experiments/run_insider_event_study.py --threshold 0.25 --lead-hours 48
"""
import argparse
import logging
import time
from collections import defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import requests
from tabulate import tabulate

from posterioralpha.polymarket.fetch import (
    fetch_market_trades,
    fetch_markets,
    fetch_token_history,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

LEDGER_PATH = Path(__file__).resolve().parents[1] / "data" / "paper_trade" / "smart_flow_positions.csv"


# ---------------------------------------------------------------------------
# Hourly price fetcher (wraps the existing fidelity param)
# ---------------------------------------------------------------------------

def fetch_hourly_prices(token_id: str) -> pd.Series:
    """Hourly Yes-price series for a token. Returns empty Series on failure."""
    try:
        s = fetch_token_history(token_id, fidelity_minutes=60, use_cache=False)
    except Exception as e:
        logger.warning("fetch_hourly_prices %s: %s", token_id[:20], e)
        return pd.Series(dtype=float)
    return s


# ---------------------------------------------------------------------------
# Flip event detection
# ---------------------------------------------------------------------------

def detect_price_flips(
    prices: pd.Series,
    threshold: float = 0.20,
    window_hours: int = 12,
    direction: str = "up",
) -> list[dict]:
    """Find windows where price moved >= threshold in <= window_hours.

    Parameters
    ----------
    prices       : hourly price series (DatetimeIndex, UTC)
    threshold    : minimum price move to count as a flip (default 0.20)
    window_hours : rolling window size in hours
    direction    : 'up' (price rises — YES buying surge) or 'down'

    Returns list of dicts: {flip_time, price_before, price_after, move, window_hours}
    """
    if len(prices) < window_hours + 1:
        return []

    events = []
    seen_windows: set[pd.Timestamp] = set()

    for i in range(window_hours, len(prices)):
        window = prices.iloc[i - window_hours: i + 1]
        p_start = float(window.iloc[0])
        p_end   = float(window.iloc[-1])
        move    = p_end - p_start if direction == "up" else p_start - p_end

        if move >= threshold:
            flip_time = prices.index[i]
            # deduplicate: skip if we already recorded a flip within window_hours
            if any(abs((flip_time - t).total_seconds()) < window_hours * 3600
                   for t in seen_windows):
                continue
            seen_windows.add(flip_time)
            events.append({
                "flip_time":    flip_time,
                "price_before": round(p_start, 4),
                "price_after":  round(p_end, 4),
                "move":         round(move, 4),
                "window_hours": window_hours,
            })

    return events


# ---------------------------------------------------------------------------
# Pre-event buyer extraction
# ---------------------------------------------------------------------------

def pre_event_buyers(
    trades: pd.DataFrame,
    flip_time: pd.Timestamp,
    lead_hours: int = 24,
    min_size: float = 10.0,
) -> pd.DataFrame:
    """Wallets that placed BUY orders in [flip_time - lead_hours, flip_time].

    Returns DataFrame[wallet, price, size, timestamp, lead_hours_before_flip].
    """
    if trades.empty:
        return pd.DataFrame()

    # Ensure UTC-aware comparison
    ft = flip_time
    if ft.tzinfo is None:
        ft = ft.tz_localize("UTC")

    window_start = ft - pd.Timedelta(hours=lead_hours)

    mask = (
        (trades["side"] == "BUY")
        & (trades["timestamp"] >= window_start)
        & (trades["timestamp"] < ft)
        & (trades["size"] >= min_size)
    )
    pre = trades[mask].copy()
    if pre.empty:
        return pd.DataFrame()

    pre["lead_hours"] = (ft - pre["timestamp"]).dt.total_seconds() / 3600
    return pre[["proxyWallet", "price", "size", "timestamp", "lead_hours"]]


# ---------------------------------------------------------------------------
# Event study — orchestration
# ---------------------------------------------------------------------------

def run_event_study(
    markets: pd.DataFrame,
    threshold: float = 0.20,
    window_hours: int = 12,
    lead_hours: int = 24,
    min_size: float = 10.0,
    sleep: float = 0.4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full event study across a set of markets.

    Returns
    -------
    events_df   : one row per (market, flip_event) with pre-event buyer count
    wallet_df   : one row per wallet with appearance stats across all events
    """
    all_events: list[dict] = []
    # wallet -> list of (lead_hours, entry_price, price_before, price_after)
    wallet_hits: dict[str, list[dict]] = defaultdict(list)
    wallet_domains: dict[str, list[str]] = defaultdict(list)

    total = len(markets)
    for idx, row in enumerate(markets.itertuples(index=False), 1):
        token       = getattr(row, "yes_token", None) or getattr(row, "token", None)
        condition   = getattr(row, "condition_id", "")
        question    = getattr(row, "question", "")
        domain      = getattr(row, "domain", "")
        outcome     = getattr(row, "outcome", float("nan"))

        if not token or not condition:
            continue

        logger.info("[%d/%d] %s…", idx, total, str(question)[:55])

        # 1. hourly prices
        prices = fetch_hourly_prices(token)
        if len(prices) < window_hours + 2:
            logger.debug("  skip — too few price points (%d)", len(prices))
            continue

        # 2. detect flips
        flips = detect_price_flips(prices, threshold=threshold, window_hours=window_hours)
        if not flips:
            logger.debug("  no flips found (threshold=%.2f)", threshold)
            continue

        logger.info("  %d flip(s) detected", len(flips))

        # 3. fetch trades
        trades = fetch_market_trades(condition, max_trades=3500, sleep=0)
        if trades.empty:
            continue
        # ensure proxyWallet column exists (fetch_market_trades returns it)
        if "proxyWallet" not in trades.columns:
            continue

        # 4. for each flip, find pre-event buyers
        for flip in flips:
            pre = pre_event_buyers(trades, flip["flip_time"], lead_hours=lead_hours,
                                   min_size=min_size)
            n_buyers = len(pre["proxyWallet"].unique()) if not pre.empty else 0

            all_events.append({
                "question":     str(question)[:60],
                "domain":       domain,
                "condition_id": condition,
                "flip_time":    flip["flip_time"],
                "price_before": flip["price_before"],
                "price_after":  flip["price_after"],
                "move":         flip["move"],
                "outcome":      outcome,
                "n_pre_buyers": n_buyers,
            })

            if not pre.empty:
                for _, t in pre.iterrows():
                    wallet = t["proxyWallet"]
                    wallet_hits[wallet].append({
                        "lead_hours":    t["lead_hours"],
                        "entry_price":   t["price"],
                        "price_before":  flip["price_before"],
                        "price_after":   flip["price_after"],
                        "move":          flip["move"],
                        "outcome":       outcome,
                        "question":      str(question)[:60],
                    })
                    wallet_domains[wallet].append(domain or "unknown")

        time.sleep(sleep)

    events_df = pd.DataFrame(all_events)

    # 5. wallet scorecard
    wallet_rows = []
    n_events = len(all_events)
    for wallet, hits in wallet_hits.items():
        n_appearances  = len(hits)
        n_events_hit   = len({h["question"] for h in hits})
        avg_lead       = np.mean([h["lead_hours"] for h in hits])
        avg_entry      = np.mean([h["entry_price"] for h in hits])
        avg_move       = np.mean([h["move"] for h in hits])
        # avg discount = how far below the post-flip price they entered
        avg_discount   = np.mean([h["price_after"] - h["entry_price"] for h in hits])
        # domain concentration (HHI)
        domains = wallet_domains[wallet]
        from collections import Counter
        counts  = Counter(domains)
        hhi     = sum((c / len(domains)) ** 2 for c in counts.values())
        top_dom = counts.most_common(1)[0][0]

        wallet_rows.append({
            "wallet":          wallet[:20] + "…",
            "wallet_full":     wallet,
            "n_pre_events":    n_events_hit,
            "n_trades_pre":    n_appearances,
            "pct_events":      round(n_events_hit / max(n_events, 1) * 100, 1),
            "avg_lead_h":      round(avg_lead, 1),
            "avg_entry":       round(avg_entry, 3),
            "avg_discount":    round(avg_discount, 3),
            "avg_move":        round(avg_move, 3),
            "domain_hhi":      round(hhi, 3),
            "top_domain":      top_dom,
        })

    wallet_df = pd.DataFrame(wallet_rows)
    if not wallet_df.empty:
        # Sort by: most events pre-bought, then earliest average lead time
        wallet_df = wallet_df.sort_values(
            ["n_pre_events", "avg_lead_h"],
            ascending=[False, True]
        ).reset_index(drop=True)

    return events_df, wallet_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-markets",    type=int,   default=80,
                    help="number of resolved markets to analyse (default 80)")
    ap.add_argument("--from-ledger",  action="store_true",
                    help="use current paper-trade ledger tokens instead of top-volume markets")
    ap.add_argument("--threshold",    type=float, default=0.20,
                    help="minimum price move to count as a flip (default 0.20)")
    ap.add_argument("--window-hours", type=int,   default=12,
                    help="rolling window to detect the move (default 12h)")
    ap.add_argument("--lead-hours",   type=int,   default=24,
                    help="pre-event window to look for buyers (default 24h)")
    ap.add_argument("--min-size",     type=float, default=10.0,
                    help="minimum trade size in USD to include (default $10)")
    ap.add_argument("--top-n",        type=int,   default=20,
                    help="number of top wallets to show in scorecard (default 20)")
    ap.add_argument("--out",          type=str,   default=None,
                    help="save wallet scorecard to CSV path")
    args = ap.parse_args()

    print("""
╔══════════════════════════════════════════════════════════╗
║  POLYMARKET INSIDER PRE-POSITIONING EVENT STUDY           ║
║  who buys before the price flips?                         ║
╚══════════════════════════════════════════════════════════╝""")

    # ── load market list ─────────────────────────────────────────────────────
    if args.from_ledger:
        if not LEDGER_PATH.exists():
            print("ERROR: ledger not found at", LEDGER_PATH)
            return
        ledger = pd.read_csv(LEDGER_PATH, dtype=str)
        # Use all positions (open + resolved) that have a condition_id
        markets = ledger[ledger["condition_id"].notna() & (ledger["condition_id"] != "")].copy()
        markets = markets.rename(columns={"token": "yes_token"}).drop_duplicates("yes_token")
        markets["outcome"] = markets["outcome"].replace("", float("nan")).astype(float)
        print(f"  Loaded {len(markets)} markets from ledger")
    else:
        print(f"  Fetching top {args.n_markets} resolved markets from Gamma…")
        markets = fetch_markets(n_markets=args.n_markets, closed=True)
        print(f"  Got {len(markets)} markets")

    if markets.empty:
        print("No markets to analyse.")
        return

    print(f"\n  Parameters: threshold={args.threshold}  window={args.window_hours}h"
          f"  lead={args.lead_hours}h  min_size=${args.min_size:.0f}\n")

    # ── run study ────────────────────────────────────────────────────────────
    events_df, wallet_df = run_event_study(
        markets,
        threshold=args.threshold,
        window_hours=args.window_hours,
        lead_hours=args.lead_hours,
        min_size=args.min_size,
    )

    # ── print flip events summary ─────────────────────────────────────────
    print(f"\n{'═'*82}")
    print(f"  FLIP EVENTS DETECTED  ({len(events_df)} total)")
    print(f"{'═'*82}")
    if events_df.empty:
        print("  None found — try lowering --threshold or --window-hours")
    else:
        show_events = events_df.sort_values("n_pre_buyers", ascending=False).head(15)
        print(tabulate(
            [[r["question"][:45], r["domain"][:10],
              r["flip_time"].strftime("%Y-%m-%d %H:%M") if hasattr(r["flip_time"], "strftime") else r["flip_time"],
              f"{r['price_before']:.3f}→{r['price_after']:.3f}",
              f"+{r['move']:.3f}", r["n_pre_buyers"]]
             for _, r in show_events.iterrows()],
            headers=["question", "domain", "flip_time", "price", "move", "pre_buyers"],
            tablefmt="rounded_grid",
        ))
        avg_pre = events_df["n_pre_buyers"].mean()
        print(f"\n  avg pre-buyers per flip: {avg_pre:.1f}")

    # ── print wallet scorecard ────────────────────────────────────────────
    print(f"\n{'═'*82}")
    print(f"  WALLET SCORECARD — top {args.top_n} pre-positioners")
    print(f"{'═'*82}")
    if wallet_df.empty:
        print("  No wallets found pre-positioning before any flip.")
    else:
        show_w = wallet_df.head(args.top_n)
        print(tabulate(
            [[r["wallet"], r["n_pre_events"], f"{r['pct_events']:.1f}%",
              f"{r['avg_lead_h']:.1f}h", f"{r['avg_entry']:.3f}",
              f"{r['avg_discount']:+.3f}", r["top_domain"], f"{r['domain_hhi']:.2f}"]
             for _, r in show_w.iterrows()],
            headers=["wallet", "n_events", "%_events", "avg_lead", "avg_entry",
                     "avg_discount", "top_domain", "HHI"],
            tablefmt="rounded_grid",
        ))
        print("""
  Columns:
    n_events    = distinct flip events where this wallet bought pre-flip
    %_events    = share of all flip events they appeared in
    avg_lead    = avg hours before the flip they bought (lower = earlier signal)
    avg_entry   = avg price they paid
    avg_discount= avg(price_after_flip - entry_price) — how cheap they got in
    HHI         = domain concentration (1.0 = only one domain, insider-like)
""")

    if args.out:
        wallet_df.to_csv(args.out, index=False)
        print(f"  Wallet scorecard saved → {args.out}")

    if not events_df.empty:
        out_events = Path(args.out).with_suffix("") if args.out else Path("data/paper_trade/event_study_flips.csv")
        events_df.to_csv(out_events, index=False)
        print(f"  Flip events saved     → {out_events}")

    print("\n✓  Event study complete.")


if __name__ == "__main__":
    main()
