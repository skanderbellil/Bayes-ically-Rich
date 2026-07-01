#!/usr/bin/env python3
"""Mark an open paper-trade position as manually closed."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "paper_trade"

CLOB_PRICE_URL = "https://clob.polymarket.com/last-trade-price"
EXIT_SLIP = 0.01   # exit-side slippage, mirrors entry-side SLIP in papertrade.py

# strategy → (ledger csv, entry-price column, question column, token-id column)
STRATEGY_MAP = {
    "midprice_yes": (DATA / "midprice_yes_positions.csv", "entry_ask",   "question", "token"),
    "smart_flow":   (DATA / "smart_flow_positions.csv",   "entry_ask",   "question", "token"),
    "smart_flow_roi": (DATA / "smart_flow_roi_positions.csv", "entry_ask", "question", "token"),
    "macro":        (DATA / "macro_positions.csv",         "entry_price", "leader_question", "leader_token"),
}


def live_price(token_id: str) -> float | None:
    """Live CLOB last-trade price for a token; None on any failure."""
    if not token_id:
        return None
    try:
        resp = requests.get(CLOB_PRICE_URL, params={"token_id": token_id}, timeout=15)
        resp.raise_for_status()
        return float(resp.json()["price"])
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Close an open paper-trade position")
    ap.add_argument("--strategy", required=True, choices=list(STRATEGY_MAP))
    ap.add_argument("--question", required=True,
                    help="Case-insensitive substring of the market question")
    args = ap.parse_args()

    path, entry_col, q_col, tok_col = STRATEGY_MAP[args.strategy]
    if not path.exists():
        raise FileNotFoundError(f"Ledger not found: {path}")

    # token ids are 70+ digit integers — force str so pandas can't mangle them
    df = pd.read_csv(path, dtype={tok_col: str})
    for c in [entry_col, "current_price", "bet_fraction"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "bet_fraction" not in df.columns:
        df["bet_fraction"] = 0.10
    df["bet_fraction"] = df["bet_fraction"].fillna(0.10)

    mask = (df["status"] == "open") & df[q_col].str.contains(args.question, case=False, na=False)
    hits = df[mask]

    if hits.empty:
        print(f"No open position matching: {args.question!r}")
        print("Open positions:")
        for q in df[df["status"] == "open"][q_col].tolist():
            print(f"  • {q}")
        return

    if len(hits) > 1:
        print(f"Multiple matches ({len(hits)}) — closing all:")

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for idx in hits.index:
        entry   = df.loc[idx, entry_col]
        current = df.loc[idx, "current_price"] if "current_price" in df.columns else None
        bf      = df.loc[idx, "bet_fraction"]

        # exit at a live price, not the (up to an hour stale) ledger mark;
        # fall back to the ledger current_price when the fetch fails
        token = df.loc[idx, tok_col] if tok_col in df.columns else ""
        exit_price = live_price(str(token) if pd.notna(token) else "")
        source = "live"
        if exit_price is None:
            exit_price, source = current, "ledger"

        if pd.notna(exit_price) and pd.notna(entry) and entry != 0:
            sell = max(exit_price - EXIT_SLIP, 0.0)   # exit-side slippage, floored at 0
            pnl = round((sell / entry - 1) * bf, 4)
        else:
            pnl = 0.0

        df.loc[idx, "status"]    = "closed"
        df.loc[idx, "exit_date"] = today
        df.loc[idx, "pnl"]       = pnl
        if pd.notna(exit_price) and "current_price" in df.columns:
            df.loc[idx, "current_price"] = exit_price
        print(f"Closed: {df.loc[idx, q_col]}  exit={exit_price} ({source})  PnL={pnl*100:+.1f}%")

    df.to_csv(path, index=False)
    print(f"Saved → {path}")


if __name__ == "__main__":
    main()
