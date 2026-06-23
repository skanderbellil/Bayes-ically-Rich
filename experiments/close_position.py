#!/usr/bin/env python3
"""Mark an open paper-trade position as manually closed."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "paper_trade"

STRATEGY_MAP = {
    "midprice_yes": (DATA / "midprice_yes_positions.csv", "entry_ask",   "question"),
    "smart_flow":   (DATA / "smart_flow_positions.csv",   "entry_ask",   "question"),
    "smart_flow_roi": (DATA / "smart_flow_roi_positions.csv", "entry_ask", "question"),
    "macro":        (DATA / "macro_positions.csv",         "entry_price", "leader_question"),
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Close an open paper-trade position")
    ap.add_argument("--strategy", required=True, choices=list(STRATEGY_MAP))
    ap.add_argument("--question", required=True,
                    help="Case-insensitive substring of the market question")
    args = ap.parse_args()

    path, entry_col, q_col = STRATEGY_MAP[args.strategy]
    if not path.exists():
        raise FileNotFoundError(f"Ledger not found: {path}")

    df = pd.read_csv(path)
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
        pnl = round((current / entry - 1) * bf, 4) \
              if pd.notna(current) and pd.notna(entry) and entry != 0 else 0.0

        df.loc[idx, "status"]    = "closed"
        df.loc[idx, "exit_date"] = today
        df.loc[idx, "pnl"]       = pnl
        print(f"Closed: {df.loc[idx, q_col]}  PnL={pnl*100:+.1f}%")

    df.to_csv(path, index=False)
    print(f"Saved → {path}")


if __name__ == "__main__":
    main()
