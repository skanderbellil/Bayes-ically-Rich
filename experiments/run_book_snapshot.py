#!/usr/bin/env python3
"""
Order-book snapshots — accumulate the depth history Polymarket doesn't keep.
============================================================================

Polymarket exposes full *live* CLOB depth but no historical books. Exactly like
``datasets/gex_snapshots.csv`` did for dealer gamma, this script turns the
hourly paper-trade cron into a data-collection instrument: it snapshots the
books of every token the open paper-trade positions care about and appends one
row per token to ``data/paper_trade/book_snapshots.csv.gz``.

Six months of these snapshots is the only depth history the project will ever
own — and the missing input for execution modeling (the payup engine's
break-even currently *assumes* a ~1¢ half-spread; this makes it measurable).

A missed snapshot is fine; a failed workflow step is not — so "no open
positions" and "API unreachable" exit 0 with a clear message. Real code bugs
still raise.

Usage
-----
  python experiments/run_book_snapshot.py             # snapshot + append
  python experiments/run_book_snapshot.py --dry-run   # print, don't write
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import _bootstrap  # noqa: F401  (adds repo root to sys.path)

from posterioralpha.polymarket import booklog


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--dry-run", action="store_true",
                   help="fetch and print the snapshot without writing anything")
    p.add_argument("--max-tokens", type=int, default=booklog.MAX_TOKENS,
                   help="cap on the token universe (default %(default)s)")
    p.add_argument("--output", type=Path, default=None,
                   help="override output path (default: data/paper_trade/"
                        "book_snapshots.csv.gz, or $BOOK_SNAPSHOT_PATH)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    tokens = booklog.collect_target_tokens(max_tokens=args.max_tokens)
    if not tokens:
        print("book-snapshot: no open positions in any ledger — nothing to snapshot (ok).")
        return

    df = booklog.snapshot_books(tokens)
    if df.empty:
        print(f"book-snapshot: 0/{len(tokens)} books fetched — API unreachable "
              "or books empty; skipping this snapshot (ok).")
        return

    if args.dry_run:
        print(df.to_string(index=False, max_rows=20))
        print(f"book-snapshot [dry-run]: {len(df)}/{len(tokens)} tokens captured, "
              "nothing written.")
        return

    path = args.output if args.output is not None else booklog.snapshot_path()
    combined = booklog.append_snapshot(df, path=path)
    size_kb = path.stat().st_size / 1024
    print(f"book-snapshot: {len(df)}/{len(tokens)} tokens captured, "
          f"{len(combined)} rows total in {path.name} ({size_kb:.1f} KB).")


if __name__ == "__main__":
    main()
