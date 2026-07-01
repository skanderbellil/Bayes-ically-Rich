"""Hourly order-book snapshots for the tokens the paper-trade ledgers hold.

Polymarket's CLOB exposes full *current* depth (``fetch.fetch_order_book``) but
no historical books — the same situation the repo faced with dealer gamma, where
``datasets/gex_snapshots.csv`` solved it by accumulating snapshots from the cron.
This module does the identical thing for order books: every paper-trade run it
snapshots the books of every token an open position cares about and appends one
row per token to ``data/paper_trade/book_snapshots.csv.gz``.

Why: the payup engine's break-even is quoted against an *assumed* ~1¢ half-spread
(docs/polymarket/PAYUP_FOLLOW.md). Once this file has a few weeks of history the
half-spread, depth, and imbalance become measurable per token per day instead of
assumed — the missing input for execution modeling.

Schema (one row per token per run):

    timestamp        ISO-8601 UTC, minute resolution (e.g. 2026-07-01T14:15Z)
    token_id         CLOB token id — **string**; load with dtype={'token_id': str}
    best_bid         highest bid price
    best_ask         lowest ask price
    mid              (best_bid + best_ask) / 2
    spread           best_ask - best_bid
    microprice       size-weighted top-of-book price
    bid_depth_10     dollar depth (Σ price·size) of bids within 10¢ of mid
    ask_depth_10     dollar depth (Σ price·size) of asks within 10¢ of mid
    depth_imbalance  top-of-book (bid_sz - ask_sz)/(bid_sz + ask_sz) ∈ [-1, 1]
    n_bid_levels     number of bid levels in the book
    n_ask_levels     number of ask levels in the book
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from .fetch import fetch_order_book, order_book_features
from .paths import REPO_ROOT

logger = logging.getLogger(__name__)

LEDGER_DIR = REPO_ROOT / "data" / "paper_trade"
SNAPSHOT_CSV = LEDGER_DIR / "book_snapshots.csv.gz"

# Environment override so tests / one-off runs can redirect the output file.
PATH_ENV_VAR = "BOOK_SNAPSHOT_PATH"

# Ledger column that holds the CLOB token id, in priority order
# (macro_positions.csv uses ``leader_token``; every other ledger uses ``token``).
_TOKEN_COLUMNS = ("token", "leader_token")

# Cap the per-run universe: 150 books ≈ 150 requests, well within an hourly cron.
MAX_TOKENS = 150

# Depth window: dollar depth within 10¢ of mid.
DEPTH_WINDOW = 0.10

# Soft size bound (compressed). ~8 MB gzip ≈ ~40 MB raw CSV for this schema;
# past that we *warn* about rotation — never delete accumulated history.
SIZE_WARN_BYTES = 8 * 1024 * 1024


# ---------------------------------------------------------------------------
# Target universe — open positions across every ledger
# ---------------------------------------------------------------------------

def collect_target_tokens(
    ledger_dir: Optional[Path] = None,
    max_tokens: int = MAX_TOKENS,
) -> list[str]:
    """Union of token ids from all OPEN positions across the paper-trade ledgers.

    Scans every ``*_positions.csv`` in ``data/paper_trade/`` defensively: a
    missing/empty/malformed ledger is skipped, the token column is whichever of
    ``token`` / ``leader_token`` exists, and rows are filtered to
    ``status == "open"`` when a status column is present. Order-preserving
    dedupe, capped at ``max_tokens``.
    """
    ledger_dir = Path(ledger_dir) if ledger_dir is not None else LEDGER_DIR
    tokens: list[str] = []
    seen: set[str] = set()

    for path in sorted(ledger_dir.glob("*_positions.csv")):
        try:
            df = pd.read_csv(path, dtype=str)
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as e:
            logger.warning("collect_target_tokens: skipping %s (%s)", path.name, e)
            continue
        col = next((c for c in _TOKEN_COLUMNS if c in df.columns), None)
        if col is None or df.empty:
            logger.debug("collect_target_tokens: no token column in %s", path.name)
            continue
        if "status" in df.columns:
            df = df[df["status"].astype(str).str.strip().str.lower() == "open"]
        for tok in df[col].dropna().astype(str).str.strip():
            if tok and tok.lower() != "nan" and tok not in seen:
                seen.add(tok)
                tokens.append(tok)

    if len(tokens) > max_tokens:
        logger.info("collect_target_tokens: capping %d open tokens at %d",
                    len(tokens), max_tokens)
        tokens = tokens[:max_tokens]
    logger.info("collect_target_tokens: %d target tokens", len(tokens))
    return tokens


# ---------------------------------------------------------------------------
# Snapshot — one row per token
# ---------------------------------------------------------------------------

def _levels(book: dict, side: str) -> list[tuple[float, float]]:
    """(price, size) tuples for one book side, tolerant of malformed levels.

    Mirrors the level parser inside ``fetch.order_book_features`` (which keeps
    it private) so depth aggregates use identical semantics.
    """
    out: list[tuple[float, float]] = []
    for lv in book.get(side, []) or []:
        try:
            out.append((float(lv["price"]), float(lv["size"])))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def snapshot_books(tokens: list[str], sleep: float = 0.1) -> pd.DataFrame:
    """Snapshot the current CLOB book of each token → one feature row per token.

    Defensive per token: an unreachable API, empty book, or one-sided book just
    drops that row — one bad book must not kill the run. Returns an empty
    DataFrame if nothing could be fetched.
    """
    ts = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%MZ")
    rows: list[dict] = []
    for i, token in enumerate(tokens):
        try:
            book = fetch_order_book(token)
            feats = order_book_features(book)
        except Exception as e:  # noqa: BLE001 — one bad book must not kill the run
            logger.warning("snapshot_books: %s… failed (%s: %s)",
                           str(token)[:16], type(e).__name__, e)
            continue
        if not feats:
            logger.debug("snapshot_books: %s… empty/one-sided book", str(token)[:16])
            continue

        bids, asks = _levels(book, "bids"), _levels(book, "asks")
        mid = feats["mid"]
        bid_depth = sum(p * s for p, s in bids if p >= mid - DEPTH_WINDOW)
        ask_depth = sum(p * s for p, s in asks if p <= mid + DEPTH_WINDOW)

        rows.append({
            "timestamp":       ts,
            "token_id":        str(token),
            "best_bid":        feats["best_bid"],
            "best_ask":        feats["best_ask"],
            "mid":             mid,
            "spread":          feats["spread"],
            "microprice":      feats["microprice"],
            "bid_depth_10":    round(bid_depth, 2),
            "ask_depth_10":    round(ask_depth, 2),
            "depth_imbalance": feats["depth_imbalance"],
            "n_bid_levels":    len(bids),
            "n_ask_levels":    len(asks),
        })
        if sleep and i < len(tokens) - 1:
            time.sleep(sleep)

    logger.info("snapshot_books: %d/%d books captured", len(rows), len(tokens))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Append — read + concat + atomic rewrite, deduped
# ---------------------------------------------------------------------------

def snapshot_path() -> Path:
    """Output file — ``BOOK_SNAPSHOT_PATH`` env override, else the default."""
    override = os.environ.get(PATH_ENV_VAR)
    return Path(override) if override else SNAPSHOT_CSV


def append_snapshot(df: pd.DataFrame, path: Optional[Path] = None) -> pd.DataFrame:
    """Append snapshot rows to the gzipped CSV, deduped on (timestamp, token_id).

    Read-existing + concat + atomic rewrite (write temp, then ``os.replace``) so
    a crash mid-write never corrupts the accumulated history. Returns the full
    combined DataFrame. Warns (does not delete) when the file grows past the
    soft rotation bound.
    """
    path = Path(path) if path is not None else snapshot_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        existing = pd.read_csv(path, dtype={"token_id": str}, compression="gzip")
        combined = pd.concat([existing, df], ignore_index=True)
    else:
        combined = df.copy()

    combined["token_id"] = combined["token_id"].astype(str)
    combined = (
        combined.drop_duplicates(subset=["timestamp", "token_id"], keep="last")
                .sort_values(["timestamp", "token_id"])
                .reset_index(drop=True)
    )

    tmp = path.with_name(path.name + ".tmp")
    combined.to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, path)

    size = path.stat().st_size
    if size > SIZE_WARN_BYTES:
        logger.warning(
            "append_snapshot: %s is %.1f MB compressed (>~%d MB ≈ 40 MB raw) — "
            "consider rotating it into a dated archive (e.g. "
            "book_snapshots_2026H1.csv.gz). History is never auto-deleted.",
            path, size / 1e6, SIZE_WARN_BYTES // (1024 * 1024),
        )
    logger.info("append_snapshot: +%d rows → %d total (%s, %.1f KB)",
                len(df), len(combined), path.name, size / 1024)
    return combined
