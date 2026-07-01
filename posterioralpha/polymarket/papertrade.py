"""Forward paper-trade tracker for the macro buy-leader strategy.

Maintains a CSV ledger at ``data/paper_trade/macro_positions.csv``. Each row
is one macro multi-outcome field: the identified leader, its entry price when
first observed, current price, and resolution outcome once settled.

The tracker is designed to run as a **daily cron** (GitHub Actions) — it is
purely additive/idempotent: new open fields are appended, resolved fields are
marked closed, nothing is ever deleted.

State columns
-------------
event_ticker   : Polymarket event ticker (stable key)
event_title    : human-readable title
leader_market  : market id of the field leader (highest mean price when entered)
leader_question: the question text
entry_date     : date the position was first logged (YYYY-MM-DD)
entry_price    : implied probability of the leader at entry
current_price  : latest mid-price (updated each daily run)
status         : open | won | lost
exit_date      : date of resolution (blank while open)
outcome        : 1.0 won / 0.0 lost / blank while open
pnl            : (outcome / (entry_price + SLIP) - 1) * fraction, blank while open
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd

from .categorize import market_category
from .fetch import GAMMA_URL, _get, _yes_index

logger = logging.getLogger(__name__)

SLIP = 0.01
STATE_FILE = Path(__file__).resolve().parents[2] / "data" / "paper_trade" / "macro_positions.csv"
CLOB_PRICE_URL = "https://clob.polymarket.com/last-trade-price"


# ---------------------------------------------------------------------------
# Fetch live open macro multi-outcome events
# ---------------------------------------------------------------------------

def _current_price(token_id: str) -> float | None:
    """Latest CLOB mid-price for a token (live)."""
    data = _get(CLOB_PRICE_URL, {"token_id": token_id})
    if data and "price" in data:
        try:
            return float(data["price"])
        except (ValueError, TypeError):
            pass
    return None


def scan_open_macro_events(n: int = 500, min_volume: float = 5_000,
                            min_candidates: int = 3) -> list[dict]:
    """Scan live Polymarket for open macro multi-outcome events.

    Returns a list of dicts, one per qualifying event, with the identified
    leader's market id, question, and current price.
    """
    found: dict[str, dict] = {}      # event_ticker → best candidate set
    offset = 0
    page_size = 100
    fetched = 0
    while fetched < n:
        params = {"limit": page_size, "offset": offset, "order": "volumeNum",
                  "ascending": "false", "closed": "false", "active": "true"}
        batch = _get(GAMMA_URL, params) or []
        if not batch:
            break
        for m in batch:
            try:
                vol = float(m.get("volumeNum") or 0)
            except (TypeError, ValueError):
                vol = 0.0
            if vol < min_volume:
                continue
            q = m.get("question", "") or ""
            evt = (m.get("events") or [{}])[0]
            ticker = evt.get("ticker", "")
            title  = evt.get("title", "")
            cat = market_category(q, ticker, title)
            if cat != "macro":
                continue
            # group by event_ticker; collect all candidates
            yi = _yes_index(m)
            if yi is None:
                continue
            ids = m.get("clobTokenIds") or []
            prices = m.get("outcomePrices") or []
            try:
                ids = [i.strip().strip('"') for i in str(ids).strip("[]").split(",")]
                prices_raw = [p.strip().strip('"') for p in str(prices).strip("[]").split(",")]
                yes_price = float(prices_raw[yi])
                token_id = ids[yi]
            except Exception:
                continue
            if not (0.05 <= yes_price <= 0.99):
                continue
            ev = found.setdefault(ticker, {"event_ticker": ticker, "event_title": title,
                                           "candidates": []})
            ev["candidates"].append({
                "market_id": m.get("id", ""),
                "question": q,
                "token_id": token_id,
                "price": yes_price,
                "volume": vol,
            })
        fetched += len(batch)
        offset += page_size
        if len(batch) < page_size:
            break

    results = []
    for ev in found.values():
        cands = ev["candidates"]
        if len(cands) < min_candidates:
            continue
        total = sum(c["price"] for c in cands)
        if not (0.80 <= total <= 1.20):     # must sum to ~1 (neg-risk enforced)
            continue
        leader = max(cands, key=lambda c: c["price"])
        results.append({
            "event_ticker": ev["event_ticker"],
            "event_title":  ev["event_title"],
            "leader_market":   leader["market_id"],
            "leader_question": leader["question"],
            "leader_token":    leader["token_id"],
            "entry_price":     leader["price"],
            "n_candidates":    len(cands),
        })
    return results


# ---------------------------------------------------------------------------
# Resolution detection (authoritative Gamma status, price heuristic fallback)
# ---------------------------------------------------------------------------

def _gamma_market(market_id: str) -> dict | None:
    """Fetch one market record from Gamma by market id (dict, or None on failure)."""
    if not market_id:
        return None
    data = _get(GAMMA_URL, {"id": market_id})
    if isinstance(data, list):          # Gamma returns a one-element list for ?id=
        data = data[0] if data else None
    return data if isinstance(data, dict) else None


def _resolve_via_gamma(market_id: str, market: dict | None = None) -> float | None:
    """Authoritative resolution outcome from the Gamma markets API.

    A market is resolved only when Gamma reports ``closed`` *and*
    ``umaResolutionStatus == "resolved"``; the settled ``outcomePrices`` then pin
    the Yes leg at ~1.0 (won) or ~0.0 (lost). Returns 1.0 / 0.0, or None while
    the market is open, its resolution is pending, or the fields are unusable.

    ``market`` may be a pre-fetched Gamma record (to avoid a duplicate request);
    otherwise the market is fetched by id.
    """
    m = market if market is not None else _gamma_market(market_id)
    if m is None:
        return None
    if not m.get("closed"):
        return None
    if str(m.get("umaResolutionStatus") or "").lower() != "resolved":
        return None
    yi = _yes_index(m.get("outcomes", ""))
    prices = m.get("outcomePrices") or ""
    try:
        prices_raw = [p.strip().strip('"') for p in str(prices).strip("[]").split(",")]
        yes_price = float(prices_raw[yi])
    except (ValueError, IndexError):
        return None
    return 1.0 if yes_price >= 0.5 else 0.0


def _resolve_status(token_id: str) -> float | None:
    """Price-heuristic fallback: CLOB last-trade ≥0.99 → won, ≤0.01 → lost.

    Only meaningful once Gamma already reports the market **closed** — callers
    must never resolve on price alone while the market is open (a 0.99 print is
    not a settlement).
    """
    p = _current_price(token_id)
    if p is None:
        return None
    if p >= 0.99:
        return 1.0
    if p <= 0.01:
        return 0.0
    return None   # too ambiguous even though closed


# ---------------------------------------------------------------------------
# Ledger read / write
# ---------------------------------------------------------------------------

_COLS = ["event_ticker", "event_title", "leader_market", "leader_question",
         "leader_token", "entry_date", "entry_price", "current_price",
         "status", "exit_date", "outcome", "pnl"]


def load_ledger() -> pd.DataFrame:
    if STATE_FILE.exists():
        return pd.read_csv(STATE_FILE, dtype=str)
    return pd.DataFrame(columns=_COLS)


def save_ledger(df: pd.DataFrame) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    df[_COLS].to_csv(STATE_FILE, index=False)


# ---------------------------------------------------------------------------
# Main update routine
# ---------------------------------------------------------------------------

def update_ledger(bet_fraction: float = 0.10) -> pd.DataFrame:
    """Run one daily update cycle: find new positions, refresh prices, mark resolutions.

    Returns the updated ledger DataFrame.
    """
    ledger = load_ledger()
    today = date.today().isoformat()

    # ── 1. find new open macro multi-outcome events ───────────────────────
    open_events = scan_open_macro_events()
    existing_tickers = set(ledger["event_ticker"].tolist()) if not ledger.empty else set()
    new_rows = []
    for ev in open_events:
        if ev["event_ticker"] in existing_tickers:
            continue
        logger.info("NEW position: %s  leader %s  price %.3f",
                    ev["event_ticker"], ev["leader_question"][:40], ev["entry_price"])
        new_rows.append({
            "event_ticker":    ev["event_ticker"],
            "event_title":     ev["event_title"],
            "leader_market":   ev["leader_market"],
            "leader_question": ev["leader_question"],
            "leader_token":    ev["leader_token"],
            "entry_date":      today,
            "entry_price":     str(round(ev["entry_price"], 4)),
            "current_price":   str(round(ev["entry_price"], 4)),
            "status":          "open",
            "exit_date":       "",
            "outcome":         "",
            "pnl":             "",
        })
    if new_rows:
        ledger = pd.concat([ledger, pd.DataFrame(new_rows)], ignore_index=True)

    # ── 2. refresh prices + mark resolutions for open positions ───────────
    for i, row in ledger[ledger["status"] == "open"].iterrows():
        token = row.get("leader_token", "")
        if not token:
            continue
        p = _current_price(token)
        if p is not None:
            ledger.at[i, "current_price"] = str(round(p, 4))

        # resolution: authoritative Gamma status first; price heuristic only
        # as a fallback when Gamma says closed but outcomePrices are unusable
        market = _gamma_market(row.get("leader_market", ""))
        res = _resolve_via_gamma(row.get("leader_market", ""), market=market)
        if res is None and market is not None and market.get("closed"):
            res = _resolve_status(token)
        if res is not None:
            entry_p = float(row["entry_price"])
            pnl_val = (res / (entry_p + SLIP) - 1) * bet_fraction
            ledger.at[i, "status"]       = "won" if res == 1.0 else "lost"
            ledger.at[i, "exit_date"]    = today
            ledger.at[i, "outcome"]      = str(res)
            ledger.at[i, "pnl"]          = str(round(pnl_val, 4))
            ledger.at[i, "current_price"] = str(res)
            logger.info("RESOLVED %s  status=%s  pnl=%.4f",
                        row["event_ticker"], ledger.at[i, "status"], pnl_val)

    save_ledger(ledger)
    return ledger
