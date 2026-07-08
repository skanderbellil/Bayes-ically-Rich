"""Forward paper-trade tracker for the smart-money consensus signal.

The live, out-of-sample analog of the `ORDER_FLOW` / `PAYUP_FOLLOW` studies, and
the honest test of their one open question — *does the gross edge survive real
spread?* Each run it asks: which **currently-open** outcome tokens have several
**non-market-maker leaderboard wallets** recently buying them (consensus
breadth)? Those become new paper longs — and crucially each position is entered
at the **live CLOB ask**, not the mid, so the bid-ask spread the backtest could
only proxy is captured for real. Positions are held to resolution and marked
hourly.

It mirrors ``papertrade.py`` (macro buy-leader): one idempotent CSV ledger at
``data/paper_trade/smart_flow_positions.csv``, additive — new consensus tokens are
appended, resolved ones marked closed, nothing deleted — designed to run as an
hourly GitHub Actions cron.

State columns
-------------
token          : the outcome-token id (stable key)
condition_id   : parent market condition id
question       : market question text
domain         : coarse topic (politics/macro/sports/...)
entry_date     : date first logged
n_smart_buyers : distinct non-MM leaderboard wallets that bought it in the window
buyers         : pipe-separated, sorted list of the buyer wallet addresses at entry
                 (recorded once at entry, never updated -- lets later analyses
                 recompute independence-style features under different thresholds
                 without re-fetching trade history)
bet_fraction   : position size fraction actually used (after Kelly scaling)
entry_mid      : CLOB mid at entry
entry_ask      : CLOB best-ask at entry (what we actually "pay" -- captures spread)
spread         : entry_ask - entry_mid (the realised half-cost)
current_price  : latest mid (refreshed each run)
status         : open | won | lost | stopped | flipped
exit_date      : resolution date (blank while open)
outcome        : 1.0 won / 0.0 lost (blank while open)
pnl            : (outcome / entry_ask - 1) * fraction  (blank while open)
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from .categorize import market_category
from .fetch import (
    GAMMA_URL,
    _get,
    fetch_market_resolution,
    fetch_order_book,
    fetch_token_history_raw,
    order_book_features,
)
from .traders import fetch_leaderboard, fetch_trader_trades

logger = logging.getLogger(__name__)

STATE_FILE = Path(__file__).resolve().parents[2] / "data" / "paper_trade" / "smart_flow_positions.csv"
# Parallel forward ledger for the ROI-selected variant (see SELECTION_SWEEP.md):
# same consensus mechanics, but the smart pool is the top-N wallets by ROI rather
# than "all winners minus volume-leaders". Kept on its own file so the original
# out-of-sample record is never contaminated.
ROI_STATE_FILE = Path(__file__).resolve().parents[2] / "data" / "paper_trade" / "smart_flow_roi_positions.csv"


def kelly_fraction(
    n_smart_buyers: int,
    base_fraction: float = 0.10,
    ref_buyers: int = 4,
    cap_multiple: float = 2.0,
) -> float:
    """Scale bet fraction proportionally to smart-buyer conviction.

    3 buyers -> 7.5%, 4 -> 10%, 5 -> 12.5%, 6 -> 15%, capped at base * cap_multiple.
    """
    raw = base_fraction * (n_smart_buyers / ref_buyers)
    return round(min(raw, base_fraction * cap_multiple), 6)

_COLS = ["token", "condition_id", "question", "domain", "end_date", "entry_date",
         "n_smart_buyers", "buyers", "bet_fraction",
         "entry_mid", "entry_ask", "spread", "current_price",
         "status", "exit_date", "outcome", "pnl"]


# ---------------------------------------------------------------------------
# Smart, non-market-maker pool
# ---------------------------------------------------------------------------

def smart_pool(per_window: int = 60) -> list[str]:
    """Recent profit-leaderboard wallets minus the volume leaders (MM proxy).

    A light, cron-friendly screen: a wallet that ranks high on *profit* but also
    high on *volume* is most likely a market maker, so we drop the intersection.
    """
    winners: list[str] = []
    for w in ("7d", "30d"):
        df = fetch_leaderboard(w, "profit", per_window)
        if not df.empty:
            winners += df["wallet"].tolist()
    winners = list(dict.fromkeys(winners))
    vol = fetch_leaderboard("30d", "volume", per_window)
    mm = set(vol["wallet"]) if not vol.empty else set()
    pool = [w for w in winners if w not in mm]
    logger.info("smart_pool: %d winners, %d after dropping volume-leaders (MM proxy)",
                len(winners), len(pool))
    return pool


# ---------------------------------------------------------------------------
# Scan recent consensus buys on currently-open markets
# ---------------------------------------------------------------------------

def _fetch_pool_trades(pool: list[str], max_trades: int = 500) -> dict[str, pd.DataFrame]:
    """Live (uncached) recent trades for every pool wallet → {wallet: DataFrame}.

    Fetched once per run and reused for both ROI ranking and consensus indexing,
    so a wallet's history is never pulled twice.
    """
    out: dict[str, pd.DataFrame] = {}
    for w in pool:
        df = fetch_trader_trades(w, max_trades=max_trades, use_cache=False)
        if not df.empty:
            out[w] = df
    return out


def wallet_roi(trades: pd.DataFrame) -> float:
    """Efficiency (PnL per dollar traded) of one wallet, from its own fills only.

    ROI = (realised cash flow + net-open shares marked at the wallet's own last
    trade price per token) / gross dollars traded. Self-contained — needs no price
    panel or extra API calls — so it is cheap enough to rank the whole pool every
    run. This is the live analogue of the ``roi`` metric that topped the selection
    sweep (``SELECTION_SWEEP.md``).
    """
    if trades is None or trades.empty:
        return 0.0
    gross = float(trades["usdcSize"].sum())
    if gross <= 0:
        return 0.0
    is_buy = trades["side"] == "BUY"
    cash = float(trades.loc[~is_buy, "usdcSize"].sum() - trades.loc[is_buy, "usdcSize"].sum())
    signed = trades["size"].where(is_buy, -trades["size"])
    t = trades.assign(_sh=signed)
    marked = 0.0
    for _tok, g in t.groupby("asset"):
        net = float(g["_sh"].sum())
        if net > 0:                                  # only open longs carry value
            last_px = float(g.sort_values("timestamp")["price"].iloc[-1])
            marked += net * last_px
    return (cash + marked) / gross


def roi_top_wallets(trades_by_wallet: dict[str, pd.DataFrame], top_n: int) -> list[str]:
    """The ``top_n`` pool wallets by :func:`wallet_roi` (descending)."""
    scored = sorted(trades_by_wallet.items(), key=lambda kv: wallet_roi(kv[1]), reverse=True)
    keep = [w for w, _ in scored[:top_n]]
    logger.info("roi_top_wallets: kept top %d of %d wallets by ROI", len(keep), len(trades_by_wallet))
    return keep


def _flow_index_from(trades_by_wallet: dict[str, pd.DataFrame], window_days: int) -> dict[str, dict]:
    """Index pre-fetched trades by token with buyer/seller sets over the recent window.

    Returns {token: {"token", "condition_id", "question", "slug", "buyers": set, "sellers": set}}.
    """
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=window_days)
    index: dict[str, dict] = {}
    for w, df in trades_by_wallet.items():
        recent = df[df["timestamp"] >= cutoff]
        for r in recent.itertuples(index=False):
            tok = str(r.asset)
            if not tok:
                continue
            e = index.setdefault(tok, {
                "token": tok,
                "condition_id": getattr(r, "conditionId", ""),
                "question": getattr(r, "title", ""),
                "slug": getattr(r, "slug", ""),
                "buyers": set(),
                "sellers": set(),
            })
            side = getattr(r, "side", "")
            if side == "BUY":
                e["buyers"].add(w)
            elif side == "SELL":
                e["sellers"].add(w)
    logger.info("_flow_index_from: %d tokens seen across %d wallets", len(index), len(trades_by_wallet))
    return index


def _build_flow_index(pool: list[str], window_days: int) -> dict[str, dict]:
    """Fetch recent trades for all pool wallets and index them by token.

    Thin wrapper over :func:`_fetch_pool_trades` + :func:`_flow_index_from`, kept
    for the existing call sites and tests.
    """
    return _flow_index_from(_fetch_pool_trades(pool), window_days)


def scan_smart_flow_entries(
    window_days: int = 7,
    min_buyers: int = 3,
    per_window: int = 60,
    min_price: float = 0.05,
    max_price: float = 0.90,
    _flow_index: dict | None = None,
) -> list[dict]:
    """Open outcome tokens with >=``min_buyers`` distinct non-MM wallets buying recently.

    Pass a pre-built ``_flow_index`` (from ``_build_flow_index``) to avoid
    re-fetching trades when called from within ``update_ledger``.
    """
    if _flow_index is not None:
        flow = _flow_index
    else:
        pool = smart_pool(per_window)
        flow = _build_flow_index(pool, window_days)

    candidates = []
    for tok, e in flow.items():
        n = len(e["buyers"])
        if n < min_buyers:
            continue
        book = fetch_order_book(tok)
        feat = order_book_features(book)
        if not feat:                       # no live book -> resolved / untradeable
            continue
        mid, ask = feat["mid"], feat["best_ask"]
        if not (min_price <= mid <= max_price):
            continue
        # market end date (Gamma) — lets the dashboard score horizon-conditioned
        # views of this forward record (e.g. the pre-registered <=3d combo cell)
        end_date = ""
        m = _get(GAMMA_URL, {"clob_token_ids": tok})
        if isinstance(m, list) and m:
            end_date = str(m[0].get("endDate") or "")[:10]
        candidates.append({
            "token": tok,
            "condition_id": e["condition_id"],
            "question": e["question"],
            "domain": market_category(e["question"] or ""),
            "end_date": end_date,
            "n_smart_buyers": n,
            "buyers": "|".join(sorted(e["buyers"])),
            "entry_mid": round(mid, 4),
            "entry_ask": round(ask, 4),
            "spread": round(ask - mid, 4),
        })
    candidates.sort(key=lambda c: c["n_smart_buyers"], reverse=True)
    logger.info("scan_smart_flow_entries: %d tokens with >=%d smart buyers (open)",
                len(candidates), min_buyers)
    return candidates


def consensus_reversal_check(
    token: str,
    flow_index: dict,
    flip_threshold: int = 1,
) -> bool:
    """Return True if smart-wallet net flow for this token has turned bearish.

    Fires when distinct sellers >= distinct buyers + flip_threshold in the
    trailing window, meaning the smart-money crowd has more sellers than buyers.
    Returns False silently if the token was not seen in any wallet's trades
    (no recent activity either way is not a reversal).
    """
    e = flow_index.get(token)
    if e is None:
        return False
    n_buy  = len(e.get("buyers",  set()))
    n_sell = len(e.get("sellers", set()))
    fired  = n_sell >= n_buy + flip_threshold
    if fired:
        logger.info("CONSENSUS FLIP candidate: token=%s  buy=%d sell=%d threshold=%d",
                    token[:20], n_buy, n_sell, flip_threshold)
    return fired


# ---------------------------------------------------------------------------
# Ledger I/O
# ---------------------------------------------------------------------------

def load_ledger(state_file: Path = STATE_FILE) -> pd.DataFrame:
    if not state_file.exists():
        return pd.DataFrame(columns=_COLS)
    df = pd.read_csv(state_file, dtype=str)
    if "bet_fraction" not in df.columns:
        # Migrate legacy CSVs: assign flat-10% default so historical PnL is unchanged.
        df.insert(df.columns.get_loc("n_smart_buyers") + 1, "bet_fraction", "0.1")
    if "buyers" not in df.columns:
        # Migrate legacy CSVs: buyer identity was never recorded for these rows --
        # backfill empty rather than guessing (no data was ever fetched for them).
        df.insert(df.columns.get_loc("n_smart_buyers") + 1, "buyers", "")
    for col in _COLS:
        if col not in df.columns:
            df[col] = ""
    return df[_COLS]


def save_ledger(df: pd.DataFrame, state_file: Path = STATE_FILE) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    df[_COLS].to_csv(state_file, index=False)


def _price(token: str) -> tuple[float | None, float | None]:
    """Live (mid, best_ask) for a token. Falls back to CLOB history if no active book."""
    feat = order_book_features(fetch_order_book(token))
    if feat:
        return feat["mid"], feat["best_ask"]
    # No active order book — market likely resolved. Fall back to CLOB history.
    try:
        hist = fetch_token_history_raw(token, fidelity_minutes=60, use_cache=False)
        if not hist.empty:
            last = float(hist.iloc[-1])
            return last, last
    except Exception:
        pass
    return None, None


# ---------------------------------------------------------------------------
# Daily update
# ---------------------------------------------------------------------------

def update_ledger(
    bet_fraction: float = 0.10,
    min_buyers: int = 3,
    window_days: int = 7,
    use_kelly: bool = False,
    ref_buyers: int = 4,
    kelly_cap: float = 2.0,
    stop_loss: float | None = None,
    consensus_exit: bool = False,
    flip_threshold: int = 1,
    per_window: int = 60,
    selector: str = "winners_minus_mm",
    top_n: int = 10,
    state_file: Path = STATE_FILE,
) -> pd.DataFrame:
    """One hourly cycle: append new consensus longs, refresh prices, mark resolutions.

    Parameters
    ----------
    bet_fraction : float
        Base bankroll fraction per position. When use_kelly=True this is the
        reference fraction for a ref_buyers-conviction trade; higher-conviction
        trades are sized proportionally larger (capped at bet_fraction * kelly_cap).
    use_kelly : bool
        If True, scale each position's fraction by n_smart_buyers / ref_buyers.
    ref_buyers : int
        Buyer count that maps to exactly bet_fraction (Kelly reference point).
    kelly_cap : float
        Maximum Kelly multiple over bet_fraction (prevents runaway sizing).
    stop_loss : float | None
        Exit an open position when current_price <= entry_ask * (1 - stop_loss).
        E.g. 0.35 exits when price falls to 65% of entry. None disables stops.
    consensus_exit : bool
        If True, exit positions where smart-wallet net flow has flipped bearish
        (more distinct sellers than buyers in the trailing window).
    flip_threshold : int
        Sellers must exceed buyers by at least this many to trigger consensus exit.
    per_window : int
        Number of wallets to fetch from the leaderboard for the smart pool.
    selector : str
        How to choose the smart pool. "winners_minus_mm" (default) keeps the
        profit-leaderboard winners minus the volume-leaders (original behaviour);
        "roi_topn" instead keeps the top_n wallets by ROI (the live analogue of
        the selection-sweep winner — see SELECTION_SWEEP.md), run as a separate
        forward track on its own state_file.
    top_n : int
        Pool size when selector="roi_topn".
    state_file : Path
        Ledger CSV to read/write. Defaults to the original STATE_FILE; the ROI
        variant uses ROI_STATE_FILE so the two forward records never mix.
    """
    ledger = load_ledger(state_file)
    today = date.today().isoformat()
    held = set(ledger["token"].tolist()) if not ledger.empty else set()

    # -- 1. build shared flow index (entries + exits share the same API fetch) -
    pool            = smart_pool(per_window)
    trades_by_wallet = _fetch_pool_trades(pool)
    if selector == "roi_topn":
        keep = set(roi_top_wallets(trades_by_wallet, top_n))
        trades_by_wallet = {w: df for w, df in trades_by_wallet.items() if w in keep}
    flow_index = _flow_index_from(trades_by_wallet, window_days)

    # -- 2. new consensus entries --
    new_rows = []
    for c in scan_smart_flow_entries(window_days=window_days, min_buyers=min_buyers,
                                     _flow_index=flow_index):
        if c["token"] in held:
            continue
        n = c["n_smart_buyers"]
        frac = kelly_fraction(n, bet_fraction, ref_buyers, kelly_cap) if use_kelly else bet_fraction
        logger.info("NEW long: %s  (%d buyers)  mid %.3f ask %.3f  frac=%.3f",
                    (c["question"] or "")[:40], n, c["entry_mid"], c["entry_ask"], frac)
        new_rows.append({
            "token": c["token"], "condition_id": c["condition_id"],
            "question": c["question"], "domain": c["domain"],
            "end_date": c.get("end_date", ""), "entry_date": today,
            "n_smart_buyers": str(n), "buyers": c.get("buyers", ""),
            "bet_fraction": str(frac),
            "entry_mid": str(c["entry_mid"]), "entry_ask": str(c["entry_ask"]),
            "spread": str(c["spread"]), "current_price": str(c["entry_mid"]),
            "status": "open", "exit_date": "", "outcome": "", "pnl": "",
        })
    if new_rows:
        ledger = pd.concat([ledger, pd.DataFrame(new_rows)], ignore_index=True)

    # -- 3. refresh + resolve + stop-loss + consensus-exit open positions --
    for i, row in ledger[ledger["status"] == "open"].iterrows():
        entry_ask = float(row["entry_ask"])
        frac = float(row["bet_fraction"]) if row.get("bet_fraction") else bet_fraction

        # resolution first, from the AUTHORITATIVE CLOB market status (closed +
        # per-token winner). This catches settled markets whose order book is gone
        # and whose fallback last-trade price never reaches the 0/1 threshold —
        # the cause of resolved positions getting stuck "open".
        res = fetch_market_resolution(row.get("condition_id", ""))
        market_closed = bool(res and res["closed"])
        outcome = None
        if market_closed:
            toks = res["tokens"]
            finalised = any(t["winner"] or t["price"] >= 0.99 for t in toks.values())
            t = toks.get(str(row["token"]))
            if finalised and t is not None:
                outcome = 1.0 if (t["winner"] or t["price"] >= 0.99) else 0.0

        mid, _ = _price(row["token"])
        if mid is not None:
            ledger.at[i, "current_price"] = str(round(mid, 4))
        if outcome is None and mid is None:
            continue  # neither resolution nor a fresh mark — leave untouched this run

        # price-heuristic fallback only when the API confirms the market is
        # closed but couldn't produce a clean per-token label — never resolve
        # on price alone while the market is still reported open
        if outcome is None and market_closed and mid is not None:
            outcome = 1.0 if mid >= 0.99 else (0.0 if mid <= 0.01 else None)
        if outcome is not None:
            pnl = (outcome / entry_ask - 1.0) * frac
            ledger.at[i, "status"]        = "won" if outcome == 1.0 else "lost"
            ledger.at[i, "exit_date"]     = today
            ledger.at[i, "outcome"]       = str(outcome)
            ledger.at[i, "pnl"]          = str(round(pnl, 4))
            ledger.at[i, "current_price"] = str(outcome)
            logger.info("RESOLVED %s  %s  pnl=%.4f",
                        (row["question"] or "")[:40], ledger.at[i, "status"], pnl)
            continue

        # stop-loss
        if stop_loss is not None:
            stop_floor = entry_ask * (1.0 - stop_loss)
            if mid <= stop_floor:
                pnl = (mid / entry_ask - 1.0) * frac
                ledger.at[i, "status"]        = "stopped"
                ledger.at[i, "exit_date"]     = today
                ledger.at[i, "outcome"]       = ""
                ledger.at[i, "pnl"]           = str(round(pnl, 4))
                logger.info("STOPPED %s  mid=%.4f floor=%.4f pnl=%.4f",
                            (row["question"] or "")[:40], mid, stop_floor, pnl)
                continue

        # consensus exit: signal reversal
        if consensus_exit:
            if consensus_reversal_check(row["token"], flow_index, flip_threshold):
                pnl = (mid / entry_ask - 1.0) * frac
                ledger.at[i, "status"]    = "flipped"
                ledger.at[i, "exit_date"] = today
                ledger.at[i, "outcome"]   = ""
                ledger.at[i, "pnl"]      = str(round(pnl, 4))
                logger.info("FLIPPED  %s  mid=%.4f  pnl=%.4f",
                            (row["question"] or "")[:40], mid, pnl)
                continue

    save_ledger(ledger, state_file)
    return ledger
