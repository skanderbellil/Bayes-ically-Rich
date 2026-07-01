#!/usr/bin/env python3
"""
Regime-calm disruptive-tail — forward paper-trade ledger
========================================================

Out-of-sample, real-time test of REGIME_CALIBRATION.md: low-priced YES contracts
on *disruptive*-domain markets (geopolitics, crypto/financial, politics, AI) are
underpriced **during calm regimes**. This logs a paper YES long (at the live ask)
in every currently-open disruptive market with YES mid <= --max-price resolving in
[--days-min, --days-max] days, **records the regime state at entry**, marks open
positions, and resolves them — never exits (hold to settlement).

Regime gate: market-wide risk (VIX). "calm" = trailing-45d max VIX below
--vix-thresh; "turbulent" otherwise. VIX is fetched live (yfinance, works in CI)
with a fallback to the committed datasets/vix_termstructure.csv. Every candidate
is logged with its regime tag and an ``in_strategy`` flag (True only in calm), so
the forward record builds the calm-vs-turbulent comparison itself — the gate is
applied at analysis time, not by silently dropping entries.

Usage:
  python experiments/run_regime_paper_update.py
  python experiments/run_regime_paper_update.py --dry-run
"""
from __future__ import annotations
import argparse, logging, re
from datetime import date
from pathlib import Path

import numpy as np, pandas as pd
import _bootstrap  # noqa
from posterioralpha.polymarket.fetch import (
    fetch_markets, fetch_order_book, order_book_features, token_resolution,
    fetch_token_history_raw)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-5s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

STATE_FILE = Path("data/paper_trade/regime_positions.csv")
COLS = ["token", "condition_id", "question", "domain", "end_date", "entry_date",
        "days_to_res", "vix_max45", "regime", "in_strategy", "entry_mid", "entry_ask",
        "spread", "current_price", "status", "exit_date", "outcome", "pnl"]

_KW = {
    "geopolitics": ["strike", "war", "warship", "invade", "invasion", "attack", "nuclear",
                    "missile", "ceasefire", "troops", "military", "iran", "russia", "ukraine",
                    "israel", "gaza", "taiwan", "north korea", "coup", "annex", "sanction", "hostage"],
    "crypto/financial": ["bitcoin", "ethereum", "solana", "crypto", "btc", "eth", "price of",
                         "reach $", "hit $", "all-time high", "recession", "s&p", "nasdaq",
                         "fed", "rate cut", "rate hike", "cpi", "inflation", "gdp", "default"],
    "ai/tech": ["openai", "gpt", "agi", "anthropic", "claude", "gemini", "llm", "ai model",
                "artificial intelligence", "deepmind", "grok", "nvidia"],
    "politics": ["election", "president", "senate", "governor", "vote", "poll", "trump", "biden",
                 "kamala", "nominee", "impeach", "resign", "primary", "parliament", "prime minister"],
}
DISRUPTIVE = set(_KW)


def classify(q: str) -> str | None:
    """Domain by whole-word keyword match (word boundaries avoid 'eth' ⊂ 'Netherlands')."""
    ql = (q or "").lower()
    for dom, kws in _KW.items():
        if any(re.search(r"\b" + re.escape(k) + r"\b", ql) for k in kws):
            return dom
    return None


def vix_trailing_max(window: int = 45):
    """(vix_max45, source). Live via yfinance (CI), else local dataset fallback."""
    s = None
    try:
        import warnings; warnings.filterwarnings("ignore")
        import yfinance as yf
        d = yf.download("^VIX", period="4mo", interval="1d", progress=False)
        if not d.empty:
            s = d["Close"].dropna(); src = "yfinance"
    except Exception as e:
        logger.info("yfinance VIX unavailable (%s); using local dataset", type(e).__name__)
    if s is None or len(s) == 0:
        df = pd.read_csv("datasets/vix_termstructure.csv", parse_dates=["Date"]).dropna(subset=["VIX"])
        s = df.set_index("Date")["VIX"]; src = f"local (≤{s.index.max().date()})"
    return float(np.asarray(s).astype(float)[-window:].max()), src


def days_to_res(end_date):
    if not end_date:
        return None
    try:
        return (pd.to_datetime(end_date, utc=True) - pd.Timestamp.now(tz="UTC")).total_seconds()/86400.0
    except (ValueError, TypeError):
        return None


def load_ledger():
    if not STATE_FILE.exists():
        return pd.DataFrame(columns=COLS)
    df = pd.read_csv(STATE_FILE, dtype=str)
    for c in COLS:
        if c not in df.columns:
            df[c] = ""
    return df[COLS]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-price", type=float, default=0.35)
    ap.add_argument("--days-min", type=int, default=2)
    ap.add_argument("--days-max", type=int, default=45)
    ap.add_argument("--min-volume", type=float, default=5_000.0)
    ap.add_argument("--vix-thresh", type=float, default=23.0, help="calm if trailing-45d max VIX below this")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    vmax, vsrc = vix_trailing_max()
    regime = "calm" if vmax <= args.vix_thresh else "turbulent"
    logger.info("regime: VIX max45 = %.1f (%s) → %s  [calm if <= %.1f]", vmax, vsrc, regime.upper(), args.vix_thresh)

    try:
        markets = fetch_markets(n_markets=1200, min_volume=args.min_volume, min_liquidity=300.0, closed=False)
    except Exception as e:
        logger.error("open-market fetch failed (%s); marking only", type(e).__name__); markets = pd.DataFrame()

    ledger = load_ledger()
    held = set(ledger["token"]) if not ledger.empty else set()
    today = date.today().isoformat()

    new = []
    for m in markets.itertuples(index=False):
        dom = classify(getattr(m, "question", ""))
        if dom is None or dom not in DISRUPTIVE:
            continue
        dtr = days_to_res(getattr(m, "end_date", None))
        if dtr is None or not (args.days_min <= dtr <= args.days_max):
            continue
        if m.yes_token in held:
            continue
        feat = order_book_features(fetch_order_book(m.yes_token))
        if not feat:
            continue
        mid, ask = feat["mid"], feat["best_ask"]
        if not (0.02 <= mid <= args.max_price):
            continue
        logger.info("ENTRY [%s/%s] %s  mid %.3f ask %.3f  %.1fd", dom, regime, (m.question or "")[:40], mid, ask, dtr)
        new.append({"token": m.yes_token, "condition_id": getattr(m, "condition_id", ""),
                    "question": m.question, "domain": dom, "end_date": getattr(m, "end_date", ""),
                    "entry_date": today, "days_to_res": round(dtr, 2), "vix_max45": round(vmax, 1),
                    "regime": regime, "in_strategy": regime == "calm", "entry_mid": round(mid, 4),
                    "entry_ask": round(ask, 4), "spread": round(ask-mid, 4), "current_price": round(mid, 4),
                    "status": "open", "exit_date": "", "outcome": "", "pnl": ""})
    logger.info("scanned %d open markets → %d new disruptive cheap-YES entries", len(markets), len(new))
    if new:
        ledger = pd.concat([ledger, pd.DataFrame(new)], ignore_index=True)

    # mark + resolve open positions (hold to settlement, no exits)
    resolved = 0
    for i, row in ledger[ledger["status"] == "open"].iterrows():
        try:
            outcome = token_resolution(row.get("condition_id", ""), row["token"])
            feat = order_book_features(fetch_order_book(row["token"]))
            mid = feat["mid"] if feat else None
            if mid is None:
                h = fetch_token_history_raw(row["token"], fidelity_minutes=60, use_cache=False)
                mid = float(h.iloc[-1]) if h is not None and not h.empty else None
        except Exception:
            outcome, mid = None, None
        if mid is not None:
            # ledger is read with dtype=str — pandas 3 str columns reject floats
            ledger.at[i, "current_price"] = str(round(mid, 4))
        if outcome is None and mid is not None:
            outcome = 1.0 if mid >= 0.99 else (0.0 if mid <= 0.01 else None)
        if outcome is not None:
            entry = float(row["entry_ask"])
            ledger.at[i, "status"] = "won" if outcome == 1.0 else "lost"
            ledger.at[i, "exit_date"] = today
            ledger.at[i, "outcome"] = str(outcome)
            ledger.at[i, "pnl"] = str(round(outcome/entry - 1.0, 4))
            resolved += 1

    # quick OOS readout: calm vs turbulent entries, settled only
    done = ledger[ledger["status"].isin(["won", "lost"])].copy()
    if not done.empty:
        done["pnl"] = pd.to_numeric(done["pnl"], errors="coerce")
        print("\n  OOS ledger so far (settled %d / total %d):" % (len(done), len(ledger)))
        for r, sub in done.groupby("regime"):
            print("    %-9s n=%2d  win=%.2f  mean_pnl/$1=%+.3f" % (r, len(sub), (sub.pnl > 0).mean(), sub.pnl.mean()))

    if args.dry_run:
        logger.info("dry-run: %d would-be new, %d resolved — NOT writing", len(new), resolved)
        return
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    ledger[COLS].to_csv(STATE_FILE, index=False)
    logger.info("ledger saved: %s  (%d open, %d settled)", STATE_FILE,
                (ledger.status == "open").sum(), (ledger.status != "open").sum())


if __name__ == "__main__":
    main()
