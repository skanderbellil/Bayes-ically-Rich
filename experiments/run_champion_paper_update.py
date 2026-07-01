#!/usr/bin/env python3
"""
Champion stack — forward, out-of-sample equity paper-trade ledger
====================================================================

The Polymarket strategies already get a survivorship-free forward record
(``data/paper_trade/*_positions.csv``, refreshed hourly by
``.github/workflows/paper_trade.yml``). The flagship equity strategy — the
"champion stack" frozen in ``experiments/run_champion_stack.py`` — didn't
have one; a backtest can always be re-run against restated data or a change
of mind about a warm-up window, a forward ledger can't. This script closes
that gap.

Each run downloads fresh daily prices, replicates the frozen spec exactly
(``posterioralpha/research/champion_live.py`` — read that module's
docstring for the exact formulas), and appends **one row per calendar date**
to ``data/paper_trade/champion_positions.csv``. Re-running the same day
overwrites that date's row (never duplicates); ``strategy_return`` and
``equity`` are recomputed from the full history every run, so a correction
anywhere in the ledger propagates forward correctly.

A missed day (yfinance down, rate-limited, FRED unreachable, etc.) is fine —
we print a clear message and exit 0 so the CI workflow doesn't go red for a
transient network hiccup. A real bug in the pipeline should NOT be
swallowed; those exceptions propagate and fail the run loudly.

Usage:
  python experiments/run_champion_paper_update.py
"""
from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from posterioralpha.research.champion_live import compute_current_weights

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-5s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

STATE_FILE = _bootstrap.ROOT / "data" / "paper_trade" / "champion_positions.csv"
COLS = ["date", "liq", "vix_ts", "credit", "dollar", "tug", "vote", "flip_f",
        "vote_eff", "exposure", "w_qld", "w_uup", "qld_close", "uup_close",
        "liq_stale_days", "strategy_return", "equity"]


def _is_network_error(exc: Exception) -> bool:
    """
    Heuristic: is this a transient data-fetch failure (network down, rate
    limited, FRED/yfinance unreachable) vs. a real bug in the pipeline?
    Only the former should be swallowed with a clean exit 0.
    """
    try:
        import requests
        if isinstance(exc, requests.exceptions.RequestException):
            return True
    except ImportError:
        pass
    name = type(exc).__name__
    mod = type(exc).__module__ or ""
    if "yfinance" in mod or "urllib3" in mod or "requests" in mod or "curl_cffi" in mod:
        return True
    return name in {
        "ConnectionError", "Timeout", "SSLError", "HTTPError", "URLError",
        "RemoteDisconnected", "YFRateLimitError", "YFPricesMissingError",
        "YFTzMissingError", "JSONDecodeError",
    }


def load_ledger() -> pd.DataFrame:
    if not STATE_FILE.exists():
        return pd.DataFrame(columns=COLS)
    df = pd.read_csv(STATE_FILE, dtype=str)
    for c in COLS:
        if c not in df.columns:
            df[c] = ""
    return df[COLS]


def _recompute_returns_and_equity(df: pd.DataFrame) -> pd.DataFrame:
    """
    strategy_return[t] = yesterday's weights applied to today's QLD/UUP
    close-to-close returns (blank for the first row); equity = cumulative
    product of (1+strategy_return), recomputed from the full column so any
    correction to a historical row propagates through.
    """
    df = df.sort_values("date").reset_index(drop=True)
    qld = pd.to_numeric(df["qld_close"], errors="coerce")
    uup = pd.to_numeric(df["uup_close"], errors="coerce")
    w_qld_prev = pd.to_numeric(df["w_qld"], errors="coerce").shift(1)
    w_uup_prev = pd.to_numeric(df["w_uup"], errors="coerce").shift(1)

    strat_ret = w_qld_prev * qld.pct_change() + w_uup_prev * uup.pct_change()
    df["strategy_return"] = strat_ret
    df["equity"] = (1.0 + strat_ret.fillna(0.0)).cumprod()
    return df


def main() -> None:
    try:
        out = compute_current_weights()
    except Exception as exc:
        if _is_network_error(exc):
            print(f"Champion paper-trade update: data fetch failed "
                  f"({type(exc).__name__}: {exc}) — a missed day is fine, "
                  f"skipping cleanly.")
            sys.exit(0)
        raise

    row = {
        "date": out["date"],
        "liq": out["liq"], "vix_ts": out["vix_ts"], "credit": out["credit"],
        "dollar": out["dollar"], "tug": out["tug"], "vote": out["vote"],
        "flip_f": out["flip_f"], "vote_eff": out["vote_eff"],
        "exposure": out["exposure"], "w_qld": out["w_qld"], "w_uup": out["w_uup"],
        "qld_close": out["qld_close"], "uup_close": out["uup_close"],
        "liq_stale_days": out["liq_stale_days"],
    }

    ledger = load_ledger()
    ledger = ledger[ledger["date"] != row["date"]]           # idempotent overwrite
    ledger = pd.concat([ledger, pd.DataFrame([row])], ignore_index=True)
    ledger = ledger.drop_duplicates(subset="date", keep="last")

    ledger = _recompute_returns_and_equity(ledger)

    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    ledger[COLS].to_csv(STATE_FILE, index=False)

    last = ledger.iloc[-1]
    stale_note = f"  [liq stale {out['liq_stale_days']}d]" if out["liq_stale"] else ""
    logger.info("ledger saved: %s  (%d rows, latest %s)", STATE_FILE, len(ledger), row["date"])
    print(f"\n{row['date']}  vote_eff={float(row['vote_eff']):+.3f}  "
          f"exposure={float(row['exposure']):.3f}  w_qld={float(row['w_qld']):.3f}  "
          f"w_uup={float(row['w_uup']):.3f}  equity={float(last['equity']):.4f}{stale_note}")


if __name__ == "__main__":
    main()
