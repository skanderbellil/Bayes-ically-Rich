#!/usr/bin/env python3
"""
Validated regime strategy — forward paper-trade ledger
======================================================

The deployable distillation of REGIME_CALIBRATION.md. run_regime_domain_validation.py
puts every disruptive domain through a strict kill battery against its *own*
domain-matched exogenous proxy; only survivors trade here. As of the latest
validation only **geopolitics** survives (GPR-calm), so this ledger is, in effect,
"buy the cheap YES leg of a geopolitical market while geopolitical risk is calm,
hold to settlement" — the one edge that withstood the hostile audit.

What it does each run:
  * reads validated_domains.json (which domains passed) + their matched proxy,
  * computes each validated domain's CURRENT regime from the committed proxy
    dataset (causal trailing-window vs trailing-1yr median; GPR moves slowly so a
    lagged committed file is fine),
  * logs a paper YES long (live ask) in every open market in a validated domain
    with YES mid <= --max-price resolving in [--days-min,--days-max] days, **only
    when that domain is currently calm** (the gate actually fires — unlike the
    exploratory all-domain ledger which logs both regimes for comparison),
  * SIZES each bet by the DEPTH of the calm (--size-mode signal, the default):
    bet_fraction = base * exp(g*z)/exp(g^2/2), capped, where z is the causal
    GPR-calm depth z-score. This deploys the sizing study's conclusion
    (run_regime_sizing_sensitivity.py): size by the SIGNAL, not the price —
    calm-depth predicts edge (corr +0.32) so deeper calm bets bigger, while
    price does not (corr +0.04) so sizing stays FLAT across price. Price risk is
    instead controlled by the --max-price cap, which the breakeven model
    (run_regime_breakeven_model.py) shows keeps the breakeven win rate <= ~0.38.
    --size-mode flat restores the old constant fraction.
  * marks + resolves to settlement (hold, no exits).

Ledger: data/paper_trade/validated_regime_positions.csv (dashboard-compatible).

Usage:
  python experiments/run_validated_regime_paper_update.py
  python experiments/run_validated_regime_paper_update.py --dry-run
"""
from __future__ import annotations
import argparse, json, logging, math
from datetime import date
from pathlib import Path

import numpy as np, pandas as pd
import _bootstrap  # noqa
from posterioralpha.polymarket.fetch import (
    fetch_markets, fetch_order_book, order_book_features, token_resolution,
    fetch_token_history_raw)
# reuse the exact domain classifier the validation/panel use
from run_regime_paper_update import classify

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-5s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

VALID_FILE = Path("data/polymarket_calibration_snapshot/validated_domains.json")
STATE_FILE = Path("data/paper_trade/validated_regime_positions.csv")
COLS = ["token", "condition_id", "question", "domain", "proxy", "proxy_value", "regime",
        "end_date", "entry_date", "days_to_res", "entry_ask", "current_price",
        "bet_fraction", "status", "exit_date", "outcome", "pnl"]

# matched proxy daily series builders (higher = more turbulent), causal trailing regime
_WIN, _REF = 45, 365


def _series(domain):
    if domain == "geopolitics":
        s = pd.read_csv("datasets/gpr_daily.csv.gz", parse_dates=["date"]).set_index("date")["GPRD"]
        return s.sort_index(), "GPR index", 45
    if domain == "crypto/financial":
        d = pd.read_csv("datasets/btc_usd_daily.csv", parse_dates=["date"]).set_index("date")["close"].sort_index()
        return (np.log(d).diff().rolling(30, min_periods=15).std() * math.sqrt(365)), "BTC 30d realized vol", 30
    if domain == "politics":
        s = pd.read_csv("datasets/epu_daily.csv", parse_dates=["date"]).set_index("date")["EPU"]
        return s.sort_index(), "EPU (daily)", 45
    return None, None, None


def regime_now(domain):
    """Current CALM/TURBULENT for a domain from committed proxy data (causal).

    Gate = level-OR-vol (the validated upgrade, see run_regime_gpr_vol.py): calm if
    the proxy's trailing-`win` mean is below its trailing-1yr median (level-calm) OR
    its trailing-`win` daily-change std is below its trailing-1yr median (vol-calm).
    The union ~doubles the deployable book while keeping the sharpest calm split.

    Also returns ``strength`` = how DEEP the calm is, as a causal z-score:
    (trailing-1yr-median - trailing-`win` mean) / trailing-1yr std of that mean.
    Positive = calmer than the recent normal; bigger = deeper calm. This is the
    SAME continuous signal whose cross-sectional tilt earned the size-weighted
    edge in run_regime_sizing_sensitivity.py (corr(edge, calm-depth)=+0.32) — the
    live ledger turns that cross-sectional tilt into a temporal one (size up when
    the calm is deep, fully causal)."""
    s, name, win = _series(domain)
    if s is None:
        return None, name, None, None
    s = s.asfreq("D").ffill().dropna()
    lvl = s.rolling(win, min_periods=max(5, win // 3)).mean()
    lvl_med = lvl.rolling(_REF, min_periods=_REF // 2).median().iloc[-1]
    lvl_std = lvl.rolling(_REF, min_periods=_REF // 2).std().iloc[-1]
    lvl_now = float(lvl.iloc[-1])
    lvl_calm = lvl_now <= lvl_med
    vol = s.diff().rolling(win, min_periods=max(5, win // 3)).std()
    vol_calm = vol.iloc[-1] <= vol.rolling(_REF, min_periods=_REF // 2).median().iloc[-1]
    calm = bool(lvl_calm or vol_calm)
    strength = float((lvl_med - lvl_now) / lvl_std) if lvl_std and not math.isnan(lvl_std) else 0.0
    return ("calm" if calm else "turbulent"), name, round(lvl_now, 2), strength


def signal_fraction(base, strength, g, cap):
    """Signal-scaled bet fraction: deeper calm -> bigger size (and vice-versa).

    multiplier = exp(g*z) / exp(g^2/2), clipped to [1/cap, cap]; fraction = base*mult.
    The exp(g^2/2) divisor is the unconditional E[exp(g*z)] for z~N(0,1) — an
    analytic constant (no lookahead) that keeps the long-run average near ``base``.
    g=0 reproduces the old flat fraction; g=1.0 (the deployed default) is the
    max-CAGR point in the sizing sweep; g~0.5 trades a little CAGR for the best
    Sortino/Calmar if you prefer a smoother ride."""
    if g == 0.0:
        return base
    mult = math.exp(g * strength) / math.exp(g * g / 2.0)
    mult = min(max(mult, 1.0 / cap), cap)
    return round(base * mult, 4)


def days_to_res(end_date):
    if not end_date:
        return None
    try:
        return (pd.to_datetime(end_date, utc=True) - pd.Timestamp.now(tz="UTC")).total_seconds() / 86400.0
    except (ValueError, TypeError):
        return None


def load_ledger():
    if not STATE_FILE.exists():
        return pd.DataFrame(columns=COLS)
    df = pd.read_csv(STATE_FILE, dtype=str)
    for c in COLS:
        if c not in df.columns:
            df[c] = ""
    # object (not the strict pandas>=3 StringArray) so in-place numeric marks
    # like current_price/outcome/pnl assign cleanly across pandas versions.
    return df[COLS].astype(object)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-price", type=float, default=0.35)
    ap.add_argument("--days-min", type=int, default=2)
    ap.add_argument("--days-max", type=int, default=45)
    ap.add_argument("--min-volume", type=float, default=5_000.0)
    ap.add_argument("--fraction", type=float, default=0.10, help="base (average) bet fraction")
    ap.add_argument("--size-mode", choices=["flat", "signal"], default="signal",
                    help="flat = constant --fraction (old behaviour); signal = scale by GPR-calm depth")
    ap.add_argument("--tilt", type=float, default=1.0, help="signal tilt strength g (0=flat; 1.0 = max-CAGR point in the sweep)")
    ap.add_argument("--size-cap", type=float, default=2.0, help="max/min size multiple vs base fraction")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not VALID_FILE.exists():
        logger.error("no validation file (%s) — run run_regime_domain_validation.py first", VALID_FILE); return
    vinfo = json.loads(VALID_FILE.read_text())
    validated = set(vinfo.get("validated_domains", []))
    if not validated:
        logger.warning("no validated domains — nothing to trade"); return

    regimes = {d: regime_now(d) for d in validated}
    for d, (reg, name, val, strength) in regimes.items():
        logger.info("validated domain %-16s proxy=%-22s now=%s (%.2f, calm-depth z=%+.2f)",
                    d, name, (reg or "?").upper(), val or float("nan"), strength or 0.0)
    calm_domains = {d for d, (reg, _, _, _) in regimes.items() if reg == "calm"}
    g = args.tilt if args.size_mode == "signal" else 0.0
    if args.size_mode == "signal":
        logger.info("sizing: signal-scaled (base %.2f, tilt g=%.2f, cap %.1fx) — deeper calm bets bigger",
                    args.fraction, g, args.size_cap)

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
        if dom not in validated or dom not in calm_domains:
            continue
        dtr = days_to_res(getattr(m, "end_date", None))
        if dtr is None or not (args.days_min <= dtr <= args.days_max) or m.yes_token in held:
            continue
        feat = order_book_features(fetch_order_book(m.yes_token))
        if not feat:
            continue
        mid, ask = feat["mid"], feat["best_ask"]
        if not (0.02 <= mid <= args.max_price):
            continue
        reg, name, val, strength = regimes[dom]
        frac = signal_fraction(args.fraction, strength, g, args.size_cap)
        logger.info("ENTRY [%s/calm] %s  ask %.3f  %.1fd  size %.3f (z=%+.2f)",
                    dom, (m.question or "")[:42], ask, dtr, frac, strength)
        new.append({"token": m.yes_token, "condition_id": getattr(m, "condition_id", ""),
                    "question": m.question, "domain": dom, "proxy": name, "proxy_value": val,
                    "regime": "calm", "end_date": getattr(m, "end_date", ""), "entry_date": today,
                    "days_to_res": round(dtr, 2), "entry_ask": round(ask, 4), "current_price": round(mid, 4),
                    "bet_fraction": frac, "status": "open", "exit_date": "", "outcome": "", "pnl": ""})
    logger.info("scanned %d open markets → %d new validated-strategy entries", len(markets), len(new))
    if new:
        ledger = pd.concat([ledger, pd.DataFrame(new)], ignore_index=True)

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
            ledger.at[i, "current_price"] = round(mid, 4)
        if outcome is None and mid is not None:
            outcome = 1.0 if mid >= 0.99 else (0.0 if mid <= 0.01 else None)
        if outcome is not None:
            entry = float(row["entry_ask"])
            try:
                row_frac = float(row.get("bet_fraction") or args.fraction)
            except (TypeError, ValueError):
                row_frac = args.fraction
            ledger.at[i, "status"] = "won" if outcome == 1.0 else "lost"
            ledger.at[i, "exit_date"] = today
            ledger.at[i, "outcome"] = outcome
            ledger.at[i, "pnl"] = round((outcome / entry - 1.0) * row_frac, 4)
            resolved += 1

    done = ledger[ledger["status"].isin(["won", "lost"])].copy()
    if not done.empty:
        done["pnl"] = pd.to_numeric(done["pnl"], errors="coerce")
        print("\n  validated-strategy ledger: settled %d / total %d  win=%.2f  realized=%+.3f"
              % (len(done), len(ledger), (done.pnl > 0).mean(), done.pnl.sum()))

    if args.dry_run:
        logger.info("dry-run: %d would-be new, %d resolved — NOT writing", len(new), resolved); return
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    ledger[COLS].to_csv(STATE_FILE, index=False)
    logger.info("ledger saved: %s  (%d open, %d settled)", STATE_FILE,
                (ledger.status == "open").sum(), (ledger.status != "open").sum())


if __name__ == "__main__":
    main()
