#!/usr/bin/env python3
"""
Market-flow imbalance backtest
==============================
Historical backtest of an order-flow signal on Polymarket resolved markets.

Signal: net dollar buy pressure on the YES token in the final <window> days
before a market's actual close (closedTime). Enter YES at mid-price at the
start of that window if net flow > 0; skip otherwise.
Compare against a baseline (always enter YES regardless of flow).

The data-api.polymarket.com/trades endpoint returns only the most-recent 3000
fills. For high-volume markets this covers the final hours/days of trading —
exactly the pre-close window we care about. We use the CLOB daily price
history (fetch_token_history) to get an objective entry price.

Usage
-----
  python experiments/run_market_flow_backtest.py
  python experiments/run_market_flow_backtest.py --n-markets 50 --window 7
  python experiments/run_market_flow_backtest.py --no-plots
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

import _bootstrap  # noqa: F401

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate

from posterioralpha.polymarket.fetch import (
    fetch_market_trades,
    fetch_markets,
    fetch_token_history,
)
from posterioralpha.polymarket.paths import RESULTS_DIR, ensure_dirs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SIZING = 0.10  # flat 10% of bankroll per trade


# ---------------------------------------------------------------------------
# Flow computation (from data-api trades)
# ---------------------------------------------------------------------------

def compute_net_flow_yes(
    trades: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> float | None:
    """Net dollar buy imbalance on YES tokens within [window_start, window_end).

    Returns None if there are no YES-token trades in the window.
    """
    mask = (trades["timestamp"] >= window_start) & (trades["timestamp"] < window_end)
    window_trades = trades[mask]
    yes_trades = window_trades[window_trades["outcome"].str.lower() == "yes"]
    if yes_trades.empty:
        return None
    dollar_vol = yes_trades["size"] * yes_trades["price"]
    buys  = dollar_vol[yes_trades["side"].str.upper() == "BUY"].sum()
    sells = dollar_vol[yes_trades["side"].str.upper() == "SELL"].sum()
    return float(buys - sells)


# ---------------------------------------------------------------------------
# Entry price (from CLOB daily price history)
# ---------------------------------------------------------------------------

def get_entry_price_from_history(
    yes_token: str,
    entry_date: pd.Timestamp,
) -> float | None:
    """Fetch YES price from CLOB history at (or just before) entry_date.

    entry_date should be tz-naive or will be stripped to tz-naive for
    comparison with the tz-naive CLOB index.

    Returns None if no price data is available at that point.
    """
    try:
        history = fetch_token_history(yes_token, use_cache=True)
    except Exception as e:
        logger.debug("fetch_token_history failed: %s", e)
        return None

    if history.empty:
        return None

    # Normalise to tz-naive for index comparison
    if hasattr(entry_date, "tz") and entry_date.tz is not None:
        entry_date_naive = entry_date.tz_localize(None)
    else:
        entry_date_naive = entry_date

    avail = history[history.index <= entry_date_naive]
    if avail.empty:
        return None
    return float(avail.iloc[-1])


# ---------------------------------------------------------------------------
# Resolve the effective close time for a market
# ---------------------------------------------------------------------------

def _parse_close_time(row: object) -> pd.Timestamp | None:
    """Best estimate of when trading actually stopped.

    Prefers ``closed_time`` (Gamma's closedTime), falls back to ``end_date``.
    Returns None if neither parses.
    """
    for attr in ("closed_time", "end_date"):
        raw = getattr(row, attr, None)
        if raw is None:
            continue
        try:
            ts = pd.to_datetime(raw, utc=True)
            if not pd.isna(ts):
                return ts
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Main backtest loop
# ---------------------------------------------------------------------------

def run_backtest(
    markets_df: pd.DataFrame,
    window_days: int = 7,
) -> pd.DataFrame:
    """Run the pre-close market-flow imbalance backtest.

    For each market:
      1. Determine the actual close time (``closed_time`` or ``end_date``).
      2. Fetch the CLOB daily price history → entry price at
         ``close_time - window_days``.
      3. Fetch the most-recent trades via the data API → compute net YES flow
         in the ``[close_time - window_days, close_time)`` window.
      4. Baseline: always enter YES at entry_price;
         Strategy: enter only when net_flow_yes > 0.

    Returns a DataFrame sorted by resolution date (end_date).
    """
    records: list[dict] = []

    for i, row in enumerate(markets_df.itertuples(index=False), 1):
        market_id    = str(row.id)
        condition_id = str(getattr(row, "condition_id", "") or "")
        yes_token    = str(getattr(row, "yes_token",    "") or "")
        question     = str(getattr(row, "question",     "") or "")
        outcome      = getattr(row, "outcome", float("nan"))
        end_date_raw = getattr(row, "end_date", None)

        # Need a clean binary outcome
        if not isinstance(outcome, (int, float)) or np.isnan(outcome):
            logger.debug("skip %s — no clean outcome", market_id[:12])
            continue

        if not condition_id or condition_id in ("None", ""):
            logger.debug("skip %s — no conditionId", market_id[:12])
            continue

        # Best close time estimate
        close_time = _parse_close_time(row)
        if close_time is None:
            logger.debug("skip %s — no parseable close time", market_id[:12])
            continue

        # Resolution date for sorting (use end_date for consistency)
        try:
            end_date = pd.to_datetime(end_date_raw, utc=True)
        except Exception:
            end_date = close_time

        logger.info(
            "[%d/%d] %s (close %s)…",
            i, len(markets_df), market_id[:16], str(close_time)[:10],
        )

        # ── Step 1: entry price from CLOB price history ───────────────────
        entry_ts = close_time - pd.Timedelta(days=window_days)
        entry_price = get_entry_price_from_history(yes_token, entry_ts)
        if entry_price is None or not (0.0 < entry_price < 1.0):
            logger.debug(
                "skip %s — bad/missing entry price %.4f (entry_ts=%s)",
                market_id[:12], entry_price or -1, str(entry_ts)[:10],
            )
            time.sleep(0.05)
            continue

        # ── Step 2: net YES flow from data-api trades ─────────────────────
        trades = fetch_market_trades(condition_id, max_trades=2000, sleep=0.2)
        if trades.empty:
            logger.debug("skip %s — no trades returned", market_id[:12])
            continue

        net_flow = compute_net_flow_yes(trades, entry_ts, close_time)
        if net_flow is None:
            logger.debug(
                "skip %s — no YES trades in window [%s, %s)",
                market_id[:12], str(entry_ts)[:10], str(close_time)[:10],
            )
            continue

        # ── Step 3: PnL at resolution ─────────────────────────────────────
        # Buy YES at entry_price; outcome is 0.0 or 1.0
        pnl = (float(outcome) - entry_price) * SIZING

        records.append({
            "market_id":    market_id,
            "question":     question,
            "end_date":     end_date,
            "close_time":   close_time,
            "net_flow":     net_flow,
            "entry_price":  entry_price,
            "outcome":      float(outcome),
            "baseline_pnl": pnl,                           # always enter YES
            "strategy_pnl": pnl if net_flow > 0 else 0.0, # enter only when flow > 0
        })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df.sort_values("end_date").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def sharpe_ratio(returns: pd.Series) -> float:
    """Per-trade info ratio (mean / std * sqrt(N)), a sample-size-adjusted metric."""
    active = returns[returns != 0.0]
    if len(active) < 2 or active.std() == 0:
        return 0.0
    return float(active.mean() / active.std() * np.sqrt(len(active)))


def print_summary(df: pd.DataFrame, window_days: int) -> None:
    """Print a formatted summary table comparing baseline vs. strategy."""
    n_total = len(df)
    n_strat  = int((df["net_flow"] > 0).sum())
    n_skip   = n_total - n_strat

    def metrics(pnl_col: str) -> list:
        s = df[pnl_col]
        active = s[s != 0.0] if pnl_col == "strategy_pnl" else s
        if active.empty:
            return [0, "N/A", "N/A", "N/A", "N/A"]
        wins = (active > 0).sum()
        wr   = wins / len(active)
        avg  = active.mean()
        tot  = active.sum()
        sr   = sharpe_ratio(active)
        return [len(active), f"{wr:.1%}", f"{avg:+.4f}", f"{tot:+.4f}", f"{sr:.2f}"]

    rows = [
        ["Baseline (always YES)"]                       + metrics("baseline_pnl"),
        [f"Strategy (flow>0, skips {n_skip}/{n_total})"] + metrics("strategy_pnl"),
    ]
    print("\n" + "=" * 74)
    print(f"  MARKET-FLOW IMBALANCE BACKTEST — SUMMARY  (window={window_days}d, sizing={SIZING:.0%})")
    print("=" * 74)
    print(tabulate(
        rows,
        headers=["Book", "N trades", "Win rate", "Avg PnL", "Total PnL", "Sharpe-like"],
        tablefmt="rounded_grid",
    ))
    print()


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def plot_results(df: pd.DataFrame, out_path) -> None:
    """Two-panel chart: equity curves (left) + signal scatter / logistic fit (right)."""
    from pathlib import Path
    out_path = Path(out_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left panel: equity curves sorted by resolution date ───────────────
    ax = axes[0]
    baseline_eq = df["baseline_pnl"].fillna(0.0).cumsum().values
    strategy_eq = df["strategy_pnl"].fillna(0.0).cumsum().values
    x = np.arange(len(df))

    ax.plot(x, baseline_eq, label="Baseline (always YES)",
            color="steelblue", lw=2)
    ax.plot(x, strategy_eq, label="Strategy (flow > 0 only)",
            color="darkorange", lw=2)
    ax.axhline(0.0, color="k", lw=0.7, alpha=0.4)
    ax.set_title("Equity Curves — sorted by resolution date", fontsize=12)
    ax.set_xlabel("Markets (chronological by resolution)")
    ax.set_ylabel("Cumulative PnL (fraction of bankroll, 10% sizing)")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.25)

    # Date labels on x-axis
    valid_dates = df["end_date"].dropna()
    if len(valid_dates) >= 2:
        n = len(df)
        ticks  = [0, n // 2, max(n - 1, 0)]
        labels = []
        for t in ticks:
            if t < len(valid_dates):
                labels.append(str(valid_dates.iloc[t])[:10])
            else:
                labels.append("")
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)

    # ── Right panel: signal vs. outcome scatter + logistic fit ────────────
    ax2 = axes[1]
    nf = df["net_flow"].values.astype(float)
    nf_std = nf.std()
    nf_norm = nf / nf_std if nf_std > 0 else nf.copy()
    outcomes = df["outcome"].values.astype(float)

    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.02, 0.02, len(outcomes))
    ax2.scatter(nf_norm, outcomes + jitter, alpha=0.40, s=30,
                color="steelblue", label="Markets", zorder=3)

    # Logistic regression line
    try:
        from sklearn.linear_model import LogisticRegression
        X = nf_norm.reshape(-1, 1)
        mask = np.isfinite(X.ravel()) & np.isfinite(outcomes)
        if mask.sum() >= 5:
            lr = LogisticRegression(max_iter=500).fit(X[mask], outcomes[mask])
            x_grid = np.linspace(nf_norm[mask].min(), nf_norm[mask].max(), 300).reshape(-1, 1)
            y_hat  = lr.predict_proba(x_grid)[:, 1]
            ax2.plot(x_grid.ravel(), y_hat, color="darkorange", lw=2.5,
                     label="Logistic fit")
    except ImportError:
        valid = np.isfinite(nf_norm) & np.isfinite(outcomes)
        if valid.sum() >= 2:
            coeffs = np.polyfit(nf_norm[valid], outcomes[valid], 1)
            x_line = np.linspace(nf_norm[valid].min(), nf_norm[valid].max(), 100)
            ax2.plot(x_line, np.polyval(coeffs, x_line), color="darkorange", lw=2,
                     label="Linear trend")

    ax2.axvline(0, color="k", lw=0.7, alpha=0.4, ls="--")
    ax2.set_ylim(-0.15, 1.15)
    ax2.set_title("Pre-Close Flow vs. Outcome", fontsize=12)
    ax2.set_xlabel("Normalised Net YES Flow (sigma units, last window)")
    ax2.set_ylabel("Outcome (1 = YES resolved)")
    ax2.legend(loc="upper left", fontsize=10)
    ax2.grid(alpha=0.25)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["NO (0)", "YES (1)"])

    fig.suptitle(
        "Polymarket Market-Flow Imbalance Backtest",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    logger.info("Saved chart → %s", out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--n-markets", type=int, default=80,
        help="Number of resolved markets to fetch from Gamma (default 80)",
    )
    ap.add_argument(
        "--window", type=int, default=7,
        help="Pre-close window in days for the flow signal (default 7)",
    )
    ap.add_argument(
        "--min-volume", type=float, default=100_000.0,
        help="Minimum lifetime volume filter (default 100000)",
    )
    ap.add_argument(
        "--no-plots", action="store_true",
        help="Skip chart generation",
    )
    args = ap.parse_args()

    ensure_dirs()

    print("""
╔══════════════════════════════════════════════════════════════════╗
║   P O L Y M A R K E T   ·   Market-Flow Imbalance Backtest       ║
║   Does pre-close buy pressure on YES predict final resolution?    ║
╚══════════════════════════════════════════════════════════════════╝""")
    logger.info(
        "Fetching %d resolved markets (min_volume=%.0f)…",
        args.n_markets, args.min_volume,
    )

    markets_df = fetch_markets(
        n_markets=args.n_markets,
        min_volume=args.min_volume,
        closed=True,
    )
    if markets_df.empty:
        logger.error("No markets returned — check network/API.")
        sys.exit(1)

    # Keep only markets with clean binary outcomes
    markets_df = markets_df.dropna(subset=["outcome"])
    markets_df = markets_df[markets_df["outcome"].isin([0.0, 1.0])].reset_index(drop=True)
    logger.info("Markets with clean binary outcome: %d", len(markets_df))

    if markets_df.empty:
        logger.error("No markets with clean binary outcomes — try a lower --min-volume.")
        sys.exit(1)

    logger.info(
        "Running backtest (window=%d days, sizing=%.0f%%)…",
        args.window, SIZING * 100,
    )
    results = run_backtest(markets_df, window_days=args.window)

    if results.empty:
        logger.error(
            "No tradeable markets found after all filters.\n"
            "  Try: --window %d (wider window) or --min-volume 50000 (more markets).",
            args.window * 2,
        )
        sys.exit(1)

    logger.info("Backtest complete: %d tradeable markets", len(results))
    print_summary(results, args.window)

    # Print top markets by signal strength
    print("  Top markets by net flow signal (positive):")
    show_cols = ["question", "net_flow", "entry_price", "outcome", "strategy_pnl"]
    top = results.nlargest(min(10, len(results)), "net_flow")[show_cols].copy()
    top["question"] = top["question"].str[:50]
    print(tabulate(
        top.values.tolist(),
        headers=list(top.columns),
        tablefmt="simple",
        floatfmt=".4f",
    ))

    if not args.no_plots:
        out_path = RESULTS_DIR / "market_flow_backtest.png"
        plot_results(results, out_path)
        print(f"\n  Chart saved → {out_path}")

    print("\n  Market-flow imbalance backtest complete.")


if __name__ == "__main__":
    main()
