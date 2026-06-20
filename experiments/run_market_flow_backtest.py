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
    fetch_market_trades_historical,
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
    historical: bool = False,
) -> pd.DataFrame:
    """Run the pre-close market-flow imbalance backtest.

    For each market:
      1. Determine the actual close time (``closed_time`` or ``end_date``).
      2. Fetch the CLOB daily price history → entry price at
         ``close_time - window_days`` (REST mode) or ``start_date`` (historical
         mode).
      3. Fetch the most-recent trades via the data API → compute net YES flow
         in the chosen window.
      4. Baseline: always enter YES at entry_price;
         Strategy: enter only when net_flow_yes > 0.

    Parameters
    ----------
    historical : If True, use ``fetch_market_trades_historical`` and anchor
        the window to the market's ``start_date`` (open date) rather than
        ``closed_time``.  This attempts to measure the *early-window* flow
        signal (first ``window_days`` after market open).

        **IMPORTANT:** As of 2025-06 there is no public source that provides
        fills from earlier than the most-recent ~3,500 trades.  The
        ``--historical`` flag applies client-side time filtering but cannot
        retrieve data that is simply not available.  Markets whose 3,500-fill
        window does not reach back to ``start_date + window_days`` will have
        ``coverage_full=False`` logged as warnings and be dropped from the
        results (no trades in window → skipped), so the result set will be
        smaller than the REST mode.  Use this flag to measure the early-window
        signal on *low-volume* markets that fit within the 3,500-fill cap.

    Returns a DataFrame sorted by resolution date (end_date).
    """
    records: list[dict] = []
    n_no_coverage = 0

    for i, row in enumerate(markets_df.itertuples(index=False), 1):
        market_id    = str(row.id)
        condition_id = str(getattr(row, "condition_id", "") or "")
        yes_token    = str(getattr(row, "yes_token",    "") or "")
        question     = str(getattr(row, "question",     "") or "")
        outcome      = getattr(row, "outcome", float("nan"))
        end_date_raw = getattr(row, "end_date", None)
        start_date_raw = getattr(row, "start_date", None)

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

        # ── Window anchor ─────────────────────────────────────────────────
        if historical and start_date_raw is not None:
            try:
                market_open = pd.to_datetime(start_date_raw, utc=True)
                window_start = market_open
                window_end   = market_open + pd.Timedelta(days=window_days)
                entry_ts     = market_open  # enter at open (first day price)
            except Exception:
                logger.debug("skip %s — bad start_date", market_id[:12])
                continue
        else:
            window_end   = close_time
            window_start = close_time - pd.Timedelta(days=window_days)
            entry_ts     = window_start

        logger.info(
            "[%d/%d] %s (%s window %s → %s)…",
            i, len(markets_df), market_id[:16],
            "early" if historical else "pre-close",
            str(window_start)[:10], str(window_end)[:10],
        )

        # ── Step 1: entry price from CLOB price history ───────────────────
        entry_price = get_entry_price_from_history(yes_token, entry_ts)
        if entry_price is None or not (0.0 < entry_price < 1.0):
            logger.debug(
                "skip %s — bad/missing entry price %.4f (entry_ts=%s)",
                market_id[:12], entry_price or -1, str(entry_ts)[:10],
            )
            time.sleep(0.05)
            continue

        # ── Step 2: net YES flow from data-api trades ─────────────────────
        if historical:
            trades = fetch_market_trades_historical(
                condition_id,
                start_ts=window_start.timestamp(),
                end_ts=window_end.timestamp(),
                max_trades=3500,
                sleep=0.2,
            )
            # Check coverage — if False, early-window data is missing
            if not trades.attrs.get("coverage_full", True):
                n_no_coverage += 1
                logger.warning(
                    "skip %s — early window not covered by available fills "
                    "(market likely has >3500 lifetime trades)",
                    market_id[:12],
                )
                continue
            # Adapt column names for compute_net_flow_yes (expects 'outcome' column)
            if not trades.empty and "outcome" not in trades.columns:
                # fetch_market_trades_historical returns [wallet, side, size, price, timestamp]
                # We need the raw trades for the outcome column; fall back to raw fetch
                trades_raw = fetch_market_trades(condition_id, max_trades=3500, sleep=0.2)
                # Apply the same time window filter client-side
                if not trades_raw.empty:
                    mask = (
                        (trades_raw["timestamp"] >= window_start)
                        & (trades_raw["timestamp"] < window_end)
                    )
                    trades = trades_raw[mask].copy()
                else:
                    trades = pd.DataFrame()
        else:
            trades = fetch_market_trades(condition_id, max_trades=2000, sleep=0.2)

        if trades.empty:
            logger.debug("skip %s — no trades returned", market_id[:12])
            continue

        net_flow = compute_net_flow_yes(trades, window_start, window_end)
        if net_flow is None:
            logger.debug(
                "skip %s — no YES trades in window [%s, %s)",
                market_id[:12], str(window_start)[:10], str(window_end)[:10],
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
            "window_start": window_start,
            "window_end":   window_end,
            "net_flow":     net_flow,
            "entry_price":  entry_price,
            "outcome":      float(outcome),
            "baseline_pnl": pnl,                           # always enter YES
            "strategy_pnl": pnl if net_flow > 0 else 0.0, # enter only when flow > 0
        })

    if historical and n_no_coverage > 0:
        logger.warning(
            "%d markets skipped: early window not covered by API's 3500-fill cap.  "
            "These are high-volume markets — no public source provides their early fills.",
            n_no_coverage,
        )

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
    ap.add_argument(
        "--historical", action="store_true",
        help=(
            "Use early-window mode: measure flow signal in the first "
            "``--window`` days after market open (start_date) rather than "
            "the last ``--window`` days before close.  "
            "NOTE: as of 2025-06, data-api.polymarket.com returns only the "
            "most-recent ≤3,500 fills regardless of any time filter.  "
            "High-volume markets (>3,500 lifetime fills) will be dropped "
            "because their early-window data is unavailable.  Only markets "
            "whose full lifetime fits inside the 3,500-fill cap will be "
            "included — these tend to be lower-volume markets."
        ),
    )
    args = ap.parse_args()

    ensure_dirs()

    mode_label = "EARLY-WINDOW (historical)" if args.historical else "PRE-CLOSE (REST)"
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║   P O L Y M A R K E T   ·   Market-Flow Imbalance Backtest       ║
║   Does buy pressure on YES predict final resolution?              ║
║   Mode: {mode_label:<56}║
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
        "Running backtest (mode=%s, window=%d days, sizing=%.0f%%)…",
        "historical" if args.historical else "pre-close",
        args.window, SIZING * 100,
    )
    results = run_backtest(markets_df, window_days=args.window, historical=args.historical)

    if results.empty:
        if args.historical:
            logger.error(
                "No tradeable markets found.\n"
                "  In --historical mode all high-volume markets are dropped because\n"
                "  data-api.polymarket.com only returns the most-recent ≤3,500 fills.\n"
                "  Try: --min-volume 10000 (smaller markets with full history) or\n"
                "       omit --historical to use pre-close mode (always works).",
            )
        else:
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
        suffix = "_historical" if args.historical else ""
        out_path = RESULTS_DIR / f"market_flow_backtest{suffix}.png"
        plot_results(results, out_path)
        print(f"\n  Chart saved → {out_path}")

    print("\n  Market-flow imbalance backtest complete.")


if __name__ == "__main__":
    main()
