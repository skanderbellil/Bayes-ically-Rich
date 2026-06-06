"""Compute earnings-surprise SUE and forward returns for PEAD testing.

Signal definition (analyst-surprise SUE):
  SUE = (Reported EPS - Estimate EPS) / Estimate EPS (in %)
  This is Yahoo's native "Surprise(%)" column, split-immune and deep-history.

Forward returns:
  Measured from next trading day after earnings announcement (t+1) to t+1,
  t+21, t+63 for the drift test. Aligned to announcement date, not filing date.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np


def load_earnings_signal(earn_csv: Path) -> pd.DataFrame | None:
    """Load a ticker's earnings-surprises file.

    Returns a frame indexed by announcement date with columns:
      - Reported EPS
      - EPS Estimate
      - Surprise(%) <- the SUE signal

    Returns None if file doesn't exist or is too small.
    """
    if not earn_csv.exists():
        return None
    df = pd.read_csv(earn_csv, index_col=0, parse_dates=True)
    if len(df) < 8:
        return None
    df.index.name = "announcement_date"
    return df.sort_index()


def load_prices(price_csv: Path) -> pd.DataFrame | None:
    """Load a ticker's adjusted close prices.

    Returns a frame indexed by trading date with column 'Close'.
    """
    if not price_csv.exists():
        return None
    df = pd.read_csv(price_csv, index_col=0, parse_dates=True)
    if "Close" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Close"})
    return df.sort_index()[["Close"]]


def align_earnings_to_prices(
    earn_df: pd.DataFrame, price_df: pd.DataFrame
) -> pd.DataFrame:
    """Align earnings announcements to trading calendar.

    For each announcement date, find the next trading day (announcement to t+1)
    and compute forward returns at t+1, t+21, t+63.

    Returns a frame indexed by announcement date with:
      - Surprise(%)
      - price_t0 (close on next trading day after announcement)
      - ret_t1, ret_t21, ret_t63 (log returns forward)
    """
    result = []
    for ann_date, row in earn_df.iterrows():
        sue = row["Surprise(%)"]

        # Find next trading day after announcement.
        future_prices = price_df[price_df.index > ann_date]
        if len(future_prices) < 2:  # need t+1 and at least t+21
            continue

        t1_idx = 1  # next trading day = index 1 into future_prices
        p_t1 = future_prices.iloc[t1_idx, 0]

        fwd_ret = {}
        fwd_ret["sue"] = sue
        fwd_ret["price_t1"] = p_t1

        # Return to t+1 (announcement day close to t+1 close).
        p_t0 = price_df[price_df.index <= ann_date].iloc[-1, 0]
        fwd_ret["ret_t1"] = np.log(p_t1 / p_t0) * 100

        # Returns to t+21 and t+63 (log, %).
        if len(future_prices) > 21:
            p_t21 = future_prices.iloc[21, 0]
            fwd_ret["ret_t21"] = np.log(p_t21 / p_t0) * 100
        else:
            fwd_ret["ret_t21"] = np.nan

        if len(future_prices) > 63:
            p_t63 = future_prices.iloc[63, 0]
            fwd_ret["ret_t63"] = np.log(p_t63 / p_t0) * 100
        else:
            fwd_ret["ret_t63"] = np.nan

        result.append((ann_date, fwd_ret))

    if not result:
        return None
    df = pd.DataFrame([r[1] for r in result], index=[r[0] for r in result])
    df.index.name = "announcement_date"
    return df


def build_signal_panel(
    tickers: list[str],
    earn_dir: Path,
    price_dir: Path,
) -> pd.DataFrame:
    """Build a cross-sectional earnings-announcement panel.

    One row per (ticker, announcement_date) with columns:
      ticker, sue, ret_t1, ret_t21, ret_t63, announcement_date.

    Drops any row with missing SUE or forward returns (t+1 and t+21).
    """
    rows = []
    for t in tickers:
        earn_csv = Path(earn_dir) / f"{t}__earnings.csv"
        price_csv = Path(price_dir) / f"{t}__prices.csv"

        earn_df = load_earnings_signal(earn_csv)
        if earn_df is None:
            continue

        price_df = load_prices(price_csv)
        if price_df is None:
            continue

        aligned = align_earnings_to_prices(earn_df, price_df)
        if aligned is None or len(aligned) == 0:
            continue

        aligned["ticker"] = t
        rows.append(aligned.reset_index())

    if not rows:
        return pd.DataFrame()

    panel = pd.concat(rows, ignore_index=True)

    # Drop missing SUE or critical forward returns.
    panel = panel.dropna(subset=["sue", "ret_t1", "ret_t21"], how="any")

    # Keep only announcements with t+1 and t+21 forward returns.
    panel = panel[["ticker", "announcement_date", "sue", "ret_t1", "ret_t21", "ret_t63"]]
    panel = panel.sort_values("announcement_date")

    return panel
