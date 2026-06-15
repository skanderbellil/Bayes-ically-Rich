"""Event-study table: do sharp upside moves predict the actual resolution?

Builds a tidy per-(market, date) episode table from the daily Yes-price panel,
the resolution labels, and the volatility features — then buckets episodes by an
upside-move metric to ask: **conditional on a sharp upside vol return, what is the
market's actual Yes-resolution rate, and does the price keep drifting up?**

Two effects must be separated:
  * **Level.** A market at 0.90 resolves Yes ~90% of the time regardless of vol —
    so we always report the mean price per bucket and the *calibration residual*
    ``yes_rate − mean_price`` (predictive content beyond the price itself).
  * **Direction.** Forward log-odds drift over ``fwd_horizon`` days tells us
    whether a sharp move continues (momentum/informative) or reverts (overreaction).

Cross-fertilisation with the rest of the repo: alongside the raw vol features we
attach the **BOCPD** changepoint probability (``research.regimes.precompute_bocpd``)
run per market on the log-odds returns — the same online change-point detector the
equity strategies use to flag regime breaks, here flagging "this market just
broke to a new level".
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from posterioralpha.research import precompute_bocpd

from .signals import to_logodds
from .volatility import (
    downside_vol,
    logodds_returns,
    realized_vol,
    sharp_up,
    upside_vol,
    vol_skew,
)

logger = logging.getLogger(__name__)


def build_event_table(
    panel: pd.DataFrame,
    outcomes: pd.Series,
    vol_window: int = 10,
    fwd_horizon: int = 10,
    sample_step: int = 3,
    min_history: int = 10,
    hazard: float = 1 / 60,
) -> pd.DataFrame:
    """One row per sampled (market, date) with causal features + the outcome.

    Columns: market, date, price, logodds, upside_vol, downside_vol, vol_skew,
    realized_vol, sharp_up, bocpd_cp, fwd_dlogodds (z_{t+h}−z_t), days_left,
    outcome (0/1).

    Only markets with a clean resolution label are included. Features at *t* use
    data ≤ *t*; ``fwd_dlogodds`` is the realised forward move (the thing we test
    against). ``sample_step`` thins each market's days to curb autocorrelation.
    """
    z = to_logodds(panel)
    uv = upside_vol(panel, vol_window)
    dv = downside_vol(panel, vol_window)
    vs = vol_skew(panel, vol_window)
    rv = realized_vol(panel, vol_window)
    su = sharp_up(panel, vol_window)
    dz = logodds_returns(panel)

    labelled = [m for m in panel.columns if m in outcomes.index]
    rows = []
    for m in labelled:
        s = panel[m].dropna()
        if len(s) < min_history + fwd_horizon + 1:
            continue
        # BOCPD on this market's log-odds returns (causal online detector)
        dzi = dz[m].reindex(s.index).fillna(0.0).values
        cp, _ = precompute_bocpd(dzi, hazard=hazard)
        cp = pd.Series(cp, index=s.index)

        idx = s.index
        last_pos = len(idx) - 1
        for pos in range(min_history, last_pos - fwd_horizon + 1, sample_step):
            t = idx[pos]
            t_fwd = idx[pos + fwd_horizon]
            rows.append({
                "market":       m,
                "date":         t,
                "price":        float(s.loc[t]),
                "logodds":      float(z.loc[t, m]),
                "upside_vol":   float(uv.loc[t, m]),
                "downside_vol": float(dv.loc[t, m]),
                "vol_skew":     float(vs.loc[t, m]),
                "realized_vol": float(rv.loc[t, m]),
                "sharp_up":     float(su.loc[t, m]),
                "bocpd_cp":     float(cp.loc[t]),
                "fwd_dlogodds": float(z.loc[t_fwd, m] - z.loc[t, m]),
                "days_left":    int(last_pos - pos),
                "outcome":      float(outcomes.loc[m]),
            })

    df = pd.DataFrame(rows).dropna(subset=["upside_vol", "logodds", "fwd_dlogodds"])
    logger.info("build_event_table: %d episodes across %d resolved markets",
                len(df), df["market"].nunique() if len(df) else 0)
    return df


def bracket_outcomes(
    events: pd.DataFrame,
    by: str = "upside_vol",
    n_brackets: int = 5,
) -> pd.DataFrame:
    """Bucket episodes into quantile brackets of ``by`` and summarise outcomes.

    Per bracket: episode count, mean of the bracketing metric, mean price,
    actual Yes-resolution rate, **calibration residual** (yes_rate − mean_price,
    the edge beyond price level), and mean forward log-odds drift.
    """
    e = events.dropna(subset=[by]).copy()
    if e.empty:
        return pd.DataFrame()
    # rank → quantile brackets (robust to ties / skew)
    e["_bracket"] = pd.qcut(e[by].rank(method="first"), n_brackets,
                            labels=[f"Q{i+1}" for i in range(n_brackets)])
    g = e.groupby("_bracket", observed=True)
    out = pd.DataFrame({
        "n":             g.size(),
        by:              g[by].mean(),
        "mean_price":    g["price"].mean(),
        "yes_rate":      g["outcome"].mean(),
        "calib_resid":   g["outcome"].mean() - g["price"].mean(),
        "fwd_dlogodds":  g["fwd_dlogodds"].mean(),
    })
    return out.reset_index().rename(columns={"_bracket": "bracket"})
