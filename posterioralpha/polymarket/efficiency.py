"""Information efficiency of Polymarket prices.

How informative is a market's price about its eventual resolution, and how does that
improve as settlement approaches? The natural scoring rule is the **Brier score**,
the mean squared error of the price as a probability forecast::

    brier = mean( (outcome - price)^2 )     # 0 = perfect, 0.25 = a coin-flip at 0.5

Lower is more informative. We measure it as a function of **days to resolution**
(does the price converge?) and per **topic** (which domains are efficient?), and we
report **Brier skill** relative to a base-rate forecaster::

    skill = 1 - brier_price / brier_baseline

where the baseline always predicts the group's unconditional Yes-rate. skill > 0
means the live price beats "just knowing how often these markets resolve Yes";
skill ≈ 0 means the market price adds no information over the base rate.

Unlike ``events.build_event_table`` this keeps a market's **entire** life including
the final days (no forward-return horizon is dropped), so the convergence curve runs
all the way into resolution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_brier_frame(panel: pd.DataFrame, outcomes: pd.Series, min_history: int = 5) -> pd.DataFrame:
    """One row per (market, day): price, outcome, days_left, brier — full market life."""
    rows = []
    for m in panel.columns:
        if m not in outcomes.index:
            continue
        s = panel[m].dropna()
        if len(s) < min_history:
            continue
        y = float(outcomes.loc[m])
        last = s.index[-1]
        for d, p in s.items():
            rows.append((m, d, float(p), y, int((last - d).days)))
    df = pd.DataFrame(rows, columns=["market", "date", "price", "outcome", "days_left"])
    df["brier"] = (df["outcome"] - df["price"]) ** 2
    return df


def brier_skill(frame: pd.DataFrame, baseline_rate: float | None = None) -> dict:
    """Brier score of the price vs a base-rate forecaster (skill > 0 ⇒ price informative).

    The base rate is computed once per *market* (not per episode) so long-lived
    markets don't dominate it; pass ``baseline_rate`` to override.
    """
    if frame.empty:
        return {"n": 0}
    if baseline_rate is None:
        baseline_rate = frame.groupby("market")["outcome"].first().mean()
    brier_price = float(frame["brier"].mean())
    brier_base = float(((frame["outcome"] - baseline_rate) ** 2).mean())
    skill = 1.0 - brier_price / brier_base if brier_base > 0 else 0.0
    return {
        "n": int(len(frame)),
        "n_markets": int(frame["market"].nunique()),
        "base_rate": float(baseline_rate),
        "brier_price": brier_price,
        "brier_base": brier_base,
        "skill": float(skill),
    }
