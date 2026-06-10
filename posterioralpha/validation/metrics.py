"""Performance metrics for portfolio evaluation."""
from typing import Dict, Optional

import numpy as np
import pandas as pd

_ANN = 252  # trading days per year


def compute_metrics(
    returns: pd.Series,
    benchmark: Optional[pd.Series] = None,
    rf: float = 0.04,
) -> Dict[str, float]:
    """
    Compute a comprehensive set of risk/return metrics.

    Parameters
    ----------
    returns   : daily arithmetic portfolio returns
    benchmark : daily arithmetic benchmark returns (for α, β, IR)
    rf        : annual risk-free rate

    Returns
    -------
    dict with keys: CAGR, Volatility, Sharpe, Sortino, Max DD,
                    Calmar, Win Rate, Total Return, [Alpha, Beta, Info Ratio]
    """
    r = returns.dropna()
    n = len(r)
    if n == 0:
        return {}

    rf_daily   = rf / _ANN
    excess     = r - rf_daily
    ann_excess = excess.mean() * _ANN

    # ── Return metrics ────────────────────────────────────────────────────
    total_ret = (1.0 + r).prod() - 1.0
    years     = n / _ANN
    cagr      = (1.0 + total_ret) ** (1.0 / years) - 1.0 if years > 0 else 0.0

    # ── Risk metrics ──────────────────────────────────────────────────────
    vol = r.std() * np.sqrt(_ANN)

    # Sharpe
    sharpe = ann_excess / (excess.std() * np.sqrt(_ANN) + 1e-9)

    # Sortino (downside deviation)
    downside_std = excess[excess < 0].std() * np.sqrt(_ANN)
    sortino = ann_excess / (downside_std + 1e-9)

    # Maximum drawdown
    cum       = (1.0 + r).cumprod()
    roll_max  = cum.cummax()
    drawdown  = (cum - roll_max) / (roll_max + 1e-9)
    max_dd    = float(drawdown.min())

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    # Win rate
    win_rate = float((r > 0).mean())

    metrics: Dict[str, float] = {
        "CAGR":         cagr,
        "Volatility":   vol,
        "Sharpe":       sharpe,
        "Sortino":      sortino,
        "Max DD":       max_dd,
        "Calmar":       calmar,
        "Win Rate":     win_rate,
        "Total Return": total_ret,
    }

    # ── Benchmark-relative metrics ────────────────────────────────────────
    if benchmark is not None:
        bench = benchmark.dropna()
        aligned = pd.concat([r, bench], axis=1).dropna()
        if len(aligned) >= 30:
            p  = aligned.iloc[:, 0]
            b  = aligned.iloc[:, 1]

            cov_mat = np.cov(p - rf_daily, b - rf_daily)
            beta    = cov_mat[0, 1] / (cov_mat[1, 1] + 1e-9)
            alpha   = ((p.mean() - rf_daily) - beta * (b.mean() - rf_daily)) * _ANN

            active  = p - b
            ir      = active.mean() / (active.std() + 1e-9) * np.sqrt(_ANN)

            metrics["Alpha"] = alpha
            metrics["Beta"]  = beta
            metrics["Info Ratio"] = ir

    return metrics
