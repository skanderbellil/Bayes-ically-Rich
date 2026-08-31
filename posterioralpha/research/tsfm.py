"""TimesFM as one more (noisy) volatility estimator.

Google's TimesFM is a pretrained decoder-only time-series foundation model. It
forecasts *any* series zero-shot, which makes it tempting to point at prices —
but `pead/bayesvol.py` already states the house view: first moments are
~unforecastable, second moments are not. So this adapter only ever shows the
model a **volatility** series and asks it to extend that.

The forecasting target throughout is realized volatility over the next
``horizon`` trading days, expressed in logs:

    rv_w(t)   = sqrt(252 / w * sum_{i<w} r_{t-i}^2)      # annualised, trailing
    x(t)      = log rv_w(t)
    target(t) = x(t + horizon)                            # covers r_{t+1..t+h}

With ``w == horizon`` the target is *exactly* the next-h-day realized vol, and
the context series ``x(.. t)`` is a strict function of returns up to t — so a
forecast made at t never touches a return it should not have seen.

Each classical baseline is exposed as a causal *path* — ``pred[t]`` is the
log-vol forecast for days t+1..t+horizon built from returns up to t — so the
bake-off in ``experiments/run_timesfm_vol_bakeoff.py`` can score TimesFM and
the baselines on one identical (ticker, date) grid.

TimesFM itself is an optional dependency (``pip install -e .[tsfm]``): importing
this module without it works fine, only `TimesFMVolForecaster` raises.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from posterioralpha.pead.bayesvol import bayes_vol_forecast

_ANN = 252

# TimesFM 2.5 (Apache-2.0 weights). 3.0 exists and is stronger, but its weights
# ship under `timesfm-non-commercial-license-v1.0` — research only, no
# production use — so the default here is the checkpoint we could actually
# deploy behind a live book.
DEFAULT_CHECKPOINT = "google/timesfm-2.5-200m-pytorch"


# ----------------------------------------------------------------- series ---
def log_returns(prices: pd.Series) -> pd.Series:
    """Daily log returns, NaNs dropped."""
    return np.log(prices / prices.shift(1)).dropna()


def realized_vol(returns: np.ndarray, window: int) -> np.ndarray:
    """Annualised trailing realized vol over `window` days (NaN until filled).

    Uses the mean of squared returns (no mean subtraction) — the standard RV
    convention; at daily frequency the drift term is negligible against it.
    """
    s = pd.Series(returns)
    ms = s.pow(2).rolling(window).mean()
    return np.sqrt(_ANN * ms).to_numpy()


def log_rv_series(returns: np.ndarray, window: int) -> np.ndarray:
    """`log` of `realized_vol`, with non-finite entries left as NaN."""
    rv = realized_vol(returns, window)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(rv > 0, np.log(rv), np.nan)


# ------------------------------------------------------------- baselines ---
# Each baseline returns a *path*: pred[t] is the log-vol forecast for days
# t+1 .. t+horizon, made from returns up to and including t. Every one of them
# is a causal filter or an explicitly purged fit, so computing the whole path
# in one pass is identical to re-running it at each evaluation date — and about
# three orders of magnitude faster for the Gamma-posterior model, which is a
# sequential Python loop. `tests/test_tsfm_vol.py` pins that equivalence.


def rw_path(returns: np.ndarray, horizon: int) -> np.ndarray:
    """Random walk on log RV: carry today's trailing RV forward.

    A deceptively strong baseline — realized vol is close to a unit root at
    daily frequency, and plenty of published vol models beat it only narrowly.
    """
    return log_rv_series(returns, horizon)


def ewma_path(returns: np.ndarray, lam: float = 0.94,
              warmup: int = 20) -> np.ndarray:
    """RiskMetrics EWMA variance filter, in log-vol units."""
    r = np.asarray(returns, dtype=float)
    v = np.full(len(r), np.nan)
    if len(r) <= warmup:
        return v
    cur = float(np.mean(r[:warmup] ** 2))
    v[warmup - 1] = cur
    for i in range(warmup, len(r)):
        cur = lam * cur + (1.0 - lam) * r[i] * r[i]
        v[i] = cur
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(v > 0, np.log(np.sqrt(_ANN * v)), np.nan)


def bayes_path(returns: np.ndarray, delta: float = 0.96,
               adapt: bool = True, sentinel: float = 0.0) -> np.ndarray:
    """PosteriorAlpha's own Gamma-posterior vol model, held flat over the horizon.

    `bayes_vol_forecast` returns, at index i, the variance forecast for day i
    built from data through i-1; we want the forecast made *at* t, so the path
    is that series shifted back by one. The final entry needs one extra step,
    which we get by appending a **sentinel** return: the extra step does its
    discount-and-forecast off the real history, and the sentinel is only
    consumed by the update that comes *after* the value we read — so it cannot
    influence the forecast. `tests/test_tsfm_vol.py` pins that invariance.

    Under the model's discounted random-walk dynamics the h-step forecast
    equals the 1-step forecast, so holding it flat across the horizon is the
    model's own view — the same way `research/amr.py` consumes it for vol
    targeting.
    """
    r = np.append(np.asarray(returns, dtype=float), sentinel)
    v = bayes_vol_forecast(r, delta=delta, adapt=adapt)[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(v > 0, np.log(np.sqrt(_ANN * v)), np.nan)


def har_path(returns: np.ndarray, horizon: int, eval_idxs: Sequence[int],
             windows: Sequence[int] = (5, 21, 63),
             min_train: int = 250) -> np.ndarray:
    """HAR-RV (Corsi 2009) in logs, refit on an expanding *purged* window.

    Regresses log RV over the next `horizon` days on trailing log RV at short,
    medium and long horizons — the standard academic benchmark that most
    "we beat GARCH" papers actually have to clear. Refit at each evaluation
    date on the pairs whose target window had already closed by then, so the
    fit never sees a return the forecast should not have seen.

    Only the requested `eval_idxs` are filled in; everything else stays NaN.
    """
    r = np.asarray(returns, dtype=float)
    feats = np.column_stack([log_rv_series(r, w) for w in windows])
    fwd = log_rv_series(r, horizon)
    target = np.full(len(r), np.nan)
    # target[i] = log RV over days i+1 .. i+horizon
    if len(r) > horizon:
        target[: len(r) - horizon] = fwd[horizon:]

    ok = np.isfinite(feats).all(axis=1) & np.isfinite(target)
    rw = rw_path(r, horizon)
    out = np.full(len(r), np.nan)
    for t in eval_idxs:
        # a training pair at i is usable only once its target window has closed
        usable = ok.copy()
        usable[max(t - horizon + 1, 0):] = False
        if usable.sum() < min_train or not np.isfinite(feats[t]).all():
            out[t] = rw[t]
            continue
        X = np.column_stack([np.ones(usable.sum()), feats[usable]])
        beta, *_ = np.linalg.lstsq(X, target[usable], rcond=None)
        out[t] = float(np.concatenate([[1.0], feats[t]]) @ beta)
    return out


def baseline_paths(returns: np.ndarray, horizon: int,
                   eval_idxs: Sequence[int]) -> Dict[str, np.ndarray]:
    """All classical baselines for one asset, as causal prediction paths."""
    return {
        "rw": rw_path(returns, horizon),
        "ewma94": ewma_path(returns),
        "har": har_path(returns, horizon, eval_idxs),
        "bayes": bayes_path(returns),
    }


def realized_target(returns: np.ndarray, horizon: int) -> np.ndarray:
    """target[t] = log realized vol over days t+1 .. t+horizon (NaN at the end)."""
    fwd = log_rv_series(returns, horizon)
    out = np.full(len(returns), np.nan)
    if len(returns) > horizon:
        out[: len(returns) - horizon] = fwd[horizon:]
    return out


# --------------------------------------------------------------- TimesFM ---
@dataclass
class TimesFMVolForecaster:
    """Zero-shot log-RV forecaster backed by TimesFM.

    The model is shown the daily log-RV series and asked to extend it by
    `horizon` steps; we read the final step, which by construction is the log
    realized vol of exactly the next `horizon` days.

    Requests are batched (`forecast_batch`) because loading and running a 200M
    transformer per (ticker, date) would dominate the study's runtime.
    """

    checkpoint: str = DEFAULT_CHECKPOINT
    max_context: int = 1024
    max_horizon: int = 32
    batch_size: int = 32
    _model: object = field(default=None, repr=False, init=False)

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            import timesfm
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "TimesFM is an optional dependency: pip install -e '.[tsfm]' "
                "(pulls torch + the timesfm package)."
            ) from exc

        torch.set_float32_matmul_precision("high")
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.checkpoint)
        model.compile(
            timesfm.ForecastConfig(
                max_context=self.max_context,
                max_horizon=self.max_horizon,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                # log-vol is negative whenever annualised vol < 100%, i.e.
                # essentially always — do not clamp forecasts at zero.
                infer_is_positive=False,
                fix_quantile_crossing=True,
            )
        )
        self._model = model

    def forecast_batch(self, contexts: List[np.ndarray],
                       horizon: int) -> np.ndarray:
        """Point forecast at step `horizon` for each context series."""
        self.load()
        out = np.full(len(contexts), np.nan)
        for lo in range(0, len(contexts), self.batch_size):
            chunk = contexts[lo: lo + self.batch_size]
            point, _ = self._model.forecast(horizon=horizon, inputs=chunk)
            out[lo: lo + len(chunk)] = np.asarray(point)[:, horizon - 1]
        return out


def build_contexts(returns: np.ndarray, t_idxs: Sequence[int], horizon: int,
                   max_context: int) -> List[np.ndarray]:
    """Log-RV context windows ending at each t in `t_idxs` (causal by slicing)."""
    contexts = []
    for t in t_idxs:
        x = log_rv_series(returns[: t + 1], horizon)
        x = x[np.isfinite(x)][-max_context:]
        contexts.append(x.astype(np.float32))
    return contexts


# ------------------------------------------------------------- evaluation ---
def qlike(rv_true: np.ndarray, rv_pred: np.ndarray) -> np.ndarray:
    """QLIKE loss on the variance scale — the standard robust vol loss.

    Robust in Patton's (2011) sense: because realized vol is only a noisy proxy
    for the latent variance, most losses rank forecasts differently depending on
    the proxy — QLIKE does not, so long as the proxy is conditionally unbiased.
    It also punishes under-prediction far more than over-prediction, which is
    the asymmetry a vol-targeted book actually lives with.
    """
    ratio = (rv_true ** 2) / (rv_pred ** 2)
    return ratio - np.log(ratio) - 1.0


def causal_bias_correction(panel: pd.DataFrame, min_obs: int = 12) -> pd.Series:
    """Per-model additive level correction in log space, estimated causally.

    A log-space point forecast is a *median*; QLIKE scores a *mean*, so a raw
    comparison would reward whichever model happens to sit higher rather than
    whichever is sharper. We therefore give every model the same expanding-window
    debiasing, fitted only on eval dates whose targets had already closed —
    dropping the last two so a 21-day target window cannot still be open.

    Returns a Series aligned to `panel` holding the correction to add to `pred`.
    """
    dates = np.sort(panel["date"].unique())
    out = pd.Series(0.0, index=panel.index)
    for model, grp in panel.groupby("model", sort=False):
        err = grp.groupby("date")["resid"].mean().reindex(dates)
        # expanding mean of residuals strictly before date i-1 (purge 2 dates)
        run = err.shift(2).expanding(min_periods=min_obs).mean()
        corr = run.reindex([d for d in grp["date"]]).to_numpy()
        out.loc[grp.index] = np.nan_to_num(corr, nan=0.0)
    return out


def newey_west_t(d: np.ndarray, lag: int = 1) -> float:
    """t-statistic on mean(d) with a Newey-West HAC variance (Bartlett kernel)."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 8 or d.std() == 0:
        return np.nan
    e = d - d.mean()
    gamma0 = float(e @ e) / n
    var = gamma0
    for k in range(1, min(lag, n - 1) + 1):
        gk = float(e[k:] @ e[:-k]) / n
        var += 2.0 * (1.0 - k / (lag + 1.0)) * gk
    if var <= 0:
        return np.nan
    return float(d.mean() / np.sqrt(var / n))
