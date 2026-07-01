"""
Champion stack — live, out-of-sample weight computation (stage 2: research).

Replicates the FROZEN SPEC from ``experiments/run_champion_stack.py`` (read
that file's module docstring for the full derivation and the ablation/LOO
evidence). Every constant and formula below is copied verbatim from that
script; nothing is re-fitted here. ``experiments/`` is not an importable
package, so this module duplicates the handful of functions it needs
(``causal_rank``, ``hmm_stance``) rather than importing them.

  FROZEN SPEC (see experiments/run_champion_stack.py for the derivation)
  ------------------------------------------------------------------------
  vote        equal-weight mean of 5 soft-signed layers:
              liq    Δ126d net liquidity, pub-lag 5
              vix_ts VIX3M/VIX − 1
              credit HYG−IEF 21d relative return
              dollar −UUP 63d return
              tug    21d overnight−intraday compounded spread (QQQ)
  slow-down   f = 1 − causal-rank(std of 4 price stances);
              vote_eff = f·MA63(vote) + (1−f)·vote
  exposure    e = 2×vote_eff, capped [0, 2]
  ceiling     e ×½ for 60bd after each debt-ceiling resolution (list
              hardcoded through Jun-2023 — APPEND NEW DEALS, see below)
  retail      w_qld = e/2 (capped [0,1]) in QLD, w_uup = 1 − w_qld in UUP

Usage
-----
    from posterioralpha.research.champion_live import compute_current_weights
    out = compute_current_weights()
    print(out["date"], out["w_qld"], out["w_uup"])
"""
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

from posterioralpha.data import load_net_liquidity
from posterioralpha.research import HMM3, precompute_bocpd, soft_sign

logger = logging.getLogger(__name__)

# ── frozen spec constants (copied verbatim from experiments/run_champion_stack.py) ──
LEV_CAP = 2.0
H_LIQ, PUB_LAG, CREDIT_WIN = 126, 5, 21
RANK_WIN, HMM_WIN, HMM_REFIT = 756, 756, 21

# Debt-ceiling resolution dates: exposure is halved for 60 business days after
# each. List hardcoded through Jun-2023 (copied as-is from run_champion_stack.py)
# — **APPEND NEW DEALS** here as Congress resolves future debt-ceiling episodes,
# or this live signal will silently stop applying the ceiling discount.
DEBT_CEILING = ["2011-08-02", "2013-10-17", "2014-02-15", "2015-11-02",
                "2017-09-08", "2018-02-09", "2019-08-02", "2021-10-14",
                "2021-12-16", "2023-06-03"]   # complete through Jun-2023

# QQQ (OHLC, for the tug layer), the 5 macro-vote tickers, and the retail
# wrapper's implementable leg. ~5y is the minimum safe warm-up for the
# flip-risk slow-down (HMM_WIN=756bd + RANK_WIN=756bd of D history for the
# causal rank to stabilise); we pull extra margin on top of that.
TICKERS = ["QQQ", "^VIX", "^VIX3M", "HYG", "IEF", "UUP", "QLD"]
HISTORY_PERIOD = "7y"


def causal_rank(s: pd.Series, win: int = RANK_WIN) -> pd.Series:
    """Copied verbatim from experiments/run_champion_stack.py."""
    return s.rolling(win, min_periods=252).apply(
        lambda x: (x < x[-1]).mean(), raw=True).shift(1)


def hmm_stance(ret: pd.Series) -> pd.Series:
    """Copied verbatim from experiments/run_champion_stack.py."""
    vals = ret.values.reshape(-1, 1)
    out = pd.Series(np.nan, index=ret.index)
    for t in range(HMM_WIN, len(ret), HMM_REFIT):
        window = vals[t - HMM_WIN:t + 1]
        model = HMM3().fit(window)
        p_bull, p_side, _ = model.regime_probs(window)
        out.iloc[t] = p_bull + 0.5 * p_side
    return out.ffill()


def _yf_session():
    """
    A plain ``requests.Session`` with a browser UA, passed explicitly to
    yfinance. yfinance's default transport (curl_cffi, browser-TLS
    impersonation) can fail under some corporate/sandboxed TLS-intercepting
    proxies; a stock ``requests`` session routes through the normal
    ``requests``/urllib3 CA + proxy machinery instead and is what actually
    works in this repo's sandboxed dev environment. Harmless in CI/plain
    internet too.
    """
    import requests
    sess = requests.Session()
    sess.headers.update({
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0 Safari/537.36"),
    })
    return sess


def _download_prices(period: str = HISTORY_PERIOD) -> pd.DataFrame:
    """
    Fresh daily history for the 7 tickers the stack needs, via yfinance.

    Returns a DataFrame with columns QQQ, QQQ_Open, VIX, VIX3M, HYG, IEF,
    UUP, QLD (all Close except QQQ_Open), reindexed to the intersection of
    dates where every column has a value (matches the dropna-intersection
    approach in experiments/run_champion_stack.py).
    """
    import yfinance as yf

    raw = yf.download(TICKERS, period=period, interval="1d", auto_adjust=False,
                       progress=False, group_by="ticker", threads=False,
                       session=_yf_session())
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance returned no data for {TICKERS}")
    have = set(raw.columns.get_level_values(0))
    missing = [t for t in TICKERS if t not in have]
    if missing:
        raise RuntimeError(f"yfinance download missing tickers: {missing}")

    price = pd.DataFrame({
        "QQQ":      raw["QQQ"]["Close"],
        "QQQ_Open": raw["QQQ"]["Open"],
        "VIX":      raw["^VIX"]["Close"],
        "VIX3M":    raw["^VIX3M"]["Close"],
        "HYG":      raw["HYG"]["Close"],
        "IEF":      raw["IEF"]["Close"],
        "UUP":      raw["UUP"]["Close"],
        "QLD":      raw["QLD"]["Close"],
    })
    idx = price.dropna().index
    if idx.empty:
        raise RuntimeError("no dates with complete data across all 7 tickers")
    return price.reindex(idx)


def _load_liquidity(asof: pd.Timestamp) -> tuple[pd.Series, bool, int]:
    """
    Net liquidity, refreshed via FRED when ``FRED_API_KEY`` is set, else the
    bundled offline snapshot at ``datasets/net_liquidity.csv``.

    Returns ``(net_liquidity_series, liq_stale, liq_stale_days)`` where
    ``liq_stale`` is True iff we could not refresh (no key, or the refresh
    failed and we fell back to the bundled CSV) and ``liq_stale_days`` is the
    gap, in calendar days, between ``asof`` and the last real FRED
    observation actually in hand (meaningful in both branches: WALCL/TGA/RRP
    are weekly-published, so even a fresh pull always lags a few days).
    """
    from posterioralpha.env import load_env
    load_env()
    api_key = os.environ.get("FRED_API_KEY")
    stale = True
    if api_key:
        try:
            from posterioralpha.data.macro import build_net_liquidity
            nl = build_net_liquidity()["net_liquidity"]
            stale = False
        except Exception as exc:
            logger.warning("FRED refresh failed (%s: %s); falling back to "
                            "bundled datasets/net_liquidity.csv", type(exc).__name__, exc)
            nl = load_net_liquidity()["net_liquidity"]
    else:
        logger.info("FRED_API_KEY not set; using bundled datasets/net_liquidity.csv")
        nl = load_net_liquidity()["net_liquidity"]

    stale_days = int((pd.Timestamp(asof).normalize() - nl.index.max().normalize()).days)
    return nl, stale, max(stale_days, 0)


def compute_current_weights(period: str = HISTORY_PERIOD) -> dict:
    """
    Compute today's champion-stack layers, vote, flip-risk-adjusted vote,
    exposure, and the retail QLD/UUP weights — faithfully replicating the
    frozen spec in ``experiments/run_champion_stack.py``.

    Returns a dict with keys: date, liq, vix_ts, credit, dollar, tug, vote,
    flip_f, vote_eff, exposure, w_qld, w_uup, qld_close, uup_close,
    liq_stale, liq_stale_days.
    """
    price = _download_prices(period)
    idx = price.index
    min_needed = HMM_WIN + 252
    if len(idx) < min_needed:
        raise RuntimeError(
            f"insufficient warm-up history: got {len(idx)} trading days, "
            f"need >= {min_needed} for the flip-risk slow-down (HMM + causal "
            f"rank warm-up) — widen `period`")

    nl_raw, liq_stale, liq_stale_days = _load_liquidity(idx.max())
    nl = nl_raw.reindex(idx.union(nl_raw.index)).sort_index().ffill().reindex(idx)

    qqq, q_o, q_c = price["QQQ"], price["QQQ_Open"], price["QQQ"]
    r_cc = qqq.pct_change()

    # ── vote (verbatim from run_champion_stack.py) ─────────────────────────
    liq = soft_sign(nl.diff(H_LIQ).shift(PUB_LAG)).shift(1)
    vix_ts = soft_sign(price["VIX3M"] / price["VIX"] - 1.0).shift(1)
    credit = soft_sign(price["HYG"].pct_change(CREDIT_WIN)
                       - price["IEF"].pct_change(CREDIT_WIN)).shift(1)
    dollar = soft_sign(-price["UUP"].pct_change(63)).shift(1)
    r_on, r_id = q_o / q_c.shift(1) - 1.0, q_c / q_o - 1.0
    tug = soft_sign(((1 + r_on).rolling(21).apply(np.prod, raw=True)
                     - (1 + r_id).rolling(21).apply(np.prod, raw=True))).shift(1)
    vote = pd.DataFrame({"liq": liq, "vix_ts": vix_ts, "credit": credit,
                         "dollar": dollar, "tug": tug}).mean(axis=1)

    # ── flip risk (verbatim) ────────────────────────────────────────────────
    _, erl = precompute_bocpd(r_cc.fillna(0.0).values)
    stances = pd.DataFrame({
        "trend":   soft_sign(qqq.pct_change(126)).shift(1),
        "meanrev": soft_sign(-qqq.pct_change(21)).shift(1),
        "bocpd":   causal_rank(pd.Series(erl, index=idx)),
        "hmm":     hmm_stance(r_cc).shift(1),
    })
    D = stances.std(axis=1).where(stances.notna().all(axis=1))
    f = 1.0 - causal_rank(D)
    vote_ma = vote.rolling(63).mean()
    vote_eff = f * vote_ma + (1.0 - f) * vote

    # ── ceiling mask (verbatim) ─────────────────────────────────────────────
    dc_mask = pd.Series(False, index=idx)
    for d in pd.to_datetime(DEBT_CEILING):
        i = idx.searchsorted(d)
        if 0 < i < len(idx):
            dc_mask.iloc[i:min(i + 61, len(idx))] = True
    dc_mask = dc_mask.shift(1).fillna(False)

    # ── exposure + retail wrapper (verbatim: "FULL STACK in QLD+UUP") ──────
    e = (2.0 * vote_eff).clip(0.0, LEV_CAP) * np.where(dc_mask, 0.5, 1.0)
    e = pd.Series(e, index=idx)

    asof = e.last_valid_index()
    if asof is None:
        raise RuntimeError("flip-risk pipeline produced no valid exposure over "
                            "the downloaded history — check warm-up length")

    e_t = float(e.loc[asof])
    w_qld = float(np.clip(e_t / 2.0, 0.0, 1.0))
    w_uup = 1.0 - w_qld

    def _f(s: pd.Series) -> float | None:
        v = s.loc[asof]
        return float(v) if pd.notna(v) else None

    return {
        "date": asof.date().isoformat(),
        "liq": _f(liq), "vix_ts": _f(vix_ts), "credit": _f(credit),
        "dollar": _f(dollar), "tug": _f(tug),
        "vote": _f(vote), "flip_f": _f(f), "vote_eff": _f(vote_eff),
        "exposure": e_t, "w_qld": w_qld, "w_uup": w_uup,
        "qld_close": float(price["QLD"].loc[asof]),
        "uup_close": float(price["UUP"].loc[asof]),
        "liq_stale": liq_stale, "liq_stale_days": liq_stale_days,
    }
