"""Weather-edge prediction-market module — London daily max-temperature contracts.

A *scheduled, objectively-resolved* prediction-market edge of the same archetype
as the macro favorite–longshot trade (``MACRO_EQUITY``): the market prices a
distribution over a number that an external, calibrated model **also** forecasts,
so mispricing is directly measurable as ``edge = P_model − P_market``. Crucially —
and unlike the politics/sports books that turned out to be volume-selection
artifacts (``STRATEGY_SYNTHESIS``) — these markets settle daily on an *objective*
reading and therefore draw volume regardless of which bucket wins, so there is no
"longshot only attracts volume once it's already winning" survivorship.

The signal is the london-edge construction made **point-in-time**:

  * forecast  — five global NWP models (ECMWF/GFS/ICON/UKMO/MeteoFrance), pulled
                from Open-Meteo's *historical-forecast archive* so each date sees
                only the forecast that existed then (no reanalysis lookahead).
  * consensus — model mean ``μ`` and inter-model dispersion ``σ`` (floored, so
                disagreement widens the implied distribution).
  * bucket P  — a normal CDF with a ±0.5°C continuity correction over the
                integer-degree buckets the market quotes.

Pipeline role::

    data      → fetch_weather_event (Gamma slug) + point_in_time_forecast (Open-Meteo)
    research  → parse_bucket, bucket_prob, compute_edges (edge / Kelly / EV)
    backtest  → experiments/run_polymarket_weather_backtest.py
    validation→ event-cluster bootstrap CI + calibration (in the experiment)

All data comes from public, key-less REST APIs.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date as _date
from typing import Optional

import numpy as np
from scipy.stats import norm

import pandas as pd

from .fetch import _get
from .paths import RAW_DIR, ensure_dirs

# ---------------------------------------------------------------------------
# Endpoints & constants
# ---------------------------------------------------------------------------
GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"
CLOB_HISTORY_URL = "https://clob.polymarket.com/prices-history"
HIST_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Five global models, exactly the london-edge panel. Open-Meteo suffixes each
# requested variable with the model id when ``&models=`` lists more than one.
MODELS = [
    "ecmwf_ifs025",
    "gfs_seamless",
    "icon_seamless",
    "ukmo_seamless",
    "meteofrance_seamless",
]

# Inter-model σ floor, in *degrees of the market's own unit*: even unanimous
# models are not certain N days out, so the implied distribution never collapses
# below this width. London-edge uses 0.7°C; we scale it to 1.3°F for Fahrenheit
# markets so the floor is the same physical uncertainty either way.
SIGMA_FLOOR_C = 0.7
SIGMA_FLOOR_F = 0.7 * 9.0 / 5.0
MIN_MODELS = 3  # require ≥3/5 models to trust the consensus (london-edge rule)

_MONTHS = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]


# ---------------------------------------------------------------------------
# City registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class City:
    """A Polymarket temperature-market location.

    ``unit`` is the market's quoting unit ('C' or 'F') — US cities quote °F in
    2-degree buckets, Europe/Asia quote °C in 1-degree buckets — and the forecast
    is pulled in that same unit so edges are computed apples-to-apples. ``lat``/
    ``lon`` are a representative city-center reading (an approximation of the
    exact resolution station; see WEATHER_EDGE.md caveats).
    """
    slug: str          # Polymarket slug fragment, e.g. "nyc"
    display: str       # human label
    lat: float
    lon: float
    timezone: str
    unit: str          # 'C' or 'F'


CITIES: dict[str, City] = {
    # Europe / Asia — °C, 1-degree buckets
    "london": City("london", "London", 51.51, -0.13, "Europe/London", "C"),
    "paris": City("paris", "Paris", 48.85, 2.35, "Europe/Paris", "C"),
    "tokyo": City("tokyo", "Tokyo", 35.68, 139.69, "Asia/Tokyo", "C"),
    "moscow": City("moscow", "Moscow", 55.76, 37.62, "Europe/Moscow", "C"),
    # North America — °F, 2-degree buckets
    "nyc": City("nyc", "New York City", 40.78, -73.97, "America/New_York", "F"),
    "chicago": City("chicago", "Chicago", 41.88, -87.63, "America/Chicago", "F"),
    "miami": City("miami", "Miami", 25.79, -80.32, "America/New_York", "F"),
    "seattle": City("seattle", "Seattle", 47.61, -122.33, "America/Los_Angeles", "F"),
    "toronto": City("toronto", "Toronto", 43.65, -79.38, "America/Toronto", "F"),
    "dallas": City("dallas", "Dallas", 32.78, -96.80, "America/Chicago", "F"),
    "los-angeles": City("los-angeles", "Los Angeles", 33.94, -118.41, "America/Los_Angeles", "F"),
    "denver": City("denver", "Denver", 39.74, -104.99, "America/Denver", "F"),
    "houston": City("houston", "Houston", 29.76, -95.37, "America/Chicago", "F"),
    "austin": City("austin", "Austin", 30.27, -97.74, "America/Chicago", "F"),
    "san-francisco": City("san-francisco", "San Francisco", 37.77, -122.42, "America/Los_Angeles", "F"),
    "atlanta": City("atlanta", "Atlanta", 33.64, -84.43, "America/New_York", "F"),
}

# Backward-compatible London constants (some callers import these directly).
LONDON = CITIES["london"]
LAT, LON, TIMEZONE = LONDON.lat, LONDON.lon, LONDON.timezone


def sigma_floor(unit: str) -> float:
    return SIGMA_FLOOR_F if unit.upper() == "F" else SIGMA_FLOOR_C


# ---------------------------------------------------------------------------
# Bucket parsing  (research)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Bucket:
    """One quoted temperature bucket and its inclusive integer-degree span.

    ``lo``/``hi`` are the *inclusive* integer edges in the market's own unit; an
    open side is ``-inf`` / ``+inf``. A single-degree bucket has ``lo == hi``; a
    US 2-degree bucket ("between 38-39°F") has ``hi == lo + 1``. ``unit`` is
    'C' or 'F'.
    """
    question: str
    lo: float
    hi: float
    label: str
    unit: str = "C"


_RE_RANGE = re.compile(r"between\s+(-?\d+)\s*[-–]\s*(-?\d+)\s*°?\s*([cf])", re.IGNORECASE)
_RE_SINGLE = re.compile(r"(-?\d+)\s*°\s*([cf])", re.IGNORECASE)


def parse_bucket(question: str) -> Optional[Bucket]:
    """Parse a 'highest temperature in <city>' market question into a Bucket.

    Handles both quoting conventions::

        "... be 6°C or below ...?"          → (-inf, 6)   °C, tmax ≤ 6
        "... be 11°C ...?"                  → (11, 11)    °C, tmax == 11
        "... be 14°C or higher ...?"        → (14, +inf)  °C, tmax ≥ 14
        "... be 37°F or below ...?"         → (-inf, 37)  °F, tmax ≤ 37
        "... be between 38-39°F ...?"       → (38, 39)    °F, 2-degree bucket
        "... be 90°F or above ...?"         → (90, +inf)  °F, tmax ≥ 90
    """
    q = question.lower()
    rng = _RE_RANGE.search(question)
    if rng:
        lo, hi = int(rng.group(1)), int(rng.group(2))
        unit = rng.group(3).upper()
        return Bucket(question, float(lo), float(hi), f"{lo}-{hi}", unit)
    m = _RE_SINGLE.search(question)
    if not m:
        return None
    t = int(m.group(1))
    unit = m.group(2).upper()
    if "or below" in q or "or lower" in q or "or colder" in q:
        return Bucket(question, -np.inf, float(t), f"≤{t}", unit)
    if "or higher" in q or "or above" in q or "or hotter" in q or "or more" in q:
        return Bucket(question, float(t), np.inf, f"≥{t}", unit)
    return Bucket(question, float(t), float(t), f"{t}", unit)


def bucket_prob(b: Bucket, mu: float, sigma: float) -> float:
    """P(tmax falls in bucket) under N(μ, σ) with a ±0.5° continuity correction.

    The quoted buckets partition the integers, so the boundary between integer
    ``d`` and ``d+1`` sits at ``d+0.5``. An open bucket integrates the whole tail.
    ``μ``/``σ`` must already be in the bucket's unit.
    """
    sigma = max(float(sigma), sigma_floor(b.unit))
    lo_edge = -np.inf if np.isneginf(b.lo) else (b.lo - 0.5)
    hi_edge = np.inf if np.isposinf(b.hi) else (b.hi + 0.5)
    p_hi = 1.0 if np.isposinf(hi_edge) else norm.cdf((hi_edge - mu) / sigma)
    p_lo = 0.0 if np.isneginf(lo_edge) else norm.cdf((lo_edge - mu) / sigma)
    return float(np.clip(p_hi - p_lo, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Forecast consensus  (data)
# ---------------------------------------------------------------------------
@dataclass
class ForecastConsensus:
    """Point-in-time multi-model max-temperature forecast for one target date."""
    target: str            # YYYY-MM-DD
    mu: float              # cross-model mean tmax (in the city's unit)
    sigma: float           # inter-model std (pre-floor; floored in bucket_prob)
    n_models: int
    unit: str = "C"
    per_model: dict = field(default_factory=dict)

    @property
    def trustworthy(self) -> bool:
        return self.n_models >= MIN_MODELS and np.isfinite(self.mu)


# plausibility window for a daily max, per unit, to drop garbage model values
_SANITY = {"C": (-40.0, 55.0), "F": (-40.0, 131.0)}


def point_in_time_forecast(target: str, city: City = LONDON) -> ForecastConsensus:
    """Five-model archived tmax forecast for ``target`` (YYYY-MM-DD) at ``city``.

    Uses Open-Meteo's *historical-forecast* archive — the forecast that the
    models actually issued for that date — rather than the ERA5 reanalysis, so a
    backtest decision on date ``t`` sees no information from after ``t``. The
    forecast is requested in the city's quoting unit (°C or °F).
    """
    unit_param = "fahrenheit" if city.unit == "F" else "celsius"
    lo_ok, hi_ok = _SANITY[city.unit]
    data = _get(HIST_FORECAST_URL, {
        "latitude": city.lat, "longitude": city.lon,
        "start_date": target, "end_date": target,
        "daily": "temperature_2m_max", "timezone": city.timezone,
        "temperature_unit": unit_param,
        "models": ",".join(MODELS),
    })
    daily = (data or {}).get("daily", {}) if isinstance(data, dict) else {}
    per_model: dict = {}
    for mdl in MODELS:
        vals = daily.get(f"temperature_2m_max_{mdl}")
        if vals and vals[0] is not None and np.isfinite(vals[0]):
            if lo_ok <= float(vals[0]) <= hi_ok:
                per_model[mdl] = float(vals[0])
    if not per_model:
        return ForecastConsensus(target, np.nan, np.nan, 0, city.unit, {})
    arr = np.array(list(per_model.values()))
    return ForecastConsensus(
        target=target,
        mu=float(arr.mean()),
        sigma=float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        n_models=len(per_model),
        unit=city.unit,
        per_model=per_model,
    )


def fetch_token_window(token_id: str, start_ts: int, end_ts: int,
                       fidelity_minutes: int = 60, use_cache: bool = True) -> pd.Series:
    """Hourly Yes-price series for a token over an explicit ``[start_ts, end_ts]`` window.

    The short-lived weather tokens return nothing from the CLOB ``interval=max``
    form that ``fetch.fetch_token_history_raw`` uses, but respond to an explicit
    ``startTs/endTs`` range — so this strategy fetches its own windowed series
    (cached separately under ``data/raw/polymarket/weatherpx/``).
    """
    cache = RAW_DIR / "weatherpx" / f"{token_id}.csv"
    if use_cache and cache.exists():
        s = pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0]
        s.name = token_id
        return s
    data = _get(CLOB_HISTORY_URL, {
        "market": token_id, "startTs": int(start_ts), "endTs": int(end_ts),
        "fidelity": fidelity_minutes,
    })
    history = (data or {}).get("history", []) if isinstance(data, dict) else []
    if not history:
        s = pd.Series(dtype=float, name=token_id)
    else:
        df = pd.DataFrame(history)
        df["ts"] = pd.to_datetime(df["t"], unit="s", utc=True)
        s = df.set_index("ts")["p"].astype(float).sort_index()
        s.name = token_id
    # only cache a real series — an empty result may just be a too-wide window
    if use_cache and not s.empty:
        cache.parent.mkdir(parents=True, exist_ok=True)
        s.to_frame(name="p").to_csv(cache)
    return s


def realized_tmax(target: str, city: City = LONDON) -> Optional[float]:
    """Observed daily max temperature (ERA5 archive) — calibration cross-check only.

    Settlement PnL uses the *market's* resolution, not this; ERA5 is here so the
    backtest can check that the model forecasts were themselves well-calibrated.
    Returned in the city's quoting unit.
    """
    data = _get(ARCHIVE_URL, {
        "latitude": city.lat, "longitude": city.lon,
        "start_date": target, "end_date": target,
        "daily": "temperature_2m_max", "timezone": city.timezone,
        "temperature_unit": "fahrenheit" if city.unit == "F" else "celsius",
    })
    daily = (data or {}).get("daily", {}) if isinstance(data, dict) else {}
    vals = daily.get("temperature_2m_max")
    if vals and vals[0] is not None and np.isfinite(vals[0]):
        return float(vals[0])
    return None


# ---------------------------------------------------------------------------
# Market event  (data)
# ---------------------------------------------------------------------------
def event_slug(d: _date, city: City = LONDON) -> str:
    """The Polymarket event slug for a given city-temperature day."""
    return f"highest-temperature-in-{city.slug}-on-{_MONTHS[d.month - 1]}-{d.day}-{d.year}"


def fetch_weather_event(d: _date, city: City = LONDON) -> Optional[dict]:
    """Fetch one day's city-temperature event from Gamma by dated slug.

    Returns ``{date, city, unit, end_date, buckets:[...]}`` where each bucket
    carries its parsed span, Yes-token id, resolution outcome (0/1 once settled)
    and volume — or ``None`` if the market does not exist for that date.
    """
    evs = _get(GAMMA_EVENTS_URL, {"slug": event_slug(d, city)})
    if not evs or not isinstance(evs, list):
        return None
    ev = evs[0]
    markets = ev.get("markets") or []
    buckets = []
    for m in markets:
        b = parse_bucket(m.get("question", ""))
        if b is None:
            continue
        try:
            token_yes = json.loads(m["clobTokenIds"])[0]
        except (KeyError, ValueError, TypeError, IndexError):
            continue
        outcome = None
        op = m.get("outcomePrices")
        if op:
            try:
                outcome = float(json.loads(op)[0]) if isinstance(op, str) else float(op[0])
            except (ValueError, TypeError, IndexError):
                outcome = None
        vol = m.get("volumeNum")
        if vol is None:
            try:
                vol = float(m.get("volume") or 0.0)
            except (ValueError, TypeError):
                vol = 0.0
        buckets.append({
            "question": b.question, "lo": b.lo, "hi": b.hi, "label": b.label,
            "unit": b.unit, "token_yes": token_yes, "outcome": outcome,
            "volume": float(vol or 0.0), "bucket": b,
        })
    if not buckets:
        return None
    return {
        "date": d.isoformat(),
        "city": city.slug,
        "unit": city.unit,
        "end_date": ev.get("endDate"),
        "closed": bool(ev.get("closed")),
        "volume": float(ev.get("volume") or 0.0),
        "buckets": buckets,
    }


# ---------------------------------------------------------------------------
# Edge construction  (research)
# ---------------------------------------------------------------------------
def compute_edges(buckets: list, consensus: ForecastConsensus,
                  market_prices: dict) -> list:
    """Annotate each bucket with model probability, market price and the edge.

    ``market_prices`` maps token_yes → the decision-time Yes price (mid). Returns
    rows with ``p_model``, ``p_market``, ``edge = p_model − p_market``, plus
    fractional-Kelly and EV for a Yes buy at that price.
    """
    rows = []
    for bk in buckets:
        p_market = market_prices.get(bk["token_yes"])
        if p_market is None or not np.isfinite(p_market):
            continue
        p_model = bucket_prob(bk["bucket"], consensus.mu, consensus.sigma)
        edge = p_model - p_market
        # fractional Kelly for a binary Yes at price p: f* = (p_model − p)/(1 − p)
        denom = max(1e-6, 1.0 - p_market)
        kelly = (p_model - p_market) / denom
        ev = p_model * (1.0 - p_market) - (1.0 - p_model) * p_market  # = edge
        rows.append({
            "date": consensus.target, "label": bk["label"], "question": bk["question"],
            "token_yes": bk["token_yes"], "lo": bk["lo"], "hi": bk["hi"],
            "p_model": p_model, "p_market": float(p_market), "edge": float(edge),
            "kelly": float(kelly), "ev": float(ev),
            "outcome": bk["outcome"], "volume": bk["volume"],
        })
    return rows
