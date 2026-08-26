"""Offline unit tests for the champion stack's price download.

Regression cover for the 2026-08-24 outage: Yahoo kept returning a ``^VIX3M``
column but with a single non-NaN close out of ~2,500 rows. The column was
present, so the existing missing-ticker check passed; the dropna-intersection
then collapsed to one row and the champion step failed on every hourly run for
two days. No network here — yfinance and the CBOE fetch are both stubbed.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
cl = pytest.importorskip("posterioralpha.research.champion_live")

FIELDS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def _raw_frame(n=1500, vix3m_valid=None):
    """A yfinance group_by='ticker' style frame; vix3m_valid=1 reproduces the
    outage (column present, one usable close)."""
    idx = pd.bdate_range("2020-01-01", periods=n)
    cols, data = [], {}
    for tkr in cl.TICKERS:
        for f in FIELDS:
            cols.append((tkr, f))
            series = np.linspace(10.0, 20.0, n)
            if tkr == "^VIX3M" and f == "Close" and vix3m_valid is not None:
                series = np.full(n, np.nan)
                series[-vix3m_valid:] = 18.5
            data[(tkr, f)] = series
    return pd.DataFrame(data, index=idx, columns=pd.MultiIndex.from_tuples(cols))


@pytest.fixture
def stub_yf(monkeypatch):
    """Install a fake yfinance module whose download() returns `holder['raw']`."""
    holder = {}
    mod = types.ModuleType("yfinance")
    mod.download = lambda *a, **k: holder["raw"]
    monkeypatch.setitem(sys.modules, "yfinance", mod)
    return holder


def test_healthy_download_needs_no_repair(stub_yf, monkeypatch):
    stub_yf["raw"] = _raw_frame()
    called = []
    monkeypatch.setattr(cl, "_vix3m_from_cboe", lambda: called.append(1) or pd.Series(dtype=float))
    out = cl._download_prices()
    assert not called, "CBOE must not be hit when Yahoo's data is complete"
    assert not out.isna().any().any()
    assert list(out.columns) == ["QQQ", "QQQ_Open", "VIX", "VIX3M", "HYG", "IEF", "UUP", "QLD"]


def test_empty_vix3m_column_is_repaired_from_cboe(stub_yf, monkeypatch):
    """The exact outage shape: column present, one usable close."""
    raw = _raw_frame(vix3m_valid=1)
    stub_yf["raw"] = raw
    idx = raw.index
    monkeypatch.setattr(cl, "_vix3m_from_cboe",
                        lambda: pd.Series(np.linspace(15.0, 25.0, len(idx)), index=idx))
    out = cl._download_prices()
    assert len(out) >= cl.HMM_WIN + 252
    assert out["VIX3M"].notna().all()


def test_repair_failure_names_the_broken_column(stub_yf, monkeypatch):
    """If the fallback is also down, the error must say WHICH feed is thin —
    a bare 'got 1 complete-data days' is what made this take two days to spot."""
    stub_yf["raw"] = _raw_frame(vix3m_valid=1)

    def boom():
        raise RuntimeError("cboe unreachable")

    monkeypatch.setattr(cl, "_vix3m_from_cboe", boom)
    monkeypatch.setattr("time.sleep", lambda *_: None)  # retry backoff is 20s+40s
    with pytest.raises(RuntimeError) as ei:
        cl._download_prices()
    msg = str(ei.value)
    assert "VIX3M" in msg
    assert "under-covered" in msg


def test_missing_ticker_still_raises(stub_yf, monkeypatch):
    """A genuinely absent ticker must not be silently repaired."""
    raw = _raw_frame().drop(columns=["HYG"], level=0)
    stub_yf["raw"] = raw
    monkeypatch.setattr(cl, "_vix3m_from_cboe", lambda: pd.Series(dtype=float))
    monkeypatch.setattr("time.sleep", lambda *_: None)  # retry backoff is 20s+40s
    with pytest.raises(RuntimeError, match="missing tickers"):
        cl._download_prices()


def test_cboe_parser_handles_the_published_schema(monkeypatch):
    """CBOE ships DATE/OPEN/HIGH/LOW/CLOSE; the parser must yield a clean,
    sorted, tz-naive close series."""
    csv = "DATE,OPEN,HIGH,LOW,CLOSE\n2026-08-25,18.0,18.9,17.8,18.21\n2026-08-24,18.4,18.8,18.1,18.56\n"

    class R:
        status_code = 200
        text = csv

        def raise_for_status(self):
            pass

    monkeypatch.setitem(sys.modules, "requests",
                        types.SimpleNamespace(get=lambda *a, **k: R()))
    s = cl._vix3m_from_cboe()
    assert list(s.index.strftime("%Y-%m-%d")) == ["2026-08-24", "2026-08-25"], "must be sorted"
    assert s.index.tz is None
    assert s.iloc[-1] == pytest.approx(18.21)
