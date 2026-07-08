"""Offline unit tests for the smart-flow consensus paper-trade ledger.

No network calls: exercises the `_COLS` schema, `load_ledger`/`save_ledger`
persistence (including the legacy-CSV migration path), and the `buyers`
wallet-set threading through `scan_smart_flow_entries` -- with
`fetch_order_book`/`order_book_features`/`_get` monkeypatched so no HTTP calls
are made.
"""
from __future__ import annotations

import pandas as pd
import pytest

sf = pytest.importorskip("posterioralpha.polymarket.smartflow_papertrade")

_COLS = sf._COLS
load_ledger = sf.load_ledger
save_ledger = sf.save_ledger
scan_smart_flow_entries = sf.scan_smart_flow_entries


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def test_buyers_column_immediately_follows_n_smart_buyers():
    idx = _COLS.index("n_smart_buyers")
    assert _COLS[idx + 1] == "buyers"


# ---------------------------------------------------------------------------
# Ledger persistence: load/save + legacy migration
# ---------------------------------------------------------------------------

def test_load_ledger_missing_file_returns_empty_frame_with_schema(tmp_path):
    df = load_ledger(tmp_path / "does_not_exist.csv")
    assert df.empty
    assert list(df.columns) == _COLS


def test_load_ledger_migrates_legacy_csv_missing_both_bet_fraction_and_buyers(tmp_path):
    """The oldest legacy CSVs predate both `bet_fraction` and `buyers`. Both
    must be backfilled (bet_fraction -> flat 10% default, buyers -> empty)
    with every other legacy value and row preserved, never crashing or
    reordering rows."""
    legacy_cols = [c for c in _COLS if c not in ("bet_fraction", "buyers")]
    rows = [{c: f"r0_{c}" for c in legacy_cols}, {c: f"r1_{c}" for c in legacy_cols}]
    legacy = pd.DataFrame(rows, columns=legacy_cols, dtype=str)
    path = tmp_path / "legacy.csv"
    legacy.to_csv(path, index=False)

    loaded = load_ledger(path)
    assert list(loaded.columns) == _COLS
    assert len(loaded) == 2
    assert loaded["bet_fraction"].tolist() == ["0.1", "0.1"]
    assert loaded["buyers"].tolist() == ["", ""]
    for i, prefix in enumerate(["r0_", "r1_"]):
        for c in legacy_cols:
            assert loaded.loc[i, c] == f"{prefix}{c}"


def test_load_ledger_migrates_legacy_csv_missing_only_buyers(tmp_path):
    """A CSV already migrated to carry `bet_fraction` but written before
    `buyers` existed (e.g. an hourly cron run that already fired against the
    pre-`buyers` schema)."""
    legacy_cols = [c for c in _COLS if c != "buyers"]
    row = {c: f"v_{c}" for c in legacy_cols}
    legacy = pd.DataFrame([row], columns=legacy_cols, dtype=str)
    path = tmp_path / "legacy_buyers_only.csv"
    legacy.to_csv(path, index=False)

    loaded = load_ledger(path)
    assert list(loaded.columns) == _COLS
    assert loaded.loc[0, "buyers"] == ""
    for c in legacy_cols:
        assert loaded.loc[0, c] == f"v_{c}"


def test_buyers_survives_save_load_roundtrip(tmp_path):
    path = tmp_path / "ledger.csv"
    row = {c: f"v_{c}" for c in _COLS}
    row["buyers"] = "0xaaa|0xbbb|0xccc"
    ledger = pd.DataFrame([row], columns=_COLS, dtype=str)

    save_ledger(ledger, path)
    loaded = load_ledger(path)

    assert loaded.loc[0, "buyers"] == "0xaaa|0xbbb|0xccc"
    pd.testing.assert_frame_equal(
        loaded.reset_index(drop=True), ledger.reset_index(drop=True), check_dtype=False,
    )


def test_save_reorders_extra_columns_to_schema(tmp_path):
    """save_ledger persists exactly _COLS, in order, regardless of input order."""
    path = tmp_path / "ledger.csv"
    shuffled = list(reversed(_COLS)) + ["scratch_col"]
    row = {c: f"v_{c}" for c in shuffled}
    save_ledger(pd.DataFrame([row], columns=shuffled, dtype=str), path)

    loaded = load_ledger(path)
    assert list(loaded.columns) == _COLS
    assert loaded.iloc[0].tolist() == [f"v_{c}" for c in _COLS]


# ---------------------------------------------------------------------------
# scan_smart_flow_entries: buyer set threaded through to the candidate dict
# ---------------------------------------------------------------------------

def test_scan_smart_flow_entries_buyers_sorted_pipe_join(monkeypatch):
    flow_index = {
        "tok1": {
            "condition_id": "c1",
            "question": "Will X happen?",
            "slug": "s1",
            "buyers": {"0xCCC", "0xaaa", "0xBbB"},
            "sellers": set(),
        },
    }
    monkeypatch.setattr(sf, "fetch_order_book", lambda tok: object())
    monkeypatch.setattr(sf, "order_book_features", lambda book: {"mid": 0.50, "best_ask": 0.52})
    monkeypatch.setattr(sf, "_get", lambda url, params: [])

    candidates = scan_smart_flow_entries(min_buyers=2, _flow_index=flow_index)

    assert len(candidates) == 1
    # sorted() lexicographic order: uppercase < lowercase in ASCII
    assert candidates[0]["buyers"] == "0xBbB|0xCCC|0xaaa"


def test_scan_smart_flow_entries_buyers_stable_regardless_of_set_build_order(monkeypatch):
    """Rebuilding the same buyer set via a different insertion order must not
    change the serialized string -- sorted() makes the join deterministic
    (byte-stable), so re-runs never spuriously diff the ledger."""
    monkeypatch.setattr(sf, "fetch_order_book", lambda tok: object())
    monkeypatch.setattr(sf, "order_book_features", lambda book: {"mid": 0.50, "best_ask": 0.52})
    monkeypatch.setattr(sf, "_get", lambda url, params: [])

    def _flow(order):
        return {
            "tok1": {
                "condition_id": "c1", "question": "Q", "slug": "s1",
                "buyers": set(order), "sellers": set(),
            },
        }

    c1 = scan_smart_flow_entries(min_buyers=2, _flow_index=_flow(["0xaaa", "0xbbb", "0xccc"]))
    c2 = scan_smart_flow_entries(min_buyers=2, _flow_index=_flow(["0xccc", "0xaaa", "0xbbb"]))
    assert c1[0]["buyers"] == c2[0]["buyers"] == "0xaaa|0xbbb|0xccc"
