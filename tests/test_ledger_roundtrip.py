"""Roundtrip test for the polymarket paper-trade ledger persistence layer.

Deliberately narrow: only load_ledger/save_ledger and the _COLS schema are
exercised (the module's resolution/trading logic is under active development).
STATE_FILE is monkeypatched to a tmp_path so nothing in the repo is touched.
"""
import pandas as pd
import pytest

pt = pytest.importorskip("posterioralpha.polymarket.papertrade")


@pytest.fixture(autouse=True)
def _isolated_state_file(tmp_path, monkeypatch):
    if not hasattr(pt, "STATE_FILE"):
        pytest.skip("papertrade.STATE_FILE not present in this revision")
    monkeypatch.setattr(pt, "STATE_FILE", tmp_path / "sub" / "ledger.csv")


def _cols():
    cols = getattr(pt, "_COLS", None)
    if not cols:
        pytest.skip("papertrade._COLS not present in this revision")
    return list(cols)


def test_load_ledger_missing_file_returns_empty_frame_with_schema():
    df = pt.load_ledger()
    assert isinstance(df, pd.DataFrame)
    assert df.empty
    assert list(df.columns) == _cols()


def test_save_then_load_roundtrips_unchanged():
    cols = _cols()
    ledger = pd.DataFrame(
        [
            {c: f"row0_{c}" for c in cols},
            {c: f"row1_{c}" for c in cols},
        ],
        columns=cols,
        dtype=str,
    )

    pt.save_ledger(ledger)
    assert pt.STATE_FILE.exists(), "save_ledger should create parent dirs + file"

    loaded = pt.load_ledger()
    assert list(loaded.columns) == cols
    pd.testing.assert_frame_equal(
        loaded.reset_index(drop=True), ledger.reset_index(drop=True),
        check_dtype=False,
    )


def test_save_reorders_extra_columns_to_schema():
    """save_ledger persists exactly _COLS, in order, regardless of input order."""
    cols = _cols()
    shuffled = list(reversed(cols)) + ["scratch_col"]
    row = {c: f"v_{c}" for c in shuffled}
    pt.save_ledger(pd.DataFrame([row], columns=shuffled, dtype=str))

    loaded = pt.load_ledger()
    assert list(loaded.columns) == cols
    assert loaded.iloc[0].tolist() == [f"v_{c}" for c in cols]
