"""Offline unit tests for the passive-entry smart-flow experiment.

No network: `smart_pool`, `_fetch_pool_trades`, `_flow_index_from`,
`scan_smart_flow_entries`, `_price` and `fetch_market_resolution` are all
monkeypatched, so every case here exercises the module's own order lifecycle
(post -> fill / expire -> resolve) against hand-built books.

The fill model is the whole experiment, so it is pinned hard: a resting bid at
the mid may fill ONLY when the offer comes down to it, never merely because the
mid drifted, and an order that is never reached must expire `unfilled` rather
than quietly becoming a position at a price we did not get.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

sfp = pytest.importorskip("posterioralpha.polymarket.smartflow_passive")

MAX_HALF_SPREAD = sfp.MAX_HALF_SPREAD
WORK_HOURS = sfp.WORK_HOURS
MIN_RESOLVED = sfp.MIN_RESOLVED
MIN_FILL_RATE = sfp.MIN_FILL_RATE
MIN_WORKED = sfp.MIN_WORKED
KILL_T_MIN = sfp.KILL_T_MIN
_COLS = sfp._COLS

T0 = datetime(2026, 9, 19, 12, 0, 0)


def _candidate(token="tok1", mid=0.50, ask=0.51, buyers=3):
    return {"token": token, "condition_id": f"cond_{token}", "question": f"q {token}",
            "domain": "other", "end_date": "2026-12-31", "n_smart_buyers": buyers,
            "buyers": "0xa|0xb|0xc", "entry_mid": mid, "entry_ask": ask,
            "spread": round(ask - mid, 4)}


def _patch(monkeypatch, candidates, price_fn, resolution_fn=lambda cid: None):
    monkeypatch.setattr(sfp, "smart_pool", lambda *a, **k: [])
    monkeypatch.setattr(sfp, "_fetch_pool_trades", lambda *a, **k: {})
    monkeypatch.setattr(sfp, "_flow_index_from", lambda *a, **k: {})
    monkeypatch.setattr(sfp, "scan_smart_flow_entries", lambda *a, **k: list(candidates))
    monkeypatch.setattr(sfp, "_price", price_fn)
    monkeypatch.setattr(sfp, "fetch_market_resolution", resolution_fn)


# ---------------------------------------------------------------------------
# Spread cap
# ---------------------------------------------------------------------------

def test_candidates_over_the_half_spread_cap_are_never_posted(tmp_path, monkeypatch):
    wide = _candidate("wide", mid=0.50, ask=0.50 + MAX_HALF_SPREAD + 0.001)
    tight = _candidate("tight", mid=0.50, ask=0.50 + MAX_HALF_SPREAD)
    _patch(monkeypatch, [wide, tight], lambda tok: (0.50, 0.99))
    led = sfp.update_ledger(state_file=tmp_path / "l.csv", now=T0)
    assert set(led["token"]) == {"tight"}, "the cap is inclusive at exactly MAX_HALF_SPREAD"


def test_the_spread_cap_is_what_makes_this_differ_from_the_incumbent(tmp_path, monkeypatch):
    """Sanity: with no wide candidates the cap must not drop anything."""
    cands = [_candidate(f"t{i}", mid=0.40, ask=0.41) for i in range(5)]
    _patch(monkeypatch, cands, lambda tok: (0.40, 0.99))
    led = sfp.update_ledger(state_file=tmp_path / "l.csv", now=T0)
    assert len(led) == 5 and set(led["status"]) == {"working"}


# ---------------------------------------------------------------------------
# Fill model
# ---------------------------------------------------------------------------

def test_a_new_order_rests_and_does_not_fill_at_the_ask(tmp_path, monkeypatch):
    """The incumbent's bug in one line: we must NOT pay the offer on arrival."""
    _patch(monkeypatch, [_candidate(mid=0.50, ask=0.51)], lambda tok: (0.50, 0.51))
    led = sfp.update_ledger(state_file=tmp_path / "l.csv", now=T0)
    assert led.loc[0, "status"] == "working"
    assert led.loc[0, "fill_px"] == "" and led.loc[0, "entry_date"] == ""


def test_fill_only_when_the_offer_reaches_our_limit(tmp_path, monkeypatch):
    path = tmp_path / "l.csv"
    _patch(monkeypatch, [_candidate(mid=0.50, ask=0.51)], lambda tok: (0.50, 0.51))
    sfp.update_ledger(state_file=path, now=T0)

    # offer drops to exactly our limit -> we are hit, at our limit, not at the ask
    _patch(monkeypatch, [], lambda tok: (0.495, 0.50))
    led = sfp.update_ledger(state_file=path, now=T0 + timedelta(hours=1))
    assert led.loc[0, "status"] == "open"
    assert float(led.loc[0, "fill_px"]) == pytest.approx(0.50)


def test_a_falling_mid_alone_does_not_fill_us(tmp_path, monkeypatch):
    """Marking down is not a fill. Only the offer crossing our bid is."""
    path = tmp_path / "l.csv"
    _patch(monkeypatch, [_candidate(mid=0.50, ask=0.51)], lambda tok: (0.50, 0.51))
    sfp.update_ledger(state_file=path, now=T0)

    # mid drops below our limit but the offer stays above it (spread widened)
    _patch(monkeypatch, [], lambda tok: (0.48, 0.505))
    led = sfp.update_ledger(state_file=path, now=T0 + timedelta(hours=1))
    assert led.loc[0, "status"] == "working"
    assert float(led.loc[0, "current_price"]) == pytest.approx(0.48), "still marked"


def test_unreached_orders_expire_unfilled_after_the_working_window(tmp_path, monkeypatch):
    path = tmp_path / "l.csv"
    _patch(monkeypatch, [_candidate(mid=0.50, ask=0.51)], lambda tok: (0.50, 0.51))
    sfp.update_ledger(state_file=path, now=T0)

    _patch(monkeypatch, [], lambda tok: (0.50, 0.51))
    still = sfp.update_ledger(state_file=path, now=T0 + timedelta(hours=WORK_HOURS - 1))
    assert still.loc[0, "status"] == "working"

    led = sfp.update_ledger(state_file=path, now=T0 + timedelta(hours=WORK_HOURS + 0.5))
    assert led.loc[0, "status"] == "unfilled"
    assert led.loc[0, "fill_px"] == "", "an expired order must not carry a fill price"


def test_an_unfilled_order_never_becomes_a_position(tmp_path, monkeypatch):
    """Regression guard: `unfilled` must stay terminal even if the book later
    trades through our old limit — we cancelled, so that print is not ours."""
    path = tmp_path / "l.csv"
    _patch(monkeypatch, [_candidate(mid=0.50, ask=0.51)], lambda tok: (0.50, 0.51))
    sfp.update_ledger(state_file=path, now=T0)
    _patch(monkeypatch, [], lambda tok: (0.50, 0.51))
    sfp.update_ledger(state_file=path, now=T0 + timedelta(hours=WORK_HOURS + 1))

    _patch(monkeypatch, [], lambda tok: (0.20, 0.21))     # blows through the old limit
    led = sfp.update_ledger(state_file=path, now=T0 + timedelta(hours=WORK_HOURS + 2))
    assert led.loc[0, "status"] == "unfilled"
    assert led.loc[0, "pnl"] == ""


def test_a_token_already_in_the_ledger_is_not_re_posted(tmp_path, monkeypatch):
    path = tmp_path / "l.csv"
    cand = _candidate(mid=0.50, ask=0.51)
    _patch(monkeypatch, [cand], lambda tok: (0.50, 0.51))
    sfp.update_ledger(state_file=path, now=T0)
    led = sfp.update_ledger(state_file=path, now=T0 + timedelta(hours=1))
    assert len(led) == 1, "the same consensus token was posted twice"


# ---------------------------------------------------------------------------
# PnL is computed off the FILL price, not the scan ask
# ---------------------------------------------------------------------------

def test_pnl_uses_the_fill_price_not_the_ask_we_declined_to_pay(tmp_path, monkeypatch):
    path = tmp_path / "l.csv"
    _patch(monkeypatch, [_candidate(mid=0.50, ask=0.51)], lambda tok: (0.50, 0.51))
    sfp.update_ledger(state_file=path, now=T0)
    _patch(monkeypatch, [], lambda tok: (0.495, 0.50))
    sfp.update_ledger(state_file=path, now=T0 + timedelta(hours=1))

    _patch(monkeypatch, [], lambda tok: (1.0, 1.0),
           resolution_fn=lambda cid: {"closed": True,
                                      "tokens": {"tok1": {"winner": True, "price": 1.0}}})
    led = sfp.update_ledger(state_file=path, now=T0 + timedelta(hours=2))
    assert led.loc[0, "status"] == "won"
    # filled at 0.50, not the 0.51 ask: (1/0.50 - 1) * 0.10 = 0.10, vs 0.0980 at the ask
    assert float(led.loc[0, "pnl"]) == pytest.approx((1.0 / 0.50 - 1.0) * 0.10, abs=1e-4)


# ---------------------------------------------------------------------------
# fill_rate / kill_check
# ---------------------------------------------------------------------------

def _led(rows: list[dict]) -> pd.DataFrame:
    recs = []
    for r in rows:
        base = {c: "" for c in _COLS}
        base.update(r)
        recs.append(base)
    return pd.DataFrame(recs, columns=_COLS, dtype=str)


def test_fill_rate_ignores_orders_still_working():
    led = _led([{"status": "open"}] * 3 + [{"status": "unfilled"}] * 1
               + [{"status": "working"}] * 96)
    fr = sfp.fill_rate(led)
    assert fr["worked"] == 4 and fr["filled"] == 3
    assert fr["rate"] == pytest.approx(0.75), "working orders must not count as misses"


def test_kill_check_is_not_armed_before_the_sample_floor():
    led = _led([{"status": "won", "fill_px": "0.5", "outcome": "1.0",
                 "exit_date": f"2026-09-{d:02d}"} for d in range(1, 11)])
    r = sfp.kill_check(led)
    assert r["armed"] is False and r["verdict"] == "insufficient data"


def test_kill_check_fires_on_a_fill_rate_floor_breach_even_with_no_resolutions():
    """Never getting filled is its own way to fail, independent of edge."""
    led = _led([{"status": "unfilled"}] * MIN_WORKED)
    r = sfp.kill_check(led)
    assert r["fill_rate"] == pytest.approx(0.0)
    assert r["viability_failed"] is True
    assert r["armed"] is True and r["verdict"] == "RETIRED"


def test_kill_check_clusters_by_settlement_day_not_by_trade():
    """120 correlated fills settling on one day are one event, not 120."""
    rows = [{"status": "won", "fill_px": "0.5", "outcome": "1.0", "exit_date": "2026-09-01"}
            for _ in range(120)]
    rows += [{"status": "lost", "fill_px": "0.5", "outcome": "0.0", "exit_date": "2026-09-02"}
             for _ in range(120)]
    r = sfp.kill_check(_led(rows))
    assert r["events"] == 2, "the t must be computed on settlement days, not trades"


def test_kill_check_retires_a_profitable_looking_book_with_no_clustered_evidence():
    """Edge positive per trade but only two settlement days -> t below the bar."""
    rows = []
    for d, (out, n) in enumerate([("1.0", 150), ("0.0", 60)], start=1):
        rows += [{"status": "won" if out == "1.0" else "lost", "fill_px": "0.5",
                  "outcome": out, "exit_date": f"2026-09-{d:02d}"} for _ in range(n)]
    r = sfp.kill_check(_led(rows))
    assert r["n_resolved"] >= MIN_RESOLVED and r["armed"] is True
    assert r["t"] < KILL_T_MIN and r["verdict"] == "RETIRED"


def test_kill_check_leaves_a_consistently_positive_book_alive():
    rows = []
    for d in range(1, 31):
        # every settlement day is modestly positive: fill 0.50, 60% winners
        rows += [{"status": "won", "fill_px": "0.50", "outcome": "1.0",
                  "exit_date": f"2026-09-{d:02d}"} for _ in range(6)]
        rows += [{"status": "lost", "fill_px": "0.50", "outcome": "0.0",
                  "exit_date": f"2026-09-{d:02d}"} for _ in range(4)]
    rows += [{"status": "open"}] * 200          # keeps the fill rate above the floor
    r = sfp.kill_check(_led(rows))
    assert r["armed"] is True and r["t"] > KILL_T_MIN
    assert r["verdict"] == "alive"
