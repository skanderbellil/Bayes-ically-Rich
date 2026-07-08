"""Offline unit tests for the independence-classified smart-flow experiment.

No network calls: `independence_features` and `classify_independence` are pure
functions over synthetic `trades_by_wallet` DataFrames, so every case here is
hand-computed and checked exactly (rounded to the module's own precision).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

smartflow_independence = pytest.importorskip("posterioralpha.polymarket.smartflow_independence")

independence_features = smartflow_independence.independence_features
classify_independence = smartflow_independence.classify_independence
TEMPORAL_SPAN_HOURS = smartflow_independence.TEMPORAL_SPAN_HOURS
PRICE_CHASE_MAX = smartflow_independence.PRICE_CHASE_MAX
JACCARD_MAX = smartflow_independence.JACCARD_MAX

TOKEN = "1234567890"
NOW = datetime.now(timezone.utc).replace(tzinfo=None)


def _trades(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal wallet trades DataFrame from {ts, side, price, asset,
    conditionId} dicts -- only the columns independence_features touches."""
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["ts"])
    df["side"] = df["side"]
    df["price"] = df["price"].astype(float)
    df["asset"] = df["asset"].astype(str)
    df["conditionId"] = df["conditionId"].astype(str)
    return df[["timestamp", "side", "price", "asset", "conditionId"]]


# ---------------------------------------------------------------------------
# independence_features: temporal span + price chase
# ---------------------------------------------------------------------------

def test_span_and_chase_hand_computed():
    """3 buyers, each with exactly one in-window BUY of the token, at known
    times/prices. earliest=w1 (t0, 0.40), latest=w2 (t0+30h, 0.55) -> span=30h,
    chase=0.15. w3's first buy at t0+10h sits strictly between and must not
    affect the earliest/latest selection."""
    t0 = NOW - timedelta(days=3)
    trades_by_wallet = {
        "w1": _trades([{"ts": t0, "side": "BUY", "price": 0.40,
                         "asset": TOKEN, "conditionId": "cA"}]),
        "w2": _trades([{"ts": t0 + timedelta(hours=30), "side": "BUY", "price": 0.55,
                         "asset": TOKEN, "conditionId": "cA"}]),
        "w3": _trades([{"ts": t0 + timedelta(hours=10), "side": "BUY", "price": 0.50,
                         "asset": TOKEN, "conditionId": "cA"}]),
    }
    feats = independence_features(TOKEN, {"w1", "w2", "w3"}, trades_by_wallet, window_days=7)
    assert feats["first_buy_span_hours"] == pytest.approx(30.0, abs=1e-6)
    assert feats["price_chase"] == pytest.approx(0.15, abs=1e-6)


def test_only_earliest_buy_per_wallet_counts():
    """A wallet's SECOND buy at a worse price must not be mistaken for its
    first-buy: earliest BUY-of-token-in-window wins, later fills are ignored."""
    t0 = NOW - timedelta(days=2)
    trades_by_wallet = {
        "w1": _trades([
            {"ts": t0, "side": "BUY", "price": 0.30, "asset": TOKEN, "conditionId": "cA"},
            {"ts": t0 + timedelta(hours=5), "side": "BUY", "price": 0.90,  # later, ignored
             "asset": TOKEN, "conditionId": "cA"},
        ]),
        "w2": _trades([{"ts": t0 + timedelta(hours=48), "side": "BUY", "price": 0.35,
                         "asset": TOKEN, "conditionId": "cB"}]),
    }
    feats = independence_features(TOKEN, {"w1", "w2"}, trades_by_wallet, window_days=7)
    assert feats["first_buy_span_hours"] == pytest.approx(48.0, abs=1e-6)
    assert feats["price_chase"] == pytest.approx(0.05, abs=1e-6)  # 0.35 - 0.30


def test_trades_outside_window_and_wrong_side_or_asset_are_ignored():
    """A BUY before the trailing window, a SELL of the token, and a BUY of a
    DIFFERENT token must all be excluded from first-buy selection."""
    t0 = NOW - timedelta(days=2)
    trades_by_wallet = {
        "w1": _trades([
            {"ts": NOW - timedelta(days=30), "side": "BUY", "price": 0.10,   # too old
             "asset": TOKEN, "conditionId": "cA"},
            {"ts": t0, "side": "SELL", "price": 0.20,                        # wrong side
             "asset": TOKEN, "conditionId": "cA"},
            {"ts": t0, "side": "BUY", "price": 0.99,                        # wrong asset
             "asset": "some_other_token", "conditionId": "cA"},
            {"ts": t0 + timedelta(hours=1), "side": "BUY", "price": 0.42,   # the real first buy
             "asset": TOKEN, "conditionId": "cA"},
        ]),
        "w2": _trades([{"ts": t0 + timedelta(hours=25), "side": "BUY", "price": 0.42,
                         "asset": TOKEN, "conditionId": "cB"}]),
    }
    feats = independence_features(TOKEN, {"w1", "w2"}, trades_by_wallet, window_days=7)
    assert feats["first_buy_span_hours"] == pytest.approx(24.0, abs=1e-6)
    assert feats["price_chase"] == pytest.approx(0.0, abs=1e-6)


def test_asof_replays_point_in_time_window():
    """`asof` lets a caller compute the trailing window from an arbitrary
    historical instant instead of wall-clock `now` -- the backtest replay path.
    A trade that is well within the window relative to a historical `asof` but
    would fall OUTSIDE the trailing window relative to real `now` must be
    picked up when `asof` is passed, and dropped when it is omitted (default
    live behaviour, unchanged)."""
    historical_asof = datetime(2020, 1, 15, 12, 0, 0)  # far in the past
    t0 = historical_asof - timedelta(days=3)  # inside a 7-day window ending at asof
    trades_by_wallet = {
        "w1": _trades([{"ts": t0, "side": "BUY", "price": 0.40,
                         "asset": TOKEN, "conditionId": "cA"}]),
        "w2": _trades([{"ts": t0 + timedelta(hours=30), "side": "BUY", "price": 0.55,
                         "asset": TOKEN, "conditionId": "cA"}]),
    }
    # With asof pinned to the historical instant, both trades fall inside the
    # trailing window and span/chase are computed exactly as hand-computed.
    feats = independence_features(TOKEN, {"w1", "w2"}, trades_by_wallet,
                                   window_days=7, asof=historical_asof)
    assert feats["first_buy_span_hours"] == pytest.approx(30.0, abs=1e-6)
    assert feats["price_chase"] == pytest.approx(0.15, abs=1e-6)

    # Without asof (default -> real now), these 2020 trades are far outside
    # any 7-day trailing window from wall-clock now, so neither buyer has an
    # in-window first buy and both fields default to 0.0.
    feats_live = independence_features(TOKEN, {"w1", "w2"}, trades_by_wallet, window_days=7)
    assert feats_live["first_buy_span_hours"] == 0.0
    assert feats_live["price_chase"] == 0.0


def test_fewer_than_two_first_buys_defaults_to_zero():
    """Defensive path: if fewer than 2 buyers actually have an in-window first
    buy (shouldn't happen once the caller's min_buyers floor is respected),
    span and chase both default to 0.0 rather than raising."""
    t0 = NOW - timedelta(days=1)
    trades_by_wallet = {
        "w1": _trades([{"ts": t0, "side": "BUY", "price": 0.5,
                         "asset": TOKEN, "conditionId": "cA"}]),
        "w2": _trades([{"ts": t0, "side": "SELL", "price": 0.5,   # no BUY -> no first buy
                         "asset": TOKEN, "conditionId": "cB"}]),
    }
    feats = independence_features(TOKEN, {"w1", "w2"}, trades_by_wallet, window_days=7)
    assert feats["first_buy_span_hours"] == 0.0
    assert feats["price_chase"] == 0.0


# ---------------------------------------------------------------------------
# independence_features: pairwise Jaccard over full history conditionIds
# ---------------------------------------------------------------------------

def test_pairwise_jaccard_hand_computed():
    """w1 history {A,B,C}, w2 {A,B,D}, w3 {A,E} (full fetched history, not
    restricted to the window or to this token):
      jaccard(w1,w2) = |{A,B}| / |{A,B,C,D}| = 2/4 = 0.50
      jaccard(w1,w3) = |{A}|   / |{A,B,C,E}| = 1/4 = 0.25
      jaccard(w2,w3) = |{A}|   / |{A,B,D,E}| = 1/4 = 0.25
      mean = (0.50 + 0.25 + 0.25) / 3 = 1/3
    """
    t0 = NOW - timedelta(days=1)

    def _hist(cond_ids):
        return _trades([{"ts": t0, "side": "BUY", "price": 0.5,
                          "asset": f"other_{i}", "conditionId": c}
                         for i, c in enumerate(cond_ids)])

    trades_by_wallet = {
        "w1": _hist(["A", "B", "C"]),
        "w2": _hist(["A", "B", "D"]),
        "w3": _hist(["A", "E"]),
    }
    feats = independence_features(TOKEN, {"w1", "w2", "w3"}, trades_by_wallet, window_days=7)
    assert feats["pairwise_jaccard"] == pytest.approx(1.0 / 3.0, abs=1e-4)


def test_pairwise_jaccard_disjoint_and_identical_extremes():
    """Fully disjoint histories -> jaccard 0.0 (maximally independent-looking);
    identical histories -> jaccard 1.0 (maximally cascade-looking)."""
    t0 = NOW - timedelta(days=1)

    def _hist(cond_ids):
        return _trades([{"ts": t0, "side": "BUY", "price": 0.5,
                          "asset": f"other_{i}", "conditionId": c}
                         for i, c in enumerate(cond_ids)])

    disjoint = {"w1": _hist(["A", "B"]), "w2": _hist(["C", "D"])}
    feats = independence_features(TOKEN, {"w1", "w2"}, disjoint, window_days=7)
    assert feats["pairwise_jaccard"] == pytest.approx(0.0, abs=1e-9)

    identical = {"w1": _hist(["A", "B"]), "w2": _hist(["A", "B"])}
    feats = independence_features(TOKEN, {"w1", "w2"}, identical, window_days=7)
    assert feats["pairwise_jaccard"] == pytest.approx(1.0, abs=1e-9)


def test_pairwise_jaccard_fewer_than_two_buyers_is_zero():
    t0 = NOW - timedelta(days=1)
    trades_by_wallet = {
        "w1": _trades([{"ts": t0, "side": "BUY", "price": 0.5,
                         "asset": TOKEN, "conditionId": "cA"}]),
    }
    feats = independence_features(TOKEN, {"w1"}, trades_by_wallet, window_days=7)
    assert feats["pairwise_jaccard"] == 0.0


def test_pairwise_jaccard_empty_histories_do_not_error():
    """Two buyers with no fetched trade history at all: empty/empty pairs must
    not raise a ZeroDivisionError and should contribute 0 (no shared history)."""
    trades_by_wallet = {
        "w1": pd.DataFrame(columns=["timestamp", "side", "price", "asset", "conditionId"]),
        "w2": pd.DataFrame(columns=["timestamp", "side", "price", "asset", "conditionId"]),
    }
    feats = independence_features(TOKEN, {"w1", "w2"}, trades_by_wallet, window_days=7)
    assert feats["pairwise_jaccard"] == 0.0
    assert feats["first_buy_span_hours"] == 0.0
    assert feats["price_chase"] == 0.0


# ---------------------------------------------------------------------------
# classify_independence: frozen thresholds + 2-of-3 rule
# ---------------------------------------------------------------------------

def test_all_three_pass_gives_independent_score_3():
    feats = {"first_buy_span_hours": 100.0, "price_chase": -0.10, "pairwise_jaccard": 0.0}
    score, cls = classify_independence(feats)
    assert score == 3
    assert cls == "independent"


def test_all_three_fail_gives_cascade_score_0():
    feats = {"first_buy_span_hours": 0.5, "price_chase": 0.50, "pairwise_jaccard": 0.90}
    score, cls = classify_independence(feats)
    assert score == 0
    assert cls == "cascade"


@pytest.mark.parametrize("span,expected_pass", [
    (TEMPORAL_SPAN_HOURS, True),        # boundary: >= passes
    (TEMPORAL_SPAN_HOURS - 1e-6, False),
    (TEMPORAL_SPAN_HOURS + 1e-6, True),
])
def test_temporal_threshold_boundary(span, expected_pass):
    feats = {"first_buy_span_hours": span, "price_chase": 1.0, "pairwise_jaccard": 1.0}
    score, _ = classify_independence(feats)
    assert (score == 1) == expected_pass


@pytest.mark.parametrize("chase,expected_pass", [
    (PRICE_CHASE_MAX, True),            # boundary: <= passes
    (PRICE_CHASE_MAX + 1e-6, False),
    (PRICE_CHASE_MAX - 1e-6, True),
])
def test_price_threshold_boundary(chase, expected_pass):
    feats = {"first_buy_span_hours": 0.0, "price_chase": chase, "pairwise_jaccard": 1.0}
    score, _ = classify_independence(feats)
    assert (score == 1) == expected_pass


@pytest.mark.parametrize("jac,expected_pass", [
    (JACCARD_MAX, True),                # boundary: <= passes
    (JACCARD_MAX + 1e-6, False),
    (JACCARD_MAX - 1e-6, True),
])
def test_comovement_threshold_boundary(jac, expected_pass):
    feats = {"first_buy_span_hours": 0.0, "price_chase": 1.0, "pairwise_jaccard": jac}
    score, _ = classify_independence(feats)
    assert (score == 1) == expected_pass


def test_two_of_three_rule_boundary():
    """Exactly 2 passing components -> independent; exactly 1 -> cascade."""
    two_pass = {"first_buy_span_hours": 100.0, "price_chase": -0.10, "pairwise_jaccard": 0.90}
    score, cls = classify_independence(two_pass)
    assert score == 2
    assert cls == "independent"

    one_pass = {"first_buy_span_hours": 100.0, "price_chase": 0.50, "pairwise_jaccard": 0.90}
    score, cls = classify_independence(one_pass)
    assert score == 1
    assert cls == "cascade"
