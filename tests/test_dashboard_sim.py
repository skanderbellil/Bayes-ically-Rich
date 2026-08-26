"""Offline unit tests for the dashboard's cash simulation (`generate_dashboard.sim`).

Regression cover for the capital-lock bug: roughly half of every Polymarket
ledger's resolved trades enter AND settle on the same date (sports markets), and
`sim` used to run each day's sell pass strictly before its buy pass. A same-day
round-trip was therefore "sold" before it was ever held, so its stake stayed in
`hold` forever. Six such trades pinned every sleeve's cash at $0 in early July
2026 and froze the whole dashboard at a fixed equity for over a month.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
gd = pytest.importorskip("generate_dashboard")

CAP0 = gd.CAP0


def _trade(ed, xd, entry, outcome, token="t"):
    """A resolved trade in the shape `load_ledger` produces."""
    return {"token": token, "ed": ed, "xd": xd, "entry": entry,
            "outcome": float(outcome), "current": None, "resolved": True, "q": ""}


def test_same_day_round_trip_releases_its_capital():
    """A trade entered and settled on one date must not stay held afterwards."""
    trades = [_trade("2026-06-26", "2026-06-26", 0.50, 1.0)]
    r = gd.sim(trades, "flat", {})
    assert r["taken"] == 1
    # won at 0.50 -> $10 stake returns $20, so +$10 realized and nothing held
    assert r["realized"] == pytest.approx(10.0)
    assert r["fin"] == pytest.approx(CAP0 + 10.0)


def test_same_day_loss_is_realized_not_parked_at_cost():
    """The old bug hid losses by leaving the position held and marked at entry."""
    trades = [_trade("2026-06-26", "2026-06-26", 0.50, 0.0)]
    r = gd.sim(trades, "flat", {})
    assert r["realized"] == pytest.approx(-10.0)
    assert r["fin"] == pytest.approx(CAP0 - 10.0)


def test_same_day_round_trips_do_not_exhaust_cash():
    """Many same-day trades used to lock the full bankroll and freeze the curve.

    Sized at 20% of equity each, 40 sequential same-day round-trips would pin
    cash at $0 within the first handful under the old ordering; every later
    entry was then skipped and the equity curve went flat forever.
    """
    trades = [_trade(f"2026-06-{d:02d}", f"2026-06-{d:02d}", 0.50, 1.0 if d % 2 else 0.0)
              for d in range(1, 29)]
    fracs = {i: 0.20 for i in range(len(trades))}
    r = gd.sim(trades, "pct", {}, fracs=fracs)
    assert r["taken"] == len(trades), "every sized trade should be affordable"
    # Entering at 0.50 doubles the stake on a win and loses it on a loss, so a
    # 20% stake compounds *1.2 / *0.8. Over 14 of each the closed form is exact
    # — it only holds if every round-trip actually returned its capital.
    # `fin` comes off the equity curve, which is rounded to cents.
    assert r["fin"] == pytest.approx(CAP0 * (1.2 * 0.8) ** 14, abs=0.01)
    assert r["maxconc"] == 1, "same-day round-trips never overlap"


def test_multi_day_hold_still_ties_up_cash_until_exit():
    """The fix must not let a genuinely open multi-day position settle early."""
    trades = [_trade("2026-06-01", "2026-06-20", 0.50, 1.0),
              _trade("2026-06-02", "2026-06-03", 0.50, 0.0)]
    fracs = {0: 0.90, 1: 0.90}
    # max_deploy=None isolates the cash constraint from the gross-exposure cap
    r = gd.sim(trades, "pct", {}, fracs=fracs, max_deploy=None)
    # trade 0 holds 90% of the bankroll through 06-20, so trade 1 on 06-02 is
    # funded only from what little cash is left — capital-constrained, not free
    assert r["constrained"] >= 1
    assert r["maxconc"] == 2


def test_gross_exposure_cap_limits_concurrent_deployment():
    """Ten positions open together, each sized at 20% of equity, must not put
    200% of the bankroll at risk — the cap holds gross exposure at max_deploy."""
    trades = [_trade("2026-06-01", "2026-07-01", 0.50, 1.0, token=f"t{i}")
              for i in range(10)]
    fracs = {i: 0.20 for i in range(10)}
    capped = gd.sim(trades, "pct", {}, fracs=fracs, max_deploy=0.30)
    assert capped["peakdep"] == pytest.approx(0.30, abs=0.02)
    uncapped = gd.sim(trades, "pct", {}, fracs=fracs, max_deploy=None)
    assert uncapped["peakdep"] > 0.9, "without the cap this deploys ~everything"


def test_gross_exposure_cap_prevents_ruin_on_a_losing_run():
    """The failure mode the cap exists for: concurrent same-day losers at full
    Kelly wiped four real sleeves to $0. Capped, the bankroll must survive."""
    trades = [_trade(f"2026-06-{d:02d}", f"2026-06-{d:02d}", 0.50, 0.0, token=f"t{d}")
              for d in range(1, 11)]
    fracs = {i: 0.50 for i in range(len(trades))}
    uncapped = gd.sim(trades, "pct", {}, fracs=fracs, max_deploy=None)
    capped = gd.sim(trades, "pct", {}, fracs=fracs, max_deploy=0.30)
    assert uncapped["fin"] < capped["fin"]
    # 10 straight losses at 30% gross exposure still leaves a live bankroll
    assert capped["fin"] > 0.0


def test_exposure_cap_is_causal():
    """The cap may only read equity and current holdings — never future data.
    Truncating the trade list must not change how earlier trades were sized."""
    trades = [_trade(f"2026-06-{d:02d}", f"2026-06-{d+5:02d}", 0.50, 1.0, token=f"t{d}")
              for d in range(1, 9)]
    fracs = {i: 0.25 for i in range(len(trades))}
    full = gd.sim(trades, "pct", {}, fracs=fracs, max_deploy=0.30)
    short = gd.sim(trades[:4], "pct", {}, fracs={i: 0.25 for i in range(4)}, max_deploy=0.30)
    assert short["stakes"], "the truncated run must still take something"
    for idx, s in short["stakes"].items():
        assert full["stakes"][idx]["stake"] == pytest.approx(s["stake"]), (
            f"trade {idx} was sized differently once later trades existed")


def test_prior_day_settlement_funds_the_same_days_entries():
    """Sells still precede buys, so today's proceeds can fund today's entries."""
    trades = [_trade("2026-06-01", "2026-06-10", 0.50, 1.0),
              _trade("2026-06-10", "2026-06-11", 0.50, 1.0)]
    fracs = {0: 1.0, 1: 1.0}
    # max_deploy=None so the full bankroll can be staked, isolating the ordering
    r = gd.sim(trades, "pct", {}, fracs=fracs, max_deploy=None)
    # trade 0 consumes the whole bankroll and doubles it on 06-10; trade 1 can
    # only be entered that same day if the sell pass ran first
    assert r["taken"] == 2
    assert r["fin"] == pytest.approx(4 * CAP0)
