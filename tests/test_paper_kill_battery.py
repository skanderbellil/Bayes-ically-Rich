"""Offline unit tests for the paper-sleeve kill battery.

The battery decides which live sleeves may claim capital, so its job is to be
hard to pass. These tests pin the properties that make it meaningful: a real
edge with enough independent evidence gets through, noise does not, and — the
one that matters most on these books — thousands of trades settling off a
handful of events cannot manufacture significance.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
kb = pytest.importorskip("run_paper_kill_battery")


class Args:
    """Stand-in for the argparse namespace `assess` consumes."""
    def __init__(self, **kw):
        self.min_events = kw.get("min_events", kb.MIN_EVENTS)
        self.t_min = kw.get("t_min", kb.T_MIN)
        self.oos_frac = kw.get("oos_frac", kb.OOS_FRAC)
        self.haircut = kw.get("haircut", kb.HAIRCUT)
        self.boot_p = kw.get("boot_p", kb.BOOT_P)
        self.top1_max = kw.get("top1_max", kb.TOP1_MAX)


def _t(day, entry, outcome, token="t"):
    return {"token": token, "ed": day, "xd": day, "entry": entry,
            "outcome": float(outcome), "current": None, "resolved": True, "q": ""}


def _days(n, start=1):
    """n distinct settlement dates spread across 2026."""
    return [f"2026-{1 + (i // 28) % 12:02d}-{1 + i % 28:02d}" for i in range(start, start + n)]


def test_event_edges_collapses_same_day_trades_to_one_observation():
    trades = [_t("2026-06-01", 0.5, 1.0), _t("2026-06-01", 0.5, 0.0),
              _t("2026-06-02", 0.5, 1.0)]
    days, ev = kb.event_edges(trades)
    assert days == ["2026-06-01", "2026-06-02"]
    assert ev[0] == pytest.approx(0.0)     # +0.5 and -0.5 average out
    assert ev[1] == pytest.approx(0.5)


def test_correlated_same_day_trades_cannot_manufacture_significance():
    """2,000 trades off 4 events must not look like 2,000 independent bets.

    This is the exact shape of the real Smart Flow book (2,119 trades, 23
    events); counting trades as independent is what made a dead sleeve look
    like it had thousands of confirmations.
    """
    trades = []
    for d in _days(4):
        for _ in range(500):
            trades.append(_t(d, 0.5, 1.0 if d == _days(4)[0] else 0.0))
    r = kb.assess("correlated", trades, mc_bar=2.9, args=Args())
    assert r["events"] == 4
    assert r["trades"] == 2000
    assert r["verdict"] == "KILLED"
    assert any("events>=" in x for x in r["reasons"])


def test_genuine_edge_with_enough_events_is_validated():
    """A large, consistent, well-spread edge must be able to pass — otherwise
    the battery is unfalsifiable rather than strict."""
    trades = [_t(d, 0.30, 1.0 if i % 4 else 0.0, token=f"t{i}")
              for i, d in enumerate(_days(120))]
    r = kb.assess("real edge", trades, mc_bar=2.9, args=Args())
    assert r["edge"] > 0
    assert r["verdict"] == "VALIDATED", r["reasons"]


def test_pure_noise_is_killed():
    rng = np.random.default_rng(7)
    trades = [_t(d, 0.50, float(rng.integers(0, 2)), token=f"t{i}")
              for i, d in enumerate(_days(120))]
    r = kb.assess("noise", trades, mc_bar=2.9, args=Args())
    assert r["verdict"] == "KILLED"


def test_concentration_check_kills_a_one_day_wonder():
    """An edge carried entirely by a single settlement day must not pass."""
    days = _days(60)
    trades = [_t(d, 0.10, 0.0, token=f"t{i}") for i, d in enumerate(days[:-1])]
    trades.append(_t(days[-1], 0.01, 1.0, token="jackpot"))
    r = kb.assess("one-day wonder", trades, mc_bar=2.9, args=Args())
    assert r["top1"] == pytest.approx(1.0)
    assert r["verdict"] == "KILLED"
    assert any("top1" in x for x in r["reasons"])


def test_multiple_comparison_bar_rises_with_candidate_count():
    """Testing more sleeves must make the winner's bar higher, not lower."""
    bars = [kb.mc_threshold(k, sims=20_000) for k in (1, 5, 15, 40)]
    assert bars == sorted(bars)
    assert bars[0] == pytest.approx(1.96, abs=0.15)   # single test -> ordinary 95%
    assert bars[2] > 2.5                              # 15 candidates -> materially stricter


def test_verdict_requires_every_check():
    """One failing check is enough to kill, regardless of the others."""
    trades = [_t(d, 0.30, 1.0 if i % 4 else 0.0, token=f"t{i}")
              for i, d in enumerate(_days(120))]
    passing = kb.assess("s", trades, mc_bar=2.9, args=Args())
    assert passing["verdict"] == "VALIDATED"
    # same data, but the selection-luck bar is raised above its t
    killed = kb.assess("s", trades, mc_bar=999.0, args=Args())
    assert killed["verdict"] == "KILLED"
    assert any("mc_bar" in x for x in killed["reasons"])


def test_haircut_is_incremental_not_a_full_spread():
    """Entry is recorded at the ask, so the haircut models only slippage beyond
    the touch. A haircut far above the observed 0.005 spread would kill every
    sleeve by assumption."""
    assert kb.HAIRCUT == pytest.approx(0.005)
    assert kb.STRESS_HAIRCUT > kb.HAIRCUT
