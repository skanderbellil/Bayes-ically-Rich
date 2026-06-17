#!/usr/bin/env python3
"""
Smart-flow paper trade: risk management backtest
=================================================

Compares four risk-management strategies on the current open positions using
Monte Carlo simulation. Since the paper trade has only been live for 2 days
with no resolved positions yet, a historical backtest is not possible. Instead
we use each position's current mid price as the market-implied Bernoulli win
probability and draw 50 000 outcome scenarios.

Strategies compared
-------------------
  baseline    : flat 10% fraction, hold to resolution  (current live system)
  kelly       : n_buyers-scaled fraction, hold to resolution
  stops       : flat 10% fraction + stop-loss at 35% drawdown from entry
  kelly+stops : n_buyers-scaled fraction + stop-loss

Stop-loss approximation
-----------------------
Without a price time-series we cannot simulate whether a position ever touched
the stop threshold mid-life. The simulation checks whether the *current* price
already breaches the threshold: if so, it locks in a realised PnL at that
price. Positions not yet at threshold resolve as Bernoulli(current_price).
This is conservative — it may undercount stops for positions that dipped and
recovered.

Correlation caveat
------------------
Several positions share the same condition_id (e.g. two outcome tokens for the
same "Iraq vs. Norway" market). Those outcomes are perfectly anti-correlated, not
independent. The simulation treats them as independent, which overstates
diversification and understates tail risk. Results should be read directionally
rather than as precise risk estimates.

Usage
-----
  python experiments/run_smart_flow_backtest.py
  python experiments/run_smart_flow_backtest.py --stop-loss 0.40 --n-sims 100000
  python experiments/run_smart_flow_backtest.py --no-plots
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tabulate import tabulate

from posterioralpha.polymarket.smartflow_papertrade import STATE_FILE, kelly_fraction
from posterioralpha.polymarket.paths import RESULTS_DIR, ensure_dirs

STRATEGIES: dict[str, dict] = {
    "baseline":    {"use_kelly": False, "stop_loss": None},
    "kelly":       {"use_kelly": True,  "stop_loss": None},
    "stops":       {"use_kelly": False, "stop_loss": 0.35},
    "kelly+stops": {"use_kelly": True,  "stop_loss": 0.35},
}

_COLORS = ["#888888", "#2196F3", "#FF9800", "#4CAF50"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_positions(path: Path = STATE_FILE) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    open_df = df[df["status"] == "open"].copy()
    n_closed = (df["status"] != "open").sum()
    if n_closed:
        print(f"  Note: {n_closed} non-open rows excluded from simulation.")
    open_df["n_smart_buyers"] = open_df["n_smart_buyers"].astype(int)
    open_df["entry_ask"]      = open_df["entry_ask"].astype(float)
    open_df["current_price"]  = open_df["current_price"].astype(float)
    return open_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Monte Carlo engine
# ---------------------------------------------------------------------------

def simulate_strategy(
    positions: pd.DataFrame,
    use_kelly: bool,
    stop_loss: float | None,
    base_fraction: float = 0.10,
    ref_buyers: int = 4,
    kelly_cap: float = 2.0,
    n_sims: int = 50_000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return portfolio PnL array of shape (n_sims,)."""
    if rng is None:
        rng = np.random.default_rng(42)

    n_pos = len(positions)
    entry_asks   = positions["entry_ask"].values
    current_ps   = positions["current_price"].values
    n_buyers_arr = positions["n_smart_buyers"].values.astype(int)

    # Per-position bet fractions
    if use_kelly:
        fracs = np.array([kelly_fraction(n, base_fraction, ref_buyers, kelly_cap)
                          for n in n_buyers_arr])
    else:
        fracs = np.full(n_pos, base_fraction)

    # Determine which positions are already stopped
    if stop_loss is not None:
        stop_floors = entry_asks * (1.0 - stop_loss)
        already_stopped = current_ps <= stop_floors
    else:
        already_stopped = np.zeros(n_pos, dtype=bool)

    # Stopped positions: fixed PnL at current price
    stopped_pnl = np.where(
        already_stopped,
        (current_ps / entry_asks - 1.0) * fracs,
        0.0,
    ).sum()

    # Active positions: Bernoulli resolution
    active_mask = ~already_stopped
    if active_mask.any():
        win_probs = np.clip(current_ps[active_mask], 0.0, 1.0)
        asks_a    = entry_asks[active_mask]
        fracs_a   = fracs[active_mask]
        outcomes  = rng.binomial(1, win_probs, size=(n_sims, active_mask.sum()))
        res_pnl   = ((outcomes / asks_a - 1.0) * fracs_a).sum(axis=1)
    else:
        res_pnl = np.zeros(n_sims)

    return stopped_pnl + res_pnl


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def stats(pnl: np.ndarray) -> dict:
    return {
        "E[PnL]":   float(pnl.mean()),
        "Median":   float(np.median(pnl)),
        "Std":      float(pnl.std()),
        "CVaR(5%)": float(np.percentile(pnl, 5)),
        "Min":      float(pnl.min()),
        "Max":      float(pnl.max()),
        "P(>0)":    float((pnl > 0).mean()),
        "Sharpe":   float(pnl.mean() / (pnl.std() + 1e-9)),
    }


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_stop_analysis(positions: pd.DataFrame, stop_loss: float) -> None:
    print(f"\n{'═'*88}")
    print(f"  STOP-LOSS ANALYSIS  (threshold = {stop_loss:.0%} drawdown from entry ask)")
    print("═"*88)
    rows = []
    for _, r in positions.iterrows():
        entry = r["entry_ask"]
        cur   = r["current_price"]
        floor = entry * (1.0 - stop_loss)
        loss  = (entry - cur) / entry
        flag  = "STOPPED ✗" if cur <= floor else "open"
        rows.append([
            (r["question"] or "")[:42],
            r["domain"][:10],
            int(r["n_smart_buyers"]),
            f"{entry:.3f}",
            f"{cur:.3f}",
            f"{floor:.3f}",
            f"{loss:+.1%}",
            flag,
        ])
    # sort: stopped first
    rows.sort(key=lambda x: (0 if "STOPPED" in x[-1] else 1, x[-2]))
    print(tabulate(rows,
                   headers=["question", "domain", "buyers", "entry", "now", "stop@", "loss%", "status"],
                   tablefmt="rounded_grid"))
    n_stopped = sum(1 for r in rows if "STOPPED" in r[-1])
    print(f"  → {n_stopped} / {len(rows)} positions currently below stop threshold.")


def print_kelly_breakdown(positions: pd.DataFrame, base_fraction: float,
                          ref_buyers: int, kelly_cap: float) -> None:
    print(f"\n{'═'*72}")
    print(f"  KELLY SIZING  (base={base_fraction:.0%} at {ref_buyers} buyers, cap={kelly_cap}×)")
    print("═"*72)
    rows = []
    for _, r in positions.iterrows():
        n    = int(r["n_smart_buyers"])
        frac = kelly_fraction(n, base_fraction, ref_buyers, kelly_cap)
        rows.append([
            (r["question"] or "")[:42],
            n,
            f"{base_fraction:.1%}",
            f"{frac:.1%}",
            f"{frac - base_fraction:+.1%}",
        ])
    rows.sort(key=lambda x: -x[1])
    print(tabulate(rows,
                   headers=["question", "buyers", "flat", "kelly", "Δ"],
                   tablefmt="rounded_grid"))


def print_summary_table(results: dict[str, tuple[np.ndarray, dict]]) -> None:
    print(f"\n{'═'*80}")
    print(f"  STRATEGY COMPARISON  ({next(iter(results.values()))[0].shape[0]:,} simulations)")
    print("═"*80)
    rows = []
    for name, (_, s) in results.items():
        rows.append([
            name,
            f"{s['E[PnL]']:+.4f}",
            f"{s['Median']:+.4f}",
            f"{s['Std']:.4f}",
            f"{s['CVaR(5%)']:+.4f}",
            f"{s['Min']:+.4f}",
            f"{s['Sharpe']:+.3f}",
            f"{s['P(>0)']:.1%}",
        ])
    print(tabulate(rows,
                   headers=["strategy", "E[PnL]", "Median", "Std", "CVaR(5%)", "Worst", "Sharpe", "P(profit)"],
                   tablefmt="rounded_grid"))
    print("  PnL is additive across positions, expressed as fraction of bankroll reference.")
    print("  CVaR(5%) = expected PnL in the worst 5% of scenarios.")


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def plot_dashboard(results: dict[str, tuple[np.ndarray, dict]], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: PnL distributions
    ax = axes[0]
    for (name, (pnl, s)), c in zip(results.items(), _COLORS):
        ax.hist(pnl, bins=120, alpha=0.50, label=name, color=c, density=True)
        ax.axvline(s["Median"], color=c, lw=1.5, ls="--", alpha=0.8)
        ax.axvline(s["CVaR(5%)"], color=c, lw=1.0, ls=":", alpha=0.7)
    ax.axvline(0, color="k", lw=1.2)
    ax.set_title("Portfolio PnL distribution — Monte Carlo (50k scenarios)\n"
                 "dashed = median, dotted = CVaR(5%)")
    ax.set_xlabel("Portfolio PnL (fraction of bankroll)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    # Panel 2: Risk metrics bar chart
    ax = axes[1]
    metrics = ["E[PnL]", "Median", "CVaR(5%)", "Min"]
    x = np.arange(len(metrics))
    w = 0.18
    for j, ((name, (_, s)), c) in enumerate(zip(results.items(), _COLORS)):
        vals = [s[m] for m in metrics]
        ax.bar(x + j * w, vals, w, label=name, color=c, alpha=0.85)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels(metrics)
    ax.set_title("Risk metrics by strategy")
    ax.set_ylabel("PnL (fraction of bankroll)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"  Chart saved → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-sims",     type=int,   default=50_000,    help="Monte Carlo draws")
    ap.add_argument("--seed",       type=int,   default=42,        help="RNG seed")
    ap.add_argument("--fraction",   type=float, default=0.10,      help="base bet fraction")
    ap.add_argument("--ref-buyers", type=int,   default=4,         help="Kelly reference buyer count")
    ap.add_argument("--kelly-cap",  type=float, default=2.0,       help="Kelly cap multiple")
    ap.add_argument("--stop-loss",  type=float, default=0.35,      help="stop-loss threshold")
    ap.add_argument("--ledger",     type=Path,  default=STATE_FILE, help="positions CSV path")
    ap.add_argument("--no-plots",   action="store_true",            help="skip chart output")
    args = ap.parse_args()

    ensure_dirs()

    print("""
╔══════════════════════════════════════════════════════════════╗
║  SMART-FLOW PAPER TRADE  ·  risk management backtest          ║
║  Monte Carlo on open positions using market-implied probs      ║
╚══════════════════════════════════════════════════════════════╝""")

    positions = load_positions(args.ledger)
    print(f"  {len(positions)} open positions loaded from {args.ledger.name}")

    rng = np.random.default_rng(args.seed)

    # Patch stop_loss from args into strategies
    strats = {
        "baseline":    {"use_kelly": False, "stop_loss": None},
        "kelly":       {"use_kelly": True,  "stop_loss": None},
        "stops":       {"use_kelly": False, "stop_loss": args.stop_loss},
        "kelly+stops": {"use_kelly": True,  "stop_loss": args.stop_loss},
    }

    results: dict[str, tuple[np.ndarray, dict]] = {}
    for name, kwargs in strats.items():
        pnl = simulate_strategy(
            positions, **kwargs,
            base_fraction=args.fraction,
            ref_buyers=args.ref_buyers,
            kelly_cap=args.kelly_cap,
            n_sims=args.n_sims,
            rng=rng,
        )
        results[name] = (pnl, stats(pnl))

    print_stop_analysis(positions, args.stop_loss)
    print_kelly_breakdown(positions, args.fraction, args.ref_buyers, args.kelly_cap)
    print_summary_table(results)

    if not args.no_plots:
        out = RESULTS_DIR / "smart_flow_backtest.png"
        plot_dashboard(results, out)

    print("\n✓  Backtest complete.")


if __name__ == "__main__":
    main()
