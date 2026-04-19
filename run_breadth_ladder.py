#!/usr/bin/env python3
"""
Universe-widening experiment: does more breadth lift MN Sharpe?

Thesis: residual stat-arb Sharpe scales roughly √N_residuals.  Going from
4 residuals (real-only) to ~29 (real + wide realistic synth) should lift
the MN sleeve's Sharpe materially — the binding constraint identified
in earlier runs was breadth, not design.

This script runs the strategy on three universes of increasing width and
prints the Sharpe ladder:

  narrow  : 4 real residuals (SPY hedge + TLT/GLD/EEM/VNQ)
  medium  : 13 residuals (narrow + 9 base realistic synth)
  wide    : 29 residuals (narrow + 9 base + 21 sectors/credit/rates/FX/style)

For a clean comparison of the breadth effect we use the OU-residual DGP
(half-life = z_window), so every synth asset has the mean-reverting
structure the strategy targets.  This isolates the breadth channel from
the "is there any alpha in the residuals" question.

When real ETF data for the wide list becomes available later, replace
`wide_realistic_spec(...)` with `fit_realistic_spec(real_returns,
factors, asset_names)` — no other changes are required.
"""
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from tabulate import tabulate

from src.market_neutral import MNParams, run_market_neutral
from src.metrics import compute_metrics
from src.synthetic_data import (
    default_realistic_spec,
    expand_universe_realistic,
    wide_realistic_spec,
    RealisticUniverseSpec, AssetParams,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S", stream=sys.stdout,
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
RF       = 0.04
BT_START = "2016-01-01"
BT_END   = "2024-12-31"
OU_HL    = 21


# ── Data ────────────────────────────────────────────────────────────────────
df     = pd.read_csv("portfolio_data.csv", parse_dates=["Date"], index_col="Date").sort_index()
r_real = df.pct_change().dropna()
spy_bt = r_real["SPY"].loc[BT_START:BT_END].rename("SPY")


def _ou_ify(spec: RealisticUniverseSpec, hl: float) -> RealisticUniverseSpec:
    """Return a copy of `spec` where every asset has OU residuals of half-life `hl`."""
    return RealisticUniverseSpec(
        factor_names=list(spec.factor_names),
        assets=[
            AssetParams(
                name=p.name, betas=p.betas, alpha_ann=0.0,
                idio_ar=0.0, idio_sd=p.idio_sd,
                sv_ar=p.sv_ar, sv_innov_sd=p.sv_innov_sd, t_df=p.t_df,
                ou_half_life=hl,
            )
            for p in spec.assets
        ],
    )


# Three universes, each with OU-residual synth where applicable
narrow_returns = r_real
medium_returns = expand_universe_realistic(
    r_real, spec=_ou_ify(default_realistic_spec(), OU_HL), seed=42
)
wide_returns = expand_universe_realistic(
    r_real,
    spec=wide_realistic_spec(ou_half_life=OU_HL, include_base=True),
    seed=42,
)

UNIVERSES = {
    "narrow (4 resid)":  narrow_returns,
    "medium (13 resid)": medium_returns,
    "wide (29 resid)":   wide_returns,
}

print("""
╔══════════════════════════════════════════════════════════════════╗
║  Breadth experiment · narrow → medium → wide synthetic universe  ║
║  OU-residual DGP (half-life = 21d) isolates the √N effect         ║
╚══════════════════════════════════════════════════════════════════╝""")
for name, rets in UNIVERSES.items():
    logger.info(f"{name:>18s}  ·  {rets.shape[1]} cols  ·  first: {list(rets.columns)[:5]}…")


# ── Strategy: weekly rebalance, symmetric hedge (we saw asymmetric is 2nd order) ──
def make_params() -> MNParams:
    return MNParams(
        market="SPY", reg_window=252, z_window=21,
        rebalance_freq="W-FRI", target_vol=0.05, vol_window=63,
        leverage_cap=2.0, tc=0.0005, hazard=1 / 252,
        hedge_mode="symmetric",
    )


results = {}
for name, rets in UNIVERSES.items():
    logger.info(f"── running  {name} ──")
    res = run_market_neutral(rets, params=make_params())
    res.returns = res.returns.loc[BT_START:BT_END]
    results[name] = res


# ── Metrics ──────────────────────────────────────────────────────────────────
COL_KEYS = ["CAGR", "Sharpe", "Sortino", "Max DD", "Calmar",
            "Volatility", "Alpha", "Beta"]


def fmt(v, k):
    if v is None or not np.isfinite(v):
        return "—"
    if k in ("CAGR", "Max DD", "Volatility", "Alpha"):
        return f"{v:.2%}"
    return f"{v:.2f}"


rows = []
for name, res in results.items():
    m = compute_metrics(res.returns, benchmark=spy_bt, rf=RF)
    rows.append([name] + [fmt(m.get(k, float("nan")), k) for k in COL_KEYS])

print("\n" + "═" * 115)
print(f"  Breadth ladder · MN strategy on OU-residual DGP · 2016–2024")
print("═" * 115)
print(tabulate(rows, headers=["universe"] + COL_KEYS, tablefmt="rounded_grid"))


# ── Plot ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 10))
fig.suptitle("Breadth ladder · market-neutral strategy on OU-residual DGP",
             fontsize=13, fontweight="bold")
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.25)
ax_cum = fig.add_subplot(gs[0, :])
ax_dd  = fig.add_subplot(gs[1, 0])
ax_sr  = fig.add_subplot(gs[1, 1])

colors = {
    "narrow (4 resid)":  "#B71C1C",
    "medium (13 resid)": "#1565C0",
    "wide (29 resid)":   "#2E7D32",
}
_PCT    = mticker.FuncFormatter(lambda x, _: f"{x:.0%}")
_DOLLAR = mticker.FuncFormatter(lambda x, _: f"${x:.2f}")

spy_cum = (1 + spy_bt).cumprod()
ax_cum.plot(spy_cum.index, spy_cum.values, color="#37474F",
            linestyle="--", linewidth=1.0, label="SPY B&H")
for name, res in results.items():
    cum = (1 + res.returns).cumprod()
    ax_cum.plot(cum.index, cum.values, color=colors[name],
                linewidth=2.0, label=name)
ax_cum.set_yscale("log")
ax_cum.yaxis.set_major_formatter(_DOLLAR)
ax_cum.set_title("Cumulative Wealth (log scale)", fontweight="bold")
ax_cum.legend(fontsize=9, ncol=4)
ax_cum.grid(alpha=0.25)

for name, res in results.items():
    cum = (1 + res.returns).cumprod()
    dd  = (cum - cum.cummax()) / (cum.cummax() + 1e-9)
    ax_dd.plot(dd.index, dd.values, color=colors[name], linewidth=1.3, label=name)
ax_dd.yaxis.set_major_formatter(_PCT)
ax_dd.set_title("Drawdown", fontweight="bold")
ax_dd.legend(fontsize=8)
ax_dd.grid(alpha=0.25)

for name, res in results.items():
    exc = res.returns - RF / 252
    rs  = exc.rolling(126).mean() / exc.rolling(126).std() * np.sqrt(252)
    ax_sr.plot(rs.index, rs.values, color=colors[name], linewidth=1.3, label=name)
ax_sr.axhline(0, color="black", lw=0.5, ls="--")
ax_sr.axhline(1, color="green", lw=0.4, ls=":", alpha=0.6)
ax_sr.set_title("Rolling 6-Month Sharpe", fontweight="bold")
ax_sr.legend(fontsize=8)
ax_sr.grid(alpha=0.25)

fig.savefig(RESULTS_DIR / "breadth_ladder.png", dpi=150, bbox_inches="tight")
plt.close(fig)
logger.info("Saved: results/breadth_ladder.png")

out = pd.DataFrame({n: r.returns for n, r in results.items()})
out["SPY"] = spy_bt
out.to_csv(RESULTS_DIR / "breadth_ladder_returns.csv")
logger.info("Saved: results/breadth_ladder_returns.csv")

print("\n✓  Done.")
