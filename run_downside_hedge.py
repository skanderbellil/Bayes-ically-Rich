#!/usr/bin/env python3
"""
Compare three ways of sizing the SPY hedge for the market-neutral book.

  symmetric   OLS β, full market neutrality in both directions.
  quantile    Koenker-Bassett β at τ = 0.10, tail-conditional slope.
  bear_hmm    β weighted by posterior P(bear) from a 3-state Gaussian HMM
              on SPY returns (regime-defined downside, not sign of today).

Residuals are always built from symmetric OLS β (the idiosyncratic return
is a well-defined statistical object); only the hedge sizing differs.

Two universes:
  • Real-only  (SPY + TLT/GLD/EEM/VNQ)                 — live market data.
  • OU-residual synth (9 extra assets, Avellaneda-Lee) — DGP with genuine
    residual mean-reversion so the strategy has something to capture and
    the hedge choice matters.
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
    AssetParams,
    RealisticUniverseSpec,
    default_realistic_spec,
    expand_universe_realistic,
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

# ── Data ────────────────────────────────────────────────────────────────────
df      = pd.read_csv("portfolio_data.csv", parse_dates=["Date"], index_col="Date").sort_index()
r_real  = df.pct_change().dropna()
spy_bt  = r_real["SPY"].loc[BT_START:BT_END].rename("SPY")

# OU-residual synth (half-life = z_window to give the strategy something to capture)
base_spec = default_realistic_spec()
OU_HL = 21
ou_spec = RealisticUniverseSpec(
    factor_names=list(base_spec.factor_names),
    assets=[
        AssetParams(
            name=p.name, betas=p.betas, alpha_ann=0.0,
            idio_ar=0.0, idio_sd=p.idio_sd,
            sv_ar=p.sv_ar, sv_innov_sd=p.sv_innov_sd, t_df=p.t_df,
            ou_half_life=OU_HL,
        )
        for p in base_spec.assets
    ],
)
r_synth = expand_universe_realistic(r_real, spec=ou_spec, seed=42)

print("""
╔══════════════════════════════════════════════════════════════════════╗
║   Downside Hedge Comparison · market-neutral residual strategy        ║
║   symmetric OLS  vs  quantile (τ=0.10)  vs  bear-HMM posterior        ║
╚══════════════════════════════════════════════════════════════════════╝""")
logger.info(f"Real universe    : {list(r_real.columns)}")
logger.info(f"Synth universe   : {list(r_synth.columns)}")
logger.info(f"OU half-life (d) : {OU_HL}")
logger.info(f"Window           : {BT_START} → {BT_END}")


# ── Run all three hedge modes on both universes ────────────────────────────
def make_params(mode: str) -> MNParams:
    return MNParams(
        market="SPY", reg_window=252, z_window=21,
        rebalance_freq="W-FRI", target_vol=0.05, vol_window=63,
        leverage_cap=2.0, tc=0.0005, hazard=1 / 252,
        hedge_mode=mode, quantile_tau=0.10, hmm_refit_every=252,
    )

MODES = ["symmetric", "quantile", "bear_hmm"]

results = {}   # (universe, mode) → BacktestResult
for universe_name, rets in [("real", r_real), ("synth", r_synth)]:
    for mode in MODES:
        logger.info(f"── {universe_name:>5s}  ·  {mode} ──")
        res = run_market_neutral(rets, params=make_params(mode))
        res.returns = res.returns.loc[BT_START:BT_END]
        results[(universe_name, mode)] = res


# ── Metrics, including downside-beta to directly check the hedge promise ───
def downside_beta(r: pd.Series, bench: pd.Series) -> float:
    """β conditional on benchmark < 0.  Measures what the asymmetric hedge
    is explicitly trying to kill; compare to full β to see if it worked."""
    a = pd.concat([r, bench], axis=1).dropna()
    mask = a.iloc[:, 1] < 0
    if mask.sum() < 20:
        return float("nan")
    b = a[mask]
    cov = np.cov(b.iloc[:, 0], b.iloc[:, 1])
    return float(cov[0, 1] / (cov[1, 1] + 1e-12))


def upside_beta(r: pd.Series, bench: pd.Series) -> float:
    a = pd.concat([r, bench], axis=1).dropna()
    mask = a.iloc[:, 1] > 0
    if mask.sum() < 20:
        return float("nan")
    b = a[mask]
    cov = np.cov(b.iloc[:, 0], b.iloc[:, 1])
    return float(cov[0, 1] / (cov[1, 1] + 1e-12))


COL_KEYS = ["CAGR", "Sharpe", "Sortino", "Max DD", "Calmar",
            "Volatility", "Alpha", "Beta"]


def fmt(v, k):
    if v is None or not np.isfinite(v):
        return "—"
    if k in ("CAGR", "Max DD", "Volatility", "Alpha"):
        return f"{v:.2%}"
    return f"{v:.2f}"


for universe_name in ("real", "synth"):
    rows = []
    for mode in MODES:
        res = results[(universe_name, mode)]
        m   = compute_metrics(res.returns, benchmark=spy_bt, rf=RF)
        b_m = m.get("Beta", float("nan"))
        b_d = downside_beta(res.returns, spy_bt)
        b_u = upside_beta(res.returns, spy_bt)
        rows.append(
            [mode]
            + [fmt(m.get(k, float("nan")), k) for k in COL_KEYS]
            + [f"{b_d:+.2f}" if np.isfinite(b_d) else "—",
               f"{b_u:+.2f}" if np.isfinite(b_u) else "—",
               f"{(b_u - b_d):+.2f}" if np.isfinite(b_u - b_d) else "—"]
        )
    print("\n" + "═" * 130)
    print(f"  {universe_name.upper():>5s}  universe  ·  2016–2024")
    print("═" * 130)
    print(tabulate(
        rows,
        headers=["hedge"] + COL_KEYS + ["β⁻ (down)", "β⁺ (up)", "β⁺−β⁻"],
        tablefmt="rounded_grid",
    ))


# ── Plot cumulative wealth for the synth universe (hedge mode matters most) ─
fig = plt.figure(figsize=(20, 10))
fig.suptitle(
    "Downside-hedge modes on OU-residual synth universe",
    fontsize=13, fontweight="bold",
)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.25)
ax_cum = fig.add_subplot(gs[0, :])
ax_dd  = fig.add_subplot(gs[1, 0])
ax_sr  = fig.add_subplot(gs[1, 1])

colors = {"symmetric": "#546E7A", "quantile": "#AD1457", "bear_hmm": "#2E7D32"}
_PCT    = mticker.FuncFormatter(lambda x, _: f"{x:.0%}")
_DOLLAR = mticker.FuncFormatter(lambda x, _: f"${x:.2f}")

spy_cum = (1 + spy_bt).cumprod()
ax_cum.plot(spy_cum.index, spy_cum.values, color="#37474F",
            linestyle="--", linewidth=1.1, label="SPY B&H")
for mode in MODES:
    cum = (1 + results[("synth", mode)].returns).cumprod()
    ax_cum.plot(cum.index, cum.values, color=colors[mode],
                linewidth=2.0, label=mode)
ax_cum.set_yscale("log")
ax_cum.yaxis.set_major_formatter(_DOLLAR)
ax_cum.set_title("Cumulative Wealth (log scale)", fontweight="bold")
ax_cum.legend(fontsize=9, ncol=4)
ax_cum.grid(alpha=0.25)

for mode in MODES:
    cum = (1 + results[("synth", mode)].returns).cumprod()
    dd  = (cum - cum.cummax()) / (cum.cummax() + 1e-9)
    ax_dd.plot(dd.index, dd.values, color=colors[mode],
               linewidth=1.3, label=mode)
ax_dd.yaxis.set_major_formatter(_PCT)
ax_dd.set_title("Drawdown", fontweight="bold")
ax_dd.legend(fontsize=8)
ax_dd.grid(alpha=0.25)

for mode in MODES:
    exc = results[("synth", mode)].returns - RF / 252
    rs  = exc.rolling(126).mean() / exc.rolling(126).std() * np.sqrt(252)
    ax_sr.plot(rs.index, rs.values, color=colors[mode],
               linewidth=1.3, label=mode)
ax_sr.axhline(0, color="black", lw=0.5, ls="--")
ax_sr.set_title("Rolling 6-Month Sharpe", fontweight="bold")
ax_sr.legend(fontsize=8)
ax_sr.grid(alpha=0.25)

fig.savefig(RESULTS_DIR / "downside_hedge.png", dpi=150, bbox_inches="tight")
plt.close(fig)
logger.info("Saved: results/downside_hedge.png")

out = pd.DataFrame({
    f"{u}_{m}": results[(u, m)].returns
    for u in ("real", "synth") for m in MODES
})
out["SPY"] = spy_bt
out.to_csv(RESULTS_DIR / "downside_hedge_returns.csv")
logger.info("Saved: results/downside_hedge_returns.csv")

print("\n✓  Done.")
