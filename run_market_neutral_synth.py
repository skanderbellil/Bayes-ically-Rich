#!/usr/bin/env python3
"""
Market-neutral strategy validated on a realistic synthetic universe.

Generator features (see src/synthetic_data.py):
  • Factor loadings on real SPY/TLT/GLD/EEM/VNQ paths
  • AR(1) mean-reverting idiosyncratic residuals
  • Log-AR(1) stochastic volatility (vol clustering)
  • Student-t innovations (fat tails)

The expanded universe (5 real + 9 realistic synth = 14 assets) gives the
market-neutral residual strategy a broader residual book than the 4-asset
real universe, which is the capacity constraint flagged earlier.

Because parameters are `fit_params_from_real`-compatible, this same script
will work on any real universe in the future — simply replace the preset
spec with `fit_realistic_spec(real_returns, factors, asset_names)` and the
synthetic siblings are matched to the new real statistical signatures.
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
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
RF       = 0.04
BT_START = "2016-01-01"
BT_END   = "2024-12-31"

# ── Data ────────────────────────────────────────────────────────────────────
df  = pd.read_csv("portfolio_data.csv", parse_dates=["Date"], index_col="Date").sort_index()
r_real = df.pct_change().dropna()
spy_bt = r_real["SPY"].loc[BT_START:BT_END].rename("SPY")

spec    = default_realistic_spec()
r_synth = expand_universe_realistic(r_real, spec=spec, seed=42)
real_assets  = list(r_real.columns)
synth_assets = [a for a in r_synth.columns if a not in real_assets]

print("""
╔══════════════════════════════════════════════════════════════════════╗
║   Market-Neutral · Realistic Synthetic Universe                      ║
║   5 real ETFs + 9 realistic synth (AR-resid + stoch-vol + fat tail)  ║
╚══════════════════════════════════════════════════════════════════════╝""")
logger.info(f"Real      : {real_assets}")
logger.info(f"Synthetic : {synth_assets}")
logger.info(f"Window    : {BT_START} → {BT_END}")


# ── Sanity: stylised facts of synth RESIDUALS (factor-beta removed) ─────────
# Total-return autocorrelation is dominated by the factor path.  For a
# stat-arb strategy what matters is the residual autocorrelation, so we
# regress each synth asset on the real factors and report ρ(ε_t, ε_{t-1}).
factor_mat = r_real[list(spec.factor_names)].values
rows = []
for a in synth_assets:
    y  = r_synth[a].values
    Xd = np.column_stack([np.ones(len(factor_mat)), factor_mat])
    coef, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    eps = y - Xd @ coef
    rho_eps = float(np.corrcoef(eps[:-1], eps[1:])[0, 1])
    rows.append([
        a,
        f"{r_synth[a].std()*np.sqrt(252):.2%}",
        f"{r_synth[a].skew():.2f}",
        f"{r_synth[a].kurt():.2f}",
        f"{rho_eps:+.3f}",
    ])
print("\nSynthetic statistical signatures (residuals against real factors):")
print(tabulate(
    rows,
    headers=["Asset", "Ann Vol", "Skew", "Excess Kurt", "Resid AR(1) ρ"],
    tablefmt="rounded_grid",
))


# ── Strategy: run on (a) real-only, (b) real+synth ──────────────────────────
params = MNParams(
    market="SPY", reg_window=252, z_window=21,
    rebalance_freq="W-FRI", target_vol=0.05, vol_window=63,
    leverage_cap=2.0, tc=0.0005, hazard=1 / 252,
)

logger.info("── Market-Neutral on REAL-only universe ──")
mn_real = run_market_neutral(r_real, params=params)

logger.info("── Market-Neutral on REAL + REALISTIC-SYNTH universe ──")
mn_full = run_market_neutral(r_synth, params=params)

# Validation: Avellaneda-Lee OU residuals with half-life matching the
# strategy's z-window.  If the strategy implementation is correct it
# must extract alpha here; if not, the implementation has a bug.
OU_HALF_LIFE = params.z_window   # 21 trading days
mr_spec = RealisticUniverseSpec(
    factor_names=list(spec.factor_names),
    assets=[
        AssetParams(
            name=p.name, betas=p.betas, alpha_ann=0.0,
            idio_ar=0.0, idio_sd=p.idio_sd,
            sv_ar=p.sv_ar, sv_innov_sd=p.sv_innov_sd, t_df=p.t_df,
            ou_half_life=OU_HALF_LIFE,
        )
        for p in spec.assets
    ],
)
r_mr = expand_universe_realistic(r_real, spec=mr_spec, seed=42)
logger.info(f"── Market-Neutral on OU-residuals synth (half-life = {OU_HALF_LIFE}d) ──")
mn_mr = run_market_neutral(r_mr, params=params)

results = {
    "Real-only (4 residuals)":             mn_real,
    "Real + realistic synth (iid resid)":  mn_full,
    "Real + OU-residual synth (HL=21d)":   mn_mr,
}
for k in results:
    results[k].returns = results[k].returns.loc[BT_START:BT_END]


# ── Metrics ─────────────────────────────────────────────────────────────────
COL_KEYS = ["CAGR", "Sharpe", "Sortino", "Max DD", "Calmar",
            "Volatility", "Alpha", "Beta", "Info Ratio"]

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
spy_m = compute_metrics(spy_bt, rf=RF)
rows.append(["SPY Buy & Hold"] + [fmt(spy_m.get(k, float("nan")), k) for k in COL_KEYS])

print("\n" + "═" * 110)
print("  RESULTS  ·  2016–2024")
print("═" * 110)
print(tabulate(rows, headers=["Universe"] + COL_KEYS, tablefmt="rounded_grid"))


# ── Plot ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 10))
fig.suptitle(
    "Market-Neutral on real vs. realistic synthetic universe",
    fontsize=13, fontweight="bold",
)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.25)
ax_cum = fig.add_subplot(gs[0, :])
ax_dd  = fig.add_subplot(gs[1, 0])
ax_sr  = fig.add_subplot(gs[1, 1])

_PCT    = mticker.FuncFormatter(lambda x, _: f"{x:.0%}")
_DOLLAR = mticker.FuncFormatter(lambda x, _: f"${x:.2f}")

colors = {
    "Real-only (4 residuals)":            "#546E7A",
    "Real + realistic synth (iid resid)": "#1565C0",
    "Real + OU-residual synth (HL=21d)":  "#2E7D32",
}

spy_cum = (1 + spy_bt).cumprod()
ax_cum.plot(spy_cum.index, spy_cum.values, color="#37474F",
            linestyle="--", linewidth=1.2, label="SPY B&H")
for name, res in results.items():
    cum = (1 + res.returns).cumprod()
    ax_cum.plot(cum.index, cum.values, color=colors[name],
                linewidth=2.0, label=name)
ax_cum.set_yscale("log")
ax_cum.yaxis.set_major_formatter(_DOLLAR)
ax_cum.set_title("Cumulative Wealth (log scale)", fontweight="bold")
ax_cum.legend(fontsize=9, ncol=3)
ax_cum.grid(alpha=0.25)

for name, res in results.items():
    cum = (1 + res.returns).cumprod()
    dd  = (cum - cum.cummax()) / (cum.cummax() + 1e-9)
    ax_dd.plot(dd.index, dd.values, color=colors[name],
               linewidth=1.3, label=name)
ax_dd.yaxis.set_major_formatter(_PCT)
ax_dd.set_title("Drawdown", fontweight="bold")
ax_dd.legend(fontsize=8)
ax_dd.grid(alpha=0.25)

for name, res in results.items():
    exc = res.returns - RF / 252
    rs  = exc.rolling(126).mean() / exc.rolling(126).std() * np.sqrt(252)
    ax_sr.plot(rs.index, rs.values, color=colors[name],
               linewidth=1.3, label=name)
ax_sr.axhline(0, color="black", lw=0.5, ls="--")
ax_sr.axhline(1, color="green", lw=0.4, ls=":", alpha=0.6)
ax_sr.set_title("Rolling 6-Month Sharpe", fontweight="bold")
ax_sr.legend(fontsize=8)
ax_sr.grid(alpha=0.25)

fig.savefig(RESULTS_DIR / "market_neutral_synth.png", dpi=150, bbox_inches="tight")
plt.close(fig)
logger.info("Saved: results/market_neutral_synth.png")

out = pd.DataFrame({n: r.returns for n, r in results.items()})
out["SPY"] = spy_bt
out.to_csv(RESULTS_DIR / "market_neutral_synth_returns.csv")
logger.info("Saved: results/market_neutral_synth_returns.csv")

print("\n✓  Done.")
