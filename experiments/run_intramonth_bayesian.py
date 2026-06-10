#!/usr/bin/env python3
"""
Bayesian Intramonth Window Detector
======================================
Instead of hardcoding the T-9→T-4 window, this module learns it from data.

Model
-----
For every day-position T-k (k = 0 … MAX_K) within the month we maintain a
conjugate Normal-Normal Bayesian model of the expected WML return:

  Prior       : θ_k  ~ N(0, σ_prior²)         ← no window assumed a-priori
  Likelihood  : y_t  ~ N(θ_k, σ_obs²)
  Posterior   : θ_k | y_{1:t} ~ N(μ_k, s_k²)  (closed-form update)

Allocation for position k on the next observation:
  z_k   = μ_k / s_k                            (posterior z-score)
  alloc = 2·Φ(z_k) − 1  ∈ [−1, 1]            (maps probability to allocation)
  → near 0 when uncertain, +1 when strongly positive, −1 when strongly negative

The detector is updated *after* each day's trade (strict no-look-ahead).

Strategies compared
-------------------
  Bayes WML        — WML × alloc_k  (fully adaptive, market-neutral)
  SPY + Bayes WML  — SPY baseline + WML × alloc_k overlay
  Fixed WML Win    — WML only on hardcoded T-9→T-4 (baseline from paper)
  SPY + Fixed Win  — SPY + hardcoded window overlay
  SPY B&H          — passive benchmark
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from tabulate import tabulate

import _bootstrap  # noqa: F401  (adds repo root to sys.path)
from posterioralpha.validation import compute_metrics
from posterioralpha.data.loaders import load_portfolio_prices

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)
RESULTS_DIR = _bootstrap.ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
CSV_PATH      = Path("portfolio_data.csv")
BT_START      = "2007-01-01"
BT_END        = "2024-12-31"
RF            = 0.04
MOMENTUM_LONG = 252
MOMENTUM_SKIP = 21
N_LONG        = 2
N_SHORT       = 2
MAX_K         = 25          # track up to T-25 positions
WINDOW_START  = 9           # fixed-window reference: T-9
WINDOW_END    = 4           # fixed-window reference: T-4
# Prior scale expressed as a multiple of the (estimated) observation std.
# 1.0 → unit-information prior (one observation worth of information)
# 0.5 → tighter / more shrinkage toward zero
PRIOR_SCALE   = 1.0

print("""
╔══════════════════════════════════════════════════════════════════════════╗
║  Bayesian Intramonth Window Detector                                     ║
║  Learning the T-k allocation signal from returns — no hardcoded window   ║
╚══════════════════════════════════════════════════════════════════════════╝""")


# ── Data ───────────────────────────────────────────────────────────────────────
logger.info(f"Loading {CSV_PATH} …")
prices   = load_portfolio_prices()
rets     = prices.pct_change().dropna()
spy_rets = rets["SPY"].rename("SPY")
assets   = rets.columns.tolist()
logger.info(f"Assets: {assets}  |  {len(rets)} trading days")


# ── Bayesian Window Detector ───────────────────────────────────────────────────

class BayesianWindowDetector:
    """
    Online conjugate Normal-Normal model for expected WML return at each T-k.

    σ_obs is estimated from a warm-up pool; σ_prior = PRIOR_SCALE × σ_obs.
    Both are re-estimated every `reestimate_every` months.
    """

    def __init__(self, max_k: int = MAX_K, reestimate_every: int = 6):
        self.max_k            = max_k
        self.reestimate_every = reestimate_every
        self._months_since    = 0

        # Sufficient statistics per position (vectorised)
        self.n   = np.zeros(max_k + 1)   # observation count
        self.s   = np.zeros(max_k + 1)   # sum of returns
        self.ssq = np.zeros(max_k + 1)   # sum of squared returns

        # Pool for σ_obs estimation (all k pooled)
        self._pool_s   = 0.0
        self._pool_ssq = 0.0
        self._pool_n   = 0

        self.sigma_obs   = None   # estimated later
        self.sigma_prior = None

    # ── Sufficient-statistic update (called AFTER each trade) ─────────────────
    def update(self, k: int, ret: float) -> None:
        if not (0 <= k <= self.max_k):
            return
        self.n[k]   += 1
        self.s[k]   += ret
        self.ssq[k] += ret * ret
        self._pool_s   += ret
        self._pool_ssq += ret * ret
        self._pool_n   += 1

    # ── σ estimation ──────────────────────────────────────────────────────────
    def _estimate_sigmas(self) -> None:
        if self._pool_n < 10:
            self.sigma_obs   = 0.01
            self.sigma_prior = 0.01
            return
        var = (self._pool_ssq
               - self._pool_s ** 2 / self._pool_n) / (self._pool_n - 1)
        self.sigma_obs   = float(np.sqrt(max(var, 1e-9)))
        self.sigma_prior = PRIOR_SCALE * self.sigma_obs

    def maybe_reestimate(self) -> None:
        self._months_since += 1
        if (self.sigma_obs is None or
                self._months_since >= self.reestimate_every):
            self._estimate_sigmas()
            self._months_since = 0

    # ── Posterior ─────────────────────────────────────────────────────────────
    def posterior_params(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (μ_post, σ_post) arrays over k = 0 … MAX_K."""
        if self.sigma_obs is None:
            self._estimate_sigmas()
        var_obs   = self.sigma_obs   ** 2
        var_prior = self.sigma_prior ** 2

        # Vectorised conjugate update
        prec_prior = 1.0 / var_prior
        prec_like  = np.where(self.n > 0, self.n / var_obs, 0.0)
        prec_post  = prec_prior + prec_like

        ybar       = np.divide(self.s, self.n, out=np.zeros_like(self.s),
                               where=self.n > 0)
        mu_post    = (prec_like * ybar) / prec_post
        sigma_post = 1.0 / np.sqrt(prec_post)

        return mu_post, sigma_post

    # ── Allocation for a single position ──────────────────────────────────────
    def allocation(self, k: int) -> float:
        """
        Smooth allocation in [−1, 1] for T-k position.
        alloc = 2·Φ(z) − 1 where z = μ_post_k / σ_post_k.
        """
        if not (0 <= k <= self.max_k):
            return 0.0
        mu, sig = self.posterior_params()
        z = mu[k] / sig[k]
        return float(2.0 * norm.cdf(z) - 1.0)

    # ── Snapshot for plotting ─────────────────────────────────────────────────
    def snapshot(self) -> dict:
        mu, sig = self.posterior_params()
        return {
            "n":    self.n.copy(),
            "mu":   mu.copy(),
            "sig":  sig.copy(),
            "alloc": np.array([self.allocation(k) for k in range(self.max_k + 1)]),
        }


# ── Helper: T-k label within a holding period ─────────────────────────────────

def day_k_labels(holding_days: pd.DatetimeIndex) -> np.ndarray:
    """k[j] = number of trading days before end-of-month for day j."""
    n = len(holding_days)
    return np.array([n - 1 - j for j in range(n)])


# ── Backtest ───────────────────────────────────────────────────────────────────

def run_backtest(rets: pd.DataFrame) -> dict:
    """
    Run five strategies in a single pass to ensure identical formation signals.

    Bayesian detector is updated online (after each day's trade).
    Snapshots of the posterior are saved monthly for plotting.
    """
    month_ends = rets.resample("ME").last().index
    min_obs    = MOMENTUM_LONG + MOMENTUM_SKIP + 5
    month_ends = month_ends[month_ends > rets.index[min_obs]]

    detector   = BayesianWindowDetector()
    snapshots  = {}          # date → detector.snapshot()

    rec = {s: {} for s in
           ["Bayes WML", "SPY+Bayes", "Fixed WML", "SPY+Fixed", "SPY B&H"]}

    spy_series = rets["SPY"]

    for i, me in enumerate(month_ends[:-1]):
        # ── Re-estimate σ at start of each month ──────────────────────────────
        detector.maybe_reestimate()
        snapshots[me] = detector.snapshot()

        # ── Momentum ranking ──────────────────────────────────────────────────
        hist = rets.loc[:me]
        if len(hist) < min_obs:
            continue

        formation = hist.iloc[-(MOMENTUM_LONG + MOMENTUM_SKIP):-MOMENTUM_SKIP]
        cum_ret   = (1.0 + formation).prod() - 1.0
        cum_ret   = cum_ret.dropna()
        if len(cum_ret) < N_LONG + N_SHORT + 1:
            continue

        ranked  = cum_ret.sort_values()
        losers  = ranked.index[:N_SHORT].tolist()
        winners = ranked.index[-N_LONG:].tolist()

        # ── Holding period ────────────────────────────────────────────────────
        next_me      = month_ends[i + 1]
        hold_mask    = (rets.index > me) & (rets.index <= next_me)
        holding_days = rets.index[hold_mask]

        bt_mask      = (holding_days >= BT_START) & (holding_days <= BT_END)
        holding_days = holding_days[bt_mask]
        if len(holding_days) == 0:
            continue

        k_labels  = day_k_labels(holding_days)
        day_rets  = rets.loc[holding_days]

        for j, date in enumerate(holding_days):
            k   = int(k_labels[j])
            dr  = day_rets.loc[date]
            spy = float(spy_series.loc[date])

            w_ret = float(dr[winners].mean())
            l_ret = float(dr[losers].mean())
            wml   = w_ret - l_ret

            # ── Allocation BEFORE updating (no look-ahead) ────────────────────
            alloc = detector.allocation(k)
            in_win = int(WINDOW_END <= k <= WINDOW_START)

            rec["Bayes WML"][date]  = wml * alloc
            rec["SPY+Bayes"][date]  = spy + wml * alloc
            rec["Fixed WML"][date]  = wml * in_win
            rec["SPY+Fixed"][date]  = spy + wml * in_win
            rec["SPY B&H"][date]    = spy

            # ── Update detector AFTER using allocation ─────────────────────────
            detector.update(k, wml)

    result = {k: pd.Series(v, name=k).sort_index() for k, v in rec.items()}
    return result, snapshots, detector


logger.info("Running Bayesian intramonth backtest …")
strats, snapshots, final_detector = run_backtest(rets)

for k in strats:
    strats[k] = strats[k].loc[BT_START:BT_END]

spy_bt = strats["SPY B&H"]


# ── Final posterior ────────────────────────────────────────────────────────────
final_snap = final_detector.snapshot()
k_vals     = np.arange(MAX_K + 1)
mu_final   = final_snap["mu"]
sig_final  = final_snap["sig"]
alloc_fin  = final_snap["alloc"]
n_final    = final_snap["n"]

logger.info(
    f"σ_obs={final_detector.sigma_obs*100:.3f}%  "
    f"σ_prior={final_detector.sigma_prior*100:.3f}%"
)


# ── Console output ─────────────────────────────────────────────────────────────
METRIC_KEYS = ["CAGR", "Volatility", "Sharpe", "Sortino", "Max DD", "Calmar"]

def fmt(v, k):
    return f"{v:.2%}" if k in ("CAGR", "Max DD", "Volatility", "Alpha") else f"{v:.2f}"

perf_map = {
    "Bayes WML":  (strats["Bayes WML"],  "Bayes WML (adaptive)"),
    "SPY+Bayes":  (strats["SPY+Bayes"],  "SPY + Bayes WML overlay"),
    "Fixed WML":  (strats["Fixed WML"],  f"Fixed WML (T-{WINDOW_START}→T-{WINDOW_END})"),
    "SPY+Fixed":  (strats["SPY+Fixed"],  f"SPY + Fixed WML overlay"),
    "SPY B&H":    (spy_bt,               "SPY Buy & Hold"),
}

perf_metrics = {key: compute_metrics(ret, benchmark=spy_bt, rf=RF)
                for key, (ret, _) in perf_map.items()}

print(f"\n{'='*80}")
print("  PERFORMANCE vs SPY BUY & HOLD")
print(f"{'='*80}")
perf_rows = []
for key, (ret, label) in perf_map.items():
    m = perf_metrics[key]
    perf_rows.append([label] + [fmt(m.get(k, 0.0), k) for k in METRIC_KEYS])
print(tabulate(perf_rows, headers=["Strategy"] + METRIC_KEYS, tablefmt="rounded_grid"))

print(f"\n{'='*80}")
print("  BENCHMARK-RELATIVE  (α, β, IR vs SPY B&H)")
print(f"{'='*80}")
rel_rows = []
for key, (ret, label) in perf_map.items():
    if key == "SPY B&H": continue
    m = perf_metrics[key]
    rel_rows.append([label,
                     fmt(m.get("Alpha", 0.0), "Alpha"),
                     f"{m.get('Beta', 0.0):.2f}",
                     f"{m.get('Info Ratio', 0.0):.2f}"])
print(tabulate(rel_rows, headers=["Strategy", "Alpha (ann.)", "Beta", "Info Ratio"],
               tablefmt="rounded_grid"))

print(f"\n{'='*80}")
print("  DELTA vs SPY BUY & HOLD")
print(f"{'='*80}")
spy_m = perf_metrics["SPY B&H"]
delta_rows = []
for key, (ret, label) in perf_map.items():
    if key == "SPY B&H": continue
    m = perf_metrics[key]
    row = [label]
    for mk in METRIC_KEYS:
        d = m.get(mk, 0.0) - spy_m.get(mk, 0.0)
        s = "+" if d >= 0 else ""
        row.append(f"{s}{d:.2%}" if mk in ("CAGR","Max DD","Volatility")
                   else f"{s}{d:.2f}")
    delta_rows.append(row)
print(tabulate(delta_rows,
               headers=["Strategy"] + [f"Δ {k}" for k in METRIC_KEYS],
               tablefmt="rounded_grid"))

print(f"\n{'='*80}")
print("  CUMULATIVE GROWTH OF $1")
print(f"{'='*80}")
cums = {key: float((1 + ret).prod()) for key, (ret, _) in perf_map.items()}
best = max(cums.values())
for key, (ret, label) in perf_map.items():
    marker = "  ◀ best" if cums[key] == best else ""
    print(f"  {label:42s}  ${cums[key]:.2f}{marker}")

print(f"\n{'='*80}")
print("  LEARNED POSTERIOR: top T-k positions by posterior mean WML return")
print(f"{'='*80}")
top_k = np.argsort(mu_final)[::-1][:10]
posterior_rows = []
for k in top_k:
    posterior_rows.append([
        f"T-{k}",
        f"{n_final[k]:.0f}",
        f"{mu_final[k]*100:+.3f}%",
        f"{sig_final[k]*100:.3f}%",
        f"{alloc_fin[k]:+.3f}",
        f"{norm.cdf(mu_final[k]/sig_final[k]):.1%}",
        "◀ paper window" if WINDOW_END <= k <= WINDOW_START else "",
    ])
print(tabulate(posterior_rows,
               headers=["Position", "N obs", "Post. mean", "Post. std",
                        "Allocation", "P(θ>0)", ""],
               tablefmt="rounded_grid"))


# ── Plots ──────────────────────────────────────────────────────────────────────
COLORS = {
    "Bayes WML": "#6A1B9A",   # purple
    "SPY+Bayes": "#1A237E",   # navy
    "Fixed WML": "#E65100",   # burnt orange
    "SPY+Fixed": "#1B5E20",   # dark green
    "SPY B&H":   "#37474F",   # slate
}

fig = plt.figure(figsize=(24, 20))
fig.suptitle(
    "Bayesian Intramonth Window Detector  ·  Learning the T-k Allocation Signal\n"
    "No hardcoded window — posterior updated online after each trade",
    fontsize=14, fontweight="bold",
)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38)

ax_cum  = fig.add_subplot(gs[0, :])    # full-width cumulative wealth
ax_post = fig.add_subplot(gs[1, 0])   # posterior mean ± 2σ by T-k
ax_alloc= fig.add_subplot(gs[1, 1])   # learned allocation by T-k
ax_dd   = fig.add_subplot(gs[1, 2])   # drawdown
ax_sr   = fig.add_subplot(gs[2, 0])   # rolling Sharpe
ax_evol = fig.add_subplot(gs[2, 1])   # allocation evolution: T-4 to T-9 over time
ax_prob = fig.add_subplot(gs[2, 2])   # P(θ_k > 0) heatmap over time

_PCT    = mticker.FuncFormatter(lambda x, _: f"{x:.0%}")
_DOLLAR = mticker.FuncFormatter(lambda x, _: f"${x:.2f}")

# ── 1. Cumulative wealth ──────────────────────────────────────────────────────
for key, (ret, label) in perf_map.items():
    cum  = (1 + ret).cumprod()
    lw   = 1.5 if key == "SPY B&H" else 2.5
    ls   = "--" if key == "SPY B&H" else "-"
    ax_cum.plot(cum.index, cum.values, label=label,
                color=COLORS[key], lw=lw, ls=ls)
ax_cum.set_yscale("log")
ax_cum.yaxis.set_major_formatter(_DOLLAR)
ax_cum.set_title("Cumulative Wealth  (log scale)", fontweight="bold")
ax_cum.legend(fontsize=9, framealpha=0.9)
ax_cum.grid(True, alpha=0.3)

# ── 2. Posterior mean ± 2σ by T-k ────────────────────────────────────────────
plot_k   = np.arange(min(MAX_K + 1, 22))
mu_plot  = mu_final[plot_k] * 100
sig_plot = sig_final[plot_k] * 100

ax_post.fill_between(plot_k, mu_plot - 2*sig_plot, mu_plot + 2*sig_plot,
                     alpha=0.20, color=COLORS["Bayes WML"], label="±2σ credible band")
ax_post.fill_between(plot_k, mu_plot - sig_plot,   mu_plot + sig_plot,
                     alpha=0.35, color=COLORS["Bayes WML"], label="±1σ credible band")
ax_post.plot(plot_k, mu_plot, color=COLORS["Bayes WML"], lw=2, label="Posterior mean")
ax_post.axhline(0, color="black", lw=0.7, ls="--")
ax_post.axvspan(WINDOW_END - 0.5, WINDOW_START + 0.5,
                alpha=0.10, color="orange",
                label=f"Paper window (T-{WINDOW_START}→T-{WINDOW_END})")
ax_post.invert_xaxis()
ax_post.set_xlabel("T-k  (days before month-end)")
ax_post.set_ylabel("Expected WML return (%)")
ax_post.set_title("Learned Posterior Mean ± Credible Bands", fontweight="bold")
ax_post.legend(fontsize=7)
ax_post.grid(True, alpha=0.3)

# ── 3. Learned allocation by T-k ──────────────────────────────────────────────
alloc_plot  = alloc_fin[plot_k]
bar_colors  = [COLORS["Bayes WML"] if a > 0.05 else
               (COLORS["Fixed WML"] if a < -0.05 else "#B0BEC5")
               for a in alloc_plot]
ax_alloc.bar(plot_k, alloc_plot, color=bar_colors, edgecolor="white", width=0.7)
ax_alloc.axhline(0, color="black", lw=0.7)
ax_alloc.axvspan(WINDOW_END - 0.5, WINDOW_START + 0.5,
                 alpha=0.10, color="orange", label=f"Paper window")
ax_alloc.invert_xaxis()
ax_alloc.set_xlabel("T-k  (days before month-end)")
ax_alloc.set_ylabel("Learned allocation  (2·Φ(z)−1)")
ax_alloc.set_title("Bayesian Allocation Signal by Day Position", fontweight="bold")
ax_alloc.set_ylim(-1.05, 1.05)
ax_alloc.legend(fontsize=7)
ax_alloc.grid(True, alpha=0.3, axis="y")

# ── 4. Drawdown ───────────────────────────────────────────────────────────────
for key, (ret, label) in perf_map.items():
    cum = (1 + ret).cumprod()
    dd  = (cum - cum.cummax()) / (cum.cummax() + 1e-9)
    ax_dd.plot(dd.index, dd.values, color=COLORS[key], lw=1.5, label=label,
               ls="--" if key == "SPY B&H" else "-")
ax_dd.yaxis.set_major_formatter(_PCT)
ax_dd.set_title("Drawdown", fontweight="bold")
ax_dd.legend(fontsize=7)
ax_dd.grid(True, alpha=0.3)

# ── 5. Rolling 6-month Sharpe ─────────────────────────────────────────────────
for key, (ret, label) in perf_map.items():
    exc = ret - RF / 252
    rs  = exc.rolling(126).mean() / (exc.rolling(126).std() + 1e-9) * 252**0.5
    ax_sr.plot(rs.index, rs.values, color=COLORS[key], lw=1.5, label=label,
               ls="--" if key == "SPY B&H" else "-")
ax_sr.axhline(0, color="black", lw=0.6, ls="--")
ax_sr.axhline(1, color="green", lw=0.5, ls=":", alpha=0.6)
ax_sr.set_title("Rolling 6-Month Sharpe", fontweight="bold")
ax_sr.legend(fontsize=7)
ax_sr.grid(True, alpha=0.3)

# ── 6. Allocation evolution: window positions (T-4 … T-9) over time ──────────
# Reconstruct from monthly snapshots
snap_dates   = sorted(snapshots.keys())
window_ks    = list(range(WINDOW_END, WINDOW_START + 1))
alloc_evol   = pd.DataFrame(
    {k: [snapshots[d]["alloc"][k] for d in snap_dates] for k in window_ks},
    index=snap_dates,
)
for k in window_ks:
    ax_evol.plot(alloc_evol.index, alloc_evol[k],
                 label=f"T-{k}", lw=1.4)
ax_evol.axhline(0, color="black", lw=0.6, ls="--")
ax_evol.axhline(0.5,  color="green", lw=0.5, ls=":", alpha=0.6)
ax_evol.axhline(-0.5, color="red",   lw=0.5, ls=":", alpha=0.6)
ax_evol.set_ylim(-1.05, 1.05)
ax_evol.set_title("Allocation Evolution: Window Positions", fontweight="bold")
ax_evol.set_ylabel("Bayesian allocation  (2·Φ(z)−1)")
ax_evol.legend(fontsize=7, ncol=2)
ax_evol.grid(True, alpha=0.3)

# ── 7. P(θ_k > 0) heatmap over time ─────────────────────────────────────────
# Build matrix: rows = dates, columns = k positions
prob_matrix = np.array([
    [norm.cdf(snapshots[d]["mu"][k] / snapshots[d]["sig"][k])
     for k in range(min(MAX_K + 1, 22))]
    for d in snap_dates
])
# Only plot once enough data has accumulated
warm_mask  = np.array([snapshots[d]["n"].sum() > 50 for d in snap_dates])
plot_dates = [d for d, m in zip(snap_dates, warm_mask) if m]
plot_prob  = prob_matrix[warm_mask]

if len(plot_dates) > 0:
    im = ax_prob.imshow(
        plot_prob.T,
        aspect="auto", origin="lower",
        extent=[0, len(plot_dates), 0, min(MAX_K + 1, 22)],
        cmap="RdYlGn", vmin=0.0, vmax=1.0,
    )
    plt.colorbar(im, ax=ax_prob, label="P(θ_k > 0 | data)")
    n_dates = len(plot_dates)
    tick_idx = np.linspace(0, n_dates - 1, min(6, n_dates), dtype=int)
    ax_prob.set_xticks(tick_idx)
    ax_prob.set_xticklabels([plot_dates[i].strftime("%Y-%m") for i in tick_idx],
                            rotation=30, ha="right", fontsize=7)
    ax_prob.axhline(WINDOW_END,     color="orange", lw=1.5, ls="--")
    ax_prob.axhline(WINDOW_START+1, color="orange", lw=1.5, ls="--",
                    label=f"Paper window T-{WINDOW_START}→T-{WINDOW_END}")
    ax_prob.set_ylabel("T-k position")
    ax_prob.set_title("P(θ_k > 0 | data) Heatmap Over Time\n"
                      "(green = Bayesian believes positive WML expected)", fontweight="bold")
    ax_prob.legend(fontsize=7)

fig.savefig(RESULTS_DIR / "intramonth_bayesian.png", dpi=150, bbox_inches="tight")
plt.close(fig)
logger.info("Saved: results/intramonth_bayesian.png")

# ── Save returns ───────────────────────────────────────────────────────────────
out_df = pd.DataFrame({k: v for k, (v, _) in perf_map.items()})
out_df.to_csv(RESULTS_DIR / "intramonth_bayesian_returns.csv")
logger.info("Saved: results/intramonth_bayesian_returns.csv")

print("\n✓  Done.  Results in results/")
