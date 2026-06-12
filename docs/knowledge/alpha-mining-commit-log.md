# Alpha Mining in This Repo — Work Log from Commit History

> Purpose: a durable record of the strategy-research line developed in this repository, reconstructed from the commit history, so future sessions don't have to re-derive what was tried, what was kept, and what was removed. Read alongside `systematic-equity-strategies.md` for the external research context.

## Timeline (oldest → newest)

| Commit | Work |
|---|---|
| `c69de85` | Initial **PosteriorAlpha** backtesting system. |
| `76f318c` | `run_local.py` — CSV-based backtest entry point (no live data dependency). |
| `3a9a3bc` | **AMR strategy** (Adaptive Market Regime) + refined Bayesian extensions; 14-asset universe. |
| `6964561` | Fix: replaced Viterbi decoding with **forward-filtered states** in `RegimeHMM` — Viterbi uses future information and is a look-ahead bias in backtests. |
| `7b51575` | Added **BOCPD** (Bayesian Online Change Point Detection) and a **3-state HMM** regime model, plus hybrid AMR strategies combining them. |
| `e680722` | `run_real_only.py` — backtest restricted to 5 real ETFs (SPY, TLT, GLD, EEM, VNQ), dropping synthetic assets. |
| `83bda78` | **BOCPD-AMR v2** — CVaR objective, multi-asset BOCPD, low-volatility tilt, adaptive leverage. |
| `7611418` | **BOCPD-AMR v3** — continuous λ via SPY **Omega ratio** + amplified sigmoid. |
| `2d4e26e` | **BOCPD-AMR v4** — ERL-adaptive EWMA halflife for the Omega estimate. |
| `9fab14e` | `run_spy_timing.py` — SPY-only risk-on/risk-off **ablation** isolating the timing signal from multi-asset rotation. |
| `00c4498` | Data upload: `sp500_top100_adj_close.csv` (top-100 S&P adjusted closes — stock-level universe for future cross-sectional work). |

## Strategy evolution: BOCPD-AMR v1 → v4

The core research line is a regime-aware allocation model on a 5-ETF universe (SPY, TLT, GLD, EEM, VNQ), 2016-01-01 → 2024-12-31, weekly (Friday) rebalancing, 5 bps transaction cost per unit of turnover.

- **v1 (`7b51575`):** BOCPD change-point probability drives a hand-tuned risk parameter: `lam = 0.55 − 0.30 × clip(cp × 30, 0, 1)`.
- **v2 (`83bda78`):** replaced the mean-variance objective with **CVaR**, ran BOCPD per-asset (multi-asset), added a low-vol tilt and adaptive leverage.
- **v3 (`7611418`):** replaced the hand-tuned λ formula with a data-driven calibration:
  `lam = Omega(r, rf) / (1 + Omega) × credibility(cp)`, where `Omega = mean(max(r − rf, 0)) / mean(max(rf − r, 0))` on a 252-day window, and the change-point probability shrinks λ toward 0.5 right after a regime break (the old Omega estimate is no longer valid).
- **v4 (`2d4e26e`):** the v3 diagnostic showed two components hurt: the ERL credibility discount fights Omega (removed), and the Kelly guard on short Sharpe windows fired too many false positives (removed). Pure EWMA Omega is self-sufficient. The kept improvement is an **ERL-adaptive EWMA halflife** (Keating & Shadwick 2002; Engle & Patton 2001):
  `halflife = clip(erl / 3, 14, 84)` — fresh regime (ERL ≈ 10d) → halflife 14 so Omega reacts fast; standard regime (ERL ≈ 126d) → halflife 42 (matches v3); long stable regime (ERL ≈ 252d) → halflife 84 for stability. Net effect: defensive faster in crashes, less whipsaw in prolonged bulls.

## The SPY timing ablation (`9fab14e`)

`run_spy_timing.py` asks whether the timing signal alone (no multi-asset rotation) beats SPY buy & hold. Three SPY-only strategies:

1. Buy & hold (baseline);
2. Binary timing — 100% SPY when λ > 0.5, else cash;
3. Proportional timing — allocation = `clip((λ − 0.25) / 0.50, 0, 1)`.

Signal: EWMA Omega on SPY with the v4 ERL-adaptive halflife, mapped via `λ = sigmoid(2.5 × log(Omega_ewma))`, clipped to [0.10, 0.80]. This isolates the value of cash-switching vs. the value of cross-asset rotation in the full AMR strategies.

## Code map

- `src/regime_models.py` — BOCPD (`precompute_bocpd`, `precompute_bocpd_multi`) and 3-state HMM.
- `src/amr.py` — AMR allocation logic, `compute_continuous_lam` (Omega → λ mapping).
- `src/hmm_filter.py` — forward-filtered HMM states (post-Viterbi fix).
- `src/backtest.py`, `src/metrics.py` — backtest engine and performance metrics.
- `run_bocpd_v2/v3/v4.py`, `run_spy_timing.py`, `run_real_only.py`, `run_refined.py`, `run_local.py` — experiment entry points; each script's docstring records its hypothesis and changes vs. the previous version.

## Lessons already paid for (don't relearn)

1. **Look-ahead bias is easy to introduce via decoders:** Viterbi smoothing uses the full sample; only forward-filtered state probabilities are valid in a backtest (`6964561`).
2. **Data-driven calibration beat the hand-tuned formula:** the Omega-ratio λ (v3) replaced magic constants and survived; prefer calibrations with an economic estimator behind them.
3. **Removing components is progress:** v4 improved by deleting the ERL credibility discount and the Kelly guard, keeping a single adaptive mechanism. Complexity that fights the main signal is negative alpha.
4. **Run ablations:** the SPY-only timing test separates the regime signal's value from rotation's value — the same discipline the knowledge base prescribes (isolate each component before crediting the combination).

## Hygiene gaps vs. the knowledge base (open items)

Checked against `systematic-equity-strategies.md` §5; none of these have been done yet:

- No **rebalance-offset robustness** check (all backtests rebalance on Fridays only).
- No **deflated Sharpe ratio** / trial-count accounting across the v1–v4 variants (each version is a trial).
- No **sub-period stability** or parameter-neighborhood perturbation reporting.
- The new `sp500_top100_adj_close.csv` enables cross-sectional (stock-level) signals, where IC analysis before backtesting should be the first step.
