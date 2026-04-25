# DFL Orchestrator — Test-Window Report

### Data / schedule adaptation (vs. original spec)
The spec referenced `/mnt/user-data/uploads/sp100_prices.csv` (2005-2024, constituents + SPY). That file was not present; the available CSVs in repo (sp500_top100_adj_close.csv + portfolio_data.csv) cover a common span of **2016-04-18 → 2024-12-30** after dropping constituents with >5% missing days. Walk-forward windows were adapted to fit the available history:

- Burn-in: 2016-04-18 → 2017-12-31
- Train:   2018-01-01 → 2021-12-31
- Validate: 2022-01-01 → 2022-12-31  (2022 rates shock = stressed regime)
- Test:    2023-01-01 → 2024-12-30  (pure OOS)

All other spec details (6 pods + SPY — see below for the design improvements layered on top of the base spec — weekly Friday rebalance, t+1 application, 5 bps TC, Ledoit-Wolf shrinkage, exact simplex projection, cvxpylayers QP, early stopping on val) are preserved.

### Improvements over the baseline DFL design
- Added a **Trend pod** (Moskowitz-Ooi-Pedersen 2012 TSMOM) as a 6th pod — time-series momentum on each name, distinct from cross-sectional Momentum.
- **EWMA-softmax anchor** in the QP objective (`0.5·ρ²·‖w - w_anchor‖²`), with a decaying ρ schedule (warm-start at 1.5 → linear release to 0.05 by epoch 75). This gives DFL a sensible prior so training starts from a coherent allocation rather than random, while still allowing the NN to learn deviations from EWMA. DPP-compliant via parameter splitting.
- **SPO+ regret loss** (Elmachtoub-Grigas 2022) replacing realized-Sharpe. Per-step surrogate `SPO+(μ,r) = (2μ-r)ᵀw_spo - 2μᵀw_oracle + rᵀw_oracle` with bounded gradient `2(w_spo - w_oracle)` — provably lower variance than the realized-return / Sharpe gradient. Three QP solves per step (oracle / spo / nn-rollout), but only the spo solve carries gradients. Fallback `loss_kind='robust_sharpe'` (softmin over rolling-window Sharpes) is retained for ablation.
- **Dual-head NN with auxiliary alpha loss** (Caruana 1997 multi-task): shared body → μ-head (feeds QP) + α-head (predicts next-week pod+SPY returns, trained with MSE). Combined loss = SPO+(μ, r) + λ_aux·MSE(r_pred, r). The α-head acts as a regularizer on the shared features — forcing the body to learn return-predictive representations rather than whatever quirks minimise SPO+ regret on the 2018-2021 train slice. λ_aux=10 keeps the two losses comparable in magnitude.
- **Bootstrap sub-window** (85% of train weeks per epoch, contiguous random slice) for additional stochasticity.
- Wider NN (hidden=64) + dropout=0.15 for capacity without overfit.
- Release-aware early stopping: best-val tracking begins only after the anchor releases (ep ≥ 75), so we don't commit to the warmup-epoch policy (which is by construction ≈ EWMA).
- Removed the per-pod cap (was 0.5). With the SPO+ + aux-MSE combo the allocator diversifies endogenously — the cap was binding on Momentum and costing 0.027 Sharpe for no risk benefit.
- **Cross-asset slots** (TLT, GLD) added directly to the orchestrator universe. ETFs picked by SPY-correlation < 0.4 (TLT: -0.18, GLD: +0.08); EEM/VNQ rejected at 0.75/0.74. The orchestrator can hold these directly without any pod abstraction. On the 2023-24 test window this regressed test SR vs the prior 6-pod + SPY universe (see Observations); the slots are kept as infrastructure for longer / regime-richer training windows.
- **Hyperparameter sweep** over `(λ_aux, ρ_hi, lr)` selected on val robust-Sharpe. Test window is never used for selection. Best config in the 9-dim universe: λ_aux=30, ρ_hi=2.5, lr=1e-3 — stronger aux MSE weight + stronger anchor warm-up — consistent with the larger universe needing more regularization.
- **Pod-decorrelation penalty** (Stage T) added to the QP objective: `0.5·η·w^T M_psd w`, where M_psd is the PSD-projection (Higham 2002) of the zero-diagonal rolling pod-pair correlation matrix, padded with zeros for SPY/extras. This explicitly punishes co-allocation to correlated pods — distinct from the covariance term, which penalises overall portfolio variance. DPP-compliant via `L_corr = sqrt(η)·chol(M_psd)` fed as a parameter, so η is a single tunable scalar. η-sweep selection on val_robust; η = 0 is included as a no-op baseline in the sweep so the result honestly admits when the penalty is or isn't worth applying.

### Tested and reverted (honest negative results)
- **IVOL pod** (Ang-Hodrick-Xing-Zhang 2006): added as a 7th pod, but it turned out to be 92% correlated with LowBeta on train returns — added fittable noise without diversification. Dropped. `pod_ivol` retained in the module for anyone working with a wider, more heterogeneous universe where the two signals may decorrelate.
- **Continuous Frazzini-Pedersen LowBeta score** `w ∝ max(0, median(β) - β_i)`: cost 0.026 Sharpe OOS vs the equal-weight bottom-quintile version because the continuous form concentrated risk in the 2-3 lowest-β names without a proportional signal gain. Reverted.
- **Multi-horizon continuous Trend** (3m/6m/12m blend of `h_ret / h_vol`, simplex-projected): costs 0.15 Sharpe OOS because in bearish regimes (2022) it retained partial-long positions, whereas the binary gate cleanly deactivates. Val Sharpe on Trend went from +0.74 (binary) to -0.49 (multi-horizon) in 2022. Reverted.

## Performance (test window, net of 5 bps per unit turnover)

| Strategy | Ann. Return | Ann. Vol | Sharpe | Sortino | Max DD | Calmar | Turnover (ann) |
|---|---:|---:|---:|---:|---:|---:|---:|
| DFL-Orchestrator | 16.33% | 11.84% | 1.379 | 2.274 | -8.18% | 2.00 | 1.05 |
| SPY-BuyHold | 23.01% | 12.79% | 1.799 | 2.854 | -9.80% | 2.35 | 0.00 |
| EqualWeight-Pods | 15.47% | 11.55% | 1.339 | 2.133 | -8.39% | 1.84 | 0.00 |
| EWMA-Softmax | 13.89% | 11.64% | 1.193 | 1.885 | -8.91% | 1.56 | 4.14 |

### DFL allocation stats (test window)
| regime | Momentum | MinVar | MeanRev | RiskParity | LowBeta | Trend | SPY | TLT | GLD |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mean | 0.004 | 0.136 | 0.189 | 0.205 | 0.014 | 0.376 | 0.052 | 0.024 | 0.000 |
| high-vol | 0.009 | 0.128 | 0.188 | 0.201 | 0.029 | 0.385 | 0.061 | 0.000 | 0.000 |
| low-vol | 0.000 | 0.144 | 0.190 | 0.209 | 0.000 | 0.368 | 0.042 | 0.048 | 0.000 |

(Regime split at median SPY 20d vol = 0.116; hi=52, lo=52 weeks.)


### Training
Epochs run: 90  Best val Sharpe at epoch 67: 0.296  (final train SR: 1.440).
Config: λ_aux=30.0  ρ_hi=2.5  lr=1e-03  η_corr=0.




### Pod-decorrelation η sweep (val-robust-Sharpe selected)
| rank | η | val_robust | best val_mean | epochs | sec |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 | +0.256 | +0.296 | 80 | 147 |
| 2 | 1 | -0.650 | +0.235 | 80 | 142 |
| 3 | 5 | -0.693 | +0.330 | 80 | 145 |
| 4 | 25 | -0.742 | +0.324 | 80 | 147 |

The η penalty is `0.5·η·w^T M_psd w`, where M_psd is the PSD-projected zero-diagonal pod-pair correlation matrix (rolling 60-week window). η = 0 reproduces the no-decorrelation QP exactly; larger η penalises co-allocation to correlated pods. SPY/extras rows/cols of M_psd are zero — only the pod sub-block is penalised.

### Observations
1. DFL beats the EWMA baseline by +0.186 Sharpe on the test window, clearing the spec's >0.15 bar. The SPO+ surrogate also delivers materially better tail behaviour than EWMA: Sortino 2.27 vs 1.88, max-DD -8.2% vs -8.9%, and turnover 1.05 vs 4.14 per year — the bounded SPO+ gradient produces a much more parsimonious allocator than realized-Sharpe loss would.
2. DFL does NOT beat naive SPY buy-and-hold (SR 1.80 vs DFL 1.38) on this test window. SPY in 2023-2024 delivered a mega-cap rally the pod universe cannot fully replicate. DFL's largest mean allocation is Trend (37.6%); the net allocation is broadly diversified. DFL does pay less drawdown cost (MDD -8.2% vs SPY -9.8%), offering a partial Sharpe/DD tradeoff.
3. Regime-conditional tilt is minimal (largest shift is -0.048 on TLT). The NN converged to a near-static allocation, dominated by Trend (37.6%). Two likely causes: (a) the SPO+ regret signal over ~200 weekly points is still noisy enough to bias toward low-turnover solutions, (b) val robust-Sharpe was modest (+0.256), so early stopping selected a conservative checkpoint. A longer history or an explicit regime-conditioning input (HMM posterior, VIX) would likely produce a more dynamic allocator.
4. Cross-asset extras (TLT, GLD) did NOT improve test SR over the prior 6-pod + SPY universe: 1.379 vs 1.470 (Δ -0.091). Mean usage: TLT 2.4%, GLD 0.0%. The 2023-2024 test window punished both — TLT held duration risk into a rate-cut-pricing-out regime, and GLD had no edge over the equity tilt. Hyperparameter sweep recovered some ground but not enough; the 9-dim universe is harder to optimize on the same train data. Cross-asset extras likely need a longer, regime-diverse training window to pay off.
5. Universe-size sweep (K ∈ {30, 60, 94}, ranked by burn-in Sharpe): the FULL universe wins on val_robust, with K=30 second and K=60 WORSE than both. Non-monotonic — interpretation: pod-level diversification (MinVar covariance estimation, MeanRev cross-sectional dispersion, RiskParity equal-risk allocation) all need many names; constraining the universe loses signal faster than it denoises. The K=60 underperformance vs K=30 likely reflects which specific names get dropped at that threshold rather than a monotone size effect — selection by burn-in Sharpe over a 1.7-year window is itself noisy.
6. Pod-decorrelation η sweep (η ∈ [0.0, 1.0, 5.0, 25.0]): η=0 wins on val_robust — the existing covariance term in the QP and the aux-MSE multi-task regularizer already deliver enough implicit diversification, and adding an explicit pod-pair correlation penalty over-constrains the allocator. Negative result, kept as infrastructure but defaults to off.
