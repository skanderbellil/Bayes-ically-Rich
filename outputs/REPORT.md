# DFL Orchestrator — Test-Window Report

### Data / schedule adaptation (vs. original spec)
The spec referenced `/mnt/user-data/uploads/sp100_prices.csv` (2005-2024, constituents + SPY). That file was not present; the available CSVs in repo (sp500_top100_adj_close.csv + portfolio_data.csv) cover a common span of **2016-04-18 → 2024-12-30** after dropping constituents with >5% missing days. Walk-forward windows were adapted to fit the available history:

- Burn-in: 2016-04-18 → 2017-12-31
- Train:   2018-01-01 → 2021-12-31
- Validate: 2022-01-01 → 2022-12-31  (2022 rates shock = stressed regime)
- Test:    2023-01-01 → 2024-12-30  (pure OOS)

All other spec details (5 pods, weekly Friday rebalance, t+1 application, 5 bps TC, Ledoit-Wolf shrinkage, exact simplex projection, cvxpylayers QP with TC penalty, -Sharpe loss, early stopping on val) are preserved.

## Performance (test window, net of 5 bps per unit turnover)

| Strategy | Ann. Return | Ann. Vol | Sharpe | Sortino | Max DD | Calmar | Turnover (ann) |
|---|---:|---:|---:|---:|---:|---:|---:|
| DFL-Orchestrator | 14.87% | 10.74% | 1.385 | 2.147 | -7.10% | 2.09 | 0.17 |
| SPY-BuyHold | 23.01% | 12.79% | 1.799 | 2.854 | -9.80% | 2.35 | 0.00 |
| EqualWeight-Pods | 15.20% | 11.47% | 1.325 | 2.115 | -8.50% | 1.79 | 0.00 |
| EWMA-Softmax | 13.39% | 11.58% | 1.156 | 1.789 | -9.11% | 1.47 | 4.49 |

### DFL allocation stats (test window)
| regime | Momentum | MinVar | MeanRev | RiskParity | LowBeta | SPY |
|---:|---:|---:|---:|---:|---:|---:|
| mean | 0.167 | 0.000 | 0.167 | 0.021 | 0.477 | 0.167 |
| high-vol | 0.167 | 0.000 | 0.167 | 0.041 | 0.458 | 0.167 |
| low-vol | 0.167 | 0.000 | 0.168 | 0.001 | 0.497 | 0.167 |

(Regime split at median SPY 20d vol = 0.118; hi=52, lo=52 weeks.)


### Training
Epochs run: 21  Best val Sharpe at epoch 11: -0.079  (final train SR: 1.100).

### Observations
1. DFL beats the EWMA baseline by +0.228 Sharpe on the test window, clearing the spec's >0.15 bar.
2. DFL does NOT beat naive SPY buy-and-hold (SR 1.80) on this test window. SPY in 2023-2024 delivered a mega-cap-driven rally that the pod universe cannot fully replicate. A diversified pod mix trades absolute Sharpe against drawdown / tail-risk resilience (DFL MDD -7.1% vs SPY MDD -9.8%).
3. Regime-conditional tilt is minimal (largest shift is +0.040 on RiskParity). The NN converged to a near-static allocation — dominated by LowBeta (47.8%) with the remaining weight spread across Momentum/MeanRev/SPY at the diversification floor. Two likely causes: (a) -Sharpe over ~200 weekly points is a high-variance training signal that biases toward low-turnover solutions, (b) validation-window Sharpe was negative (-0.079), so early stopping selected a conservative checkpoint. A longer history, lower-variance loss (e.g. Smart Predict-then-Optimize per Elmachtoub-Grigas 2022), or an explicit regime-conditioning input would likely produce a more dynamic allocator.
