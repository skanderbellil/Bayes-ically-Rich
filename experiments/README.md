# Experiments

Reproducible studies that wire the `posterioralpha` pipeline stages together.
Each script loads data, runs one or more backtests, prints a metrics table, and
writes plots/CSVs to `results/`.

Run from anywhere — `_bootstrap.py` puts the repo root on `sys.path`:

```bash
python experiments/run_real_only.py
```

| script | universe | what it studies |
|--------|----------|-----------------|
| `build_etf_universe.py` | ~1,300 → top-250 ETFs | **Builds the large universe.** financedatabase (universe + info) + yfinance (history), screened for coverage & dollar-volume liquidity → `datasets/etf_universe_*`. Network; run once. |
| `run_large_universe.py` | 250 liquid US ETFs | Universe summary from fd info + diversified cross-asset allocation on a basket built from that info (most-liquid ETF per asset class) vs SPY. Offline. |
| `main.py` | S&P 500 (live download) | Multi-seed robustness of the Bayesian family vs. SPY; saves dashboards, multi-seed bars, λ heatmap |
| `run_local.py` | 5 ETFs + 9 synthetic = 14 | Full Bayesian × AMR comparison on an expanded factor-model universe |
| `run_real_only.py` | 5 real ETFs | Headline comparison on real data only — no synthetic assets |
| `run_refined.py` | 14 (synthetic-expanded) | Newer models only (`bocpd_amr`, `hmm3_amr`) over a shorter, faster window |
| `run_bocpd_v2.py` | 5 real ETFs | BOCPD-AMR v2: multi-asset BOCPD + CVaR + low-vol tilt + adaptive leverage |
| `run_bocpd_v3.py` | 5 real ETFs | BOCPD-AMR v3: continuous λ via the Omega ratio |
| `run_bocpd_v4.py` | 5 real ETFs | BOCPD-AMR v4: ERL-adaptive EWMA halflife for Omega |
| `run_spy_timing.py` | SPY only | Can a simple risk-on/risk-off rule on SPY beat buy-and-hold? |
| `run_market_neutral.py` | 5 real ETFs | Beta-hedged residual mean-reversion (market-neutral alpha sleeve) with BOCPD gating + vol targeting |
| `run_hurst_multiasset.py` | 5 real ETFs | Cross-sectional Hurst (R/S) trend ranking, top-K inverse-vol with IVOL-5 fallback (IS/OOS split) |
| `run_hurst_bull_adaptive.py` | SPY only | Hurst-bull SPY timing with adaptive (rolling-quantile) H/ERL thresholds + walk-forward |
| `run_intramonth_momentum.py` | 5 real ETFs | Nathan-Suominen-Tasa (2026): does WML momentum concentrate in the T-9→T-4 intramonth window? |
| `run_intramonth_bayesian.py` | 5 real ETFs | Learns the intramonth allocation per T-k position via an online Normal-Normal model (no hardcoded window) |
| `dfl_orchestrator.py` | S&P 100 + ETFs | Decision-focused multi-pod orchestrator: 6 asset-level pods + a differentiable mean-variance QP layer. Requires `pip install -e .[dfl]` |

### PEAD (Post-Earnings-Announcement-Drift) — run in order

| script | what it does |
|--------|--------------|
| `download_finance_data.py` | One-time: download the FinanceDatabase equity universe → `financial_data/`. Requires `pip install -e .[pead]`. |
| `run_pead.py` | Stratified-sample the universe, fetch earnings+prices (checkpoint/resume), build the SUE panel, run Fama-MacBeth IC tests overall + per cap bucket. Writes `results/pead/signal_panel.csv`. |
| `walk_forward_pead.py` | Walk-forward long-short validation on the panel produced by `run_pead.py`. |

`main.py` and the PEAD runners require network access (yfinance / FinanceDatabase);
the other portfolio studies run offline against the bundled `datasets/` files.

`_bootstrap.py` is a shared shim, not a study — import it first in every script.
