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
| `main.py` | S&P 500 (live download) | Multi-seed robustness of the Bayesian family vs. SPY; saves dashboards, multi-seed bars, λ heatmap |
| `run_local.py` | 5 ETFs + 9 synthetic = 14 | Full Bayesian × AMR comparison on an expanded factor-model universe |
| `run_real_only.py` | 5 real ETFs | Headline comparison on real data only — no synthetic assets |
| `run_refined.py` | 14 (synthetic-expanded) | Newer models only (`bocpd_amr`, `hmm3_amr`) over a shorter, faster window |
| `run_bocpd_v2.py` | 5 real ETFs | BOCPD-AMR v2: multi-asset BOCPD + CVaR + low-vol tilt + adaptive leverage |
| `run_bocpd_v3.py` | 5 real ETFs | BOCPD-AMR v3: continuous λ via the Omega ratio |
| `run_bocpd_v4.py` | 5 real ETFs | BOCPD-AMR v4: ERL-adaptive EWMA halflife for Omega |
| `run_spy_timing.py` | SPY only | Can a simple risk-on/risk-off rule on SPY beat buy-and-hold? |

`main.py` requires network access (yfinance); all others run offline against the
bundled `datasets/` files.

`_bootstrap.py` is a shared shim, not a study — import it first in every script.
