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
| `build_etf_universe.py` | ~1,300 → top-250 ETFs | **Builds the large ETF universe.** financedatabase (universe + info) + yfinance (history), screened for coverage & dollar-volume liquidity → `datasets/etf_universe_*`. Network; run once. |
| `build_equity_universe.py` | ~820 → top-500 equities | **Builds the large equity universe.** US large/mega-cap names from financedatabase + yfinance, same liquidity screen → `datasets/equity_universe_*`. Network; run once. ⚠️ current-membership universe → survivorship-biased history. |
| `run_large_universe.py` | 250 liquid US ETFs | Universe summary from fd info + diversified cross-asset allocation on a basket built from that info (most-liquid ETF per asset class) vs SPY. Offline. |
| `run_equity_cross_section.py` | 500 liquid US equities | Exploratory 12-1 cross-sectional momentum (monthly decile L/S) vs SPY, gross/net of turnover cost. Offline. ⚠️ survivorship-biased universe. |
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
| `run_net_liquidity.py` | SPY/TLT + FRED | Net liquidity (Fed assets − TGA − RRP) vs equities: horizon lead-lag scan + three liquidity-driven strategies (binary on/off, scaled exposure, SPY↔TLT rotation) vs SPY. Offline (cached); `--build` refreshes FRED (needs `FRED_API_KEY`). |
| `run_gamma_exposure.py` | SPX/SPY options | Dealer GEX snapshot via the CBOE CDN (exchange greeks + OI, no key; SPX default) or yfinance: total gamma, long/short-gamma regime, zero-gamma flip, per-strike profile. Network. `--log` appends to `datasets/gex_snapshots.csv` (run daily to accumulate a GEX history, since free historical OI is scarce). |
| `run_net_liquidity_universe.py` | 250 ETFs + 500 equities + FRED | Where the net-liquidity effect lives: per-asset & per-asset-class sensitivity (ETF universe) + per-sector sensitivity (equity universe), most/least-sensitive names, and an out-of-sample regime-spread check. Offline. |
| `run_liquidity_trend.py` | SPY + FRED | Net liquidity × 200-day price trend: each signal alone, an AND gate, and a vote (0/½/1 exposure). The vote blend tops both standalone signals and SPY on Sharpe/Sortino. Offline. |
| `run_liquidity_predictive.py` | SPY + FRED + VIX/HYG | Net liquidity × *forward-looking* layers (VIX term structure, HYG−IEF credit appetite) instead of the lagging MA: pairwise and three-way votes vs the trend baseline and SPY. Best blend of the thread (3-way vote Sharpe 0.75, DD −20%). Offline. |
| `run_liquidity_continuous.py` | SPY + cyclical basket | Continuous fit-free vote (soft-sign strengths, no thresholds) on SPY and the XLY/XLK/XLF/XLE/XLI sleeve, with rolling 3y OOS win rates and per-year edge table. Best drawdown/Calmar of the thread; overlay beats its underlying in 60–68% of windows. Offline. |
| `run_liquidity_levered.py` | SPY + QQQ | Levering the vote for absolute return: four exposure mappings (vol-target, 2×vote, convex, vol-braked) with honest financing/TC. 2×vote on QQQ beats SPY's CAGR in 84% of 3y windows (median +8.4pp/yr) and QQQ's in 78% — at −48% max DD. Offline. Uses `research.overlay.liquidity_vote`. |
| `run_liquidity_4layer.py` | QQQ + FRED + VIX/UUP + 500 equities | Fourth vote layer head-to-head against the 2022 grinding-bear hole: dollar (−UUP 63d) vs equity breadth. Dollar wins — cuts 2022 loss and max DD, lifts the rolling CAGR win rate vs SPY to 88% with no Sharpe loss; breadth too slow, rejected. Offline. |
| `run_liquidity_stress.py` | QQQ/SPY/EEM + cyclicals, 2007→ | Stress-tests the 4-layer 2×vote: GFC-inclusive sample (live 2008-04, −8% through Sep08–Jun09 vs SPY −27%; survives the one non-QE-rescue crisis) passes; diversifying the underlying to a QQQ+EEM+cyclicals basket fails (deeper DD, lower return). Offline. |
| `run_liquidity_interaction.py` | SPY + FRED + VIX | Is liquidity state-dependent? corr(Δ6m liq, next-6m SPY) by causal VIX-percentile regime: +0.34 fragile vs −0.11 calm — a real conditional structure, but a naive fragile-only switch still trails buy & hold. Actionable as signal *weighting*, not a switch. Offline. |
| `run_liquidity_regime_weighted.py` | QQQ + FRED + VIX/UUP | Cashes in the interaction finding: liquidity layer weight = fragility rank inside the 4-layer 2×vote. Centered (w_liq = 2f) nudges the champion to its best OOS scores (90% CAGR / 74% Sharpe win vs SPY, Calmar 0.51→0.53); calm-mute (w_liq = f) is worse. Marginal refinement, not a breakthrough — 2022 unchanged. Offline. |
| `run_liquidity_disagreement.py` | QQQ + FRED + VIX/UUP | Cross-layer *disagreement* (std of the 4 strengths) as a risk gauge and throttle. Nearly VIX-orthogonal, yet high d cuts fwd 6m return +13%→+6%, fattens the 5th-pct tail +1%→−23% and survives a VIX-held-fixed control. Throttling the 2×vote by confidence crushes 2022 (−39%→−8%) and DD (−42%→−18%) at preserved Sharpe; the causally renormalized version recovers champion CAGR with the thread's best OOS consistency (97% CAGR-win vs SPY). Offline. |
| `run_model_disagreement.py` | QQQ + FRED + VIX/UUP | Mines the strategy graveyard: 5 model stances (liquidity vote, trend, mean-reversion, BOCPD regime age, walk-forward HMM) → cross-MODEL dispersion as a "model-implied uncertainty" gauge. **Hypothesis inverts**: high disagreement = strong trends = *good* times (fwd 6m +13% vs +8%, thinner tail), and 200d-MA flips follow trend/meanrev *agreement* (28% vs 7%) — near transitions all models drift to neutral, so dispersion is low. Indecision, not conflict, is the tell. Agreement-throttles fail; the post-hoc inverted throttle (2022 −5%, DD −21%, Sharpe 0.83) is flagged as unvalidated. Offline. |
| `run_overnight_split.py` | QQQ/SPY OHLC + FRED + VIX/UUP | Splits close-to-close into overnight (close→open) and intraday (open→close). The anomaly holds (QQQ overnight-only Sharpe 0.70 vs 24h 0.37, DD −34% vs −83%) and the sharp hypothesis **confirms: liquidity loads on the overnight leg** (corr +0.19 vs +0.03 intraday; vote terciles monotone overnight only). But the trade fails honestly: 2 trades/day ≈ 5–10%/yr drag, and post-2010 the overnight leg's DD advantage vanished (2022's bear was an overnight phenomenon: −23%). A diagnostic win, not a strategy. Offline (cached `overnight_panel.csv.gz`; `--refresh` refetches). |

### PEAD (Post-Earnings-Announcement-Drift) — run in order

| script | what it does |
|--------|--------------|
| `download_finance_data.py` | One-time: download the FinanceDatabase equity universe → `financial_data/`. Requires `pip install -e .[pead]`. |
| `run_pead.py` | Stratified-sample the universe, fetch earnings+prices (checkpoint/resume), build the SUE panel, run Fama-MacBeth IC tests overall + per cap bucket. Writes `results/pead/signal_panel.csv`. |
| `walk_forward_pead.py` | Walk-forward long-short validation on the panel produced by `run_pead.py`. |

`main.py` and the PEAD runners require network access (yfinance / FinanceDatabase);
the other portfolio studies run offline against the bundled `datasets/` files.

`_bootstrap.py` is a shared shim, not a study — import it first in every script.
