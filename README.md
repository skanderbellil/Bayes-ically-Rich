# PosteriorAlpha

**A Bayesian adaptive portfolio research framework — from strategy research to backtest and validation.**

PosteriorAlpha builds portfolios that *adapt* to changing market regimes. Instead of
trusting a single historical estimate, it continuously asks *"how much should I trust
recent data versus the long-run prior?"* and shifts allocations accordingly — via
Bayesian shrinkage, online change-point detection (BOCPD), and hidden-Markov regime
filters.

The codebase is organised as an explicit four-stage pipeline so any strategy can be
traced cleanly from idea to validated result:

```
   ┌──────────┐    ┌────────────┐    ┌────────────┐    ┌──────────────┐
   │  1. DATA │ ─▶ │ 2.RESEARCH │ ─▶ │ 3.BACKTEST │ ─▶ │ 4.VALIDATION │
   └──────────┘    └────────────┘    └────────────┘    └──────────────┘
   market access    models &           time-stepping     metrics &
   synthetic        optimisers &       engines (no        plots /
   loaders          regime signals     lookahead)         dashboards
```

---

## Repository layout

```
posterioralpha/                # the framework package
├── data/                      # ── Stage 1: DATA ──
│   ├── universe.py            #   large liquid ETF + equity universes: financedatabase (info) + yfinance (history)
│   ├── macro.py               #   FRED (keyed, .env): net liquidity + curated macro panel
│   ├── market.py              #   live download + S&P 500 universe (yfinance)
│   ├── synthetic.py           #   factor-model synthetic universe expansion
│   └── loaders.py             #   robust loaders for bundled datasets
├── research/                  # ── Stage 2: STRATEGY RESEARCH ──
│   ├── bayesian.py            #   moments, λ via Mahalanobis, posterior blend, optimisers
│   ├── amr.py                 #   AMR/CVaR/HRP optimisers, Ω-ratio λ, vol-targeting overlay
│   ├── gamma.py               #   dealer gamma exposure (GEX): CBOE/yfinance chains, zero-gamma flip, GEX series
│   └── regimes.py             #   RegimeHMM (2-state), BOCPD, HMM3 (3-state)
├── backtest/                  # ── Stage 3: BACKTEST ENGINES ──
│   ├── bayesian.py            #   monthly-rebalanced engine + BacktestResult
│   └── amr.py                 #   weekly-rebalanced engine + AMRResult + sensitivity sweep
├── validation/                # ── Stage 4: VALIDATION ──
│   ├── metrics.py             #   CAGR, Sharpe, Sortino, Max DD, Calmar, α/β/IR
│   └── plots.py               #   dashboards, multi-seed bars, λ heatmap
├── pead/                      # ── PEAD strategy module (spans all 4 stages) ──
│   ├── universe.py·fetch.py   #   data:     equity universe + earnings/price fetch
│   ├── signals.py·momentum.py·bayesvol.py   # research: SUE, momentum, Bayes vol
│   ├── walk_forward.py·backtest.py·corrected.py·costs.py  # backtest engines
│   └── fama_macbeth.py·research_utils.py     # validation: IC tests + metrics/plots
├── council/                   # ── blinded period-council backtest (spans stages 2–4) ──
│   ├── blinding.py            #   no-dates/no-tickers/no-levels period contexts + mirror copies
│   ├── specialists.py         #   domain analysts (rates/credit/vol/dollar/liquidity/trend)
│   ├── llm.py                 #   Claude-backed specialists (structured outputs + Batches API)
│   └── council.py             #   walk-forward engine + solo attribution + mirror leakage audit
└── mining/                    # ── automated alpha mining (spans stages 2–4) ──
    ├── signals.py             #   cross-sectional alpha families (incl. BOCPD/macro-gated)
    ├── evaluation.py          #   lagged, cost-aware dollar-neutral L/S construction
    ├── validation.py          #   randomized gauntlet: random purged windows,
    │                          #   block bootstrap, permutation test, Deflated SR
    ├── miner.py               #   evolutionary search loop + holdout leaderboard
    └── timing.py              #   timing-dial mining on one underlying (null = buy & hold)

experiments/                   # reproducible studies that wire the stages together
datasets/                      # bundled price data (250-ETF + 500-equity liquid universes + info, 5 ETFs, ~100 S&P 500)
docs/pead/                     # PEAD research write-ups & headline results
results/                       # generated plots & CSVs (gitignored)
```

Each stage depends only on the stages *above* it: `research` is pure and
time-step-agnostic; `backtest` engines walk research primitives through history;
`validation` turns the resulting return streams into evidence.

---

## Install

```bash
pip install -e .          # installs posterioralpha + dependencies
# or, without installing the package:
pip install -r requirements.txt

cp .env.example .env      # then fill in FRED_API_KEY (free) for keyed FRED access
```

Requires Python ≥ 3.10.

---

## Quickstart

Work on the **large universe** — 250 liquid US ETFs (2010–today) bundled in
`datasets/`, no network needed:

```bash
python experiments/run_large_universe.py   # universe summary + cross-asset allocation vs SPY
```

```python
from posterioralpha.data import load_etf_universe_returns, load_etf_universe_info
from posterioralpha.backtest import run_market_neutral, MNParams
from posterioralpha.validation import compute_metrics

returns = load_etf_universe_returns()                  # 250 ETFs  (stage 1)
info    = load_etf_universe_info()                     # financedatabase metadata + median ADV
res     = run_market_neutral(returns, MNParams(market="SPY"))   # stages 2+3
print(compute_metrics(res.returns))                    # stage 4
```

The universe is assembled by **financedatabase** (investable ETF universe +
instrument info) and **yfinance** (adjusted price history), screened for
coverage and dollar-volume liquidity. Rebuild / refresh it with:

```bash
python experiments/build_etf_universe.py --start 2010-01-01 --top-n 250   # needs network
```

Smaller bundled datasets remain available (`load_portfolio_returns` — 5 ETFs;
`load_sp500_prices` — ~100 S&P 500 names) for quick checks.

---

## Strategy catalogue

### Bayesian family — monthly rebalance (`backtest.bayesian`)
| key | description |
|-----|-------------|
| `bayesian` | Bayesian shrinkage: λ blends recent vs. prior moments via Mahalanobis divergence |
| `equal_weight` | 1/N baseline |
| `min_variance` | long-only global minimum variance |
| `hist_max_sharpe` | max-Sharpe on the full prior (no adaptation) |
| `bayesian_ewma` | EWMA "recent" estimate + transaction-cost penalty |
| `bayesian_hmm` | per-regime portfolios blended by HMM posterior + uncertainty penalty |
| `bayesian_full` | all extensions combined |

### AMR family — weekly rebalance (`backtest.amr`)
| key | description |
|-----|-------------|
| `amr` | Asymmetric Min-Risk (downside − λ·upside) + vol targeting |
| `inv_vol` | inverse-volatility baseline |
| `amr_no_vt` | AMR without the vol-targeting overlay (ablation) |
| `bocpd_amr` | BOCPD sets adaptive lookback + λ, then AMR |
| `bocpd_amr_v2` | multi-asset BOCPD + CVaR objective + dynamic low-vol tilt |
| `bocpd_amr_v3` | continuous λ via EWMA Omega ratio |
| `bocpd_amr_v4` | ERL-adaptive EWMA halflife for Omega |
| `hmm3_amr` | 3-state HMM blends regime-specific AMR portfolios |

### Market-neutral & specialist strategies
| key / module | rebalance | description |
|--------------|-----------|-------------|
| `run_market_neutral` (`backtest.market_neutral`) | weekly | Beta-hedged residual mean-reversion — rolling OLS β hedges out market exposure, contrarian on residual z-scores, BOCPD-gated, vol-targeted. A long/short alpha sleeve (β ≈ 0). |
| `run_hurst_multiasset` (experiment) | weekly | Cross-sectional Hurst (R/S) trend ranking: top-K persistent assets, inverse-vol sized, with an IVOL-5 fallback. Uses `research.hurst`. |
| `run_hurst_bull_adaptive` (experiment) | weekly | SPY trend-timing with adaptive rolling-quantile H/ERL thresholds + walk-forward re-optimisation. |
| `run_intramonth_momentum` (experiment) | monthly | Tests the Nathan-Suominen-Tasa (2026) T-9→T-4 intramonth momentum window. Uses `research.intramonth`. |
| `run_intramonth_bayesian` (experiment) | monthly | Learns the per-T-k allocation online via a conjugate Normal-Normal model — no hardcoded window. |
| `dfl_orchestrator` (experiment, `.[dfl]`) | — | Decision-focused multi-pod system: 6 asset-level pods + a differentiable mean-variance QP layer trained end-to-end on realised Sharpe (cvxpylayers). |

Reusable signal primitives live in `research/` (`rs_hurst`, `rolling_hurst`,
`intramonth_window_mask`, `wml_formation`); the studies above wire them through the
backtest + validation stages in `experiments/`.

**Reference configuration** (`experiments/run_champion_stack.py`): the frozen
5-layer-vote stack — flip-risk slow-down, debt-ceiling ×½, dollar (UUP) slack —
implemented as *w = exposure/2 in QLD, 1−w in UUP* (two tickers, real fund
prints).  Best validated scorecard in the project: Sharpe 0.96, max DD −28%,
Calmar 0.87, 2022 −24%, 100%/97% rolling-3y CAGR/Sharpe win vs SPY (2011→,
honest costs).  See `experiments/README.md` for the study trail behind every
component.

---

### PEAD — Post-Earnings-Announcement-Drift (`pead`)

A separate, **event-driven cross-sectional equity-anomaly** pipeline (distinct from
the time-series portfolio strategies above, so it lives in its own subpackage that
still follows the four-stage layering). It tests the analyst-surprise SUE signal
across the market-cap spectrum.

```bash
python experiments/download_finance_data.py   # one-time: fetch universe (needs `pip install -e .[pead]`)
python experiments/run_pead.py                # Fama-MacBeth IC test across cap buckets
python experiments/walk_forward_pead.py       # walk-forward long-short backtest
```

Findings (`docs/pead/`) are deliberately self-critical: the signal is statistically
significant (IC t-stats > 6, strongest in small/micro-cap) but a prior **+23.6%/yr**
result was traced to a **winners-only calculation bug** and corrected to ~+11–15%/yr,
and the live-names-only universe makes long-short returns **survivorship-biased upward**.
Treat it as a research artefact, not a deployable edge.

### Polymarket — cross-market momentum (`polymarket`)

A **prediction-market** module, deliberately a *different asset class* from every
time-series / cross-sectional **equity** strategy above. A Polymarket market
resolves Yes/No and its Yes-token price *is* the implied probability, so edge comes
from probability mispricing/drift rather than mean–variance, and the natural
coordinate is **log-odds**. Like `pead`/`council` it is a self-contained subpackage
spanning all four stages, on live data (Gamma metadata + CLOB price history):

```bash
python experiments/run_polymarket_momentum.py            # cached panel + 4 books
python experiments/run_polymarket_momentum.py --refresh  # re-pull live Polymarket data
python experiments/run_polymarket_vol_outcome.py --refresh   # sharp-move → outcome study
```

**Cross-market momentum** (`run_polymarket_momentum.py`) pits cross-sectional
momentum vs. its contrarian mirror (`xs_reversal`), per-market time-series
momentum, and a `long_all` baseline — all causal, Bayesian-shrunk by each
market's log-odds noise, cost-aware. First finding (`docs/polymarket/`,
**research artefact, not a deployable edge**): on the top resolved markets the
weekly cross-section **mean-reverts** rather than trends (favorite–longshot bias),
the signal is weak (|Sharpe| ≈ 0.18), and turnover cost is the binding constraint
— it does not survive 50 bps/turn.

**Sharp move → actual outcome** (`run_polymarket_vol_outcome.py`) is an event
study using the settled resolution as ground truth: it buckets `(market, day)`
episodes by recent upside log-odds volatility (and `sharp_up` / `vol_skew` /
`bocpd_cp`, reusing the repo's BOCPD detector) and reads off the actual Yes-rate,
the calibration residual vs. price, and the forward drift. Finding: a sharp
upside move is **mildly informative about the eventual outcome** (markets
underprice Yes by ~+3.6% in the top bucket) but the sharpest spikes **overreact
short-term** (forward 10-day log-odds drift flips negative) — the same mean-reversion
theme as the momentum study. Full order-book depth is reachable live
(`fetch.order_book_features`: spread / microprice / depth imbalance) for future
live-signal work, though Polymarket exposes no historical book.

**Harvesting the edge** (`run_polymarket_event_trades.py`) turns that split into
trades with realistic frictions (fill *after* the move, slippage per book crossing,
one position per market). The **exit policy is the whole game**: detecting a
top-bucket upside spike and **holding Yes to resolution** makes money while **trading
the drift** (take-profit / fixed horizon) breaks even or loses to the short-term
overreaction (`docs/polymarket/EVENT_TRADE_HARVEST.md`).

**Level control** (`run_polymarket_band_control.py`) then *corrects* that reading.
Comparing high- vs low-vol episodes **within fixed price bands** shows the
hold-to-resolution PnL is the favorite–longshot **calibration of the entry band**,
not the vol detection — a vol filter slightly *reduces* it (+9¢ vs +11¢ per $1
contract). The level-controlled vol signal is real but **sign-flips** (spiking
longshots revert; spiking favorites resolve Yes +29% more) and is **already priced**
(trading it loses net of slippage). So the genuine effect here is *structural*
favorite–longshot mispricing of mid-low-probability markets — not a volatility/timing
edge. A cautionary, self-correcting result in the spirit of the PEAD writeup
(`docs/polymarket/BAND_CONTROL.md`).

**Universe robustness** (`run_polymarket_universe_robustness.py`) asks whether the
[0.15,0.30] underpricing is structural or a 2024 artifact: it pulls a larger 514-market
universe, tags each by topic (`categorize.py`), and re-runs the calibration per category.
Verdict — **political, not structural**: the residual is +42% in politics and +22% in
geopolitics but **≈0 in sports** (the largest category) and crypto, and negative in
macro. The "favorite–longshot edge" was the 2024-election + war complex; it does not
generalise. No deployable, universe-robust edge survived the four-study arc — but a
clean, reusable prediction-market research stack did (`docs/polymarket/UNIVERSE_ROBUSTNESS.md`).

**Digging into politics** (`run_polymarket_politics_deep.py`) stress-tests the one
positive lead. Good news: the underpricing *is* diverse — 25 distinct events across
many countries, persisting 2024–2025, and politics is genuinely **mis-calibrated**
(underdogs underpriced, favorites at 0.7–0.9 resolving Yes only ~44%) while
non-politics hugs the calibration diagonal. But traded honestly — entered at realistic
prices, **PnL clustered per event**, bootstrap CI — every strategy's interval spans
zero, it's statistically **indistinguishable from the non-political baseline**, and it
decays in 2026 (+10%→+12%→−4% by year). A real behavioural mis-calibration, but one
correlated political era — a directional regime bet, not a confirmed edge. Upgrading it
needs genuine out-of-sample cycles (pre-2024 history) — `docs/polymarket/POLITICS_DEEP.md`.

**Information efficiency** (`run_polymarket_efficiency.py`) steps back from edge-hunting
to ask how informative prices are (Brier score) over a market's life. Prices converge
(Brier 0.081→0.028 into the last 3 days), and **efficiency varies hugely by domain**:
Brier-skill-vs-base-rate is +84% in macro (Fed telegraphs), +39% in sports, but only
**+4.5% in geopolitics** — far-from-resolution geopolitical prices are near coin-flips
the market can't forecast. The inefficient domains are exactly the mis-calibrated ones,
which reframes the missing edge: where the market looks least efficient there is mostly
**irreducible uncertainty, not exploitable mispricing** (`docs/polymarket/EFFICIENCY.md`).

**Strategy synthesis** (`run_polymarket_strategy_synthesis.py`) answers "can any of
this be traded?". Directional alpha: no. **One** structural trade survives — the classic
favorite–longshot bias in **macro** (scheduled Fed/rate markets): tail events priced
4–20% that resolved Yes 0%, so shorting them (buy No, hold to resolution) earns
+9¢/contract, CI [+0.073,+0.110], surviving 3% slippage. But it's a **short-volatility**
bet — ~100% hit only because no tail fired in the calm 2024–25 regime — with tiny
capacity and unobserved tail risk. The politics/sports "underpricing" that looked like
edge is shown to be **volume-selection bias** (high-volume longshots are far more
"underpriced" — exciting underdogs only draw volume once they're winning). A single
honestly-caveated candidate, not a free lunch (`docs/polymarket/STRATEGY_SYNTHESIS.md`).

**Cross-outcome structure** (`run_polymarket_cross_outcome.py`) maps the *joint*
behaviour of a multi-candidate event's outcomes (which the rest of the thread, treating
each market alone, never did). Fields are internally **arbitrage-free** (sum-to-one to
0.7%), substitution is **concentrated** — 82% of a shocked candidate's mass flows to one
beneficiary, and the dominant negatively-correlated pair recovers each real head-to-head
(Biden↔Harris −0.83, Republican↔Democrat −0.98, Fed cut↔hold), so any field reduces to a
2-horse race plus dead longshots. Redistribution slightly over-reacts (beneficiary
reverts, but not tradably), and within fields the favorite is under-bet / the challenger
over-bet (one-regime caveat). A rich structural primitive for hedging/modelling, even
without a new edge (`docs/polymarket/CROSS_OUTCOME.md`).

## Methodology notes

- **No lookahead bias.** Regime filters use forward-filtered posteriors only (no Viterbi
  backward pass); moments and signals at date *t* use data up to *t*.
- **Warm-up.** Engines reserve an initial history window before the first rebalance so
  priors are meaningful from day one.
- **Costs.** Transaction costs are charged on effective (post-leverage) turnover.

See `experiments/README.md` for the index of studies.
