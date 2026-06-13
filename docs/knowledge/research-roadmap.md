# Research Roadmap — Applying the Knowledge Base to PosteriorAlpha

> Companion to `systematic-equity-strategies.md` (external corpus) and
> `alpha-mining-commit-log.md` (the program's arcs). Maps the knowledge
> onto the framework and ranks next experiments. Written 2026-06-12 against
> the `claude/automated-alpha-mining` branch.

## 1. Where the knowledge base meets the framework

| KB source | Status in this repo |
|---|---|
| López de Prado — DSR, purged windows, bootstrap | Already first-class: `mining/validation.py` runs the full gauntlet on every mined candidate. |
| Harvey et al. — trials accounting | Enforced inside the miner; NOT yet applied retroactively to hand-built families (BOCPD-AMR v1→v4, vote-layer variants). `validation/robustness.py::dsr_report` now exists for that. |
| Grinold & Kahn — IC before backtest | Was PEAD-only (`pead/fama_macbeth.py`); now generic via `mining/ic.py`. Use it as the gate BEFORE a family enters the miner: a family with no rank IC shouldn't consume gauntlet trials. |
| Newfound — rebalance timing luck | Was absent everywhere. `validation/robustness.py::rebalance_offset_dispersion` + `rebal_anchor` in `backtest/amr.py`. First result: §6 below. |
| Man AHL — vol targeting | Implemented (plain + regime-adaptive cap in `backtest/amr.py`); the governor studies independently confirmed "vol management on levered barbells is real". |
| AQR — value/momentum blending; QMJ conditioning | Momentum families exist; value/quality need fundamentals data — open gap (fundamentals are also what PEAD's SUE pipeline already fetches; reuse it). |
| HXZ — survivorship hygiene | Known and flagged (equity universes are current-membership). A point-in-time universe remains the single biggest data upgrade available. |
| Ilmanen — style completeness | Defensive ✓ (low-vol, vol targeting), momentum ✓ (families), carry ✗, value ✗. |

## 2. Novel ideas (ranked)

### Idea 1 — Per-stock regime-gated momentum — STARTED
`stock_regime_momentum` family: each stock's momentum shrunk by its OWN
BOCPD regime age (`ctx.erl_panel`), vs. the existing market-level gate.
Daniel–Moskowitz crash logic applied cross-sectionally. First IC results: §5.
Next: IC with gate applied only at extreme freshness (ERL < 21); test the
gate on the short leg; if IC survives, let the miner search the family and
see what the gauntlet says.

### Idea 2 — `regime_age` as a standalone defensive factor
Price-only quality proxy. First IC pass inconclusive (§5) — parked unless a
different universe (ETFs, where BOCPD regimes are cleaner) revives it.

### Idea 3 — Factor momentum across mined finalists (Ehsani–Linnainmaa)
The miner's leaderboard is a panel of factor returns; time them by their own
trailing 12-month performance. Cheap meta-strategy on existing output.

**OpenAP variant — TESTED, NULL vs benchmark.** On the 212-predictor panel
(live 1997→2024) the Ehsani-Linnainmaa fact replicates — XS winners Sharpe
0.83 (t 4.4), TS winners 1.01 (t 5.4), 12m spread +9.5%/yr (t 2.6), all
sub-period stable — but the spread is ~1.9× beta to the EW-ALL benchmark
with alpha +0.9%/yr (t 0.25), and the pre-declared 50/50 vol-matched combo
lowers the Sharpe (1.09 vs 1.29). Factor momentum on this panel is
time-varying zoo exposure, not a new return stream; hold the zoo, don't
time it. Still open: the original Idea-3 target (the miner's own
leaderboard, where the EW benchmark is much weaker).
`run_factor_momentum.py`.

### Idea 4 — Retroactive DSR on the hand-built families — DONE, split verdict
Ran `dsr_report` (excess returns, rf 4%) over both flagship families, with a
DSR-vs-assumed-trials sensitivity curve since nominal counts undercount the
true search. **5-ETF v1→v4 arc (9 variants, 2016-2024): FAILS** — best is
`bocpd_amr_v4` at DSR 0.91 (bar: 0.95), 0.88 at n=80; v3 ≡ v4 and v2 < v1,
so the version ladder was noise. **Champion-stack ladder on QQQ (7 variants,
2011→): PASSES** — DSR ≈ 1.00 across the board, robust to n=80 and to a
14× wider trial dispersion. Consequence: retire the 5-ETF allocation arc
from promotion language; the QQQ vote stack carries the program.
`run_retroactive_dsr.py`.

### Idea 5 — IC gate inside the miner loop
Wire `mining/ic.py` in as a cheap pre-filter: candidates whose 21d rank IC
t-stat < 1 on the training window are rejected before portfolio construction
— saves gauntlet budget and reduces effective trials.

### Idea 6 — Value/quality via the PEAD fundamentals pipe
PEAD already fetches earnings data; SUE's cousins (gross profitability,
accruals, net issuance — the HXZ survivors) are one fetch away. Closes the
Ilmanen style gap with infrastructure that already exists.

### Idea 7 — Alpha half-life (zoo age structure) — TESTED, NULL
McLean-Pontiff decay as an allocation rule: long YOUNG factors (1-5y since
publication), short OLD (>15y), on the OpenAP panel. The decay fact
replicates (50% post-pub haircut, M-P report ~58%) but the event-time
profile is a STEP at publication followed by a 25-year plateau at ~4%/yr —
age beyond publication carries no information, YOUNG−OLD is dead
(−0.7%/yr, t −0.3). Keep the byproduct: EW of all published factors runs
Sharpe 1.21 gross (t 5.9, maxDD −6%) and is sub-period stable — the zoo's
value is diversification, not its age structure. `run_alpha_halflife.py`.

### Idea 8 — Champion attribution — DONE, verdict in
FF5+UMD Newey-West regression (2011-10 → 2026-04): full stack
α = +6.1%/yr (t 2.34), retail QLD+UUP wrapper +6.9% (t 2.56), vs the QQQ
B&H control's +2.6% (t 2.07). Average market beta ≈ 1.0 (the 2× cap is
rarely fully spent), anti-value tilt (HML −0.36), R² 0.81. Read: the
timing machinery adds ~+3.5-4.3pp/yr beyond static growth beta —
classically significant, below Harvey's t > 3 bar. The thread deserves
continued investment but not promotion language stronger than "fragile
alpha". `run_factor_attribution.py`.

### Idea 9 — Factor crowding gauges on the zoo — TESTED, NULL
Three causal gauges (36m avg pairwise corr, PC1 share, 12m dispersion) on
the OpenAP panel as warning dials for the EW-zoo holding. The zoo
*de*-correlated as it grew (PC1 share 0.44→0.28 since the 90s); the
correlation gauges disagree on sign within eras (neither t>2); the only
borderline signal is dispersion→forward-EW (IC +0.14, t 2.2 post-2004) —
an opportunity-set effect, not a crowding alarm. No dial to build. Gross
panel can't see cost/capacity crowding, so this null does not clear live
risk. `run_factor_crowding.py`.

### Idea 10 — Meta-factors / cluster structure of the zoo — TESTED, ONE SURVIVOR
The factor-of-factors sweep beyond momentum (user prompt 2026-06-13).
Clusters are real and nowcastable (K=6, 36m corr, median ARI 0.75) but
cluster-balanced weighting decays (α t 4.9 → 2.0 across 2004) and cluster
rotation is dead. LT reversal and low-vol at the factor level: dead
(low-vol is backwards). **Factor-level seasonality survives everything**:
Keloharju same-calendar-month signal gives α vs EW-ALL of +9.5%/yr
(t 3.6, β −0.12), is not momentum (corr −0.20), holds in both eras, and
the 50/50 vol-matched combo lifts the EW holder's Sharpe 1.29 → 1.42 with
a smaller maxDD in both sub-periods. First clear t>3 positive on the
OpenAP panel, with KLN's published priors behind it.
`run_zoo_metafactors.py`.

**OOS update (`run_seasonality_oos.py`):** the JKP cross-region test
confirms direction, not magnitude. True-OOS regions all positive and
half-stable (Sharpe 0.37-0.42, t 2.1-2.5) but ~1/6 the OpenAP size; the
US theme-level panel is flat (t 0.76 / 69y). Seasonality is a breadth
effect — 4-per-leg theme baskets can't carry what 60-per-leg predictor
baskets can. Still open if pursued: cost model for the 2× monthly-turnover
sleeve; within-cluster seasonality (cluster bet or not); an OOS test at
matched breadth (e.g. the 153-factor JKP set, not its 13 themes).

### Idea 11 — Factor-level BOCPD: regime age in the zoo's cross-section —
### TESTED, MECHANISM YES / EDGE NO
The genuinely novel one (no published precedent in the KB): per-factor
online Bayesian changepoint posteriors, regime age (ERL) as a meta-signal.
The Daniel-Moskowitz logic transfers — stale-regime factors beat fresh
(α t 2.2 vs −0.4), regime-conditioned momentum orders exactly as predicted
in both eras, and momentum return is near-monotone in ERL quintile
(Sharpe 0.66 freshest → 1.03 stalest). But no construction clears the EW
benchmark or t>2 alpha (spread t 1.5, conditioned mom t 1.25), and the
cross-sectional changepoint dial is dead. Mechanism worth keeping in the
KB; edge not present at monthly/gross resolution. Possible revival: daily
factor returns (JKP daily?) where changepoints are sharper, or ERL as a
*risk* model (position sizing) rather than a return signal.
`run_factor_regimes.py`.

## 3. Standing hygiene rules (KB §5, adopted)

1. IC analysis before any backtest (`mining/ic.py` is the gate).
2. Every variant is a trial → family-level `dsr_report` before claiming
   version-over-version progress.
3. Weekly strategies report `rebalance_offset_dispersion`.
4. Survivorship caveat on all current-membership universes.
5. Costs ≥ 5 bps × turnover everywhere; the miner's evaluator is already
   cost-aware and lagged.

## 4. Architecture note

The cross-sectional additions live in `mining/` (ic lab, per-stock families)
and `validation/` (robustness) — stages 2 and 4 of the pipeline. The
per-stock ERL panel is cached on `SignalContext` (first call ~1s/name).

## 4b. Factor data enrichment (2026-06-13)

The KB's free data sources are now bundled (built by
`experiments/build_factor_data.py`, loaders in `posterioralpha.data`):

| dataset | contents | unblocks |
|---|---|---|
| `ff_factors_monthly.csv` / `ff_factors_daily.csv.gz` | US Mkt-RF, SMB, HML, RMW, CMA, RF, Mom (1963→, daily too) | Idea: FF5+UMD alpha attribution of the champion stack / barbell (the "Buffett's Alpha" regression) on daily returns |
| `ff_europe_monthly.csv` | Europe 5F + WML (1990→) | European-universe attribution |
| `ff_industry12_monthly.csv` | 12 VW industry portfolios (1926→) | cheap cross-sectional test assets |
| `jkp_factors.csv.gz` | 13 JKP themes × usa/developed/emerging/world_ex_us, monthly vw_cap (1926→) | cross-region OOS validation of any signal family |
| `openap_ls_returns.csv.gz` + `openap_signal_doc.csv` | 212 published predictors' monthly L/S returns + doc table (165 'clear' predictors) | factor momentum (Idea 3), factor crowding/PCA, decay studies — without rebuilding 200 signals |

Sanity-checked on load: FF equity premium 0.6%/mo at 16.2% ann vol; all
series stored as decimals.

## 5. First IC results (2026-06-12, 94 names, 2016-04 → 2026-04)

Monthly Spearman rank IC vs. forward returns, overlap-corrected t-stats;
current-membership S&P-100 → survivorship-biased. Regenerate via
`python experiments/run_signal_lab.py`.

| signal | IC 21d (t) | IC 63d (t) | IC 126d (t) | hit 21d | Q5−Q1 ann. |
|---|---|---|---|---|---|
| momentum_12_1 | 0.024 (1.21) | 0.031 (0.95) | 0.037 (0.78) | 0.56 | +3.8% |
| reversal_1m | 0.004 (0.20) | 0.003 (0.10) | −0.005 (−0.12) | 0.52 | +2.3% |
| low_vol_1y | −0.032 (−1.20) | −0.050 (−1.09) | −0.055 (−0.84) | 0.42 | −14.0% |
| regime_age | −0.013 (−1.05) | −0.002 (−0.11) | 0.016 (0.53) | 0.45 | −2.5% |
| stock_regime_momentum | 0.021 (1.10) | 0.026 (0.87) | 0.027 (0.60) | 0.58 | +2.8% |

Honest read: **nothing passes t > 2 on this sample** — the lab doing its
job. ~107 monthly observations with IC vol ≈ 0.2 cannot resolve a true IC
of 0.03 (needs ~15 years); mega-caps are the lowest-breadth, most efficient
corner. Momentum has the right shape (IC grows with horizon). The per-stock
gate trades IC magnitude for consistency (higher hit rate, lower IC vol than
raw momentum). Low-vol's strongly negative IC is the survivorship bias made
visible (the index filter kept the high-vol winners). Trials ledger: 5
signals × 3 horizons on one sample.

## 6. First timing-luck check (core AMR, 5 ETFs, 2005 → 2026)

`rebalance_offset_dispersion(run_amr_backtest, strategy="amr")`:

| anchor | CAGR | Sharpe | MaxDD |
|---|---|---|---|
| W-MON | 0.096 | 0.545 | −0.199 |
| W-TUE | 0.098 | 0.565 | −0.234 |
| W-WED | 0.096 | 0.551 | −0.224 |
| W-THU | 0.098 | 0.563 | −0.226 |
| W-FRI | 0.095 | 0.543 | −0.234 |
| **range** | **0.003** | **0.022** | **0.035** |

Verdict: the core AMR engine is robust to the anchor (0.3 pp CAGR range is
noise-level), and the historically-used W-FRI was the *worst* anchor, so
published results were never flattered by it. Open: same check on
BOCPD-AMR v2–v4, whose regime signals could interact with the anchor more.
