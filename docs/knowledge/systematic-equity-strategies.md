# Systematic / Quant Equity Strategies — Knowledge Base

> Purpose: curated, research-ready reference for building and evaluating systematic equity strategies. Each entry includes core findings, key mechanics, and how to actually use it in research.

-----

## 1. Academic Papers & Research Corpora

### 1.1 AQR Research Library

**Link:** <https://www.aqr.com/Insights/Research>

The largest free practitioner-grade research corpus. AQR papers are unusual in that they’re written by people running real money, so they consistently address implementation costs, capacity, and turnover — not just gross alphas.

**Core papers to extract:**

**“Value and Momentum Everywhere” (Asness, Moskowitz, Pedersen, 2013, Journal of Finance)**

- Documents value and momentum premia across 8 markets/asset classes: US/UK/Europe/Japan equities, country indices, currencies, bonds, commodities.
- Key empirical fact: value and momentum are **negatively correlated** with each other (~ -0.5 to -0.6 within asset classes) but **positively correlated with themselves across asset classes**. This means a 50/50 value+momentum combo has a dramatically higher Sharpe than either alone.
- Research use: the negative correlation is the single most exploitable structural fact in factor investing. Any multi-factor equity strategy should test value/momentum blending before anything else. Also suggests common global factor structure → liquidity risk partially explains comovement.
- Implementation note: they use BE/ME for value and 12-1 month returns for momentum (skip the most recent month due to short-term reversal).

**“Quality Minus Junk” (Asness, Frazzini, Pedersen, 2019)**

- Defines quality as a composite: **profitability** (GPOA, ROE, ROA, CFOA, margins), **growth** (5y growth in profitability measures), **safety** (low beta, low leverage, low bankruptcy risk, low earnings volatility).
- QMJ factor earns significant alpha vs. FF4 in 24 countries. Quality stocks are persistently *underpriced* relative to what their fundamentals warrant.
- Key insight for research: quality works best as a **conditioning variable** — “value among quality” (avoiding value traps) and “quality at a reasonable price” outperform either factor alone.
- Construction detail: each metric is cross-sectionally ranked and z-scored, then averaged within sub-composite, then averaged again. Robust to metric substitutions — the signal is in the breadth, not any single ratio.

**“Betting Against Beta” (Frazzini & Pedersen, 2014)**

- Low-beta assets earn higher risk-adjusted returns than high-beta assets, in equities (US + 19 intl markets), bonds, credit, futures.
- Mechanism: leverage-constrained investors (mutual funds, retail) overpay for high-beta stocks to get implicit leverage → high beta is structurally overpriced.
- BAB factor construction: long low-beta levered to beta=1, short high-beta delevered to beta=1. Beta estimated with 1y daily correlations + 5y volatilities, shrunk toward 1 (shrinkage factor 0.6).
- Research use: explains why min-variance and low-vol strategies have historically beaten cap-weighted benchmarks. Caution: BAB performance is sensitive to funding conditions — it suffers when leverage constraints tighten (e.g., crisis deleveraging).

**“Buffett’s Alpha” (Frazzini, Kabiller, Pedersen, 2018)**

- Decomposes Berkshire’s returns: essentially long cheap, safe, quality stocks with ~1.6x leverage at low cost (insurance float). Once you control for BAB + QMJ factors, the alpha is statistically insignificant.
- Research use: a template for **strategy reverse-engineering** — regress any track record on factor returns before crediting skill. Methodologically useful for evaluating funds or your own backtests.

-----

### 1.2 Fama & French — Factor Model Papers

**Links:** SSRN / Journal of Financial Economics; data at the French Data Library (see §4.1)

**“Common Risk Factors in the Returns on Stocks and Bonds” (1993)**

- The 3-factor model: Market (Mkt-RF), Size (SMB), Value (HML). Built with the now-canonical 2x3 double sort: 2 size buckets (NYSE median) × 3 BE/ME buckets (30/40/30 percentiles), factors = average of corner portfolios.
- Why this matters for research: the **2x3 sort methodology is the industry-standard construction recipe**. When you build any factor, replicating this construction makes your results comparable to the literature.

**“A Five-Factor Asset Pricing Model” (2015)**

- Adds RMW (profitability: robust minus weak operating profitability) and CMA (investment: conservative minus aggressive asset growth).
- Critical finding: with RMW and CMA included, **HML becomes redundant** — its alpha vs. the other four factors is ~0. Value is largely subsumed by profitability + investment.
- Known weakness (admitted in the paper): the model fails to price small stocks with low profitability that invest aggressively (small-growth lottery stocks). This residual is a persistent anomaly worth knowing about.
- Research use: this is the **default benchmark model** for testing whether a new signal adds alpha. Always report alphas vs. FF5 + momentum (6-factor), not just CAPM, otherwise the result is uninformative.

-----

### 1.3 “…and the Cross-Section of Expected Returns” (Harvey, Liu, Zhu, 2016)

**Link:** SSRN 2249314

- Catalogues 316 published factors and applies multiple-testing corrections (Bonferroni, Holm, BHY false discovery rate).
- Headline result: given the volume of testing in the field, a newly proposed factor should clear a **t-statistic of ~3.0+, not the classical 2.0**. Most published factors fail this bar.
- Estimates that a large fraction of published factors are false positives from data mining.
- Research use: this paper sets your **statistical hygiene standards**. Practical rules to adopt:
  - Require t > 3 for in-sample discovery.
  - Track the number of strategies/variants you’ve tested and adjust significance accordingly (each backtest variation is a trial).
  - Prefer signals with economic rationale stated *before* testing.
- Companion: Harvey & Liu, “Backtesting” (2015) — gives a haircutting formula for Sharpe ratios based on number of trials.

-----

### 1.4 “Replicating Anomalies” (Hou, Xue, Zhang, 2020, RFS)

**Link:** SSRN 2961979

- Replicates **452 published anomalies** with consistent methodology: NYSE breakpoints, value-weighted returns (vs. the equal-weighted, all-breakpoint constructions common in original papers).
- Result: **65% of anomalies fail replication** (|t| < 1.96). With t > 2.78 (multiple-testing adjusted), ~82% fail.
- The failures concentrate in: trading-frictions category (almost entirely fails), and anomalies driven by microcaps (60% of stocks but ~3% of market cap).
- What survives robustly: **momentum variants, investment/asset growth, profitability, accruals, net stock issuance**.
- Research use: this is the **filter list**. Before researching any anomaly from the literature, check whether it survives HXZ replication. Also adopt their hygiene: value-weight your backtests and use NYSE breakpoints (or large/mid-cap-only universes) to avoid microcap mirages — especially relevant in Europe where microcap liquidity is worse.

-----

### 1.5 Open Source Asset Pricing (Chen & Zimmermann)

**Link:** <https://www.openassetpricing.com> / SSRN 3604626

- Open-source replication of **200+ cross-sectional predictors** with full code (Stata/R) and downloadable monthly returns for each anomaly portfolio.
- Contrary to HXZ’s pessimism, they find most predictors *do* replicate when constructed as in original papers — the disagreement is about construction choices, which is itself instructive.
- Research use:
  - Free dataset of anomaly long-short returns → use directly as inputs to factor-timing, factor-momentum, or ML meta-strategy research without rebuilding 200 signals.
  - Their documentation spreadsheet (signal definitions, sample periods, original t-stats vs. replicated) is the best single reference table of the factor zoo.
  - Post-publication decay: combine with McLean & Pontiff (2016) finding that anomaly returns decay ~58% post-publication → discount any literature Sharpe by roughly half.

-----

## 2. Books

### 2.1 *Quantitative Equity Portfolio Management* — Chincarini & Kim

**The end-to-end textbook for building an equity quant process.** Covers the full pipeline:

- **Factor model taxonomy:** fundamental factor models (BARRA-style: factor exposures known, returns estimated) vs. economic factor models (factor returns known, betas estimated) vs. statistical (PCA). Knowing which type you’re building determines your estimation procedure and risk decomposition.
- **Alpha modeling chapters:** Z-score aggregation across signals, sector/industry neutralization, outlier treatment (winsorization at ±3σ), turnover-aware signal blending.
- **The “fundamental law” applied:** stock screening vs. weighting; aggregate Z-score model construction step by step.
- **Portfolio construction:** mean-variance with constraints, tracking-error budgeting, transaction-cost-aware rebalancing, tax considerations.
- **Backtesting chapter:** explicit treatment of survivorship bias, look-ahead bias (use point-in-time fundamentals — report dates, not fiscal dates), and data snooping.
- Research use: when building an internal tool or backtest framework, this book is effectively the **spec document**. The Z-score aggregation + neutralization recipe is the default architecture for multi-signal equity models.

### 2.2 *Expected Returns* — Antti Ilmanen

- Organizes everything around three lenses: **asset class premia, factor/style premia, and strategy premia**, with historical estimates and forward-looking frameworks for each.
- Key conceptual contributions:
  - **Required vs. expected returns:** premia exist as compensation for risk (bad-times covariance), or from structural/behavioral frictions. Classify every signal you research into one of these buckets — it predicts persistence.
  - Survey of the major style premia (value, momentum, carry, defensive/low-risk) **across asset classes**, with consistent evidence tables.
  - Honest treatment of estimation uncertainty: historical average returns have huge standard errors (σ/√T) — 50 years of data gives ±2-3% confidence bands on equity premium estimates.
- Research use: the carry/value/momentum/defensive 4-style framework is a useful **completeness check** — for any universe you trade, ask which of the four styles you’ve implemented and why the missing ones are excluded.
- 2022 follow-up: *Investing Amid Low Expected Returns* — updates for the post-QE regime.

### 2.3 *Advances in Financial Machine Learning* — Marcos López de Prado

The reference for ML applied to investing without fooling yourself. Most valuable chapters for equity research:

- **Ch. 3 — Labeling:** the triple-barrier method (label by which barrier is hit first: profit-take, stop-loss, or time expiry) and meta-labeling (a second model that predicts whether the primary model’s signal should be acted on, sizing bets by confidence). Meta-labeling is directly applicable to filtering any rule-based signal.
- **Ch. 4 — Sample weights:** financial labels overlap in time → observations aren’t IID → standard cross-validation is broken. Weight samples by uniqueness.
- **Ch. 7 — Purged K-fold CV with embargo:** removes training samples whose labels overlap the test set, plus an embargo period after the test window. **This is the correct CV scheme for any financial ML, full stop.** Standard sklearn KFold on financial data leaks and inflates performance.
- **Ch. 8 — Feature importance:** MDI (in-sample, biased toward high-cardinality features), MDA (out-of-sample permutation importance), Single Feature Importance. Use MDA on purged CV.
- **Ch. 10 — Bet sizing** from predicted probabilities (sigmoid sizing of signal confidence).
- **Ch. 14 — Backtest statistics** and the **Deflated Sharpe Ratio**: corrects observed Sharpe for the number of trials, skewness, and kurtosis. Formula inputs: observed SR, variance of SRs across trials, number of trials, sample length, skew, kurtosis. Adopt as standard output of any backtest report.
- Warning the book itself makes: most ML fund failures come from backtest overfitting, not bad models.

### 2.4 *Active Portfolio Management* — Grinold & Kahn

The theoretical backbone of active quant investing. Core machinery:

- **The Fundamental Law of Active Management:**
  `IR ≈ IC × √BR × TC`
  where IC = information coefficient (correlation between forecasts and realized returns, typically 0.02–0.10 for good signals), BR = breadth (number of *independent* bets per year), TC = transfer coefficient (correlation between ideal and implemented portfolio, degraded by constraints/costs).
  - Implications you’ll use constantly: a weak signal (IC=0.03) applied across 1,000 stocks monthly beats a strong signal (IC=0.15) applied to 10 stocks annually. Breadth is the quant’s structural edge. Long-only constraints typically cut TC to ~0.3–0.6 → roughly half the potential IR is lost to the no-shorting constraint.
- **Forecasting rule of thumb:** `Alpha = IC × volatility × score` (score = cross-sectional z-score of the signal). This converts any raw signal into return forecasts with sane magnitudes for an optimizer.
- **Information ratio as the objective:** value-added = α − λ·ω² (alpha minus risk-aversion-penalized active variance); optimal active risk = IR/(2λ).
- Research use: the IC framework gives you a **signal evaluation pipeline before any backtest**: compute monthly cross-sectional IC of signal vs. forward returns, its mean, std, and IC t-stat (mean IC / std IC × √N). An IC consistently > 0.02 with t > 2 is a real signal; backtest after, not before.

-----

## 3. Practitioner Blogs & Research Sites

### 3.1 Alpha Architect — <https://alphaarchitect.com/blog>

- Publishes accessible summaries of academic papers (several per week), each with construction details and “why it matters” sections. Run by Wes Gray (ex-academic, runs factor ETFs).
- Strongest content areas: value investing research, momentum, trend following, and **factor crash risk** (e.g., momentum crashes of 2009 — momentum loses ~70%+ in sharp reversals after market crashes due to its implicit short-beta position post-drawdown).
- Research use: efficient paper-screening layer — read their summary, then pull the SSRN original only if relevant. Their archive is searchable by factor.

### 3.2 Quantpedia — <https://quantpedia.com>

- An encyclopedia of **700+ trading strategies** extracted from academic papers, each entry: markets, instruments, backtest period, reported Sharpe, turnover, complexity rating, and source paper.
- Free tier includes ~70 strategy descriptions and the excellent **Screener taxonomy** (by asset class, style, rebalancing frequency).
- Research use: idea generation and prior-art search. Before researching a strategy, check Quantpedia to see if it’s documented, what the literature Sharpe is, and what the known caveats are. Their blog also runs original replications with out-of-sample updates — useful decay evidence.

### 3.3 Robeco Quant Research — <https://www.robeco.com/en-int/insights>

- The most academically rigorous European quant shop publishing publicly. Key recurring contributions:
  - **“Volatility effect” papers (Blitz & van Vliet):** the European/global evidence base for low-volatility investing, predating and complementing BAB.
  - **Factor investing in emerging markets and credits** — evidence that the same styles work outside US large-cap.
  - **“Resurrecting the Value Premium”** and machine-learning-for-factors papers — pragmatic ML applications with economic priors.
  - **Conservative formula** (van Vliet): low vol + momentum + net payout yield, a simple implementable 3-signal model with a long backtest (1929+) — a good baseline to benchmark more complex models against.
- Research use: when researching anything in **European equities**, Robeco is the first stop — their samples and conclusions account for European market structure (different sector mix, lower liquidity tail).

### 3.4 Man Institute / Man AHL — <https://www.man.com/insights>

- Practitioner papers on trend following, execution, and portfolio construction with unusual transparency on implementation.
- Notable recurring themes: **trend-following as crisis alpha / portfolio insurance** (performance conditional on equity drawdowns), **volatility targeting** (vol-scaled positions improve Sharpe and cut tail risk for risk assets because volatility is persistent while returns aren’t), and machine learning in execution.
- Research use: the volatility-targeting literature here is directly portable: scaling any equity strategy’s exposure by 1/σ̂ (recent realized vol) typically improves its Sharpe and materially cuts left tail. Cheap, robust overlay worth testing on every strategy.

### 3.5 Flirting with Models (Newfound Research) — <https://blog.thinknewfound.com>

- Deep, original work on **the mechanics of strategy implementation**, especially:
  - **Rebalance timing luck:** two identical strategies rebalanced on different days of the month can diverge by several % per year. The fix: tranche the portfolio across multiple rebalance dates (overlapping portfolios). Quantifies this for value, momentum, trend.
  - **Trend-following speed diversification:** blending lookback windows beats picking one.
  - “Liquidity cascades” and payoff-replication views of strategies (momentum ≈ long straddle-ish convexity; value ≈ short-vol-ish).
- Research use: the rebalance-timing-luck result is a mandatory robustness check — run every backtest at multiple rebalance offsets and report the dispersion. If strategy ranking changes with offset, the edge is noise.

-----

## 4. Data & Replication Resources

### 4.1 Kenneth French Data Library

**Link:** <https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html>

- Free, monthly-updated returns since 1926 (US): Mkt-RF, SMB, HML, RMW, CMA, momentum (UMD), short/long-term reversal, plus **industry portfolios** (5/10/12/17/30/38/48/49 industries) and bivariate sorted portfolios (size×value, size×momentum, etc.).
- Also: **developed international and European factor returns** (relevant for European universes).
- Research uses:
  - Benchmark regressions: regress your strategy returns on FF5+UMD to compute alpha and factor loadings — the standard attribution.
  - The 25 size×value portfolios are the canonical **test assets** for evaluating any pricing model.
  - Industry portfolios: quick datasets for testing cross-sectional ideas without stock-level data.
- Caveats: factors are academic constructions — no transaction costs, monthly rebalanced, includes hard-to-short microcaps in some legs. Treat as upper bounds.

### 4.2 Open Source Asset Pricing data (see §1.5)

- Download: monthly long-short returns + underlying portfolio returns for 200+ predictors, plus the signal documentation spreadsheet. Updated periodically.
- Practical use: ready-made input matrix for **factor momentum** research (Ehsani & Linnainmaa: factors themselves exhibit momentum — last year’s winning factors outperform), factor crowding studies, and PCA on the anomaly space (most anomalies load on a few principal components).

### 4.3 JKP Global Factor Data — <https://jkpfactors.com>

- From Jensen, Kelly & Pedersen, **“Is There a Replication Crisis in Finance?” (2023, JF)** — the optimistic rejoinder to HXZ: using a Bayesian hierarchical model and consistent construction, **most factors replicate**, and evidence is strengthened by internal validity across 93 countries.
- The data: **153 factor returns × 93 countries**, monthly, free, with code (GitHub: bkelly-lab/replication-crisis). Factors grouped into 13 themes (value, momentum, quality, low risk, etc.).
- Research uses:
  - Out-of-sample validation: test whether your signal’s cousin-factors work in other countries — cross-country consistency is strong evidence against data mining.
  - European factor returns at the theme level → direct relevance for a European equity universe.
  - The 13-theme clustering is a good dimensionality structure for organizing your own signal library.

-----

## 5. Cross-Cutting Research Hygiene (synthesis)

Rules distilled from the corpus above — apply to every project:

1. **Signal evaluation before backtesting:** rank IC analysis (Grinold & Kahn). Mean IC, IC t-stat, IC decay profile across horizons (1m/3m/6m/12m forward).
1. **Statistical bar:** t > 3 in-sample (Harvey et al.); deflate Sharpe for number of trials (López de Prado DSR); expect ~50% post-publication decay (McLean & Pontiff).
1. **Construction hygiene:** value-weight or restrict to investable universe, NYSE-style breakpoints, point-in-time fundamentals, sector-neutralize unless the sector bet is the thesis (HXZ; Chincarini & Kim).
1. **Robustness battery:** multiple rebalance offsets (Newfound), sub-period stability, parameter neighborhoods (perturb lookbacks ±25%), cross-country check (JKP data).
1. **Cost realism:** turnover × spread + impact estimate; long-only TC penalty (~halves IR vs. long-short).
1. **Overlay candidates that usually help:** volatility targeting (Man AHL), value+momentum blending (AQR), quality conditioning (QMJ), tranched rebalancing.
1. **CV for ML:** purged K-fold with embargo only; never plain KFold (López de Prado).

-----

## 6. Suggested Reading Order

1. Grinold & Kahn ch. 1–6 (the IR/IC framework) → 2. Value & Momentum Everywhere → 3. Hou-Xue-Zhang (what survives) → 4. Harvey et al. (statistical bar) → 5. Chincarini & Kim (build pipeline) → 6. Ilmanen (style completeness) → 7. López de Prado (ML layer, if/when needed). Blogs in parallel as a screening layer.
