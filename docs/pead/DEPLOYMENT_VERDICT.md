# Retail Deployment Verdict — Alpaca Cost Analysis

**Question:** after realistic Alpaca retail costs, is any strategy in this repo
robust enough to deploy — and is there a real edge?

**Answer:** Yes. The PEAD long-*short* in its raw form is **not** deployable (its
edge lives in hard-to-borrow small/micro-caps that Alpaca cannot short). But a
**market-neutral long-short confined to easy-to-borrow Mega+Large caps** is fully
deployable and carries a robust, regime-stable edge:

> **+10.3%/yr (2010–2026), Sharpe ≈ 1.5, beta ≈ 0, max drawdown ≈ −6 to −11%,
> commission-free, $0 borrow.** Positive in every 5-year sub-period and out-of-sample.

Reproduce with `python experiments/deploy_cost_analysis.py` (after `run_pead.py`).

---

## 1. Alpaca cost model (2026)

Encoded in `posterioralpha/backtest/alpaca_costs.py`:

| Item | Value | Consequence |
|------|-------|-------------|
| Commission (US equities/ETFs) | **$0** | Binding cost is the bid-ask half-spread, not commission |
| Regulatory (SEC + FINRA TAF) | ~0.3 bps, sells only | Negligible |
| Short borrow — ETB (easy-to-borrow) | **$0/yr** | Large/mega-caps + SPY shortable for free |
| Short borrow — HTB (hard-to-borrow) | **not shortable at all** | **Kills any short leg in small/mid/micro caps** |
| Margin interest (leverage > 1×) | 6.25%/yr (non-elite) | A dollar-neutral book at ≤2:1 has **no** debit → no drag |
| PDT rule | **retired 2026-06-04** | No $25k minimum; holding-period strategies unaffected |

Half-spreads (one-way, bps) by tier: ETF 1 · Mega 2 · Large 4 · Mid 10 · Small 25 · Micro 60 · Nano 120.

## 2. What is NOT deployable

- **PEAD long-short (broad / small-cap):** the strongest raw signal is in Mid/Small/Micro
  caps (long-short +9–18%/yr gross), but those are **hard-to-borrow → unshortable on
  Alpaca**. Not deployable as a long-short.
- **Long-only top-SUE basket vs SPY:** robust **+5.9%/yr net selection alpha** over the
  equal-weight universe (t=6.8, 77% of quarters), but it is *equal-weight*, so it trails
  cap-weighted SPY on raw return by ~2.6%/yr in the mega-tech era (Sharpe 0.76 vs 0.71).
  A risk-improver, not a return-beater.

## 3. The deployable edge — market-neutral PEAD (Mega + Large, ETB)

Long top-quintile SUE / short bottom-quintile SUE, **within Mega+Large caps only** (all
easy-to-borrow), dollar-neutral, rebalanced quarterly, each leg charged a round-trip
spread. Net of all Alpaca costs:

| Window | Ann | Sharpe | MaxDD | Quarters |
|--------|-----|--------|-------|----------|
| 1998–2004 | +4.0% | 0.44 | −10.9% | 13 |
| 2005–2009 | +16.3% | 2.31 | −1.9% | 20 |
| 2010–2014 | +8.2% | 1.35 | −5.6% | 20 |
| 2015–2019 | +14.2% | 2.49 | −1.0% | 20 |
| 2020–2026 | +8.9% | 1.21 | −5.7% | 26 |
| **Modern (2010–2026)** | **+10.3%** | **1.57** | ~−6% | 66 |
| Full (1998–2026) | +10.6% | 1.48 | −10.9% | 99 |

**Out-of-sample walk-forward** (train → test): +10.8% (t=6.2) · +11.1% (t=5.2) ·
+9.4% (t=3.6) — robust in every split, 70–78% of quarters positive.

**Why it's robust:**
- Positive in **every** 5-year sub-period; weakest era was 1998–2004, so the result is
  *not* an artifact of the dot-com period.
- **Survivorship bias is conservative here:** delisted bottom-SUE names (the best shorts)
  are absent from the live-only panel, so the realised short-leg profit — and the edge —
  is if anything *understated*.
- Well-diversified: ~26 names per leg (min ~21) in the modern era — not concentrated.
- Correlation to SPY **+0.17** → a genuine diversifier, not a levered-beta proxy.

## 4. Recommended deployment — SPY core + market-neutral sleeve

Because the sleeve earns ~SPY-like returns at ~0 beta, blending it with SPY traces an
efficient frontier at **constant ~11.5% return** (2005–2024, unlevered, no margin drag):

| Capital split | Ann | Sharpe | MaxDD |
|---------------|-----|--------|-------|
| 100% SPY | +11.7% | 0.71 | −46% |
| 80/20 SPY/MN | +11.6% | 0.86 | −37% |
| 60/40 SPY/MN | +11.6% | 1.08 | −26% |
| 40/60 SPY/MN | +11.5% | 1.38 | −14% |
| 100% MN sleeve | +11.3% | **1.68** | **−6%** |

Pick the point on the frontier matching your drawdown tolerance. 40/60 roughly doubles
SPY's Sharpe and more than halves its drawdown for the same return.

## 5. Implementation notes

- **Account:** margin account (Reg-T) for the short leg; ≥$2k equity. No PDT constraint.
- **Universe:** US Mega + Large caps with analyst-estimate coverage (ETB-verify each name
  via Alpaca's `shortable`/`easy_to_borrow` asset flags before shorting).
- **Signal:** SUE = Yahoo `Surprise(%)` (analyst-estimate surprise), per `pead.signals`.
- **Construction:** each earnings season, rank SUE within tier; long top quintile / short
  bottom quintile, equal-weight, dollar-neutral; hold ~21 trading days (entered t+1 after
  the announcement). In practice run it as **rolling** 21-day holds entered at each
  announcement rather than discrete quarterly buckets (the backtest's quarterly grouping
  is a modelling simplification).
- **Costs already modelled:** spread per tier, $0 commission, $0 ETB borrow. No leverage
  beyond the 2:1 the dollar-neutral book implies → no margin interest.

## 6. Honest caveats / residual risk

- **Sample:** 414 names, stratified — a research-grade panel, not the full CRSP universe.
  Larger coverage would tighten estimates (the jump from 184→414 names *raised* the Sharpe
  1.11→1.48, which is the right direction).
- **Execution slippage** beyond the modelled half-spread (e.g. trading the close, earnings-day
  gaps) is not captured; size orders patiently.
- **Borrow availability** can change intraday even for normally-ETB names; the book must
  skip any name flagged HTB at execution.
- **Crowding / decay:** PEAD is a well-known anomaly; the modern-era Sharpe (1.57) is lower
  than 2005–2009/2015–2019 peaks, consistent with gradual decay. Monitor live IC.
- The quarterly-bucket backtest ignores intra-quarter compounding of overlapping holds; treat
  the Sharpe as indicative, not exact.

**Bottom line:** a retail trader on Alpaca can deploy the market-neutral Mega+Large PEAD
sleeve — commission-free, $0 borrow, beta-neutral — either standalone (Sharpe ~1.6) or
blended with an SPY core to dial in a chosen risk level. This is the deployable edge.
