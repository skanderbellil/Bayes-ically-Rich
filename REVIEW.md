# PEAD Research — Critical Review & Corrections

A code-and-data review of the PEAD pipeline. Five flaws materially inflated the
reported edge. After fixing them and benchmarking against **real** SPY, the
tradeable strategy **does not beat buy-and-hold** on a fully-invested basis.

## Summary of corrected numbers

Small-cap, top-25% SUE, correct return math, no costs:

| Accounting | Annual | Sharpe | Max DD |
|---|---|---|---|
| As-backtested (anchored to pre-announcement close, **untradeable**) | +18.6% | 1.04 | −21% |
| Tradeable (enter at t+1 close) | **+8.85%** | 0.63 | −40% |

Tradeable + trailing-percentile selection + Alpaca costs, by bucket:

| Bucket | Gross | Net | Sharpe | Max DD |
|---|---|---|---|---|
| Mega Cap | +7.88% | **+7.64%** | 0.85 | −25% |
| Large Cap | +4.51% | +4.03% | 0.39 | −25% |
| Mid Cap | +7.37% | +6.41% | 0.60 | −32% |
| Small Cap | +7.63% | +5.71% | 0.52 | −40% |
| **SPY (real, buy & hold)** | **+10.43%** | **+10.43%** | **0.60** | — |

**Verdict:** best tradeable bucket (Mega) nets +7.64%/yr vs SPY +10.43%/yr →
**−2.79%/yr net alpha**, while being only ~1 month/quarter invested (cash drag
not even charged) and resting on a survivorship-biased universe. The one bright
spot: mega-cap's risk-adjusted return (Sharpe 0.85) edges SPY's 0.60 — but on
lower absolute return and the least survivorship-exposed bucket.

---

## The five flaws

### 1. Untradeable announcement jump (critical)
`signals.py:88` anchors all forward returns to `p_t0`, the close **on/before**
the announcement. Earnings post after the close, so you cannot transact at `p_t0`.
Earliest entry is the t+1 close. Since the stored returns are **log** returns,
tradeable drift is additive: `drift = ret_t21 − ret_t1`.

Effect on the SUE signal (Spearman IC, all caps):

| Horizon | IC |
|---|---|
| `ret_t1` (pure jump) | +0.186 |
| `ret_t21` (incl. jump, as used) | +0.130 |
| **tradeable drift (t+1→t+21)** | **+0.028** |

~85% of the apparent signal was the announcement-day pop. Fix: trade
`drift_t1_to_t21`.

### 2. Log returns averaged as simple returns (critical)
`backtest.py:78` averages log returns across names and compounds with `(1+r)`.
You cannot do that. Evidence: stored `ret_t21` spans −310%…+156%, `ret_t63`
to −373% — impossible as simple returns. This produced the **−150% to −172%
"max drawdowns"** (a long book can't lose >100%). Fix: `expm1(logret/100)` per
name **before** averaging (`corrected.py:portfolio_quarter_return`).

### 3. Mocked benchmark hid negative alpha (critical)
`research_strategy_comparison` / the "simple→complex" chart generated SPY with
`np.random.normal(...)`. Its own printout was strategy +5.23% net vs SPY +9.92%
— i.e. it **lost** to buy-and-hold — but rendered the gap as `+-4.69%`. Fix:
cached real SPY (`results/pead/spy_quarterly.csv`, +10.43%/yr, Sharpe 0.60).

### 4. In-quarter quantile look-ahead (major)
`group["sue"].quantile(0.75)` ranks a name against others that report **later**
the same quarter. Fix: expanding/trailing percentile
(`corrected.py:trailing_percentile_mask`).

### 5. Survivorship bias + inconsistent Sharpe (major/minor)
The universe is live-only; yfinance returns nothing for delisted tickers, so
bankrupt/negative-SUE small-caps are missing — biasing returns **up** exactly in
the previously-recommended bucket. Separately, `backtest.py` annualized Sharpe
with `√252` on quarterly data (inflated); standardized to `√4` here. Overlapping
21-day holds bucketed to quarters also violate the IID assumption behind Sharpe.

---

## What changed in code
- `src/pead/corrected.py` — correct return math, tradeable drift, trailing
  percentile, √4 Sharpe, real-SPY loader.
- `research_pead_corrected.py` — honest backtest + `corrected_vs_spy.png`.
- `results/pead/spy_quarterly.csv` — cached real SPY quarterly returns.

## What to do next (if pursuing PEAD)
1. Source point-in-time, survivorship-free earnings + delisting returns.
2. Model cash drag honestly (overlapping cohorts to stay ~fully invested).
3. Re-test mega-cap tradeable drift out-of-sample — it is the only bucket with a
   plausible, survivorship-robust, risk-adjusted edge, and it is small.

---

## Update: a real, robust, dynamic improvement (the overlay)

After fixing the flaws, the standalone tradeable PEAD sleeve does not beat SPY.
But blending the **mega-cap tradeable sleeve** (survivorship-robust; corr 0.58 to
SPY) with the index does improve risk-adjusted return — and it is robust because
even the zero-parameter static blend works.

| Strategy | Sharpe (full) | Sharpe (OOS half) | Annual | Max DD |
|---|---|---|---|---|
| SPY buy & hold | 0.81 | 0.94 | +13.2% | -46% |
| Static 50/50 (0 params) | 0.91 | 1.20 | +10.5% | -31% |
| Dynamic-weight overlay | 0.97 | 1.20 | +12.6% | -37% |

- **Robust**: the parameter-free 50/50 blend already lifts Sharpe and cuts the
  worst drawdown by a third; holds in the out-of-sample second half.
- **Dynamic**: PEAD weight scales with the sleeve's trailing Sharpe (regime
  awareness), using only past data (`.shift(1)`) — no look-ahead.
- **Why it works**: the mega-cap PEAD sleeve sits in cash ~2 months/quarter, so it
  is naturally low-beta and weakly correlated with SPY; combining a defensive,
  low-correlation sleeve with the index raises Sharpe and reduces drawdown. This is
  portfolio diversification, not data-mining.

Code: `research_pead_overlay.py`, `src/pead/corrected.py`. Plot:
`results/pead/overlay_vs_spy.png`.

Honest caveats: returns assume the PEAD sleeve can be deployed each quarter (point-
in-time earnings needed live); mega-cap minimizes but does not eliminate data
issues; the absolute return is slightly below SPY — the gain is risk-adjusted
(higher Sharpe, lower drawdown), which is what was asked for.
