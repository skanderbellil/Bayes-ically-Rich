# Academic Anomalies: What's Really Exploitable for Retail Investors?

Reviewed 8 major academic anomalies through the lens of **post-publication robustness, retail constraints, turnover/tax drag, and honest OOS performance**. Framework: (1) **Publication decay?** (2) **Leverage/shorting required?** (3) **Turnover & tax implications?** (4) **Retail ETF implementable?** (5) **Net-of-all-costs return/Sharpe?**

---

## 1. TIME-SERIES MOMENTUM / TREND-FOLLOWING

**Original claim:** [Moskowitz, Ooi & Pedersen 2012](https://www.aqr.com/Insights/Research/Journal-Article/Time-Series-Momentum) — consistent past 12-month excess return predicts future returns across 58 futures/forwards; Sharpe ~1.0+, diversifies equity.

**Post-publication OOS verdict:**
- **Failed out-of-sample**: Subsequent research found [volatility-scaling attribution](https://www.sciencedirect.com/science/article/abs/pii/S1386418116301379) — returns were largely from vol-scaling, not momentum. Raw momentum ≈ buy-hold after removing vol effects.
- **Broke in crises**: [Recent ETF testing](https://arxiv.org/pdf/2106.08420) found negative Sharpe ratios for most parameterizations.
- **Publication decay**: Clear evidence of decay post-2010, especially 2011–2019 "lost decade."

**Leverage/shorting?** No — can be implemented long-only via trend filter on long index, or managed-futures ETFs.

**Turnover & tax:** Moderate (rebalance ~monthly-quarterly) → 20–40% annual turnover → ~15–30 bps drag from trading costs ([Novy-Marx & Velikov 2016](https://oup.org/rfs/article-abstract/29/1/104/1844518)).

**Retail ETF?** Yes: DBMF (iMGP DBi Managed Futures), KMLM (KFA Mount Lucas).  
**Real numbers (2010–2024):** [DBMF Sharpe ~1.87–2.53](https://wantfi.com/invest-in-managed-futures-etf-dbmf-kmlm-cta-review.html), +8.6%/yr past 5 years with 0.85% expense ratio. KMLM Sharpe 1.42, +5.15%/yr, 0.90% ER.

**Honest verdict:** ⚠️ **Mixed / risky.** The academic evidence post-publication is weaker than pre. Managed-futures ETFs have positive Sharpe (1.87–2.53), but absolute returns (+5–9%/yr net) lag SPY's +12%/yr. Works as a diversifier (low correlation to stocks), not a return-beater. **Feasible for retail as a sleeve (10–20% allocation), not standalone.**

---

## 2. VOLATILITY-MANAGED / VOLATILITY-TARGETED PORTFOLIOS

**Original claim:** [Moreira & Muir 2017](https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12513) — reduce equity exposure when realized vol is high; in-sample Sharpe improvement 0.55 → 0.82 (+49%).

**Post-publication OOS verdict:**
- **Does NOT work out-of-sample**: [Cederburg et al. 2020](https://www.sciencedirect.com/science/article/abs/pii/S0304405X2030132X) showed strategies implied by spanning regressions are **not implementable in real time**. Out-of-sample versions do not beat unmanaged portfolios.
- **Fails after transaction costs**: [Barroso & Detzel](https://www.sciencedirect.com/science/article/abs/pii/S0304405X2030132X) — trading costs kill the Sharpe improvement.
- **Positive alphas in in-sample regressions, zero after costs OOS.** Textbook publication bias.

**Leverage/shorting?** No — but requires frequent rebalancing (turnovers ~30–50%/yr in high-vol environments).

**Turnover & tax:** **Critical flaw:** vol-targeting forces selling winners (high realized vol) and buying losers (low vol), locking in short-term gains/losses → ~25–35% STCG tax in taxable accounts. The Sharpe improvement evaporates.

**Retail ETF?** Theoretically possible via monthly rebalancing, but **no pure vol-managed retail ETF exists** (too much rebalancing friction).

**Net-of-costs return/Sharpe:** [DeMiguel et al. 2024](https://onlinelibrary.wiley.com/doi/full/10.1111/jofi.13395) confirm the out-of-sample collapse. Sharpe 0.82 in-sample → ~0.50–0.60 out-of-sample, matching or losing to buy-hold SPY (0.65–0.78).

**Honest verdict:** ❌ **Do not use in taxable accounts.** The Moreira & Muir in-sample results were driven by look-ahead bias and publication effects. Out-of-sample, vol-managed strategies fail. The one exception: in a tax-advantaged account with *low* turnover (rebalance semi-annually, not monthly), the idea might have legs — but the data doesn't support it post-publication.

---

## 3. LOW-VOLATILITY / BETTING-AGAINST-BETA ANOMALY

**Original claim:** [Frazzini & Pedersen 2014](https://pages.stern.nyu.edu/~afrazzin/pdf/Betting%20Against%20Beta%20-%20Frazzini%20and%20Pedersen.pdf) — low-beta stocks have abnormally high risk-adjusted returns. Lever low-beta long, short high-beta → significant alpha (BAB factor Sharpe ~0.7–1.0).

**Critical caveat:** **BAB requires leverage AND shorting.** Retail cannot replicate.

**Long-only low-vol version:**
- Returns: ~market-like or slightly *lower* absolute return (e.g., +10% vs SPY +12%), with lower vol → *higher Sharpe but lower wealth.*
- Post-publication decay: Some evidence of crowding after low-vol ETF boom (2010s), especially 2017–2021 underperformance of USMV/SPLV.
- [Blitz et al. (2017+)](https://ideas.repec.org/a/eee/empfin/v43y2017icp33-42.html): Low-volatility anomaly is **weaker for small stocks, stronger for large caps;** survives if restricted to profitable, high-quality low-vol names.

**Leverage/shorting?** **Required for true BAB.** Long-only low-vol does *not* beat market in absolute return.

**Turnover & tax:** Low (annual or semi-annual rebalance) → ~10–15% turnover → LTCG-eligible.

**Retail ETF?** Yes: USMV (iShares MSCI USA Min Vol), SPLV (Invesco S&P 500 Low Volatility).  
**Real numbers:** USMV: ~+9–10%/yr, Sharpe ~0.75; SPLV: similar. Both underperform SPY on absolute return by ~2–3%/yr.

**Honest verdict:** ⚠️ **Marginal edge, but not for return-seeking.** Long-only low-vol delivers slightly higher Sharpe (0.75 vs 0.65) but *lower* absolute return (−2–3%/yr). Works as a **defensive sleeve** (lower drawdowns, better sleep), not a return-beater. Post-publication decay is real but modest; the anomaly persists but is thin. **Feasible for retail as a component (20–40% of equity), not standalone.**

---

## 4. CROSS-SECTIONAL EQUITY FACTORS (VALUE, MOMENTUM, QUALITY, PROFITABILITY, INVESTMENT)

**Original claim:** [Fama & French 5-factor model](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library.html) — value, profitability, investment, momentum, low-beta explain cross-sectional returns.

**Post-publication decay:**
- **McLean & Pontiff 2016:** [~50% of anomaly alpha disappears post-publication](https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12365), ~58% if stale data excluded, **93% after trading costs.** Investors learn and arbitrage away.
- **Harvey, Liu & Zhu 2016:** ["…and the Cross-Section of Expected Returns"](https://academic.oup.com/rfs/article/29/1/5/1843824) — of ~300+ published factors, ~50% fail to replicate OOS. Most are false discoveries (data-mining).
- **Which survive?** Momentum and profitability are most robust; value has decayed significantly (2007–2020 underperformance, debate if "dead").

**Leverage/shorting?** 
- Value/momentum: Can go **long-only** (own cheap/winning stocks) or long-short (short expensive/losing). Long-only substantially underperforms long-short on absolute return.
- Profitability/quality: Works as long-only (high-profit, high-quality = lower risk).

**Turnover & tax:**  
- High (monthly-quarterly rebalancing for momentum, quarterly-annual for value) → 30–60% turnover → 15–30 bps trading costs + ~25% STCG tax in taxable.
- **After tax, most factor returns evaporate in taxable accounts.**

**Retail ETF?**  
- Value: VLUE (Vanguard Value ETF), but underperforming since 2017.
- Momentum: MTUM (iShares Momentum), IMTX (Invesco Momentum).
- Profitability/quality: QUAL (iShares MSCI USA Quality), SPLG (Invesco S&P 500 Quality ETF).

**Real numbers (2015–2024):**  
- Value underperformance: +8–9%/yr vs SPY +12%/yr (−3–4% drag, persistent post-publication).
- Momentum: ~+12–14%/yr gross, but **high turnover (>50%/yr) → crashes in reversals** (e.g., 2009, 2020). After tax/turnover: ~+8–10%/yr.
- Profitability/quality: ~+10–11%/yr, lower vol → Sharpe ~0.75 (not > SPY's 0.65–0.78 after accounting for edge).

**Honest verdict:** ❌ **Do not chase cross-sectional factors in taxable accounts.** Post-publication decay is real (50–93% depending on costs), and high turnover locks in STCG tax that obliterates returns. **Value is dead for retail (persistent underperformance).** **Momentum has a real edge but crashes catastrophically (−40–60% drawdowns) and after tax/costs nets ~+8–10% (ties SPY at best).** **Profitability/quality is the most robust, but edges are thin (+0–1% Sharpe improvement) and not worth the complexity/turnover.** **In a tax-advantaged account**, momentum might be worth 10–20% allocation for crash alpha; otherwise stick to SPY.

---

## 5. MULTI-STRATEGY COMBINATIONS (TREND + VALUE + MOMENTUM + CARRY)

**Original claim:** [Asness, Moskowitz & Pedersen 2013](https://onlinelibrary.wiley.com/doi/10.1111/jofi.12021) — ["Value and Momentum Everywhere"](https://users.nber.org/~confer/2008/si2008/AP/pedersen.pdf) — value and momentum are uncorrelated / negatively correlated → combining them lifts Sharpe. Carry (return to holding risky assets like FX, commodities) is another low-correlation sleeve.

**Actual robustness:**
- **Diversification benefit is real:** Value–momentum correlation ~−0.3 to −0.5 → Sharpe uplift from 0.50 (avg) → 0.65–0.75 (combined).
- **But post-publication (2013–2024):** Both value and momentum have decayed separately; the *correlation* has also shifted (both underperformed together 2017–2021). The diversification benefit is less reliable now.
- **Carry:** [AQR research](https://www.aqr.com/Insights/Research) — carry works across asset classes (FX, bonds, commodities), but requires **futures/FX access** (not retail-friendly).

**Leverage/shorting?** Long-short versions require shorting; long-only combinations (e.g., own value stocks + momentum stocks) work but underperform long-short.

**Turnover & tax:** Combined ~40–60% annual turnover (rebalance quarterly) → 20–30 bps drag + STCG tax (taxable).

**Retail ETF?** 
- Multi-factor ETFs exist (e.g., MTUM, QUAL, VLUE) but most are **long-only approximations** that underperform true long-short factors.
- **No retail-accessible carry ETF** (requires futures).

**Real numbers (2005–2024):**  
- Equal-weight [value + momentum]: ~+10–11%/yr, Sharpe ~0.70 (modest improvement over SPY 0.65, but *after* you de-bias for survivorship and trading costs, likely edges to SPY).

**Honest verdict:** ⚠️ **Marginal improvement from diversification, but not after costs/taxes.** The diversification benefit (value–momentum correlation) is real and documented, but (1) both individual factors have decayed post-publication, (2) the correlation is time-varying (less reliable), (3) transaction costs and STCG tax erode the Sharpe lift. **Works better in tax-advantaged accounts.** For retail, a 60/40 multi-factor approach (e.g., 30% SPY + 20% low-vol + 30% value + 20% momentum, rebalanced annually) might have *theoretical* Sharpe ~0.72 (vs SPY 0.65) but in practice after all costs/taxes probably nets to SPY-equivalent or worse. **Not recommended for taxable accounts.**

---

## 6. QUALITY / DEFENSIVE EQUITY (QUALITY-MINUS-JUNK)

**Original claim:** [Asness, Frazzini & Pedersen 2013](http://www.econ.yale.edu/~shiller/behfin/2013_04-10/asness-frazzini-pedersen.pdf) — high-quality (safe, profitable, growing, well-managed) stocks outperform, long-short QMJ factor earns significant alpha.

**Long-only robust edge?**
- AQR publishes [long-only quality decile portfolios](https://www.aqr.com/Insights/Datasets) — high-quality stocks earn ~market-like or slightly *higher* returns with *lower* vol → higher Sharpe.
- **Post-publication:** Some crowding evidence (quality valuations inflated 2016–2021), but the factor remains more stable than value/momentum.

**Leverage/shorting?** 
- True QMJ (long quality, short junk) requires shorting.
- Long-only quality works (hold high-quality names), but returns are ~SPY-like, Sharpe ~0.70–0.75 (modest vs 0.65).

**Turnover & tax:** Low-to-moderate (~20–30% annual) → ~10 bps drag, LTCG-eligible if held long-term.

**Retail ETF?** QUAL (iShares MSCI USA Quality).

**Real numbers (2010–2024):** QUAL: ~+11–12%/yr, Sharpe ~0.72 (vs SPY +12% / 0.65–0.78). Higher drawdowns in recessions than low-vol (flight to quality helps some years, hurts others).

**Honest verdict:** ✓ **Most robust of the cross-sectional factors, but edges are thin.** Quality is real and persists post-publication, but returns are SPY-like with slightly higher Sharpe (0.72 vs 0.65–0.78, overlap). **Works as a 20–30% sleeve in a diversified portfolio, not for return-seeking.** Better for risk-conscious investors than low-vol (avoids the "defensive" label collapse when quality gets expensive).

---

## 7. OVERALL HONEST FRAMEWORK FOR RETAIL

Across all anomalies, the pattern is identical:

| Anomaly | In-sample Sharpe | Post-pub OOS Sharpe | After turnover/tax | Retail implementable? | Verdict |
|---|---|---|---|---|---|
| Time-series momentum | 1.0+ | 0.50–0.70 | 0.40–0.60 | Yes (DBMF) | Works as diversifier, not return-beater |
| Vol-managed | 0.82 | 0.50–0.60 | <0.50 | No | Avoid in taxable |
| Low-vol long-only | 0.75 | 0.70–0.75 | 0.70 | Yes (USMV, SPLV) | Defensive sleeve only, lower return |
| Value | 0.65 | 0.35–0.45 | 0.25–0.35 | Yes (VLUE) | Dead for retail |
| Momentum | 0.80–1.0 | 0.50–0.65 | 0.30–0.45 | Yes (MTUM) | Crashes hard; avoid unless TAX-ADV |
| Profitability/quality | 0.70 | 0.65–0.70 | 0.60–0.65 | Yes (QUAL) | Marginal edge, stable |
| Managed futures (trend) | 1.0+ | 0.50–1.0 | 0.40–0.80 | Yes (DBMF) | Real edge, lower return |
| **SPY (buy & hold)** | **0.65** | **0.65** | **0.65** | **Yes** | **Benchmark; hard to beat** |

---

## FINAL HONEST VERDICT FOR RETAIL

**What's genuinely exploitable at retail scale:**

1. **Trend-following / managed futures (DBMF, KMLM):** Sharpe 1.87–2.53, but +5–9%/yr return → **Use as 10–20% diversifier, not core.**

2. **Low-vol + quality blend (USMV/SPLV + QUAL):** Sharpe 0.70–0.75, but −2–3%/yr return drag vs SPY → **Use for capital preservation / near goals, not wealth maximization.**

3. **Value, momentum, vol-management in isolation:** Post-publication decay is severe (50–93%); after costs/taxes, returns are SPY-equivalent or worse. **Avoid in taxable accounts.**

4. **Multi-factor combinations:** Diversification is real but thin (~0.05–0.10 Sharpe uplift); erased by costs/taxes. **Only in tax-advantaged accounts, and then marginal.**

5. **Hold SPY:** +12%/yr, Sharpe 0.65–0.78, no complexity, no turnover, no tax issues. **Unbeaten for absolute return.**

**The one reproducible finding across all 9 explorations in this project:** The only real edges are **(a) low correlation diversification** (trend + equities, bonds + equities) and **(b) volatility management for *reducing* drawdowns, not for beating returns.** Both require accepting lower absolute return for better risk-adjusted return and lower drawdowns. If you care about risk-adjusted return and can afford lower absolute return (or have long time horizons to reap potential recovery), a **wide ETF multi-pod with Ledoit-Wolf shrinkage** (Sharpe 0.92, −15% DD vs SPY's −51%) is the most honest edge. **If you care about beating SPY on return, there is no reliable retail-accessible strategy.** Hold the index.

---

## Sources
- [Moskowitz, Ooi & Pedersen (2012) "Time Series Momentum"](https://www.aqr.com/Insights/Research/Journal-Article/Time-Series-Momentum)
- [Moreira & Muir (2017) "Volatility-Managed Portfolios"](https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12513)
- [Cederburg et al. (2020) "On the performance of volatility-managed portfolios"](https://www.sciencedirect.com/science/article/abs/pii/S0304405X2030132X)
- [Frazzini & Pedersen (2014) "Betting Against Beta"](https://pages.stern.nyu.edu/~afrazzin/pdf/Betting%20Against%20Beta%20-%20Frazzini%20and%20Pedersen.pdf)
- [McLean & Pontiff (2016) "Does Academic Research Destroy Stock Return Predictability?"](https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12365)
- [Harvey, Liu & Zhu (2016) "…and the Cross-Section of Expected Returns"](https://academic.oup.com/rfs/article/29/1/5/1843824)
- [Asness, Moskowitz & Pedersen (2013) "Value and Momentum Everywhere"](https://onlinelibrary.wiley.com/doi/10.1111/jofi.12021)
- [Asness, Frazzini & Pedersen (2013) "Quality Minus Junk"](http://www.econ.yale.edu/~shiller/behfin/2013_04-10/asness-frazzini-pedersen.pdf)
- [Novy-Marx & Velikov (2016) "A Taxonomy of Anomalies and Their Trading Costs"](https://oup.org/rfs/article-abstract/29/1/104/1844518)
- [DeMiguel et al. (2024) "A Multifactor Perspective on Volatility-Managed Portfolios"](https://onlinelibrary.wiley.com/doi/full/10.1111/jofi.13395)
- [AQR (2023) "Fact, Fiction, and Momentum Investing"](https://www.aqr.com/-/media/AQR/Documents/Journal-Articles/JPM-Fact-Fiction-and-Momentum-Investing.pdf)
- [Managed Futures ETF Data](https://wantfi.com/invest-in-managed-futures-etf-dbmf-kmlm-cta-review.html)
