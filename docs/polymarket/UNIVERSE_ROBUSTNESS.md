# Universe robustness — the mispricing is political, not structural

> **Verdict:** the [0.15,0.30] favorite–longshot underpricing (`BAND_CONTROL.md`)
> is **not** a universal feature of prediction-market pricing. It is concentrated in
> **politics (+42%) and geopolitics (+22%)** event markets and **vanishes in sports
> (≈0, the largest category) and crypto**. The "structural edge" was an event-domain
> artifact — and politics is dominated by the single 2024-election path.

```bash
python experiments/run_polymarket_universe_robustness.py --refresh --n-markets 600
```

## Setup

Pulled a larger, more diverse universe — **514 resolved markets** (998 days,
25,478 episodes), well beyond the election-heavy top-volume set — and tagged each by
topic (`categorize.market_category`, keyword rules over question + parent-event
title). Then re-ran the [0.15,0.30] calibration **per category**.

## Universe composition

| category | markets | base Yes-rate | [0.15,0.30) episodes |
|---|---:|---:|---:|
| sports | 185 | 4.3% | 426 |
| politics | 163 | 16.0% | 562 |
| macro | 53 | 22.6% | 110 |
| other | 44 | 29.5% | 209 |
| geopolitics | 37 | 43.2% | 142 |
| crypto | 20 | 15.0% | 227 |
| culture | 12 | 0.0% | 37 |

## [0.15,0.30] calibration residual (Yes-rate − price) by category

| | residual | reading |
|---|---:|---|
| **politics** | **+41.7%** | priced ~0.23, resolved Yes **64%** — massive underpricing |
| **geopolitics** | **+21.8%** | war / ceasefire markets, also strongly underpriced |
| sports | **+0.3%** | well-calibrated (n=426, the largest category) |
| crypto | −1.4% | well-calibrated |
| other | −6.5% | — |
| macro | −18.5% | Fed/rates markets *over*priced in the band |
| culture | −20.0% | small n |

Aggregated three ways:

| group | episodes | mean price | Yes-rate | calib residual |
|---|---:|---:|---:|---:|
| ALL | 1713 | 0.22 | 34.7% | +13.0% |
| **politics** | 562 | 0.23 | 64.4% | **+41.7%** |
| **non-politics** | 1151 | 0.21 | 20.2% | **−1.1%** |

## What it means

1. **The edge is political/geopolitical, not structural.** Strip politics out and the
   band's calibration residual is ≈ 0 (−1.1%). The +13% pooled "structural mispricing"
   was the 2024-election + war complex, where mid-priced contested Yes-outcomes
   resolved Yes far more often than priced. In **sports** — the largest non-political
   category — markets are essentially **well-calibrated** (+0.3%): no edge.
2. **Even the political edge is a path artifact.** One election cycle and one
   Russia/Ukraine/Iran war period is a *single realised macro path*. "Contested
   political/war Yes-outcomes underpriced" could be a real behavioural effect (people
   under-bet likely-but-scary outcomes) or simply that 2024–25's specific outcomes
   landed Yes. With n≈1 event cycle the two are inseparable.
3. **A note on the trade number.** The per-category *hold-to-resolution trade* shows
   non-politics +0.09/trade despite the ≈0 episode residual — because it enters once,
   on a market's *first* touch of the band (a momentum-selected sample) and is carried
   by geopolitics. The unbiased episode-level residual (≈0 ex-politics) is the
   statistic to trust; the trade figure is a weaker, entry-timing-dependent claim.

## Bottom line for the whole Polymarket thread

Across four studies the honest arc is: a sharp-move "edge" → revealed as price-level
calibration → revealed as the favorite–longshot underpricing of mid-priced markets →
revealed as a **politics/geopolitics, largely-2024 artifact** that does not generalise
to sports or crypto. No deployable, universe-robust timing or calibration edge
survived. What *did* survive is a clean, reusable prediction-market research stack
(live data → log-odds signals → BOCPD events → cost-aware trades → level & universe
controls) and a worked example of confounds being peeled away one layer at a time.

The remaining genuinely-open question — not answerable from resolved history — is
whether **live order-book imbalance** at a spike can separate informed political/war
moves from noise *before* the price adjusts.
