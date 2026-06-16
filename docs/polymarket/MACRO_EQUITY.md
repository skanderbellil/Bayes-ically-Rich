# Macro buy-leader — the $1,000 equity curve

Visualises the **one surviving directional candidate** of the entire Polymarket thread.
In **macro** multi-outcome fields (Fed decisions, CPI/inflation prints, rate-path
events) the field leader is materially under-priced — it wins **93%** of the time at a
mean entry price of **0.73**, worth **+19.8¢/event** with an event-clustered **t ≈ 3**
(`FIELD_SHAPE.md`, `STRATEGY_SYNTHESIS.md`). Tripling the universe (24 → 83 fields)
sharpened it rather than killing it, while the *general* leader edge dissolved to
t=1.35 — so macro stands alone.

```bash
python experiments/run_polymarket_macro_equity.py --start 1000
```

## The record (15 macro fields, chronological)

- **14 wins / 1 loss.** Leader win-rate 0.93, mean entry 0.726, mean per-stake return
  **+28.7%**. Per-win returns span +3% to +63% (pay ≈0.73, collect 1.00).
- The single loss is a macro favorite that resolved No on **2024-09-18** (it lands
  first, so the curve dips at the open then climbs through 14 straight wins).

## $1,000 → (Sept 2024 → May 2026, ≈20 months)

| sizing per bet | final | max drawdown |
|---|---:|---:|
| 10% of bankroll | **$1,511** | −10% |
| 20% of bankroll | **$2,205** | −20% |
| quarter-Kelly (≈19%) | $2,121 | −19% |

## How to read it honestly

* **Endpoints are order-invariant.** With a single losing event, fixed-fraction
  compounding commutes — the final dollars don't depend on the sequence, and the worst
  single-event drawdown is *exactly* the bet fraction. The dip-first shape is cosmetic;
  the $ outcomes are robust to ordering.
* **Don't size up.** Full-Kelly here is ≈0.76 of bankroll — an artefact of a 93% win
  rate over **15 in-sample, one-regime events**. The downside is −100% of stake, so an
  out-of-sample cluster of losses (which 15 events cannot reveal) is ruinous at high
  fractions. 10–20% is the honest band, and even that trusts a win-rate that will not
  fully persist.
* **It is a short-volatility trade.** You win small and often, and the rare loss is
  near-total. The Sharpe looks great precisely because the sample contains only one
  realisation of the tail.

## Verdict

A clean, real-looking **+50% to +120% over ~1.7 years** at sane sizing — but it is the
*in-sample* picture of a 15-event short-tail trade in one calm-ish regime. The only way
to earn trust in it is **forward, out-of-sample**, on the next FOMC / CPI / macro
fields — which is exactly what the paper-trade tracker (`PAPER_TRADE.md`) is for.
