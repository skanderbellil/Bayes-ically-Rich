# Pay-up follow book — the synthesis that survives out of sample

> The two strongest behavioural findings compose into one signal that holds up.
> `TRADE_QUALITY` said **pay-up urgency** marks the informed fill; `ORDER_FLOW`
> said **breadth/consensus** predicts the cross-section. Fuse them — an
> **informed-flow** signal that sums the pool's directional dollars *weighted by
> how far each fill paid through the mid* — trade it **dollar-neutral** (so the
> favorite-drift beta cancels), and it nets Sharpe **0.70** after 50 bps, beating
> the plain breadth (0.45) and imbalance (0.25) books. The clincher: on a strict
> **out-of-sample slice** (post-election, 2025-01 →), it still nets **0.67** —
> gross actually *rises* to 1.87 — so it is **not** a 2024-cycle artefact. This is
> the closest thing to a deployable edge in the whole Polymarket thread.

```bash
python experiments/run_polymarket_payup.py
python experiments/run_polymarket_payup.py --lookback 14 --start 2025-01-01
```

## The signal

For each token-day, over a trailing 14-day window, sum across the non-MM pool's
fills::

    informed_flow = Σ  sign(side) · usdcSize · max(price − mid, 0)

i.e. dollars that **paid up** in their direction (a BUY above the mid, a SELL
below) — passive fills inside the mid contribute nothing. It is *consensus
weighted by urgency*: many traders aggressively crossing the spread the same way.
Traded long-top / short-bottom 20% by the day's cross-sectional z-score, held one
day, 50 bps on turnover — the same dollar-neutral engine as `ORDER_FLOW`, so the
common favorite-drift beta is removed and only ranking power is paid out.

## Result (177 wallets → 238 priced tokens, dollar-neutral, 50 bps/turn)

| signal | full gross | full net | full PnL | full maxDD | OOS gross | **OOS net** | OOS PnL |
|--------|-----------:|---------:|---------:|-----------:|----------:|------------:|--------:|
| `informed` (pay-up × consensus) | 1.45 | **0.70** | +0.38 | −0.16 | 1.87 | **0.67** | +0.21 |
| `breadth` | 1.29 | 0.45 | +0.24 | −0.24 | 1.69 | 0.31 | +0.09 |
| `imbalance` | 1.05 | 0.25 | +0.14 | −0.20 | 1.37 | 0.10 | +0.03 |

Three reads:

1. **The synthesis beats its parts.** Weighting flow by pay-up urgency lifts net
   Sharpe from 0.45 (breadth) to 0.70 and cuts the drawdown (−0.16 vs −0.24) —
   urgency adds genuine information on top of consensus, exactly as the
   trade-level study predicted.

2. **It holds out of sample.** The whole ranking (informed > breadth > imbalance)
   is preserved on the post-election slice, and the informed book's *gross* Sharpe
   is actually higher there (1.87). The edge is not the 2024 election complex.

3. **Order ranks with information content.** imbalance (raw dollars) < breadth
   (distinct-trader consensus) < informed (consensus × urgency) — each refinement
   that moves closer to "aggressive agreement" earns more.

## Honest verdict & limitations

This is a real, beta-neutral, cost-surviving, out-of-sample-stable signal — the
strongest in the thread. But "deployable" still needs work it hasn't had:

- **Long-tilt residue.** Like `ORDER_FLOW`, most of the edge is in the long (high
  informed-flow) leg; the short leg mainly buys beta-neutrality.
- **Spread is the real cost.** Marks to the CLOB mid and charges a flat 50 bps;
  Polymarket's true cost is spread-dominated and worst on the thin, away-from-
  headline books the signal often points at. The pay-up tell also carries the
  mild price-impact endogeneity flagged in `TRADE_QUALITY`.
- **In-sample knobs.** Lookback (14d), top-frac (20%) and the pay-up weighting are
  not tuned out-of-sample; the OOS *slice* validates the period, not the
  hyper-parameters.
- **Pool survivorship.** Leaderboard-seeded pool, tokens need ≥10 daily prices.

Read it as: *the behavioural signal is real and robust; turning it into a live
book requires honest spread modelling and walk-forward hyper-parameter selection.*
