# Harvesting the sharp-move edge — exit policy is the whole game

The vol→outcome study (`SHARP_MOVE_OUTCOME.md`) found two opposite-signed effects
after a sharp upside move: the market **underprices Yes** (a positive *calibration*
edge, realised only at settlement) but **overreacts short-term** (negative forward
drift). This study turns those facts into trades and shows the exit policy decides
whether the edge is harvested or thrown away.

```bash
python experiments/run_polymarket_event_trades.py --refresh
python experiments/run_polymarket_event_trades.py --metric sharp_up --slippage 0.02
```

## Setup — with reality checks

A trade opens when a market's `upside_vol` crosses into its top quantile (Q5 =
top 20%, Q4+Q5 = top 40%). Frictions are deliberately conservative:

- **causal trigger** — the metric at *t* uses prices ≤ *t*;
- **fill after the move** — entry at *t+1*, at the already-elevated price (we never
  get the pre-spike price);
- **slippage** on every book crossing (1–2¢); settlement at resolution pays 0/1 with
  no exit slippage;
- one position per market, re-entry cooldown.

Two exit policies:
- **Idea 1 — hold to resolution.** PnL = side·(outcome − entry). Harvests the
  calibration edge; the short-term overreaction is irrelevant.
- **Idea 2 — trade the drift.** Take-profit at +10¢ / stop −25¢, or mark out after
  a fixed horizon. Directly exposed to the overreaction.

## Result (250 resolved markets, base Yes-rate 16%, 1% slippage/crossing)

| config | n | hit | avg PnL/trade | total PnL | trade-Sharpe | avg entry | avg hold |
|---|---:|---:|---:|---:|---:|---:|---:|
| **res · Q5 · Yes** (Idea 1) | 101 | 32.7% | **+0.049** | +5.0 | 0.111 | 0.27 | 86d |
| res · Q4Q5 · Yes (Idea 1) | 117 | 31.6% | +0.039 | +4.5 | 0.086 | 0.27 | 108d |
| res · Q5 · directional | 101 | 38.6% | +0.025 | +2.5 | 0.055 | 0.27 | 86d |
| TP · Q5 · Yes (Idea 2) | 174 | 49.4% | −0.002 | −0.3 | −0.006 | 0.32 | 26d |
| horizon10 · Q5 · Yes (Idea 2) | 207 | 34.3% | −0.011 | −2.3 | −0.051 | 0.32 | 9d |

Three clean readings:

1. **Idea 1 works — and it's the calibration edge being paid out.** Buying Yes on
   a Q5 upside spike and holding to settlement earns **~+5¢ per $1 contract** (avg
   PnL +0.049), and the cumulative curve climbs to +5–6 over the sample. It is the
   mechanical payout of "entry ≈ 0.27, but these markets resolve Yes 32.7% of the
   time." It **survives 2% slippage** (avg PnL +0.039).
2. **Idea 2 fails — the overreaction eats it.** Trading the drift breaks even
   (take-profit) or loses (fixed horizon, −0.011), exactly as the negative Q5
   forward drift predicted. The take-profit's flashy 49% hit-rate is a mirage: it
   banks small winners and lets the reverting losers run to the stop. The cumulative
   curves drift *down* to −2…−4.
3. **The side matters, asymmetrically.** Letting `vol_skew` pick the side
   (`directional`) is *weaker* than always buying Yes (+0.025 vs +0.049) — the edge
   is specifically *Yes-underpricing after an up-move*, not a generic "trade the
   spike" effect. The asymmetry is the signal.

So: **the exit policy is the entire trade.** The same detection, held to resolution,
makes money; chased for drift, loses it.

### Caveats — why this is a research artifact, not a deployable edge

- **Thin and lumpy.** ~100 trades, trade-Sharpe ≈ 0.1, binary payoffs; trades cluster
  in time (the 2024-election complex dominates the universe), so the effective
  sample is far smaller than n suggests. No significance is claimed — read the sign,
  not the t-stat.
- **Capital lockup.** Winners hold ~86 days to settlement; the per-trade PnL is *not*
  time-annualised, and you can't freely redeploy locked collateral.
- **Survivorship / universe tilt + level confound** carry over from the vol→outcome
  study (top-volume resolved markets; `upside_vol` partly proxies "alive market").
- **Settlement assumption.** Resolution PnL settles at the Gamma `outcome` label, not
  a tradable exit; real fills, depth, and No-leg pricing are idealised.

Next steps: price-band-matched entries to kill the level confound, a downside-spike /
buy-No symmetric test, capital-aware (annualised, collateral-adjusted) sizing, and
joining live order-book imbalance (`fetch.order_book_features`) to flag which spikes
will revert hardest.
