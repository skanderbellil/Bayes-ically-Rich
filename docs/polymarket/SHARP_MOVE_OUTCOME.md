# Sharp upside moves vs. the actual outcome

> Does a *sharp upside volatility return* in a Polymarket probability predict the
> market's actual resolution — and does the price keep going, or revert?

An event study that cross-fertilises the repo's volatility / change-point
machinery (`research.regimes.BOCPD`) onto prediction markets. Run it with:

```bash
python experiments/run_polymarket_vol_outcome.py --refresh        # pull live labels
python experiments/run_polymarket_vol_outcome.py --by sharp_up    # other metrics
```

## Method

For every sampled `(market, day)` we measure — **causally**, from prices up to
that day — how sharp and upside-tilted the recent log-odds churn has been, then
bucket episodes into quantile brackets and read off three things the brackets
can't fake:

- **actual Yes-resolution rate** of the markets in the bracket;
- **calibration residual** = `yes_rate − mean price` — predictive content *beyond*
  the price level (a market at 0.9 resolves Yes ~90% regardless of vol);
- **forward log-odds drift** over `fwd_horizon` days — does a sharp up-move
  continue (momentum) or revert (overreaction)?

Bracketing metrics (`--by`): `upside_vol` (RMS of positive log-odds moves),
`sharp_up` (largest single-day positive jump), `vol_skew` (upside − downside
semivol), `bocpd_cp` (BOCPD changepoint probability). Resolution labels come from
Gamma's settled `outcomePrices` (`["1","0"]`=Yes); features from CLOB mid-price
history. **Base Yes-rate of the universe is ~16%** — a stark favorite–longshot
base rate (most "Will X happen?" markets resolve No).

## Result (250 top resolved markets, 11,262 episodes, 10d window/horizon)

| upside_vol bracket | n | mean price | actual Yes-rate | calib resid | fwd 10d Δlog-odds |
|---|---:|---:|---:|---:|---:|
| Q1 (no upside vol) | 2253 | 0.01 | 2.0% | +0.6% | +0.008 |
| Q2 | 2252 | 0.01 | 1.1% | −0.0% | +0.003 |
| Q3 | 2252 | 0.01 | 4.2% | +2.8% | +0.007 |
| Q4 | 2252 | 0.20 | 25.4% | +5.0% | +0.002 |
| **Q5 (sharp upside)** | 2253 | 0.26 | **29.4%** | **+3.6%** | **−0.027** |

Two complementary reads — and they point opposite ways, which is the whole point:

1. **For the eventual outcome, a sharp upside move is mildly *informative*.** In
   every bracket the actual Yes-rate sits *above* the price (positive calibration
   residual): these markets slightly **underprice** Yes, most so in the
   high-upside-vol buckets (+3.6% in Q5, +5.0% in Q4). The up-move carried real
   information about where the market settles.
2. **For the short-term path, the sharpest upside spikes *overreact*.** Forward
   10-day drift is positive in Q1–Q4 but flips sharply **negative** in Q5
   (−0.027 log-odds): right after the biggest upside spikes, prices tend to
   **revert**. `sharp_up` shows the same pattern (Q5 fwd Δz −0.021).

This is consistent with the cross-market momentum study (`POLYMARKET_MOMENTUM.md`):
at short horizons the cross-section **mean-reverts**. A sharp upside move tells you
something true about the destination, but the market gets there by overshooting
first.

### Caveats

- **Level confound.** Upside vol is near-zero for the many markets parked at
  0.01, so the bracketing is partly a proxy for "this market is alive/contested"
  (note price climbing across brackets). The **forward-drift** column is the
  level-robust signal; the calibration residual is suggestive but not
  price-matched. A cleaner test would bucket *within* price bands.
- **Overlapping episodes.** Days within one market share a resolution label and
  overlapping forward windows, so effective sample size ≪ 11k; treat magnitudes,
  not significance.
- **Mid-price only.** Features use the CLOB mid; spread/slippage are ignored, and
  Polymarket exposes no *historical* order book (only live depth, wired up in
  `fetch.order_book_features` for forward live-signal work).
- **Survivorship / universe tilt.** Top-volume resolved markets, dominated by the
  2024 election complex — not a representative cross-section.

Research artifact, not a deployable edge. Natural next steps: price-band-matched
brackets, downside-move symmetry, and joining live order-book imbalance to the
forward-reversion signal.
