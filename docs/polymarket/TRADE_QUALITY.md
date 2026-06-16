# What marks an informed fill — pay-up dominates

> Zooming from positions to individual fills (28,336 non-MM trades on the large
> universe), one behavioural tell towers over the rest: **aggressiveness**.
> Trades that *pay up* — cross the CLOB mid by >1¢ to get filled now — drift
> **+11.2¢** in their direction over the next 3 days (t≈36) and match the eventual
> resolution **83%** of the time, versus **−4.8¢ / 57%** for trades that sit below
> the mid. Betting **big** (size >1σ above the wallet's own norm, +3.0¢/74%) and
> **initiating** a position (first fill, +4.9¢/80%) are real, weaker tells.
> **Patience** is *not* a clean tell — long holders and flippers disagree across
> the two outcome measures. Baseline: +1.0¢ drift, 71% hit.

```bash
python experiments/run_polymarket_trade_quality.py
python experiments/run_polymarket_trade_quality.py --fwd 5
```

## The four tells

For every fill we record properties knowable *at the trade* and an outcome
knowable only *after* it — the forward 3-day directional drift (signed by side,
so a good BUY and a good SELL both score positive) and whether the side matched
the token's eventual 0/1 resolution.

### Aggressiveness — pay-up is the signal (and it's not subtle)

| bucket | n | fwd drift | t | resolution hit |
|--------|--:|----------:|--:|---------------:|
| paid up >1¢ through mid | 5,653 | **+0.1115** | 35.7 | **0.827** |
| at / inside mid | 14,058 | +0.0042 | 3.3 | 0.724 |
| below mid <−1¢ | 8,625 | **−0.0483** | −21.7 | **0.565** |

Monotonic and enormous. Urgency — paying through the book to get filled — is the
single strongest informed-trade tell in the data: those traders are right 83% of
the time; the ones taking the passive/cheap side are right only 57% (barely above
the 50/50 you'd get by accident on a binary market).

### Conviction — betting big

| bucket | n | fwd drift | t | hit |
|--------|--:|----------:|--:|----:|
| big bet (size z>1) | 4,524 | +0.0298 | 8.5 | 0.745 |
| normal (|z|≤1) | 21,002 | +0.0051 | 4.2 | 0.701 |
| small (z<−1) | 2,810 | +0.0112 | 2.3 | 0.725 |

A fill large relative to the wallet's *own* typical size drifts ~3× baseline and
is right more often. Conviction sizing carries information.

### Entry vs add — initiations are sharper

| bucket | n | fwd drift | t | hit |
|--------|--:|----------:|--:|----:|
| entry (first fill in market) | 214 | +0.0487 | 2.8 | 0.803 |
| add-on | 28,122 | +0.0093 | 8.0 | 0.711 |

Initiating a position beats averaging into one (80% vs 71% hit) — but the entry
count is small (the capped trade history rarely contains a wallet's true first
fill), so treat this as suggestive.

### Patience — no clean tell

| bucket | n | fwd drift | t | hit |
|--------|--:|----------:|--:|----:|
| patient (hold > 3d) | 12,215 | +0.0138 | 6.5 | 0.560 |
| flipper (hold ≤ 3d) | 16,121 | +0.0065 | 5.1 | 0.925 |

The two outcome measures *contradict* each other — patient wallets show more
short-horizon drift but a low resolution hit-rate, flippers the reverse (flippers
trade near-resolution favorites, inflating their hit-rate). No coherent
patience signal; drop it.

## So what's exploitable

The tells compose into a clean filter: **follow the aggressive, big, initiating
fills of the smart crowd** — exactly the trades that pay up with conviction. This
is the natural sharpening of the `ORDER_FLOW` breadth signal: weight consensus by
*urgency*, not just count. A pay-up-weighted follow book is the obvious next
build.

## The one honest caveat

Aggressive fills partly **move the mark themselves**: a large pay-up BUY can lift
the day's last price, so some of the +11¢ 3-day "drift" is the trade's own price
impact and short-horizon continuation, not pure foresight. That inflates the
*drift* magnitude. It does **not** explain the **resolution** gap, though — 83% vs
57% correct at settlement is a forecasting-accuracy difference, mechanically
unrelated to intraday impact. So the *direction and significance* are solid; read
the 11¢ as an upper bound on the tradable drift.

## Limitations

- **Impact endogeneity** on the drift metric (above) — the resolution hit-rate is
  the cleaner read.
- **Mid proxy.** "Premium to mid" uses the daily last price as the mid; a true
  best-bid/ask mid (no intraday history on the CLOB) would sharpen the split.
- **Pool / priced-universe selection.** Same caveats as `SMART_MONEY` /
  `SPECIALISTS`; 48 wallets × 67 tokens clear the price-band + forward-drift
  filter, 28k fills.
- **Descriptive, not a backtest.** These are conditional forward-drift / hit-rate
  tables, not a cost-charged book — the pay-up-weighted follow book is the test
  that would turn this into a strategy.
