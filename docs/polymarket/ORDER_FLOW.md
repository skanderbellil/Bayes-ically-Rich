# Smart-crowd order flow — breadth beats imbalance, and survives cost

> Forget *who* is trading: does the candidate pool's **aggregate flow** predict
> the cross-section of next-day moves? Yes, and cleanly. Traded **dollar-neutral**
> (so the favorite-drift beta is removed), a long-top / short-bottom book on
> **breadth** — the count of distinct wallets net-buying a token — runs gross
> Sharpe **1.1** and, with a 14-day formation window refreshed daily, nets **0.45**
> after 50 bps. That makes it the *first* beta-neutral Polymarket signal in this
> thread to clear cost with margin (cross-market price momentum did not). Two
> honest qualifiers: the predictiveness lives almost entirely in the **top
> quintile** (a high-consensus long tilt, not a symmetric long/short), and it is
> **fast-decaying** — slow the rebalance and the gross signal collapses, so
> turnover is permanently the binding constraint.

```bash
python experiments/run_polymarket_flow.py
python experiments/run_polymarket_flow.py --lookback 14
python experiments/run_polymarket_flow.py --refresh
```

## Two signals, one beta to kill

From the (non-MM) pool's fills over a trailing window we build, per token-day:

- **breadth** — distinct wallets net-buying minus distinct net-selling. A
  *consensus* signal: many independent traders agreeing is information a single
  whale's size cannot manufacture.
- **imbalance** (OFI) — net signed dollars (buys − sells). Classic order-flow
  pressure.

The universe has a pervasive **favorite-drift beta** (a naive long-all-tokens
book is already ~0.5 Sharpe — see `SPECIALISTS`), so a raw long book would just
re-earn that. To isolate the *signal*, the engine z-scores each signal across the
live tokens every day and goes **long the top 20% / short the bottom 20%, equal
dollars** — the common drift cancels, leaving only ranking power.

## Result (177 wallets → 238 priced tokens, dollar-neutral, 50 bps/turn)

| book | gross Sharpe | net Sharpe | net PnL | net maxDD |
|------|-------------:|-----------:|--------:|----------:|
| `breadth` (7d) | 1.07 | 0.20 | +0.10 | −0.21 |
| `imbalance` (7d) | 0.98 | 0.12 | +0.07 | −0.20 |
| `breadth_fade` | −1.07 | −1.87 | −1.00 | −1.00 |
| `imbalance_fade` | −0.98 | −1.79 | −1.06 | −1.08 |

The fade legs are the exact negatives of the follow legs (gross), the
consistency check that the sign is real: **following** the flow is right,
**fading** it is wrong. Breadth edges imbalance everywhere — *consensus among
distinct wallets beats raw dollar pressure*, exactly the prior the first probe
suggested.

**Calibration — where the signal lives.** Mean next-day Δp by signal quintile
(pooled, Q1 = lowest → Q5 = highest):

| signal | Q1 | Q2 | Q3 | Q4 | Q5 |
|--------|---:|---:|---:|---:|---:|
| breadth   | +0.0018 | +0.0003 | +0.0003 | +0.0007 | **+0.0099** |
| imbalance | +0.0011 | −0.0001 | +0.0011 | +0.0026 | **+0.0078** |

The effect is **top-quintile-loaded**: the most-bought-by-consensus tokens drift
+0.99¢ the next day, 5–10× the flat middle. So the book is really a
high-consensus *long tilt*; the short leg mostly buys dollar-neutrality, not
negative drift.

## It's fast alpha — turnover is the binding constraint

Slowing the rebalance does not save the net Sharpe, it **destroys the gross
signal** — the predictive content decays within a day or two:

| config | gross | net | turnover |
|--------|------:|----:|---------:|
| breadth 7d, daily | 1.07 | 0.20 | 0.101 |
| breadth 7d, every 5d | 0.28 | −0.10 | 0.041 |
| breadth 7d, every 10d | −0.08 | −0.51 | 0.031 |
| **breadth 14d, daily** | **1.29** | **0.45** | 0.098 |
| breadth 14d, every 5d | 0.24 | −0.13 | 0.041 |

So you must rebalance daily to capture it, which costs ~10 bps/day of turnover.
The one configuration that clears cost comfortably is a **longer (14-day)
formation window refreshed daily**: the smoother consensus measure is more
persistent (gross 1.29) while turnover stays put, netting **0.45**.

## Verdict

A genuine, beta-neutral cross-sectional signal — smart-crowd consensus (breadth)
predicts next-day prediction-market moves, follow-not-fade, concentrated in the
high-consensus tail. Net of realistic cost it is **marginal but positive** (0.20
at 7d, 0.45 at 14d), where the price-momentum book was negative — the order-flow
view adds something the price view didn't. Still in-sample, marked-to-mid, and a
long tilt more than a true long/short, so it's a research signal, not a deployable
strategy.

## Limitations

- **Top-quintile / long tilt.** The short leg adds little; honestly this is "high-
  consensus names outperform," dressed as long/short for beta-neutrality.
- **Fast decay + cost.** Daily rebalance is mandatory; at real spread-dominated
  Polymarket costs (> 50 bps on thin books) the net edge would compress further.
- **Pool & priced-universe selection.** Same caveats as `SMART_MONEY` /
  `SPECIALISTS`: leaderboard-seeded pool, tokens need ≥10 daily price points.
- **In-sample.** Lookback / holding / top-frac are not tuned out-of-sample; read
  the magnitudes as a sign.
