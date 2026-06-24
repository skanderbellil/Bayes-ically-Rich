# Bankroll reality check — what a real $1,000 account would have done

The paper-trade ledgers report cumulative PnL as a **sum of fraction-of-bankroll
bets with no cash constraint**. When a signal wants 150 concurrent positions at
10% each, that headline silently assumes ~15× leverage no real account (and no
Polymarket account — there is no leverage) could deploy. `run_bankroll_sim.py`
replaces that with an honest cash simulation and three side-by-side sizing
conventions on the *same* trades.

```bash
python experiments/run_bankroll_sim.py --capital 1000 --stake 0.10
```

Writes `data/paper_trade/bankroll_summary.csv` (committed hourly by the cron) and
feeds the **"Real $1k acct"** line on each dashboard card.

## The three conventions

| convention | meaning |
|---|---|
| **additive (headline)** | Σ `stake · rᵢ` — sum of 10%-of-bankroll bets, unconstrained |
| **de-levered (100%)** | `mean rᵢ` — same trades, weights rescaled to sum to 100% (no leverage) |
| **real return** | $1,000 start, 10% of equity per trade, **a trade is skipped if there isn't free cash** (the actual capacity limit), payouts recycle into cash |

Identity: `additive = de-levered × (N · stake)`. So the headline is just the
de-levered return times the nominal summed exposure — when many trades overlap it
is a **leverage artifact, not extra revenue**.

## Result (snapshot)

| strategy | real $1k | additive | de-levered | peak leverage | edge t |
|---|---:|---:|---:|---:|---:|
| Smart Flow | **−39%** | +124% | +7% | 12.1× | +0.34 |
| Smart Flow (ROI) | −19% | −20% | — (n=2) | 0.1× | — |
| Mid-priced YES | **+63%** | +39% | +35% | 1.8× | +0.90 |
| Dip-Confirm YES | +54%¹ | +12% | — (n=2) | 0.8× | — |
| Macro | +1%¹ | — | — | — | — |

¹ mostly *unrealized* — few positions resolved yet.

## What it says

- **Smart Flow's profit was leverage, not edge.** +124% headline ⇒ +7% once
  de-levered to 100%, and **−39%** on a real $1k (192 of 212 signals were
  unfundable; the few longshot winners that carried the headline sit in the
  unfundable pile). Its edge t-stat is 0.34 — indistinguishable from fair pricing.
  Scaling does **not** preserve the revenue.
- **Mid-priced YES is the one that holds up.** additive +39% ≈ de-levered +35%
  because it was barely levered (1.8× peak) — its return is scale-robust. Real
  $1k account: +63%. (Edge t=0.90 — promising but not yet significant.)
- **Dip-Confirm / Macro** are too young to read (mostly unrealized).

## Caveats

- `max_dd` and the equity curve mark open positions at **cost basis** between
  entry and resolution (no daily path), so intra-life drawdown is understated.
- The real-return sim is path-dependent (first-affordable-first); it is one
  realistic account, not an ensemble average.
- Fills assume top-of-book at the recorded ask with no depth/slippage — fine for
  small size, optimistic at scale.
