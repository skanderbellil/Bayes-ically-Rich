# Does the smart-flow *wallet selection* premise hold?

A falsification test of the assumption underneath the smart-flow consensus
signal (`SMART_FLOW_PAPER`, `smartflow_papertrade.py`): that **recent
profit-leaderboard wallets carry forward edge** — i.e. *trader momentum*. The
paper tracker follows those wallets' consensus buys; this study asks the prior
question those entries take for granted.

Reproduce:

```bash
python experiments/run_smart_money_wallet_test.py            # n_follow=20
python experiments/run_smart_money_wallet_test.py --n-follow 5
```

Snapshot artifacts (git-tracked, in `data/polymarket_smart_money_snapshot/`):
`wallet_test_summary_n{5,20}.csv`, `wallet_test_equity_n*.csv`,
`wallet_test_cohorts_n*.json`, `wallet_test_meta_n*.json`, and
`raw_cache_2026-06-23.tar.gz` (the exact 58 MB leaderboard/fills/CLOB cache the
run was built on, compressed to ~4 MB — point-in-time, so it can't be re-pulled).

## What "wallets" means here

The pool is **not hand-picked**. `candidate_universe()` pulls the Polymarket
profit leaderboard for the 7d/30d/all windows (top 60 each), de-dupes to 121
wallets, then a behavioural MM screen (`mm_fingerprint`) flags 12 as
market-maker-like. The four falsifiable books differ *only* in which wallets
they follow on each rebalance date (selection is strictly point-in-time on PnL
reconstructed from each wallet's own fills — no lookahead):

| book | cohort | tests |
|------|--------|-------|
| `smart_money` | top-N non-MM by point-in-time PnL | the hypothesis |
| `all_leaders` | every non-MM, inventory-weighted | does *selection* help? |
| `anti_smart`  | bottom-N non-MM | negative control (follow the losers) |
| `with_mm`     | top-N including MMs | does the MM screen do work? |

Run: 121 wallets (12 MM-flagged), 82-token universe, 901 days
(2024-01-05 → 2026-06-23), $1 gross, rebalance 7d, 0.5%/turn cost, burn-in 45d.

## Result

**n_follow = 20** (cohorts overlap 97% — *underpowered*, see below):

| book | net Sharpe | net PnL | net maxDD | avg #followed |
|------|-----------:|--------:|----------:|--------------:|
| smart_money | 0.12 | 0.15 | −0.55 | 9.4 |
| all_leaders | 0.19 | 0.26 | −0.55 | 9.8 |
| anti_smart  | 0.19 | 0.26 | −0.55 | 9.4 |
| with_mm     | 0.12 | 0.15 | −0.55 | 9.4 |

**n_follow = 5** (cohorts disjoint, Jaccard 0.02 — *discriminating*):

| book | net Sharpe | net PnL | net maxDD | avg #followed |
|------|-----------:|--------:|----------:|--------------:|
| smart_money | **−0.09** | **−0.02** | −0.13 | 3.4 |
| all_leaders |  0.19 |  0.26 | −0.55 | 9.8 |
| anti_smart  | **+0.29** | **+0.50** | −0.68 | 3.4 |
| with_mm     | −0.09 | −0.02 | −0.13 | 3.4 |

## Reading it

1. **The wallet-selection premise does not hold in this sample — if anything it
   inverts.** Once the cohorts are actually separated (n=5), following the top-PnL
   "smart" wallets *loses* money net of cost (Sharpe −0.09) while following the
   recent *losers* is the single best book (+0.29 Sharpe, +0.50 PnL). This is
   trader-performance **reversal**, the opposite of the momentum the signal
   assumes. `all_leaders` (no selection) beats `smart_money` at every cohort size.

2. **The MM screen does no work here.** `with_mm` is *identical* to `smart_money`
   at both sizes — the top-PnL cohort contains no MM-flagged wallets, so including
   them changes nothing. The screen isn't wrong, it's just inert on this pool.

3. **n_follow=20 is uninformative, not confirmatory.** Only ~29 wallets are ever
   rankable and ~10 are active per rebalance, so "top-20" and "bottom-20" select
   almost the same ~10 wallets (97% Jaccard, identical −0.55 maxDD). The four
   books collapse onto one portfolio. *This is why the live paper tracker, which
   leans on the same pool, can't be assumed to have selection edge.*

## Honest caveats

- **Pool survivorship.** Every candidate is on the *current* leaderboard, so all
  are eventual winners. `anti_smart` is therefore "recently-cold but ultimately
  good" wallets — its edge may be buy-the-dip on proven traders, not a generic
  "follow losers" rule. The reversal is real *within the winner pool*; it does
  not license shorting random losing wallets.
- **One point-in-time snapshot** (2026-06-23), 82 tokens, tiny n=5 cohorts (high
  variance — note anti's −0.68 maxDD). Directional, not a sized claim.
- **This is the daily inventory-follow book, not the exact live signal.** The
  paper tracker enters on ≥3 distinct smart buyers and holds to resolution; this
  mirrors aggregate net dollar inventory daily. It tests the *premise* (winners'
  flow has edge), not the precise consensus-breadth entry rule.

## So — does the idea hold?

On this evidence, **no**: selecting recent leaderboard winners shows no edge over
following everyone, and underperforms following the pool's recent losers. Before
trusting the live smart-flow consensus tracker, the next tests worth running are
(a) a **much deeper, more separable pool** (larger `per_window`, wider token
universe) so selection can be measured rather than collapsing, and (b) testing
the **exact ≥3-buyer hold-to-resolution signal** directly rather than the
inventory-follow proxy.
