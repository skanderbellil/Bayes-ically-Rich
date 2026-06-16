# Follow the smart money? Trader momentum on Polymarket

> Short answer: **no.** Ranking Polymarket wallets by the PnL they have realised
> *so far* and mirroring the recent winners' positions is, net of cost, the
> **worst** of the four books — flat-to-dead. Following the recent *losers*
> (`anti_smart`) is never worse and usually better. The cross-section of trader
> flow **mean-reverts**: "smart money" momentum is the favorite–longshot bias
> wearing a different hat, the same finding the rest of this thread keeps hitting.
> And the market-maker screen turns out **inert** here — MMs never rank top by
> marked PnL on this universe, so excluding them changes nothing.

```bash
python experiments/run_polymarket_smart_money.py            # cached panel + 4 books
python experiments/run_polymarket_smart_money.py --refresh  # re-pull live trader data
python experiments/run_polymarket_smart_money.py --n-follow 10 --rebalance 5
```

## The idea, and the one trap that kills naïve versions

A Polymarket leaderboard publishes the top wallets by realised PnL. The obvious
trade: find the recent winners, screen out the market makers, and **copy their
positions** — trader momentum. The trap is that a leaderboard is computed *as of
now*, so "recent winners" are people you already know turned out to win. Replay
their past trades and the backtest looks brilliant for entirely circular reasons.

So the whole study is built point-in-time:

- the end-of-sample leaderboard only seeds **which wallets exist** in the
  candidate pool (a documented survivorship caveat — see *Limitations*);
- **who we follow on date *t*** is chosen by the PnL each wallet had realised
  *strictly before t*, reconstructed from its own fills marked against the CLOB
  price panel (`trader_equity_curve`) — never the current leaderboard;
- **how we follow** is *mirror-and-exit*: the cohort's target in each outcome
  token is the selected leaders' aggregate net **dollar** inventory, normalised
  to $1 gross, held while they hold and decaying to zero as they exit, marked to
  the CLOB mid daily with a turnover cost.

Four books make the claim falsifiable — `smart_money` (top-N non-MM by
point-in-time PnL, the hypothesis), `all_leaders` (every non-MM wallet — does
*selection* add anything?), `anti_smart` (bottom-N — the negative control), and
`with_mm` (top-N *including* market makers — does the screen do work?).

## Result: selection is inverse-predictive

98 wallets pooled from the 7d / 30d / all-time profit leaderboards, 48 priced
outcome tokens (the cohort's most-traded markets, 2024-01 → 2026-06, election-
dominated), weekly cohort re-selection, 45-day burn-in, 50 bps/turn, $1 gross:

| book | gross Sharpe | net Sharpe | net PnL | net maxDD | avg #followed |
|------|-------------:|-----------:|--------:|----------:|--------------:|
| `smart_money` (top-20 winners) | 0.29 | **0.01** | +0.01 | −0.57 | 9.4 |
| `all_leaders` (everyone) | 0.46 | 0.23 | +0.32 | −0.58 | 9.5 |
| `anti_smart` (bottom-20 losers) | 0.54 | **0.33** | +0.49 | −0.58 | 9.4 |
| `with_mm` (top-20 incl. MMs) | 0.29 | 0.01 | +0.01 | −0.57 | 9.4 |

Two things jump out, and both survive a cohort-size sweep (`--n-follow`):

1. **Following winners is the worst book; following losers the best.** Across
   `--n-follow ∈ {5, 10, 20, 30}`, `anti_smart`'s net Sharpe is **never below**
   `smart_money`'s (0.44 vs −0.05 at 5; a 0.39 tie at 10; 0.33 vs 0.01 at 20;
   0.30 vs 0.17 at 30). The winner-following book is never the best in any
   configuration. The cross-section of trader flow **reverts**: leaders climb the
   board by holding pumped favorites near resolution (capped upside, prone to
   revert), so buying *after* them is buying high.

2. **The MM screen is inert here.** `with_mm` is byte-identical to `smart_money`
   in every configuration — the wallets the fingerprint flags as market makers
   (11/98) never rank in the top-N by *marked PnL on this universe*, so removing
   them changes no cohort. The screen is built and correct (it flags heavy
   two-sided, high-turnover, thin-edge churn), it simply has nothing to bite on
   once you rank by directional PnL.

The effect is **weak and config-dependent** — at `--n-follow 10` the winner and
loser books tie at 0.39, and the small-cohort books just hold fewer names (lower
turnover, smaller drawdown) rather than smarter ones. This is a directional
*sign*, consistent everywhere, not a stable magnitude.

## Why this is the favorite–longshot bias again, not a new edge

The whole thread keeps landing on the same place: prediction-market probabilities
**mean-revert** at the weekly horizon (`POLYMARKET_MOMENTUM`), longshots are
over-priced and favorites under-bet (`STRATEGY_SYNTHESIS`, `FIELD_SHAPE`,
`CROSS_OUTCOME`). Trader momentum is just that result viewed through the order
flow: the wallets *currently* winning are disproportionately long favorites that
have already run, so mirroring them is a momentum bet on names that revert. The
contrarian leg wins for the same reason the contrarian *price* leg wins.

## Limitations

- **Pool survivorship.** The candidate pool is seeded by the *current*
  leaderboard, so it is enriched for wallets that eventually did well. Selection
  *timing* is clean (point-in-time PnL), but a wallet that blew up and vanished
  is not in the pool at all. The honest scope is "among persistent leaderboard
  names, does trailing PnL rank predict forward flow quality?" — and the answer
  is no/inverse.
- **Universe coverage.** Only 48 of the cohort's top-100 tokens have enough daily
  CLOB history to price; the set is dominated by the 2024 US-election complex —
  the same concentration that caveats `POLYMARKET_MOMENTUM`. Not a representative
  cross-section.
- **Marking, not fills.** PnL marks to the CLOB mid; it ignores spread, depth and
  the fact that copying a whale's fill at their price is often impossible. Real
  copy-trading frictions are *worse* than the 50 bps charged here.
- **In-sample.** Cohort size, rebalance and burn-in are not tuned out-of-sample;
  the magnitudes should be read as a sign, not a backtest to deploy.
