# Follow traders in the domain they're good at

> The plain smart-money follow failed — but it failed for a fixable reason. On a
> **large, diverse universe** (177 wallets × 238 priced tokens, vs the 48-token
> election sliver before), conditioning skill on **topic** rescues it: following
> the per-domain specialists earns net Sharpe **0.99** (PnL +2.25, maxDD −0.38)
> and beats the domain-**blind** ranking (0.84) on both Sharpe and drawdown. The
> winner-beats-loser ordering, *inverted* on the tiny universe, is **restored**
> at scale (specialist 0.99 > anti 0.74). Two honest asterisks: most of the level
> is following the smart crowd at all — `all_leaders` ties the specialist (0.99) —
> and a chunk is a long-favorite-drift **beta** (a naive long-all-tokens book is
> already 0.51), so domain-conditioning is a *risk-and-selection* refinement, not
> a new source of alpha.

```bash
python experiments/run_polymarket_specialists.py
python experiments/run_polymarket_specialists.py --refresh
python experiments/run_polymarket_specialists.py --n-per-domain 3
```

## Why condition on domain

`SMART_MONEY` ranked each wallet by *total* PnL and found following the winners
lost to following the losers. But the efficiency and field-shape studies already
showed forecasting skill is **domain-specific** — a wallet can be a genuine
politics forecaster and a coin-flipper in sports. Ranking it globally blends the
two, so the "top" cohort is a muddle and the selection carries no real skill.

The fix: tag every outcome token by topic (`behavior.token_domains`, reusing the
keyword classifier on the market title), reconstruct each wallet's **point-in-time
PnL *within each domain*** (`behavior.trader_token_pnl`, marked PnL attributed
per token then summed over a domain), and at each weekly rebalance follow the
per-domain top-N specialists **only on that domain's markets**. Same engine
guarantees as before: selection uses only PnL realised before *t*, positions are
mirror-and-exit (hold proportional to the cohort's net dollar inventory, decay out
as they exit), marked to the CLOB mid on $1 gross, costs on turnover.

## Result (177 wallets × 238 tokens, 2024-01 → 2026-06, 50 bps/turn)

Universe domain mix: other 157 · geopolitics 40 · politics 19 · sports 17 ·
macro 4 · crypto 1.

| book | gross Sharpe | net Sharpe | net PnL | net maxDD | avg #followed |
|------|-------------:|-----------:|--------:|----------:|--------------:|
| `specialist` (top-5 / domain) | 1.23 | **0.99** | +2.25 | **−0.38** | 5.6 |
| `global` (top-K by *total* PnL) | 1.11 | 0.84 | +1.82 | −0.52 | 11.1 |
| `anti_special` (bottom-5 / domain) | 0.96 | 0.74 | +1.92 | −0.70 | 5.2 |
| `all_leaders` (every non-MM wallet) | 1.26 | 0.99 | +2.13 | −0.37 | 11.9 |
| *long-all-tokens beta (reference)* | — | *0.51* | *+0.58* | — | — |

Four reads, in order of how much they survive scrutiny:

1. **Domain-conditioning beats domain-blind selection.** `specialist` (0.99,
   DD −0.38) dominates `global` (0.84, DD −0.52) — same idea, skill ranked within
   topic instead of pooled — with a *tighter* book (5.6 names vs 11.1). Ranking a
   wallet globally picks the wrong specialists; conditioning on domain fixes it
   and cuts the drawdown by a quarter.

2. **The inversion was a small-sample artefact.** On the 48-token election sliver,
   following losers beat following winners. On the diverse universe the natural
   order returns: `specialist` (0.99) > `anti_special` (0.74), and the
   loser-following book carries the worst drawdown (−0.70). In-domain winner flow
   is mildly predictive after all.

3. **Most of the level is "follow the smart crowd," not fine selection.**
   `all_leaders` — just mirror *every* non-MM wallet's aggregate inventory — ties
   the specialist at 0.99 (and the specialist only pulls ahead, 1.09 vs 0.99, when
   the per-domain cohort is widened to 10). So the edge over the `global` book is
   mostly about *avoiding bad concentration*, not surgically picking the best five.

4. **A real slice is favorite-drift beta.** A naive equal-weight long of every
   priced token is already 0.51 Sharpe: the priced universe is enriched for
   favorites that drifted to resolution. The follow books clear it comfortably
   (0.74–0.99 ≫ 0.51), so they are *not* just the beta — but they ride it, and the
   honest attribution is "smart-crowd selection on top of a favorite-drift tilt."

**Robustness (`--n-per-domain` sweep).** The ordering is stable: `specialist`
beats `global` at every cohort size (net Sharpe 0.99 vs 0.80 at 3, 0.99 vs 0.84
at 5, 1.09 vs 0.88 at 10) and beats `anti_special` at every size (0.99 vs 0.58,
0.99 vs 0.74, 1.09 vs 0.87) — and the winner-over-loser gap is *widest* under the
tightest selection (n=3), exactly what a real skill signal should do.

## What changed from `SMART_MONEY`, and the honest verdict

Same engine, same point-in-time discipline — only the **universe size/diversity**
and the **domain conditioning** differ, and that is enough to flip the conclusion
from "trader-following is inverse-predictive" to "trader-following works, modestly,
once you (a) give it a broad universe and (b) rank skill by topic." The deployable
claim is weak (it leans on a beta and ties a no-selection baseline), but the
research claim is clean: **smart-money flow is predictive when you stop diluting
skill across domains.**

## Limitations

- **Coarse taxonomy.** 157/238 tokens fall in `other` (the classifier only firmly
  tags politics/geopolitics/sports/macro/crypto), so "domain-conditioning" is a
  few real domains plus a large grab-bag. A finer event taxonomy would sharpen it.
- **Priced-universe selection.** Tokens need ≥10 daily price points, which skews
  toward longer-lived, eventually-resolved markets — the source of the +0.51
  favorite-drift beta. Not a representative cross-section of all flow.
- **Pool survivorship.** As in `SMART_MONEY`, the pool is seeded by the current
  leaderboard; selection *timing* is clean, but blown-up wallets are absent.
- **Marking, not fills.** Marks to the CLOB mid — ignores spread, depth, and the
  feasibility of copying a whale's fill. In-sample; magnitudes are a sign, not a
  backtest to deploy.
