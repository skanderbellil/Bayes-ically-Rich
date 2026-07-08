# Independence-classified smart-flow consensus — pre-registered forward experiment

**Registered:** 2026-07-08, before the first data point. Everything in the
"Frozen thresholds" and "Kill criteria" sections below is fixed *before* this
ledger sees a single resolved position. Any retuning after data arrives is a
different, new experiment — it gets a new ledger, not a rewrite of this one.

```bash
python experiments/run_smart_flow_indep_update.py                # hourly update
python experiments/run_smart_flow_indep_update.py --dry-run      # scan only, no write
```

## Motivation

The incumbent smart-flow consensus strategy (`SMART_FLOW_PAPER.md`) longs any
open outcome token that **>= N distinct non-market-maker leaderboard wallets**
bought recently. Its forward record is bad: **-52% MTM**, edge **t~+0.09** over
822 markets seen. That is not "no edge" noise — counting co-buyers is a
plausible-sounding rule that keeps losing money, which means the mechanism it's
built on is probably wrong, not just under-tuned.

`docs/knowledge/epistemology/README.md` (Lesson 2, social epistemology) names
the exact failure mode: **Condorcet's jury theorem only delivers "wisdom of the
crowd" when the voters are independent.** A crowd of a million has the epistemic
weight of the three people who thought first the moment everyone after them is
just imitating, not re-deriving — an information cascade. "Three wallets bought
this token" is ambiguous between two very different underlying processes:

- **Independent discovery**: three unrelated traders separately notice the same
  mispricing from separate evidence/models, and buy at separate times. This is
  the Condorcet-informative case — real signal.
- **Information cascade**: one wallet buys, others see the tape move (or copy
  its known positions) and pile on. This adds *zero* independent evidence no
  matter how many wallets do it — cascades can look identical to consensus in
  a raw buyer count.

The incumbent's buyer-count rule cannot tell these apart. This experiment adds
a classifier that tries to, and tests whether the split explains the incumbent's
losses: if the -52%/t~0.09 record is actually an *average* of a good
independent-discovery edge and a bad cascade edge, splitting them should reveal
both.

## Hypothesis

> Consensus classified **independent** has positive edge net of the real
> bid-ask spread (entering at the live ask, exactly like the incumbent).
> Consensus classified **cascade** has zero or negative edge.

The claim being tested is the **difference** between the two classes, not the
absolute level of either one in isolation — a discriminator that doesn't
discriminate is worthless even if both classes happen to be flat.

Mechanically, both classes are scanned by the *same* consensus gate the
incumbent uses (`>= min_buyers` non-MM wallets buying in the trailing window),
entered at the *same* live ask, sized at the *same* flat `bet_fraction` (no
Kelly, no conviction scaling), and held/exited with the *same* rules
(hold-to-resolution + optional consensus-exit on flow reversal). The only thing
that differs between rows is the `indep_class` label — so any PnL difference
between classes is attributable to the classifier, not to a confound in how the
two groups are traded.

## Frozen component tests (pre-registered, not tunable after data arrives)

Three cheap proxies, computed from the same per-wallet trade histories already
fetched to build the flow index (no extra API calls):

| component | feature | passes (independent-leaning) if |
|---|---|---|
| temporal | `first_buy_span_hours` — hours between the earliest and latest buyer's *first* buy of the token in the trailing window | `>= 24.0` |
| price | `price_chase` — price paid by the chronologically *last* first-buyer minus price paid by the *earliest* first-buyer | `<= 0.05` |
| co-movement | `pairwise_jaccard` — mean pairwise Jaccard similarity of the buyers' full traded-history `conditionId` sets | `<= 0.20` |

`indep_score` = count of the three passed (0-3). `indep_class` = **independent**
if `indep_score >= 2`, else **cascade** (2-of-3 rule).

These three thresholds (24h, 0.05, 0.20) and the 2-of-3 rule are **value
choices, frozen now**. They will not be nudged after seeing which side of the
line makes the ledger look better — that would just be overfitting the
classifier to its own forward record. If the frozen thresholds turn out to be
poorly calibrated, the honest move is a *new* experiment with a *new* ledger,
not editing these numbers in place.

## Kill criteria (frozen)

**Primary — the discriminator claim itself.** Once **>= 40 resolved positions
exist in EACH class** (independent and cascade), compute the one-sided
two-sample t-statistic of per-position PnL, **independent minus cascade**. If
`t < 1.0`, the discriminator does not survive contact with data: retire it and
freeze this ledger (stop opening new positions; let existing ones resolve for
the record, but do not restart the experiment on the same file).

**Secondary — is it even tradeable?** If the independent class itself has mean
PnL `<= 0` after `>= 40` resolved positions, the strategy is not tradeable
regardless of whether it beats the cascade class — a "less bad than cascade"
edge that is still non-positive is not a strategy.

## Kill criteria for the incumbent smart-flow strategies (also now pre-registered)

The incumbent Smart Flow and Smart Flow (ROI) forward ledgers (`SMART_FLOW_PAPER.md`)
did not have a pre-registered kill rule before now. Backfilling one: each is
retired once **>= 50 positions have resolved** in its own ledger, if its edge
t-statistic is **`< 0`** at that point. (Smart Flow is already at t~+0.09 over
822 markets seen — close to the line; this makes the retirement rule explicit
rather than a judgment call made after the fact.)

## Ledger

`data/paper_trade/smart_flow_indep_positions.csv` — same idempotent-additive
CSV convention as every other ledger in this repo (new tokens appended,
resolved ones marked closed, nothing deleted). Columns are the incumbent Smart
Flow schema plus, inserted after `n_smart_buyers`: `first_buy_span_hours`,
`price_chase`, `pairwise_jaccard`, `indep_score`, `indep_class`. Runs hourly in
the same Actions cron (`.github/workflows/paper_trade.yml`) as every other
strategy, `continue-on-error` like its siblings so a transient failure here
doesn't block the other ledgers.

## Reading the ledger

Open `data/paper_trade/smart_flow_indep_positions.csv` on GitHub, or run
`python experiments/run_smart_flow_indep_update.py` for a summary table split
by `indep_class` (count, open, resolved, win rate, mean PnL per class) — that
split is the entire point of this file. The dashboard (`generate_dashboard.py`)
also carries three views: the full ledger ("Smart Flow (indep exp)") and two
filtered views ("SF independent" / "SF cascade") re-slicing the same ledger by
`indep_class`, consistent with the existing `Smart Flow ∩ band` derived-view
pattern (`MIDPRICE_SMARTFLOW_COMBO.md`).

## Retrospective replay (exploratory — 2026-06-23 cache)

**Run 2026-07-08.** Before the forward ledger above has a single resolved
position, `experiments/run_smartflow_indep_backtest.py` replays the same
consensus + classification mechanics offline against the frozen wallet-fill
cache (`data/polymarket_smart_money_snapshot/raw_cache_2026-06-23.tar.gz`: 126
wallets, 224,927 fills, right-truncated at 2026-06-23). It walks day by day,
fires one consensus event per token the first day >= 3 *eligible* wallets
(cached history starting before `t - 7d`, to avoid left-truncation faking a
"late arrival") have bought it in the trailing 7-day window, classifies each
event by calling the shipped `independence_features` / `classify_independence`
verbatim (no reimplementation; a wallet's frame is filtered to `timestamp <=
t` before the call, so there is no look-ahead), and resolves outcomes via
`fetch_market_resolution` with the same 0.05–0.90 entry-price band the live
scanner uses. Full script + method notes: `experiments/run_smartflow_indep_backtest.py`.

Event-level output: `data/polymarket_smart_money_snapshot/smartflow_indep_backtest_events.csv`.
Resolution cache: `data/polymarket_smart_money_snapshot/smartflow_indep_resolution_cache.csv`.

**Headline numbers** — 102 consensus events fired (95 within the 0.05–0.90
entry-price band, 7 excluded); 89 of the in-band events resolved, 6
unresolved/unfetchable (attrition, mostly still-open Iran-deal and World Cup
markets as of the 2026-06-23 cache cutoff):

| class | n resolved | win rate | mean pnl | std pnl |
|---|---|---|---|---|
| independent | 73 | 60.3% | +0.6301 | 2.0838 |
| cascade | 16 | 50.0% | -0.0843 | 1.0352 |

Welch t-stat (independent − cascade, two-sided): **t = +2.009, p = 0.050**
(df ≈ 46). This is the discriminator's own claim (one-sided in the
pre-registration): independent-classified consensus outperforms
cascade-classified consensus in this sample, right at conventional
significance with a small cascade class (n=16).

Per-component diagnostics (resolved events, pass vs. fail on each proxy
alone, pooled across both classes):

| component | pass n / mean pnl | fail n / mean pnl |
|---|---|---|
| temporal (span ≥ 24h) | 22 / +1.1157 | 67 / +0.3000 |
| price (chase ≤ 0.05) | 67 / +0.7598 | 22 / -0.2844 |
| co-movement (jaccard ≤ 0.20) | 84 / +0.4242 | 5 / +1.8027 |

`indep_score` distribution across all 102 fired events (in-band + out-of-band):
0 → 1 event, 1 → 22, 2 → 65, 3 → 14.

The price-chase component alone shows the cleanest split of the three; the
co-movement component's "fail" bucket is too thin (n=5) to read anything into.

### Mandatory caveats — read before treating this as evidence

**(a) Look-ahead / survivorship in the wallet pool.** The 126 wallets are the
2026-06-23 leaderboard-derived snapshot — selecting *these* wallets already
uses information (that they existed, traded, and were worth caching) from
*after* the replayed trades. This inflates the ABSOLUTE edge for **both**
classes. The only number this replay can honestly speak to is the
**BETWEEN-CLASS separation** (independent vs. cascade) — and that separation
shares the same bias, since both classes are drawn from the identical
survivorship-affected pool. Do not read either class's absolute mean pnl as a
forward-tradeable edge.

**(b) No bid-ask spread.** Entries use the fill price of the chronologically
last first-buyer as an ask proxy — there is no historical order book in this
replay, so the real bid-ask spread the live ledger captures (`entry_ask` vs.
`entry_mid`) is entirely absent here. Live entries pay the ask; this replay
does not.

**(c) Per-wallet history truncation.** Coverage is right-truncated at
2026-06-23 and left-truncated per wallet (only 76/126 wallets reach back to
2026-06-16, 35/126 to 2026-04-23), which limits how far back — and how densely
— consensus events could be detected, especially before mid-2026.

**(d) Exploratory evidence only.** This replay is a fast, biased read, not a
substitute for the pre-registered forward ledger above, which remains the
deciding test (kill criteria: >= 40 resolved positions per class, t < 1.0
retires the discriminator). The frozen component thresholds (24h / 0.05 / 0.20,
2-of-3 rule) were **not** retuned after seeing these numbers — they were
frozen before this replay ran, per the pre-registration, and a retuned
threshold would be a new experiment on a new ledger, not an edit to this one.
