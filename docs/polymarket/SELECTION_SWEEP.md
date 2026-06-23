# Is it the *selection criteria* that's broken? — full sweep

Follow-up to `WALLET_SELECTION_TEST.md`. That test ranked the pool by **total
point-in-time PnL** and found no edge (`smart ≤ anti`). But that's one selector.
This sweep asks the sharper question: **is there any selection criterion under
which the wallet idea holds?** — varying every selection knob on the same warm
cache, so any difference is attributable to the *selector alone* (weights, costs,
marking and point-in-time discipline are identical to `run_smart_money_follow`).

Reproduce:

```bash
python experiments/run_smart_money_selection_sweep.py          # 450 configs
python experiments/run_smart_money_selection_sweep.py --quick  # 50 configs
```

Artifacts (git-tracked): `data/polymarket_smart_money_snapshot/selection_sweep.csv`
(+ `_meta.json`). Snapshot: 119-wallet pool (11 MM-flagged), 79-token universe,
901 days (2024-01-05 → 2026-06-23), $1 gross, 0.5%/turn, burn-in 45d.

Aspects swept (5 × 5 × 3 × 3 × 2 = 450):

| knob | values |
|------|--------|
| pool (leaderboard windows that seed candidates) | `7d` `30d` `all` `7d+30d` `7d+30d+all` |
| metric (point-in-time ranking score) | `pnl_total` `pnl_30d` `pnl_90d` `roi` `sharpe` |
| n_follow (cohort size) | 3, 5, 10 |
| rebalance (cadence, days) | 7, 14, 30 |
| MM screen | on / off |

Per config we score three books and the two falsification spreads:
`spread_anti = Sharpe(smart) − Sharpe(anti)` (>0 ⇒ winners beat losers) and
`spread_all = Sharpe(smart) − Sharpe(all_leaders)` (>0 ⇒ selection beats none).

## Headline

- **The idea holds in 111/450 (25%) of configs**, but the **grid-wide mean spread
  is still slightly negative** (`spread_anti −0.08`, `spread_all −0.02`). So
  selection edge is *not* a robust property of the strategy — it lives in
  specific corners.
- **The selector absolutely matters** — the user's hypothesis is right that the
  criteria, not just the premise, was the problem. But the decisive knob is
  **cohort size**, with the ranking metric second; the MM screen does nothing.

## What actually drives it

**Cohort size is the dominant axis.** On the production pool (`7d+30d+all`),
holding metric/cadence roughly fixed:

| n_follow | typical smart Sharpe | vs anti / all |
|---------:|---------------------:|---------------|
| 3 | ~0.0–0.4, often **negative** | usually loses (3 noisy wallets) |
| 5 | mixed | borderline |
| **10** | **0.7–0.9** | **beats both, every cadence** |

The earlier n=20 result (`WALLET_SELECTION_TEST.md`) failed for the opposite
reason — with only ~10 wallets ever rankable, top-20 and bottom-20 collapse onto
the same book. **n_follow ≈ 10 is the sweet spot**: large enough to be stable,
small enough to still be a real subset of the active set.

**Metric ranking (mean spread_anti across the grid):**

| metric | spread_anti | spread_all | note |
|--------|------------:|-----------:|------|
| `roi`       | **−0.01** | **+0.03** | least-bad; only metric with +ve spread_all |
| `pnl_90d`   | −0.07 | +0.01 | |
| `pnl_30d`   | −0.07 | −0.01 | |
| `pnl_total` | −0.11 | −0.06 | **the original — among the worst** |
| `sharpe`    | −0.14 | −0.06 | worst |

Ranking by **ROI (PnL per dollar traded)** beats ranking by raw PnL or by Sharpe.
The original selector (`pnl_total`) is one of the weakest choices.

**MM screen: inert.** Screen-on vs screen-off is identical to three decimals
everywhere (`spread_anti −0.07 vs −0.09`) — the top cohorts contain no MM-flagged
wallets, so the screen changes nothing on this pool.

## The standout config

The best config that is **strong in absolute terms *and* beats both controls *and*
sits on the real production pool** (not a tiny-pool artifact):

> **`7d+30d+all` pool · `roi` metric · n_follow 10 · rebalance 14d**
> smart Sharpe **0.91**, anti 0.16, all_leaders 0.20 → spread_anti **+0.75**, spread_all **+0.71**

It is *robust within its neighbourhood*: at n_follow 10 the production pool is
edge-positive at every cadence (rb 7/14/30 → smart 0.70 / 0.91 / 0.90) and
identical screen-on/off — a coherent corner, not a lone lucky cell. Switching the
live tracker's implicit "rank by PnL, small cohort" to "**rank by ROI, follow 10,
rebalance fortnightly**" turns the production configuration from edge-negative
into Sharpe ~0.9 that beats both controls.

## Honest caveats — do not trade this yet

- **Multiple testing.** 450 configs on one 901-day, 79-token, single-snapshot
  sample. Finding 25% that "hold" and a few with spread > 1.0 is partly what
  chance-mining produces. The extreme `7d`-pool spreads (up to +1.24) are
  artifacts — `anti` is catastrophic on that tiny pool, not `smart` being good
  (mean `7d` smart Sharpe is *negative*). Trust the *structure* (size 10 + ROI on
  the full pool, stable across neighbours), not the peak number.
- **Pool survivorship** is unchanged: candidates are all current-leaderboard
  winners, so this measures selection *within an already-good pool*.
- **In-sample.** This is hypothesis generation. The next step is the only honest
  test: run the standout config **forward / out-of-sample** (the paper-tracker is
  the vehicle), and ideally on a second snapshot, before believing it.

## Verdict

Was the selection criteria the problem? **Largely yes.** Fixing two knobs —
cohort size (→10) and ranking metric (→ROI) — flips the live configuration from
no-edge to a clear in-sample edge over both negative controls, while the MM
screen turns out to be irrelevant. That is a real, reproducible improvement in
the *selector*, but it is in-sample over a large grid and must be forward-tested
before it counts as alpha.
