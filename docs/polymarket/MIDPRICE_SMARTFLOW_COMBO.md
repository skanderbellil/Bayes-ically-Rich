# Mid-priced YES × smart-flow consensus — the combo backtest

`experiments/run_midprice_smartflow_combo.py` · snapshot in
`data/polymarket_combo_snapshot/` · first run 2026-07-02

## Question

The two live paper books each trade one signal — the mid-price band
(hold-to-resolution upward drift of in-band markets) and smart-flow consensus
(≥N non-MM leaderboard wallets net-buying). Does the intersection beat either
alone? Four books per (band, horizon) isolate the attribution: `midprice`
(every in-band market), `combo` (in-band AND consensus), `anti` (in-band, zero
smart bulls), `flow_only` (consensus outside the band).

## Data reality (worth recording on its own)

* The trades data-api **hard-caps offset paging at 3,000 fills** and the
  pool's hyper-active wallets burn that in **under a day** — a naive
  per-wallet fetch gives near-zero historical coverage (the honesty gate
  zeroed the first run's sample rather than let it lie).
* Fix: the API accepts an (undocumented) **`end` timestamp** — fills are
  walked back in time per wallet under a request budget, cached under
  `data/raw/polymarket/traders_window/`.
* Feasible window ≈ the last ~90 days of resolutions (354 markets, 536
  usable obs at ≥15 visible wallets, 2026-03 → 2026-06). Deeper history
  needs either bigger request budgets or the forward ledgers.
* The pool (52 wallets: profit-leaders minus volume-leaders) touched only
  **28%** of observations — consensus is a sparse overlay, not a re-ranking
  of the whole band.

## Findings (haircut 2¢, window 7d, CIs cluster resolution dates)

1. **At the 7d horizon flow adds nothing to the band.** `combo` ≈ `anti` ≈
   `midprice` (mean +0.08 / +0.25 / +0.23 per $1, all CIs spanning 0). The
   band's fat PnL sits in the cheap [0.10,0.30) lottery slice — and there
   the pool-ignored names (`anti` +0.99) *beat* the pool-touched ones.

2. **The one live cell: 3d horizon × mid band [0.30,0.70).** The level
   effect is dead there (`midprice` +0.003, n=105) but consensus ranks it:
   `combo≥1` **+0.220 per $1, CI [+0.083, +0.623]**, win 69% (n=35 over 16
   independent resolution days) vs `anti` −0.059. The wide-band `combo≥2` at
   3d also clears zero (+0.135, CI [+0.006, +0.662]). Reads as: smart flow
   is *short-horizon information* — it pays in the final days, in the price
   region where the market is genuinely uncertain, not in the
   favorite-longshot tails the band already harvests.

3. **Flow without the level filter is an anti-signal.** `flow_only` (pool
   consensus on markets priced <0.10 / ≥0.90) is a disaster at both horizons
   (−0.86 and −0.88 per $1, CIs entirely negative, win 12–13%): the pool's
   extreme-price buys are toxic longshot lottery tickets. The SMART_MONEY
   inverse-selection result in one more costume — and the reason the live
   consensus tracker should never buy tail-priced tokens.

## Caveats (read before trading anything)

* **Trials.** ~96 (band × horizon × threshold × book) cells were examined;
  the highlighted cell would not survive a formal DSR charge on its own.
  Treat as a *pre-registered hypothesis for the forward test*, not an edge.
* Pool membership is today's leaderboard (survivorship in who exists);
  fills themselves are timestamped and causal.
* One 90-day regime; entries at daily close + 2¢ assumed spread (the
  `book_snapshots.csv.gz` accumulation will make this measurable).

## Bankroll terms (the dashboard's exact convention)

$1,000 start, **no leverage**: a position ties up cash entry → resolution,
stake = min(10%·equity, cash), skipped when fully committed — the same cash
sim the dashboard runs (positions marked at cost between entry and
resolution here; the dashboard marks hourly). Daily-series Sharpe annualised
√365. 83-day sample (2026-04-11 → 2026-07-02). **CAGR is NOT annualised** —
extrapolating 83 days is meaningless; total return is the honest number.

| book (3d horizon, band [0.30,0.70)) | n | win | total ret | ann. Sharpe | maxDD | flat-$10 |
|---|---|---|---|---|---|---|
| midprice (all in-band)   | 105 | 53% | **−34%** | +0.05 | −71% | +0.3% |
| **combo ≥1 bull**        | 35  | 69% | **+74%** | **+3.60** | −36% | +7.7% |
| anti (0 bulls)           | 59  | 47% | **−45%** | −0.76 | −71% | −3.5% |
| 7d wide-band midprice (context) | 125 | 55% | +47% | +1.88 | −66% | — |

The flat-$10 column (fixed $10/trade, no compounding or sizing) is the raw
per-trade edge — the same statistic the dashboard's "flat" mode reports.
Note the accounting lesson: an unconstrained 10%/trade sim (allowed to hold
unlimited concurrent positions — implicit leverage) showed the wide-band
book at +466%; the honest no-leverage sim cuts it to +47% at −66% DD. The
combo book barely moves under the constraint (35 short-hold trades rarely
overlap), which is itself evidence its result is not a sizing artifact.
Same trials caveat as above — the forward test decides.

The live dashboard now carries a derived **"Smart Flow ∩ band .30–.70"**
sleeve (filter view of the smart-flow ledger — band + tail-price exclusion;
not added to COMBINED, which would double-count). Its forward KPIs score the
pre-registered criterion below; the ≤3-days-to-resolution leg isn't
recoverable for the ledger's open rows (no end_date column), so the sleeve is
band-only — strictly more conservative than the backtested cell.

## Pre-registered confirmation (do not tune)

Forward criterion, decided 2026-07-02 before any new data: on markets
resolving after this date, `combo≥1` entries in [0.30,0.70) with ≤3 days to
resolution should show mean net PnL > 0 at the same 2¢ haircut. The live
smart-flow tracker already records everything needed to score this; the
midprice ledgers provide the `midprice`/`anti` controls.
