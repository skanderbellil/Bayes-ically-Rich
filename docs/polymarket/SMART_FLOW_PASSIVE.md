# Passively entered smart-flow consensus — pre-registered forward experiment

**Registered:** 2026-09-19, before the first data point. Everything in "Frozen
spec" and "Kill criteria" below is fixed *before* this ledger sees a single
resolved position. Any retuning after data arrives is a different, new
experiment — it gets a new ledger, not a rewrite of this one.

```bash
python experiments/run_smart_flow_passive_update.py              # hourly update
python experiments/run_smart_flow_passive_update.py --dry-run    # scan only, no write
```

## Motivation

The independence experiment (`SMART_FLOW_INDEPENDENCE.md`) was retired on
2026-09-19 when both of its frozen kill criteria fired. Taking that ledger apart
left one finding standing, and it is not about which token to buy — it is about
how the position is acquired. Over its 7,788 resolved positions:

| measured on the same 7,788 resolved positions | edge/$1 | event-clustered t |
|---|---|---|
| at the **ask** — what the ledger actually paid | **−0.00730** | −1.93 |
| at the **mid** — the same trades, without crossing | **+0.00862** | +2.36 |

with a mean recorded half-spread of **0.01731**. Those reconcile exactly:
−0.0090 + 0.0173 = +0.0083 per trade. The consensus signal is not noise. It is
worth roughly 0.9¢/$1 at mid, and the sleeve pays about 1.7¢/$1 to cross the
spread and collect it. **The loss is the spread.**

This is the most out-of-sample-stable pattern in that ledger. Splitting by
half-spread, edge at the mid is roughly flat across buckets while edge at the ask
collapses monotonically as the spread widens — and the collapse reproduces
out-of-sample, which almost nothing else in this book does:

| half-spread | IS edge@ask | OOS edge@ask | IS edge@mid | OOS edge@mid |
|---|---|---|---|---|
| ≤ 1¢ | +0.0120 | −0.0008 | +0.0173 | +0.0045 |
| 1–2¢ | +0.0065 | −0.0216 | +0.0188 | −0.0091 |
| 2–5¢ | −0.0416 | −0.0199 | −0.0137 | +0.0076 |
| > 5¢ | **−0.1201** | **−0.1257** | −0.0074 | −0.0061 |

(IS/OOS split at the 70th percentile of settlement dates: 44 in-sample days,
20 out-of-sample.) The >5¢ bucket loses ~12% per trade at the ask in both halves
and ~0.7% at the mid. That is not a bad signal being found out; it is a toll
booth.

## Hypothesis

> Entering the same consensus candidates **passively** — a resting limit at the
> mid, on tokens whose half-spread is within 2¢ — produces positive edge/$1 at
> the realised fill price, where crossing the spread on the same candidates
> produces negative edge.

The claim is about **execution**, not selection. The consensus gate, the wallet
pool, the flow window, the hold-to-resolution rule and the flat sizing are all
byte-identical to the incumbent. Only two things change, and both are about how
the position is acquired.

## Frozen spec (pre-registered, not tunable after data arrives)

| knob | value | why this value |
|---|---|---|
| `MAX_HALF_SPREAD` | **0.02** | The 2–5¢ and >5¢ buckets are where edge@ask collapses, in-sample and out-of-sample alike. 2¢ is the boundary of the last bucket that is not obviously a toll booth — not the value that maximised anything. |
| `LIMIT_AT_MID` | **True** | Post at the mid exactly. Not bid+tick (which is a different, more passive experiment), not mid−ε (which games the fill test). |
| `WORK_HOURS` | **24.0** | The scan runs hourly on a 7-day flow window; a day of working gives the book a fair chance to come to us without letting a stale signal sit for a week. |
| `bet_fraction` | **0.10**, flat | No Kelly, no conviction scaling. Sizing must not vary with anything, or it confounds the execution comparison. |
| consensus gate | `min_buyers=3`, `window_days=7` | Unchanged from the incumbent, deliberately. |

### The fill model

A resting bid fills only when the **offer comes down to it**: on each hourly
poll, if `best_ask <= limit_px` the order is marked filled at `limit_px`. A
falling *mid* is not a fill — that is a mark, and the row stays `working`. If the
book never reaches the limit within `WORK_HOURS`, the order is cancelled and
recorded `unfilled`. An `unfilled` row is terminal: if the market later trades
through that old limit, the print is not ours, because we cancelled.

## The honest failure mode, stated up front

**Passive fills are adversely selected.** A resting bid gets hit precisely when
somebody wants to sell into it, which correlates with the price being about to
fall. The +0.00862 mid-edge above was measured on fills we *know* we would have
got, because we crossed for them. A passive order does not get that set of fills;
it gets a worse-selected subset. The entire question this ledger answers is
whether enough of the mid-edge survives that selection to pay for itself.

`unfilled` is therefore a **first-class recorded outcome, not an error**. A
strategy that fills only when it is wrong is worthless however good its fill
price looks, and that shows up here as a high fill rate on losers and a low one
on winners. Any reading of this ledger that quietly drops the unfilled rows is
wrong.

**A second caveat**, equally important: the mid-edge itself does **not** survive
the independence ledger's own out-of-sample split — in-sample t = +2.95 over 44
settlement days, out-of-sample t = −0.32 over 20. So the correct reading of the
motivation above is "the cost structure is the binding constraint", **not**
"there is a proven edge waiting behind it". This experiment is worth running. It
is not a sure thing, and it was not registered as one.

## Kill criteria (frozen)

**Primary — is there edge at the price we actually get?** Once **≥ 200 filled
positions have resolved**, compute the mean edge/$1 at the fill price, clustered
one mean per settlement date. If the event-clustered `t < 1.0`, retire the sleeve
and freeze this ledger (stop opening, let the rest resolve for the record, do not
restart the experiment on the same file).

Clustering by settlement day is not optional here. This book settles on the order
of 120 correlated positions on a single date, so a per-trade t overstates the
independent evidence by roughly an order of magnitude — that is the same error
`run_paper_kill_battery.py` exists to prevent, and the reason the independence
ledger's 7,788 trades were only ever 64 events.

**Viability — can we even get filled?** If fewer than **20%** of the first
**200 terminal orders** (filled or expired, excluding still-working) fill, passive
entry at this limit is untradeable regardless of the edge on the fills we did
get. Retire it. The honest follow-up would be a *new* ledger at a more aggressive
limit (bid+tick, or mid+1 tick), not a loosened threshold on this one.

Both criteria are recomputed from the ledger by
`smartflow_passive.kill_check()`, so the verdict is reproducible rather than
asserted. The hourly runner prints it every run.

## Ledger

`data/paper_trade/smart_flow_passive_positions.csv` — same idempotent-additive
convention as every other ledger here (new tokens appended, terminal ones marked,
nothing deleted). Runs hourly in the same Actions cron
(`.github/workflows/paper_trade.yml`), `continue-on-error` like its siblings.

Status lifecycle:

```
working ──(offer reaches our limit)──> open ──> won | lost | flipped
   └─────(WORK_HOURS elapsed)────────> unfilled          [terminal]
```

`entry_date` is the **fill** date (blank while working), and `fill_px` is what we
paid — deliberately named so `generate_dashboard.py`'s loader picks the ledger up
with no special case. `working` and `unfilled` rows are neither resolved nor open,
so the dashboard skips them: an order we never got is not a position.

## Reading the ledger

`python experiments/run_smart_flow_passive_update.py` prints working orders,
fills, the fill rate, the mean spread saved versus crossing, and the current kill
verdict. The dashboard carries it as **"Smart Flow (passive)"**.

The number to watch first is the **fill rate**, not the PnL. If it is high (say
>60%), be suspicious rather than pleased: filling that easily at the mid on a
signal the crowd is also trading suggests we are the liquidity of last resort,
and adverse selection should show up in the edge shortly after.
