# Smart-money consensus — forward paper-trade tracker

The live, out-of-sample test of the trader-behaviour signals (`ORDER_FLOW`,
`PAYUP_FOLLOW`) — and the honest resolution of their open question: **does the
gross edge survive real spread?** Backtests on the leaderboard-seeded universe are
in-sample and survivorship-prone; a forward ledger is immune to both, and by
entering at the **live CLOB ask** it pays the real bid-ask spread that the
backtest could only proxy.

```bash
python experiments/run_smart_flow_paper_update.py            # daily update
python experiments/run_smart_flow_paper_update.py --dry-run  # scan only, no write
```

## What it does (daily)

1. Build the **smart pool**: recent profit-leaderboard wallets (7d + 30d) minus
   the volume leaders — a light market-maker screen (high profit *and* high volume
   ⇒ likely MM).
2. Pull each pool wallet's most recent fills (uncached — a live signal), keep the
   last 7 days of BUYs, and count **distinct buyers per outcome token**.
3. For every open token with **≥3 distinct smart buyers** (consensus breadth),
   fetch the live order book, confirm it's tradeable, and log a paper **long at
   the best ask** — recording entry mid, ask and **spread**.
4. Mark open positions to the current mid each run; close to 0/1 on resolution
   with realised PnL `(outcome / entry_ask − 1) · fraction`.

Idempotent on the token id (like the macro tracker): new consensus tokens are
appended, resolved ones marked closed, nothing deleted. Runs as a daily GitHub
Actions cron (`.github/workflows/smart_flow_paper.yml`), committing the ledger
back to `data/paper_trade/smart_flow_positions.csv`.

## Why this is the right forward test

- **Spread is measured, not assumed.** The validation showed the book breaks even
  at ≈1¢ half-spread. Here every entry records the *actual* ask−mid, so the ledger
  accumulates the real cost distribution — the `drift` column already shows each
  position down by its spread at entry (we mark mid, paid ask).
- **No survivorship.** Positions are taken on *currently-open* markets going
  forward; the leaderboard only defines the pool, never the outcome.
- **Hold-to-resolution.** Discrete, low-turnover positions (vs the daily-rebalanced
  backtest) — the cleanest read on whether consensus buys actually resolve right.

## ROI-selected variant (forward out-of-sample test)

The selection sweep (`SELECTION_SWEEP.md`) found the original "all winners minus
volume-leaders" pool is a weak selector, and that ranking the pool by **ROI** (PnL
per dollar traded) and following the **top ~10** wallets was the configuration that
beat both negative controls in-sample. To test whether that edge is real rather
than overfit, a second forward ledger runs that exact selector alongside the
original — same consensus mechanics, hold-to-resolution, live-ask entry:

```bash
python experiments/run_smart_flow_paper_update.py \
    --selector roi_topn --top-n 10 --min-buyers 2 --consensus-exit --flip-threshold 1
```

It writes a **separate** ledger, `data/paper_trade/smart_flow_roi_positions.csv`,
so the original out-of-sample record is never contaminated; both run hourly in the
same Actions cron. `min-buyers` is 2 (not 3) because consensus among only 10 elite
wallets is a far higher bar — this is a low-frequency, high-conviction track. The
honest comparison is the resolved-PnL of the two ledgers over the coming weeks: if
ROI selection has real edge, its ledger should out-resolve the original's.

## Reading the ledger

Open `data/paper_trade/smart_flow_positions.csv` on GitHub, or ask Claude Code to
summarise it. Seeded 2026-06-16 with 17 open positions (entry spreads 0.1–3.5¢).
Honest caveats: marks to the CLOB mid (not last trade), the MM screen is a coarse
profit∩volume proxy, and sizing is flat-fraction — this is a forward *signal*
validator, not a sized book.
