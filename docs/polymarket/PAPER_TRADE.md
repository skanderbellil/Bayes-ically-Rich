# Macro paper-trade tracker

Forward, out-of-sample tracking of the **macro buy-leader** strategy — the one
candidate that survived 11 in-sample studies with t≈3 and CI excluding zero
(`MACRO_EQUITY.md`, `FIELD_SHAPE.md`). Every day a GitHub Actions cron scans live
Polymarket for open macro multi-outcome fields, logs new positions, refreshes prices,
and marks resolutions. The ledger lives at `data/paper_trade/macro_positions.csv`
committed to the repo.

## Strategy rules (as coded)

| parameter | value |
|---|---|
| universe | macro multi-outcome events (≥3 candidates summing to ~1) |
| entry | field leader (highest implied prob at time of discovery) |
| entry price | mid-price when first scanned |
| hold | to resolution (no take-profit, no stop-loss) |
| sizing | 10% of bankroll per position (configurable `--fraction`) |
| slippage | 1% flat per crossing |
| exit | position settles at 0 or 1 via resolution |

## How to check your positions (from a phone)

Three options — no server needed:

**1. GitHub (easiest).** Open the repo on GitHub → navigate to
`data/paper_trade/macro_positions.csv` → GitHub renders it as a table. Updated daily
by the Actions cron at 09:00 UTC; the last commit message shows the date.

**2. Ask Claude Code.** In any Claude Code session on your phone:
> *"Show me the current paper trade state"*

Claude reads `data/paper_trade/macro_positions.csv` and summarises it. You can also
ask it to run the update script to get a real-time refresh.

**3. GitHub Actions tab.** Go to Actions → *Macro paper-trade tracker* → click any
run to see the full output. You can also click **Run workflow** to trigger an
off-schedule refresh.

## How to run it manually

```bash
# standard daily update (writes ledger)
python experiments/run_paper_trade_update.py

# scan only — print qualifying events without writing
python experiments/run_paper_trade_update.py --dry-run

# use a different sizing fraction
python experiments/run_paper_trade_update.py --fraction 0.20
```

## Honest caveats

* **15 in-sample events, one calm-ish regime.** The t≈3 edge is real by the
  usual standard, but 15 events means the true win-rate confidence interval is
  wide. A cluster of 2–3 out-of-sample losses in the same cycle would not be
  surprising, and sizing at 10% means each costs you 10% of bankroll.
* **The bet is short-volatility.** Win small (+38% of stake at mean entry 0.73)
  and lose large (−100% of stake). A bad macro year gives correlated losses.
* **The sample is political.** The 2024 Fed-path fields landed in a specific
  rate-cycle. If the out-of-sample regime differs materially (e.g. rapid cuts
  surprise the market), the calibration bias may not persist.
* **1% slippage is approximate.** Real execution cost depends on the CLOB spread
  at the time you trade, which can be wider on thin markets.

## Current positions

See `data/paper_trade/macro_positions.csv` — updated daily by the cron.

*First live position logged 2026-06-16:*
> **how-many-fed-rate-cuts-in-2026** · "Will no Fed rate cuts happen in 2026?" ·
> leader price **0.696** · 3-candidate field

This is the genuine out-of-sample test. The in-sample record was 14W/1L over
~20 months. We track forward from here.
