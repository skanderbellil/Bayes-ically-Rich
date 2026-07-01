# Champion stack — forward paper-trade ledger

Out-of-sample, real-time paper-trade record of the flagship equity strategy —
the **champion stack** frozen in `experiments/run_champion_stack.py`. The
Polymarket strategies have had a survivorship-free forward ledger for a
while (`docs/polymarket/PAPER_TRADE.md`); the equity side didn't, even though
it's the strategy the repo is built around. This closes that gap.

A GitHub Actions cron (`.github/workflows/paper_trade.yml`, hourly — the
"Champion stack (equity forward ledger)" step) runs
`experiments/run_champion_paper_update.py`, which downloads fresh daily
prices, computes today's weights, and appends one row per calendar date to
`data/paper_trade/champion_positions.csv`, committed back to the repo.

## Why this exists

A backtest can always be re-run against restated data, a different warm-up
window, or a tweaked parameter — a forward ledger can't. This is the one
piece of evidence about the champion stack that is generated *before* the
outcome is known, with no ability to revise history. It won't have enough
observations to be statistically decisive for a long time; treat it as a
live audit trail, not (yet) as validation.

## Frozen spec (reference, not re-derived here)

The ledger replicates `experiments/run_champion_stack.py`'s frozen
configuration exactly — read that script's module docstring for the full
derivation and the ablation-ladder / leave-one-out evidence for why each
piece earns its place. The live computation lives in
`posterioralpha/research/champion_live.py::compute_current_weights()`,
which duplicates (not imports — `experiments/` isn't a package) the same
formulas:

- **vote** — equal-weight mean of 5 soft-signed layers: `liq` (Δ126d net
  liquidity, publication-lagged 5bd), `vix_ts` (VIX3M/VIX − 1), `credit`
  (HYG−IEF 21d relative return), `dollar` (−UUP 63d return), `tug` (21d
  overnight−intraday compounded spread on QQQ).
- **flip-risk slow-down** — `f = 1 − causal-rank(std of 4 price stances)`
  (trend, mean-reversion, BOCPD run-length rank, walk-forward HMM3 stance);
  `vote_eff = f·MA63(vote) + (1−f)·vote`.
- **exposure** — `e = 2×vote_eff`, capped to `[0, 2]`.
- **debt-ceiling ceiling** — `e ×½` for 60 business days after each
  debt-ceiling resolution.
- **retail wrapper** — `w_qld = e/2` (capped `[0, 1]`) held in QLD,
  `w_uup = 1 − w_qld` held in UUP.

## Ledger columns

`data/paper_trade/champion_positions.csv`, one row per calendar date:

| column | meaning |
|---|---|
| `date` | the trading date the signal is asof |
| `liq`, `vix_ts`, `credit`, `dollar`, `tug` | the 5 soft-signed layer strengths, `[0,1]` |
| `vote` | equal-weight mean of the 5 layers |
| `flip_f` | flip-risk slow-down weight, `[0,1]` |
| `vote_eff` | flip-risk-adjusted vote |
| `exposure` | `e`, `[0,2]`, ceiling-adjusted |
| `w_qld`, `w_uup` | retail wrapper weights |
| `qld_close`, `uup_close` | closes used for the return calc |
| `liq_stale_days` | see caveat below |
| `strategy_return` | *yesterday's* `w_qld`/`w_uup` applied to *today's* QLD/UUP close-to-close returns (blank on the first row) |
| `equity` | cumulative product of `(1 + strategy_return)` starting at 1.0, **recomputed from the full column every run** so any correction propagates through the whole history |

Re-running the update script on the same day overwrites that date's row —
it never appends a duplicate.

## Caveats

**Net-liquidity staleness.** `liq` needs Fed net liquidity (WALCL − TGA −
RRP). If `FRED_API_KEY` is set (via the `FRED_API_KEY` repo secret, wired
into the workflow step's `env:`), it's refreshed live each run. If not — or
if the refresh fails — the script falls back to the bundled
`datasets/net_liquidity.csv` snapshot and the row's `liq_stale_days` records
the gap, in calendar days, between the signal date and the last real FRED
observation actually in hand. Because WALCL/TGA/RRP are weekly-published,
`liq_stale_days` is not zero even on a fresh pull (typically a handful of
days) — it's only a red flag when it's large (multi-week), meaning the
`liq` layer is running on old information.

**Debt-ceiling list caveat.** `DEBT_CEILING` in `champion_live.py` is
hardcoded through Jun-2023, copied as-is from `run_champion_stack.py`.
**It must be appended by hand** as Congress resolves future debt-ceiling
episodes — there is no live feed for this. If the list goes stale, the
ceiling discount will simply stop firing at the next resolution instead of
erroring, so it's worth checking this list occasionally.

## Running it manually

```bash
python experiments/run_champion_paper_update.py
```

Network failures (yfinance down/rate-limited, FRED unreachable) print a
message and exit 0 — a missed day is fine, a red CI run for a transient
network hiccup is not. Real bugs in the pipeline (bad warm-up length, a
formula error) raise and fail the run loudly.

## Current state

See `data/paper_trade/champion_positions.csv` — updated hourly by the cron.
