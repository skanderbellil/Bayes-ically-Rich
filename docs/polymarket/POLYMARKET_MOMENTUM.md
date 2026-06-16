# Polymarket cross-market momentum

A prediction-market strategy module, deliberately **outside** the equity/ETF
portfolio work that fills the rest of the repo. It lives in its own subpackage
(`posterioralpha/polymarket/`, like `pead/` and `council/`) and spans all four
pipeline stages.

## Why prediction markets are a different asset class

A Polymarket market resolves Yes/No, and its two outcome tokens trade between
$0 and $1. **The Yes-token price *is* the implied probability** of the event.
That changes everything relative to a long-only equity book:

- prices are bounded in (0, 1), so raw price changes are not comparable across
  markets (a 0.50→0.60 move ≠ a 0.90→0.95 move);
- "return" is awkward — a $1 stake in a 2¢ longshot has a wildly different
  payoff profile than the same stake in a 50/50;
- edge comes from **probability mispricing / drift**, not mean–variance;
- a position can be held to resolution, where the price jumps to exactly 0 or 1.

So we work in **log-odds** space, `z = logit(p)`, the natural unbounded
coordinate for a probability (a fixed step in `z` is an equal multiplicative
odds update — the Bayesian belief-update scale), and we size positions as a
fraction of **$1 gross notional**, marking them to market daily. Per-day PnL of a
weight `w` in market `i` is `w · Δp_i`, accumulated **additively** (a long captures
up-moves; a short — a negative Yes weight, i.e. buying No — captures down-moves).
This "$1-of-max-loss" convention avoids the 1/p blow-up of a naïve percentage
return on cheap longshots and keeps cross-market sizing honest.

## Pipeline

| stage | module | what it does |
|-------|--------|--------------|
| 1 · data | `fetch.py` | Live **Gamma** (market metadata + `clobTokenIds`) + **CLOB** (`/prices-history`) → a daily Yes-price panel (dates × markets). Checkpoint/resume CSV cache. |
| 2 · research | `signals.py` | Log-odds momentum (trailing `z_t − z_{t−L}`), cross-sectional z-score across the live universe, and a Normal–Normal **Bayesian shrink** by each market's own log-odds noise (σ²/(σ²+s²)). |
| 3 · backtest | `backtest.py` | No-lookahead daily engine: signal from prices up to *t*, book held into *t+1*, turnover cost charged at *t*. Four books: `xs_momentum`, `xs_reversal`, `ts_momentum`, `long_all`. |
| 4 · validation | `posterioralpha.validation.compute_metrics` | Reused. Sharpe is scale-invariant; cumulative PnL / drawdown are reported additively on $1 gross. |

```bash
python experiments/run_polymarket_momentum.py                 # cached panel
python experiments/run_polymarket_momentum.py --refresh        # re-pull live data
python experiments/run_polymarket_momentum.py --holding 1 --cost 0.0   # gross, daily
```

## First-pass finding (research artifact, **not** a deployable edge)

On the ~76 highest-volume **resolved** markets (2024-01 → 2026-06, dominated by
the 2024 US-election complex), weekly rebalance, lookback 7d:

| book | gross Sharpe | net Sharpe (50 bps) |
|------|-------------:|--------------------:|
| `xs_momentum` | −0.18 | −0.70 |
| `xs_reversal` | **+0.18** | −0.35 |
| `ts_momentum` | −0.64 | −1.18 |
| `long_all`    | −0.57 | −0.76 |

Two honest reads:

1. **Direction.** `xs_reversal` is the exact negative book of `xs_momentum`, so
   their opposite-signed gross Sharpes are a consistency check, not two findings.
   The positive leg is **reversal**, not momentum: at the weekly horizon the
   cross-section of prediction-market probabilities **mean-reverts** — consistent
   with the well-documented favorite–longshot bias. The signal is weak
   (|Sharpe| ≈ 0.18).
2. **Frictions kill it.** Even the positive-gross reversal book does **not**
   survive a 50 bps turnover cost (net Sharpe −0.35). Turnover is the binding
   constraint; `long_all` bleeds too, reflecting the base rate that most
   "Will X happen?" YES tokens drift toward 0.

### Caveats / known limitations

- **Survivorship & selection.** Only the highest-volume *resolved* markets are
  pulled; the universe is dominated by one event complex (2024 election). Not a
  representative or tradable cross-section.
- **Fills & depth.** PnL marks to the CLOB mid; it ignores spread, slippage, and
  the thin books typical away from headline markets.
- **No resolution capture.** The engine trades intra-life price drift (mark to
  market); it does not model holding to the 0/1 resolution jump.
- **Costs are a guess.** 50 bps/turn is a placeholder; Polymarket's real cost is
  spread-dominated and market-specific.

Treat this as the framework's first wired-up study — a clean, no-lookahead,
real-data prediction-market pipeline — not as evidence of an edge. Next steps:
sweep lookback/holding, widen the universe to non-election markets, model spread
from the order book, and test holding-to-resolution payoffs.
