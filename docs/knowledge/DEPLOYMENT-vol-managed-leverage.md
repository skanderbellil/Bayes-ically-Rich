# Deployment Spec — Volatility-Managed Leveraged Equity (retail / Alpaca)

> The session's deployable conclusion. A long-only, no-shorting book that
> compounds faster than QQQ by managing **volatility** (predictable), never
> **returns** (noise). Validated out-of-sample and across the parameter grid
> (`experiments/run_deploy_spec.py`). **This is an aggressive growth sleeve —
> read the risks.**

## The one rule

Each day, size leveraged-equity exposure inversely to recent volatility:

```
vol_QQQ_t   = EWMA( daily returns of QQQ, halflife = 20 trading days )
              annualized = vol_QQQ_t * sqrt(252)
w_t         = clip( TARGET_VOL / (2 * vol_QQQ_t) , 0 , 1 )     # 2 = QLD's leverage
```

Hold `w_t` of the book in **QLD** (2× QQQ) and `1 − w_t` in the **safe asset**.
`w_t` uses *yesterday's* vol (no look-ahead). Calm tape → w≈1 (full 2×);
vol spike → w→0 (out of leverage).

## Tickers

| leg | ticker | role |
|---|---|---|
| risk | **QLD** (2× QQQ) | the return engine (use SSO/2× SPY for less tech concentration) |
| safe | **DBMF** (managed futures) or **BIL/SGOV** (T-bills) | where capital sits when de-risked; DBMF adds crisis convexity, cash is simplest |

All long-only, all on Alpaca, no shorting, no margin.

## Parameters (canonical — robust, not optimized)

| param | value | notes |
|---|---|---|
| EWMA halflife | 20 trading days | robust across 10–40 (grid-tested) |
| TARGET_VOL | **0.25** (moderate) or **0.30** (aggressive) | this is a *risk preference*, not a fitted edge |
| exposure cap | 1.0 (100% QLD = 2× net) | no leverage beyond the ETF |
| rebalance | daily check; trade only when `w` drifts > 10% | controls turnover (~switch cost 10 bp modeled) |

## What to expect (2011–2026, net of cost)

| book | CAGR | Sharpe | maxDD |
|---|---|---|---|
| QQQ buy & hold | +19% | 0.75 | −35% |
| QLD buy & hold (2×) | +32% | 0.78 | −64% |
| **vol-managed QLD @ 25%** | **+23%** | **0.78–0.87** | **−30%** |
| vol-managed QLD @ 30% | +26% | 0.80 | −35% |

Out-of-sample (2017–26): **+26% CAGR / 0.87 Sharpe vs QQQ +22% / 0.80.**
Crisis years (vol-managed @25%): 2018 −3%, 2020 +30%, 2022 −23%, 2025 +10%.

## Why it works — no return forecast anywhere

Compound growth ≈ `μ − σ²/2` (the variance drain). You cannot forecast `μ`
(returns are noise), but you *can* forecast `σ` (volatility clusters —
today's vol predicts tomorrow's). A 2× ETF has ~4× the variance drain, so
holding it only when vol is low harvests most of the leverage's return while
shedding the drain that destroys it. The edge is a mathematical identity made
tradable, not a pattern fit to history.

## Risks — read before deploying

- **2× leverage. Drawdowns of −30% to −35% are expected**, and worse is
  possible. This is not a conservative allocation.
- **Blind spot:** a slow grind-down where price falls while volatility stays
  *low* would keep the book levered into losses. The signal reacts to vol, not
  price — it protects against volatile crashes (2020, 2022), not quiet bleeds.
- **Leveraged daily-reset decay** still bites in choppy, elevated-vol sideways
  regimes (the signal reduces but does not eliminate it).
- **One regime of history.** 2011–2026 is a single (tech-heavy, mostly-bull)
  sample with two crises; the mechanism is robust in theory and OOS, but size
  to a loss you can hold through, and treat TARGET_VOL as your risk dial.
- **Taxes/turnover:** daily-checked but band-triggered trading keeps turnover
  modest; still a taxable-account consideration.

## Provenance
`experiments/run_vol_managed.py` (mechanism + dashboard),
`run_daily_leading_signal.py` (the binary-signal cousin, higher Sharpe / has a
threshold), `run_deploy_spec.py` (OOS + grid validation). Roadmap: the
"wall broken" / "pure vol management" entries under Idea 18.
