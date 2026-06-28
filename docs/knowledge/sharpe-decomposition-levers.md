# Sharpe / IC Decomposition — the four levers, and why the formula holds

> Source: Giuseppe Paleologo thread (2026) on decomposing the information
> coefficient and Sharpe ratio through the Fundamental Law of Active
> Management. The point: a strategy's Sharpe factors into independent,
> multiplicative levers, so a losing book can be diagnosed by *which* term is
> near zero — instead of blindly re-tuning a model whose signal may not even
> matter.

## The decomposition

Start from the Fundamental Law of Active Management:

```
Sharpe = IC · √(n · T)        IC  = info coefficient (corr of forecast vs return)
                              n   = independent bets in the cross-section
                              T   = independent decision periods per year
```

In practice you almost never forecast **returns** directly. You forecast a
**driver variable** Z (next quarter's EPS, a flow, realized vol …), and Z maps
to returns. Split the IC along that chain:

- **ρ₁** = how well your forecast tracks the driver (your skill at predicting Z)
- **ρ₂** = how well a *perfect* forecast of Z would track returns (how much Z
  actually moves price)

Under a partial-correlation chain (Z is the only channel from forecast to
returns), the realized correlation is the product `IC = ρ₁ · ρ₂`. Substituting,
and writing `ρ₁ = √(R²)` (R² of your forecast on the driver) and `ρ₂ = IC_ideal`
(the "oracle" IC of a perfect driver-forecaster):

```
Sharpe = √(R²) · IC_ideal · √(breadth) · √(decisions per year)
            │         │          │              │
       forecast    oracle    # independent    how often
       accuracy    value        bets          you bet
```

## The four levers (each is independently diagnosable)

| # | Lever | What it measures | How to improve | How to estimate |
|---|---|---|---|---|
| 1 | **Forecast accuracy `√(R²)`** | skill at predicting the *driver* | better features/models on the driver itself | regress your forecast on the realized driver |
| 2 | **Oracle IC `IC_ideal`** | how much the driver maps to returns *if known perfectly* | switch to a driver that actually moves price | correlate a perfect-foresight driver vs returns |
| 3 | **Breadth `√n`** | number of *independent* bets | widen universe; de-correlate signals | effective N (haircut for cross-correlation) |
| 4 | **Frequency `√(decisions/yr)`** | rebalances per year | trade more often *if* the signal decays slowly | turnover / holding period |

Because the terms **multiply**, a near-zero in any one kills the whole book —
and the decomposition tells you which knob to turn:

1. **`√(R²)` low** — you're bad at predicting the thing. Fix: better
   models/data on the driver.
2. **`IC_ideal` low** — the thing doesn't matter. You could predict EPS
   *perfectly* and still earn nothing because EPS surprises aren't moving these
   names. **This is the term people ignore** — no amount of forecasting skill
   rescues a dead signal. Fix: pick a driver that actually drives price.
3. **`√n · √(decisions/yr)` low** — you aren't betting on enough independent
   names, often enough. Fix: widen the universe, de-correlate, trade more.

**Research discipline:** always decompose a backtest's Sharpe into these four
*before* "improving" it, so you fix the binding constraint rather than
over-tuning a model whose driver is worthless.

## Why this *is* the conventional Sharpe (μ/σ), just derived

`SR = IC·√(nT)` is not a new definition — it is `μ/σ` after annualizing a
cross-section of bets.

**One bet, one period.** Standardize the forecast `x` (mean 0, var 1) and the
realized return `r` (var 1); by definition `corr(x, r) = IC`. Size the position
∝ forecast. Then:
- Expected PnL per period = `E[x·r] = IC`
- SD of PnL ≈ 1 (for small IC)
- So `SR_per bet = μ/σ ≈ IC`. **The IC is a one-shot Sharpe.**

**Aggregate.** With `n` independent bets in the cross-section and `T`
independent periods per year — `nT` independent bets total — means add linearly
(∝ nT) and SDs add in quadrature (∝ √(nT)):

```
SR_annual = (nT · IC · σ) / (√(nT) · σ) = IC · √(nT)
```

That `√(nT)` is the exact same √-time annualization as `√252` for a daily
Sharpe: Sharpe grows with the square root of the number of *independent*
observations. The Fundamental Law is just `μ/σ` + √-time annualization applied
to a cross-section.

## When the identity breaks (it's a ceiling, not an equality)

The formula is exact only under idealized assumptions: bets independent,
equal-vol, ~normal, sized optimally. Reality breaks each, and each break has a
name:

- **Bets aren't independent** → true breadth ≪ nominal `n`. Correlated names
  inflate the apparent breadth (Buckle's "effective breadth" critique). Use a
  correlation-haircut effective N, not the raw count.
- **You can't fully act on forecasts** (constraints, costs, risk limits) →
  multiply by the **transfer coefficient**: `IR = TC · IC · √BR`
  (Clarke–de Silva–Thorley, 2002). A long-only book often runs TC ≈ 0.3–0.5,
  losing half the theoretical Sharpe right there.
- **Estimation error, fat tails, costs** → realized Sharpe < formula.

So treat `IC·√(nT)` as the *ceiling* a clean `μ/σ` would reach, useful for
diagnosis — not as a literal equality to a realized backtest. Measure realized
return over realized SD for the real number; the formula tells you why it's
that size and where it leaks.

## Side note: Sharpe is dimensionless — don't quote it in %

`μ/σ` is return over return; the units cancel. That is the whole point — it lets
you compare a bond book to a crypto book. A pure number *can* technically be
written as a percent (like a probability 0.5 → "50%"), so "1.5 → 150%" is not
dimensionally illegal, but it is **non-standard and invites misreading** (people
expect "Sharpe 1.5", and may decode "150%" as a return or an information ratio).
The one real error to rule out: make sure a "%" Sharpe isn't a mislabeled raw
**return** or an information **ratio** (IR) vs **coefficient** (IC) — *that*
would be a genuine mistake, not just a convention slip.

## Where this lives in the repo

`posterioralpha/research/breadth.py` implements the diagnostics:

- `effective_breadth(returns | n, avg_corr)` → `N / (1 + (N−1)·ρ̄)`
- `average_pairwise_correlation`, `participation_ratio` (spectral alternative)
- `transfer_coefficient(w_actual, w_ideal, cov)` — risk-adjusted weight
  correlation (Clarke et al.); `empirical_transfer_coefficient(sr_c, sr_i)` —
  Sharpe-ratio proxy (model-free but **unbounded — only trust it when the ideal
  Sharpe is solidly positive**; prefer the weight-based TC otherwise)
- `fundamental_law_sharpe(ic, breadth, periods_per_year, tc)`,
  `cross_sectional_ic(forecast, realized)`

Wired into `experiments/run_equity_cross_section.py` (a "Fundamental-Law
diagnostics" block). **Empirical finding on the 500-name US universe (12-1
momentum, decile L/S):** a 100-name book at ρ̄ ≈ 0.35 has an *effective breadth
of ~2.8 — just 3% of nominal* (correlated names ≠ independent bets), while the
decile truncation transfers TC ≈ 0.70 of the full-rank ideal. The breadth
illusion, not the truncation, is the dominant leak here.

## TL;DR

- Sharpe `= √(R²) · IC_ideal · √(breadth) · √(decisions/yr)` — four independent
  multiplicative levers; diagnose the near-zero one.
- It's the same `μ/σ`, derived: IC is a one-shot Sharpe, `√(nT)` is √-time
  annualization over independent bets.
- It's a ceiling — correlated bets (effective breadth) and the transfer
  coefficient (TC ≈ 0.3–0.5 long-only) pull realized Sharpe below it.
- Sharpe is unitless; quoting it in % is a clarity smell, not a math error —
  but check it isn't a mislabeled return or IR.
