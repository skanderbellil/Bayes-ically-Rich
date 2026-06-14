# Forecastable Mechanical Return Levers — strip the math, element by element

> User question (2026-06-14): volatility worked because it is (a) FORECASTABLE
> and (b) linked to terminal wealth by a MATHEMATICAL IDENTITY, not a return
> forecast. What ELSE has those two properties? This decomposes geometric
> (compound) growth term by term; each term that is forecastable-and-mechanical
> is a lever that raises return WITHOUT forecasting returns.

## The decomposition

For a rebalanced portfolio, geometric growth (what actually compounds) is:

```
g  =   Σ wᵢ μᵢ              (A) DRIFT          — the return term
     − ½ σ_p²               (B) VARIANCE DRAIN — vol of the portfolio
     + ½(Σ wᵢσᵢ² − σ_p²)    (C) REBALANCING    — Fernholz excess growth (≥0)
     + Σ wᵢ yᵢ              (D) CARRY / YIELD  — contractual income
     + (skew, kurtosis …)   (E) HIGHER MOMENTS — tail asymmetry
     − costs − taxes        (F) FRICTIONS
```

Term (A) — drift — is the only one that requires forecasting RETURNS, and it
is pure noise (the whole session proved it). **Every other term is
forecastable from quantities that persist, and each is a lever.**

## The levers (ranked: forecastability × mechanical strength × retail-fit × untapped)

| # | Lever | Forecastable input (persists) | Mechanism (no return forecast) | Status |
|---|---|---|---|---|
| 1 | **Variance drain (B)** | volatility (clusters) | vol-target leverage → less σ²/2 drag | **DONE: +6pp geo on QLD** (`run_vol_managed.py`) |
| 2 | **Rebalancing premium (C)** | vol + correlation (persist) | rebalance volatile, uncorrelated assets | **TESTED: +2–5pp** for TQQQ+DBMF etc. (`run_rebalancing_premium.py`) |
| 3 | **Carry / VRP (D)** | implied−realized vol; option premium (persistent ~3–4 vol pts) | sell covered calls / cash-secured puts on the held sleeve → harvest the volatility risk premium ON TOP of the holding | **NEXT — strongest untapped** |
| 4 | **Roll yield (D)** | futures curve shape (contango/backwardation, observable today) | hold backwardated commodity/bond futures; bonds roll down the curve | untested here |
| 5 | **Low-beta premium (A-structural)** | beta (more stable than returns) | tilt low-beta, lever to target beta (BAB; leverage-constraint premium) | KB-known, not retail-built |
| 6 | **Regime: trend vs mean-revert (Hurst)** | Hurst/autocorrelation (somewhat persistent) | apply the matching harvester — rebalance if mean-reverting, trend-follow if trending | program has Hurst/BOCPD |
| 7 | **Higher moments (E)** | options skew, vol-of-vol (forecastable) | avoid negative-skew exposures; size by skew-adjusted growth | conceptual |
| 8 | **Frictions (F)** | your OWN turnover & tax (fully known) | minimize turnover; tax-loss harvest; ETF > mutual fund; band-rebalance | mechanical, always available |
| 9 | **Flow / liquidity calendar** | index reconstitution, month-end, dividend reinvest, net-liquidity (scheduled) | position ahead of forced flows | program's net-liquidity arc |

## The headline hypotheses (testable next)

**H1 — Carry stacking (lever #3) is the biggest untapped one.** A covered call
on the held equity sleeve harvests the volatility risk premium (implied vol
systematically exceeds realized by ~3–4 points — one of the most persistent,
forecastable quantities in markets) as *contractual income*, no return
forecast. Hypothesis: writing systematic calls on the vol-managed QLD sleeve
(or holding JEPI/QYLD-style) adds carry that raises geometric growth in flat/
down tape, at the cost of capped upside — a forecastable trade of convexity
for yield. Test: buy-write overlay on the deployable sleeve, net of the
upside cap.

**H2 — Rebalancing premium is a *timeable* overlay (lever #2).** The premium
(C) is large precisely when vol is high and correlation low — both
forecastable. Hypothesis: scale rebalancing frequency/aggressiveness by the
ex-ante (vol × (1−corr)) of the basket; harvest more when the dispersion is
there. Our QLD+managed-futures book already earns +5pp of this passively;
timing it could add more.

**H3 — Roll yield (lever #4) is the purest "known return".** The futures curve
shape is observable TODAY. Backwardated commodities/managed-futures carry a
positive roll; this is income, not a forecast. Hypothesis: a backwardation
filter on the commodity/managed-futures sled improves its carry.

## The unifying principle

You cannot forecast **returns** (term A — noise). You CAN forecast
**volatility, correlation, carry, curve shape, beta, and your own frictions**
(terms B–F — all persist), and each connects to terminal wealth through
ALGEBRA, not prediction. The session's results are two instances:
- vol management harvests term **B** (variance drain): +6pp on leverage.
- the QLD + managed-futures book harvests term **C** (rebalancing): +2–5pp.
The roadmap forward is to stack the remaining forecastable terms — carry/VRP
(D) next — onto the same sleeve. Each is additive and none needs a return
forecast.

## Honest caveats
- Term (C) is positive in *expectation* but the realized harvest needs mean-
  reversion; a persistently trending asset makes rebalancing LOSE (SPY+TLT
  −3pp 2021–26 as TLT trended down). Forecasting the trend/mean-revert regime
  (lever #6) is the guard.
- Carry/VRP (D) sells convexity — it earns steadily then gives back in crashes
  (the short-vol risk). It must be sized as such, not as free yield.
- All gross of tax; rebalancing and option-writing are taxable events.
