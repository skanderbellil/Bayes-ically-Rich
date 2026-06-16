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
| 3 | **Carry / VRP (D)** | implied−realized vol (persistent, forecastable) | sell covered calls / short vol | **TESTED — NEGATIVE: forecastable but NOT orthogonal (same crash risk as equity); covered calls cap upside** (`run_carry_vrp.py`) |
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

## REFINEMENT (2026-06-14): forecastable is necessary but NOT sufficient — also need ORTHOGONAL

Testing lever #3 (carry/VRP, `run_carry_vrp.py`) revealed a second condition.
The VRP is genuinely forecastable (implied > realized 59% of months, fatter
when VIX is high) — yet harvesting it does NOT add return to leveraged
equity, because it is NOT ORTHOGONAL: short-vol / covered-call positions are
+0.66 to +0.88 correlated with QQQ and lose 64-67% of QQQ's drop in crashes.
Carry/VRP is the SAME crash risk you already own, repackaged as yield; and
covered calls cap the upside you leveraged for (capture ~50% up, ~50% down).

So the levers sort into THREE tiers:
  • Tier 1 — IDENTITY (truly mechanical, no risk premium): variance drain
    (lever B / vol management). Pure algebra; nearly free. **Best.**
  • Tier 2 — ORTHOGONAL risk premium: rebalancing with managed futures
    (lever C). A different risk (corr≈0, pays in crashes) → diversifies and
    adds geometric return. **Works.**
  • Tier 3 — SAME-RISK premium: carry/VRP (lever D). Forecastable, but it
    pays you for the crash risk you ALREADY bear → concentrates, not
    diversifies. **Does not add return to an equity sleeve.**

The test for any candidate lever is now TWO-PART: (1) is the input
forecastable (persists)? (2) is the payoff ORTHOGONAL to what you already
hold (corr≈0 AND ideally positive in your drawdowns)? Volatility and the
managed-futures rebalancing premium pass both; carry/VRP passes only the
first. Re-rank the remaining levers (#4 roll yield, #5 low-beta, #6 regime,
#7 skew, #8 frictions, #9 flow) by BOTH tests before building.

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
