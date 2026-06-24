# Regime-dependent calibration error in Polymarket

> **Result: the behavioural hypothesis survives a first pass — a candidate, not a
> settled, effect.** Low-probability *disruptive* events (geopolitics, crypto/
> financial, politics, AI) are systematically **underpriced during calm regimes**
> and that underpricing **vanishes after recent shocks** — exactly the
> regime-dependent calibration error the hypothesis predicts.

Experiment: `experiments/run_regime_calibration.py`. Panel of **1,576 resolved
legs** (top-volume markets, 2024–26), decision price = last daily mid **7 days
before resolution** (reusing the cached token histories), outcome = the market's
own settlement. No price lookahead; surprise intensity is strictly lagged.

## The hypothesis (formalised)

Traders **underestimate disruptive events during stable regimes** and
**overestimate them after recent shocks**, so prices lag the true hazard rate.
Two falsifiable predictions:

* **P1 — calm underpricing:** for disruptive domains, calibration error
  `CE = outcome − price > 0`, strongest at low prices (tail-adjacent events).
* **P2 — regime dependence:** `CE` is larger after calm and smaller/negative
  after recent shocks; `CE` falls as a lagged per-domain *surprise intensity* `S`
  rises.

`S` for a market = number of disruptive **longshot-YES** resolutions (price <0.20,
outcome 1) in the **same domain** that settled in the 90 days **before this
market's decision date** (leave-one-out, strictly past).

## Finding 0 — favorite–longshot *inversion* (robust)

Across all priced legs, the market is **overconfident at both tails**: longshots
resolve YES more often than priced, favorites less often. Monotonic, both ends
significant.

| price bucket | priced | realized | CE | t |
|---|---:|---:|---:|---:|
| 0.00–0.10 | 0.051 | 0.113 | +0.061 | **3.03** |
| 0.10–0.20 | 0.147 | 0.211 | +0.064 | 1.88 |
| 0.20–0.35 | 0.281 | 0.355 | +0.075 | 2.10 |
| 0.35–0.50 | 0.430 | 0.493 | +0.064 | 1.91 |
| 0.50–0.65 | 0.572 | 0.514 | −0.059 | −1.75 |
| 0.65–0.80 | 0.722 | 0.660 | −0.062 | −1.78 |
| 0.80–1.00 | 0.913 | 0.849 | −0.065 | **−3.58** |

(This is the same phenomenon the mid-price band strategy monetises — here measured
as calibration error rather than PnL.)

## Finding 1 (P1) — tail underpricing is disruptive-specific

At price <0.35:

| group | n | priced | realized | CE | t |
|---|---:|---:|---:|---:|---:|
| **disruptive** | 317 | 0.137 | 0.205 | **+0.068** | **3.14** |
| other-binary | 223 | 0.156 | 0.197 | +0.042 | 1.59 |
| sports (control) | 25 | 0.262 | 0.520 | +0.258 | 2.59 (tiny n) |

Disruptive low-priced events are underpriced and significant; generic binaries
weaker. (Sports has almost no <0.35 legs — moneylines cluster near 0.5 — so the
control n is too small to read.)

## Finding 2 (P2) — regime dependence ✓

Disruptive legs, price <0.35 (where the mispricing lives), split by lagged
surprise intensity `S`:

| regime | n | CE | t | realized | priced |
|---|---:|---:|---:|---:|---:|
| **CALM** (S ≤ domain mean) | 165 | **+0.115** | **3.69** | 0.242 | 0.127 |
| **TURBULENT** (S > domain mean) | 152 | +0.017 | 0.58 | 0.164 | 0.147 |

Difference +0.098 in the predicted direction. Within-domain OLS:
**dCE/dS = −0.039 (t −2.34)** — more recent surprises ⇒ less underpricing, as P2
predicts. After a domain has been shocked, the crowd raises prices on new
tail-events (sometimes to fair, occasionally past it); during calm it under-reacts.

## Honest caveats

1. **Forking paths.** The P2 segment (tail, disruptive) was chosen because P1
   flagged it; t −2.34 is moderate, not decisive.
2. **Endogenous regime measure.** `S` is built from the panel's own longshot-YES
   resolutions. It is strictly lagged and leave-one-out, but a clean confirmation
   needs **exogenous** regime indicators (VIX for financial, GDELT conflict counts
   for geopolitics, model-release cadence for AI).
3. **Time confounding.** Calm/turbulent may partly proxy calendar time; rerun with
   time fixed-effects.
4. **Selection.** 2024–26 top-volume markets only; thin narrative markets (where
   the effect should be strongest) are under-sampled by the volume filter.

## What would settle it

* Swap the endogenous `S` for exogenous per-domain regime indicators.
* Add time fixed-effects and a domain×time panel regression.
* A **forward paper ledger** buying low-priced disruptive legs during calm regimes
  (the tradeable form of the effect) — the only out-of-sample killer.

## Data preserved

`data/polymarket_calibration_snapshot/`: the panel (`regime_calibration_panel.csv`,
1,576 legs with domain/price/outcome/lagged-S), the by-domain summary, and
`token_history_cache_2026-06-24.tar.gz` — the 4,002 cached daily token-price
histories the mid-price and calibration studies were built on (expensive to
re-pull; unpack into `data/raw/`).
