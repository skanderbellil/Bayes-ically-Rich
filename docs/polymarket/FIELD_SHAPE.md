# Does field *shape* condition predictiveness? — peaked vs flat fields

A multi-outcome event can be **peaked** — one candidate towering over the field (an
incumbent at 0.85) — or **flat** — several comparably-priced outcomes splitting the
mass (a genuine toss-up). Both are valid probability vectors, but they encode very
different states of knowledge. This study asks the question directly: *conditional on
its price, does the leader of a peaked field predict the winner differently than the
leader of a flat field?* Shape is the **normalised Shannon entropy** of the
renormalised price vector — `H = -Σpᵢlog pᵢ / log k`, 0 = peaked, 1 = flat.

One row per event (the field leader over a 7–90-day pre-resolution window) so the
binary "did the leader win" fact is counted once per market.

> **Universe.** Multi-outcome fields are scarce and their longshot constituents are
> low-volume, so a tight volume screen both starves the field count and *selects* the
> most-traded (most political) fields — which is exactly how a spurious "leader edge"
> sneaks in. This study therefore pulls a deliberately wide, low-floor universe
> (`--n-markets 2500 --min-volume 3000`) → **83 multi-outcome fields** (80
> winner-take-all), up from the 24 a 20k-volume screen yields. Cached after the first
> (~10 min) pull.

```bash
python experiments/run_polymarket_field_shape.py
```

## 1. Leader calibration does *not* depend on shape

| shape | n | entropy | leader price | leader win | resid | Brier skill |
|---|---:|---:|---:|---:|---:|---:|
| **peaked** | 42 | 0.32 | 0.755 | 0.810 | **+0.055** | **0.60** |
| **flat**   | 41 | 0.69 | 0.489 | 0.585 | **+0.096** | **0.23** |

The leader of a peaked field and the leader of a flat field are **under-priced by
similar amounts** (+0.055 vs +0.096). Event-clustered, the gap is **peaked − flat =
−0.042, 95% CI [−0.234, +0.139]** — indistinguishable from zero. So shape does *not*
re-tune the leader's calibration: the favorite-under-bet tilt is present in both
regimes, no stronger in one. The first-order answer is **no** — a clear favorite is
not differently predictive of *its own* outcome than a leader that barely leads.

## 2. …but shape strongly conditions *information content*

The column that moves hard is **Brier skill vs a uniform 1/k forecaster: 0.60 peaked
vs 0.23 flat.** A peaked field's prices beat "everyone equally likely" by far more
than a flat field's do. Part of that is mechanical (a peaked field is nearer
certainty), but it is the honest, strategy-relevant truth: **flat fields carry
genuinely more irreducible uncertainty** — the price adds little over ignorance,
exactly where a "live" market tempts you to think it's telling you something.

## 3. Where the mispricing lives — price-matched across all constituents

Scoring **every** constituent (window-mean price vs outcome), matched within price
bands so the comparison isn't just "peaked fields have a higher-priced favorite":

| price band | n (peaked) | resid (peaked) | n (flat) | resid (flat) |
|---|---:|---:|---:|---:|
| [0.00, 0.10) | 271 | −0.006 | 231 | −0.002 |
| **[0.10, 0.35)** | 29 | **+0.012** | 75 | **−0.056** |
| [0.35, 0.65) | 13 | +0.113 | 36 | +0.001 |
| [0.65, 1.00) | 33 | +0.003 | 5 | +0.082 |

The **challenger-over-bet effect lives in flat fields**: in the 0.10–0.35 band the
flat-field outcomes resolve Yes *less* than priced (resid −0.056, now over n=75) while
same-priced peaked-field outcomes are calibrated (+0.012). A flat field is precisely
where several plausible challengers invite over-betting. Longshots (<0.10) are
well-calibrated in both.

## 4. "The leader wins 70–83% — can't we just size it and win?"

The tempting trade: buy the field leader, hold to resolution. It *looks* like a big
edge (win 0.83 at price 0.69 ⇒ gross EV ≈ +0.13). Tested honestly on the wide
universe, pooled and split by topic (1% slip, event-clustered bootstrap):

| group | n | leader win | price | PnL/event | 95% CI | t |
|---|---:|---:|---:|---:|---|---:|
| **ALL** | 83 | 0.70 | 0.624 | +0.065 | [−0.029, +0.158] | **1.35** |
| politics | 39 | 0.72 | 0.657 | +0.051 | [−0.096, +0.188] | 0.69 |
| **macro** | 15 | **0.93** | 0.726 | **+0.198** | **[+0.051, +0.298]** | **2.96** |
| sports | 9 | 0.56 | 0.453 | +0.092 | [−0.181, +0.356] | 0.67 |
| crypto | 7 | 0.57 | 0.417 | +0.144 | [−0.204, +0.456] | 0.86 |

*(geopolitics, culture: n=2 each — not interpretable.)*

Three things kill the "just size it" intuition:

* **The 83% was selection.** On the narrow 24-field universe the leader won 83%; on
  the un-selected 83-field universe it wins **70%** at price 0.62 — barely positive EV.
  The original number was inflated by screening to high-volume (political) fields.
* **It isn't significant.** Pooled t = 1.35, CI spans zero. The losses are full-stake
  (a favorite that resolves No costs ≈ −0.70) and **correlated** (one political
  regime), so they don't diversify — sizing a basket of correlated favorites draws
  down together. Kelly only helps a *known* edge; over-betting an unproven correlated
  one is how you blow up.
* **Only MACRO survives:** 15 fields, leaders win 93%, +19.8¢/event, CI [+5.1¢, +29.8¢],
  **t = 2.96.** Tripling the data didn't kill it — it sharpened it. This is the same
  favorite-longshot / short-tail candidate flagged in `STRATEGY_SYNTHESIS.md`, now
  confirmed at higher n. Still a short-volatility trade in a calm-ish regime, but it is
  the one place the "size the favorite" logic is statistically legitimate.

## Takeaway

Field shape answers the question on two levels. **Calibration of the leader is
shape-invariant** — a towering favorite is no better-tuned to its own outcome than a
narrow one. What shape *governs* is **information content** (peaked fields far more
predictive per outcome; flat fields genuinely more uncertain) and the **location of
the challenger-over-bet** mispricing (flat fields' 0.10–0.35 band). And the direct
sizing question resolves cleanly with more data: the *general* leader edge is a
selection artifact (t=1.35), while **macro** is the lone, now-firmer, tradable
favorite-longshot candidate (t≈3) — exactly where the rest of the thread already
pointed.
