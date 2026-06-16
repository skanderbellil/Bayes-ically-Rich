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

```bash
python experiments/run_polymarket_field_shape.py
```

## 1. Leader calibration does *not* depend on shape

| shape | n | entropy | leader price | leader win | resid | Brier skill |
|---|---:|---:|---:|---:|---:|---:|
| **peaked** | 12 | 0.33 | 0.741 | 0.833 | **+0.092** | **0.72** |
| **flat**   | 12 | 0.55 | 0.646 | 0.750 | **+0.104** | **0.51** |

The leader of a peaked field and the leader of a flat field are **under-priced by
essentially the same amount** (+0.092 vs +0.104). Event-clustered, the gap is
**peaked − flat = −0.012, 95% CI [−0.309, +0.292]** — indistinguishable from zero.
So shape does *not* re-tune the leader's calibration: the favorite-under-bet tilt
(seen in `CROSS_OUTCOME.md`'s rank table) is present at ~+0.1 in both regimes, no
stronger in one than the other. The first-order answer to the question is **no** —
a clear favorite is not differently predictive of *its own* outcome than a leader
that barely leads.

## 2. …but shape strongly conditions *information content*

The one column that moves is **Brier skill vs a uniform 1/k forecaster: 0.72 peaked
vs 0.51 flat.** A peaked field's prices beat "everyone equally likely" by far more
than a flat field's do. Part of that is mechanical (a peaked field is nearer
certainty), but it is the honest statement of the obvious truth a strategy must
respect: **flat fields carry genuinely more irreducible uncertainty** — the price
adds less over ignorance, exactly where you'd be tempted to think a "live" market is
telling you something. Predictiveness *per outcome* is lower in a scrum, even though
the leader's calibration bias is the same.

## 3. Where the mispricing lives — price-matched across all constituents

Reducing to the leader throws away the also-rans. Scoring **every** constituent
(window-mean price vs outcome), matched within price bands so the comparison isn't
just "peaked fields have a higher-priced favorite":

| price band | n (peaked) | resid (peaked) | n (flat) | resid (flat) |
|---|---:|---:|---:|---:|
| [0.00, 0.10) | 74 | −0.011 | 23 | −0.028 |
| **[0.10, 0.35)** | 9 | **+0.023** | 13 | **−0.093** |
| [0.35, 0.65) | 3 | +0.171 | 4 | +0.164 |
| [0.65, 1.00) | 10 | +0.012 | 7 | +0.120 |

The **challenger-over-bet effect lives in flat fields**: in the 0.10–0.35 band the
flat-field outcomes resolve Yes *less* than priced (resid −0.093) while the same-priced
peaked-field outcomes are calibrated (+0.023). That fits the intuition — a flat field
is precisely where there are several plausible challengers for the crowd to over-bet,
and it's where the over-pricing of the second/third horse concentrates. Longshots
(<0.10) are well-calibrated in both. But the populated cells are n≈10 over **24
independent fields in one political-era regime**: read the direction, not the
significance.

## 4. Not tradable

Buying the leader to resolution (1% slip) returns **+0.082/event peaked** and
**+0.094/event flat**, both with 95% CIs spanning zero (n=12 each). There is no
shape-conditioned leader edge to harvest — consistent with every other directional
result in this thread.

## Takeaway

Field shape answers the question on two levels. **Calibration of the leader is
shape-invariant** — a towering favorite is no better-tuned to its own outcome than a
narrow one (both +0.1 under-priced, the familiar favorite-under-bet tilt). What shape
*does* govern is **information content** (peaked fields are far more predictive per
outcome; flat fields are genuinely more uncertain) and the **location of the
challenger-over-bet** mispricing (it concentrates in flat fields' 0.1–0.35 band).
Useful as a conditioning lens — "trust a flat field's prices less, and distrust its
mid-priced challengers specifically" — but, like the rest of the structural work, it
rests on ~24 one-regime fields and yields no harvestable directional edge.
