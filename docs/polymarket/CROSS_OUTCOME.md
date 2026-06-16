# Anatomy of a multi-outcome field — cross-outcome structure

The outcomes of a single event are not independent: exactly one resolves Yes, so
the candidate prices are mechanically coupled (sum ~1, changes sum ~0). This study
maps that joint structure across ~20 well-covered fields (≥4 candidates, prices
summing to ~1) — the angle the rest of the thread, which treated each market in
isolation, never touched.

```bash
python experiments/run_polymarket_cross_outcome.py
```

## 1. The field is internally arbitrage-free

Median sum-to-one tightness **|S−1| = 0.007** across fields — the prices track a
coherent probability distribution to within ~0.7%. Polymarket's neg-risk mechanism
enforces it, so there is **no static dutch-book** (closing the over-round idea
definitively). The one loose field (Bitcoin price targets, |S−1|=0.28) is correctly
flagged — those outcomes aren't mutually exclusive.

## 2. Substitution is concentrated — the data recovers each real head-to-head

Mean within-field pairwise correlation of log-odds changes is only **−0.07** — *not*
strongly negative, because redistribution doesn't spread across the field, it
concentrates in **one dominant substitute pair**. Ranking each field by its most
negative pair recovers the actual contest automatically:

| event | dominant substitute pair | ρ |
|---|---|---:|
| democratic-nominee-2024 | **Biden ↔ Harris** | −0.83 |
| which-party-wins-presidency | **Republican ↔ Democrat** | −0.98 |
| new-york-city-mayoral | **Mamdani ↔ Cuomo** | −0.80 |
| fed-decision-* | **"cut" ↔ "no change"** | −0.85 |
| next-president-of-south-korea | **Kim ↔ Han** | −0.57 |

And when a candidate is shocked, **82% of the freed probability flows to a single
beneficiary**, not the field. Practical upshot: a messy N-candidate field is, to
first order, a **2-horse race** between the favorite and its dominant substitute plus
dead longshots — useful for hedging (the clean hedge for candidate A is its dominant
substitute, not the basket) and for modelling. The Biden→Harris handoff and the
presidential-field correlation heatmap (`results/polymarket/cross_outcome.png`)
illustrate it.

## 3. Redistribution over-reacts (but isn't tradable)

After a sharp single-day drop in one candidate, the biggest beneficiary's forward
5-day log-odds drift is **−0.095** (only 37% keep climbing; 63% revert). So the
immediate beneficiary *overshoots* and gives some back — the within-field version of
the reversion theme seen everywhere in this project. Fading it (short the beneficiary,
exit 5d, 1% slip) yields PnL/event **−0.009, 95% CI [−0.023, +0.005]** — indistinguishable
from zero. Real overshoot in log-odds, not harvestable net of cost.

## 4. Within-field calibration by rank: leader under-bet, challenger over-bet

In genuine winner-take-all fields (outcomes sum to 1):

| rank group | n | mean price | actual Yes-rate | residual |
|---|---:|---:|---:|---:|
| **favorite** | 19 | 0.638 | 0.789 | **+0.152** |
| **2nd** | 19 | 0.238 | 0.158 | **−0.080** |
| trailing | 88 | 0.017 | 0.011 | −0.006 |

The leader is **under-priced** and the **challenger over-priced** (trailing also-rans
are calibrated) — a "back the second horse" behavioural tilt. It implies a
field-relative-value trade (long favorite / short 2nd), but it rests on just **19
mostly-political fields**, i.e. the same single correlated regime flagged in
`POLITICS_DEEP.md`; the trailing-longshot short's CI also spans zero. Suggestive, not
established.

## Takeaway

The cross-outcome structure is the richest *descriptive* result in the thread: fields
are arbitrage-free, substitution is sharply concentrated into recoverable head-to-heads,
and shocks redistribute to a single substitute with a slight over-shoot. None of the
*dynamic* signals (beneficiary fade, spread reversion) beats costs, and the one
calibration tilt (favorite under-bet) is the familiar one-regime political effect. But
the structural map itself — reduce any field to its 2-horse core, hedge with the dominant
substitute — is a genuinely useful primitive for any future Polymarket modelling or
market-making work.
