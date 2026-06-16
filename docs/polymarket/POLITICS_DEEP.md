# Digging into the political edge — real mis-calibration, not a confirmed trade

> Fair challenge: if the underpricing spans many different political markets, why not
> exploit it? Answer: the **mis-calibration is real and diversified**, but once PnL is
> entered at realistic prices and **clustered by event**, its confidence interval spans
> zero and it is **statistically indistinguishable from the non-political baseline** —
> and it is already decaying in 2026. A real behavioural fact; not a confirmed edge.

```bash
python experiments/run_polymarket_politics_deep.py
```

## It *is* diverse — that part of the intuition holds

The mid-band ([0.15,0.30]) underpricing is **not** one event. Across the 514-market
universe it spans **25 distinct events** — NYC mayoral, Canada PM, Romania / Poland /
Portugal presidents, Fed-chair nominee, Democratic nominee, popular-vote markets — and
**persists across years** (2024 +10%, 2025 +12%). Episode-weighting inflates it
(+41.7%), but de-correlated to one observation per event it is still **+25.6%**, and an
event-clustered bootstrap over [0.15,0.30] gives a CI that excludes zero. My earlier
"it's just 2024" framing was too harsh, and the multi-candidate-field hypothesis was
wrong (the effect is actually *stronger* in binary markets, +31%, than fields, +10%).

## The calibration curve — politics is genuinely mis-calibrated

One observation per market, equal-weighted:

| price band | n | mean price | actual Yes-rate | residual |
|---|---:|---:|---:|---:|
| [0.02, 0.10) | 74 | 0.05 | 9.6% | +4.7% |
| [0.10, 0.20) | 40 | 0.15 | 53.6% | +20.5% |
| [0.20, 0.35) | 38 | 0.27 | 57.0% | +12.3% |
| [0.35, 0.50) | 32 | 0.43 | 37.4% | +8.8% |
| [0.50, 0.65) | 28 | 0.56 | 54.9% | −6.8% |
| [0.65, 0.80) | 21 | 0.72 | 23.6% | **−29.0%** |
| [0.80, 0.98) | 16 | 0.88 | 53.2% | **−43.1%** |

The non-political curve hugs the 45° diagonal (`results/polymarket/politics_calibration.png`);
the political one is an inverted-S — **underdogs underpriced, favorites badly
overpriced** (a 0.88-priced political "lock" resolved Yes only 53%, driven partly by
dramatic favorite-collapses like Biden's withdrawal). This is a real overconfidence-in-
favorites behavioural pattern, and it is specific to politics.

## But as a trade it does not survive (event-clustered, realistic entry, 1% slippage)

PnL per **event** (correlated sub-markets collapsed to one observation) + bootstrap CI:

| strategy | n_trd | n_evt | PnL/event | 95% CI (per event) |
|---|---:|---:|---:|---|
| politics: long [0.10,0.50] Yes | 61 | 37 | +0.077 | **[−0.047, +0.212]** |
| politics: short [0.65,0.95] favorites (No) | 33 | 27 | +0.125 | **[−0.042, +0.295]** |
| politics: long + short combo | 94 | 40 | +0.048 | **[−0.034, +0.133]** |
| non-politics: long [0.10,0.50] Yes (control) | 128 | 82 | +0.049 | [−0.036, +0.136] |

Three things kill the "exploit it" case:

1. **Every CI spans zero.** Point estimates are positive (+0.05…+0.13/event) but, with
   only ~one political era of events, none is statistically distinguishable from zero.
2. **It does not beat the baseline.** The political long-underpriced strategy
   (+0.077/event) is statistically indistinguishable from the *non-political* control
   (+0.049/event). The big calibration residual was inflated by episode-tenure
   weighting and by ignoring realistic entry + slippage; trading it honestly shrinks it
   to baseline.
3. **It is decaying.** By year the mid-band residual runs 2024 +10%, 2025 +12%,
   **2026 −4%** — the most recent political markets are reverting.

## Verdict

The political mis-calibration is the most promising lead in the whole Polymarket thread
and is a genuine, diversified behavioural fact — but it is **one correlated political
era**, so the ~25 events share a latent regime (this period's elections broke toward
outsiders and against high-priced favorites). That makes it a **directional regime
bet**, not an established free lunch: entered honestly and judged by event-clustered
inference, its edge is indistinguishable from zero and from non-political markets, and
it is fading in the newest data.

The only way to upgrade this from "promising" to "confirmed" is **genuine out-of-sample
political cycles** — which this sample can't manufacture. The concrete next step is to
pull **pre-2024 Polymarket history** (2020–2023 elections and referenda) and re-run the
event-clustered test on cycles the 2024 regime didn't generate; if the favorites-
overpriced / underdogs-underpriced curve holds there too, it becomes tradable. Until
then: real finding, unproven edge.
