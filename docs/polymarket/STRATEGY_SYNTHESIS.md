# Can any of this become a strategy? The synthesis

> Short answer: **directional alpha, no.** One narrow structural trade survives —
> the classic favorite–longshot bias in **macro** markets — but it is a
> short-volatility bet in a single calm regime, not a free lunch. And the things
> that *looked* like edges in politics/sports are largely a **volume-selection
> artifact**, demonstrated below.

```bash
python experiments/run_polymarket_strategy_synthesis.py
```

## The one trade that survives: short macro longshots

Shorting over-priced longshots (buy No on [0.03,0.20], hold to resolution),
event-clustered with a bootstrap CI, **per domain**:

| domain | n_evt | PnL/event | 95% CI | hit | worst trade | verdict |
|---|---:|---:|---|---:|---:|---|
| **macro** | 15 | **+0.091** | **[+0.073, +0.110]** | 100% | +0.021 | ✓ robust |
| sports | 17 | −0.077 | [−0.221, +0.027] | 89% | −0.976 | ~ noise |
| politics | 39 | −0.170 | [−0.289, −0.062] | 82% | −0.973 | ✗ loses |
| crypto | 9 | −0.088 | [−0.402, +0.138] | 86% | −0.845 | ~ noise |
| geopolitics | 27 | −0.221 | [−0.399, −0.057] | 63% | −0.969 | ✗ loses |

Only **macro** (scheduled Fed/rate markets) shows the textbook bias: longshot tail
events ("Fed cuts 50bps", "hike 25bps") priced 4–20% that resolved Yes **0%** of the
time. Buying No earns ~+9¢ per contract, the CI is tight and excludes zero, and it
survives 3% slippage. Everywhere else shorting longshots loses or is noise — and note
the **worst trades of ≈ −0.97**: a single underdog win wipes out a long string of
small No-bet wins.

## Why the other "edges" aren't real: volume-selection bias

The politics/sports "underpricing" we kept finding is mostly survivorship. A longshot
only attracts *volume* once it becomes *exciting* — i.e., once the underdog is already
surging toward a win — so a top-volume panel is tilted toward longshots-that-won.
The diagnostic (longshot calibration residual by volume tercile) confirms it:

| domain | low-volume | mid-volume | high-volume |
|---|---:|---:|---:|
| sports | +0.114 | −0.048 | **+0.219** |
| politics | +0.085 | +0.000 | **+0.234** |
| macro | −0.048 | +0.114 | −0.148 |

In sports and politics the high-volume longshots are *far* more "underpriced" than the
low-volume ones — that is selection, not a real-time edge (you couldn't know which
longshots would draw the volume). Macro shows **no** such pattern (scheduled events
draw volume regardless of outcome), which is exactly why its favorite–longshot signal
is trustworthy.

## The honest verdict

- **Directional alpha across domains: no.** Momentum/reversion revert to a zero
  baseline; the political mis-calibration is one correlated regime; the sports/politics
  underpricing is volume-selection.
- **One structural candidate: short macro longshots.** Real, classic, tight in-sample,
  not selection-biased.
- **But treat it as a short-vol trade, not free money.** The ~100% hit rate exists
  because *no tail fired* in the calm 2024–25 rate regime; the true risk (a Fed
  surprise → lose ~0.9 on the No bet) is **unobserved** in this sample. Tiny capacity,
  ~15 events, one regime. Deploy only with explicit tail-risk sizing (e.g., cap loss
  per event, diversify across many independent scheduled events, stop if a regime
  shift makes surprises likely).

## What the project actually produced

Not a deployable edge — but a complete, reusable prediction-market research stack
(live data → log-odds signals → BOCPD events → cost-aware trade sim → level/universe/
event-cluster controls → calibration, efficiency & selection-bias diagnostics) and a
disciplined, self-correcting worked example: every apparent edge was chased to its real
cause (price level → favorite–longshot → one political regime → volume-selection →
irreducible uncertainty). The most valuable output is the machinery to tell a real
bias from an artifact — and a single, honestly-caveated trade candidate to paper-trade
forward.
