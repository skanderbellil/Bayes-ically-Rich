# Price information efficiency — convergence and which domains are (in)efficient

A descriptive study (not an edge hunt) using the **Brier score** — mean squared
error of the price as a probability forecast — over each market's full life.

```bash
python experiments/run_polymarket_efficiency.py
```

## 1. Prices converge as resolution approaches

| days to resolution | Brier | % "decided" (p<0.1 or >0.9) |
|---|---:|---:|
| 120+ | 0.081 | 75% |
| 60–120 | 0.074 | 78% |
| 30–60 | 0.065 | 78% |
| 14–30 | 0.067 | 79% |
| 7–14 | 0.053 | 81% |
| 3–7 | 0.045 | 83% |
| 0–3 | **0.028** | **88%** |

Monotone: the price becomes a better forecast the closer it gets to settlement, and
by the final three days 88% of markets are already at the rails. Nothing surprising —
but it's the baseline the domain split is read against.

## 2. Efficiency varies enormously by domain

Brier and **Brier skill** (1 − Brier_price/Brier_base, base = predict the group's
Yes-rate) at a fixed 30–60 days before resolution:

| domain | n | base Yes | Brier(price) | skill |
|---|---:|---:|---:|---:|
| **macro** (Fed/rates) | 53 | 0.23 | 0.028 | **+83.8%** |
| other | 36 | 0.28 | 0.092 | +56.5% |
| **sports** | 185 | 0.04 | 0.022 | +39.4% |
| politics | 156 | 0.15 | 0.100 | +21.3% |
| crypto | 12 | 0.25 | 0.132 | +19.4% |
| **geopolitics** | 32 | 0.50 | 0.239 | **+4.5%** |

- **Macro is the most informative.** Fed/rate markets are ~84% of the way to a perfect
  forecast a month out — the Fed telegraphs, and the market reads it.
- **Sports is efficient too** (low absolute Brier; the market clearly adds info over the
  4% base rate).
- **Geopolitics is essentially a coin flip the market cannot forecast.** A month out the
  price carries **~4.5% skill** over simply saying "50%". The convergence chart
  (`results/polymarket/efficiency.png`) shows geopolitics pinned near Brier 0.25 almost
  all the way to resolution while macro/sports collapse toward zero.

## Why this ties the whole thread together

The domains we earlier found **mis-calibrated** (politics, geopolitics) are exactly the
**low-skill / inefficient** ones; the **well-calibrated** domain (sports) is **efficient**.
And it reframes the "missing edge": the inefficiency in geopolitics/politics is mostly
**irreducible uncertainty** — these outcomes genuinely aren't knowable far ahead, so the
price *can't* be informative — not exploitable mispricing that a smarter trader could
harvest. That is the deepest reason the calibration "edge" never became a trade: where the
market looks least efficient, there is the least real signal to capture, only noise.

The practical takeaway for any future Polymarket strategy: **trust the price (and
momentum) in macro and sports; treat far-from-resolution geopolitics/crypto prices as
near-noise** and expect mean-reversion there (consistent with the momentum study, where
the cross-section reverted). Skill, not just calibration, is the variable that tells you
which regime a market is in.
