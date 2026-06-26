# Geo-calm sleeve — the bias, the strategy, the conclusions

> **One line:** we buy cheap *YES* legs of short-dated **geopolitical** prediction
> markets **when geopolitical risk is calm**, because in calm periods traders
> under-price the chance of disruptive events — and we hold to settlement. The
> edge is a *calibration gap*, not a trade signal.

This is the colleague-facing summary. Background hypothesis: `REGIME_CALIBRATION.md`.
Experiments: `experiments/run_validated_regime_backtest.py` (sleeve),
`run_regime_sizing_sensitivity.py`, `run_regime_breakeven_model.py`,
`run_regime_take_profit.py`, `run_regime_trend_exit.py`,
`run_regime_path_distribution.py`, `run_regime_price_calibration.py`.

---

## 1. The bias we exploit

**Availability / salience bias in tail pricing during calm regimes.**

A prediction market price *is* the crowd's probability. When recent geopolitical
news is **quiet**, that crowd under-weights the chance of a disruptive event
actually resolving YES — the event isn't *salient*, so few people pay up for the
cheap YES contracts. The result is a **systematic under-pricing of cheap
geopolitical YES legs in calm regimes**: they resolve YES *more often than their
price implies*. After a recent shock the bias flips (events feel imminent, the
cheap legs get bid up / over-priced) and the edge disappears.

So the edge is a **regime-conditional calibration error**, summarised as
`CE = outcome − price > 0`, concentrated in (a) low prices, (b) the geopolitics
domain, (c) calm regimes.

**Is it real, and how big?** The full prediction-market panel is *well
calibrated* overall — price ≈ win probability (`run_regime_price_calibration.py`,
ECE 0.06 over 4,516 observations). The edge is a **narrow, modest tilt on top of
an efficient market**: geopolitics YES legs alone show an upward miscalibration
(ECE ≈ 0.20) — cheap ones win a bit more than priced. It is *not* a free lunch or
a "0.50 means certain win" effect; it is a small, domain- and regime-specific
mispricing we harvest with size and selection.

## 2. Why a *calm* filter (and how we measure calm)

The behavioural claim is specifically about *calm* periods, so we need an
exogenous calm/turbulent gauge that is **not** built from the markets themselves
(to avoid circularity) and uses **no future information**.

We use the **Caldara–Iacoviello Geopolitical Risk (GPR) index** — a daily,
news-based measure. A decision day is **calm** if, using only data up to that day:

- *level-calm:* trailing-45-day mean GPR ≤ its own trailing-365-day median, **OR**
- *vol-calm:* trailing-45-day volatility of GPR ≤ its trailing-365-day median.

Both are causal (backward-looking) and the threshold floats against the *recent*
normal, so structurally elevated years (2023–26) aren't all flagged "turbulent."

## 3. The strategy (the sleeve)

| Element | Rule |
|---|---|
| Universe | Polymarket **geopolitics** markets |
| Side | **YES** leg, bought at the ask |
| Entry price | mid ≤ **0.35** (the cheap tail, where the bias lives) |
| Regime | **GPR-calm** on the decision day (above) |
| Horizon | resolves in **2–45 days** |
| Exit | **hold to settlement** — winners settle $1, losers $0 |
| Cost | entry spread only (~3¢ haircut); no exit cost when held |

On the validated book (35 trades, 2024–26): ~54% win rate, **net edge ≈ +0.38
per $1** at a realistic 3¢ spread, with the cheap-price convexity doing the
compounding. *Caveat: n is small and spans one calm stretch — treat magnitudes as
indicative.*

## 4. Position sizing — size by the **signal**, not the price (deployed)

We tested tilting bet size by entry price vs by *how deep the calm is*
(`run_regime_sizing_sensitivity.py`):

- **Price is a poor sizing variable** (corr with edge ≈ +0.04). Betting *more* on
  dearer contracts *cut* returns (it starves the convex cheap legs); any
  cheap-tilt gain was indistinguishable from a random-order placebo.
- **Calm-depth predicts edge** (corr ≈ +0.32). Sizing **up when the calm is
  deeper** raised return and improved drawdown/Sortino.

**Deployed:** the live ledger now scales each bet by causal calm-depth
(`bet_fraction = base · exp(g·z)/exp(g²/2)`, capped; `g = 1.0`, `base = 10%`).
Deeper calm → bigger bet; shallow / vol-only calm → floored small. Price is *not*
used to size; price risk is controlled by the ≤0.35 cap.

## 5. Breakeven economics (why cheap entries matter)

For a YES bought at effective price `ep` (`run_regime_breakeven_model.py`):

- one win pays for **`(1−ep)/ep`** losing $1 bets, and
- the **breakeven win rate is just the price** (`p* = ep`).

At `ep ≈ 0.13` one win covers ~6.7 losses and you only need to be right ~13% of
the time to break even. The book's realised win rate (~54%) clears breakeven by a
wide margin (~+0.41), which is the margin of safety the cheap-tail cap buys us.

## 6. Exit research — hold vs take-profit

Idea tested: enter cheap, take profit at "the mids" instead of holding.

- **Dumb fixed take-profit hurts** (`run_regime_take_profit.py`): on these 35
  trades, holding beat every take-profit threshold. The take-profit only **capped
  winners** and saved no losers.
- **Trend-conditional exit** (bank a *stall* in the mids, ride an *uptrend*,
  `run_regime_trend_exit.py`) **fixes** that — it nearly matches hold (and posts a
  slightly higher Sharpe), but doesn't beat it here.

**Important honesty caveat.** The reason holding won is *not* "reaching the mids
guarantees a win." The path-distribution chart that suggested that
(`run_regime_path_distribution.py`) was **small-sample + post-selection** — the
mid-price bins held only 1–5 observations, and the market is otherwise efficient
(a 0.50 contract is ~a coin flip). On those particular 35 trades the losers simply
never rallied; in a larger sample a **stall-exit could** save losers. We keep
**hold-to-settlement** as the default but treat the conservative trend-exit as
cheap, optional insurance.

## 7. Bottom line for the desk

1. **Edge:** a modest, regime-conditional mispricing — cheap geopolitical YES
   legs are under-priced *when geopolitical risk is calm* (availability bias).
2. **Harvest:** buy the cheap YES tail in GPR-calm geopolitics, hold to
   settlement; select hard on price (≤0.35) and regime.
3. **Size with the signal:** bigger in deep calm, smaller otherwise (live, g=1.0).
   Do **not** size by price.
4. **Don't over-trade the exit:** hold; the market is efficient mid-life, so
   take-profit mostly forfeits convex upside.
5. **Caveat:** small sample, one calm regime, optimistic depth/capacity
   assumptions. This is a validated *candidate* sleeve, sized conservatively — not
   a settled, high-capacity strategy.
