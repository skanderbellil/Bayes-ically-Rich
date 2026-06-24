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

## Tradeable form — backtest + hostile audit

`experiments/run_regime_strategy_backtest.py` trades the effect: buy the cheap
(price ≤ 0.35) YES leg of a disruptive-domain market a lookback before
resolution, **only when the domain has been calm** (lagged S below its per-domain
median). 1% stake, 1¢ haircut, hold to settlement. Built on the committed panel.

**Does the calm filter add value? (the whole thesis)**

| book | n | edge/$1 | t | CAGR | Sharpe | maxDD |
|---|---:|---:|---:|---:|---:|---:|
| all disruptive-tail | 240 | +0.100 | 3.84 | +65% | 1.31 | −12% |
| **calm only (strategy)** | 128 | **+0.140** | 3.78 | +47% | 1.35 | **−9%** |
| turbulent only | 112 | +0.054 | 1.51 | +15% | 0.54 | −29% |

Conditioning on calm ~3× the per-$1 edge of the turbulent book and cuts drawdown
from −29% to −9%. The regime filter *is* the alpha — not just "buy cheap
disruptive."

**Audit (calm book, n=128):** positive in all three years (2024 t 2.54, 2025
t 2.33, 2026 t 1.76); survives walk-forward (OOS t 2.90 / 2.17); cost-robust (even
5¢ leaves +0.090/$1, CAGR +21%) because the edge is large; week-clustered
bootstrap 5th-pct +0.088, P(edge≤0)=0; Calmar 5.0, maxDD −9%, Ulcer 0.032.

**Weaknesses (why it's niche, not scalable):**
- **Concentrated in geopolitics** (calm edge +0.251, t 3.84); crypto (+0.060,
  t 0.94) and politics (+0.071, t 1.26) are weak. It is largely a
  "war/escalation underpriced during calm" effect.
- **Decaying:** first half +0.189 (t 3.50) → second half +0.090 (t 1.80). The
  market may be partly arbitraging it (the predicted fate).
- **Winner-concentrated:** top 10 trades = 57% of gross profit (n=128 is small; a
  few escalation events drive it).
- **Capacity:** geopolitical/narrative markets are thin — the deployability
  ceiling, as expected.

**Verdict:** a *real, regime-conditioned, geopolitics-driven* edge — clean risk
profile, survives the audit — but small-sample, concentrated, possibly decaying,
and capacity-limited. A niche overlay, not a scalable strategy. Next: exogenous
regime triggers (VIX/GDELT), time fixed-effects, and a forward paper ledger.

Graph: `regime_strategy.png` (equity calm-vs-not · calibration · per-domain edge).

## Exogenous-regime robustness (resolves the endogeneity caveat)

`experiments/run_regime_exogenous.py` re-tests the effect with a **fully
exogenous** regime indicator — market-wide risk (VIX) and credit stress (BofA US
HY OAS) from the repo datasets, known as-of the decision date — instead of the
home-made surprise intensity `S`. Recent-shock = trailing-45d max VIX above its
median (≈23); calm = below.

**Disruptive tail (price<0.35, n=317):**

| regime (VIX) | n | CE | t |
|---|---:|---:|---:|
| **CALM** (low trailing VIX) | 172 | **+0.098** | 3.08 |
| TURBULENT (recent VIX spike) | 145 | +0.032 | 1.14 |

Same direction, calm ~3× larger — and the **VIX-calm flag is uncorrelated with the
original S-calm flag (agreement 50%, corr 0.00)**. Two *independent* regime
measures producing the same effect is much stronger than either alone; the
endogeneity worry is largely resolved. Geopolitics again carries it (VIX-calm
+0.205 t 3.48 vs turbulent +0.107 t 1.81). The strategy book replicates: VIX-calm
edge +0.127 (t 3.51) vs turbulent +0.062 (t 1.70).

**Remaining honesty:** the *continuous* slope CE~VIX-level is insignificant
(t −0.44) — the effect is **state-like (calm vs recently-shocked), not a smooth
gradient** — and VIX is a global-risk proxy, not a pure geopolitical-conflict
index (GDELT would be the ideal next data source). But the result no longer
depends on the home-made regime measure.

## Domain-matched proxy: the GPR index (the confidence test)

`experiments/run_regime_gpr.py` re-runs the **geopolitics** legs — the domain that
carries the whole effect — against the **Caldara–Iacoviello Geopolitical Risk
(GPR) index** (`datasets/gpr_daily.csv.gz`, daily, news-based, 1985→2026-06). This
is the *right-domain* exogenous regime measure: VIX is financial risk, near-blind
to geopolitical calm, whereas GPR is built to measure exactly this. It also lets us
put numbers on the two open method questions — *is the calm/turbulent split robust,
and is it real or a labelling artifact?*

**No-lookahead construction.** Every regime quantity uses only GPR dated ≤ the
decision date: a trailing-45d GPR mean and a trailing-365d GPR *median* baseline
(both causal — pandas `.rolling` sees only past rows), attached with
`merge_asof(direction="backward")`. **Calm = recent GPR below its own running
1-yr normal** (a moving-average crossover). The baseline is *trailing*, not
all-history, because GPR is structurally elevated in 2023–26 — an absolute 1985
reference would mislabel every recent day "turbulent". The decision price is still
the mid 7d pre-resolution; outcome is settlement.

**Headline — geopolitics tail (price<0.35, n=117):**

| regime (GPR) | n | CE | t |
|---|---:|---:|---:|
| **CALM** (trailing GPR < 1-yr median) | 25 | **+0.372** | 3.80 |
| TURBULENT (recent GPR spike) | 92 | +0.105 | 2.34 |

Difference **+0.266** in the predicted direction — *larger* than either S or VIX
gave. The matched proxy strengthens the effect, exactly as it should if the effect
is genuinely geopolitical.

**The three confidence tests all pass:**

1. **Continuous dose-response (the thing VIX failed):** within geopolitics,
   `dCE/d(GPR) = −0.0018 per GPR-pt, t −2.42`. Higher recent geopolitical risk ⇒
   less underpricing, as a **smooth gradient** — not just an on/off state. VIX's
   continuous slope was t −0.44; the domain-matched index recovers the gradient,
   which is strong evidence the regime variable is measuring the right thing.
2. **Robustness grid** (trailing-mean window 30/45/60/90d × reference 365/730d,
   all causal): **7 of 8 cells positive**, diff range −0.013…+0.312, median +0.232.
   The effect lives at the **30–60d horizon**; it washes out at 90d (calm is a
   relatively short-memory state). One cell (90d/730d) is marginally negative —
   the honest edge of the grid, not the centre.
3. **Placebo test:** reshuffling the calm label 5,000× gives a null centred at 0
   (95% [−0.19, +0.20]); the real +0.266 sits in the tail, **one-sided p = 0.005**
   (two-sided 0.008). The split is not an artifact of arbitrary labelling.

**Caveats that remain.** GPR is geopolitics-specific — applied to the *whole*
disruptive tail it adds nothing (calm vs turb diff +0.004), confirming the effect
is specifically a geopolitical-calm phenomenon and not a generic one. Samples are
still small (calm cell n=25 at baseline) and the price<0.35 / geopolitics segment
was inherited, not re-derived, so forking-paths risk is reduced (pre-specified grid
+ placebo) but not zero.

**Confidence verdict.** Three *independent* regime measures — endogenous surprise
intensity `S`, exogenous VIX, and now the domain-matched GPR index — all show
disruptive-tail underpricing concentrated in calm regimes, and GPR additionally
delivers the **significant continuous gradient** the other two lacked. That moves
the geopolitics result from "candidate" toward **solid**. The remaining honest
limits are sample size and capacity, not the regime definition.

## Per-domain validation with matched proxies — what actually trades

`experiments/run_regime_domain_validation.py` does the disciplined version of "is
this real per domain": each disruptive domain is judged against **its own
exogenous proxy** (not one global index), then put through a strict kill battery
(base t, calm−turb gap, walk-forward OOS, cost @3¢/5¢, week-clustered bootstrap,
placebo regime, winner concentration). A domain is VALIDATED only if it clears
*all* gates; otherwise KILLED.

| domain | matched proxy | calm edge/$1 (t) | OOS edge (t) | boot P≤0 | top1 | verdict |
|---|---|---:|---:|---:|---:|:--|
| **geopolitics** | GPR index | **+0.406 (3.93)** | +0.357 (3.31) | 0.00 | 17% | **VALIDATED** |
| crypto/financial | BTC 30d realized vol | +0.044 (0.69) | +0.072 (0.81) | 0.26 | 36% | killed (insignificant) |
| politics | EPU (daily) | +0.062 (0.38) | +0.082 (0.40) | 0.35 | 100% | killed (6 calm legs, 1 winner) |
| ai/tech | — | — | — | — | — | killed (no tail data) |

**Only geopolitics survives.** Crypto's calm edge points the right way (gap +0.089)
but base t 0.69 and bootstrap P 0.26 — not real. Politics has 6 calm legs and a
single winner (top1 100%). ai/tech has no cheap-tail legs. This is the rigorous
confirmation of the earlier hunch: the effect is a *war/escalation-underpriced-
during-calm* phenomenon, nothing broader. The verdict is written to
`data/polymarket_calibration_snapshot/validated_domains.json`, which drives what
trades live.

## Deployed: the validated strategy (live, on the cron + dashboard)

`experiments/run_validated_regime_paper_update.py` is the forward paper ledger of
the distilled strategy: **buy the cheap (≤0.35) YES leg of a geopolitical market
while geopolitical risk is calm** (GPR trailing-45d mean below its trailing-1yr
median, computed causally from the committed dataset), 10%/bet, hold to settlement.
It reads `validated_domains.json`, so if a domain's verdict changes the live book
follows automatically. It is wired into the hourly `paper_trade.yml` workflow
(validation → ledger → dashboard) and shows up as **"Regime (geo-calm)"** on the
dashboard, with the per-domain kill-test table on the Backtests tab. The regime
gate genuinely fires: when GPR is elevated (as during an active Middle-East
flare-up) the strategy logs no new entries and waits for calm.
