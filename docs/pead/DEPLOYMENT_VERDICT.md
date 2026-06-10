# Retail Deployment Verdict — Alpaca Cost Analysis

**Question:** after realistic Alpaca retail costs, is any strategy in this repo
robust enough to deploy — and is there a real edge?

**Answer (final, OHLC-resolved): no robustly-tradeable retail edge. A real PEAD anomaly
exists, but as a strategy it is marginal at best, and the exit-losers stop — the
centerpiece of earlier drafts — actively *hurts* once modelled with real intraday fills.**

> ### ✅ DEFINITIVE ASSESSMENT (real daily OHLC; supersedes all sections below)
> The fill uncertainty is now resolved with actual open/high/low data
> (`refetch_ohlc.py` + `pead_ohlc_stops.py`), not bounded:
> - **Fill quality is good** (you were right): a 3% intraday stop fills at *exactly* −3%
>   **67%** of the time, gaps through worse only **9%**, never fires 24%.
> - **But the stop still hurts.** Net of exact fills, every stop is *worse* than no stop:
>   no-stop OOS≥2014 Sharpe **0.55**; 3% **0.05**; 5% **−0.08**; 10% **0.13**. Full-history
>   (2001–2026) all negative.
> - **Why the earlier "+stop triples Sharpe" was an artifact:** the old backtest checked the
>   *closing* price against the stop (rarely below it), but a real intraday stop triggers on
>   the *intraday low* — and **76% of positions touch −3% intraday** at some point in the
>   hold. Cutting them all at −3% forgoes the dips that close green. The stop only looked
>   good because the close-based check almost never fired.
>
> **Net:** the genuinely-tradeable, fill-robust version is **no-stop, hold-to-next-earnings**,
> which is **OOS Sharpe ~0.55 (2014–2026) but negative over the full 2001–2026 history** —
> marginal and regime-dependent, not a deployable alpha. The statistically-significant PEAD
> *anomaly* is real (IC t > 6); a profitable retail *strategy* on Mega+Large with realistic
> execution is not demonstrable here. Honest recommendation: **do not deploy**; if curious,
> paper-trade the no-stop version and watch live IC.

Reproduce with `python experiments/pead_ohlc_stops.py` (after `run_pead.py` + `refetch_ohlc.py`).

> ### ⚠️ Realism correction (supersedes the first version of this doc)
> An earlier version of this analysis reported **+10.3%/yr, Sharpe ~1.5** for the
> plain 21-day market-neutral book. That number is **overstated**: the signal panel
> measures returns from `p_t0` = the *pre-announcement* close, so it captures the
> announcement-day jump (`p_t0 → t+1`) — **which a retail trader cannot trade**, since
> the surprise isn't public until the release. That untradeable gap is **~93%** of the
> Mega+Large LS spread (`ret_t1` LS ≈ `ret_t21` LS). Entering at the t+1 close — the
> first transactable price — the plain 21-day drift is **~0% after costs**. The
> genuinely tradeable edge comes from holding longer (drift peaks ~day 42–55) **and**
> cutting losers with a stop. See §3a. `deploy_cost_analysis.py` still reports the
> p_t0-based (optimistic) numbers and is labelled accordingly.

---

## 1. Alpaca cost model (2026)

Encoded in `posterioralpha/backtest/alpaca_costs.py`:

| Item | Value | Consequence |
|------|-------|-------------|
| Commission (US equities/ETFs) | **$0** | Binding cost is the bid-ask half-spread, not commission |
| Regulatory (SEC + FINRA TAF) | ~0.3 bps, sells only | Negligible |
| Short borrow — ETB (easy-to-borrow) | **$0/yr** | Large/mega-caps + SPY shortable for free |
| Short borrow — HTB (hard-to-borrow) | **not shortable at all** | **Kills any short leg in small/mid/micro caps** |
| Margin interest (leverage > 1×) | 6.25%/yr (non-elite) | A dollar-neutral book at ≤2:1 has **no** debit → no drag |
| PDT rule | **retired 2026-06-04** | No $25k minimum; holding-period strategies unaffected |

Half-spreads (one-way, bps) by tier: ETF 1 · Mega 2 · Large 4 · Mid 10 · Small 25 · Micro 60 · Nano 120.

## 2. What is NOT deployable

- **PEAD long-short (broad / small-cap):** the strongest raw signal is in Mid/Small/Micro
  caps (long-short +9–18%/yr gross), but those are **hard-to-borrow → unshortable on
  Alpaca**. Not deployable as a long-short.
- **Long-only top-SUE basket vs SPY:** robust **+5.9%/yr net selection alpha** over the
  equal-weight universe (t=6.8, 77% of quarters), but it is *equal-weight*, so it trails
  cap-weighted SPY on raw return by ~2.6%/yr in the mega-tech era (Sharpe 0.76 vs 0.71).
  A risk-improver, not a return-beater.

## 3. Upper-bound (p_t0 entry) — OPTIMISTIC, includes the untradeable gap

Long top-quintile SUE / short bottom-quintile SUE, **within Mega+Large caps only** (all
easy-to-borrow), dollar-neutral, rebalanced quarterly, each leg charged a round-trip
spread. **These returns are measured from the pre-announcement close, so they include
the untradeable announcement-day jump (see the realism correction above). Treat as an
upper bound, not a deployable result.** Net of all Alpaca costs:

| Window | Ann | Sharpe | MaxDD | Quarters |
|--------|-----|--------|-------|----------|
| 1998–2004 | +4.0% | 0.44 | −10.9% | 13 |
| 2005–2009 | +16.3% | 2.31 | −1.9% | 20 |
| 2010–2014 | +8.2% | 1.35 | −5.6% | 20 |
| 2015–2019 | +14.2% | 2.49 | −1.0% | 20 |
| 2020–2026 | +8.9% | 1.21 | −5.7% | 26 |
| **Modern (2010–2026)** | **+10.3%** | **1.57** | ~−6% | 66 |
| Full (1998–2026) | +10.6% | 1.48 | −10.9% | 99 |

**Out-of-sample walk-forward** (train → test): +10.8% (t=6.2) · +11.1% (t=5.2) ·
+9.4% (t=3.6) — robust in every split, 70–78% of quarters positive.

**Why it's robust:**
- Positive in **every** 5-year sub-period; weakest era was 1998–2004, so the result is
  *not* an artifact of the dot-com period.
- **Survivorship bias is conservative here:** delisted bottom-SUE names (the best shorts)
  are absent from the live-only panel, so the realised short-leg profit — and the edge —
  is if anything *understated*.
- Well-diversified: ~26 names per leg (min ~21) in the modern era — not concentrated.
- Correlation to SPY **+0.17** → a genuine diversifier, not a levered-beta proxy.

## 3a. The actually-deployable edge — t+1 entry + exit-losers stop

Rebuilding each position's daily path from the cached prices and entering at the **t+1
close** (`experiments/pead_dynamic_exit.py`), net of Alpaca costs. Numbers below are on
the **representative 259-name Mega+Large universe** (~196 names/quarter):

| Tradeable strategy (Mega+Large, market-neutral) | Full Sharpe | OOS≥2014 Ann | OOS Sharpe | OOS MaxDD |
|---|---|---|---|---|
| Fixed 21d (the naive book) | −0.40 | −0.8% | −0.16 | −21.2% |
| Fixed 42d | −0.25 | +2.4% | 0.42 | −13.0% |
| Fixed 63d | −0.26 | +3.2% | 0.54 | −8.1% |
| Stop sweep, hold-to-next-earnings — **3%** | 1.06 | +9.8% | **1.49** | −4.6% |
| … **5%** | 0.67 | +9.2% | **1.25** | −6.1% |
| … **6%** | 0.53 | +8.1% | 1.08 | −9.4% |
| … **8%** | 0.40 | +8.4% | 1.01 | −11.4% |
| HMM/regime hold (calm 63d / storm 21d) | −0.18 | +2.2% | 0.38 | −14.5% |

**Breadth did not help — it gave a more sober estimate.** Going from 150 → 259 Mega+Large
names (the deployable universe is more fully covered) *lowered* the central Sharpe: the
6% stop went from OOS 1.18 to 1.08, and the no-stop drift from 0.90 to 0.54. The extra
large-cap names carry noisier raw PEAD, so the stop does even more of the work. The
earlier, smaller sample was modestly optimistic; ~1.0–1.25 (at a 5–6% stop) is the honest
deployable range, with tight 3–4% stops scoring higher on paper but leaning on exact-fill
assumptions.

Findings:

- **Exit losers (stop-loss) is the real edge — and it's the only exit worth timing.**
  A per-name stop turns the near-zero naive book (no-stop OOS Sharpe 0.54) into the
  deployable strategy (OOS Sharpe ~1.0–1.5 depending on tightness) and cuts drawdown to
  ~−5 to −6%. It works because ~42% of early losers keep deteriorating — capping them
  while letting the ~79% of winners run is a genuine asymmetry. Stops help at *every*
  level (3–20% all beat no-stop); see `pead_stop_sweep.py`.
- **The hold cap is NOT a cherry-picked number.** OOS Sharpe is a *plateau* across the
  hold cap (25–63 days), not a spike — the exact number doesn't matter. The natural,
  non-arbitrary cap is **"hold until the next earnings report" (~63 trading days = one
  quarter)** — the SUE signal is about *this* event and goes stale at the next one. That
  is a signal-decay horizon, not a tuned parameter.
- **The holding period is already dynamic per position.** Under "6% stop, hold-to-next-
  earnings," **58% of positions stop out early** (median hold 41d, **IQR 13–63d**) — the
  losers exit when *they* signal it; only the winners ride to the cap. Nothing is pinned
  to a fixed horizon.
- **Timing the *winner* exit doesn't help** (this is what the HMM/trailing/regime ideas
  tried). A horizon-free trailing stop gives OOS Sharpe 0.55–0.93; the per-position HMM
  0.71; a market-regime hold 0.57 — all *below* "hold + cut losers." PEAD drift is a slow,
  noisy grind, so any rule that exits winners on a pullback cuts the drift short. The data
  rewards letting winners run to signal decay and only acting on losers.
- **Vol-scaling the stop is a wash** (`pead_volscaled_stop.py`). Setting each name's stop
  to `k × its own monthly volatility` (to equalise stop-out probability) does not beat a
  fixed % stop: at a matched ~6% average level, OOS Sharpe 1.19 vs 1.20 and a slightly
  *worse* drawdown (−12.3% vs −11.2%); full-sample it is worse (0.71 vs 0.87). Within
  Mega+Large the vol dispersion is modest and diversifies across ~26 names, and a fixed %
  stop already equalises each name's *dollar* loss — which is what controls portfolio
  drawdown. Vol-scaling actually widens stops on the high-vol names you most want to cut.

**Meta-finding:** every elaboration of the exit rule — HMM, market regime, trailing stop,
per-name vol-scaling — ties or underperforms a plain fixed % stop with a hold-to-next-
earnings cap. The edge is real but modest, and is *not* improvable by clever exit
engineering; the robust, simplest rule is the right one to deploy.
- **Neither HMM variant beats the stop.** Two were tried (`pead_hmm_drift.py`):
  *(a)* a market-wide vol-regime hold (OOS Sharpe 0.57), and *(b)* a **per-position**
  2-state Gaussian HMM that filters each name's own path to decide if the drift is
  "confirmed" → extend to day 63, else exit (best config decision-day 15: OOS Sharpe
  0.71, better drawdown than fixed-hold but still < the stop's 1.18). The daily P&L path
  is too noisy for an HMM to confirm drift in ~15 days, and the stop's value is mostly in
  hard-cutting tail losers — which the HMM's extend-winners logic doesn't replicate. The
  exit decision that matters is a simple per-position level stop, not a latent-state model.

See `results/pead/hmm_vs_spy.png` for equity + drawdown curves vs SPY, including the
recommended 60/40 SPY-core + MN-sleeve blend (SPY-like return at a much shallower drawdown).

Mechanics (how it runs live): the strategy is **continuous and self-renewing**. The
−6% stop is measured **from entry** (not a trailing/drawdown stop — the trailing version
was tested and lost). The hold cap is the **next earnings report**, which *refreshes* the
signal: at each report you re-rank and roll (same side), flip, or exit. A continuous-roll
backtest that charges turnover only on real trades (`pead_continuous_roll.py`) lands
within +0.2%/yr of the batch number — because PEAD turnover is inherently high (only ~14%
of position-quarters are rolls; a name's surprise rank rarely persists), so the batch
"full round-trip per quarter" cost was already nearly right. Live stats: median holding
**40 trading days (IQR 13–63)**, ~60% of positions stop out early, ~3–4 round-trips/yr per
slot — cheap in liquid large-caps (~8 bps round-trip).

Caveat on the stop: the backtest exits exactly at −stop%, which assumes no gap-through.
Earnings names can gap, so realised stop fills may be a little worse; size patiently and
treat OOS Sharpe ~1.15 as indicative.

## 3b. The entry signal — raw SUE wins; nothing improves it

Tested whether combining/replacing the entry signal sharpens the market-neutral book
(`pead_entry_signals.py`, Mega+Large, t+1 entry, 6% stop, hold-to-next-earnings, net):

| Entry signal | OOS Ann | OOS Sharpe | OOS MaxDD |
|--------------|---------|-----------|-----------|
| **Raw SUE (analyst Surprise %)** | **+11.1%** | **1.25** | −11.2% |
| Standardized SUE (vs firm's own surprise history) | +8.9% | 1.16 | −7.9% |
| Announcement reaction (Brandt et al.) | +7.1% | 0.79 | −11.7% |
| Reaction 0.5 + SUE 0.5 | +9.8% | 1.10 | −11.2% |
| Momentum only | +9.3% | 0.83 | −9.8% |
| SUE confirmed by momentum sign | +7.9% | 0.79 | −13.0% |
| SUE within low-vol half | +2.7% | 0.36 | −22.2% |
| Composite SUE + momentum (any weight) | +9–10% | 0.85–1.00 | −9 to −11% |

**Raw SUE beats every alternative and every combination.** Momentum, the
announcement-day reaction, firm-standardized SUE, a low-vol filter, and all blends
dilute or underperform it in the large-cap universe. (Standardized SUE ties on a
risk-adjusted basis with a shallower drawdown — the one defensible alternative — but
does not raise the Sharpe.)

**Meta-finding (entry + exit).** Across *both* sides of the trade, every elaboration —
HMM/regime/trailing/vol-scaled exits, and momentum/reaction/standardized/composite
entries — ties or underperforms the simplest rule: **rank on raw SUE, enter at t+1, cut
losers with a fixed % stop, hold to the next earnings report.** The edge is real but
modest, and is not enhanceable with the signals available here. The one lever that *did*
help was **universe breadth** (going 184 → 414 names raised the Sharpe by cutting
per-leg noise) — pointing to coverage, not signal engineering, as the remaining upside.

## 4. Recommended deployment — SPY core + market-neutral sleeve

> The frontier below uses the §3 (p_t0) sleeve and is therefore an **upper bound**.
> With the realistic §3a tradeable sleeve (OOS Sharpe ~1.15, ~+8%/yr, ~0 beta) the
> *shape* is the same — a low-correlation diversifier that lifts a SPY portfolio's
> Sharpe and cuts its drawdown — but the gains are more modest than shown here.

Because the sleeve earns ~SPY-like returns at ~0 beta, blending it with SPY traces an
efficient frontier at **constant ~11.5% return** (2005–2024, unlevered, no margin drag):

| Capital split | Ann | Sharpe | MaxDD |
|---------------|-----|--------|-------|
| 100% SPY | +11.7% | 0.71 | −46% |
| 80/20 SPY/MN | +11.6% | 0.86 | −37% |
| 60/40 SPY/MN | +11.6% | 1.08 | −26% |
| 40/60 SPY/MN | +11.5% | 1.38 | −14% |
| 100% MN sleeve | +11.3% | **1.68** | **−6%** |

Pick the point on the frontier matching your drawdown tolerance. 40/60 roughly doubles
SPY's Sharpe and more than halves its drawdown for the same return.

## 5. Implementation notes

- **Account:** margin account (Reg-T) for the short leg; ≥$2k equity. No PDT constraint.
- **Universe:** US Mega + Large caps with analyst-estimate coverage (ETB-verify each name
  via Alpaca's `shortable`/`easy_to_borrow` asset flags before shorting).
- **Signal:** SUE = Yahoo `Surprise(%)` (analyst-estimate surprise), per `pead.signals`.
- **Construction:** each earnings season, rank SUE within tier; long top quintile / short
  bottom quintile, equal-weight, dollar-neutral; hold ~21 trading days (entered t+1 after
  the announcement). In practice run it as **rolling** 21-day holds entered at each
  announcement rather than discrete quarterly buckets (the backtest's quarterly grouping
  is a modelling simplification).
- **Costs already modelled:** spread per tier, $0 commission, $0 ETB borrow. No leverage
  beyond the 2:1 the dollar-neutral book implies → no margin interest.

## 6. Honest caveats / residual risk

- **Sample:** 414 names, stratified — a research-grade panel, not the full CRSP universe.
  Larger coverage would tighten estimates (the jump from 184→414 names *raised* the Sharpe
  1.11→1.48, which is the right direction).
- **Execution slippage** beyond the modelled half-spread (e.g. trading the close, earnings-day
  gaps) is not captured; size orders patiently.
- **Borrow availability** can change intraday even for normally-ETB names; the book must
  skip any name flagged HTB at execution.
- **Crowding / decay:** PEAD is a well-known anomaly; the modern-era Sharpe (1.57) is lower
  than 2005–2009/2015–2019 peaks, consistent with gradual decay. Monitor live IC.
- The quarterly-bucket backtest ignores intra-quarter compounding of overlapping holds; treat
  the Sharpe as indicative, not exact.

**Bottom line:** a retail trader on Alpaca can deploy the market-neutral Mega+Large PEAD
sleeve — commission-free, $0 borrow, beta-neutral — **provided it is run realistically:
enter at t+1, hold ~42 days, and cut losers with a 6–8% stop.** So run, it delivers OOS
~+8%/yr at Sharpe ~1.15 with a ~−6% drawdown (§3a). The naive 21-day book is ~zero once
the untradeable announcement gap is excluded. Blend the sleeve with an SPY core to dial
in a chosen risk level. This is the honest deployable edge.

## 7. Other branch ideas tested (so this isn't re-litigated)

Mined from the prior two days' work and stress-tested against the same Alpaca costs:

| Idea (source branch) | Verdict |
|----------------------|---------|
| **Beta-hedged market-neutral residual** (`market-neutral-alpha-strategy`) | **Adopted** — the key unlock: hedge with shortable ETB names, not HTB small-caps. Became the deployable edge above. |
| **"Beat the EW-survivor universe, not SPY"** (`pead` `factor_lowturnover`) | **Adopted as methodology.** Confirms SUE is the one signal clearing the survivorship bar; their low-vol/momentum factors did not. |
| **Hurst-bull ×1.5 SPY timing** (`fix-lever-scaling`) | **Negative on return.** Honest IS/OOS (tune 2005–14, test 2015–24) with margin cost: OOS +9.0%/yr Sharpe 1.07 MaxDD −15% vs SPY +14.8%/0.79/−34%. Higher Sharpe + lower DD but **−5.8%/yr return** — a risk-reducer, not the "beats SPY on every metric" the branch claimed. Doesn't add a return edge the MN sleeve doesn't already provide. |
| **Vol-targeting the sleeve** (`pead` `vol_targeting`) | **Promising lever.** Scaling the MN sleeve to an 8% causal vol target lifted Sharpe further in spot checks; left as a tuning knob, not yet hardened. |
| **Signal-strength / top-decile thresholds** (`pead` `signal_filters`) | Concentrates the edge per-name; quintile used here for diversification. A knob, not a regime change. |

The recurring repo-wide pattern: in the 2015–2024 bull, **defensive/timing overlays improve
Sharpe and cut drawdown but trail SPY on raw return.** The market-neutral PEAD sleeve is
different — it earns an *absolute, beta-neutral* return, which is why it is the one
deployable edge rather than just a lower-beta SPY.
