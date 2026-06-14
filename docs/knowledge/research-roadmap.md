# Research Roadmap — Applying the Knowledge Base to PosteriorAlpha

> Companion to `systematic-equity-strategies.md` (external corpus) and
> `alpha-mining-commit-log.md` (the program's arcs). Maps the knowledge
> onto the framework and ranks next experiments. Written 2026-06-12 against
> the `claude/automated-alpha-mining` branch.

## 1. Where the knowledge base meets the framework

| KB source | Status in this repo |
|---|---|
| López de Prado — DSR, purged windows, bootstrap | Already first-class: `mining/validation.py` runs the full gauntlet on every mined candidate. |
| Harvey et al. — trials accounting | Enforced inside the miner; NOT yet applied retroactively to hand-built families (BOCPD-AMR v1→v4, vote-layer variants). `validation/robustness.py::dsr_report` now exists for that. |
| Grinold & Kahn — IC before backtest | Was PEAD-only (`pead/fama_macbeth.py`); now generic via `mining/ic.py`. Use it as the gate BEFORE a family enters the miner: a family with no rank IC shouldn't consume gauntlet trials. |
| Newfound — rebalance timing luck | Was absent everywhere. `validation/robustness.py::rebalance_offset_dispersion` + `rebal_anchor` in `backtest/amr.py`. First result: §6 below. |
| Man AHL — vol targeting | Implemented (plain + regime-adaptive cap in `backtest/amr.py`); the governor studies independently confirmed "vol management on levered barbells is real". |
| AQR — value/momentum blending; QMJ conditioning | Momentum families exist; value/quality need fundamentals data — open gap (fundamentals are also what PEAD's SUE pipeline already fetches; reuse it). |
| HXZ — survivorship hygiene | Known and flagged (equity universes are current-membership). A point-in-time universe remains the single biggest data upgrade available. |
| Ilmanen — style completeness | Defensive ✓ (low-vol, vol targeting), momentum ✓ (families), carry ✗, value ✗. |

## 2. Novel ideas (ranked)

### Idea 1 — Per-stock regime-gated momentum — STARTED
`stock_regime_momentum` family: each stock's momentum shrunk by its OWN
BOCPD regime age (`ctx.erl_panel`), vs. the existing market-level gate.
Daniel–Moskowitz crash logic applied cross-sectionally. First IC results: §5.
Next: IC with gate applied only at extreme freshness (ERL < 21); test the
gate on the short leg; if IC survives, let the miner search the family and
see what the gauntlet says.

### Idea 2 — `regime_age` as a standalone defensive factor
Price-only quality proxy. First IC pass inconclusive (§5) — parked unless a
different universe (ETFs, where BOCPD regimes are cleaner) revives it.

### Idea 3 — Factor momentum across mined finalists (Ehsani–Linnainmaa)
The miner's leaderboard is a panel of factor returns; time them by their own
trailing 12-month performance. Cheap meta-strategy on existing output.

**OpenAP variant — TESTED, NULL vs benchmark.** On the 212-predictor panel
(live 1997→2024) the Ehsani-Linnainmaa fact replicates — XS winners Sharpe
0.83 (t 4.4), TS winners 1.01 (t 5.4), 12m spread +9.5%/yr (t 2.6), all
sub-period stable — but the spread is ~1.9× beta to the EW-ALL benchmark
with alpha +0.9%/yr (t 0.25), and the pre-declared 50/50 vol-matched combo
lowers the Sharpe (1.09 vs 1.29). Factor momentum on this panel is
time-varying zoo exposure, not a new return stream; hold the zoo, don't
time it. Still open: the original Idea-3 target (the miner's own
leaderboard, where the EW benchmark is much weaker).
`run_factor_momentum.py`.

### Idea 4 — Retroactive DSR on the hand-built families — DONE, split verdict
Ran `dsr_report` (excess returns, rf 4%) over both flagship families, with a
DSR-vs-assumed-trials sensitivity curve since nominal counts undercount the
true search. **5-ETF v1→v4 arc (9 variants, 2016-2024): FAILS** — best is
`bocpd_amr_v4` at DSR 0.91 (bar: 0.95), 0.88 at n=80; v3 ≡ v4 and v2 < v1,
so the version ladder was noise. **Champion-stack ladder on QQQ (7 variants,
2011→): PASSES** — DSR ≈ 1.00 across the board, robust to n=80 and to a
14× wider trial dispersion. Consequence: retire the 5-ETF allocation arc
from promotion language; the QQQ vote stack carries the program.
`run_retroactive_dsr.py`.

### Idea 5 — IC gate inside the miner loop
Wire `mining/ic.py` in as a cheap pre-filter: candidates whose 21d rank IC
t-stat < 1 on the training window are rejected before portfolio construction
— saves gauntlet budget and reduces effective trials.

### Idea 6 — Value/quality via the PEAD fundamentals pipe
PEAD already fetches earnings data; SUE's cousins (gross profitability,
accruals, net issuance — the HXZ survivors) are one fetch away. Closes the
Ilmanen style gap with infrastructure that already exists.

### Idea 7 — Alpha half-life (zoo age structure) — TESTED, NULL
McLean-Pontiff decay as an allocation rule: long YOUNG factors (1-5y since
publication), short OLD (>15y), on the OpenAP panel. The decay fact
replicates (50% post-pub haircut, M-P report ~58%) but the event-time
profile is a STEP at publication followed by a 25-year plateau at ~4%/yr —
age beyond publication carries no information, YOUNG−OLD is dead
(−0.7%/yr, t −0.3). Keep the byproduct: EW of all published factors runs
Sharpe 1.21 gross (t 5.9, maxDD −6%) and is sub-period stable — the zoo's
value is diversification, not its age structure. `run_alpha_halflife.py`.

### Idea 8 — Champion attribution — DONE, verdict in
FF5+UMD Newey-West regression (2011-10 → 2026-04): full stack
α = +6.1%/yr (t 2.34), retail QLD+UUP wrapper +6.9% (t 2.56), vs the QQQ
B&H control's +2.6% (t 2.07). Average market beta ≈ 1.0 (the 2× cap is
rarely fully spent), anti-value tilt (HML −0.36), R² 0.81. Read: the
timing machinery adds ~+3.5-4.3pp/yr beyond static growth beta —
classically significant, below Harvey's t > 3 bar. The thread deserves
continued investment but not promotion language stronger than "fragile
alpha". `run_factor_attribution.py`.

### Idea 9 — Factor crowding gauges on the zoo — TESTED, NULL
Three causal gauges (36m avg pairwise corr, PC1 share, 12m dispersion) on
the OpenAP panel as warning dials for the EW-zoo holding. The zoo
*de*-correlated as it grew (PC1 share 0.44→0.28 since the 90s); the
correlation gauges disagree on sign within eras (neither t>2); the only
borderline signal is dispersion→forward-EW (IC +0.14, t 2.2 post-2004) —
an opportunity-set effect, not a crowding alarm. No dial to build. Gross
panel can't see cost/capacity crowding, so this null does not clear live
risk. `run_factor_crowding.py`.

### Idea 10 — Meta-factors / cluster structure of the zoo — TESTED, ONE SURVIVOR
The factor-of-factors sweep beyond momentum (user prompt 2026-06-13).
Clusters are real and nowcastable (K=6, 36m corr, median ARI 0.75) but
cluster-balanced weighting decays (α t 4.9 → 2.0 across 2004) and cluster
rotation is dead. LT reversal and low-vol at the factor level: dead
(low-vol is backwards). **Factor-level seasonality survives everything**:
Keloharju same-calendar-month signal gives α vs EW-ALL of +9.5%/yr
(t 3.6, β −0.12), is not momentum (corr −0.20), holds in both eras, and
the 50/50 vol-matched combo lifts the EW holder's Sharpe 1.29 → 1.42 with
a smaller maxDD in both sub-periods. First clear t>3 positive on the
OpenAP panel, with KLN's published priors behind it.
`run_zoo_metafactors.py`.

**OOS update (`run_seasonality_oos.py`):** the JKP cross-region test
confirms direction, not magnitude. True-OOS regions all positive and
half-stable (Sharpe 0.37-0.42, t 2.1-2.5) but ~1/6 the OpenAP size; the
US theme-level panel is flat (t 0.76 / 69y). Seasonality is a breadth
effect — 4-per-leg theme baskets can't carry what 60-per-leg predictor
baskets can.

**MATCHED-BREADTH OOS + cost model — RESOLVED (`run_seasonality_matched_oos.py`,
2026-06-13):** built the JKP 153 individual-factor panel × 4 regions
(new `build_jkp_individual` / `load_jkp_individual`, cached as
`datasets/jkp_factors_individual.csv.gz`) and reran at breadth that rivals
OpenAP, closing the "is it breadth-illusory?" question and the cost
question together. Verdict:
  • **Geographic OOS confirms the signal is real, not breadth-illusory:**
    developed α +3.4%/yr (t 3.9), world_ex_us +3.4% (t 4.3; β 0.02,
    corr_mom −0.01 → a nearly pure orthogonal stream), emerging +2.3%
    (t 3.1); momentum-orthogonal, sub-period stable in both halves.
  • **But the OpenAP +9.5% magnitude was an artifact:** the true
    cross-region effect is ~3%/yr gross (a third), and **usa on JKP does
    not replicate** (t(α) 1.78, β 1.18, dead post-2004) — the OpenAP US
    headline was panel-specific.
  • **Costs cap it:** ~12×/yr turnover, break-even ≈25-30 bps per unit
    one-way (net Sharpe ~0.4-0.5 at 10 bps, ~0 at 30 bps). Survives
    institutional frictions, dies at retail.
Net: seasonality downgrades from "the headline positive" to "real, modest,
momentum-orthogonal, cost-fragile." Its cleanest form (world_ex_us, β≈0,
corr_mom≈0) is a candidate market-neutral overlay sleeve (cf. Idea 15) —
but only if turnover is engineered down. Remaining open: low-turnover
construction (membership buffering/hysteresis) to lift the cost break-even;
 within-cluster seasonality.

**LOW-TURNOVER construction — RESOLVED (`run_seasonality_lowturn.py`,
2026-06-13):** tested whether the ~12-25×/yr turnover is structural or
reducible. Three builds (hard tercile, continuous z-score, hysteresis-
buffered). **Buffering is strictly better than the baseline** — higher
gross Sharpe (developed 0.65→0.73, world_ex_us 0.70→0.78) AND ~27% lower
turnover, because the hysteresis band stops trading boundary flicker — but
the cost break-even only rises from ~15 to ~18 bps. Continuous cuts
turnover ~45% but dilutes the signal (gross ~1.8%/yr). **All three die by
20-30 bps**: turnover is largely STRUCTURAL — the seasonality cross-section
reconstitutes every calendar month by construction, so the book must churn.
Final verdict on seasonality: a real, momentum-orthogonal *institutional*
market-neutral overlay sleeve, best implemented buffered (~18 bps
break-even); not retail-viable at any construction. Seasonality arc closed.

### Idea 11 — Factor-level BOCPD: regime age in the zoo's cross-section —
### TESTED, MECHANISM YES / EDGE NO
The genuinely novel one (no published precedent in the KB): per-factor
online Bayesian changepoint posteriors, regime age (ERL) as a meta-signal.
The Daniel-Moskowitz logic transfers — stale-regime factors beat fresh
(α t 2.2 vs −0.4), regime-conditioned momentum orders exactly as predicted
in both eras, and momentum return is near-monotone in ERL quintile
(Sharpe 0.66 freshest → 1.03 stalest). But no construction clears the EW
benchmark or t>2 alpha (spread t 1.5, conditioned mom t 1.25), and the
cross-sectional changepoint dial is dead. Mechanism worth keeping in the
KB; edge not present at monthly/gross resolution. Possible revival: daily
factor returns (JKP daily?) where changepoints are sharper, or ERL as a
*risk* model (position sizing) rather than a return signal.
`run_factor_regimes.py`.

### Idea 12 — Absolute-return blends on the zoo — TESTED, LEVERAGE WINS
The objective flip (total return, beta indifferent). VM 50/50 at the
factor level fails its premise: MOM/VAL corr +0.21 (≈0 post-2004), factor
value too weak to blend (Sharpe 0.13), so the blend trails momentum alone.
VMS (with seasonality) reaches 0.76; every stack that includes the EW
carry dilutes it. Winner: causally vol-targeted EW zoo @10% (cap 3×) —
+10.9%/yr gross, Sharpe 1.37, maxDD −15%, era-stable. The panel's final
word in both objectives: selection adds nothing; leverage converts the
zoo's Sharpe into return. `run_zoo_absolute.py`.

### Idea 13 — Long-only tracking of the levered zoo — TESTED, UNSPANNED
Can a reachable book (long-only, no leverage, +cash) replicate Idea-12's
levered EW stream? No. Min-TE on FF12 industries and 27 ETFs, in-sample
floor + walk-forward: the optimizer parks ~90-97% in cash, in-sample TE
floor (7-8%) barely beats the target's own 8% vol, and equities are
*negatively* correlated with the stream (SPY −0.42). The levered zoo is
the orthogonal complement of a long-only portfolio — addable via
market-neutral vehicles, not rebuildable from beta. `run_zoo_tracking.py`.

### Idea 14 — The reachable frontier — TESTED, CONCENTRATION WON THE REGIME
The practical landing: best long-only, no-leverage, NET book from 20 liquid
ETFs. Over 2011-2026 nothing beat SPY risk-adjusted (SPY Sharpe 0.90 on
realized RF, 60/40 0.89, every diversified book 0.62-0.72). Diversification
delivered drawdown insurance (maxDD −12-14% vs −24%, 2022 −8% vs −18%) but
not Sharpe. All reachable books are negatively correlated with the levered
zoo (−0.31 to −0.41), so none of them harvest it — the zoo is reachable
only through genuine market-neutral vehicles. Whole sub-arc conclusion:
the absolute-return king is orthogonal to long-only space; in that space,
over this regime, concentration paid and diversification was insurance.
Caveat: one bull-dominated regime + 2022. `run_reachable_frontier.py`.

### Idea 15 — Zoo as a market-neutral overlay — TESTED, THE ARC'S PAYOFF
Relax long-only and the orthogonality pays. Core = FF total market
(Sharpe 0.50), overlay = levered EW zoo, corr −0.61. The diversifier
hurdle ρ·S_core is negative, so the overlay survives an ~11.3%/yr cost
break-even. Implementable walk-forward (36m tangency dose, cap 50%, no
future info) at a hard 6% cost: equity core 0.46 / +9.1% / −50% maxDD →
0.71 / +11.6% / −42% maxDD — better on all three axes, out of sample.
Resolution of the whole absolute-return arc: the zoo's value IS its
market-neutrality, which long-only access (Ideas 13-14) destroys; with
shorting/alts access a 30-50% overlay is the reachable upgrade.
`run_zoo_overlay.py`.

### Idea 16 — Factor lead-lag network: do some factors lead others? —
### TESTED, MECHANISM YES (cross-region) / EDGE NO
User prompt 2026-06-13: "construction of factor matters, maybe some factors
lead others." A genuinely novel directed-network question (no explicit KB
precedent), distinct from own-momentum (Idea 3) and the XS winners effect.
Tested at the JKP 13-theme level × 4 regions (dimensionality + cross-region
discipline against the 13×13 = 156-test overfit surface).
  • **The pre-declared fast→slow hypothesis is the robust win.** Market-
    priced FAST factors (momentum, short-term reversal, low-risk) lead
    accounting-priced SLOW factors (investment, accruals, profitability,
    value) in ALL FOUR regions: lag-1 corr +0.07 to +0.14 (t 1.6-4.0,
    three of four >2), while the reverse direction slow→fast is always ~0
    (|t|<1.6). The asymmetry rules out a shared contemporaneous factor —
    this is genuine directed lead-lag, and construction-speed is the
    mechanism, exactly as hypothesized.
  • **Leadership network exists but is ex-US-stable only.** Net-leadingness
    ranking agrees across developed/emerging/world_ex_us (Spearman
    0.65-0.95) but the US is idiosyncratic (0.10-0.21). Robust pattern:
    **size leads**, profitability/quality/profit_growth lag.
  • **Not tradable.** The walk-forward cross-factor predictor has real
    gross predictability (t 2.9-4.4 in 3 regions) but dies at 20 bps,
    does not beat own-momentum cross-region, and nothing beats simply
    holding EW-ALL (Sharpe 1.1-1.5). Same structural verdict as the arc:
    timing the zoo loses to holding it.
Net: confirms the user's intuition at the mechanism level (fast factors
lead slow, cross-region) — keep as a KB fact and a candidate *forecasting/
risk* input — but no net-of-cost return edge at monthly/gross resolution.
Possible revival: daily factor returns (sharper lead-lag), or the fast→slow
signal as a conditioner on the Idea-15 overlay rather than a standalone
sleeve. `run_factor_leadlag.py`.

### Idea 17 — Retail-scale IC improvement, net of Alpaca costs —
### LOOP RESULT: found on ETFs, decayed, best as an overlay sleeve
User loop 2026-06-13 ("until next IC improvement at retail scale, alpaca
costs"). Grounded in the repo's Alpaca cost model (commission-free; binding
cost = half-spread: 1bp ETF / 2bp mega / 4bp large; mid-and-smaller not
shortable). Two arenas:
  • **Single-name large-caps (490 names) — NULL.** Three levers (z-score
    composite, sector-neutralization, multi-horizon 21/63/126d) all confirm
    cross-sectional stock-selection IC ≈ 0 (momentum IC 0.001-0.002,
    |t|<0.2). HXZ confirmed: survivorship-biased large-caps over a bull
    decade are the wrong arena. Incumbent long-only momentum nets Sharpe
    1.15 but at ~0 IC (tail-driven). `run_retail_signal_lab.py`.
  • **Cross-asset ETFs (235 names) — FOUND.** 12-1 momentum IC 0.058
    (t 2.40) at 21d, 0.089 (t 2.11) at 63d — first t>2 of the loop, net of
    1bp cost. Matches the program's prior ("cross-asset ETF portfolios =
    the one real robust result"). Caveats: DECAYED (IC 0.09/t2.3 ≤2017 →
    0.023/t0.78 >2017 — post-publication crowding); standalone long-only
    loses to SPY (0.78 vs 0.97). Durable use = the market-neutral L/S
    sleeve (corr −0.04 to SPY, +10.3%/yr net, Sharpe 0.52), a retail
    diversifier that feeds the Idea-15 overlay. `run_retail_etf_lab.py`.
Recurring loop lesson: signal COMBINATION lost to the single best signal in
BOTH arenas — blending dilutes when components differ in quality. Loop
goal met (a real net-of-cost IC improvement exists at retail scale); the
honest framing is "modest, decayed, overlay-grade," not a headline winner.

### Idea 18 — The orthogonal-alpha ensemble — BREAKTHROUGH (the capstone)
User push 2026-06-13 ("find a breakthrough"). The whole program kept finding
mutually-uncorrelated alpha streams and judging each in isolation, where
each looks modest. This assembles them — the synthesis nobody ran.
Three sleeves on a common 2011-2024 monthly timeline:
  core   FF total market (equity beta)
  zoo    levered EW OpenAP zoo @10% vol (Idea 15), net of a cost grid
  etfmom decile L/S on 235 ETFs (Idea 17), net of 1bp Alpaca
**They are mutually orthogonal AND stably so OOS**: corr(core,zoo) −0.44,
corr(core,etf) +0.01, corr(zoo,etf) +0.03 — holding in both halves. So the
fundamental law's √N diversification of INDEPENDENT bets applies, and it
delivers a step-change, not an increment:
  • risk-parity (walk-forward inv-vol, 10% target): **Sharpe 0.93 → 1.70**
    net of cost, maxDD −20% (< equity −25%), **+14% in 2022** (equity −20%),
    sub-periods 2.16 / 1.27 (both beat equity's 1.17 / 0.80).
  • naive equal-sum is WORSE (1.28, −41% DD) — it over-weights the high-vol
    ETF sleeve; risk-parity is the correct construction (methodological
    point worth keeping).
**Decisive bound (zoo-cost sensitivity):** the ensemble beats equity until
~12%/yr drag on the zoo leg — Sharpe 2.10 @0%, 1.71 @4%, 1.31 @8%, 0.91
@12%. Realistic institutional market-neutral frictions are 2-6%, so the
result is robust across the whole plausible range, not knife-edge.
**Honest caveats:** (1) the zoo is the workhorse (core+zoo alone already
1.56; the third sleeve adds 1.56→1.70) and is the least-implementable —
needs market-neutral access, gross/survivorship-flattered OpenAP, an upper
bound; (2) the RETAIL-only ensemble (core+etfmom, no zoo) is just 0.96 — the
breakthrough REQUIRES the institutional zoo sleeve; (3) both alpha sleeves
decayed post-2018, so the recent half is 1.27 not 2.16 (still a wide margin
over equity). Net: the breakthrough is not a new signal — it is the proof
that the program's orthogonal alphas STACK to a step-change Sharpe exactly
as the fundamental law guarantees, robust to realistic costs, conditional
on market-neutral access. The deployable next step is to source a real
liquid-alts / market-neutral vehicle for the zoo leg.
`run_orthogonal_ensemble.py`.

**RETAIL-DEPLOYABLE (`run_retail_alts_ensemble.py`, 2026-06-13):** the breakthrough needs a market-neutral sleeve; the academic zoo gets it by shorting hundreds of stocks (impossible on Alpaca). Tested two retail proxies from real ETF prices (returns net of expense ratios). **Beta-hedged factor ETFs FAIL** (long MTUM/VLUE/QUAL/USMV/SIZE − β·SPY): residual beta +0.20, killed by the post-2019 factor winter. **Liquid alts WORK**: an EW basket of BTAL (anti-beta) + MNA (merger arb) + DBMF (managed futures), held LONG (the funds short internally — the retail investor never shorts a stock), keeps corr −0.42 to SPY STABLY across both halves (−0.38/−0.47), low standalone Sharpe 0.26 (a diversifier, the zoo's character). Result: risk-parity{SPY, ETF-momentum L/S, liquid-alts} → **Sharpe 1.42 vs SPY 0.99**, maxDD −14% (vs −24%), **+15% in 2022** (SPY −18%), and sub-period STABLE (1.48/1.42) — MORE robust than the institutional ensemble (2.16→1.27 decay) because managed-futures/anti-beta don't fade like academic factors. Everything trades on Alpaca. This is the program's most deployable result: a 3-sleeve retail portfolio that lifts equity Sharpe ~43% net of cost. `run_retail_alts_ensemble.py`.

**LONG-ONLY + DAILY + NO-LEVERAGE reality check (`run_retail_longonly_daily.py`, 2026-06-13):** the hard retail constraints. Liquid alts survive (bought long), but shorting and leverage do not — so the ETF-momentum L/S becomes a long-only 200d-MA multi-asset trend sleeve, and there's no vol-targeting up. **Decisive finding: the Idea-18 Sharpe jump (0.93→1.70) was substantially a leverage+shorting phenomenon.** Long-only no-leverage CANNOT beat SPY's Sharpe (best ≈0.59-0.63 vs 0.62 full-sample 2009-2026). What the orthogonal diversification DOES buy, long-only, is drawdown and tail protection: 80/20 SPY/trend is a near-Pareto win (maxDD −34%→−27% at ~SPY Sharpe); 60/20/20 with alts cuts the 2022 tail −18%→−11% and maxDD to −22% for ~4pp/yr of give-up. And in the post-2019 regime every diversified book beats SPY on Sharpe (0.70 vs 0.64) — pure equity only dominated the 2009-2019 megabull. Naive inverse-vol weighting fails (over-weights the −0.17-Sharpe alts); equity-dominant fixed weights are correct. Deployable long-only recommendation: 70/20/10 equity/trend/alts — SPY-like return, a third less drawdown, half the crash tail, zero shorting, zero leverage, all Alpaca ETFs. `run_retail_longonly_daily.py`.

**LIQUID ALTS = the long-only zoo (`run_liquid_alts_basket.py`, 2026-06-13):** the user's insight — each liquid-alt ETF is one packaged orthogonal sleeve you buy LONG, so a basket is a retail long-only zoo. Confirmed, with the winner being MANAGED FUTURES (DBMF/KMLM/CTA): the only alt class uncorrelated to BOTH stocks (−0.10..−0.20) AND bonds (−0.34..−0.49), with crisis convexity (+0.8% in worst-decile equity months, +22-24% in 2022). Adding a MF sled to SPY (long-only, no leverage, 2020-26) monotonically improves Sharpe 0.68→0.78, maxDD −24%→−14%, 2022 tail −18%→−7% — and for 60/40, 0.47→0.63 with maxDD −21%→−7%. Unlike BTAL (bleeds), MF have positive carry + negative corr + convexity. Honest counterweight: full-cycle (WTMF 2012-26, incl. the 2011-2019 MF drought) shows MF slightly LOWER Sharpe (0.78→0.76) while still cutting drawdown — the recent window flatters them. Net: managed futures are the retail, no-shorting, buy-on-Alpaca version of the Idea-18 zoo overlay — crisis insurance with positive carry, sized ~10-15%. Best long-only book to date: equity core + ~15% managed futures (a cleaner orthogonal sleeve than the 200d-MA trend, which whipsaws). `run_liquid_alts_basket.py`.

**'BETTER SAUCE' — leveraged equity + managed futures (`run_levered_alts.py`, 2026-06-13):** the diversified books smooth SPY but the user wants to beat QQQ on RETURN. Long-only leverage = leveraged ETFs (QLD/SSO/UPRO) hedged by the crisis-convex managed-futures sleeve. **QLD 60% + MF 40% beats QQQ on both axes: CAGR +24.3% vs +21%, Sharpe 0.84 vs 0.82, at maxDD −35% ≈ QQQ −33%** (2020-26). The MF cushion QLD's standalone −61% crash to −35% while 2× leverage lifts return; monthly rebalancing harvests the 2022 crash. QLD 70/MF 30 = +26.5%/−42%. Mechanism is honest: MF make leverage SURVIVABLE (risk-adjusted ≈ QQQ, scaled up), not free alpha. Heavy caveats: short window flatters both the tech bull and the 2022 MF bonanza; QLD full-history maxDD −81% (2008/2022) and the MF hedge only exists from 2019 so a 2008-with-MF-failing scenario is untested; leveraged daily-reset decay in choppy tape. This is the program's highest-RETURN retail book — an aggressive growth sleeve sized to risk tolerance, the natural top of the long-only menu: managed futures protection scales from a 10-15% insurance tilt (run_liquid_alts_basket) up to a 40% leverage-enabler here. `run_levered_alts.py`.

**ZOO OF LIQUID ALTS + LEADERSHIP ROTATION (`run_alt_zoo_rotation.py`, 2026-06-13) — the session's synthesis & the wall:** 25 liquid alts across 11 categories, each a packaged orthogonal sleeve; rotate into the leaders by trailing-6m momentum (Idea-3/16 logic on a tradable alt universe). (1) Rotation WORKS: top-8 momentum Sharpe 0.47 / +7.8%/yr vs static EW zoo 0.30 / +5.5%, +18% in 2022 vs +1% — timing the alt zoo beats holding it. (2) But even rotated, the alt sleeve is a low-Sharpe (0.35-0.47) diversifier — can't out-RETURN QQQ. (3) QLD 60 + rotated-alt 40 beats QQQ on return (+28.3% vs +24.7%) and the rotated zoo makes leverage survivable (maxDD −37% vs QLD-alone −59%), but Sharpe 0.97 < QQQ 1.06. THE WALL (definitive across the session's 6+ return-hunt experiments): over 2016-26 nothing in the long-only liquid-alt + leverage toolkit beats QQQ's Sharpe 1.06 — a generational tech-bull number. You can buy more RETURN with leverage (rotated alts keep it holdable), not more risk-adjusted return. The Idea-18 ensemble beat *equity* (0.93) only via leverage+shorting+the gross institutional zoo; long-only liquid alts have Sharpes too low (0.3-0.5) to lift QQQ. Honest menu endpoint: match QQQ return at less drawdown (QLD50+alt50, −31% vs −32%) or beat its return at more (QLD60-70+alt, −37/−43%); its Sharpe is the ceiling. `run_alt_zoo_rotation.py`.

**UNIFIED ZOO — leveraged equity AS A ZOO MEMBER (`run_unified_zoo_rotation.py`, 2026-06-14):** the user's capstone idea — stop treating equity as a fixed core; put SPY/QQQ/QLD/TQQQ INTO the rotation zoo and let dual momentum (200d trend + 6m relative strength) pick everything, so the book is levered-long in bull trends and flees to managed futures in busts. Tested with vol-targeting too. Clean NEGATIVE: the leveraged rotation captures QQQ-like return (+19.9%) and nails the 2022 SLOW bear (−4% vs QQQ −33% — it really did rotate to MF), but Sharpe 0.63 < QQQ 0.86 at maxDD −54% vs −33%, because monthly momentum exits FAST crashes late (held leverage into COVID-2020) and leveraged ETFs decay in whipsaws; vol-targeting can't fix it (vol estimate lags). DEFINITIVE WALL across the session's 8 return experiments: no long-only retail toolkit — diversification, liquid alts, leadership rotation, leverage, or adaptive rotation over a unified zoo — beats QQQ's risk-adjusted return over 2016-26. The structural reasons: liquid alts are low-Sharpe (0.3-0.5) diversifiers; monthly signals lag fast crashes; leveraged ETFs decay; QQQ was a generational run. The only thing that would break it is a genuine equity-SELECTION alpha with Sharpe near QQQ's, additive to it — which needs point-in-time fundamental stock data the repo lacks (the standing HXZ/Idea-6 gap). `run_unified_zoo_rotation.py`.

**WALL BROKEN — DAILY + LEADING vol signal (`run_daily_leading_signal.py`, 2026-06-14):** the user's two corrections to the failed monthly rotation: (1) rebalance DAILY, (2) use a LEADING signal (vol spikes at stress ONSET, before momentum/trend confirms). De-risk leveraged equity (QLD) into managed-futures-or-cash on the signal, daily, net 10bp/switch. Just moving the lagging 200d-MA to daily lifts Sharpe 0.63->0.82. The winner is fast REALIZED VOL (20d, threshold 1.3-1.5x 1y median): CAGR ~30-32% vs QQQ 19%, Sharpe ~0.9-0.98 vs 0.75, maxDD -42% vs QLD-alone -64% — robust across the central window/threshold grid and BOTH sub-periods (Sh<=18 0.79-0.93, Sh>18 0.99-1.05), and it holds with plain CASH as the safe asset (not reliant on the recent MF funds). Mechanism: leveraged ETFs decay MOST in high vol, so exiting on the vol spike dodges both the decay AND the crash; daily speed catches the fast crashes monthly missed (COVID close-up: the VIX-TS leading signal got +3%/-21% vs QQQ -29% and the lagging trend -43%). VIX-TS alone whipsaws (0.75); realized vol is the clean leading signal. THIS REVISES THE WALL: monthly momentum couldn't beat QQQ, but daily + a leading vol signal DOES — the first risk-adjusted QQQ-beater of the session's return-hunt. Honest caveats: still -42% maxDD (2x leverage, ~2x QQQ's drawdown for ~1.6x the return); 2011-26 has COVID/2022 in-sample (mitigated by parameter + sub-period robustness); leveraged daily-reset decay remains in choppy high-vol grinds. This is the program's best return-AND-risk-adjusted retail book. `run_daily_leading_signal.py`.

**PURE VOL MANAGEMENT — more return without estimating returns (`run_vol_managed.py`, 2026-06-14):** the user's principle — returns are noise, so strip out return estimation entirely; the only input is VOLATILITY (predictable, it clusters) and the compounding math delivers via VARIANCE DRAIN: g ~ mu - sigma^2/2. Continuous exposure = clip(target_vol/EWMA_vol(QQQ), 0, 1) in QLD, rest in MF/cash, daily. Quantifies leverage decay cleanly: QLD bleeds -4.4%/yr to variance drain (9x QQQ's -0.5%). Vol-managed QLD @30% compounds at GEO +25.7%/yr vs QQQ +19.0% at the SAME -35% maxDD; @25% beats QQQ's return at LESS drawdown (-30%), Sharpe ~0.80. This is the PRINCIPLE behind every leverage result this session: you cannot forecast returns, but you can forecast (and stabilise) volatility, and the geometric-return math converts stable vol into compound growth - the AMR philosophy (no mu estimation) + Man-AHL vol targeting + the variance-drain identity, unified. Continuous targeting is the purest form (parameter-light, no thresholds); the binary fast-vol signal trades a threshold for higher Sharpe (0.9-0.98). `run_vol_managed.py`.

## 3. Standing hygiene rules (KB §5, adopted)

1. IC analysis before any backtest (`mining/ic.py` is the gate).
2. Every variant is a trial → family-level `dsr_report` before claiming
   version-over-version progress.
3. Weekly strategies report `rebalance_offset_dispersion`.
4. Survivorship caveat on all current-membership universes.
5. Costs ≥ 5 bps × turnover everywhere; the miner's evaluator is already
   cost-aware and lagged.

## 4. Architecture note

The cross-sectional additions live in `mining/` (ic lab, per-stock families)
and `validation/` (robustness) — stages 2 and 4 of the pipeline. The
per-stock ERL panel is cached on `SignalContext` (first call ~1s/name).

## 4b. Factor data enrichment (2026-06-13)

The KB's free data sources are now bundled (built by
`experiments/build_factor_data.py`, loaders in `posterioralpha.data`):

| dataset | contents | unblocks |
|---|---|---|
| `ff_factors_monthly.csv` / `ff_factors_daily.csv.gz` | US Mkt-RF, SMB, HML, RMW, CMA, RF, Mom (1963→, daily too) | Idea: FF5+UMD alpha attribution of the champion stack / barbell (the "Buffett's Alpha" regression) on daily returns |
| `ff_europe_monthly.csv` | Europe 5F + WML (1990→) | European-universe attribution |
| `ff_industry12_monthly.csv` | 12 VW industry portfolios (1926→) | cheap cross-sectional test assets |
| `jkp_factors.csv.gz` | 13 JKP themes × usa/developed/emerging/world_ex_us, monthly vw_cap (1926→) | cross-region OOS validation of any signal family |
| `jkp_factors_individual.csv.gz` | 153 individual JKP characteristic factors × 4 regions, premium-oriented, monthly (1926→) | matched-breadth (rivals OpenAP's 212) cross-region OOS — resolved the seasonality magnitude question |
| `openap_ls_returns.csv.gz` + `openap_signal_doc.csv` | 212 published predictors' monthly L/S returns + doc table (165 'clear' predictors) | factor momentum (Idea 3), factor crowding/PCA, decay studies — without rebuilding 200 signals |

Sanity-checked on load: FF equity premium 0.6%/mo at 16.2% ann vol; all
series stored as decimals.

## 5. First IC results (2026-06-12, 94 names, 2016-04 → 2026-04)

Monthly Spearman rank IC vs. forward returns, overlap-corrected t-stats;
current-membership S&P-100 → survivorship-biased. Regenerate via
`python experiments/run_signal_lab.py`.

| signal | IC 21d (t) | IC 63d (t) | IC 126d (t) | hit 21d | Q5−Q1 ann. |
|---|---|---|---|---|---|
| momentum_12_1 | 0.024 (1.21) | 0.031 (0.95) | 0.037 (0.78) | 0.56 | +3.8% |
| reversal_1m | 0.004 (0.20) | 0.003 (0.10) | −0.005 (−0.12) | 0.52 | +2.3% |
| low_vol_1y | −0.032 (−1.20) | −0.050 (−1.09) | −0.055 (−0.84) | 0.42 | −14.0% |
| regime_age | −0.013 (−1.05) | −0.002 (−0.11) | 0.016 (0.53) | 0.45 | −2.5% |
| stock_regime_momentum | 0.021 (1.10) | 0.026 (0.87) | 0.027 (0.60) | 0.58 | +2.8% |

Honest read: **nothing passes t > 2 on this sample** — the lab doing its
job. ~107 monthly observations with IC vol ≈ 0.2 cannot resolve a true IC
of 0.03 (needs ~15 years); mega-caps are the lowest-breadth, most efficient
corner. Momentum has the right shape (IC grows with horizon). The per-stock
gate trades IC magnitude for consistency (higher hit rate, lower IC vol than
raw momentum). Low-vol's strongly negative IC is the survivorship bias made
visible (the index filter kept the high-vol winners). Trials ledger: 5
signals × 3 horizons on one sample.

## 6. First timing-luck check (core AMR, 5 ETFs, 2005 → 2026)

`rebalance_offset_dispersion(run_amr_backtest, strategy="amr")`:

| anchor | CAGR | Sharpe | MaxDD |
|---|---|---|---|
| W-MON | 0.096 | 0.545 | −0.199 |
| W-TUE | 0.098 | 0.565 | −0.234 |
| W-WED | 0.096 | 0.551 | −0.224 |
| W-THU | 0.098 | 0.563 | −0.226 |
| W-FRI | 0.095 | 0.543 | −0.234 |
| **range** | **0.003** | **0.022** | **0.035** |

Verdict: the core AMR engine is robust to the anchor (0.3 pp CAGR range is
noise-level), and the historically-used W-FRI was the *worst* anchor, so
published results were never flattered by it. Open: same check on
BOCPD-AMR v2–v4, whose regime signals could interact with the anchor more.
