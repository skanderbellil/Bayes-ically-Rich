# Mid-price band strategy — hostile validation audit

> **Verdict in one line:** a *real but regime-young* edge (favorite–longshot
> inversion: mid-priced underdogs are underpriced), broad across categories and
> trades, with **no lookahead** — but it lives almost entirely in 2025–26, dies
> past ~3¢ of cost, and its capacity is unmeasured. Pilot-worthy, not size-worthy.

Audited book: `run_midprice_full_backtest.py --n-markets 2000` (the "both" book),
908 entries, 2023-05 → 2026-06 resolution. Per-trade record in
`data/paper_trade/midprice_full_backtest_trades.csv`.

## What the +454% headline hides

- **It is not "YES contracts."** 379 `yes` / 257 `no` / **272 multi-outcome
  sports & team legs**. The book is "buy the mispriced underdog," mostly in
  **sports moneylines and binary event markets** — a favorite–longshot
  *inversion*, not generic mispricing.
- **There is no fair-value model.** The code buys *every* in-band leg; the
  "underpriced vs estimated fair value" framing overstates it. "Signal strength"
  reduces to cheapness.
- **It is a 2025–26 phenomenon.** Resolution-year counts: 2023:1, 2024:99,
  2025:276, 2026:531. ~96% of profit is 2025–26.

## Where the edge lives (category)

| category | n | win | edge/$1 | t | % of profit |
|---|---:|---:|---:|---:|---:|
| other-binary | 432 | 0.34 | +0.045 | 2.01 | 48% |
| sports/multi | 272 | 0.45 | +0.071 | 2.34 | 32% |
| crypto/price | 66 | 0.41 | **+0.120** | 2.03 | 15% |
| politics | 123 | 0.33 | +0.044 | 1.10 | 6% |
| macro | 15 | 0.20 | +0.032 | 0.35 | −2% |

Three categories independently clear ~t≈2; strongest per-$1 in crypto/price and
sports underdogs; **weak in politics, absent/negative in macro** (scheduled, calm,
well-arbitraged). Not a single-category artifact.

## The 12 tests

1. **Time stability** — 2026 = 56% of profit (**flag: >50%**), 2025 = 40%, 2024
   marginal (PF 1.16), 2023 nil. Sharpe 1.8 (2025) / 1.6 (2026).
1b. **First vs second half** — edge +0.054 (t 2.46) → **+0.061 (t 2.76)**. *Not*
   decaying; the market is not arbitraging it away yet. Caveat: both halves sit
   inside the 2025–26 regime.
2. **Walk-forward** — train ≤2024 edge +0.0095 (**t 0.21 — no edge in 2024**);
   OOS 2025+ edge +0.064 (t 3.83, +388%). The edge appeared in 2025; it is
   regime-young. Biggest yellow flag.
3. **Price buckets** — win rate monotonic with price (0.19→0.50, sane). Edge
   lumpy (0.064/0.127/0.032/0.070/0.040), sweet spot 0.15–0.20, sag 0.20–0.30;
   positive everywhere.
4. **Entry timing** — edge grows with horizon: 20+ days +0.104 (t 2.29) vs 0–2
   days +0.047 (t 1.05). Alpha comes from entering early.
5. **Cost stress** — 1¢:+63% / 2¢:+47% / 3¢:+34% / 5¢:+12% (Sharpe 0.49) /
   10¢:−14%. **Survives to ~2–3¢, dies by 5¢.**
6. **Liquidity** — *unresolved:* record has lifetime market volume (median $6.8M)
   but no L2 depth. High volume ⇒ capacity exists; bucket depth 2–10 days out is
   the real unknown. Biggest data gap.
7. **Bootstrap** — iid 5th-pct +0.032; **resolution-week-clustered** 5th-pct
   +0.028; **P(edge≤0)=0** in both. MC order-reshuffle final +422%/+452%/+486%.
   Robust, not order-dependent.
8. **Winner concentration** — **top 10 = 9%, top 20 = 17% of gross profit.**
   Broad-based; not a lottery. Strongest point in its favor.
9. **Leakage** — ✅ no post-resolution info (entry price ≥2d before resolution;
   outcome only settles), ✅ no future prices, ✅ no fair-value leak (no model).
   ⚠️ **selection bias**: universe = top-volume *closed* markets only, so the
   edge is proven only on mega-volume resolved markets (generalization untested;
   1,200→2,000 markets *raised* return, mildly reassuring). ⚠️ **look-elsewhere**:
   1 of ~11 strategies explored here — multiple comparisons inflate any single t.
10. **Sizing** — equal/fixed: CAGR 39%, Sharpe 1.43, DD −18%. 1% risk: 63%,
    Sharpe 1.38, DD −21%. **Kelly is suicidal** (DD −79% to −89%). Flat only.
11. **Risk (1% book)** — CAGR 63%, Sharpe 1.38, Sortino 1.34, Calmar 2.97, maxDD
    −21%, Ulcer 0.063, PF 1.39, skew +1.6, ES95 −1.00 (every loss total).
12. **Failure analysis** — 571/908 (63%) lose; the 20 worst are **all entries at
    ~0.50** (top of band) — coin-flip games + a few geopolitics. Losses cluster
    where edge is thinnest (0.40–0.50); trimming the band to <0.40 would cut them.

## Yes/No decomposition

The entire edge is on the **YES** side: yes-only +0.100/$1 (t 4.59) vs no-only
−0.007/$1 (t −0.28). Buying mid-priced *No* (fading favorites) is dead weight.
Concentrating on the Yes leg raises per-$1 edge and significance.

## Scorecard

| dimension | score | note |
|---|---|---|
| Statistical credibility | 6.5/10 | t 3.7, survives clustered bootstrap & both halves — but one regime |
| Economic significance | 6/10 | +5.8¢/contract; dies past 3¢ cost |
| Robustness | 5/10 | stable across halves/bootstrap; cost-fragile, regime-limited, lumpy |
| Capacity | 4/10 | high $-volume but L2 depth unmeasured |
| Production readiness | 3/10 | needs ≤2¢ execution, selection/fair-value layer, forward ledger |

## Answers

1. **Edge real?** Probably, within the 2025–26 high-volume regime (~65–70%).
2. **P(overfit)?** Classic curve-fit low (no fitted params); P(+454% materially
   overstates live edge via selection + look-elsewhere + one regime) ~50–60%;
   P(zero edge) ~25–30%.
3. **Max deployable?** Unknown without L2 depth; rough $50k–$250k before 3¢
   slippage halves CAGR. Capacity, not signal, binds.
4. **Single most-informative next test?** A **forward real-time paper ledger** —
   kills the "efficient-right-after-discovery" and selection concerns at once.
5. **Trade own money?** Small live pilot yes; meaningful size no, until a forward
   ledger and a real slippage/depth model exist.

## Open research question

Regime-dependent calibration (tails underpriced after calm, overpriced after a
shock) is plausible and **suggestively consistent** here: edge concentrates in
crypto/price and disruptive event binaries, and is absent in calm macro. Direct
test: condition each market's edge on a recent-disruption indicator (underlying
realized vol / time since last shock-resolution in its category).
