# Pre-resolution timing — is informed timing a *persistent* skill?

> Some wallets buy right before favourable moves. The question that decides
> whether that is exploitable — rather than luck or un-actionable leakage — is
> **persistence**: does a wallet that timed well early keep timing well later? A
> strict split-half test says **yes, but modestly**. The rank correlation of
> early-vs-late lead scores across wallets is **Spearman +0.28**, and the wallets
> in the top half on early timing stay positive on late timing (**+0.042** mean
> forward drift) while the early-poor half goes negative (**−0.026**). So informed
> timing is a partly repeatable trait, not pure noise — but it is weak, and the
> "best timers" leaderboard is contaminated by survivorship artefacts (see below),
> so *standalone* "follow the top timers" is fragile. It corroborates the
> pay-up/conviction tells more than it stands alone.

```bash
python experiments/run_polymarket_timing.py
python experiments/run_polymarket_timing.py --fwd 5
```

## The test

Each wallet's **lead score** is the mean forward 5-day directional drift of its
fills — how reliably its trades *precede* moves in their favour. To separate skill
from luck we split each wallet's fills at its own median trade time, score the
lead in each half, and correlate across wallets. A positive cross-half
correlation is the signature of a repeatable timing skill.

## Result (large universe, 5-day forward drift)

**Persistence (36 wallets with both halves scorable):**

| measure | value |
|---|---|
| early→late lead correlation (Pearson) | +0.73 *(outlier-inflated — see caveat)* |
| early→late lead correlation (**Spearman**, robust) | **+0.28** |
| late lead given top-half-early timers | **+0.0424** |
| late lead given bottom-half-early timers | **−0.0256** |

The rank-based Spearman is the trustworthy read: a **modest positive** persistence.
The actionable split is cleaner than the correlation — being a good timer early
buys you a positive (if small) late lead, and being a poor one early predicts a
negative late lead. Timing is a real, repeatable wallet trait.

## The honest caveats (why "follow the timers" is fragile)

- **Pearson is an artefact.** The top of the lead-score table contains wallets
  with impossible t-stats (one at t≈8,672, lead +0.72, 100% resolution hit) — a
  wallet whose fills all sit in a handful of tokens that resolved Yes, so its
  "forward drift" is near-constant. That is **selected survivorship in the priced
  universe**, not foresight, and it dominates the Pearson. The Spearman/rank view
  neutralises it; trust that.
- **Small and noisy.** Spearman +0.28 over 36 wallets, and the late-lead gap is
  ~7 bps per fill. Real, but thin — a standalone "copy the top timers" book would
  ride a few contaminated names.
- **Impact endogeneity & spread.** Same as `TRADE_QUALITY`: forward drift after a
  fill partly reflects the fill's own impact, and marks ignore spread.

## Where this leaves the behaviour thread

Timing skill *persists weakly*, which is consistent with — and best used as a
confirmation of — the stronger, cleaner signals: the **pay-up follow book**
(`PAYUP_FOLLOW`) already harvests "aggressive, urgent, consensus flow" and
survives out of sample at net Sharpe 0.67. Persistent timing is the *why* behind
that book working (informed wallets really do lead moves repeatably); it is not a
better standalone signal than the flow synthesis itself.
