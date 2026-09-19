# Retired strategies — pulled from the paper-trade dashboard

`experiments/generate_dashboard.py` still runs, sizes, and validates every
sleeve in its `REGISTRY`/`DERIVED` lists — `run_paper_kill_battery.py`
imports those lists directly and re-scores all of them on every run, so a
retired sleeve that starts winning forward can still re-qualify. What
changed is display only: `DASHBOARD_SIDS` in `generate_dashboard.py` limits
`data/paper_trade/dashboard.html` to a chosen shortlist. Everything below is
still tracked, just no longer shown on the dashboard day to day.

> ## ⚠️ Correction, 2026-09-19 — the original figures here were wrong
>
> This page previously said the shortlist was "the sleeves that are actually
> making money paper-trading". **That was not true, and the numbers that
> supported it were an artifact of the simulator, not a result.**
>
> `sim()` used to fund each day's entries in **ledger row order** until its
> 30%-of-equity exposure cap filled, then silently drop the rest. Sleeves that
> enter many positions a day were therefore scored on whichever rows the CSV
> happened to list first. For `smart_flow_indep` (~120 entries/day, so ~4 in 5
> candidates dropped) that reported **+39%**, while 60 random shuffles of the
> *same trades* returned a median of **−18%** and never once reached +39%. The
> dashboard was showing the best case of an arbitrary ordering.
>
> `sim()` now allocates **pro-rata** across all of a day's candidates and is
> order-invariant (regression test: `tests/test_dashboard_sim.py::
> test_capped_day_is_invariant_to_ledger_row_order`). Every figure on this page
> has been recomputed under the corrected sim.
>
> **Under the corrected sim, every sleeve in this book is mark-to-market
> negative.** There is no "still making money" group. The shortlist on the
> dashboard is an explicitly chosen set to watch, not a P&L ranking.

## Corrected figures — all sleeves, 2026-09-19

Mark-to-market (realized + unrealized) from a $1,000-per-sleeve, no-leverage
cash sim at a flat $10/trade stake, under the order-invariant allocation.
"Shown" marks the sleeves currently in `DASHBOARD_SIDS`.

| Sleeve | sid | Shown | Trades (open) | Win rate | MTM | Flat-stake return |
|---|---|---|---|---|---|---|
| Macro (Fed cuts) | `macro` | | 0 (1) | — | +$3.74 | +0.4% |
| Smart Flow (indep exp) | `smart_flow_indep` | ✓ | 7788 (827) | 60.9% | −$17.93 | −1.8% |
| SF independent | `smart_flow_indep_independent` | ✓ | 6366 (713) | 60.4% | −$26.34 | −2.6% |
| YES [20–40%] | `midprice_yes_20_40` | ✓ | 301 (1) | 29.9% | −$82.47 | −8.2% |
| YES [10–50%] | `midprice_yes` | | 572 (9) | 31.3% | −$84.90 | −8.5% |
| Smart Flow ∩ band .30–.70 | `smart_flow_combo` | | 4094 (277) | 53.3% | −$159.21 | −15.9% |
| SF cascade | `smart_flow_indep_cascade` | | 1422 (114) | 63.4% | −$211.20 | −21.1% |
| YES [10–20%] | `midprice_yes_10_20` | | 226 (3) | 14.2% | −$222.92 | −22.3% |
| YES [10–30%] | `midprice_yes_10_30` | | 332 (3) | 18.7% | −$258.08 | −25.8% |
| Smart Flow | `smart_flow` | | 8738 (810) | 59.8% | −$281.11 | −28.1% |
| Regime (all-domain, exploratory) | `regime_all` | | 181 (40) | 11.0% | −$326.08 | −32.6% |
| Regime (geo-calm) | `validated_regime` | | 68 (15) | 8.8% | −$403.83 | −40.4% |
| Smart Flow (ROI) | `smart_flow_roi` | | 360 (54) | 43.9% | −$447.40 | −44.7% |
| Combo cell (band ∩ ≤3d) | `smart_flow_combo3d` | | 3104 (90) | 53.3% | −$471.65 | −47.2% |
| Dip-Confirm YES | `dip_confirm` | | 341 (2) | 34.0% | −$529.48 | −52.9% |

`Macro (Fed cuts)` is not a winner — it has **zero resolved trades** and one
small open position, so its +$3.74 is an unrealized mark on a sleeve with no
trading record at all.

`Smart Flow (passive)` (`smart_flow_passive`) is also shown on the dashboard
and is absent from this table because it has no history yet: it is the live
pre-registered successor experiment, registered 2026-09-19. See
`docs/polymarket/SMART_FLOW_PASSIVE.md`.

## Statistical validation is a separate, harsher bar

Every sleeve above — shown ones included — also fails the kill-battery in
`data/paper_trade/sleeve_validation.json` (`verdict: "KILLED"` for all 15
tracked sleeves, `validated: []`; see `experiments/run_paper_kill_battery.py`).
That battery answers "is there provable statistical edge" and currently says no
for everything. The table above is the narrower, purely financial cut.

Note the battery's `trades_per_event` column when reading any of this:
`smart_flow_indep` carries 7,788 trades across only **64 settlement-day
events**. Nominal trade counts in this book overstate independent evidence by
roughly two orders of magnitude.

## Why retire instead of delete

Ledgers, sim code, and kill-battery scoring are untouched — only
`DASHBOARD_SIDS` changed. Deleting the underlying data or REGISTRY/DERIVED
entries would stop `run_paper_kill_battery.py` from ever re-evaluating these
sleeves, which is the one way a currently-losing sleeve could still prove
itself later (regime shift, bug fix in entry logic, etc.). Losing money in
this sim so far doesn't retroactively make the ledger uninteresting — it's
kept as a live record, just off the dashboard people look at day to day.

One sleeve is retired in a stronger sense: `smart_flow_indep` has
`ENTRIES_FROZEN = True` in `posterioralpha/polymarket/smartflow_independence.py`,
because both of its *pre-registered* kill criteria fired. It opens no new
positions; the cron only resolves what is already open. That is a commitment
made before the data arrived, not a reaction to the figures above.

## Re-promoting a retired sleeve

Add its `sid` back to `DASHBOARD_SIDS` in `experiments/generate_dashboard.py`
and regenerate (`python experiments/generate_dashboard.py`). Do this only
once the sleeve has turned mark-to-market positive over a meaningful stretch
of forward (not just recent) data — the numbers above are a point-in-time
snapshot, not a permanent verdict, and will keep moving as more trades in
`data/paper_trade/*.csv` resolve.
