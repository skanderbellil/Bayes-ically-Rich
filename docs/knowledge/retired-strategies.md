# Retired strategies — pulled from the paper-trade dashboard

`experiments/generate_dashboard.py` still runs, sizes, and validates every
sleeve in its `REGISTRY`/`DERIVED` lists — `run_paper_kill_battery.py`
imports those lists directly and re-scores all of them on every run, so a
retired sleeve that starts winning forward can still re-qualify. What
changed is display only: `DASHBOARD_SIDS` in `generate_dashboard.py` now
limits `data/paper_trade/dashboard.html` to the sleeves that are actually
making money paper-trading (`smart_flow_indep`, its `smart_flow_indep_independent`
filter view, and `midprice_yes_20_40`). Everything below is still tracked,
just no longer shown on the dashboard, because it was losing real
(paper-trade) dollars.

Figures are mark-to-market (realized + unrealized) as of 2026-09-19, from a
$1,000-per-sleeve, no-leverage cash sim at a flat $10/trade stake (the
`retflat`/`mtm` columns `generate_dashboard.py` computes) — the walk-forward
Kelly column (`ret10`) is noisier at low trade counts and mostly reads $0
(no proven edge) for these sleeves, so flat-stake MTM is the more honest
"did this lose money" measure. Registry column is each ledger's dashboard id
(`sid`), for tracing back to `data/paper_trade/*.csv` and `REGISTRY`/`DERIVED`
in `generate_dashboard.py`.

| Sleeve | sid | Trades (open) | Win rate | Realized | MTM | Flat-stake return |
|---|---|---|---|---|---|---|
| Smart Flow | `smart_flow` | 8738 (810) | 59.8% | −$778 | −$776 | −77.6% |
| Smart Flow ∩ band .30–.70 | `smart_flow_combo` | 4094 (277) | 53.3% | −$178 | −$197 | −19.7% |
| Combo cell (band ∩ ≤3d) | `smart_flow_combo3d` | 3104 (90) | 53.3% | −$662 | −$662 | −66.2% |
| SF cascade | `smart_flow_indep_cascade` | 1422 (114) | 63.4% | −$29 | −$33 | −3.3% |
| YES [10–50%] | `midprice_yes` | 572 (9) | 31.3% | −$66 | −$85 | −8.5% |
| Smart Flow (ROI) | `smart_flow_roi` | 360 (54) | 43.9% | −$436 | −$422 | −42.2% |
| Dip-Confirm YES | `dip_confirm` | 341 (2) | 34.0% | −$523 | −$530 | −52.9% |
| YES [10–30%] | `midprice_yes_10_30` | 332 (3) | 18.7% | −$236 | −$258 | −25.8% |
| YES [10–20%] | `midprice_yes_10_20` | 226 (3) | 14.2% | −$200 | −$223 | −22.3% |
| Regime (all-domain, exploratory) | `regime_all` | 181 (40) | 11.0% | −$177 | −$171 | −17.1% |
| Regime (geo-calm) | `validated_regime` | 68 (15) | 8.8% | −$353 | −$404 | −40.4% |
| Macro (Fed cuts) | `macro` | 0 (1) | — | $0 | +$4 | +0.4% |

Every sleeve in this table — kept ones included — also fails the
statistical kill-battery in `data/paper_trade/sleeve_validation.json`
(`verdict: "KILLED"` for all 15 tracked sleeves, `validated: []`; see
`experiments/run_paper_kill_battery.py`). That battery answers "is there
provable statistical edge" and currently says no for everything, including
the sleeves still shown. This retirement list is a narrower, purely
financial cut: these twelve are the ones that lost real paper-trade dollars
outright, on top of failing validation. `Macro (Fed cuts)` is the one
exception — it has zero resolved trades and near-zero MTM (one small open
position), so it isn't a loser by any measure; it's retired here only
because it has no trading activity to show and was dropped in the same
dashboard cleanup.

## Why retire instead of delete

Ledgers, sim code, and kill-battery scoring are untouched — only
`DASHBOARD_SIDS` changed. Deleting the underlying data or REGISTRY/DERIVED
entries would stop `run_paper_kill_battery.py` from ever re-evaluating these
sleeves, which is the one way a currently-losing sleeve could still prove
itself later (regime shift, bug fix in entry logic, etc.). Losing money in
this sim so far doesn't retroactively make the ledger uninteresting — it's
kept as a live record, just off the dashboard people look at day to day.

## Re-promoting a retired sleeve

Add its `sid` back to `DASHBOARD_SIDS` in `experiments/generate_dashboard.py`
and regenerate (`python experiments/generate_dashboard.py`). Do this only
once the sleeve has turned mark-to-market positive over a meaningful stretch
of forward (not just recent) data — the numbers above are a point-in-time
snapshot, not a permanent verdict, and will keep moving as more trades in
`data/paper_trade/*.csv` resolve.
