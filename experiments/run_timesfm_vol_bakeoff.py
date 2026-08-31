"""Does a time-series foundation model forecast volatility better than we do?

Evaluation of Google's **TimesFM** against this repo's own volatility models on
the only forecasting job the house view says is winnable: second moments.

Target — for each asset and each month-end t, the log realized volatility of
days t+1 .. t+21. Every forecaster sees returns only up to t (see
`research/tsfm.py`; `tests/test_tsfm_vol.py` pins causality).

Contenders
    rw       carry today's trailing 21d realized vol forward (near-unit-root, hard to beat)
    ewma94   RiskMetrics EWMA variance
    har      HAR-RV (Corsi 2009) in logs, expanding purged refit — the academic benchmark
    bayes    `pead/bayesvol.py` — Gamma posterior on precision, adaptive discount
    timesfm  TimesFM 2.5 (200M, Apache-2.0 weights), zero-shot on the log-RV series

Scoring — RMSE in log-vol space (transformation-neutral, so no model is rewarded
for sitting higher) plus QLIKE on the variance scale, the loss a vol-targeting
book actually cares about. Both after the *same* causal expanding-window level
debiasing. Model-vs-model significance is a Diebold-Mariano test on
cross-sectionally averaged loss differentials (so the 20 ETFs sharing a date do
not count as 20 independent observations), Newey-West at lag 1.

⚠️ Pretraining leakage: TimesFM's weights were fixed at a checkpoint date, and
nothing in this repo's gauntlet can see contamination that happened during
*pretraining*. The full-sample column is therefore not a clean out-of-sample
read for TimesFM alone; `--oos-start` (default = the 2.5 release) reports the
slice that is, at the cost of a much smaller n.

    python experiments/run_timesfm_vol_bakeoff.py
    python experiments/run_timesfm_vol_bakeoff.py --no-timesfm      # baselines only
"""
import argparse
import time

import _bootstrap  # noqa: F401  (adds repo root to sys.path)
import numpy as np
import pandas as pd

from posterioralpha.research.tsfm import (
    DEFAULT_CHECKPOINT,
    TimesFMVolForecaster,
    baseline_paths,
    build_contexts,
    causal_bias_correction,
    log_returns,
    newey_west_t,
    qlike,
    realized_target,
)

PRICES = _bootstrap.ROOT / "datasets" / "etf_universe_prices.csv.gz"
OUT_DIR = _bootstrap.ROOT / "results"

# Liquid cross-asset basket, all with full history back to 2010: equity beta,
# size, style, international, duration, credit, and the loudest sectors.
DEFAULT_TICKERS = [
    "SPY", "QQQ", "IWM", "MDY", "EFA", "EEM", "TLT", "LQD", "HYG", "GLD",
    "XLE", "XLF", "XLK", "XLV", "XLU", "XLP", "XLY", "XLI", "IYR", "GDX",
]

# TimesFM 2.5 was released 2025-09-15; anything before it could in principle
# sit inside the pretraining corpus.
CHECKPOINT_RELEASE = "2025-09-15"


def month_end_positions(index: pd.DatetimeIndex) -> np.ndarray:
    """Positional index of the last trading day of each calendar month."""
    s = pd.Series(np.arange(len(index)), index=index)
    return s.resample("ME").last().dropna().astype(int).to_numpy()


def build_panel(prices: pd.DataFrame, tickers, horizon: int, min_history: int,
                start: str) -> pd.DataFrame:
    """One row per (date, ticker, model) with the forecast and the realized target."""
    rows = []
    for tic in tickers:
        px = prices[tic].dropna()
        if len(px) < min_history + horizon + 50:
            print(f"  skip {tic}: only {len(px)} observations")
            continue
        r = log_returns(px)
        dates = r.index
        n = len(r)

        evals = month_end_positions(dates)
        evals = evals[(evals >= min_history) & (evals + horizon < n)]
        evals = evals[dates[evals] >= pd.Timestamp(start)]
        if len(evals) == 0:
            continue

        rv = r.to_numpy()
        target = realized_target(rv, horizon)
        paths = baseline_paths(rv, horizon, evals)

        for name, path in paths.items():
            for t in evals:
                rows.append((dates[t], tic, name, t, path[t], target[t]))

    panel = pd.DataFrame(rows, columns=["date", "ticker", "model", "t", "pred", "true"])
    return panel


def add_timesfm(panel: pd.DataFrame, prices: pd.DataFrame, horizon: int,
                context: int, batch_size: int, checkpoint: str) -> pd.DataFrame:
    """Run TimesFM over exactly the (ticker, date) grid the baselines used."""
    grid = panel[panel["model"] == "rw"][["date", "ticker", "t", "true"]].copy()
    forecaster = TimesFMVolForecaster(checkpoint=checkpoint, max_context=context,
                                      max_horizon=max(32, horizon), batch_size=batch_size)
    print(f"\nTimesFM: {len(grid)} forecasts, context {context}, horizon {horizon}")
    t0 = time.time()
    forecaster.load()
    print(f"  checkpoint loaded in {time.time() - t0:.1f}s")

    out = []
    for tic, grp in grid.groupby("ticker", sort=False):
        rv = log_returns(prices[tic].dropna()).to_numpy()
        idxs = grp["t"].to_numpy()
        contexts = build_contexts(rv, idxs, horizon, context)
        preds = forecaster.forecast_batch(contexts, horizon)
        g = grp.copy()
        g["pred"] = preds
        g["model"] = "timesfm"
        out.append(g)
        print(f"  {tic}: {len(idxs)} forecasts  [{time.time() - t0:.0f}s elapsed]")

    tf = pd.concat(out, ignore_index=True)[["date", "ticker", "model", "t", "pred", "true"]]
    return pd.concat([panel, tf], ignore_index=True)


def score(panel: pd.DataFrame, label: str) -> pd.DataFrame:
    """Loss table over one slice of evaluation dates."""
    rows = []
    for model, g in panel.groupby("model", sort=False):
        g = g[np.isfinite(g["pred_adj"]) & np.isfinite(g["true"])]
        if len(g) == 0:
            continue
        err = g["true"] - g["pred_adj"]
        rmse = float(np.sqrt(np.mean(err ** 2)))
        ql = float(np.mean(qlike(np.exp(g["true"]), np.exp(g["pred_adj"]))))
        # Mincer-Zarnowitz: regress realized on forecast; slope 1 = unbiased
        X = np.column_stack([np.ones(len(g)), g["pred_adj"]])
        beta, *_ = np.linalg.lstsq(X, g["true"].to_numpy(), rcond=None)
        r2 = 1.0 - float(np.mean(err ** 2)) / float(np.var(g["true"]))
        rows.append(dict(slice=label, model=model, n=len(g), rmse_log=rmse,
                         qlike=ql, r2=r2, mz_slope=float(beta[1])))
    out = pd.DataFrame(rows).sort_values("qlike").reset_index(drop=True)
    return out


def dm_table(panel: pd.DataFrame, benchmark: str, label: str) -> pd.DataFrame:
    """Diebold-Mariano of every model against `benchmark`, on QLIKE.

    Loss differentials are averaged cross-sectionally per date first: the 20
    ETFs share market-wide vol shocks, so treating them as independent
    observations would inflate the t-stats several-fold.
    """
    wide = panel.pivot_table(index=["date", "ticker"], columns="model", values="loss")
    if benchmark not in wide.columns:
        return pd.DataFrame()
    rows = []
    for model in wide.columns:
        if model == benchmark:
            continue
        d = (wide[model] - wide[benchmark]).groupby(level="date").mean().dropna()
        rows.append(dict(slice=label, model=model, vs=benchmark,
                         mean_delta=float(d.mean()), t_stat=newey_west_t(d.to_numpy()),
                         n_dates=len(d)))
    return pd.DataFrame(rows).sort_values("mean_delta").reset_index(drop=True)


def diagnostics(panel: pd.DataFrame) -> None:
    """Where do the errors sit, and does any model carry orthogonal information?

    The headline table ranks models; these three answer *why*. Conditioning on
    how far vol actually moved separates "tracks the level" from "calls the
    turn", and a 50/50 log-space blend against `bayes` is the cheapest possible
    test of whether a candidate knows anything the incumbent does not — if the
    blend does not beat `bayes` alone, the candidate is redundant.
    """
    d = panel.copy()
    d["err"] = d["true"] - d["pred_adj"]  # positive => model under-predicted vol
    models = sorted(d["model"].unique())

    err = d.pivot_table(index=["date", "ticker"], columns="model", values="err")
    rw = d[d["model"] == "rw"].set_index(["date", "ticker"])
    move = (rw["true"] - rw["pred"]).reindex(err.index)
    err["regime"] = pd.qcut(move, 5, labels=["1 vol crush", "2", "3 flat", "4", "5 vol spike"])

    print("\nMean signed error by realized-vol move quintile  (+ = under-predicted):")
    print(err.groupby("regime", observed=True)[models].mean()
          .to_string(float_format=lambda v: f"{v:8.3f}"))

    print("\nRMSE (log-vol) by the same quintiles:")
    print(err.groupby("regime", observed=True)[models]
          .apply(lambda g: np.sqrt((g ** 2).mean()))
          .to_string(float_format=lambda v: f"{v:8.3f}"))

    loss = d.pivot_table(index=["date", "ticker"], columns="model", values="loss")
    share = loss.idxmin(axis=1).value_counts(normalize=True).mul(100)
    print("\nShare of (date, ticker) cells won outright on QLIKE (%):")
    print(share.to_string(float_format=lambda v: f"{v:6.1f}"))

    if "bayes" not in loss.columns:
        return
    truth = d[d["model"] == "bayes"].set_index(["date", "ticker"])["true"]
    base = d[d["model"] == "bayes"].set_index(["date", "ticker"])["pred_adj"]

    def _ql(pred: pd.Series) -> float:
        return float(np.mean(qlike(np.exp(truth), np.exp(pred))))

    print("\nDoes anything add information on top of bayes? (50/50 log-space blend)")
    print(f"  bayes alone           {_ql(base):9.4f}")
    for other in [m for m in models if m != "bayes"]:
        o = d[d["model"] == other].set_index(["date", "ticker"])["pred_adj"]
        joined = pd.concat([base, o], axis=1).dropna()
        blend = joined.mean(axis=1)
        tag = "" if _ql(blend.reindex(base.index).fillna(base)) < _ql(base) else "   (redundant)"
        print(f"  bayes + {other:<14}{_ql(blend.reindex(base.index).fillna(base)):9.4f}{tag}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    ap.add_argument("--horizon", type=int, default=21, help="forecast horizon in trading days")
    ap.add_argument("--context", type=int, default=1024, help="TimesFM context length")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--min-history", type=int, default=756, help="days of returns before the first eval date")
    ap.add_argument("--start", default="2013-01-01")
    ap.add_argument("--oos-start", default=CHECKPOINT_RELEASE,
                    help="start of the post-checkpoint slice that is clean of pretraining leakage")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--no-timesfm", action="store_true", help="baselines only (no model download)")
    args = ap.parse_args()

    prices = pd.read_csv(PRICES, index_col=0, parse_dates=True)
    tickers = [t for t in args.tickers if t in prices.columns]
    missing = set(args.tickers) - set(tickers)
    if missing:
        print(f"not in the panel, skipping: {sorted(missing)}")

    print(f"Building baselines over {len(tickers)} ETFs, horizon {args.horizon}d ...")
    panel = build_panel(prices, tickers, args.horizon, args.min_history, args.start)
    print(f"  {panel['date'].nunique()} eval dates, "
          f"{len(panel) // max(panel['model'].nunique(), 1)} forecasts per model")

    if not args.no_timesfm:
        panel = add_timesfm(panel, prices, args.horizon, args.context,
                            args.batch_size, args.checkpoint or DEFAULT_CHECKPOINT)

    panel = panel[np.isfinite(panel["pred"]) & np.isfinite(panel["true"])].copy()
    panel["resid"] = panel["true"] - panel["pred"]
    panel["pred_adj"] = panel["pred"] + causal_bias_correction(panel)
    panel["loss"] = qlike(np.exp(panel["true"]), np.exp(panel["pred_adj"]))

    OUT_DIR.mkdir(exist_ok=True)
    panel.to_csv(OUT_DIR / "timesfm_vol_bakeoff.csv", index=False)

    slices = [("full", panel)]
    oos = panel[panel["date"] >= pd.Timestamp(args.oos_start)]
    if len(oos):
        slices.append((f"post-{args.oos_start}", oos))

    for label, sl in slices:
        print(f"\n{'=' * 78}\n{label}  ({sl['date'].nunique()} eval dates, "
              f"{sl['ticker'].nunique()} ETFs)\n{'=' * 78}")
        print(score(sl, label).to_string(index=False,
                                         float_format=lambda v: f"{v:9.4f}"))
        for bench in ("bayes", "rw"):
            dm = dm_table(sl, bench, label)
            if len(dm):
                print(f"\n  Diebold-Mariano vs {bench} (QLIKE; negative = better than {bench}):")
                print(dm.to_string(index=False, float_format=lambda v: f"{v:9.4f}"))

    diagnostics(panel)

    print(f"\nwrote {OUT_DIR / 'timesfm_vol_bakeoff.csv'}")


if __name__ == "__main__":
    main()
