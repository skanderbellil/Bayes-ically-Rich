#!/usr/bin/env python3
"""
The GFC test: the final configuration meets the crisis it never saw.

Everything alive in this thread — the QLD/GLD barbell, the council dial as
rotation, the return-tilt frontier — was validated on 2011→2026, a sample
with no systemic crisis. QLD has real prints from 2006-06-21; GLD from
2004; the macro panel reaches 1990; net liquidity is rebuilt here from
2005. This experiment stitches pre-2010 history onto the bundled data and
runs the FINAL configurations through 2007–2009 with zero re-fitting:
the council is fit-free, the barbell weights are fixed, volmgmt is
parameter-free. Nothing here was chosen after seeing 2008.

Caveats disclosed: the liquidity seat only has data from ~2005 (the
council runs 5–6 seats before that, by construction); QLD's 2006 start
means the GFC entry is ~16 months after inception; pre-2010 GLD/QQQ come
from a different adjusted-close snapshot and are stitched via returns.

Network on first run (GLD pre-2010 via yfinance; net liquidity via FRED);
cached afterward.
"""
import _bootstrap  # noqa: F401  (adds repo root to sys.path, loads .env)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from posterioralpha.council import CouncilBacktest, CouncilConfig
from posterioralpha.data import load_etf_universe_prices, load_fred_macro

OUT_DIR = _bootstrap.ROOT / "results" / "pod_policies"
COUNCIL_DIR = _bootstrap.ROOT / "results" / "council"
TC = 2e-4
GFC = slice("2007-10-09", "2009-03-09")     # SPX peak → trough


def fetch_fresh(ticker: str) -> pd.Series:
    cache = _bootstrap.ROOT / "datasets" / f"fresh_{ticker}.csv"
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["Date"], index_col="Date")["Close"]
    import yfinance as yf
    df = yf.download(ticker, start="1999-01-01", end="2010-01-01",
                     auto_adjust=True, progress=False)
    s = df["Close"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s.name = "Close"
    s.to_frame().to_csv(cache, index_label="Date")
    return s


def stitch(old: pd.Series, new: pd.Series) -> pd.Series:
    """Join two adjusted-close snapshots via returns (no level mixing)."""
    old_r = old.pct_change().loc[: new.index[0]].iloc[:-1]
    full_r = pd.concat([old_r, new.pct_change()]).dropna()
    full_r = full_r[~full_r.index.duplicated()]
    return (1 + full_r).cumprod()


def ensure_liquidity() -> pd.Series:
    nl_path = _bootstrap.ROOT / "datasets" / "net_liquidity.csv"
    nl = pd.read_csv(nl_path, parse_dates=["date"], index_col="date")
    if nl.index[0].year > 2006:
        from posterioralpha.data import build_net_liquidity
        nl = build_net_liquidity(start="2005-01-01")
    return nl["net_liquidity"]


def stats(r: pd.Series) -> dict:
    r = r.dropna()
    cum = (1 + r).cumprod()
    dd = float(((cum - cum.cummax()) / cum.cummax()).min())
    years = len(r) / 252
    cagr = float(cum.iloc[-1] ** (1 / years) - 1)
    return {"cagr": round(cagr, 3), "sharpe": round(float(r.mean() / r.std() * np.sqrt(252)), 2),
            "max_dd": round(dd, 3), "calmar": round(cagr / abs(dd), 2),
            "wealth": round(float(cum.iloc[-1]), 1)}


def main():
    uni = load_etf_universe_prices()
    lev = pd.read_csv(_bootstrap.ROOT / "datasets" / "levered_etfs.csv.gz",
                      index_col=0, parse_dates=True)
    qqq = stitch(fetch_fresh("QQQ"), uni["QQQ"].dropna())
    gld = stitch(fetch_fresh("GLD"), uni["GLD"].dropna())
    qld = lev["QLD"].dropna()
    liquidity = ensure_liquidity()
    macro = load_fred_macro()

    # ── council on the stitched QQQ (fit-free; minutes from ~2000) ────────
    minutes_path = COUNCIL_DIR / "minutes_QQQ_ext.csv"
    if minutes_path.exists():
        minutes = pd.read_csv(minutes_path, index_col=0, parse_dates=True)
    else:
        bt = CouncilBacktest(qqq, macro=macro, liquidity=liquidity,
                             config=CouncilConfig(seed=7))
        res = bt.run()
        print("council mirror audit (extended sample):")
        print(res.audit.to_string())
        minutes = res.minutes
        COUNCIL_DIR.mkdir(parents=True, exist_ok=True)
        minutes.to_csv(minutes_path)

    # ── books from QLD inception + 1 trading year ──────────────────────────
    start = qld.index[0] + pd.Timedelta(days=30)
    idx = qld.loc[start:].index
    r_qld = qld.pct_change().reindex(idx).fillna(0.0)
    r_gld = gld.pct_change().reindex(idx).fillna(0.0)
    r_qqq = qqq.pct_change().reindex(idx).fillna(0.0)
    e = (minutes["exposure"].reindex(qld.index).ffill()
         .reindex(idx).fillna(0.5))

    def book(w_leg, scale):
        wl = (w_leg * scale).shift(1).fillna(0.0)
        wg = ((1 - w_leg) * scale).shift(1).fillna(0.0)
        turn = wl.diff().abs().fillna(0) + wg.diff().abs().fillna(0)
        return wl * r_qld + wg * r_gld - TC * turn

    half = pd.Series(0.5, index=idx)
    ones = pd.Series(1.0, index=idx)
    static_mix = 0.5 * r_qld + 0.5 * r_gld
    vol = static_mix.rolling(21).std()
    vm = (vol.rolling(252, min_periods=126).median() / vol).clip(0, 1).fillna(1.0)

    books = {
        "QQQ B&H": r_qqq, "QLD B&H": r_qld,
        "static 50/50": book(half, ones),
        "rotation (council)": book(e, ones),
        "volmgmt 50/50": book(half, vm),
    }

    full = pd.DataFrame({n: stats(b) for n, b in books.items()}).T
    gfc = pd.DataFrame({n: {"GFC_return": round(float((1 + b.loc[GFC]).prod() - 1), 3)}
                        for n, b in books.items()}).T
    y2008 = pd.DataFrame({n: {"2008": round(float((1 + b.loc["2008"]).prod() - 1), 3),
                              "2009": round(float((1 + b.loc["2009"]).prod() - 1), 3)}
                          for n, b in books.items()}).T
    table = full.join(gfc).join(y2008)

    print(f"\nGFC TEST ({idx[0].date()} → {idx[-1].date()}; zero re-fitting, "
          "QLD real prints, stitched GLD/QQQ, Alpaca-tier costs)")
    print("=" * 100)
    print(table.to_string())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "barbell_gfc.csv")

    fig, ax = plt.subplots(figsize=(11.5, 6))
    colors = {"QQQ B&H": "#bbbbbb", "QLD B&H": "#888888",
              "static 50/50": "#e0a020", "volmgmt 50/50": "#7a4fb5",
              "rotation (council)": "#b5341a"}
    for n, c in colors.items():
        st = stats(books[n])
        ax.plot((1 + books[n]).cumprod(), color=c,
                lw=1.9 if n == "rotation (council)" else 1.1,
                label=f"{n} (GFC {gfc.loc[n, 'GFC_return']:+.0%}, "
                      f"maxDD {st['max_dd']:.0%}, wealth {st['wealth']}x)")
    ax.axvspan(pd.Timestamp("2007-10-09"), pd.Timestamp("2009-03-09"),
               color="#cc4444", alpha=0.08)
    ax.set_yscale("log")
    ax.set_ylabel("growth of $1 (log)")
    ax.set_title("The crisis the strategy never saw: QLD/GLD configurations through 2008\n"
                 "(council fit-free, weights fixed, nothing chosen after seeing the GFC)")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "barbell_gfc.png", dpi=150)
    print(f"\nSaved to {OUT_DIR}/: barbell_gfc.csv, barbell_gfc.png")


if __name__ == "__main__":
    main()
