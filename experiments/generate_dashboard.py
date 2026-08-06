#!/usr/bin/env python3
"""Generate a SIMPLE self-contained HTML dashboard from the paper-trade ledgers.

Overview-only design:
  * Refresh button (reload) + Run button (opens the Actions workflow to dispatch).
  * Last run / next run (hourly cron at :15 UTC) with a live countdown.
  * Combined KPIs across all strategies.
  * Per-strategy KPIs assuming a $1,000 bankroll from start, comparing
    10%-of-equity staking (compounding) vs a flat $10 stake each trade.
  * Toggle buttons to chart the global equity curve or any single strategy
    (both sizing schemes), drawn inline (no external libraries).

Economics: staking $S on a YES contract at price p returns S*(outcome/p - 1)
(win: S*(1/p-1), loss: -S). So 10% staking compounds equity *= 1+0.10*r and the
flat scheme adds 10*r, r = outcome/entry - 1, per resolved trade in time order.

Output: data/paper_trade/dashboard.html
"""
from __future__ import annotations
import json
import math
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:                 # so `posterioralpha` imports when run as a script
    sys.path.insert(0, str(ROOT))
DATA = ROOT / "data" / "paper_trade"
OUT  = DATA / "dashboard.html"
REPO = "skanderbellil/Bayes-ically-Rich"
RUN_URL = f"https://github.com/{REPO}/actions/workflows/paper_trade.yml"

CAP0 = 1000.0      # assumed starting bankroll
STAKE_FRAC = 0.10  # 10% of equity per trade (compounding)
FLAT = 10.0        # flat $ per trade

# (file, entry column, question column, label)
REGISTRY = [
    ("validated_regime_positions.csv",  "entry_ask",   "question",        "Regime (geo-calm)"),
    ("midprice_yes_positions.csv",      "entry_ask",   "question",        "YES [10–50%]"),
    ("midprice_yes_10_20_positions.csv","entry_ask",   "question",        "YES [10–20%]"),
    ("midprice_yes_10_30_positions.csv","entry_ask",   "question",        "YES [10–30%]"),
    ("midprice_yes_20_40_positions.csv","entry_ask",   "question",        "YES [20–40%]"),
    ("smart_flow_positions.csv",        "entry_ask",   "question",        "Smart Flow"),
    ("smart_flow_roi_positions.csv",    "entry_ask",   "question",        "Smart Flow (ROI)"),
    ("smart_flow_indep_positions.csv",  "entry_ask",   "question",        "Smart Flow (indep exp)"),
    ("macro_positions.csv",             "entry_price", "leader_question", "Macro (Fed cuts)"),
    ("dip_confirm_positions.csv",       "entry_ask",   "question",        "Dip-Confirm YES"),
]


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def load_ledger(path: Path, entry_col: str, q_col: str | None = None, row_filter=None):
    """Return a list of trades with real entry/exit dates:
    {ed, xd, entry, outcome, current, resolved, q}. Resolved trades have xd+outcome;
    open trades have current (marked at current_price, held to the end). `q` is the
    market question, truncated, for the per-position detail table.
    `row_filter(df) -> bool mask` optionally restricts the ledger rows (used for
    derived filter views, e.g. the combo band on the smart-flow book)."""
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if row_filter is not None:
        df = df[row_filter(df)].reset_index(drop=True)
    if entry_col not in df.columns:
        entry_col = "entry_ask" if "entry_ask" in df.columns else ("entry_price" if "entry_price" in df.columns else None)
    if entry_col is None:
        return []
    df["_entry"] = _num(df[entry_col]).clip(lower=0.01, upper=0.99)
    df["_cur"] = _num(df.get("current_price"))
    df["_out"] = _num(df.get("outcome"))
    status = df.get("status", pd.Series(["" for _ in range(len(df))])).astype(str)
    out = df["_out"].where(df["_out"].notna(), status.str.lower().map({"won": 1.0, "lost": 0.0}))
    tokens = df.get("token", pd.Series(["" for _ in range(len(df))])).astype(str)
    if q_col and q_col in df.columns:
        qs = df[q_col].astype(str)
    else:
        qs = pd.Series(["" for _ in range(len(df))])

    trades = []
    for i, r in df.iterrows():
        e = r["_entry"]
        if pd.isna(e):
            continue
        st = status.iloc[i].lower()
        ed = str(r.get("entry_date"))[:10] if pd.notna(r.get("entry_date")) else None
        tok = tokens.iloc[i]
        q = qs.iloc[i].strip()
        q = (q[:80] + "…") if len(q) > 80 else q
        if st in ("won", "lost") or (pd.notna(out.iloc[i]) and st not in ("open", "watching")):
            o = out.iloc[i]
            if pd.isna(o):
                continue
            xd = str(r.get("exit_date"))[:10] if pd.notna(r.get("exit_date")) else ed
            trades.append({"token": tok, "ed": ed or xd, "xd": xd or ed, "entry": float(e),
                           "outcome": float(o), "current": None, "resolved": True, "q": q})
        elif st == "open":
            cur = float(r["_cur"]) if pd.notna(r["_cur"]) else float(e)
            trades.append({"token": tok, "ed": ed, "xd": None, "entry": float(e),
                           "outcome": None, "current": cur, "resolved": False, "q": q})
    return [t for t in trades if t["ed"]]


# --- hourly (intraday) price-mark cache (cache resolved markets once, refresh open ones) ---
MARKS_PATH = DATA / "token_intraday.csv.gz"


def load_marks():
    if not MARKS_PATH.exists():
        return {}
    try:
        df = pd.read_csv(MARKS_PATH, dtype={"token": str})
        df["ts"] = pd.to_datetime(df["ts"])
        return {tok: g.set_index("ts")["p"].sort_index() for tok, g in df.groupby("token")}
    except Exception:
        return {}


def save_marks(marks):
    rows = [(tok, ts.isoformat(), float(p)) for tok, s in marks.items() for ts, p in s.items()]
    pd.DataFrame(rows, columns=["token", "ts", "p"]).to_csv(MARKS_PATH, index=False, compression="gzip")


def fetch_intraday(token):
    """Hourly Yes-price series for a token (lazy import; None on any failure)."""
    try:
        from posterioralpha.polymarket.fetch import fetch_token_history_raw
        s = fetch_token_history_raw(str(token), fidelity_minutes=60, use_cache=False)
        if s is None or len(s) == 0:
            return None
        s.index = pd.to_datetime(s.index).tz_localize(None)
        return s.sort_index()
    except Exception:
        return None


def update_marks(token_resolved):
    """token_resolved: {token: fully_resolved?}. Keep cached resolved tokens (static);
    (re)fetch open or uncached ones. Degrades gracefully with no network/deps."""
    marks = load_marks()
    changed = False
    for tok, resolved in token_resolved.items():
        if not tok or tok in ("", "nan"):
            continue
        if tok in marks and resolved:
            continue
        s = fetch_intraday(tok)
        if s is not None and len(s):
            marks[tok] = s; changed = True
    if changed:
        try:
            save_marks(marks)
        except Exception:
            pass
    return marks


def kelly_fraction(returns, cap=0.5):
    """Growth-optimal (full Kelly) staking fraction from the realized per-trade
    return distribution: argmax_f Σ log(1 + f·r_i). Solved by bisection on the
    derivative Σ r/(1+f·r). 0 if no positive expectancy (Kelly says don't bet).
    In-sample (fit on a strategy's own history) and capped at `cap` for sanity —
    full Kelly on noisy binary samples is otherwise punishingly aggressive."""
    rs = [float(r) for r in returns if r is not None]
    if not rs or sum(rs) <= 0:
        return 0.0

    def g(f):
        return sum(r / (1.0 + f * r) for r in rs)

    lo, hi = 0.0, 0.999
    if g(hi) > 0:
        f = hi
    else:
        for _ in range(80):
            mid = (lo + hi) / 2.0
            if g(mid) > 0:
                lo = mid
            else:
                hi = mid
        f = (lo + hi) / 2.0
    return round(min(f, cap), 4)


def walkforward_kelly(trades, min_hist=5, cap=0.5):
    """Per-trade WALK-FORWARD Kelly (no lookahead): trade t is sized from the Kelly
    fit on the EVENT-CLUSTERED returns of trades that had settled before t was
    entered (settlement date <= t's entry date).

    Clustering = one observation per settlement date (mean return across same-day
    trades), not one per trade. Without this, correlated same-event trades (e.g.
    several props on one match, all settling together) get counted as independent
    evidence and inflate Kelly's apparent edge/confidence — exactly the
    nominal-vs-effective-breadth gap documented in
    docs/knowledge/sharpe-decomposition-levers.md. `min_hist` is independent
    EVENTS, not raw trade count: no bet until `min_hist` prior events have
    settled, even if hundreds of (correlated) trades have.

    `min_hist=5` (not the pre-clustering 10) because event-clustered samples are
    already lower-noise per observation (each is a mean over several correlated
    trades) — but Kelly needs both a mean AND a variance estimate, so this isn't
    lowered further just because a smaller strategy would otherwise show 0%
    (that would be tuning the threshold to the output, the exact p-hacking this
    whole diagnostic exists to catch). A strategy with < 5 independent
    settlement days genuinely has no statistically defensible size yet — 0% is
    the correct answer, not a bug. Returns (fracs[idx], current_rate) where
    current_rate is the all-history estimate (the stake you'd use on the next
    trade)."""
    settled = sorted(((t["xd"], t["outcome"] / t["entry"] - 1.0)
                      for t in trades if t["resolved"] and t["xd"]), key=lambda x: x[0])
    sx = [d for d, _ in settled]
    sr = [r for _, r in settled]

    def event_means(k):
        by_event: dict[str, list] = {}
        for d, r in zip(sx[:k], sr[:k]):
            by_event.setdefault(d, []).append(r)
        return [float(np.mean(v)) for v in by_event.values()]

    import bisect
    fracs = {}
    for idx, t in enumerate(trades):
        k = bisect.bisect_right(sx, t["ed"])            # how many had settled by entry
        sample = event_means(k)
        if len(sample) < min_hist:
            fracs[idx] = 0.0
        else:
            fracs[idx] = round(kelly_fraction(sample, cap=cap) * min(1.0, len(sample) / 20.0), 4)
    final_sample = event_means(len(sx))
    cur = (round(kelly_fraction(final_sample, cap=cap) * min(1.0, len(final_sample) / 20.0), 4)
           if len(final_sample) >= min_hist else 0.0)
    return fracs, cur


def sim(trades, mode, marks=None, frac=STAKE_FRAC, fracs=None):
    """Realistic cash sim from $1,000 — NO LEVERAGE. A position ties up cash from
    entry to exit; new trades are sized min(target, cash) so overlapping positions
    can't be funded on margin. Buy/sell DECISIONS execute at the start of their
    recorded date (the ledger only records entry/exit at daily granularity), but
    between decisions OPEN positions are marked HOURLY to their real intraday price
    (from the marks cache) — an actually-intraday equity curve, not a daily
    staircase. mode 'pct' = `frac` of equity (Kelly), 'flat'=$10."""
    marks = marks or {}
    if not trades:
        return dict(pts=[["", CAP0]], fin=CAP0, ret=0.0, dd=0.0, realized=0.0, unreal=0.0,
                    taken=0, constrained=0, maxconc=0, peakdep=0.0, stakes={})
    buys, sells = {}, {}
    for idx, t in enumerate(trades):
        buys.setdefault(t["ed"], []).append(idx)
        if t["resolved"]:
            sells.setdefault(max(t["xd"], t["ed"]), []).append(idx)
    start = min(t["ed"] for t in trades)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    today = now.strftime("%Y-%m-%d")
    end = max([today] + [t["xd"] for t in trades if t["resolved"] and t["xd"]])
    end_ts = min(now, pd.Timestamp(end) + pd.Timedelta(days=1))
    hours = pd.date_range(pd.Timestamp(start), end_ts, freq="h")
    if len(hours) == 0:
        hours = pd.DatetimeIndex([pd.Timestamp(start)])

    def mp(idx, ts):
        t = trades[idx]; s = marks.get(t["token"])
        if s is not None and len(s):
            v = s.asof(ts)
            if pd.notna(v):
                return float(v)
        return t["current"] if t["current"] is not None else t["entry"]

    cash, hold, pts = CAP0, {}, []
    taken = constrained = maxconc = 0
    realized = peakdep = 0.0
    stakes = {}                            # idx -> {"frac": fraction used, "stake": $ at entry}
    cur_day = None

    def settle(day):
        """Close every position settling on `day` that is currently held.
        Idempotent — a second call skips what the first already popped."""
        nonlocal cash, realized
        for idx in sells.get(day, []):
            h = hold.pop(idx, None)
            if h:
                proc = h["shares"] * trades[idx]["outcome"]
                cash += proc; realized += proc - h["cost"]

    for ts in hours:
        day = ts.strftime("%Y-%m-%d")
        if day != cur_day:
            cur_day = day
            # Sell first so capital from today's settlements funds today's entries.
            settle(day)
            for idx in buys.get(day, []):
                eq = cash + sum(h["shares"] * mp(i, ts) for i, h in hold.items())
                fr = (fracs.get(idx, 0.0) if fracs is not None else frac)
                target = fr * eq if mode == "pct" else FLAT
                if target <= 1e-9:
                    continue                           # Kelly says don't bet (no edge / no history yet)
                stake = min(target, cash)
                if stake > 1e-6:
                    if stake < target - 1e-6:
                        constrained += 1                # capital-limited (real margin constraint)
                    hold[idx] = {"shares": stake / trades[idx]["entry"], "cost": stake}
                    cash -= stake; taken += 1
                    maxconc = max(maxconc, len(hold))
                    if mode == "pct":
                        stakes[idx] = {"frac": fr, "stake": round(stake, 2)}
            # Same-day round-trips: ~half of these ledgers' trades enter AND
            # settle on one date (sports markets). The pre-buy sell pass ran
            # before they were held, so without this second pass their capital
            # is locked forever — six such trades pinned every sleeve's cash at
            # $0 from 2026-07-01 and froze the whole dashboard.
            settle(day)
        cost = sum(h["cost"] for h in hold.values())
        mv = sum(h["shares"] * mp(i, ts) for i, h in hold.items())
        peakdep = max(peakdep, cost / (cost + cash) if (cost + cash) > 0 else 0.0)
        pts.append([ts.isoformat(), round(cash + mv, 2)])

    final = pts[-1][1] if pts else CAP0
    unreal = sum(hold[i]["shares"] * mp(i, hours[-1]) - hold[i]["cost"] for i in hold)
    peak, dd = CAP0, 0.0
    for _, e in pts:
        peak = max(peak, e); dd = min(dd, e / peak - 1.0 if peak > 0 else 0.0)
    return dict(pts=pts, fin=final, ret=final / CAP0 - 1.0, dd=dd, realized=realized,
                unreal=unreal, taken=taken, constrained=constrained, maxconc=maxconc,
                peakdep=peakdep, stakes=stakes)


def kpis_for(trades, label, sid=None, marks=None):
    resolved = [t for t in trades if t["resolved"]]
    opens = [t for t in trades if not t["resolved"]]
    win = sum(t["outcome"] == 1 for t in resolved) / len(resolved) if resolved else 0.0
    # WALK-FORWARD Kelly: each trade sized only from edge known before it (no lookahead)
    wf_fracs, cur_kelly = walkforward_kelly(trades)
    p = sim(trades, "pct", marks, fracs=wf_fracs); f = sim(trades, "flat", marks)
    fk = cur_kelly  # the rate the model recommends for the NEXT trade (all history)
    k = dict(label=label, n=len(resolved), open=len(opens), win=win, kelly=fk,
             fin10=p["fin"], ret10=p["ret"], dd10=p["dd"],
             finflat=f["fin"], retflat=f["ret"], dd_flat=f["dd"],
             realized=f["realized"], unreal=f["unreal"], mtm=f["realized"] + f["unreal"],
             constrained=p["constrained"], taken=p["taken"], maxconc=p["maxconc"], peakdep=p["peakdep"])
    if sid is not None:
        k["id"] = sid
    return k, {"label": label, "pts10": p["pts"], "ptsf": f["pts"], "kelly_stakes": p["stakes"]}


def build_positions(trades, kelly_stakes=None):
    """Per-position table rows for the strategy-detail drill-down: one row per
    trade with entry/exit price, status, PnL at a flat-$10 stake AND at the
    walk-forward Kelly stake actually chosen for that trade (the dollar amount
    it was entered with, and the resulting PnL; 0 if Kelly said don't bet — no
    edge proven yet or capital-constrained). Resolved trades use the realized
    outcome; open trades mark-to-current. Sorted most-recent first."""
    kelly_stakes = kelly_stakes or {}
    rows = []
    for idx, t in enumerate(trades):
        entry = t["entry"]
        if t["resolved"]:
            exitp = t["outcome"]
            status = "WON" if exitp >= 0.5 else "LOST"
        else:
            exitp = t["current"] if t["current"] is not None else entry
            status = "OPEN"
        pnl_pct = exitp / entry - 1.0
        ks = kelly_stakes.get(idx)
        kfrac = ks["frac"] if ks else 0.0
        kstake = ks["stake"] if ks else 0.0
        rows.append({
            "q": t.get("q", ""), "ed": t["ed"], "xd": t["xd"],
            "entry": round(entry, 3), "exit": round(exitp, 3), "status": status,
            "pnl_pct": round(pnl_pct * 100, 1), "pnl10": round(10 * pnl_pct, 2),
            "kfrac": round(kfrac * 100, 1), "kstake": round(kstake, 2),
            "kpnl": round(kstake * pnl_pct, 2),
        })
    rows.sort(key=lambda r: r["xd"] or r["ed"] or "", reverse=True)
    return rows


def _streak(resolved_sorted):
    """Current win/loss streak (e.g. 'W3', 'L2') from chronologically-sorted
    resolved trades. '—' if there's no resolved history."""
    if not resolved_sorted:
        return "—"
    last_won = resolved_sorted[-1]["outcome"] >= 0.5
    n = 0
    for t in reversed(resolved_sorted):
        if (t["outcome"] >= 0.5) != last_won:
            break
        n += 1
    return f"{'W' if last_won else 'L'}{n}"


def assess_strategy(trades, label):
    """Deep-dive diagnostics for the strategy-detail tab, organized around the
    Fundamental-Law decomposition (docs/knowledge/sharpe-decomposition-levers.md):
    Sharpe ~ edge x sqrt(independent breadth x cadence). Two t-stats are reported
    for the same edge — a NAIVE one (treats every trade as independent) and an
    EVENT-CLUSTERED one (groups trades resolving the same day, since same-event
    markets move together) — the gap between them is exactly the "effective
    breadth" lesson: nominal trade count overstates independent bets when many
    positions resolve off the same real-world event.
    """
    resolved = sorted([t for t in trades if t["resolved"]], key=lambda t: t["xd"] or t["ed"] or "")
    opens = [t for t in trades if not t["resolved"]]
    n = len(resolved)
    if n == 0:
        return dict(label=label, n=0, open=len(opens))

    edges = np.array([t["outcome"] - t["entry"] for t in resolved])
    rets = np.array([t["outcome"] / t["entry"] - 1.0 for t in resolved])
    pnl10 = 10.0 * rets
    wins = pnl10[pnl10 > 0]
    losses = pnl10[pnl10 < 0]

    edge_mean = float(edges.mean())
    edge_se = float(edges.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    edge_t = edge_mean / edge_se if edge_se else float("nan")

    by_event: dict[str, list] = {}
    for t, e in zip(resolved, edges):
        by_event.setdefault(t["xd"] or t["ed"], []).append(e)
    event_means = np.array([float(np.mean(v)) for v in by_event.values()])
    n_events = len(event_means)
    if n_events > 1 and event_means.std(ddof=1) > 0:
        event_t = float(event_means.mean() / (event_means.std(ddof=1) / math.sqrt(n_events)))
    else:
        event_t = float("nan")

    gross_win = float(wins.sum())
    gross_loss = float(-losses.sum())
    if gross_loss > 0:
        profit_factor = gross_win / gross_loss
    else:
        profit_factor = float("inf") if gross_win > 0 else float("nan")

    r_mean, r_std = float(rets.mean()), (float(rets.std(ddof=1)) if n > 1 else float("nan"))
    sharpe_trade = (r_mean / r_std) if r_std else float("nan")

    span_days = max(1, (pd.to_datetime(resolved[-1]["xd"] or resolved[-1]["ed"])
                         - pd.to_datetime(resolved[0]["ed"])).days)
    trades_per_year = n / span_days * 365.25
    sharpe_ann = sharpe_trade * math.sqrt(trades_per_year) if sharpe_trade == sharpe_trade else float("nan")

    hold_days = [(pd.to_datetime(t["xd"]) - pd.to_datetime(t["ed"])).days
                 for t in resolved if t["xd"] and t["ed"]]

    return dict(
        label=label, n=n, open=len(opens),
        win_rate=float((rets >= 0).mean()),
        avg_entry=float(np.mean([t["entry"] for t in resolved])),
        avg_hold=float(np.mean(hold_days)) if hold_days else float("nan"),
        trades_per_week=n / span_days * 7,
        edge_mean=edge_mean, edge_t=edge_t,
        n_events=n_events, breadth_pct=(n_events / n if n else 0.0), event_t=event_t,
        expectancy10=float(pnl10.mean()), profit_factor=profit_factor,
        avg_win10=float(wins.mean()) if len(wins) else 0.0,
        avg_loss10=float(losses.mean()) if len(losses) else 0.0,
        best10=float(pnl10.max()), worst10=float(pnl10.min()),
        streak=_streak(resolved),
        sharpe_trade=sharpe_trade, sharpe_ann=sharpe_ann, trades_per_year=trades_per_year,
    )


def _sum_curves(sers, key, cap_each=CAP0):
    """Sum independent per-sleeve daily equity curves on a common date grid (each
    sleeve flat at its $1k before its first trade)."""
    s_list = []
    for ser in sers:
        pts = ser[key]
        if not pts or not pts[0][0]:
            continue
        s = pd.Series({d: e for d, e in pts})
        s.index = pd.to_datetime(s.index)
        s_list.append(s[~s.index.duplicated(keep="last")].sort_index())
    if not s_list:
        return [["", cap_each]]
    idx = pd.DatetimeIndex(sorted(set().union(*[set(s.index) for s in s_list])))
    total = None
    for s in s_list:
        a = s.reindex(idx).ffill().fillna(cap_each)
        total = a if total is None else total + a
    return [[d.strftime("%Y-%m-%d"), round(float(v), 2)] for d, v in total.items()]


def _combo_band(df):
    """The combo-backtest filter (MIDPRICE_SMARTFLOW_COMBO.md): entry mid in the
    [0.30, 0.70) band — which also excludes the tail prices (<0.10 / >=0.90)
    where smart-pool consensus backtested strongly NEGATIVE."""
    mid = _num(df.get("entry_mid", df.get("entry_ask")))
    return (mid >= 0.30) & (mid < 0.70)


def _combo_cell(df):
    """The full pre-registered cell (band AND <=3 days to resolution at entry).
    Requires the end_date column the tracker records from 2026-07-02 — legacy
    rows without it are excluded, so this view is a clean forward score."""
    band = _combo_band(df)
    end = pd.to_datetime(df.get("end_date"), errors="coerce")
    ent = pd.to_datetime(df.get("entry_date"), errors="coerce")
    dtr = (end - ent).dt.days
    return band & dtr.notna() & (dtr <= 3)


def _indep_class_filter(cls: str):
    """Row filter factory: keep only rows whose `indep_class` column equals
    ``cls`` (the pre-registered independent/cascade split from the
    smart-flow-independence experiment; see SMART_FLOW_INDEPENDENCE.md).
    Missing column -> empty mask (handled gracefully like an absent ledger)."""
    def _f(df):
        if "indep_class" not in df.columns:
            return pd.Series(False, index=df.index)
        return df["indep_class"].astype(str) == cls
    return _f


def _in_strategy(df):
    """Exploratory regime ledger logs both regimes for comparison — chart only
    the rows the strategy would actually have taken."""
    if "in_strategy" not in df.columns:
        return pd.Series(True, index=df.index)
    return df["in_strategy"].astype(str).str.lower().isin(("true", "1", "1.0"))


# Derived filter views: (source file, dashboard id, label, row filter)
DERIVED = [
    ("regime_positions.csv", "regime_all",
     "Regime (all-domain, exploratory)", _in_strategy),
    ("smart_flow_positions.csv", "smart_flow_combo",
     "Smart Flow ∩ band .30–.70", _combo_band),
    ("smart_flow_positions.csv", "smart_flow_combo3d",
     "Combo cell (band ∩ ≤3d)", _combo_cell),
    ("smart_flow_indep_positions.csv", "smart_flow_indep_independent",
     "SF independent", _indep_class_filter("independent")),
    ("smart_flow_indep_positions.csv", "smart_flow_indep_cascade",
     "SF cascade", _indep_class_filter("cascade")),
]


def build(fetch_marks=True):
    per_ledger = []
    all_trades = []
    for fname, ecol, qcol, label in REGISTRY:
        trades = load_ledger(DATA / fname, ecol, qcol)
        per_ledger.append((fname, label, trades))
        all_trades.extend(trades)
    # derived views re-slice an existing ledger — their trades are NOT added to
    # all_trades/COMBINED (that would double-count the underlying positions)
    for fname, sid, label, filt in DERIVED:
        trades = load_ledger(DATA / fname, "entry_ask", "question", row_filter=filt)
        per_ledger.append((f"{sid}_positions.csv", label, trades))

    # cache-once + refresh-open: a token is static only if ALL its trades are resolved
    token_resolved = {}
    for t in all_trades:
        token_resolved[t["token"]] = token_resolved.get(t["token"], True) and t["resolved"]
    marks = update_marks(token_resolved) if fetch_marks else load_marks()

    derived_ids = {sid for _, sid, _, _ in DERIVED}
    strategies, series, active_ser = [], {}, []
    for fname, label, trades in per_ledger:
        sid = fname.replace("_positions.csv", "")
        k, ser = kpis_for(trades, label, sid, marks)
        k["derived"] = sid in derived_ids
        strategies.append(k)
        if k["n"] or k["open"]:
            ser["positions"] = build_positions(trades, ser.pop("kelly_stakes", {}))
            ser["assess"] = assess_strategy(trades, label)
            series[sid] = ser
            if not k["derived"]:      # filter views re-slice a ledger already
                active_ser.append((k, ser))   # counted — keep them out of COMBINED

    # COMBINED = SUM of independent $1k sleeves (each strategy its own bankroll)
    act = [k for k, _ in active_ser]
    nact = len(act)
    cap = nact * CAP0
    n_tot = sum(k["n"] for k in act)
    g10 = _sum_curves([s for _, s in active_ser], "pts10")
    gf = _sum_curves([s for _, s in active_ser], "ptsf")

    def dd_of(pts):
        peak = dd = 0.0
        for _, e in pts:
            peak = max(peak, e); dd = min(dd, e / peak - 1.0 if peak > 0 else 0.0)
        return dd

    fin10 = g10[-1][1] if g10 and g10[0][0] else cap
    finflat = gf[-1][1] if gf and gf[0][0] else cap
    combined = dict(
        label="ALL COMBINED", n=n_tot, open=sum(k["open"] for k in act),
        win=(sum(k["win"] * k["n"] for k in act) / n_tot) if n_tot else 0.0,
        kelly=None, cap=cap, nsleeves=nact,
        fin10=fin10, ret10=fin10 / cap - 1.0 if cap else 0.0, dd10=dd_of(g10),
        finflat=finflat, retflat=finflat / cap - 1.0 if cap else 0.0,
        realized=sum(k["realized"] for k in act), unreal=sum(k["unreal"] for k in act),
        mtm=sum(k["mtm"] for k in act),
        constrained=sum(k["constrained"] for k in act),
        peakdep=max([k["peakdep"] for k in act], default=0.0))
    if nact:
        series = {"GLOBAL": {"label": "All combined (sum of sleeves)", "pts10": g10, "ptsf": gf}, **series}
    return combined, strategies, series


def fmt_money(v):
    return f"${v:,.0f}"


def signed(v):
    return f"${v:+,.0f}"


def _sc(v):
    return "pos" if v >= 0 else "neg"


def kpi_card(s):
    c10, cf = _sc(s["ret10"]), _sc(s["retflat"])
    derived_tag = (" · <b>filter view — not counted in COMBINED</b>"
                   if s.get("derived") else "")
    return f"""
    <div class="card">
      <div class="card-h">{s['label']}</div>
      <div class="card-sub">{s['n']} trades · win {s['win']*100:.0f}% · open {s['open']}{derived_tag}</div>
      <div class="kgrid">
        <div class="k"><div class="kl">Kelly {s['kelly']*100:.0f}%</div><div class="kv {c10}">{fmt_money(s['fin10'])}</div><div class="kd {c10}">{s['ret10']*100:+.1f}%</div></div>
        <div class="k"><div class="kl">flat $10</div><div class="kv {cf}">{fmt_money(s['finflat'])}</div><div class="kd {cf}">{s['retflat']*100:+.1f}%</div></div>
      </div>
      <div class="pnl3">
        <div><span class="kl">Realized</span><span class="{_sc(s['realized'])}">{signed(s['realized'])}</span></div>
        <div><span class="kl">Unrealized</span><span class="{_sc(s['unreal'])}">{signed(s['unreal'])}</span></div>
        <div><span class="kl">MTM</span><span class="{_sc(s['mtm'])}">{signed(s['mtm'])}</span></div>
      </div>
      <div class="card-f">{deploy_note(s)}</div>
    </div>"""


def deploy_note(s):
    """Footer line for the Kelly (10%-of-equity) sim. When walk-forward Kelly
    staked $0 on every trade (no proven edge, or not enough independent
    settled events yet), say so — 'max DD 0% · peak 0% deployed' reads like a
    rendering bug otherwise."""
    if s.get("n") and not s.get("maxconc"):
        return ("Kelly never deployed — walk-forward edge ≤ 0 or &lt; 5 independent "
                "settled events, so the correct stake was $0 on every trade "
                "(flat $10 column shows the raw edge)")
    return (f"max DD {s['dd10']*100:.0f}% · peak {s['peakdep']*100:.0f}% deployed "
            f"· max {s['maxconc']} open{cap_note(s)}")


def cap_note(s):
    return f" · {s['constrained']} entries capital-capped" if s.get("constrained") else ""


def fnum(v, suf="", dp=2, sign=False):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "+∞" if (isinstance(v, float) and math.isinf(v) and v > 0) else "—"
    return f"{v:+.{dp}f}{suf}" if sign else f"{v:.{dp}f}{suf}"


def assess_card(a):
    """Deep-dive stat grid for the strategy-detail tab: sample/calibration, edge
    significance (naive vs event-clustered — the effective-breadth lesson),
    PnL quality, and risk-adjusted return. See assess_strategy() for the math."""
    if not a or not a.get("n"):
        return '<div class="note">No resolved trades yet for this strategy — stats need at least one settled position.</div>'
    return f"""
    <div class="astats">
      <div class="asec">
        <div class="asec-h">Sample &amp; calibration</div>
        <div class="agrid">
          <div class="a"><div class="al">Resolved / open</div><div class="av">{a['n']} / {a['open']}</div></div>
          <div class="a"><div class="al">Win rate</div><div class="av">{a['win_rate']*100:.0f}%</div></div>
          <div class="a"><div class="al">Avg entry price</div><div class="av">{a['avg_entry']*100:.1f}¢</div></div>
          <div class="a"><div class="al">Avg hold</div><div class="av">{fnum(a['avg_hold'], ' d', 1)}</div></div>
          <div class="a"><div class="al">Cadence</div><div class="av">{a['trades_per_week']:.1f}<span class="asub">/wk</span></div></div>
        </div>
      </div>
      <div class="asec">
        <div class="asec-h">Edge &amp; significance <span class="asub">Sharpe ≈ edge × √(breadth × cadence)</span></div>
        <div class="agrid">
          <div class="a"><div class="al">Mean edge</div><div class="av {_sc(a['edge_mean'])}">{a['edge_mean']*100:+.1f}¢</div></div>
          <div class="a"><div class="al">Edge t-stat (naive)</div><div class="av {_sc(a['edge_t'])}">{fnum(a['edge_t'], '', 2, True)}</div></div>
          <div class="a"><div class="al">Independent events</div><div class="av">{a['n_events']}<span class="asub"> ({a['breadth_pct']*100:.0f}% of n)</span></div></div>
          <div class="a"><div class="al">Edge t-stat (clustered)</div><div class="av {_sc(a['event_t'])}">{fnum(a['event_t'], '', 2, True)}</div></div>
        </div>
      </div>
      <div class="asec">
        <div class="asec-h">PnL quality <span class="asub">flat $10 stake</span></div>
        <div class="agrid">
          <div class="a"><div class="al">Expectancy/trade</div><div class="av {_sc(a['expectancy10'])}">{signed(a['expectancy10'])}</div></div>
          <div class="a"><div class="al">Profit factor</div><div class="av">{fnum(a['profit_factor'])}</div></div>
          <div class="a"><div class="al">Avg win / avg loss</div><div class="av">{signed(a['avg_win10'])} / {signed(a['avg_loss10'])}</div></div>
          <div class="a"><div class="al">Best / worst trade</div><div class="av">{signed(a['best10'])} / {signed(a['worst10'])}</div></div>
          <div class="a"><div class="al">Current streak</div><div class="av">{a['streak']}</div></div>
        </div>
      </div>
      <div class="asec">
        <div class="asec-h">Risk-adjusted <span class="asub">per-trade cadence, not calendar time</span></div>
        <div class="agrid">
          <div class="a"><div class="al">Sharpe (per-trade)</div><div class="av {_sc(a['sharpe_trade'])}">{fnum(a['sharpe_trade'], '', 2, True)}</div></div>
          <div class="a"><div class="al">Sharpe (annualized)</div><div class="av {_sc(a['sharpe_ann'])}">{fnum(a['sharpe_ann'], '', 2, True)}</div></div>
          <div class="a"><div class="al">Trades / yr (cadence)</div><div class="av">{a['trades_per_year']:.0f}</div></div>
        </div>
      </div>
    </div>"""


def generate():
    combined, strategies, series = build()
    now = datetime.now(timezone.utc)
    nxt = now.replace(minute=15, second=0, microsecond=0)
    if nxt <= now:
        nxt += timedelta(hours=1)

    first_sid = next(iter(series), "")
    btns = "".join(
        f'<button class="chartbtn{" active" if sid == first_sid else ""}" data-sid="{sid}">{sd["label"]}</button>'
        for sid, sd in series.items())
    order = sorted(strategies, key=lambda s: (-(s["id"] == "validated_regime"), -s["n"]))
    cards = "\n".join(kpi_card(s) for s in order)

    # Strategy-detail tab: dropdown + per-strategy KPI card, keyed off the same
    # sids as the chart buttons (those with at least one trade or open position).
    detail_sids = [sid for sid in series if sid != "GLOBAL"]
    first_detail = detail_sids[0] if detail_sids else ""
    strat_options = "".join(
        f'<option value="{sid}">{series[sid]["label"]}</option>' for sid in detail_sids)
    cards_map = {s["id"]: kpi_card(s) for s in strategies if s.get("id") in detail_sids}
    assess_cards = {sid: assess_card(series[sid].get("assess")) for sid in detail_sids}
    cm = combined
    cret10 = "pos" if cm["ret10"] >= 0 else "neg"
    cretf = "pos" if cm["retflat"] >= 0 else "neg"

    html = f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Paper-trade dashboard</title>
<style>
:root{{--bg:#0f1216;--panel:#171c22;--line:#262d36;--ink:#e6edf3;--mut:#8b97a5;--pos:#2ec27e;--neg:#e5534b;--acc:#3b82f6;}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 -apple-system,Segoe UI,Roboto,sans-serif}}
.wrap{{max-width:1100px;margin:0 auto;padding:18px}}
.bar{{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:16px}}
h1{{font-size:18px;margin:0;font-weight:650}}.spacer{{flex:1}}
button{{cursor:pointer;border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:8px;padding:8px 12px;font-size:13px}}
button:hover{{border-color:var(--acc)}}
.btn-run{{background:var(--acc);border-color:var(--acc);color:#fff;font-weight:600}}
.runinfo{{font-size:12px;color:var(--mut)}}.runinfo b{{color:var(--ink)}}
.combined{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin-bottom:18px}}
.cell{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px}}
.cell .l{{font-size:11px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}}
.cell .l .tag{{text-transform:none;letter-spacing:0;opacity:.7;font-size:10px}}
.cell .v{{font-size:20px;font-weight:680;margin-top:3px}}.cell .v.small{{font-size:17px}}
.kd{{font-size:12px}}
.section-h{{font-size:12px;color:var(--mut);text-transform:uppercase;letter-spacing:.05em;margin:18px 0 8px}}
#chartbtns{{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px}}
.chartbtn{{font-size:12px;padding:8px 11px}}.chartbtn.active{{background:var(--acc);border-color:var(--acc);color:#fff}}
.chartwrap{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}}
.legend{{display:flex;gap:16px;font-size:12px;color:var(--mut);margin-bottom:6px;flex-wrap:wrap}}
.legend i{{display:inline-block;width:14px;height:3px;vertical-align:middle;margin-right:5px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:10px}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px}}
.card-h{{font-weight:640}}.card-sub{{font-size:12px;color:var(--mut);margin:2px 0 8px}}
.kgrid{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}
.k{{background:#10151b;border:1px solid var(--line);border-radius:8px;padding:8px}}
.kl{{font-size:10px;color:var(--mut);text-transform:uppercase}}.kv{{font-size:17px;font-weight:680}}
.pnl3{{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-top:8px}}
.pnl3 div{{background:#10151b;border:1px solid var(--line);border-radius:7px;padding:6px;display:flex;flex-direction:column;gap:2px}}
.pnl3 span:last-child{{font-size:13px;font-weight:640}}
.card-f{{font-size:11px;color:var(--mut);margin-top:8px}}.pos{{color:var(--pos)}}.neg{{color:var(--neg)}}
.note{{font-size:12px;color:var(--mut);background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px 12px;margin-bottom:6px}}.note b{{color:var(--ink)}}
.tabs{{display:flex;gap:6px;margin-bottom:16px}}
.tabbtn{{font-size:13px;font-weight:600}}.tabbtn.active{{background:var(--acc);border-color:var(--acc);color:#fff}}
.tabview{{display:none}}.tabview.active{{display:block}}
.stratbar{{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:14px}}
select{{background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:8px 12px;font-size:13px}}
.postable-wrap{{background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:auto;max-height:480px}}
table.postable{{width:100%;border-collapse:collapse;font-size:12px;white-space:nowrap}}
table.postable th{{position:sticky;top:0;background:var(--panel);text-align:left;color:var(--mut);text-transform:uppercase;font-size:10px;letter-spacing:.04em;padding:8px 10px;border-bottom:1px solid var(--line)}}
table.postable td{{padding:7px 10px;border-bottom:1px solid #1c222a;white-space:normal}}
table.postable td:nth-child(2){{min-width:220px;color:var(--ink)}}
.astats{{display:flex;flex-direction:column;gap:10px}}
.asec{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px}}
.asec-h{{font-size:12px;font-weight:650;margin-bottom:10px}}
.asec-h .asub{{font-weight:400;color:var(--mut);font-size:11px;margin-left:6px;text-transform:none;letter-spacing:0}}
.agrid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px}}
.agrid .a{{background:#10151b;border:1px solid var(--line);border-radius:8px;padding:8px}}
.agrid .al{{font-size:10px;color:var(--mut);text-transform:uppercase;letter-spacing:.03em}}
.agrid .av{{font-size:16px;font-weight:660;margin-top:2px}}
.agrid .av .asub{{font-size:10px;color:var(--mut);font-weight:400}}
@media(max-width:600px){{.wrap{{padding:12px}}.bar{{gap:8px}}h1{{font-size:16px;width:100%}}
  .runinfo{{width:100%;order:3}}.bar button{{flex:1}}.cell .v{{font-size:18px}}
  #chart,#chart2{{height:260px}}.section-h{{margin:14px 0 6px}}}}
</style></head><body><div class="wrap">

<div class="bar">
  <h1>📈 Paper-trade dashboard</h1>
  <div class="spacer"></div>
  <div class="runinfo">last run <b>{now:%Y-%m-%d %H:%M} UTC</b> · next <b>{nxt:%H:%M}</b> (<span id="cd"></span>)</div>
  <button onclick="location.reload(true)">↻ Refresh</button>
  <button class="btn-run" onclick="window.open('{RUN_URL}','_blank')">▶ Run now</button>
</div>

<div class="tabs">
  <button class="tabbtn active" data-tab="overview">Overview</button>
  <button class="tabbtn" data-tab="strategy">Strategy Detail</button>
</div>

<div id="tab-overview" class="tabview active">

<div class="combined">
  <div class="cell"><div class="l">Combined trades</div><div class="v">{cm['n']}</div></div>
  <div class="cell"><div class="l">Win rate</div><div class="v">{cm['win']*100:.0f}%</div></div>
  <div class="cell"><div class="l">Open positions</div><div class="v">{cm['open']}</div></div>
  <div class="cell"><div class="l">${cm['cap']//1000}k @ Kelly <span class="tag">per-strat</span></div><div class="v {cret10}">{fmt_money(cm['fin10'])}</div><div class="kd {cret10}">{cm['ret10']*100:+.1f}%</div></div>
  <div class="cell"><div class="l">${cm['cap']//1000}k @ flat $10</div><div class="v small {cretf}">{fmt_money(cm['finflat'])}</div><div class="kd {cretf}">{cm['retflat']*100:+.1f}%</div></div>
  <div class="cell"><div class="l">Realized <span class="tag">flat $10</span></div><div class="v small {_sc(cm['realized'])}">{signed(cm['realized'])}</div></div>
  <div class="cell"><div class="l">Unrealized <span class="tag">open marks</span></div><div class="v small {_sc(cm['unreal'])}">{signed(cm['unreal'])}</div></div>
  <div class="cell"><div class="l">MTM <span class="tag">real+unreal</span></div><div class="v small {_sc(cm['mtm'])}">{signed(cm['mtm'])}</div></div>
  <div class="cell"><div class="l">Peak deployed <span class="tag">≤100% = no leverage</span></div><div class="v small">{cm['peakdep']*100:.0f}%</div></div>
</div>
<div class="note">Realistic <b>hourly-marked</b>, <b>no-leverage</b> cash sim: each strategy trades its <b>own $1,000</b> and the combined is the <b>sum of the {cm['nsleeves']} base sleeves</b> (${cm['cap']//1000}k total) — <b>filter views</b> (e.g. Smart Flow ∩ band) re-slice a ledger already counted and are <b>excluded</b>, so cards don't sum to the total. Positions tie up cash entry→exit (downsized/skipped when committed, never on margin); entries/exits execute on their recorded date, but open positions are marked to their real <b>intraday</b> price between decisions for a smooth curve. <b>Kelly %</b> is <b>walk-forward</b> (out-of-sample) and <b>event-clustered</b>: every trade is staked from the edge known from trades settled <i>before</i> it, averaged one observation per independent settlement day — not per trade, so correlated same-event trades (e.g. several props on one match) can't masquerade as independent evidence. No bet until ≥5 prior independent event-days, no lookahead. The % shown is the rate recommended now; 0 = no proven edge yet → don't bet (capped 50%).</div>

<div class="section-h">Equity curve — $1,000 per strategy · combined = sum of sleeves · hourly marks</div>
<div id="chartbtns">{btns}</div>
<div class="chartwrap">
  <div class="legend"><span><i style="background:#3b82f6"></i>Kelly stake (per strategy)</span><span><i style="background:#e0a93b"></i>flat $10/trade</span></div>
  <svg id="chart" viewBox="0 0 960 340" width="100%" height="340" preserveAspectRatio="xMidYMid meet"></svg>
</div>

<div class="section-h">Per-strategy — own $1,000 · Kelly stake vs flat $10</div>
<div class="grid">{cards}</div>

</div>

<div id="tab-strategy" class="tabview">
  <div class="stratbar">
    <select id="stratSelect">{strat_options}</select>
  </div>
  <div id="stratCards" class="grid" style="margin-bottom:16px"></div>

  <div class="section-h">Deep dive</div>
  <div id="stratAssess" style="margin-bottom:18px"></div>

  <div class="section-h">Equity curve — $1,000 bankroll · hourly marks</div>
  <div class="chartwrap" style="margin-bottom:18px">
    <div class="legend"><span><i style="background:#3b82f6"></i>Kelly stake</span><span><i style="background:#e0a93b"></i>flat $10/trade</span></div>
    <svg id="chart2" viewBox="0 0 960 340" width="100%" height="340" preserveAspectRatio="xMidYMid meet"></svg>
  </div>

  <div class="section-h">Positions — per-trade PnL, flat $10 vs the walk-forward Kelly stake actually chosen</div>
  <div class="postable-wrap">
    <table class="postable" id="posTable">
      <thead><tr><th>Date</th><th>Question</th><th>Entry</th><th>Exit</th><th>Status</th><th>PnL %</th><th>Flat $10 PnL</th><th>Kelly %</th><th>Kelly stake</th><th>Kelly PnL</th></tr></thead>
      <tbody></tbody>
    </table>
  </div>
</div>

<script>
const SERIES = {json.dumps(series)};
const CARDS = {json.dumps(cards_map)};
const ASSESS = {json.dumps(assess_cards)};
const CAP0 = {CAP0};
const nextTs = new Date("{nxt.isoformat()}").getTime();
function tick(){{const d=Math.max(0,nextTs-Date.now());const m=Math.floor(d/60000),s=Math.floor(d%60000/1000);const el=document.getElementById('cd');if(el)el.textContent='in '+m+'m '+String(s).padStart(2,'0')+'s';}}
setInterval(tick,1000);tick();
function fmtTick(iso){{
  const d=new Date(iso); if(isNaN(d)) return iso;
  const mm=String(d.getUTCMonth()+1).padStart(2,'0'),dd=String(d.getUTCDate()).padStart(2,'0'),hh=String(d.getUTCHours()).padStart(2,'0');
  return `${{mm}}-${{dd}} ${{hh}}:00`;
}}
function drawTo(sid, elId){{
  const s=SERIES[sid]; if(!s) return;
  const W=960,H=340,P=46,a=s.pts10,b=s.ptsf,n=a.length;
  const base=a.length?a[0][1]:CAP0;            // starting capital ($1k/strategy, $Nk combined)
  const allY=a.map(p=>p[1]).concat(b.map(p=>p[1]));
  const ymin=Math.min(...allY,base),ymax=Math.max(...allY,base);
  const X=i=>P+(W-2*P)*(n<=1?0.5:i/(n-1));
  const Y=v=>H-P-(H-2*P)*((v-ymin)/((ymax-ymin)||1));
  const line=pts=>pts.map((p,i)=>(i?'L':'M')+X(i).toFixed(1)+' '+Y(p[1]).toFixed(1)).join(' ');
  let g='';
  for(let k=0;k<=4;k++){{const v=ymin+(ymax-ymin)*k/4,y=Y(v);g+=`<line x1="${{P}}" y1="${{y}}" x2="${{W-P}}" y2="${{y}}" stroke="#262d36"/><text x="${{P-8}}" y="${{y+4}}" fill="#8b97a5" font-size="11" text-anchor="end">$${{Math.round(v)}}</text>`;}}
  const yb=Y(base);g+=`<line x1="${{P}}" y1="${{yb}}" x2="${{W-P}}" y2="${{yb}}" stroke="#3a4453" stroke-dasharray="4 4"/>`;
  [0,Math.floor((n-1)/2),n-1].forEach(i=>{{if(i>=0&&i<n)g+=`<text x="${{X(i)}}" y="${{H-14}}" fill="#8b97a5" font-size="11" text-anchor="middle">${{fmtTick(a[i][0])}}</text>`;}});
  g+=`<path d="${{line(b)}}" fill="none" stroke="#e0a93b" stroke-width="2"/>`;
  g+=`<path d="${{line(a)}}" fill="none" stroke="#3b82f6" stroke-width="2.4"/>`;
  document.getElementById(elId).innerHTML=g;
}}
function draw(sid){{ drawTo(sid, 'chart'); }}
document.querySelectorAll('.chartbtn').forEach(btn=>btn.addEventListener('click',()=>{{
  document.querySelectorAll('.chartbtn').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');draw(btn.dataset.sid);
}}));
draw(Object.keys(SERIES)[0]);

// --- Strategy Detail tab ----------------------------------------------------
function escapeHtml(s){{return (s||'').replace(/[&<>"]/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}})[c]);}}
function renderStrategy(sid){{
  const s=SERIES[sid]; if(!s) return;
  document.getElementById('stratCards').innerHTML = CARDS[sid] || '';
  document.getElementById('stratAssess').innerHTML = ASSESS[sid] || '';
  drawTo(sid, 'chart2');
  const tbody=document.querySelector('#posTable tbody');
  const rows=s.positions||[];
  tbody.innerHTML = rows.length ? rows.map(p=>{{
    const scls = p.status==='WON'?'pos':(p.status==='LOST'?'neg':'');
    const pcls = p.pnl_pct>=0?'pos':'neg';
    const kbet = p.kstake>0;
    const kcls = kbet ? (p.kpnl>=0?'pos':'neg') : '';
    return `<tr><td>${{p.xd||p.ed||''}}</td><td>${{escapeHtml(p.q)}}</td>`+
      `<td>${{(p.entry*100).toFixed(1)}}¢</td><td>${{(p.exit*100).toFixed(1)}}¢</td>`+
      `<td class="${{scls}}">${{p.status}}</td>`+
      `<td class="${{pcls}}">${{p.pnl_pct>=0?'+':''}}${{p.pnl_pct}}%</td>`+
      `<td class="${{pcls}}">${{p.pnl10>=0?'+':''}}$${{p.pnl10.toFixed(2)}}</td>`+
      `<td>${{kbet?p.kfrac.toFixed(1)+'%':'—'}}</td>`+
      `<td>${{kbet?'$'+p.kstake.toFixed(2):'—'}}</td>`+
      `<td class="${{kcls}}">${{kbet?(p.kpnl>=0?'+':'')+'$'+p.kpnl.toFixed(2):'—'}}</td></tr>`;
  }}).join('') : '<tr><td colspan="10" style="color:var(--mut)">No positions</td></tr>';
}}
const stratSelect=document.getElementById('stratSelect');
if(stratSelect){{
  stratSelect.addEventListener('change', e=>renderStrategy(e.target.value));
  renderStrategy(stratSelect.value);
}}
document.querySelectorAll('.tabbtn').forEach(btn=>btn.addEventListener('click',()=>{{
  document.querySelectorAll('.tabbtn').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('.tabview').forEach(v=>v.classList.remove('active'));
  document.getElementById('tab-'+btn.dataset.tab).classList.add('active');
  if(btn.dataset.tab==='strategy' && stratSelect) renderStrategy(stratSelect.value);
}}));
</script>
</div></body></html>"""
    OUT.write_text(html, encoding="utf-8")
    print(f"Dashboard written → {OUT}")
    print(f"  combined: {combined['n']} trades, $1k@10%→{fmt_money(combined['fin10'])} ({combined['ret10']*100:+.1f}%), "
          f"flat$10→{fmt_money(combined['finflat'])} ({combined['retflat']*100:+.1f}%)")
    for s in sorted(strategies, key=lambda x: -x["n"]):
        print(f"    {s['label']:<20} n={s['n']:<3} 10%→{fmt_money(s['fin10'])} ({s['ret10']*100:+.0f}%)  flat→{fmt_money(s['finflat'])} ({s['retflat']*100:+.0f}%)")


if __name__ == "__main__":
    generate()
