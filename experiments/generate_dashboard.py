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
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
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
    ("macro_positions.csv",             "entry_price", "leader_question", "Macro (Fed cuts)"),
    ("dip_confirm_positions.csv",       "entry_ask",   "question",        "Dip-Confirm YES"),
]


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def load_ledger(path: Path, entry_col: str):
    """Return a list of trades with real entry/exit dates:
    {ed, xd, entry, outcome, current, resolved}. Resolved trades have xd+outcome;
    open trades have current (marked at current_price, held to the end)."""
    if not path.exists():
        return []
    df = pd.read_csv(path)
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

    trades = []
    for i, r in df.iterrows():
        e = r["_entry"]
        if pd.isna(e):
            continue
        st = status.iloc[i].lower()
        ed = str(r.get("entry_date"))[:10] if pd.notna(r.get("entry_date")) else None
        tok = tokens.iloc[i]
        if st in ("won", "lost") or (pd.notna(out.iloc[i]) and st not in ("open", "watching")):
            o = out.iloc[i]
            if pd.isna(o):
                continue
            xd = str(r.get("exit_date"))[:10] if pd.notna(r.get("exit_date")) else ed
            trades.append({"token": tok, "ed": ed or xd, "xd": xd or ed, "entry": float(e),
                           "outcome": float(o), "current": None, "resolved": True})
        elif st == "open":
            cur = float(r["_cur"]) if pd.notna(r["_cur"]) else float(e)
            trades.append({"token": tok, "ed": ed, "xd": None, "entry": float(e),
                           "outcome": None, "current": cur, "resolved": False})
    return [t for t in trades if t["ed"]]


# --- daily price-mark cache (cache resolved markets once, refresh open ones) ----
MARKS_PATH = DATA / "token_daily.csv.gz"


def load_marks():
    if not MARKS_PATH.exists():
        return {}
    try:
        df = pd.read_csv(MARKS_PATH, dtype={"token": str})
        df["date"] = pd.to_datetime(df["date"])
        return {tok: g.set_index("date")["p"].sort_index() for tok, g in df.groupby("token")}
    except Exception:
        return {}


def save_marks(marks):
    rows = [(tok, d.strftime("%Y-%m-%d"), float(p)) for tok, s in marks.items() for d, p in s.items()]
    pd.DataFrame(rows, columns=["token", "date", "p"]).to_csv(MARKS_PATH, index=False, compression="gzip")


def fetch_daily(token):
    """Daily Yes-price series for a token (lazy import; None on any failure)."""
    try:
        from posterioralpha.polymarket.fetch import fetch_token_history
        s = fetch_token_history(str(token), fidelity_minutes=1440, use_cache=False)
        if s is None or len(s) == 0:
            return None
        s.index = pd.to_datetime(s.index)
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
        s = fetch_daily(tok)
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


def walkforward_kelly(trades, min_hist=10, cap=0.5):
    """Per-trade WALK-FORWARD Kelly (no lookahead): trade t is sized from the Kelly
    of only the trades that had SETTLED before t was entered (exit_date <= t entry).
    No bet until `min_hist` prior settles. Returns (fracs[idx], current_rate) where
    current_rate is the all-history estimate (the stake you'd use on the next trade)."""
    settled = sorted(((t["xd"], t["outcome"] / t["entry"] - 1.0)
                      for t in trades if t["resolved"] and t["xd"]), key=lambda x: x[0])
    sx = [d for d, _ in settled]
    sr = [r for _, r in settled]
    import bisect
    fracs = {}
    for idx, t in enumerate(trades):
        k = bisect.bisect_right(sx, t["ed"])            # how many had settled by entry
        prior = sr[:k]
        if len(prior) < min_hist:
            fracs[idx] = 0.0
        else:
            fracs[idx] = round(kelly_fraction(prior, cap=cap) * min(1.0, len(prior) / 20.0), 4)
    cur = round(kelly_fraction(sr, cap=cap) * min(1.0, len(sr) / 20.0), 4) if len(sr) >= min_hist else 0.0
    return fracs, cur


def sim(trades, mode, marks=None, frac=STAKE_FRAC, fracs=None):
    """Realistic daily cash sim from $1,000 — NO LEVERAGE. A position ties up cash
    from entry to exit; new trades are sized min(target, cash) so overlapping
    positions can't be funded on margin. Each day, OPEN positions are marked to
    their real daily price (from the marks cache), giving a smooth equity curve
    instead of cost-held steps. mode 'pct' = `frac` of equity (Kelly), 'flat'=$10."""
    marks = marks or {}
    if not trades:
        return dict(pts=[["", CAP0]], fin=CAP0, ret=0.0, dd=0.0, realized=0.0, unreal=0.0,
                    taken=0, constrained=0, maxconc=0, peakdep=0.0)
    buys, sells = {}, {}
    for idx, t in enumerate(trades):
        buys.setdefault(t["ed"], []).append(idx)
        if t["resolved"]:
            sells.setdefault(max(t["xd"], t["ed"]), []).append(idx)
    start = min(t["ed"] for t in trades)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    end = max([today] + [t["xd"] for t in trades if t["resolved"] and t["xd"]])
    days = [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end, freq="D")]

    def mp(idx, day):
        t = trades[idx]; s = marks.get(t["token"])
        if s is not None:
            v = s.asof(pd.Timestamp(day))
            if pd.notna(v):
                return float(v)
        return t["current"] if t["current"] is not None else t["entry"]

    cash, hold, pts = CAP0, {}, []
    taken = constrained = maxconc = 0
    realized = peakdep = 0.0
    for day in days:
        for idx in sells.get(day, []):
            h = hold.pop(idx, None)
            if h:
                proc = h["shares"] * trades[idx]["outcome"]
                cash += proc; realized += proc - h["cost"]
        for idx in buys.get(day, []):
            eq = cash + sum(h["shares"] * mp(i, day) for i, h in hold.items())
            fr = (fracs.get(idx, 0.0) if fracs is not None else frac)
            target = fr * eq if mode == "pct" else FLAT
            if target <= 1e-9:
                continue                               # Kelly says don't bet (no edge / no history yet)
            stake = min(target, cash)
            if stake > 1e-6:
                if stake < target - 1e-6:
                    constrained += 1                   # capital-limited (real margin constraint)
                hold[idx] = {"shares": stake / trades[idx]["entry"], "cost": stake}
                cash -= stake; taken += 1
                maxconc = max(maxconc, len(hold))
        cost = sum(h["cost"] for h in hold.values())
        mv = sum(h["shares"] * mp(i, day) for i, h in hold.items())
        peakdep = max(peakdep, cost / (cost + cash) if (cost + cash) > 0 else 0.0)
        pts.append([day, round(cash + mv, 2)])

    final = pts[-1][1] if pts else CAP0
    unreal = sum(hold[i]["shares"] * mp(i, days[-1]) - hold[i]["cost"] for i in hold)
    peak, dd = CAP0, 0.0
    for _, e in pts:
        peak = max(peak, e); dd = min(dd, e / peak - 1.0 if peak > 0 else 0.0)
    return dict(pts=pts, fin=final, ret=final / CAP0 - 1.0, dd=dd, realized=realized,
                unreal=unreal, taken=taken, constrained=constrained, maxconc=maxconc, peakdep=peakdep)


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
    return k, {"label": label, "pts10": p["pts"], "ptsf": f["pts"]}


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


def build(fetch_marks=True):
    per_ledger = []
    all_trades = []
    for fname, ecol, qcol, label in REGISTRY:
        trades = load_ledger(DATA / fname, ecol)
        per_ledger.append((fname, label, trades))
        all_trades.extend(trades)

    # cache-once + refresh-open: a token is static only if ALL its trades are resolved
    token_resolved = {}
    for t in all_trades:
        token_resolved[t["token"]] = token_resolved.get(t["token"], True) and t["resolved"]
    marks = update_marks(token_resolved) if fetch_marks else load_marks()

    strategies, series, active_ser = [], {}, []
    for fname, label, trades in per_ledger:
        sid = fname.replace("_positions.csv", "")
        k, ser = kpis_for(trades, label, sid, marks)
        strategies.append(k)
        if k["n"] or k["open"]:
            series[sid] = ser
            active_ser.append((k, ser))

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
    return f"""
    <div class="card">
      <div class="card-h">{s['label']}</div>
      <div class="card-sub">{s['n']} trades · win {s['win']*100:.0f}% · open {s['open']}</div>
      <div class="kgrid">
        <div class="k"><div class="kl">Kelly {s['kelly']*100:.0f}%</div><div class="kv {c10}">{fmt_money(s['fin10'])}</div><div class="kd {c10}">{s['ret10']*100:+.1f}%</div></div>
        <div class="k"><div class="kl">flat $10</div><div class="kv {cf}">{fmt_money(s['finflat'])}</div><div class="kd {cf}">{s['retflat']*100:+.1f}%</div></div>
      </div>
      <div class="pnl3">
        <div><span class="kl">Realized</span><span class="{_sc(s['realized'])}">{signed(s['realized'])}</span></div>
        <div><span class="kl">Unrealized</span><span class="{_sc(s['unreal'])}">{signed(s['unreal'])}</span></div>
        <div><span class="kl">MTM</span><span class="{_sc(s['mtm'])}">{signed(s['mtm'])}</span></div>
      </div>
      <div class="card-f">max DD {s['dd10']*100:.0f}% · peak {s['peakdep']*100:.0f}% deployed · max {s['maxconc']} open{cap_note(s)}</div>
    </div>"""


def cap_note(s):
    return f" · {s['constrained']} entries capital-capped" if s.get("constrained") else ""


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
@media(max-width:600px){{.wrap{{padding:12px}}.bar{{gap:8px}}h1{{font-size:16px;width:100%}}
  .runinfo{{width:100%;order:3}}.bar button{{flex:1}}.cell .v{{font-size:18px}}
  #chart{{height:260px}}.section-h{{margin:14px 0 6px}}}}
</style></head><body><div class="wrap">

<div class="bar">
  <h1>📈 Paper-trade dashboard</h1>
  <div class="spacer"></div>
  <div class="runinfo">last run <b>{now:%Y-%m-%d %H:%M} UTC</b> · next <b>{nxt:%H:%M}</b> (<span id="cd"></span>)</div>
  <button onclick="location.reload(true)">↻ Refresh</button>
  <button class="btn-run" onclick="window.open('{RUN_URL}','_blank')">▶ Run now</button>
</div>

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
<div class="note">Realistic daily, <b>no-leverage</b> cash sim: each strategy trades its <b>own $1,000</b> and the combined is the <b>sum of the {cm['nsleeves']} sleeves</b> (${cm['cap']//1000}k total). Positions tie up cash entry→exit (downsized/skipped when committed, never on margin); open positions are marked to their real daily price. <b>Kelly %</b> is <b>walk-forward</b> (out-of-sample): every trade is staked only from the edge known from trades settled <i>before</i> it — no bet until ≥10 prior settles, no lookahead. The % shown is the rate recommended now; 0 = no proven edge → don't bet (capped 50%).</div>

<div class="section-h">Equity curve — $1,000 per strategy · combined = sum of sleeves</div>
<div id="chartbtns">{btns}</div>
<div class="chartwrap">
  <div class="legend"><span><i style="background:#3b82f6"></i>Kelly stake (per strategy)</span><span><i style="background:#e0a93b"></i>flat $10/trade</span></div>
  <svg id="chart" viewBox="0 0 960 340" width="100%" height="340" preserveAspectRatio="xMidYMid meet"></svg>
</div>

<div class="section-h">Per-strategy — own $1,000 · Kelly stake vs flat $10</div>
<div class="grid">{cards}</div>

<script>
const SERIES = {json.dumps(series)};
const CAP0 = {CAP0};
const nextTs = new Date("{nxt.isoformat()}").getTime();
function tick(){{const d=Math.max(0,nextTs-Date.now());const m=Math.floor(d/60000),s=Math.floor(d%60000/1000);const el=document.getElementById('cd');if(el)el.textContent='in '+m+'m '+String(s).padStart(2,'0')+'s';}}
setInterval(tick,1000);tick();
function draw(sid){{
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
  [0,Math.floor((n-1)/2),n-1].forEach(i=>{{if(i>=0&&i<n)g+=`<text x="${{X(i)}}" y="${{H-14}}" fill="#8b97a5" font-size="11" text-anchor="middle">${{a[i][0]}}</text>`;}});
  g+=`<path d="${{line(b)}}" fill="none" stroke="#e0a93b" stroke-width="2"/>`;
  g+=`<path d="${{line(a)}}" fill="none" stroke="#3b82f6" stroke-width="2.4"/>`;
  document.getElementById('chart').innerHTML=g;
}}
document.querySelectorAll('.chartbtn').forEach(btn=>btn.addEventListener('click',()=>{{
  document.querySelectorAll('.chartbtn').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');draw(btn.dataset.sid);
}}));
draw(Object.keys(SERIES)[0]);
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
