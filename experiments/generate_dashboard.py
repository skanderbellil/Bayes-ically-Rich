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
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
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
    """Return (resolved, opens). resolved: [{date, entry, ret}] in resolution
    order; opens: [{entry, ret}] unrealized at current_price."""
    if not path.exists():
        return [], []
    df = pd.read_csv(path)
    if entry_col not in df.columns:
        entry_col = "entry_ask" if "entry_ask" in df.columns else ("entry_price" if "entry_price" in df.columns else None)
    if entry_col is None:
        return [], []
    df["_entry"] = _num(df[entry_col]).clip(lower=0.01, upper=0.99)
    df["_cur"] = _num(df.get("current_price"))
    df["_out"] = _num(df.get("outcome"))
    status = df.get("status", pd.Series(["" for _ in range(len(df))])).astype(str)
    out = df["_out"].where(df["_out"].notna(), status.str.lower().map({"won": 1.0, "lost": 0.0}))

    resolved, opens = [], []
    for i, r in df.iterrows():
        e = r["_entry"]
        if pd.isna(e):
            continue
        st = status.iloc[i].lower()
        if st in ("won", "lost") or (pd.notna(out.iloc[i]) and st not in ("open", "watching")):
            o = out.iloc[i]
            if pd.isna(o):
                continue
            d = r.get("exit_date") if pd.notna(r.get("exit_date")) else r.get("entry_date")
            resolved.append({"date": str(d)[:10], "entry": float(e), "ret": float(o) / float(e) - 1.0})
        elif st == "open" and pd.notna(r["_cur"]):
            opens.append({"entry": float(e), "ret": float(r["_cur"]) / float(e) - 1.0})
    resolved.sort(key=lambda x: x["date"])
    return resolved, opens


def equity_curves(resolved):
    """Two equity series from CAP0: 10%-compounding and flat-$10."""
    if not resolved:
        return [], [], dict(n=0, win=0.0, fin10=CAP0, ret10=0.0, finflat=CAP0, retflat=0.0, dd10=0.0)
    start = resolved[0]["date"]
    eq10, eqf, peak, dd, wins = CAP0, CAP0, CAP0, 0.0, 0
    pts10, ptsf = [[start, CAP0]], [[start, CAP0]]
    for t in resolved:
        r = t["ret"]
        eq10 *= (1.0 + STAKE_FRAC * r)
        eqf += FLAT * r
        pts10.append([t["date"], round(eq10, 2)])
        ptsf.append([t["date"], round(eqf, 2)])
        peak = max(peak, eq10); dd = min(dd, eq10 / peak - 1.0); wins += (r > 0)
    n = len(resolved)
    return pts10, ptsf, dict(n=n, win=wins / n, fin10=eq10, ret10=eq10 / CAP0 - 1.0,
                             finflat=eqf, retflat=eqf / CAP0 - 1.0, dd10=dd)


def build():
    strategies, series, all_resolved = [], {}, []
    for fname, ecol, qcol, label in REGISTRY:
        resolved, opens = load_ledger(DATA / fname, ecol)
        all_resolved.extend(resolved)
        pts10, ptsf, k = equity_curves(resolved)
        sid = fname.replace("_positions.csv", "")
        strategies.append(dict(id=sid, label=label, open=len(opens),
                               unreal=sum(FLAT * o["ret"] for o in opens), **k))
        if k["n"]:
            series[sid] = {"label": label, "pts10": pts10, "ptsf": ptsf}
    all_resolved.sort(key=lambda x: x["date"])
    gp10, gpf, gk = equity_curves(all_resolved)
    if gk["n"]:
        series = {"GLOBAL": {"label": "All combined", "pts10": gp10, "ptsf": gpf}, **series}
    combined = dict(label="ALL COMBINED", open=sum(s["open"] for s in strategies),
                    unreal=sum(s["unreal"] for s in strategies), **gk)
    return combined, strategies, series


def fmt_money(v):
    return f"${v:,.0f}"


def kpi_card(s):
    c10 = "pos" if s["ret10"] >= 0 else "neg"
    cf = "pos" if s["retflat"] >= 0 else "neg"
    return f"""
    <div class="card">
      <div class="card-h">{s['label']}</div>
      <div class="card-sub">{s['n']} trades · win {s['win']*100:.0f}% · open {s['open']}</div>
      <div class="kgrid">
        <div class="k"><div class="kl">10% stake</div><div class="kv {c10}">{fmt_money(s['fin10'])}</div><div class="kd {c10}">{s['ret10']*100:+.1f}%</div></div>
        <div class="k"><div class="kl">flat $10</div><div class="kv {cf}">{fmt_money(s['finflat'])}</div><div class="kd {cf}">{s['retflat']*100:+.1f}%</div></div>
      </div>
      <div class="card-f">max DD {s['dd10']*100:.0f}% · unreal (flat) ${s['unreal']:+,.0f}</div>
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
.combined{{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:18px}}
.cell{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px}}
.cell .l{{font-size:11px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}}
.cell .v{{font-size:20px;font-weight:680;margin-top:3px}}.cell .v.small{{font-size:16px}}
.kd{{font-size:12px}}
.section-h{{font-size:12px;color:var(--mut);text-transform:uppercase;letter-spacing:.05em;margin:18px 0 8px}}
#chartbtns{{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px}}
.chartbtn{{font-size:12px;padding:6px 10px}}.chartbtn.active{{background:var(--acc);border-color:var(--acc);color:#fff}}
.chartwrap{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}}
.legend{{display:flex;gap:16px;font-size:12px;color:var(--mut);margin-bottom:6px}}
.legend i{{display:inline-block;width:14px;height:3px;vertical-align:middle;margin-right:5px}}
.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px}}
.card-h{{font-weight:640}}.card-sub{{font-size:12px;color:var(--mut);margin:2px 0 8px}}
.kgrid{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}
.k{{background:#10151b;border:1px solid var(--line);border-radius:8px;padding:8px}}
.kl{{font-size:10px;color:var(--mut);text-transform:uppercase}}.kv{{font-size:17px;font-weight:680}}
.card-f{{font-size:11px;color:var(--mut);margin-top:8px}}.pos{{color:var(--pos)}}.neg{{color:var(--neg)}}
@media(max-width:820px){{.combined{{grid-template-columns:repeat(2,1fr)}}.grid{{grid-template-columns:1fr}}}}
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
  <div class="cell"><div class="l">$1k @ 10% stake</div><div class="v {cret10}">{fmt_money(cm['fin10'])}</div><div class="kd {cret10}">{cm['ret10']*100:+.1f}%</div></div>
  <div class="cell"><div class="l">$1k @ flat $10</div><div class="v small {cretf}">{fmt_money(cm['finflat'])}</div><div class="kd {cretf}">{cm['retflat']*100:+.1f}%</div></div>
  <div class="cell"><div class="l">Open positions</div><div class="v">{cm['open']}</div></div>
</div>

<div class="section-h">Equity curve — $1,000 from start</div>
<div id="chartbtns">{btns}</div>
<div class="chartwrap">
  <div class="legend"><span><i style="background:#3b82f6"></i>10% stake (compounding)</span><span><i style="background:#e0a93b"></i>flat $10/trade</span></div>
  <svg id="chart" viewBox="0 0 960 340" width="100%" height="340" preserveAspectRatio="xMidYMid meet"></svg>
</div>

<div class="section-h">Per-strategy — $1,000 from start · 10% stake vs flat $10</div>
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
  const allY=a.map(p=>p[1]).concat(b.map(p=>p[1]));
  const ymin=Math.min(...allY,CAP0),ymax=Math.max(...allY,CAP0);
  const X=i=>P+(W-2*P)*(n<=1?0.5:i/(n-1));
  const Y=v=>H-P-(H-2*P)*((v-ymin)/((ymax-ymin)||1));
  const line=pts=>pts.map((p,i)=>(i?'L':'M')+X(i).toFixed(1)+' '+Y(p[1]).toFixed(1)).join(' ');
  let g='';
  for(let k=0;k<=4;k++){{const v=ymin+(ymax-ymin)*k/4,y=Y(v);g+=`<line x1="${{P}}" y1="${{y}}" x2="${{W-P}}" y2="${{y}}" stroke="#262d36"/><text x="${{P-8}}" y="${{y+4}}" fill="#8b97a5" font-size="11" text-anchor="end">$${{Math.round(v)}}</text>`;}}
  const yb=Y(CAP0);g+=`<line x1="${{P}}" y1="${{yb}}" x2="${{W-P}}" y2="${{yb}}" stroke="#3a4453" stroke-dasharray="4 4"/>`;
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
