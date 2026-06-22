#!/usr/bin/env python3
"""Generate a self-contained HTML dashboard from the three paper-trade ledgers.

Output: data/paper_trade/dashboard.html
Run directly or called by the cron workflows after each ledger update.
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "paper_trade"
OUT  = DATA / "dashboard.html"

REPO        = "skanderbellil/Bayes-ically-Rich"
ACTIONS_URL = f"https://github.com/{REPO}/actions/workflows/close_position.yml"


# ---------------------------------------------------------------------------
# Load & compute MTM for one ledger
# ---------------------------------------------------------------------------

def load_strategy(path: Path, entry_col: str, q_col: str,
                  label: str, strategy_id: str) -> dict:
    if not path.exists():
        return {"label": label, "strategy_id": strategy_id,
                "open": [], "resolved": [], "unrealized": 0,
                "realized": 0, "nW": 0, "nL": 0, "nF": 0}
    df = pd.read_csv(path)
    for c in [entry_col, "current_price", "pnl", "bet_fraction"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "bet_fraction" not in df.columns or df["bet_fraction"].isna().all():
        df["bet_fraction"] = 0.10

    opened   = df[df["status"] == "open"].copy()
    resolved = df[df["status"].notna() & (df["status"] != "open")].copy()
    opened["mtm"] = (opened["current_price"] / opened[entry_col] - 1) * opened["bet_fraction"]

    def safe_str(val, fallback=""):
        return str(val) if pd.notna(val) else fallback

    def row_to_dict(r, include_mtm=False):
        d = {
            "question":   str(r[q_col]),
            "entry":      round(float(r[entry_col]), 4) if pd.notna(r[entry_col]) else None,
            "current":    round(float(r["current_price"]), 4) if "current_price" in r and pd.notna(r["current_price"]) else None,
            "status":     str(r["status"]),
            "pnl":        round(float(r["pnl"]), 4) if pd.notna(r.get("pnl")) else None,
            "exit_date":  safe_str(r.get("exit_date")),
            "entry_date": safe_str(r.get("entry_date")),
        }
        if include_mtm:
            d["mtm"] = round(float(r["mtm"]), 4) if pd.notna(r.get("mtm")) else None
        return d

    open_rows     = [row_to_dict(r, include_mtm=True)
                     for _, r in opened.sort_values("mtm", ascending=False).iterrows()]
    resolved_rows = [row_to_dict(r)
                     for _, r in resolved.iterrows()]

    return {
        "label":       label,
        "strategy_id": strategy_id,
        "open":        open_rows,
        "resolved":    resolved_rows,
        "unrealized":  round(float(opened["mtm"].sum()), 4),
        "realized":    round(float(resolved["pnl"].dropna().sum()), 4),
        "nW": int((resolved["status"] == "won").sum()),
        "nL": int((resolved["status"] == "lost").sum()),
        "nF": int((resolved["status"] == "flipped").sum()),
    }


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1">
<title>Paper-Trade Dashboard</title>
<style>
:root {
  --bg:#0f1117; --card:#1a1d27; --border:#2a2d3e;
  --green:#22c55e; --red:#ef4444; --yellow:#f59e0b;
  --blue:#3b82f6; --text:#e2e8f0; --muted:#94a3b8; --flip:#a78bfa;
}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--text);font-family:'Inter',system-ui,sans-serif;font-size:14px;padding-bottom:72px;}

/* ── PAGES ── */
.page{display:none;padding:16px 16px 28px;}
.page.active{display:block;}
.page-hdr{margin-bottom:14px;}
.page-hdr h1{font-size:20px;font-weight:700;}
.page-hdr h2{font-size:17px;font-weight:700;}
.updated{font-size:11px;color:var(--muted);margin-top:2px;}

/* ── BOTTOM NAV ── */
.bottom-nav{
  position:fixed;bottom:0;left:0;right:0;
  background:var(--card);border-top:1px solid var(--border);
  display:flex;z-index:50;
  padding-bottom:env(safe-area-inset-bottom,0px);
}
.nav-btn{
  flex:1;background:none;border:none;padding:10px 0 8px;
  display:flex;flex-direction:column;align-items:center;gap:2px;
  cursor:pointer;color:var(--muted);transition:color .15s;
  -webkit-tap-highlight-color:transparent;
}
.nav-btn.active{color:var(--blue);}
.nav-icon{font-size:18px;line-height:1;}
.nav-lbl{font-size:10px;font-weight:500;}

/* ── DESKTOP TABS (sticky header, wide screens only) ── */
.dtabs-wrap{
  display:none;background:var(--card);border-bottom:1px solid var(--border);
  position:sticky;top:0;z-index:40;
}
.dtabs{display:flex;max-width:1200px;margin:0 auto;padding:0 20px;}
.dtab{
  background:none;border:none;padding:14px 16px;font-size:13px;font-weight:500;
  color:var(--muted);cursor:pointer;border-bottom:2px solid transparent;
  margin-bottom:-1px;transition:color .15s;white-space:nowrap;
}
.dtab.active{color:var(--blue);border-bottom-color:var(--blue);}

/* ── CARDS ── */
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:16px;margin-bottom:12px;}
.card-title{font-size:14px;font-weight:600;margin-bottom:10px;display:flex;align-items:center;gap:8px;flex-wrap:wrap;}

/* ── KPIs ── */
.port-kpis{display:grid;grid-template-columns:repeat(2,1fr);gap:8px;margin-bottom:14px;}
.port-kpi{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:12px 14px;}
.kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:10px;}
.kpi{background:rgba(255,255,255,.04);border-radius:8px;padding:10px 12px;}
.kpi-lbl{font-size:10px;color:var(--muted);margin-bottom:2px;text-transform:uppercase;letter-spacing:.4px;}
.kpi-val{font-size:18px;font-weight:700;}
.port-kpi .kpi-lbl{font-size:10px;color:var(--muted);margin-bottom:3px;text-transform:uppercase;letter-spacing:.4px;}
.port-kpi .kpi-val{font-size:22px;font-weight:700;}

/* ── BADGES & PILLS ── */
.badge{font-size:11px;padding:2px 8px;border-radius:99px;font-weight:500;}
.badge-green{background:rgba(34,197,94,.18);color:var(--green);}
.badge-red{background:rgba(239,68,68,.18);color:var(--red);}
.badge-grey{background:rgba(148,163,184,.12);color:var(--muted);}
.pill{display:inline-block;font-size:10px;padding:1px 6px;border-radius:4px;font-weight:600;}
.pill-won{background:rgba(34,197,94,.18);color:var(--green);}
.pill-lost{background:rgba(239,68,68,.18);color:var(--red);}
.pill-flipped{background:rgba(167,139,250,.18);color:var(--flip);}
.pill-open{background:rgba(59,130,246,.12);color:var(--blue);}

/* ── COLOR HELPERS ── */
.pos{color:var(--green);}.neg{color:var(--red);}.neu{color:var(--text);}

/* ── W/L/F BAR ── */
.wlf-row{display:flex;align-items:center;gap:8px;margin-bottom:8px;}
.wlf-bar{flex:1;height:5px;border-radius:3px;overflow:hidden;display:flex;gap:1px;}
.wlf-seg{height:100%;border-radius:2px;}
.wlf-lbl{font-size:11px;color:var(--muted);white-space:nowrap;}

/* ── TABLES ── */
.tbl-wrap{overflow-x:auto;-webkit-overflow-scrolling:touch;margin-top:8px;}
table{width:100%;border-collapse:collapse;font-size:12px;}
th{text-align:left;padding:5px 8px;color:var(--muted);font-weight:500;border-bottom:1px solid var(--border);white-space:nowrap;}
td{padding:5px 8px;border-bottom:1px solid rgba(42,45,62,.6);vertical-align:middle;}
tr:last-child td{border-bottom:none;}
.q{max-width:180px;word-break:break-word;}

/* ── STRATEGY FILTER PILLS ── */
.seg{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px;}
.seg-btn{
  background:var(--card);border:1px solid var(--border);border-radius:6px;
  padding:5px 11px;font-size:12px;font-weight:500;color:var(--muted);
  cursor:pointer;transition:all .15s;-webkit-tap-highlight-color:transparent;
}
.seg-btn.active{background:var(--blue);border-color:var(--blue);color:#fff;}

/* ── CUT BUTTON ── */
.cut-btn{
  background:rgba(239,68,68,.1);color:var(--red);
  border:1px solid rgba(239,68,68,.25);border-radius:4px;
  padding:3px 8px;font-size:10px;font-weight:600;
  cursor:pointer;white-space:nowrap;transition:background .15s;
}
.cut-btn:hover{background:rgba(239,68,68,.25);}

/* ── MODAL (slides up from bottom on mobile) ── */
.modal-backdrop{
  display:none;position:fixed;inset:0;background:rgba(0,0,0,.75);
  z-index:100;align-items:flex-end;justify-content:center;
}
.modal-backdrop.open{display:flex;}
.modal{
  background:#1a1d27;border:1px solid #2a2d3e;
  border-radius:16px 16px 0 0;padding:24px;
  width:100%;max-height:90vh;overflow-y:auto;
}
.modal h3{font-size:15px;margin-bottom:6px;}
.modal-q{font-size:12px;color:var(--muted);margin-bottom:16px;line-height:1.5;}
.modal-section{margin-bottom:14px;}
.modal-lbl{font-size:10px;text-transform:uppercase;letter-spacing:.5px;color:var(--muted);margin-bottom:5px;}
.code-box{
  background:#0f1117;border:1px solid #2a2d3e;border-radius:6px;
  padding:10px 12px;font-family:'Menlo','Consolas',monospace;
  font-size:11px;color:#e2e8f0;word-break:break-all;line-height:1.6;white-space:pre-wrap;
}
.modal-actions{display:flex;gap:8px;margin-top:16px;flex-wrap:wrap;}
.btn{
  padding:10px 14px;border-radius:8px;font-size:13px;font-weight:600;
  cursor:pointer;border:none;transition:background .15s;
  -webkit-tap-highlight-color:transparent;
}
.btn-copy{background:rgba(59,130,246,.18);color:var(--blue);border:1px solid rgba(59,130,246,.35);}
.btn-copy:hover{background:rgba(59,130,246,.32);}
.btn-gh{
  background:rgba(34,197,94,.12);color:var(--green);
  border:1px solid rgba(34,197,94,.28);text-decoration:none;
  display:inline-flex;align-items:center;gap:5px;
}
.btn-gh:hover{background:rgba(34,197,94,.25);}
.btn-cancel{background:rgba(255,255,255,.05);color:var(--muted);border:1px solid #2a2d3e;}
.btn-cancel:hover{background:rgba(255,255,255,.1);}

/* ── EQUITY CHART ── */
.eq-sub{font-size:12px;color:var(--muted);margin-bottom:10px;}

/* ── META ── */
.meta{font-size:11px;color:var(--muted);margin-top:4px;line-height:1.7;padding:14px 16px;border-top:1px solid var(--border);}

/* ── DESKTOP OVERRIDES ── */
@media(min-width:640px){
  body{padding-bottom:0;}
  .bottom-nav{display:none;}
  .dtabs-wrap{display:block;}
  .page{max-width:1200px;margin:0 auto;padding:24px;}
  .port-kpis{grid-template-columns:repeat(4,1fr);}
  .modal-backdrop{align-items:center;padding:20px;}
  .modal{border-radius:14px;max-width:520px;max-height:none;}
  .strategy-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px;}
  .strategy-grid .card{margin-bottom:0;}
  .kpi-val{font-size:20px;}
}
@media(max-width:380px){
  .hide-xs{display:none;}
}
</style>
</head>
<body>

<!-- Desktop sticky tab bar -->
<div class="dtabs-wrap">
  <div class="dtabs">
    <button class="dtab active" onclick="showPage('overview')" id="dtab-overview">📊 Overview</button>
    <button class="dtab" onclick="showPage('open')" id="dtab-open">📋 Open</button>
    <button class="dtab" onclick="showPage('equity')" id="dtab-equity">📈 Equity</button>
    <button class="dtab" onclick="showPage('history')" id="dtab-history">🏁 History</button>
  </div>
</div>

<!-- ── OVERVIEW ── -->
<div id="page-overview" class="page active">
  <div class="page-hdr">
    <h1>Polymarket Paper-Trade</h1>
    <div class="updated">Last updated: <strong id="ts">—</strong></div>
  </div>
  <div id="port-kpis" class="port-kpis"></div>
  <div id="strategy-grid" class="strategy-grid"></div>
  <div class="meta">
    10 % of bankroll per bet · PnL shown as % of that allocation<br>
    <strong>Mid-priced YES</strong>: buy YES tokens in [0.10, 0.50] ~5 d before resolution, hold to settlement.<br>
    <strong>Smart Flow</strong>: follow ≥ 3 smart-wallet consensus buys, exit on reversal signal.<br>
    <strong>Macro</strong>: hold dominant leg of macro resolution markets until year-end.
  </div>
</div>

<!-- ── OPEN POSITIONS ── -->
<div id="page-open" class="page">
  <div class="page-hdr"><h2>Open Positions</h2></div>
  <div id="open-filter" class="seg"></div>
  <div id="open-list"></div>
</div>

<!-- ── EQUITY CURVES ── -->
<div id="page-equity" class="page">
  <div class="page-hdr"><h2>Equity Curves</h2></div>
  <div id="equity-charts"></div>
</div>

<!-- ── HISTORY ── -->
<div id="page-history" class="page">
  <div class="page-hdr"><h2>History</h2></div>
  <div id="history-list"></div>
</div>

<!-- Bottom nav (mobile) -->
<nav class="bottom-nav">
  <button class="nav-btn active" onclick="showPage('overview')" id="nav-overview">
    <span class="nav-icon">📊</span><span class="nav-lbl">Overview</span>
  </button>
  <button class="nav-btn" onclick="showPage('open')" id="nav-open">
    <span class="nav-icon">📋</span><span class="nav-lbl">Open</span>
  </button>
  <button class="nav-btn" onclick="showPage('equity')" id="nav-equity">
    <span class="nav-icon">📈</span><span class="nav-lbl">Equity</span>
  </button>
  <button class="nav-btn" onclick="showPage('history')" id="nav-history">
    <span class="nav-icon">🏁</span><span class="nav-lbl">History</span>
  </button>
</nav>

<!-- Cut-position modal -->
<div id="cut-modal" class="modal-backdrop" onclick="hideCut()">
  <div class="modal" onclick="event.stopPropagation()">
    <h3>✂ Close position</h3>
    <div class="modal-q" id="modal-q"></div>
    <div class="modal-section">
      <div class="modal-lbl">Run locally</div>
      <div class="code-box" id="modal-cmd"></div>
    </div>
    <div class="modal-section">
      <div class="modal-lbl">GitHub Actions — paste these when prompted</div>
      <div class="code-box" id="modal-inputs"></div>
    </div>
    <div class="modal-actions">
      <button class="btn btn-copy" id="copy-btn" onclick="copyCmd()">📋 Copy command</button>
      <a class="btn btn-gh" id="modal-gh" href="" target="_blank" rel="noopener">⚡ Open Actions</a>
      <button class="btn btn-cancel" onclick="hideCut()">Cancel</button>
    </div>
  </div>
</div>

<script>
const DATA        = %%DATA%%;
const ACTIONS_URL = %%ACTIONS_URL%%;
const S_COLORS    = ['#3b82f6', '#22c55e', '#f59e0b', '#a78bfa'];

// ── NAVIGATION ───────────────────────────────────────────────────────────────
function showPage(id) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-btn, .dtab').forEach(b => b.classList.remove('active'));
  document.getElementById('page-' + id).classList.add('active');
  const nb = document.getElementById('nav-' + id);   if (nb) nb.classList.add('active');
  const dt = document.getElementById('dtab-' + id);  if (dt) dt.classList.add('active');
  window.scrollTo(0, 0);
}

// ── FORMATTERS ───────────────────────────────────────────────────────────────
function fmtPct(v) {
  if (v === null || v === undefined) return '—';
  return (v >= 0 ? '+' : '') + (v * 100).toFixed(1) + '%';
}
function cls(v)        { return v > 0.001 ? 'pos' : v < -0.001 ? 'neg' : 'neu'; }
function pill(s)       { return `<span class="pill pill-${s}">${s}</span>`; }
const esc = s => String(s)
  .replace(/&/g,'&amp;').replace(/"/g,'&quot;')
  .replace(/</g,'&lt;').replace(/>/g,'&gt;');

// ── SPARKLINE ────────────────────────────────────────────────────────────────
function miniTrack(entry, current) {
  if (entry == null || current == null) return '<span style="color:var(--border)">—</span>';
  const W = 80, H = 14, mid = 7;
  const clamp = v => Math.min(Math.max(v, 0), 1);
  const ex = Math.round(clamp(entry)   * W);
  const cx = Math.round(clamp(current) * W);
  const col = current >= entry ? '#22c55e' : '#ef4444';
  const lx = Math.min(ex, cx), rx = Math.max(ex, cx);
  return [
    `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" style="display:block">`,
    `<rect x="0" y="${mid-1}" width="${W}" height="2" fill="#2a2d3e" rx="1"/>`,
    `<rect x="${lx}" y="${mid-3}" width="${Math.max(rx-lx,1)}" height="6" fill="${col}" opacity="0.3" rx="2"/>`,
    `<line x1="${ex}" y1="2" x2="${ex}" y2="${H-2}" stroke="#94a3b8" stroke-width="1.5" stroke-linecap="round"/>`,
    `<circle cx="${cx}" cy="${mid}" r="3.5" fill="${col}"/>`,
    '</svg>',
  ].join('');
}

// ── W/L/F BAR ────────────────────────────────────────────────────────────────
function wlfRow(nW, nL, nF) {
  const tot = nW + nL + nF;
  if (!tot) return '';
  const segs = [
    nW ? `<div class="wlf-seg" style="width:${(nW/tot*100).toFixed(1)}%;background:var(--green)"></div>` : '',
    nL ? `<div class="wlf-seg" style="width:${(nL/tot*100).toFixed(1)}%;background:var(--red)"></div>`   : '',
    nF ? `<div class="wlf-seg" style="width:${(nF/tot*100).toFixed(1)}%;background:var(--flip)"></div>`  : '',
  ].join('');
  return `<div class="wlf-row">
    <div class="wlf-bar">${segs}</div>
    <span class="wlf-lbl">${nW}W&nbsp;${nL}L${nF ? '&nbsp;' + nF + 'F' : ''}</span>
  </div>`;
}

// ── EQUITY CURVE DATA ─────────────────────────────────────────────────────────
function strategyEquity(s) {
  const today = DATA.generated.split(' ')[0];

  // Find earliest entry date to anchor the curve at 0
  const allDates = [...s.open, ...s.resolved].map(r => r.entry_date).filter(Boolean).sort();
  const pts = allDates.length ? [{date: allDates[0], cum: 0}] : [];

  // Walk resolved positions in exit-date order
  const closed = s.resolved
    .filter(r => r.exit_date && r.pnl != null)
    .sort((a, b) => a.exit_date.localeCompare(b.exit_date));

  let cum = 0;
  for (const r of closed) {
    cum += r.pnl;
    if (pts.length && pts[pts.length - 1].date === r.exit_date) {
      pts[pts.length - 1].cum = +cum.toFixed(4);
    } else {
      pts.push({date: r.exit_date, cum: +cum.toFixed(4)});
    }
  }

  // Today's trailing point = realized + unrealized
  const todayCum = +(cum + s.unrealized).toFixed(4);
  if (pts.length && pts[pts.length - 1].date === today) {
    pts[pts.length - 1].cum = todayCum;
  } else {
    pts.push({date: today, cum: todayCum});
  }

  return pts;
}

function portfolioEquity() {
  const today = DATA.generated.split(' ')[0];

  const allStartDates = DATA.strategies
    .flatMap(s => [...s.open, ...s.resolved].map(r => r.entry_date).filter(Boolean))
    .sort();
  const pts = allStartDates.length ? [{date: allStartDates[0], cum: 0}] : [];

  const allClosed = DATA.strategies
    .flatMap(s => s.resolved.filter(r => r.exit_date && r.pnl != null))
    .sort((a, b) => a.exit_date.localeCompare(b.exit_date));

  let cum = 0;
  for (const r of allClosed) {
    cum += r.pnl;
    if (pts.length && pts[pts.length - 1].date === r.exit_date) {
      pts[pts.length - 1].cum = +cum.toFixed(4);
    } else {
      pts.push({date: r.exit_date, cum: +cum.toFixed(4)});
    }
  }

  const totalUnrealized = DATA.strategies.reduce((s, d) => s + d.unrealized, 0);
  const todayCum = +(cum + totalUnrealized).toFixed(4);
  if (pts.length && pts[pts.length - 1].date === today) {
    pts[pts.length - 1].cum = todayCum;
  } else {
    pts.push({date: today, cum: todayCum});
  }

  return pts;
}

// ── EQUITY SVG ────────────────────────────────────────────────────────────────
function equitySVG(pts, color) {
  const W = 320, H = 155;
  const pad = {t: 18, r: 14, b: 28, l: 46};
  const cW = W - pad.l - pad.r, cH = H - pad.t - pad.b;

  if (!pts || pts.length < 2) {
    return `<svg viewBox="0 0 ${W} ${H}" width="100%" style="display:block">
      <text x="${W/2}" y="${H/2}" fill="#94a3b8" font-size="11"
            text-anchor="middle" dominant-baseline="middle">No resolved trades yet</text>
    </svg>`;
  }

  const cums  = pts.map(p => p.cum);
  const minC  = Math.min(Math.min(...cums), 0);
  const maxC  = Math.max(Math.max(...cums), 0);
  const range = maxC - minC || 0.001;
  const n     = pts.length;

  const xOf = i => pad.l + (i / (n - 1)) * cW;
  const yOf = c => pad.t + cH - ((c - minC) / range) * cH;
  const y0  = yOf(0);

  // Polyline + area
  const pstr  = pts.map((p, i) => `${xOf(i).toFixed(1)},${yOf(p.cum).toFixed(1)}`).join(' ');
  const areaD = `M${xOf(0).toFixed(1)},${y0.toFixed(1)} ` +
    pts.map((p, i) => `L${xOf(i).toFixed(1)},${yOf(p.cum).toFixed(1)}`).join(' ') +
    ` L${xOf(n-1).toFixed(1)},${y0.toFixed(1)} Z`;

  const lastCum  = cums[n - 1];
  const dotColor = lastCum >= 0 ? '#22c55e' : '#ef4444';

  // Y axis ticks: zero (solid), max, min
  const ticks = [{v: 0, y: y0, lbl: '0%'}];
  if (maxC > 0.005) ticks.push({v: maxC, y: pad.t,      lbl: (maxC * 100).toFixed(0) + '%'});
  if (minC < -0.005) ticks.push({v: minC, y: pad.t + cH, lbl: (minC * 100).toFixed(0) + '%'});

  const gridLines = ticks.map(t =>
    `<line x1="${pad.l}" y1="${t.y.toFixed(1)}" x2="${(pad.l+cW).toFixed(1)}" y2="${t.y.toFixed(1)}"
           stroke="#2a2d3e" stroke-width="1"${t.v === 0 ? '' : ' stroke-dasharray="3,2"'}/>`
  ).join('');
  const yLabels = ticks.map(t =>
    `<text x="${pad.l - 5}" y="${(t.y + 3.5).toFixed(1)}"
           fill="#94a3b8" font-size="9" text-anchor="end">${t.lbl}</text>`
  ).join('');

  // X axis labels: first, mid (if enough points), last
  const xLbl = (i, anchor) =>
    `<text x="${xOf(i).toFixed(1)}" y="${(H - 6).toFixed(1)}"
           fill="#94a3b8" font-size="9" text-anchor="${anchor}">${pts[i].date.slice(5)}</text>`;
  let xLabels = xLbl(0, 'middle') + xLbl(n - 1, 'middle');
  if (n > 4) {
    const mi = Math.round((n - 1) / 2);
    xLabels += xLbl(mi, 'middle');
  }

  // Segment step-fill (positive green / negative red bands)
  const bands = [];
  for (let i = 0; i < n - 1; i++) {
    const x1 = xOf(i), x2 = xOf(i + 1);
    const c1 = pts[i].cum, c2 = pts[i + 1].cum;
    const yTop1 = yOf(Math.max(c1, 0)), yTop2 = yOf(Math.max(c2, 0));
    const yBot1 = yOf(Math.min(c1, 0)), yBot2 = yOf(Math.min(c2, 0));
    if (Math.max(c1, c2) > 0)
      bands.push(`<polygon points="${x1.toFixed(1)},${y0.toFixed(1)} ${x1.toFixed(1)},${yTop1.toFixed(1)} ${x2.toFixed(1)},${yTop2.toFixed(1)} ${x2.toFixed(1)},${y0.toFixed(1)}" fill="#22c55e" opacity="0.10"/>`);
    if (Math.min(c1, c2) < 0)
      bands.push(`<polygon points="${x1.toFixed(1)},${y0.toFixed(1)} ${x1.toFixed(1)},${yBot1.toFixed(1)} ${x2.toFixed(1)},${yBot2.toFixed(1)} ${x2.toFixed(1)},${y0.toFixed(1)}" fill="#ef4444" opacity="0.10"/>`);
  }

  // Dot labels for every resolved point (skip first anchor)
  const dotLabels = [];
  pts.slice(1, -1).forEach((p, ii) => {
    const i = ii + 1;
    const x = xOf(i), y = yOf(p.cum);
    const dc = p.cum >= 0 ? '#22c55e' : '#ef4444';
    dotLabels.push(`<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="2.5" fill="${dc}" opacity="0.8"/>`);
  });

  return `<svg viewBox="0 0 ${W} ${H}" width="100%" style="display:block">
    ${gridLines}
    ${yLabels}
    ${bands.join('')}
    <polyline points="${pstr}" fill="none" stroke="${color}" stroke-width="2"
              stroke-linejoin="round" stroke-linecap="round"/>
    ${dotLabels.join('')}
    <circle cx="${xOf(n-1).toFixed(1)}" cy="${yOf(lastCum).toFixed(1)}"
            r="4.5" fill="${dotColor}" stroke="var(--bg)" stroke-width="2"/>
    ${xLabels}
  </svg>`;
}

// ── PAGE: OVERVIEW ────────────────────────────────────────────────────────────
function buildOverview() {
  const totalU   = DATA.strategies.reduce((s, d) => s + d.unrealized, 0);
  const totalR   = DATA.strategies.reduce((s, d) => s + d.realized,   0);
  const totalMTM = totalU + totalR;
  const totalO   = DATA.strategies.reduce((s, d) => s + d.open.length, 0);

  document.getElementById('port-kpis').innerHTML = `
    <div class="port-kpi">
      <div class="kpi-lbl">Portfolio MTM</div>
      <div class="kpi-val ${cls(totalMTM)}">${fmtPct(totalMTM)}</div>
    </div>
    <div class="port-kpi">
      <div class="kpi-lbl">Unrealized</div>
      <div class="kpi-val ${cls(totalU)}">${fmtPct(totalU)}</div>
    </div>
    <div class="port-kpi">
      <div class="kpi-lbl">Realized</div>
      <div class="kpi-val ${cls(totalR)}">${fmtPct(totalR)}</div>
    </div>
    <div class="port-kpi">
      <div class="kpi-lbl">Open Positions</div>
      <div class="kpi-val neu">${totalO}</div>
    </div>`;

  document.getElementById('strategy-grid').innerHTML = DATA.strategies.map(d => {
    const tot  = d.unrealized + d.realized;
    const bdge = tot > 0.001 ? '<span class="badge badge-green">▲ profitable</span>'
               : tot < -0.001 ? '<span class="badge badge-red">▼ drawdown</span>'
               : '<span class="badge badge-grey">flat</span>';
    const nR = d.nW + d.nL + d.nF;
    return `<div class="card">
      <div class="card-title">${d.label} ${bdge}</div>
      <div class="kpis">
        <div class="kpi"><div class="kpi-lbl">Total MTM</div><div class="kpi-val ${cls(tot)}">${fmtPct(tot)}</div></div>
        <div class="kpi"><div class="kpi-lbl">Unrealized</div><div class="kpi-val ${cls(d.unrealized)}">${fmtPct(d.unrealized)}</div></div>
        <div class="kpi"><div class="kpi-lbl">Realized</div><div class="kpi-val ${cls(d.realized)}">${fmtPct(d.realized)}</div></div>
      </div>
      <div style="font-size:12px;color:var(--muted);margin-bottom:${nR?'6px':'0'}">
        ${d.open.length} open · ${d.resolved.length} resolved${nR ? ' (' + d.nW + 'W/' + d.nL + 'L' + (d.nF ? '/' + d.nF + 'F' : '') + ')' : ''}
      </div>
      ${nR ? wlfRow(d.nW, d.nL, d.nF) : ''}
    </div>`;
  }).join('');
}

// ── PAGE: OPEN POSITIONS ──────────────────────────────────────────────────────
function buildOpen() {
  // Filter pill buttons
  document.getElementById('open-filter').innerHTML = DATA.strategies.map((d, i) =>
    `<button class="seg-btn${i===0?' active':''}" onclick="filterOpen(${i})" id="ofilt-${i}">
      ${d.label} <span style="opacity:.65">${d.open.length}</span>
    </button>`
  ).join('');

  // Strategy sections
  document.getElementById('open-list').innerHTML = DATA.strategies.map((d, i) => {
    const show = i === 0 ? 'block' : 'none';
    if (!d.open.length) {
      return `<div id="open-sec-${i}" style="display:${show}">
        <div class="card"><div style="color:var(--muted);font-size:13px;text-align:center;padding:20px">No open positions</div></div>
      </div>`;
    }
    const rows = d.open.map(r => `<tr>
      <td class="q">${r.question}</td>
      <td style="padding-right:2px">${miniTrack(r.entry, r.current)}</td>
      <td class="hide-xs">${r.entry   != null ? r.entry.toFixed(3)   : '—'}</td>
      <td>${r.current != null ? r.current.toFixed(3) : '—'}</td>
      <td class="${cls(r.mtm)}">${fmtPct(r.mtm)}</td>
      <td><button class="cut-btn" data-sid="${d.strategy_id}" data-q="${esc(r.question)}" onclick="openCut(this)">✂</button></td>
    </tr>`).join('');

    return `<div id="open-sec-${i}" style="display:${show}">
      <div class="card">
        <div class="card-title">${d.label}
          <span style="font-size:12px;font-weight:400;color:var(--muted)">${d.open.length} positions</span>
        </div>
        <div class="tbl-wrap">
          <table>
            <thead><tr>
              <th>Question</th><th>Track</th>
              <th class="hide-xs">Entry</th><th>Now</th><th>MTM</th><th></th>
            </tr></thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
      </div>
    </div>`;
  }).join('');
}

function filterOpen(idx) {
  DATA.strategies.forEach((_, i) => {
    const sec = document.getElementById('open-sec-' + i);
    const btn = document.getElementById('ofilt-' + i);
    if (sec) sec.style.display = i === idx ? 'block' : 'none';
    if (btn) btn.classList.toggle('active', i === idx);
  });
}

// ── PAGE: EQUITY CURVES ───────────────────────────────────────────────────────
function buildEquity() {
  const portPts  = portfolioEquity();
  const portLast = portPts.length ? portPts[portPts.length - 1].cum : 0;

  let html = `<div class="card">
    <div class="card-title">📊 All Strategies Combined</div>
    <div class="eq-sub ${cls(portLast)}">${fmtPct(portLast)} combined MTM</div>
    ${equitySVG(portPts, '#3b82f6')}
  </div>`;

  DATA.strategies.forEach((d, i) => {
    const pts  = strategyEquity(d);
    const last = pts.length ? pts[pts.length - 1].cum : 0;
    const col  = S_COLORS[(i + 1) % S_COLORS.length];
    const nR   = d.nW + d.nL + d.nF;
    html += `<div class="card">
      <div class="card-title">${d.label}</div>
      <div class="eq-sub ${cls(last)}">${fmtPct(last)} · ${d.open.length} open${nR ? ' · ' + d.nW + 'W/' + d.nL + 'L' + (d.nF ? '/' + d.nF + 'F' : '') : ''}</div>
      ${equitySVG(pts, col)}
    </div>`;
  });

  document.getElementById('equity-charts').innerHTML = html;
}

// ── PAGE: HISTORY ─────────────────────────────────────────────────────────────
function buildHistory() {
  const sections = DATA.strategies.map(d => {
    if (!d.resolved.length) return '';
    const rows = [...d.resolved]
      .sort((a, b) => (b.exit_date || '').localeCompare(a.exit_date || ''))
      .map(r => `<tr>
        <td class="q">${r.question}</td>
        <td>${pill(r.status)}</td>
        <td class="${cls(r.pnl)}">${fmtPct(r.pnl)}</td>
        <td style="color:var(--muted);white-space:nowrap">${r.exit_date}</td>
      </tr>`).join('');
    return `<div class="card">
      <div class="card-title">${d.label}</div>
      <div class="tbl-wrap">
        <table>
          <thead><tr><th>Question</th><th>Status</th><th>PnL</th><th>Date</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    </div>`;
  }).join('');

  document.getElementById('history-list').innerHTML = sections.trim() ||
    '<div class="card"><div style="color:var(--muted);font-size:13px;text-align:center;padding:20px">No resolved positions yet</div></div>';
}

// ── MODAL ─────────────────────────────────────────────────────────────────────
let _currentCmd = '';

function openCut(btn) {
  const sid = btn.dataset.sid;
  const q   = btn.dataset.q;
  const cmd = `python experiments/close_position.py --strategy ${sid} --question "${q.replace(/\\/g,'\\\\').replace(/"/g,'\\"')}"`;
  _currentCmd = cmd;
  document.getElementById('modal-q').textContent      = q;
  document.getElementById('modal-cmd').textContent    = cmd;
  document.getElementById('modal-inputs').textContent = `Strategy : ${sid}\nQuestion : ${q}`;
  document.getElementById('modal-gh').href            = ACTIONS_URL;
  document.getElementById('copy-btn').textContent     = '📋 Copy command';
  document.getElementById('cut-modal').classList.add('open');
}
function hideCut() { document.getElementById('cut-modal').classList.remove('open'); }
function copyCmd() {
  navigator.clipboard.writeText(_currentCmd).then(() => {
    const b = document.getElementById('copy-btn');
    b.textContent = '✓ Copied!';
    setTimeout(() => { b.textContent = '📋 Copy command'; }, 1800);
  });
}
document.addEventListener('keydown', e => { if (e.key === 'Escape') hideCut(); });

// ── INIT ──────────────────────────────────────────────────────────────────────
document.getElementById('ts').textContent = DATA.generated;
buildOverview();
buildOpen();
buildEquity();
buildHistory();
</script>
</body>
</html>
"""


def generate() -> None:
    strategies = [
        load_strategy(
            DATA / "midprice_yes_positions.csv", "entry_ask", "question",
            "Mid-priced YES", "midprice_yes"),
        load_strategy(
            DATA / "smart_flow_positions.csv", "entry_ask", "question",
            "Smart Flow", "smart_flow"),
        load_strategy(
            DATA / "macro_positions.csv", "entry_price", "leader_question",
            "Macro (Fed cuts)", "macro"),
    ]
    payload = {
        "generated":  datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "strategies": strategies,
    }
    html = _HTML.replace("%%DATA%%",        json.dumps(payload, ensure_ascii=False))
    html = html.replace("%%ACTIONS_URL%%",  json.dumps(ACTIONS_URL))
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"Dashboard written → {OUT}")
    for s in strategies:
        total = s["unrealized"] + s["realized"]
        sign  = "+" if total >= 0 else ""
        print(f"  {s['label']:22s}  MTM {sign}{total*100:.1f}%  "
              f"(open {len(s['open'])}, {s['nW']}W/{s['nL']}L"
              + (f"/{s['nF']}flip" if s["nF"] else "") + ")")


if __name__ == "__main__":
    generate()
