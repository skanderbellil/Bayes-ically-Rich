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

    def row_to_dict(r):
        return {
            "question":  str(r[q_col]),
            "entry":     round(float(r[entry_col]), 4) if pd.notna(r[entry_col]) else None,
            "current":   round(float(r["current_price"]), 4) if pd.notna(r.get("current_price")) else None,
            "mtm":       round(float(r["mtm"]), 4) if "mtm" in r and pd.notna(r["mtm"]) else None,
            "status":    str(r["status"]),
            "pnl":       round(float(r["pnl"]), 4) if pd.notna(r.get("pnl")) else None,
            "exit_date": str(r.get("exit_date", "")) if pd.notna(r.get("exit_date", "")) else "",
        }

    open_rows     = [row_to_dict(r) for _, r in opened.sort_values("mtm", ascending=False).iterrows()]
    resolved_rows = [row_to_dict(r) for _, r in resolved.iterrows()]

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
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Polymarket Paper-Trade Dashboard</title>
<style>
  :root {
    --bg: #0f1117; --card: #1a1d27; --border: #2a2d3e;
    --green: #22c55e; --red: #ef4444; --yellow: #f59e0b;
    --blue: #3b82f6; --text: #e2e8f0; --muted: #94a3b8;
    --flip: #a78bfa;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Inter', system-ui, sans-serif; font-size: 14px; padding: 24px; }
  h1 { font-size: 22px; font-weight: 700; margin-bottom: 4px; }
  .updated { color: var(--muted); font-size: 12px; margin-bottom: 24px; }
  .strategies { display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 20px; margin-bottom: 32px; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }
  .card-title { font-size: 15px; font-weight: 600; margin-bottom: 12px; display: flex; align-items: center; gap: 8px; }
  .badge { font-size: 11px; padding: 2px 8px; border-radius: 99px; font-weight: 500; }
  .badge-green { background: rgba(34,197,94,.18); color: var(--green); }
  .badge-red   { background: rgba(239,68,68,.18);  color: var(--red); }
  .badge-grey  { background: rgba(148,163,184,.12); color: var(--muted); }
  .kpis { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-bottom: 14px; }
  .kpi { background: rgba(255,255,255,.04); border-radius: 8px; padding: 10px 12px; }
  .kpi-label { font-size: 11px; color: var(--muted); margin-bottom: 3px; text-transform: uppercase; letter-spacing: .4px; }
  .kpi-value { font-size: 18px; font-weight: 700; }
  .pos { color: var(--green); } .neg { color: var(--red); } .neu { color: var(--text); }
  details { margin-top: 10px; }
  summary { cursor: pointer; font-size: 12px; color: var(--muted); user-select: none; padding: 4px 0; }
  summary:hover { color: var(--text); }
  table { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 12px; }
  th { text-align: left; padding: 5px 8px; color: var(--muted); font-weight: 500; border-bottom: 1px solid var(--border); white-space: nowrap; }
  td { padding: 5px 8px; border-bottom: 1px solid rgba(42,45,62,.6); vertical-align: middle; }
  tr:last-child td { border-bottom: none; }
  .q { max-width: 200px; }
  .pill { display: inline-block; font-size: 10px; padding: 1px 6px; border-radius: 4px; font-weight: 600; }
  .pill-won     { background: rgba(34,197,94,.18);  color: var(--green); }
  .pill-lost    { background: rgba(239,68,68,.18);  color: var(--red); }
  .pill-flipped { background: rgba(167,139,250,.18); color: var(--flip); }
  .pill-open    { background: rgba(59,130,246,.12);  color: var(--blue); }
  .meta { font-size: 11px; color: var(--muted); margin-top: 20px; border-top: 1px solid var(--border); padding-top: 12px; }
  /* W/L/F bar */
  .wlf-row { display: flex; align-items: center; gap: 8px; margin-bottom: 10px; }
  .wlf-bar { flex: 1; height: 5px; border-radius: 3px; overflow: hidden; display: flex; gap: 1px; }
  .wlf-seg { height: 100%; border-radius: 2px; }
  .wlf-label { font-size: 11px; color: var(--muted); white-space: nowrap; }
  /* Cut button */
  .cut-btn { background: rgba(239,68,68,.1); color: var(--red); border: 1px solid rgba(239,68,68,.25);
             border-radius: 4px; padding: 2px 7px; font-size: 10px; font-weight: 600; cursor: pointer;
             line-height: 1.7; white-space: nowrap; transition: background .15s; }
  .cut-btn:hover { background: rgba(239,68,68,.25); }
  /* Modal */
  .modal-backdrop { display: none; position: fixed; inset: 0; background: rgba(0,0,0,.7);
                    z-index: 100; align-items: center; justify-content: center; padding: 20px; }
  .modal-backdrop.open { display: flex; }
  .modal { background: #1a1d27; border: 1px solid #2a2d3e; border-radius: 14px; padding: 24px;
           max-width: 520px; width: 100%; }
  .modal h3 { font-size: 15px; margin-bottom: 6px; }
  .modal-q { font-size: 12px; color: var(--muted); margin-bottom: 18px; line-height: 1.5; }
  .modal-section { margin-bottom: 14px; }
  .modal-lbl { font-size: 10px; text-transform: uppercase; letter-spacing: .5px;
               color: var(--muted); margin-bottom: 5px; }
  .code-box { background: #0f1117; border: 1px solid #2a2d3e; border-radius: 6px; padding: 10px 12px;
              font-family: 'Menlo','Consolas',monospace; font-size: 11px; color: #e2e8f0;
              word-break: break-all; line-height: 1.6; white-space: pre-wrap; }
  .modal-actions { display: flex; gap: 8px; margin-top: 18px; flex-wrap: wrap; }
  .btn { padding: 8px 14px; border-radius: 7px; font-size: 12px; font-weight: 600;
         cursor: pointer; border: none; transition: background .15s; }
  .btn-copy   { background: rgba(59,130,246,.18); color: var(--blue); border: 1px solid rgba(59,130,246,.35); }
  .btn-copy:hover { background: rgba(59,130,246,.32); }
  .btn-gh { background: rgba(34,197,94,.12); color: var(--green); border: 1px solid rgba(34,197,94,.28);
            text-decoration: none; display: inline-flex; align-items: center; gap: 5px; }
  .btn-gh:hover { background: rgba(34,197,94,.25); }
  .btn-cancel { background: rgba(255,255,255,.05); color: var(--muted); border: 1px solid #2a2d3e; }
  .btn-cancel:hover { background: rgba(255,255,255,.1); }
</style>
</head>
<body>
<h1>📊 Polymarket Paper-Trade Dashboard</h1>
<div class="updated">Last updated: <strong id="ts">—</strong> &nbsp;·&nbsp; auto-refreshes each cron run</div>

<div class="strategies" id="strategies"></div>

<div class="meta">
  Positions sized at 10% of bankroll per bet. PnL shown as % of that allocation.<br>
  <strong>Mid-priced YES</strong>: buy YES tokens in [0.10,0.50] ~5d before resolution, hold to settlement.<br>
  <strong>Smart Flow</strong>: follow ≥3 smart-wallet consensus buys, consensus-exit on reversal signal.<br>
  <strong>Macro</strong>: hold the dominant leg of macro resolution markets until year-end.
</div>

<!-- Close-position modal -->
<div id="cut-modal" class="modal-backdrop" onclick="hideCut()">
  <div class="modal" onclick="event.stopPropagation()">
    <h3>✂ Close position</h3>
    <div class="modal-q" id="modal-q"></div>

    <div class="modal-section">
      <div class="modal-lbl">Run locally</div>
      <div class="code-box" id="modal-cmd"></div>
    </div>

    <div class="modal-section">
      <div class="modal-lbl">Or use GitHub Actions — paste these values when prompted</div>
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

// ── formatting ───────────────────────────────────────────────────────────────
// All PnL / MTM values are displayed as percentages (e.g. 0.17 → +17.3%)
function fmtPct(v) {
  if (v === null || v === undefined) return '—';
  return (v >= 0 ? '+' : '') + (v * 100).toFixed(1) + '%';
}
function cls(v) { return v > 0.001 ? 'pos' : v < -0.001 ? 'neg' : 'neu'; }
function statusPill(s) { return `<span class="pill pill-${s}">${s}</span>`; }

// ── mini price-track SVG (0 → 1 number line) ─────────────────────────────────
// Grey tick = entry price, coloured dot = current price, filled band shows movement
function miniTrack(entry, current) {
  if (entry == null || current == null) return '<span style="color:var(--border)">—</span>';
  const W = 88, H = 14, mid = 7;
  const clamp = v => Math.min(Math.max(v, 0), 1);
  const ex  = Math.round(clamp(entry)   * W);
  const cx  = Math.round(clamp(current) * W);
  const up  = current >= entry;
  const col = up ? '#22c55e' : '#ef4444';
  const lx  = Math.min(ex, cx), rx = Math.max(ex, cx);
  return [
    `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" style="display:block">`,
    `<rect x="0" y="${mid-1}" width="${W}" height="2" fill="#2a2d3e" rx="1"/>`,
    `<rect x="${lx}" y="${mid-3}" width="${Math.max(rx-lx,1)}" height="6" fill="${col}" opacity="0.3" rx="2"/>`,
    `<line x1="${ex}" y1="2" x2="${ex}" y2="${H-2}" stroke="#94a3b8" stroke-width="1.5" stroke-linecap="round"/>`,
    `<circle cx="${cx}" cy="${mid}" r="3.5" fill="${col}"/>`,
    `</svg>`,
  ].join('');
}

// ── W/L/F horizontal bar ─────────────────────────────────────────────────────
function wlfRow(nW, nL, nF) {
  const total = nW + nL + nF;
  if (!total) return '';
  const pW = (nW / total * 100).toFixed(1);
  const pL = (nL / total * 100).toFixed(1);
  const pF = (nF / total * 100).toFixed(1);
  const segs = [
    nW ? `<div class="wlf-seg" style="width:${pW}%;background:var(--green)"></div>` : '',
    nL ? `<div class="wlf-seg" style="width:${pL}%;background:var(--red)"></div>`   : '',
    nF ? `<div class="wlf-seg" style="width:${pF}%;background:var(--flip)"></div>`  : '',
  ].join('');
  return `<div class="wlf-row">
    <div class="wlf-bar">${segs}</div>
    <span class="wlf-label">${nW}W ${nL}L${nF ? ' ' + nF + 'F' : ''}</span>
  </div>`;
}

// ── modal ─────────────────────────────────────────────────────────────────────
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
    const btn = document.getElementById('copy-btn');
    btn.textContent = '✓ Copied!';
    setTimeout(() => { btn.textContent = '📋 Copy command'; }, 1800);
  });
}

document.addEventListener('keydown', e => { if (e.key === 'Escape') hideCut(); });

// ── card builder ──────────────────────────────────────────────────────────────
function buildCard(d) {
  const total  = d.unrealized + d.realized;
  const totCls = cls(total);
  const badge  = total > 0 ? '<span class="badge badge-green">▲ profitable</span>'
               : total < 0 ? '<span class="badge badge-red">▼ drawdown</span>'
               :              '<span class="badge badge-grey">flat</span>';
  const nR = d.nW + d.nL + d.nF;
  const res_summary = nR
    ? `${d.nW}W / ${d.nL}L${d.nF ? ' / ' + d.nF + ' flip' : ''}`
    : 'none yet';

  // HTML-escape for use in data attributes
  const esc = s => s.replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;').replace(/>/g,'&gt;');

  const openRows = d.open.map(r => `
    <tr>
      <td class="q">${r.question}</td>
      <td style="width:96px;padding-right:4px">${miniTrack(r.entry, r.current)}</td>
      <td>${r.entry   != null ? r.entry.toFixed(3)   : '—'}</td>
      <td>${r.current != null ? r.current.toFixed(3) : '—'}</td>
      <td class="${cls(r.mtm)}">${fmtPct(r.mtm)}</td>
      <td><button class="cut-btn" data-sid="${d.strategy_id}" data-q="${esc(r.question)}" onclick="openCut(this)">✂ cut</button></td>
    </tr>`).join('');

  const resolvedRows = d.resolved.map(r => `
    <tr>
      <td class="q">${r.question}</td>
      <td>${statusPill(r.status)}</td>
      <td class="${cls(r.pnl)}">${fmtPct(r.pnl)}</td>
      <td style="color:var(--muted)">${r.exit_date}</td>
    </tr>`).join('');

  return `
  <div class="card">
    <div class="card-title">${d.label} ${badge}</div>
    <div class="kpis">
      <div class="kpi">
        <div class="kpi-label">Total MTM</div>
        <div class="kpi-value ${totCls}">${fmtPct(total)}</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Unrealized</div>
        <div class="kpi-value ${cls(d.unrealized)}">${fmtPct(d.unrealized)}</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Realized</div>
        <div class="kpi-value ${cls(d.realized)}">${fmtPct(d.realized)}</div>
      </div>
    </div>
    <div style="font-size:12px;color:var(--muted);margin-bottom:${nR ? 5 : 8}px">
      ${d.open.length} open &nbsp;·&nbsp; ${d.resolved.length} resolved (${res_summary})
    </div>
    ${nR ? wlfRow(d.nW, d.nL, d.nF) : ''}

    ${d.open.length ? `
    <details open>
      <summary>Open positions (${d.open.length})</summary>
      <table>
        <thead><tr><th>Question</th><th>Track (0→1)</th><th>Entry</th><th>Now</th><th>MTM %</th><th></th></tr></thead>
        <tbody>${openRows}</tbody>
      </table>
    </details>` : ''}

    ${d.resolved.length ? `
    <details>
      <summary>Resolved (${d.resolved.length})</summary>
      <table>
        <thead><tr><th>Question</th><th>Status</th><th>PnL %</th><th>Date</th></tr></thead>
        <tbody>${resolvedRows}</tbody>
      </table>
    </details>` : ''}
  </div>`;
}

document.getElementById('ts').textContent       = DATA.generated;
document.getElementById('strategies').innerHTML = DATA.strategies.map(buildCard).join('');
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
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "strategies": strategies,
    }
    html = _HTML.replace("%%DATA%%", json.dumps(payload, ensure_ascii=False))
    html = html.replace("%%ACTIONS_URL%%", json.dumps(ACTIONS_URL))
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"Dashboard written → {OUT}")
    for s in strategies:
        total = s["unrealized"] + s["realized"]
        sign  = "+" if total >= 0 else ""
        print(f"  {s['label']:20s}  MTM {sign}{total*100:.1f}%  "
              f"(open {len(s['open'])}, {s['nW']}W/{s['nL']}L"
              + (f"/{s['nF']}flip" if s["nF"] else "") + ")")


if __name__ == "__main__":
    generate()
