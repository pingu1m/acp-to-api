"""Embedded web dashboard: TraceHub for real-time event streaming and single-page HTML UI."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TraceEvent:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    ts: float = field(default_factory=time.time)
    protocol: str = "rest"
    direction: str = "inbound"
    provider: str = ""
    method: str = ""
    summary: str = ""
    body: str = ""
    status: int | None = None
    duration_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


MAX_WS_CLIENTS = 10


class TraceHub:
    """In-memory ring buffer with fan-out to WebSocket subscribers."""

    def __init__(self, maxlen: int = 2000) -> None:
        self._buffer: deque[TraceEvent] = deque(maxlen=maxlen)
        self._subscribers: set[asyncio.Queue[TraceEvent]] = set()

    def push(self, event: TraceEvent) -> None:
        self._buffer.append(event)
        for q in self._subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass

    def recent(self, n: int = 200) -> list[dict[str, Any]]:
        items = list(self._buffer)[-n:]
        return [e.to_dict() for e in items]

    def subscribe(self) -> asyncio.Queue[TraceEvent] | None:
        if len(self._subscribers) >= MAX_WS_CLIENTS:
            return None
        q: asyncio.Queue[TraceEvent] = asyncio.Queue(maxsize=500)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[TraceEvent]) -> None:
        self._subscribers.discard(q)


DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>acp-to-api dashboard</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#0f0f17;--surface:#1a1a2e;--border:#2a2a40;--text:#e0e0e8;
  --dim:#888;--accent:#6c63ff;--green:#4caf50;--blue:#2196f3;
  --red:#ef5350;--orange:#ff9800;--cyan:#00bcd4;
}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,monospace;background:var(--bg);color:var(--text);height:100vh;display:flex;flex-direction:column;overflow:hidden}
a{color:var(--accent)}
header{display:flex;align-items:center;gap:16px;padding:10px 20px;background:var(--surface);border-bottom:1px solid var(--border);flex-shrink:0}
header h1{font-size:16px;font-weight:600;white-space:nowrap}
header .meta{font-size:12px;color:var(--dim);display:flex;gap:16px}
header .ws-status{margin-left:auto;font-size:12px;display:flex;align-items:center;gap:6px}
header .ws-dot{width:8px;height:8px;border-radius:50%;background:var(--red)}
header .ws-dot.ok{background:var(--green)}
.main{display:flex;flex:1;overflow:hidden}
.sidebar{width:280px;min-width:240px;border-right:1px solid var(--border);display:flex;flex-direction:column;background:var(--surface);flex-shrink:0}
.sidebar h2{font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:var(--dim);padding:12px 16px 6px}
.provider-list{flex:1;overflow-y:auto;padding:0 8px 8px}
.prov-card{background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:10px 12px;margin-bottom:6px;display:flex;justify-content:space-between;align-items:center}
.prov-card .name{font-weight:600;font-size:13px}
.prov-card .cmd{font-size:11px;color:var(--dim)}
.prov-card button{background:none;border:1px solid var(--red);color:var(--red);border-radius:4px;padding:2px 8px;cursor:pointer;font-size:11px}
.prov-card button:hover{background:var(--red);color:#fff}
.sidebar-actions{padding:8px;border-top:1px solid var(--border)}
.sidebar-actions button{width:100%;padding:6px;margin-bottom:4px;border:1px solid var(--border);background:var(--bg);color:var(--text);border-radius:4px;cursor:pointer;font-size:12px}
.sidebar-actions button:hover{border-color:var(--accent);color:var(--accent)}
.add-form{padding:8px 12px;border-top:1px solid var(--border);display:none}
.add-form.open{display:block}
.add-form input{width:100%;padding:5px 8px;margin-bottom:4px;background:var(--bg);border:1px solid var(--border);color:var(--text);border-radius:4px;font-size:12px}
.add-form .btn-row{display:flex;gap:4px}
.add-form .btn-row button{flex:1;padding:5px;font-size:12px;border-radius:4px;cursor:pointer;border:1px solid var(--border);background:var(--bg);color:var(--text)}
.add-form .btn-row button.primary{border-color:var(--green);color:var(--green)}
.add-form .btn-row button.primary:hover{background:var(--green);color:#fff}
.trace-panel{flex:1;display:flex;flex-direction:column;overflow:hidden}
.trace-toolbar{display:flex;align-items:center;gap:8px;padding:8px 12px;border-bottom:1px solid var(--border);background:var(--surface);flex-wrap:wrap}
.trace-toolbar select,.trace-toolbar input{background:var(--bg);border:1px solid var(--border);color:var(--text);border-radius:4px;padding:4px 8px;font-size:12px}
.trace-toolbar input[type=text]{width:180px}
.trace-toolbar button{background:var(--bg);border:1px solid var(--border);color:var(--text);border-radius:4px;padding:4px 10px;cursor:pointer;font-size:12px}
.trace-toolbar button:hover{border-color:var(--accent);color:var(--accent)}
.trace-toolbar button.active{border-color:var(--orange);color:var(--orange)}
.trace-toolbar .count{font-size:11px;color:var(--dim);margin-left:auto}
.trace-list{flex:1;overflow-y:auto;font-size:12px}
.trace-row{display:flex;align-items:flex-start;gap:8px;padding:6px 12px;border-bottom:1px solid var(--border);cursor:pointer;transition:background .1s}
.trace-row:hover{background:rgba(108,99,255,.07)}
.trace-row .time{color:var(--dim);white-space:nowrap;width:85px;flex-shrink:0;font-variant-numeric:tabular-nums}
.badge{display:inline-block;padding:1px 6px;border-radius:3px;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.04em}
.badge.acp{background:rgba(76,175,80,.18);color:var(--green)}
.badge.rest{background:rgba(33,150,243,.18);color:var(--blue)}
.arrow{font-size:14px;width:18px;text-align:center;flex-shrink:0}
.arrow.in{color:var(--cyan)}
.arrow.out{color:var(--orange)}
.trace-row .prov{color:var(--accent);width:80px;flex-shrink:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.trace-row .meth{color:var(--text);width:220px;flex-shrink:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.trace-row .stat{width:32px;flex-shrink:0;text-align:right}
.trace-row .stat.ok{color:var(--green)}
.trace-row .stat.err{color:var(--red)}
.trace-row .dur{width:50px;flex-shrink:0;text-align:right;color:var(--dim)}
.trace-row .summ{flex:1;color:var(--dim);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.detail-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.6);z-index:100;justify-content:center;align-items:center}
.detail-overlay.open{display:flex}
.detail-box{background:var(--surface);border:1px solid var(--border);border-radius:8px;width:80%;max-width:900px;max-height:80vh;display:flex;flex-direction:column}
.detail-header{display:flex;justify-content:space-between;align-items:center;padding:12px 16px;border-bottom:1px solid var(--border)}
.detail-header h3{font-size:14px}
.detail-header button{background:none;border:none;color:var(--dim);cursor:pointer;font-size:18px}
.detail-body{flex:1;overflow:auto;padding:16px;font-family:monospace;font-size:12px;white-space:pre-wrap;word-break:break-all;line-height:1.6}
.empty{padding:40px;text-align:center;color:var(--dim)}
</style>
</head>
<body>
<header>
  <h1>acp-to-api</h1>
  <div class="meta">
    <span id="hConfigPath"></span>
    <span id="hHostPort"></span>
  </div>
  <div class="ws-status"><span class="ws-dot" id="wsDot"></span><span id="wsLabel">connecting</span></div>
</header>
<div class="main">
  <div class="sidebar">
    <h2>Providers</h2>
    <div class="provider-list" id="provList"></div>
    <div class="sidebar-actions">
      <button id="btnToggleAdd">+ Add Provider</button>
      <button id="btnReloadConfig">Reload Config</button>
    </div>
    <div class="add-form" id="addForm">
      <input id="addName" placeholder="name (e.g. cursor)">
      <input id="addCmd" placeholder="command (e.g. agent)">
      <input id="addArgs" placeholder="args, comma-separated (e.g. acp)">
      <div class="btn-row">
        <button id="btnCancelAdd">Cancel</button>
        <button class="primary" id="btnSubmitAdd">Add</button>
      </div>
    </div>
  </div>
  <div class="trace-panel">
    <div class="trace-toolbar">
      <select id="fProto"><option value="">All</option><option value="rest">REST</option><option value="acp">ACP</option></select>
      <select id="fDir"><option value="">All</option><option value="inbound">Inbound</option><option value="outbound">Outbound</option></select>
      <select id="fProv"><option value="">All providers</option></select>
      <input type="text" id="fSearch" placeholder="Search...">
      <button id="btnPause">Pause</button>
      <button id="btnClear">Clear</button>
      <span class="count" id="traceCount">0 events</span>
    </div>
    <div class="trace-list" id="traceList"><div class="empty">Waiting for events...</div></div>
  </div>
</div>
<div class="detail-overlay" id="detailOverlay">
  <div class="detail-box">
    <div class="detail-header"><h3 id="detailTitle">Event</h3><button id="btnCloseDetail">&times;</button></div>
    <div class="detail-body" id="detailBody"></div>
  </div>
</div>
<script>
const API = location.origin;
const WS_URL = (location.protocol === 'https:' ? 'wss:' : 'ws:') + '//' + location.host + '/api/v1/ws/traces';
let ws, paused = false, events = [], providerSet = new Set();

function $(id) { return document.getElementById(id); }

// --- Config ---
async function loadConfig() {
  try {
    const r = await fetch(API + '/api/v1/config');
    const c = await r.json();
    $('hConfigPath').textContent = c.config_path || 'no config file';
    $('hHostPort').textContent = c.host + ':' + c.port;
  } catch(e) { console.error('config load failed', e); }
}

// --- Providers ---
async function loadProviders() {
  try {
    const r = await fetch(API + '/api/v1/providers');
    const list = await r.json();
    const el = $('provList');
    el.innerHTML = '';
    const sel = $('fProv');
    const cur = sel.value;
    sel.innerHTML = '<option value="">All providers</option>';
    list.forEach(p => {
      const card = document.createElement('div');
      card.className = 'prov-card';
      const info = document.createElement('div');
      const nameEl = document.createElement('div'); nameEl.className = 'name'; nameEl.textContent = p.name;
      const cmdEl = document.createElement('div'); cmdEl.className = 'cmd'; cmdEl.textContent = p.command + ' ' + (p.args||[]).join(' ');
      info.appendChild(nameEl); info.appendChild(cmdEl);
      const btn = document.createElement('button'); btn.textContent = 'remove';
      btn.addEventListener('click', () => removeProv(p.name));
      card.appendChild(info); card.appendChild(btn);
      el.appendChild(card);
      providerSet.add(p.name);
      const opt = document.createElement('option');
      opt.value = p.name; opt.textContent = p.name;
      sel.appendChild(opt);
    });
    sel.value = cur;
  } catch(e) { console.error('providers load failed', e); }
}

async function removeProv(name) {
  if (!confirm('Remove provider "' + name + '"?')) return;
  await fetch(API + '/api/v1/providers/' + encodeURIComponent(name), {method:'DELETE'});
  loadProviders();
}

async function reloadConfig() {
  const r = await fetch(API + '/api/v1/providers/reload', {method:'POST'});
  const d = await r.json();
  loadProviders();
  alert('Reload: ' + JSON.stringify(d));
}

function toggleAdd() { $('addForm').classList.toggle('open'); }

async function submitAdd() {
  const name = $('addName').value.trim();
  const cmd = $('addCmd').value.trim();
  const args = $('addArgs').value.trim().split(',').map(s=>s.trim()).filter(Boolean);
  if (!name || !cmd) { alert('Name and command are required'); return; }
  const r = await fetch(API + '/api/v1/providers', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({name, command: cmd, args})
  });
  if (r.ok) {
    $('addName').value = ''; $('addCmd').value = ''; $('addArgs').value = '';
    toggleAdd(); loadProviders();
  } else {
    const e = await r.json(); alert('Error: ' + (e.detail || r.status));
  }
}

// --- Trace ---
function connectWS() {
  ws = new WebSocket(WS_URL);
  ws.onopen = () => { $('wsDot').classList.add('ok'); $('wsLabel').textContent = 'connected'; };
  ws.onclose = () => {
    $('wsDot').classList.remove('ok'); $('wsLabel').textContent = 'reconnecting...';
    setTimeout(connectWS, 2000);
  };
  ws.onmessage = (e) => {
    const data = JSON.parse(e.data);
    if (Array.isArray(data)) { data.forEach(addEvent); }
    else { addEvent(data); }
  };
}

function addEvent(ev) {
  events.push(ev);
  if (ev.provider) providerSet.add(ev.provider);
  if (!paused) renderEvent(ev);
  $('traceCount').textContent = events.length + ' events';
}

function renderEvent(ev) {
  if (!matchesFilter(ev)) return;
  const list = $('traceList');
  if (list.querySelector('.empty')) list.innerHTML = '';
  const row = document.createElement('div');
  row.className = 'trace-row';
  row.onclick = () => showDetail(ev);
  const t = new Date(ev.ts * 1000);
  const ts = t.toLocaleTimeString('en-US', {hour12:false}) + '.' + String(t.getMilliseconds()).padStart(3,'0');
  const dirCls = ev.direction === 'inbound' ? 'in' : 'out';
  const dirIcon = ev.direction === 'inbound' ? '&#x2192;' : '&#x2190;';
  const statCls = ev.status ? (ev.status < 400 ? 'ok' : 'err') : '';
  const statTxt = ev.status || '';
  const durTxt = ev.duration_ms != null ? Math.round(ev.duration_ms) + 'ms' : '';
  row.innerHTML =
    '<span class="time">' + ts + '</span>' +
    '<span class="badge ' + ev.protocol + '">' + ev.protocol + '</span>' +
    '<span class="arrow ' + dirCls + '">' + dirIcon + '</span>' +
    '<span class="prov">' + esc(ev.provider) + '</span>' +
    '<span class="meth">' + esc(ev.method) + '</span>' +
    '<span class="stat ' + statCls + '">' + statTxt + '</span>' +
    '<span class="dur">' + durTxt + '</span>' +
    '<span class="summ">' + esc(ev.summary) + '</span>';
  list.prepend(row);
  while (list.children.length > 1000) list.removeChild(list.lastChild);
}

function matchesFilter(ev) {
  const fp = $('fProto').value;
  if (fp && ev.protocol !== fp) return false;
  const fd = $('fDir').value;
  if (fd && ev.direction !== fd) return false;
  const fv = $('fProv').value;
  if (fv && ev.provider !== fv) return false;
  const fs = $('fSearch').value.toLowerCase();
  if (fs && !(ev.method + ' ' + ev.summary + ' ' + ev.body).toLowerCase().includes(fs)) return false;
  return true;
}

function togglePause() {
  paused = !paused;
  $('btnPause').textContent = paused ? 'Resume' : 'Pause';
  $('btnPause').classList.toggle('active', paused);
  if (!paused) rerender();
}

function clearTrace() {
  events = [];
  $('traceList').innerHTML = '<div class="empty">Cleared</div>';
  $('traceCount').textContent = '0 events';
}

function rerender() {
  $('traceList').innerHTML = '';
  events.filter(matchesFilter).slice(-500).forEach(renderEvent);
  if (!$('traceList').children.length) $('traceList').innerHTML = '<div class="empty">No matching events</div>';
}

['fProto','fDir','fProv','fSearch'].forEach(id => $(id).addEventListener('change', rerender));
$('fSearch').addEventListener('input', rerender);

function showDetail(ev) {
  $('detailTitle').textContent = ev.protocol.toUpperCase() + ' ' + ev.direction + ' ' + ev.method;
  let bodyText = ev.body || ev.summary || '(empty)';
  try { bodyText = JSON.stringify(JSON.parse(bodyText), null, 2); } catch(e) {}
  $('detailBody').textContent = bodyText;
  $('detailOverlay').classList.add('open');
}
function closeDetail() { $('detailOverlay').classList.remove('open'); }
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeDetail(); });

function esc(s) { const d = document.createElement('div'); d.textContent = s || ''; return d.innerHTML; }

$('btnToggleAdd').addEventListener('click', toggleAdd);
$('btnCancelAdd').addEventListener('click', toggleAdd);
$('btnSubmitAdd').addEventListener('click', submitAdd);
$('btnReloadConfig').addEventListener('click', reloadConfig);
$('btnPause').addEventListener('click', togglePause);
$('btnClear').addEventListener('click', clearTrace);
$('btnCloseDetail').addEventListener('click', closeDetail);
$('detailOverlay').addEventListener('click', e => { if (e.target === $('detailOverlay')) closeDetail(); });
loadConfig();
loadProviders();
connectWS();
setInterval(loadProviders, 15000);
</script>
</body>
</html>
"""
