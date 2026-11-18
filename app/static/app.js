/* Market Derivatives Analytics — front-end.
   Read-only: calls the API, renders order book + payoff + Greeks. */

"use strict";

const state = {
  provider: "",
  markets: [],
  selected: null,       // full market object
  optionType: "call",
  side: "long",
  strike: 0.5,
};

const charts = {};
const COLORS = {
  accent: "#e0a63c",
  pos: "#3fb950",
  neg: "#e05b5b",
  ink: "#e6e9ef",
  dim: "#6b7686",
  line: "#262d38",
};

const el = (id) => document.getElementById(id);
const fmtPct = (x) => (x == null ? "—" : (x * 100).toFixed(1) + "%");
const fmtNum = (x, d = 4) => (x == null ? "—" : Number(x).toFixed(d));

async function api(path) {
  const r = await fetch(path);
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  return r.json();
}

/* ---------- backend status chip ---------- */
async function loadHealth() {
  try {
    const h = await api("/api/health");
    const chip = el("backendChip");
    if (h.using_mock_backends) {
      chip.textContent = "demo data (mock backends)";
      chip.classList.add("mock");
    } else {
      chip.textContent = "live data";
      chip.classList.add("live");
    }
  } catch (_) { /* non-fatal */ }
}

/* ---------- market search + list ---------- */
let searchTimer = null;
function onSearchInput() {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(loadMarkets, 180);
}

async function loadMarkets() {
  const q = el("searchInput").value.trim();
  const params = new URLSearchParams();
  if (q) params.set("q", q);
  if (state.provider) params.set("provider", state.provider);
  try {
    const data = await api("/api/markets?" + params.toString());
    state.markets = data.markets;
    renderMarketList();
  } catch (e) {
    el("marketList").innerHTML = `<li class="placeholder">Error: ${e.message}</li>`;
  }
}

function renderMarketList() {
  const list = el("marketList");
  if (!state.markets.length) {
    list.innerHTML = `<li class="placeholder">No markets match.</li>`;
    return;
  }
  list.innerHTML = "";
  for (const m of state.markets) {
    const li = document.createElement("li");
    li.className = "market-item" + (state.selected && state.selected.id === m.id ? " selected" : "");
    li.innerHTML = `
      <div class="mi-title">${escapeHtml(m.title)}</div>
      <div class="mi-meta">
        <span class="badge ${m.provider}">${m.provider}</span>
        <span>${escapeHtml(m.category || "")}</span>
        <span class="mi-yes">YES ${fmtPct(m.yes_price)}</span>
      </div>`;
    li.addEventListener("click", () => selectMarket(m.id));
    list.appendChild(li);
  }
}

/* ---------- market detail + order book ---------- */
async function selectMarket(id) {
  try {
    state.selected = await api("/api/market/" + encodeURIComponent(id));
    renderMarketList();
    renderMarketDetail();
    await refreshDerivative();
  } catch (e) {
    el("marketDetail").innerHTML = `<p class="placeholder">Error: ${e.message}</p>`;
  }
}

function renderMarketDetail() {
  const m = state.selected;
  const detail = el("marketDetail");
  detail.classList.remove("empty");
  const ob = m.order_book || { bids: [], asks: [] };
  const maxSize = Math.max(
    1,
    ...ob.bids.map((r) => r[1]),
    ...ob.asks.map((r) => r[1])
  );
  const bookRow = (r, side) => {
    const w = ((r[1] / maxSize) * 100).toFixed(1);
    return `<div class="ob-row"><span class="depth" style="width:${w}%"></span>
      <span class="ob-px">${fmtPct(r[0])}</span><span class="ob-sz">${r[1].toLocaleString()}</span></div>`;
  };

  detail.innerHTML = `
    <h3 class="md-title">${escapeHtml(m.title)}</h3>
    <div class="md-sub"><span class="badge ${m.provider}">${m.provider}</span> · ${escapeHtml(m.category || "")} · closes ${escapeHtml((m.close_time || "").slice(0, 10))}</div>

    <div class="price-row">
      <div class="price-card yes"><div class="pc-label">YES</div><div class="pc-val">${fmtPct(m.yes_price)}</div></div>
      <div class="price-card no"><div class="pc-label">NO</div><div class="pc-val">${fmtPct(m.no_price)}</div></div>
    </div>

    <div class="ob-head">Order book — depth (price × size)</div>
    <div class="orderbook"><div class="ob-cols">
      <div class="ob-col bids"><h4>Bids</h4>${ob.bids.map((r) => bookRow(r, "bid")).join("")}</div>
      <div class="ob-col asks"><h4>Asks</h4>${ob.asks.map((r) => bookRow(r, "ask")).join("")}</div>
    </div></div>

    <div class="synth">
      <h3>Synthetic option</h3>
      <div class="ctrl-row">
        <div class="seg" id="typeSeg">
          <button data-v="call" class="${state.optionType === "call" ? "active" : ""}">Call</button>
          <button data-v="put" class="${state.optionType === "put" ? "active" : ""}">Put</button>
        </div>
        <div class="seg" id="sideSeg">
          <button data-v="long" class="${state.side === "long" ? "active" : ""}">Long</button>
          <button data-v="short" class="${state.side === "short" ? "active" : ""}">Short</button>
        </div>
      </div>
      <div class="strike-ctrl">
        <label>Strike (probability) <span class="strike-val" id="strikeVal">${fmtPct(state.strike)}</span></label>
        <input type="range" id="strikeRange" min="0" max="1" step="0.01" value="${state.strike}" />
      </div>
      <div class="synth-summary" id="synthSummary"></div>
    </div>`;

  // wire controls
  el("typeSeg").querySelectorAll("button").forEach((b) =>
    b.addEventListener("click", () => { state.optionType = b.dataset.v; renderMarketDetail(); refreshDerivative(); }));
  el("sideSeg").querySelectorAll("button").forEach((b) =>
    b.addEventListener("click", () => { state.side = b.dataset.v; renderMarketDetail(); refreshDerivative(); }));
  const range = el("strikeRange");
  range.addEventListener("input", () => {
    state.strike = parseFloat(range.value);
    el("strikeVal").textContent = fmtPct(state.strike);
  });
  range.addEventListener("change", refreshDerivative);
}

/* ---------- derivative bundle + charts ---------- */
async function refreshDerivative() {
  if (!state.selected) return;
  const params = new URLSearchParams({
    option_type: state.optionType,
    side: state.side,
    strike: String(state.strike),
  });
  try {
    const b = await api(`/api/derivative/${encodeURIComponent(state.selected.id)}?` + params.toString());
    renderSynthSummary(b);
    renderPayoff(b);
    renderGreeks(b);
    el("chartNote").textContent =
      `${state.side} ${state.optionType} · strike ${fmtPct(b.strike)} · underlying YES ${fmtPct(b.underlying_prob)}`;
  } catch (e) {
    el("chartNote").textContent = "Error: " + e.message;
  }
}

function renderSynthSummary(b) {
  const s = el("synthSummary");
  if (!s) return;
  s.innerHTML = `
    <div class="ss"><div class="ss-label">Premium</div><div class="ss-val">${fmtNum(b.premium)}</div></div>
    <div class="ss"><div class="ss-label">Break-even</div><div class="ss-val">${fmtPct(b.break_even)}</div></div>
    <div class="ss"><div class="ss-label">Implied prob</div><div class="ss-val">${fmtPct(b.implied_prob)}</div></div>`;
}

const baseAxis = {
  grid: { color: COLORS.line },
  ticks: { color: COLORS.dim, font: { family: "monospace", size: 10 } },
};
function lineChart(canvasId, labels, data, color, zeroLine) {
  if (charts[canvasId]) charts[canvasId].destroy();
  const ctx = el(canvasId).getContext("2d");
  charts[canvasId] = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [{
        data, borderColor: color, borderWidth: 2,
        pointRadius: 0, tension: 0.15,
        fill: zeroLine ? { target: "origin", above: "rgba(63,185,80,0.08)", below: "rgba(224,91,91,0.08)" } : false,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      animation: false,
      plugins: { legend: { display: false }, tooltip: { enabled: true, intersect: false, mode: "index" } },
      scales: {
        x: { ...baseAxis, ticks: { ...baseAxis.ticks, maxTicksLimit: 6, callback: (v, i) => (i % 20 === 0 ? labels[i] : "") } },
        y: baseAxis,
      },
    },
  });
}

function renderPayoff(b) {
  const labels = b.payoff_curve.map((d) => (d.p * 100).toFixed(0));
  const data = b.payoff_curve.map((d) => d.payoff);
  lineChart("payoffChart", labels, data, COLORS.accent, true);
}

function renderGreeks(b) {
  const g = b.greek_curves;
  const labels = g.delta.map((d) => (d.p * 100).toFixed(0));
  lineChart("deltaChart", labels, g.delta.map((d) => d.value), "#7aa7ff", true);
  lineChart("gammaChart", labels, g.gamma.map((d) => d.value), "#8fd3b6", true);
  lineChart("thetaChart", labels, g.theta.map((d) => d.value), COLORS.neg, true);
  lineChart("vegaChart", labels, g.vega.map((d) => d.value), COLORS.accent, true);

  el("greekScalars").innerHTML = `
    <span class="gs">Δ <b>${fmtNum(b.greeks.delta, 3)}</b></span>
    <span class="gs">Γ <b>${fmtNum(b.greeks.gamma, 3)}</b></span>
    <span class="gs">Θ <b>${fmtNum(b.greeks.theta, 3)}</b></span>
    <span class="gs">ν <b>${fmtNum(b.greeks.vega, 3)}</b></span>`;
}

/* ---------- utils ---------- */
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* ---------- init ---------- */
function init() {
  el("searchInput").addEventListener("input", onSearchInput);
  document.querySelectorAll(".prov").forEach((b) =>
    b.addEventListener("click", () => {
      document.querySelectorAll(".prov").forEach((x) => x.classList.remove("active"));
      b.classList.add("active");
      state.provider = b.dataset.provider;
      loadMarkets();
    }));
  loadHealth();
  loadMarkets();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
