/* Market Derivatives Analytics — front-end.
   Robinhood-style single-position flow. Read-only: calls the API, renders the
   payoff hero chart, position stats, a compact collapsible order book, a roomy
   option builder, and a compact secondary Greeks panel. */

"use strict";

const state = {
  provider: "",
  markets: [],
  selected: null,       // full market object (with order_book)
  optionType: "call",
  side: "long",
  strike: 0.5,
  obExpanded: false,
  obLevels: 3,
};

const charts = {};
const reduceMotion =
  window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/* KTP palette, kept in sync with the CSS variables. */
const COLORS = {
  blue: "#458fff",
  navy: "#0b2a5b",
  pos: "#0b8f4e",
  neg: "#c23b3b",
  ink: "#111827",
  grid: "#e3e8ef",
  dim: "#6b7280",
};

const el = (id) => document.getElementById(id);
const fmtPct = (x) => (x == null ? "—" : (x * 100).toFixed(1) + "%");
const fmtNum = (x, d = 4) => (x == null ? "—" : Number(x).toFixed(d));
const fmtVol = (v) => {
  if (v == null) return "";
  if (v >= 1e6) return "$" + (v / 1e6).toFixed(1) + "M";
  if (v >= 1e3) return "$" + (v / 1e3).toFixed(0) + "K";
  return "$" + Number(v).toFixed(0);
};

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

/* ---------- market search + list (popular first) ---------- */
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
    el("marketList").innerHTML = `<li class="placeholder">Error: ${escapeHtml(e.message)}</li>`;
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
    const vol = m.volume != null ? `<span class="mi-vol">${fmtVol(m.volume)}</span>` : "";
    li.innerHTML = `
      <div class="mi-title">${escapeHtml(m.title)}</div>
      <div class="mi-meta">
        <span class="badge ${m.provider}">${m.provider}</span>
        ${vol}
        <span class="mi-yes">YES ${fmtPct(m.yes_price)}</span>
      </div>`;
    li.addEventListener("click", () => selectMarket(m.id));
    list.appendChild(li);
  }
}

/* ---------- select a market -> show the single-position card ---------- */
async function selectMarket(id) {
  try {
    state.selected = await api("/api/market/" + encodeURIComponent(id));
    renderMarketList();
    el("emptyState").hidden = true;
    el("positionBody").hidden = false;
    renderMarketHeader();
    renderOrderBook();
    await refreshDerivative();
  } catch (e) {
    el("emptyState").hidden = false;
    el("emptyState").innerHTML = `<p class="placeholder">Error: ${escapeHtml(e.message)}</p>`;
    el("positionBody").hidden = true;
  }
}

function renderMarketHeader() {
  const m = state.selected;
  el("posProvider").textContent = m.provider;
  el("posProvider").className = "badge " + m.provider;
  el("posTitle").textContent = m.title;
  const close = (m.close_time || "").slice(0, 10);
  const parts = [`YES ${fmtPct(m.yes_price)}`];
  if (m.category) parts.push(m.category);
  if (close) parts.push("closes " + close);
  el("posMeta").textContent = parts.join(" · ");
}

function updateChosenBadge(b) {
  el("posChosen").textContent = `${b.side} ${b.option_type} @ ${fmtPct(b.strike)}`;
}

/* ---------- compact, collapsible order book ---------- */
function bookStats() {
  const ob = (state.selected && state.selected.order_book) || { bids: [], asks: [] };
  const bestBid = ob.bids.length ? ob.bids[0][0] : null;
  const bestAsk = ob.asks.length ? ob.asks[0][0] : null;
  const mid = bestBid != null && bestAsk != null ? (bestBid + bestAsk) / 2 : null;
  const spread = bestBid != null && bestAsk != null ? bestAsk - bestBid : null;
  return { ob, bestBid, bestAsk, mid, spread };
}

function renderOrderBook() {
  const { bestBid, bestAsk, mid, spread } = bookStats();
  el("obSummary").textContent =
    `bid ${fmtPct(bestBid)} · ask ${fmtPct(bestAsk)} · mid ${fmtPct(mid)} · spread ${spread == null ? "—" : (spread * 100).toFixed(1) + "pp"}`;
  renderOrderBookDepth();
}

function renderOrderBookDepth() {
  const { ob } = bookStats();
  const n = state.obLevels;
  const bids = ob.bids.slice(0, n);
  const asks = ob.asks.slice(0, n);
  const maxSize = Math.max(
    1,
    ...bids.map((r) => r[1]),
    ...asks.map((r) => r[1])
  );
  const row = (r) => {
    const w = ((r[1] / maxSize) * 100).toFixed(1);
    return `<div class="ob-row"><span class="depth" style="width:${w}%"></span>` +
      `<span class="ob-px">${fmtPct(r[0])}</span><span class="ob-sz">${r[1].toLocaleString()}</span></div>`;
  };
  el("obBids").innerHTML = bids.length ? bids.map(row).join("") : `<div class="ob-row"><span class="ob-sz">no bids</span></div>`;
  el("obAsks").innerHTML = asks.length ? asks.map(row).join("") : `<div class="ob-row"><span class="ob-sz">no asks</span></div>`;
  const maxLevels = Math.max(ob.bids.length, ob.asks.length, 1);
  const slider = el("obLevels");
  slider.max = String(maxLevels);
  el("obLevelsVal").textContent = `${n} level${n === 1 ? "" : "s"}`;
}

function toggleOrderBook() {
  state.obExpanded = !state.obExpanded;
  el("obToggle").setAttribute("aria-expanded", String(state.obExpanded));
  el("obDepth").hidden = !state.obExpanded;
}

/* ---------- derivative bundle -> payoff + stats + Greeks ---------- */
async function refreshDerivative() {
  if (!state.selected) return;
  const params = new URLSearchParams({
    option_type: state.optionType,
    side: state.side,
    strike: String(state.strike),
  });
  try {
    const b = await api(`/api/derivative/${encodeURIComponent(state.selected.id)}?` + params.toString());
    updateChosenBadge(b);
    renderStatRow(b);
    renderPayoff(b);
    renderGreeks(b);
  } catch (e) {
    el("chartNote").textContent = "Error: " + e.message;
  }
}

function renderStatRow(b) {
  const payoffs = b.payoff_curve.map((d) => d.payoff);
  const maxPnl = Math.max(...payoffs);
  const minPnl = Math.min(...payoffs);
  const atUnderlying = b.payoff_curve.reduce((best, d) =>
    Math.abs(d.p - b.underlying_prob) < Math.abs(best.p - b.underlying_prob) ? d : best,
    b.payoff_curve[0]);
  const stat = (label, val, cls) =>
    `<div class="stat"><div class="stat-label">${label}</div><div class="stat-val ${cls || ""}">${val}</div></div>`;
  el("statRow").innerHTML =
    stat("Premium / fair value", fmtNum(b.premium, 3)) +
    stat("Break-even", fmtPct(b.break_even)) +
    stat("Implied prob", fmtPct(b.implied_prob)) +
    stat("P&L at current", fmtNum(atUnderlying.payoff, 3), atUnderlying.payoff >= 0 ? "pos" : "neg") +
    stat("Max gain", fmtNum(maxPnl, 3), "pos") +
    stat("Max loss", fmtNum(minPnl, 3), minPnl < 0 ? "neg" : "");
}

/* ---------- Chart.js helpers ---------- */
const baseAxis = {
  grid: { color: COLORS.grid, drawTicks: false },
  border: { display: false },
  ticks: { color: COLORS.dim, font: { size: 10 } },
};

function lineChart(canvasId, labels, data, color, fillZero, compact) {
  if (charts[canvasId]) charts[canvasId].destroy();
  const ctx = el(canvasId).getContext("2d");
  charts[canvasId] = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [{
        data, borderColor: color, borderWidth: 2,
        pointRadius: 0, tension: 0.15,
        fill: fillZero
          ? { target: "origin", above: "rgba(11,143,78,0.10)", below: "rgba(194,59,59,0.10)" }
          : false,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      animation: reduceMotion ? false : { duration: 200 },
      plugins: { legend: { display: false }, tooltip: { enabled: true, intersect: false, mode: "index" } },
      scales: {
        x: {
          ...baseAxis,
          ticks: {
            ...baseAxis.ticks,
            maxTicksLimit: compact ? 3 : 6,
            callback: (v, i) => (i % (compact ? 50 : 20) === 0 ? labels[i] + "%" : ""),
          },
        },
        y: { ...baseAxis, ticks: { ...baseAxis.ticks, maxTicksLimit: compact ? 4 : 6 } },
      },
    },
  });
}

function renderPayoff(b) {
  const labels = b.payoff_curve.map((d) => (d.p * 100).toFixed(0));
  const data = b.payoff_curve.map((d) => d.payoff);
  lineChart("payoffChart", labels, data, COLORS.blue, true, false);
}

function renderGreeks(b) {
  const g = b.greek_curves;
  const labels = g.delta.map((d) => (d.p * 100).toFixed(0));
  lineChart("deltaChart", labels, g.delta.map((d) => d.value), COLORS.blue, true, true);
  lineChart("gammaChart", labels, g.gamma.map((d) => d.value), COLORS.navy, true, true);
  lineChart("thetaChart", labels, g.theta.map((d) => d.value), COLORS.neg, true, true);
  lineChart("vegaChart", labels, g.vega.map((d) => d.value), COLORS.pos, true, true);

  el("gvDelta").textContent = fmtNum(b.greeks.delta, 3);
  el("gvGamma").textContent = fmtNum(b.greeks.gamma, 3);
  el("gvTheta").textContent = fmtNum(b.greeks.theta, 3);
  el("gvVega").textContent = fmtNum(b.greeks.vega, 3);

  el("greekScalars").innerHTML = `
    <span class="gs">Δ <b>${fmtNum(b.greeks.delta, 2)}</b></span>
    <span class="gs">Γ <b>${fmtNum(b.greeks.gamma, 2)}</b></span>
    <span class="gs">Θ <b>${fmtNum(b.greeks.theta, 2)}</b></span>
    <span class="gs">ν <b>${fmtNum(b.greeks.vega, 2)}</b></span>`;
  el("chartNote").textContent =
    `${b.side} ${b.option_type} · strike ${fmtPct(b.strike)} · underlying YES ${fmtPct(b.underlying_prob)}`;
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

  // builder controls
  el("typeSeg").querySelectorAll("button").forEach((btn) =>
    btn.addEventListener("click", () => {
      state.optionType = btn.dataset.v;
      syncSeg("typeSeg", state.optionType);
      refreshDerivative();
    }));
  el("sideSeg").querySelectorAll("button").forEach((btn) =>
    btn.addEventListener("click", () => {
      state.side = btn.dataset.v;
      syncSeg("sideSeg", state.side);
      refreshDerivative();
    }));
  syncSeg("typeSeg", state.optionType);
  syncSeg("sideSeg", state.side);

  const range = el("strikeRange");
  range.addEventListener("input", () => {
    state.strike = parseFloat(range.value);
    el("strikeVal").textContent = fmtPct(state.strike);
  });
  range.addEventListener("change", refreshDerivative);

  // order book controls
  el("obToggle").addEventListener("click", toggleOrderBook);
  el("obLevels").addEventListener("input", () => {
    state.obLevels = parseInt(el("obLevels").value, 10);
    renderOrderBookDepth();
  });

  loadHealth();
  loadMarkets();
}

function syncSeg(segId, value) {
  el(segId).querySelectorAll("button").forEach((b) =>
    b.classList.toggle("active", b.dataset.v === value));
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
