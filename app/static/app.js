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
  tau: 1.0,             // modeled time-to-resolution in (0, 1]; 1 = far from resolving
  bundle: null,        // last derivative bundle (for the what-if scrub)
  obExpanded: false,
  obLevels: 3,
};

const charts = {};
// Test hook: expose the live chart instances so the browser regression test
// can introspect config/guides. Harmless in production (read-only reference).
if (typeof window !== "undefined") window.__charts = charts;
const reduceMotion =
  window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/* KTP palette (ktpvandy.org), kept in sync with the CSS variables in styles.css. */
const COLORS = {
  blue: "#2f4a8a",
  navy: "#1b2a52",
  pos: "#0b8f4e",
  neg: "#c23b3b",
  ink: "#0f172a",
  grid: "#e2e8f0",
  dim: "#5b6678",
  gold: "#c8a951",
};

const el = (id) => document.getElementById(id);
const fmtPct = (x) => (x == null ? "—" : (x * 100).toFixed(1) + "%");

/* Normalize -0 (and values that round to zero at the chosen precision) so the
   UI never prints a misleading "-0.00" / "−$0.000". */
const noNegZero = (x) => (x === 0 ? 0 : x);

/* Adaptive precision: show a fixed number of decimals, but if the value is
   small-but-non-zero and would round to all-zeros at that precision, widen the
   decimals until the first significant digit shows (capped) so a real non-zero
   number never reads as 0.00. Zero stays "0". */
function adaptiveFixed(x, baseDecimals) {
  if (x == null || !Number.isFinite(Number(x))) return null;
  const v = noNegZero(Number(x));
  if (v === 0) return (0).toFixed(baseDecimals);
  let d = baseDecimals;
  // widen until the rounded value is non-zero, up to 8 decimals
  while (d < 8 && Math.abs(Number(v.toFixed(d))) === 0) d++;
  return v.toFixed(d);
}

const fmtNum = (x, d = 4) => {
  const s = adaptiveFixed(x, d);
  return s == null ? "—" : s;
};
const fmtVol = (v) => {
  if (v == null) return "";
  if (v >= 1e6) return "$" + (v / 1e6).toFixed(1) + "M";
  if (v >= 1e3) return "$" + (v / 1e3).toFixed(0) + "K";
  return "$" + Number(v).toFixed(0);
};

async function api(path, signal) {
  const r = await fetch(path, signal ? { signal } : undefined);
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  return r.json();
}

/* Per-flow AbortControllers so a newer request cancels the one in flight and a
   late/aborted response can never overwrite fresher data. Aborted requests are
   swallowed silently (that is the expected outcome, not an error). */
const inflight = { markets: null, market: null, derivative: null };
function freshSignal(flow) {
  if (inflight[flow]) inflight[flow].abort();
  const ctrl = new AbortController();
  inflight[flow] = ctrl;
  return ctrl.signal;
}
const isAbort = (e) => e && (e.name === "AbortError" || e.code === 20);

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
    const data = await api("/api/markets?" + params.toString(), freshSignal("markets"));
    state.markets = data.markets;
    renderMarketList();
  } catch (e) {
    if (isAbort(e)) return;  // superseded by a newer search; ignore silently
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
    li.title = m.title;  // full title on hover; the visible title is clamped
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
  // A new selection cancels any in-flight market load and the previous market's
  // derivative request, so a slow older response can't repaint a stale market.
  const signal = freshSignal("market");
  if (inflight.derivative) inflight.derivative.abort();
  try {
    state.selected = await api("/api/market/" + encodeURIComponent(id), signal);
    renderMarketList();
    el("emptyState").hidden = true;
    el("positionBody").hidden = false;
    renderMarketHeader();
    renderOrderBook();
    await refreshDerivative();
  } catch (e) {
    if (isAbort(e)) return;  // superseded by a newer selection; ignore silently
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

/* ---------- derivative bundle -> payoff + summary + sensitivities ---------- */
async function refreshDerivative() {
  if (!state.selected) return;
  const params = new URLSearchParams({
    option_type: state.optionType,
    side: state.side,
    strike: String(state.strike),
    tau: String(state.tau),
  });
  try {
    const b = await api(`/api/derivative/${encodeURIComponent(state.selected.id)}?` + params.toString(), freshSignal("derivative"));
    state.bundle = b;
    updateChosenBadge(b);
    renderSummaryStrip(b);
    renderPayoff(b);
    renderGreeks(b);
    renderProbScrub();
  } catch (e) {
    if (isAbort(e)) return;  // superseded by a newer builder change; ignore silently
    el("chartNote").textContent = "Error: " + e.message;
  }
}

/* Money is $ per $1 contract; always carry the sign so a loss reads as negative.
   Adaptive precision keeps a tiny non-zero value from collapsing to $0.000, and
   negative zero is normalized so "max loss" never shows "−$0.000". */
const fmtMoney = (x) => {
  if (x == null || !Number.isFinite(Number(x))) return "—";
  const v = noNegZero(Number(x));
  const mag = adaptiveFixed(Math.abs(v), 3);
  return (v < 0 ? "−$" : "$") + mag;
};

function renderSummaryStrip(b) {
  // Use the API's own side-signed scalars, not a re-derivation.
  // 'You pay' and 'Chance of profit' are the headline numbers (hero tiles).
  const tile = (label, val, cls, sub, hero) =>
    `<div class="sum-tile${hero ? " hero" : ""}">
       <div class="sum-label">${label}</div>
       <div class="sum-val ${cls || ""}">${val}</div>
       ${sub ? `<div class="sum-sub">${sub}</div>` : ""}
     </div>`;

  el("summaryStrip").innerHTML =
    tile("You pay", fmtMoney(b.premium), "accent", "premium, per $1 contract", true) +
    tile("Chance of profit", fmtPct(b.prob_of_profit), "", "model-based", true) +
    tile("Max you'd make", fmtMoney(b.max_gain), "pos", "per $1 contract") +
    tile("Max you'd lose", fmtMoney(b.max_loss), b.max_loss < 0 ? "neg" : "", "per $1 contract") +
    tile("Break-even", fmtPct(b.break_even), "", "YES probability");

  el("summaryNote").textContent =
    "“You pay”, “max make” and “max lose” are exact from the payoff — model-free. " +
    "“Chance of profit” is model-based (Bachelier), so treat it as a modeled estimate, not odds.";
}

/* ---------- Chart.js global styling (Inter, muted grid, no borders) ---------- */
function configureCharts() {
  if (typeof Chart === "undefined") return;
  Chart.defaults.font.family =
    '"Inter var", "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
  Chart.defaults.font.size = 11;
  Chart.defaults.color = COLORS.dim;
  // Keep Chart.js's own animation config object intact — replacing it wholesale
  // (e.g. with `false`) drops the internal fields Chart.js expects and throws
  // "this._fn is not a function", blanking every chart. Only tune duration, and
  // honour reduced motion by zeroing the duration rather than nulling the object.
  Chart.defaults.animation.duration = reduceMotion ? 0 : 200;
  Chart.defaults.plugins.legend.display = false;
  Chart.defaults.plugins.tooltip.backgroundColor = COLORS.navy;
  Chart.defaults.plugins.tooltip.titleColor = "#fff";
  Chart.defaults.plugins.tooltip.bodyColor = "#e7edf8";
  Chart.defaults.plugins.tooltip.borderColor = "rgba(255,255,255,0.15)";
  Chart.defaults.plugins.tooltip.borderWidth = 1;
  Chart.defaults.plugins.tooltip.padding = 8;
  Chart.defaults.plugins.tooltip.cornerRadius = 8;
  Chart.defaults.plugins.tooltip.displayColors = false;
}

/* ---------- Chart.js helpers ---------- */
const baseAxis = {
  grid: { color: COLORS.grid, drawTicks: false, lineWidth: 1 },
  border: { display: false },
  ticks: { color: COLORS.dim, font: { size: 10 } },
};

function lineChart(canvasId, labels, data, color, fillZero, compact, guides, moneyY) {
  if (charts[canvasId]) charts[canvasId].destroy();
  const ctx = el(canvasId).getContext("2d");
  charts[canvasId] = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [{
        data, borderColor: color, borderWidth: 2,
        pointRadius: 0, pointHoverRadius: 4, pointHoverBackgroundColor: color,
        tension: 0.15,
        fill: fillZero
          ? { target: "origin", above: "rgba(11,143,78,0.12)", below: "rgba(194,59,59,0.12)" }
          : false,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      animation: { duration: reduceMotion ? 0 : 200 },
      interaction: { intersect: false, mode: "index" },
      plugins: {
        legend: { display: false },
        // Pass the guide values through the plugin options so the plugin reads
        // them from its `opts` argument on every draw (including the first).
        // Attaching them to the chart after construction meant the first draw
        // had no guides and the break-even / current-price / max lines never
        // appeared.
        payoffGuides: guides || undefined,
        tooltip: {
          enabled: true, intersect: false, mode: "index",
          callbacks: {
            title: (items) => (items.length ? `YES ${items[0].label}%` : ""),
            label: (item) => {
              const v = item.parsed.y;
              if (v == null) return "";
              return moneyY ? fmtMoney(v) + " per $1" : fmtNum(v, 3);
            },
          },
        },
      },
      scales: {
        x: {
          ...baseAxis,
          ticks: {
            ...baseAxis.ticks,
            autoSkip: false,
            // Label every tick we keep. `afterBuildTicks` below pins the ticks
            // to round percentages, so each one gets a "NN%" label — the old
            // `i % 20` test keyed off the post-skip tick index (not a data
            // index) and left every chart showing a single "0%".
            callback(value) {
              const pct = Number(this.getLabelForValue(value));
              return Number.isFinite(pct) ? pct + "%" : "";
            },
          },
          // Pin ticks to round percentages: 0/20/40/60/80/100 (or 0/50/100 for
          // the compact Greek charts). labels are "0".."100", so the tick value
          // is the data index == the percentage.
          afterBuildTicks(axis) {
            const wanted = compact ? [0, 50, 100] : [0, 20, 40, 60, 80, 100];
            const maxIdx = labels.length - 1;  // 100 for the 0..100 grid
            axis.ticks = wanted
              .filter((p) => p <= maxIdx)
              .map((p) => ({ value: p }));
          },
        },
        y: { ...baseAxis, ticks: { ...baseAxis.ticks, maxTicksLimit: compact ? 4 : 6 } },
      },
    },
    plugins: guides ? [payoffGuides] : [],
  });
  return charts[canvasId];
}

function renderPayoff(b) {
  const labels = b.payoff_curve.map((d) => (d.p * 100).toFixed(0));
  const data = b.payoff_curve.map((d) => d.payoff);
  lineChart("payoffChart", labels, data, COLORS.blue, true, false, {
    breakEven: b.break_even,      // YES probability 0..1
    underlying: b.underlying_prob,
    maxGain: b.max_gain,
    maxLoss: b.max_loss,
  }, true);
}

/* Draw the break-even vertical line, the current-underlying marker, and the
   max-gain / max-loss horizontal guides directly on the payoff curve, so the
   three summary numbers are visible ON the chart. Pure canvas, no plugin dep. */
const payoffGuides = {
  id: "payoffGuides",
  afterDatasetsDraw(chart, _args, opts) {
    const g = opts;
    if (!g || g.breakEven == null) return;
    const { ctx, chartArea: area, scales } = chart;
    const xAt = (p) => scales.x.getPixelForValue(Math.round(p * 100));
    const yAt = (v) => scales.y.getPixelForValue(v);
    ctx.save();
    ctx.font = '10px "Inter var", "Inter", -apple-system, "Segoe UI", sans-serif';

    // horizontal guides for max gain / max loss
    const hline = (val, color, label) => {
      if (val == null || val < scales.y.min || val > scales.y.max) return;
      const y = yAt(val);
      ctx.strokeStyle = color; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(area.left, y); ctx.lineTo(area.right, y); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = color; ctx.textAlign = "right"; ctx.textBaseline = val >= 0 ? "bottom" : "top";
      ctx.fillText(label, area.right - 4, y + (val >= 0 ? -2 : 2));
    };
    hline(g.maxGain, COLORS.pos, "max gain " + fmtMoney(g.maxGain));
    hline(g.maxLoss, COLORS.neg, "max loss " + fmtMoney(g.maxLoss));

    // vertical break-even line
    if (g.breakEven != null) {
      const x = xAt(g.breakEven);
      ctx.strokeStyle = COLORS.navy; ctx.lineWidth = 1.5; ctx.setLineDash([5, 4]);
      ctx.beginPath(); ctx.moveTo(x, area.top); ctx.lineTo(x, area.bottom); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = COLORS.navy; ctx.textAlign = x > (area.left + area.right) / 2 ? "right" : "left";
      ctx.textBaseline = "top";
      ctx.fillText("break-even " + (g.breakEven * 100).toFixed(0) + "%", x + (ctx.textAlign === "right" ? -4 : 4), area.top + 2);
    }

    // current-underlying marker (where the market sits now)
    if (g.underlying != null) {
      const x = xAt(g.underlying);
      ctx.strokeStyle = COLORS.gold; ctx.lineWidth = 1; ctx.setLineDash([2, 3]);
      ctx.beginPath(); ctx.moveTo(x, area.top); ctx.lineTo(x, area.bottom); ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.restore();
  },
};

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

/* ---------- what-if probability scrub ---------- */
/* Reads premium (model fair value), payoff and the modeled chance-of-profit at a
   hypothetical YES probability the user drags to. Payoff is exact at that level
   (from the curve); premium/PoP are the model's, shown for intuition only. */
function renderProbScrub() {
  const b = state.bundle;
  const scrub = el("probScrub");
  if (!b) return;
  // default the marker to the current underlying the first time
  if (scrub.dataset.touched !== "1") {
    scrub.value = String(b.underlying_prob);
  }
  const p = parseFloat(scrub.value);
  el("probScrubVal").textContent = fmtPct(p);

  // nearest payoff sample on the 0..1 grid
  const pt = b.payoff_curve.reduce((best, d) =>
    Math.abs(d.p - p) < Math.abs(best.p - p) ? d : best, b.payoff_curve[0]);
  const side = p >= b.break_even ? "above" : "below";
  el("probScrubReadout").innerHTML =
    `If it settles around <b>${fmtPct(p)}</b> YES: payoff ` +
    `<b class="${pt.payoff >= 0 ? "pos" : "neg"}">${fmtMoney(pt.payoff)}</b> per $1 contract ` +
    `(${side} your ${fmtPct(b.break_even)} break-even).`;
}

/* ---------- utils ---------- */
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* ---------- init ---------- */
function init() {
  configureCharts();
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

  // time-to-resolution (tau): a modeled resolution horizon, not a listed expiry
  const tau = el("tauRange");
  const fmtTau = (t) =>
    t <= 0.12 ? "about to resolve" : t >= 0.9 ? "far off" : "partway";
  tau.addEventListener("input", () => {
    state.tau = parseFloat(tau.value);
    el("tauVal").textContent = fmtTau(state.tau);
  });
  tau.addEventListener("change", refreshDerivative);
  el("tauVal").textContent = fmtTau(state.tau);

  // what-if probability scrub (read-only; does not re-request the bundle)
  const scrub = el("probScrub");
  scrub.addEventListener("input", () => {
    scrub.dataset.touched = "1";
    renderProbScrub();
  });

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
