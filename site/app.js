// NIFTY direction website.
// data/results.json = test results (made once by live/make_results.py)
// data/live.json    = daily predictions (updated by GitHub Actions every weekday)

const LIVE_ON_GITHUB = "https://raw.githubusercontent.com/Harshul0025/nifty-direction/main/site/data/live.json";
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const C = { s1: css("--s1"), s2: css("--s2"), s3: css("--s3"), gray: css("--gray"), card: css("--card"),
            text: css("--text"), soft: css("--soft"), muted: css("--muted") };

// ---------- small helpers ----------
async function getJSON(url) {
  const reply = await fetch(url, { cache: "no-store" });
  if (!reply.ok) throw new Error(url + " " + reply.status);
  return reply.json();
}

// make an element: el("div", "class names", "text", parent)
function el(tag, cls, text, parent) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined && text !== null) e.textContent = text;
  if (parent) parent.appendChild(e);
  return e;
}

const toDate = s => new Date(s + "T00:00:00");
const niceDate = s => toDate(s).toLocaleDateString("en-IN", { day: "numeric", month: "short", year: "numeric" });
const longDate = s => toDate(s).toLocaleDateString("en-IN", { weekday: "long", day: "numeric", month: "short", year: "numeric" });
const monthYear = s => toDate(s).toLocaleDateString("en-IN", { month: "short", year: "numeric" });
const signed = v => (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(2) + "%";

// a simple table (used for "Show as table" under each chart)
function makeTable(box, headers, rows) {
  const table = el("table", "", null, box);
  const head = el("tr", "", null, table);
  headers.forEach(h => el("th", "", h, head));
  rows.forEach(r => { const tr = el("tr", "", null, table); r.forEach(v => el("td", "", v, tr)); });
}

// legend made of small colored keys + text
function makeLegend(id, items) {
  const box = document.getElementById(id);
  items.forEach(([keyClass, color, label]) => {
    const item = el("span", "", null, box);
    const key = el("i", keyClass, null, item);
    if (color) key.style[keyClass === "key-ring" ? "borderColor" : "background"] = color;
    item.appendChild(document.createTextNode(label));
  });
}

// ---------- tooltip for the HTML charts ----------
const tipBox = document.getElementById("tip");
const tips = new Map();                                  // element -> [title, detail]
function addTip(element, title, detail) {
  tips.set(element, [title, detail]);
  element.tabIndex = 0;
}
function showTip(element, x, y) {
  const [title, detail] = tips.get(element);
  tipBox.replaceChildren();
  el("b", "", title, tipBox);
  el("span", "", detail, tipBox);
  tipBox.hidden = false;
  const w = tipBox.offsetWidth, h = tipBox.offsetHeight;
  tipBox.style.left = Math.max(8, Math.min(x + 14, innerWidth - w - 8)) + "px";
  tipBox.style.top = (y + 18 + h > innerHeight ? y - h - 12 : y + 18) + "px";
}
const tipTarget = e => { let t = e.target; while (t && !tips.has(t)) t = t.parentElement; return t; };
document.addEventListener("pointermove", e => { const t = tipTarget(e); t ? showTip(t, e.clientX, e.clientY) : (tipBox.hidden = true); });
document.addEventListener("focusin", e => { const t = tipTarget(e); if (t) { const r = t.getBoundingClientRect(); showTip(t, r.left, r.bottom - 10); } });
document.addEventListener("focusout", () => { tipBox.hidden = true; });

// ---------- 1. the signal card ----------
const INPUT_NAMES = {
  sp500_night: "S&P 500 last night", usvix_night: "US VIX", crude_night: "Crude oil",
  usdinr_night: "USD/INR", nikkei_gap: "Nikkei open gap", hangseng_gap: "Hang Seng open gap",
};

function drawGaugeZones() {
  // point on the half circle for a chance between 0 and 1
  const point = t => { const a = Math.PI * (1 - t); return [110 + 90 * Math.cos(a), 110 - 90 * Math.sin(a)]; };
  const arc = (from, to) => { const [x0, y0] = point(from), [x1, y1] = point(to); return `M${x0} ${y0} A90 90 0 0 1 ${x1} ${y1}`; };
  document.getElementById("zone-down").setAttribute("d", arc(0, 0.395));     // below 40%: DOWN
  document.getElementById("zone-mid").setAttribute("d", arc(0.405, 0.595));  // 40% to 60%: not sure
  document.getElementById("zone-up").setAttribute("d", arc(0.605, 1));       // above 60%: UP
}

function showSignal(live) {
  const p = live.predictions[live.predictions.length - 1];
  const todayIST = new Date().toLocaleDateString("en-CA", { timeZone: "Asia/Kolkata" });
  const isToday = p.date === todayIST;
  const set = (id, text) => { document.getElementById(id).textContent = text; };

  set("sig-title", isToday ? "Today's signal" : "Latest signal");
  set("sig-date", longDate(p.date));
  const badge = document.getElementById("sig-stage");
  badge.textContent = p.stage;
  badge.classList.toggle("live", p.stage === "live");

  set("sig-pct", Math.round(p.p_up * 100) + "%");
  setTimeout(() => { document.getElementById("needle").style.transform = `rotate(${(p.p_up - 0.5) * 180}deg)`; }, 300);

  const verdict = document.getElementById("sig-verdict");
  verdict.replaceChildren();
  const up = p.pred === "UP";
  el("span", "arrow " + (up ? "up" : "down"), up ? "▲ " : "▼ ", verdict);
  verdict.appendChild(document.createTextNode((p.confident ? "" : "Leaning ") + p.pred));
  set("sig-sub", p.confident ? "The model is confident (chance above 60% or below 40%)."
                             : "Not sure. The chance is between 40% and 60%.");

  let outcome = "The result comes after the market closes at 3:30 PM IST.";
  if (p.actual === "CLOSED") outcome = "The market was closed that day.";
  else if (p.actual) outcome = `NIFTY closed ${p.actual} (${signed(p.ret)}). The call was ${p.correct ? "correct ✓" : "wrong ✗"}`;
  set("sig-outcome", outcome);
  set("sig-next", isToday ? "" : "Next signal: the next trading morning, about 7:50 AM IST.");

  const chips = document.getElementById("sig-inputs");
  chips.replaceChildren();
  for (const [key, label] of Object.entries(INPUT_NAMES)) {
    const chip = el("div", "chip", null, chips);
    el("span", "", label, chip);
    const market = key.replace("_gap", "");
    const closed = key.endsWith("_gap") && p.asia_open && p.asia_open[market] === false;
    el("b", "", closed ? "closed" : signed(p.inputs[key]), chip);
  }
  set("sig-foot", `Made ${p.made_at} · trained on ${p.trained_on_days.toLocaleString("en-IN")} days, up to ${niceDate(p.last_finished_day)}`);
  set("updated", "Data last updated: " + live.updated);
}

// ---------- 2. numbers inside the text: <span data-v="walk_forward.acc"> ----------
function fillNumbers(results) {
  document.querySelectorAll("[data-v]").forEach(span => {
    const value = span.dataset.v.split(".").reduce((obj, key) => obj[key], results);
    const fmt = span.dataset.fmt;
    span.textContent = fmt === "int" ? value.toLocaleString("en-IN") : fmt === "pct" ? value.toFixed(1)
                     : fmt === "month" ? monthYear(value) : fmt === "date" ? niceDate(value) : value;
  });
}

// ---------- 3. dot plot: models against simple rules ----------
function drawModels(results) {
  const LOW = 45, HIGH = 65;
  const place = v => ((v - LOW) / (HIGH - LOW)) * 100 + "%";
  const box = document.getElementById("chart-models");
  const rows = [...results.models].sort((a, b) => b.test_acc - a.test_acc);

  makeLegend("dots-legend", [["key-dot", C.s1, "final model"], ["key-dot", C.gray, "other models"], ["key-ring", C.gray, "simple rules"]]);
  rows.forEach(m => {
    const row = el("div", "row " + m.kind, null, box);
    el("div", "name", m.name, row);
    const track = el("div", "track", null, row);
    el("div", "mid", null, track).style.left = place(50);
    el("div", "dot", null, track).style.left = place(m.test_acc);
    el("div", "value", m.test_acc.toFixed(1) + "%", row);
    addTip(row, m.test_acc.toFixed(1) + "% correct", m.name + (m.test_auc ? ` · test AUC ${m.test_auc} · validation AUC ${m.val_auc}` : " · a rule, not a model"));
  });
  const axis = el("div", "axis", null, box);
  el("div", "", null, axis);
  const ticks = el("div", "ticks", null, axis);
  [45, 50, 55, 60, 65].forEach(t => { el("span", "", t + "%", ticks).style.left = place(t); });

  makeTable(document.getElementById("table-models"), ["Model or rule", "Test accuracy", "Test AUC", "Validation AUC"],
    rows.map(m => [m.name, m.test_acc.toFixed(1) + "%", m.test_auc ?? "–", m.val_auc ?? "–"]));
}

// ---------- 4. horizontal bars: confidence and importance ----------
function drawBar(box, name, small, share, valueText, extraClass) {
  const row = el("div", "row", null, box);
  const label = el("div", "name", name, row);
  if (small) el("small", "", small, label);
  const bar = el("div", "bar " + (extraClass || ""), null, el("div", "track", null, row));
  el("div", "value", valueText, row);
  requestAnimationFrame(() => requestAnimationFrame(() => { bar.style.width = share + "%"; }));
  return row;
}

function drawConfidence(results) {
  const box = document.getElementById("chart-confidence");
  const names = ["All days", "Above 55% or below 45%", "Above 60% or below 40%", "Above 65% or below 35%", "Above 70% or below 30%"];
  results.confidence.forEach((c, i) => {
    const row = drawBar(box, names[i], `${c.share}% of days`, c.acc, c.acc.toFixed(1) + "%");
    addTip(row, c.acc.toFixed(1) + "% correct", `${c.days} of ${results.data.test_days} test days have a chance this far from 50%`);
  });
  makeTable(document.getElementById("table-confidence"), ["Only days where the chance of UP is", "Days", "Share of days", "Accuracy"],
    results.confidence.map((c, i) => [names[i], c.days, c.share + "%", c.acc.toFixed(1) + "%"]));
}

const FEATURE_NAMES = {
  hangseng_gap: "Hang Seng opening gap", nikkei_gap: "Nikkei opening gap", sp500_night: "S&P 500 last night",
  usvix_night: "US VIX last night", crude_night: "Crude oil last night", usdinr_night: "USD/INR last night",
  prev_ret_1: "NIFTY return yesterday", prev_gap: "NIFTY opening gap yesterday",
  prev_dist_52w_high: "Distance from 1-year high", prev_macd_hist: "MACD yesterday",
};

function drawImportance(results) {
  const box = document.getElementById("chart-importance");
  const biggest = results.importance[0].drop;
  makeLegend("legend-importance", [["key-box", C.s2, "overnight and Asia inputs"], ["key-box", C.gray, "yesterday's NIFTY inputs"]]);
  results.importance.forEach(f => {
    const name = FEATURE_NAMES[f.name] || f.name;
    const row = drawBar(box, name, null, (f.drop / biggest) * 100, f.drop.toFixed(3), f.overnight ? "night" : "day");
    row.querySelector(".bar.day")?.style.setProperty("background", C.gray);
    addTip(row, "AUC drops by " + f.drop.toFixed(3), `${name} (${f.name}) · ${f.overnight ? "overnight input" : "yesterday's input"}`);
  });
  makeTable(document.getElementById("table-importance"), ["Input", "Group", "Drop in test AUC"],
    results.importance.map(f => [FEATURE_NAMES[f.name] || f.name, f.overnight ? "overnight / Asia" : "yesterday", f.drop.toFixed(3)]));
}

function drawConfusion(results) {
  const c = results.confusion, box = document.getElementById("confusion");
  const cell = (n, text, right) => { const d = el("div", "cell" + (right ? " right" : ""), null, box); el("b", "", n, d); d.appendChild(document.createTextNode(text)); };
  el("div", "", "", box); el("div", "head", "NIFTY closed UP", box); el("div", "head", "NIFTY closed DOWN", box);
  el("div", "side", "Model said UP", box); cell(c.up_right, "correct", true); cell(c.up_wrong, "wrong", false);
  el("div", "side", "Model said DOWN", box); cell(c.down_wrong, "wrong", false); cell(c.down_right, "correct", true);
}

// ---------- 5. line charts (Chart.js) ----------
// thin vertical line under the pointer
const crosshair = {
  id: "crosshair",
  afterDatasetsDraw(chart) {
    const active = chart.tooltip?.getActiveElements() || [];
    if (!active.length) return;
    const { ctx, chartArea } = chart, x = active[0].element.x;
    ctx.save(); ctx.strokeStyle = "rgba(255,255,255,0.3)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x, chartArea.top); ctx.lineTo(x, chartArea.bottom); ctx.stroke(); ctx.restore();
  },
};
// the last value of each line, written at its end (pushed apart if two would overlap)
const endLabels = {
  id: "endLabels",
  afterDatasetsDraw(chart, args, options) {
    const labels = chart.data.datasets.map((d, i) => {
      const last = chart.getDatasetMeta(i).data.at(-1);
      return { x: last.x, y: last.y, text: options.format(d.data.at(-1)) };
    }).sort((a, b) => a.y - b.y);
    for (let i = 1; i < labels.length; i++) if (labels[i].y - labels[i - 1].y < 15) labels[i].y = labels[i - 1].y + 15;
    const ctx = chart.ctx;
    ctx.save(); ctx.fillStyle = C.text; ctx.font = "600 12px 'JetBrains Mono', monospace"; ctx.textBaseline = "middle";
    labels.forEach(l => ctx.fillText(l.text, l.x + 10, l.y));
    ctx.restore();
  },
};

function lineChart(id, labels, series, yAxis, xTicks, format) {
  Chart.defaults.color = C.muted;
  Chart.defaults.font.family = "'Space Grotesk', system-ui, sans-serif";
  return new Chart(document.getElementById(id), {
    type: "line",
    data: {
      labels,
      datasets: series.map(s => ({
        label: s.name, data: s.data, borderColor: s.color, backgroundColor: s.color, borderWidth: 2,
        pointRadius: s.points ? 5 : 0, pointHoverRadius: 6, pointBorderColor: C.card, pointBorderWidth: 2, tension: 0,
      })),
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      layout: { padding: { right: 58, top: 8 } },
      scales: {
        x: { grid: { display: false }, border: { color: "rgba(255,255,255,0.15)" }, ticks: xTicks },
        y: yAxis,
      },
      plugins: {
        legend: { display: false },
        endLabels: { format },
        tooltip: {
          backgroundColor: "#191d33", borderColor: "rgba(255,255,255,0.14)", borderWidth: 1, padding: 12,
          titleColor: C.soft, bodyColor: C.text, boxWidth: 14, boxHeight: 2, bodyFont: { weight: 600 },
          callbacks: { label: item => ` ${format(item.raw)}  ${item.dataset.label}` },
        },
      },
    },
    plugins: [crosshair, endLabels],
  });
}

function drawYears(results) {
  const w = results.walk_forward;
  const series = [
    { name: "Model", data: w.model_by_year, color: C.s1, points: true },
    { name: 'Rule: "follow the US"', data: w.follow_us_by_year, color: C.s2, points: true },
  ];
  makeLegend("legend-years", series.map(s => ["key-line", s.color, s.name]));
  lineChart("chart-years", w.years, series, {
    min: 40, max: 80, border: { display: false },
    ticks: { stepSize: 10, callback: v => v + "%" },
    grid: { color: g => (g.tick.value === 50 ? "rgba(255,255,255,0.35)" : "rgba(255,255,255,0.07)") },   // 50% = coin flip
  }, { maxRotation: 0 }, v => v.toFixed(1) + "%");
  makeTable(document.getElementById("table-years"), ["Year", "Model", "Follow the US"],
    w.years.map((y, i) => [y, w.model_by_year[i].toFixed(1) + "%", w.follow_us_by_year[i].toFixed(1) + "%"]));
}

function drawMoney(results) {
  const m = results.money;
  const series = [
    { name: "Impossible: buy at yesterday's close", data: m.impossible, color: C.s2 },
    { name: "Real: buy at open, sell at close, after costs", data: m.real, color: C.s1 },
    { name: "Buy and hold NIFTY", data: m.hold, color: C.s3 },
  ];
  makeLegend("legend-money", series.map(s => ["key-line", s.color, s.name]));
  lineChart("chart-money", m.dates.map(monthYear), series, {
    type: "logarithmic", min: 0.2, max: 40, border: { display: false },
    afterBuildTicks: axis => { axis.ticks = [0.25, 0.5, 1, 2, 5, 10, 20, 40].map(value => ({ value })); },
    ticks: { callback: v => "₹" + v },
    grid: { color: g => (g.tick.value === 1 ? "rgba(255,255,255,0.35)" : "rgba(255,255,255,0.07)") },    // ₹1 = where it started
  }, {
    autoSkip: false, maxRotation: 0,
    callback(value) {                                    // one tick per year (every second year on a phone)
      const label = this.getLabelForValue(value), year = Number(label.slice(-4));
      return label.startsWith("Jan") && (this.chart.width > 520 || year % 2 === 1) ? year : null;
    },
  }, v => "₹" + v.toFixed(2));
}

// ---------- 6. track record ----------
function drawRecord(live, results) {
  const table = document.getElementById("record-table");
  const head = el("tr", "", null, table);
  ["Date", "Signal", "NIFTY that day", "Result", "Type"].forEach(h => el("th", "", h, head));
  [...live.predictions].reverse().forEach(p => {
    const tr = el("tr", "", null, table);
    el("td", "", niceDate(p.date), tr);
    el("td", "", `${p.pred === "UP" ? "▲" : "▼"} ${p.pred} · ${Math.round(p.p_up * 100)}%`, tr);
    el("td", "", p.ret === null ? "–" : signed(p.ret), tr);
    const result = el("td", "", null, tr);
    if (p.actual === "CLOSED") result.textContent = "market closed";
    else if (p.correct === null) result.textContent = "waiting";
    else el("span", p.correct ? "ok" : "bad", p.correct ? "✓ correct" : "✗ wrong", result);
    el("span", "tag " + p.stage, p.stage, el("td", "", null, tr));
  });

  const score = stage => {
    const rows = live.predictions.filter(p => p.stage === stage && p.correct !== null);
    return `${rows.filter(p => p.correct).length} of ${rows.length} correct`;
  };
  const anyLive = live.predictions.some(p => p.stage === "live");
  document.getElementById("record-summary").textContent =
    (anyLive ? `Live: ${score("live")}. ` : "Live predictions start on the next trading morning. ") + `Replays: ${score("replay")}.`;

  const strip = document.getElementById("strip");
  results.recent.forEach(d => {
    const sq = el("span", "sq " + (d.hit ? "hit" : "miss"), null, strip);
    addTip(sq, `${niceDate(d.date)}: ${d.hit ? "correct" : "wrong"}`, `Chance of UP ${Math.round(d.p_up * 100)}% · NIFTY ${signed(d.ret)}`);
  });
  const hits = results.recent.filter(d => d.hit).length;
  document.getElementById("strip-summary").textContent =
    `${hits} of ${results.recent.length} correct. Each square is one trading day, oldest first.`;
}

// ---------- start ----------
async function start() {
  drawGaugeZones();
  const results = await getJSON("data/results.json");
  results.follow_us_test = results.models.find(m => m.name === "Rule: follow the US").test_acc;
  // newest predictions straight from GitHub; if that fails, use the copy that was deployed with the site
  const live = await getJSON(LIVE_ON_GITHUB).catch(() => getJSON("data/live.json"));

  showSignal(live);
  fillNumbers(results);
  drawModels(results);
  drawConfidence(results);
  drawImportance(results);
  drawConfusion(results);
  drawYears(results);
  drawMoney(results);
  drawRecord(live, results);

  // fade things in as they scroll into view
  const watcher = new IntersectionObserver(entries => entries.forEach(e => e.isIntersecting && e.target.classList.add("seen")), { threshold: 0.08 });
  document.querySelectorAll(".card, h2, .section-lead, .timeline").forEach(e => { e.classList.add("reveal"); watcher.observe(e); });
}
start();
