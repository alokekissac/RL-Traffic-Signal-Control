// Simulation + UI. The environment below mirrors traffic_rl/env.py exactly.
import { IntersectionScene } from "./scene.js";

const NS_MAX = 3, EW_MAX = 5, EW = 0, NS = 1;
const ARRIVALS = [0, 1, 2], DEPARTURES = [1, 2];
const $ = (id) => document.getElementById(id);

const LABELS = {
  q_learning: "Q-learning", monte_carlo: "Monte Carlo", optimal: "Optimal (value iteration)",
  optimal_overflow_aware: "Optimal, overflow-aware", longest_queue: "Longest queue first",
  fixed_time: "Fixed-time", random: "Random",
};
const DESCRIPTIONS = {
  q_learning: "Tabular Q-learning trained for 5,000 episodes (α 0.1, γ 0.9, ε 1.0 → 0.01). Acts greedily on its learned Q-table.",
  monte_carlo: "First-visit Monte Carlo control, same budget. Learns from complete episodes rather than single steps.",
  optimal: "The exact optimal policy, solved by value iteration on the known transition model. The benchmark to beat.",
  optimal_overflow_aware: "Optimal once every turned-away car costs 2 points: it serves North–South more and turns away fewer cars.",
  longest_queue: "Classic actuated rule: give the green to whichever road has more cars waiting.",
  fixed_time: "A traditional timer: the green alternates every 2 ticks, whatever the traffic.",
  random: "Flips a coin every tick. A floor that any sensible controller should beat.",
};
const COLORS = { q_learning: "#4b93ea", monte_carlo: "#f07a4a", optimal: "#22c08a", optimal_overflow_aware: "#6fd3ae" };

// ------------------------------------------------------------------ RNG + env
function mulberry32(a) {
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const pick = (rng, arr) => arr[Math.floor(rng() * arr.length)];

/** One tick. Returns the new state plus a breakdown the 3D scene can animate. */
function envStep(ns, ew, action, rng) {
  const aNS = pick(rng, ARRIVALS), aEW = pick(rng, ARRIVALS), d = pick(rng, DEPARTURES);
  const road = (q, arr, green, cap) => {
    const pre = q + arr;
    const after = green ? Math.max(0, pre - d) : pre;
    const departed = pre - after;
    const dropped = Math.max(0, after - cap);
    const next = Math.min(after, cap);
    const departFromQueue = Math.min(departed, q);
    const passThrough = departed - departFromQueue;
    return { next, dropped, anim: { departFromQueue, passThrough, join: arr - passThrough - dropped, dropped } };
  };
  const n = road(ns, aNS, action === NS, NS_MAX), e = road(ew, aEW, action === EW, EW_MAX);
  return { ns: n.next, ew: e.next, dropped: n.dropped + e.dropped, waiting: n.next + e.next, anim: { ns: n.anim, ew: e.anim } };
}

// ------------------------------------------------------------------ policies
let TABLES = {};
function makePolicy(name, rng) {
  if (TABLES[name]) {
    const Q = TABLES[name].Q;
    return (ns, ew) => (Q[ns][ew][1] > Q[ns][ew][0] ? NS : EW);
  }
  if (name === "longest_queue") return (ns, ew) => (ns > ew ? NS : EW);
  if (name === "fixed_time") return (ns, ew, t) => (Math.floor(t / 2) % 2 === 0 ? EW : NS);
  return () => (rng() < 0.5 ? EW : NS);
}

// ------------------------------------------------------------------ state
const sim = {
  policyName: "q_learning", policy: null, rng: mulberry32(Date.now() & 0xffff),
  ns: 1, ew: 3, t: 0, action: EW, playing: true, speed: 1, timer: null, busy: false,
  total: 0, waitingSum: 0, dropped: 0, history: [],
};
let scene;
const D = () => 1500 / sim.speed;

function setPolicy(name) {
  sim.policyName = name;
  sim.policy = makePolicy(name, sim.rng);
  $("hudPolicy").textContent = LABELS[name];
  $("policyDesc").textContent = DESCRIPTIONS[name];
  updateQ();
}

function tick() {
  if (sim.busy) return;
  sim.busy = true;
  const next = sim.policy(sim.ns, sim.ew, sim.t);
  const dur = D();
  let delay = 0;
  if (next !== sim.action) {
    // amber phase on the road losing its green
    scene.setLights(sim.action, sim.action === NS ? "ns" : "ew");
    setPhase(sim.action, true);
    delay = dur * 0.28;
  }
  setTimeout(() => {
    sim.action = next;
    scene.setLights(next);
    setPhase(next, false);
    const out = envStep(sim.ns, sim.ew, next, sim.rng);
    scene.animateRoad("ns", out.anim.ns, dur);
    scene.animateRoad("ew", out.anim.ew, dur);
    sim.ns = out.ns; sim.ew = out.ew; sim.t += 1;
    sim.total -= out.waiting;
    sim.waitingSum += out.waiting;
    sim.dropped += out.dropped;
    sim.history.push(out.waiting);
    if (sim.history.length > 60) sim.history.shift();
    setTimeout(() => { updateStats(); updateQ(); }, dur * 0.55);
    setTimeout(() => { sim.busy = false; schedule(); }, dur);
  }, delay);
}

function schedule() {
  clearTimeout(sim.timer);
  if (sim.playing) sim.timer = setTimeout(tick, 120 / sim.speed);
}

function resetSim(ns = Math.floor(Math.random() * 3), ew = Math.floor(Math.random() * 4)) {
  Object.assign(sim, { ns, ew, t: 0, total: 0, waitingSum: 0, dropped: 0, history: [] });
  scene.setQueues(ns, ew);
  updateStats(); updateQ();
}

// ------------------------------------------------------------------ UI
function setPhase(action, amber) {
  const el = $("phase");
  el.classList.toggle("amber", amber);
  $("phaseText").textContent = amber ? "Changing…" : action === NS ? "North–South green" : "East–West green";
}

function updateStats() {
  $("hudTick").textContent = sim.t;
  $("qNS").textContent = sim.ns; $("qEW").textContent = sim.ew;
  $("qlabelNS").classList.toggle("full", sim.ns === NS_MAX);
  $("qlabelEW").classList.toggle("full", sim.ew === EW_MAX);
  $("qlabelNS").classList.toggle("green", sim.action === NS);
  $("qlabelEW").classList.toggle("green", sim.action === EW);
  $("sWaiting").textContent = sim.ns + sim.ew;
  $("sAvg").textContent = sim.t ? (sim.waitingSum / sim.t).toFixed(2) : "–";
  $("sDropped").textContent = sim.dropped;
  $("sReward").textContent = sim.total;
  drawSpark();
}

function updateQ() {
  const table = TABLES[sim.policyName] || TABLES.optimal;
  if (!table) return;
  const own = !!TABLES[sim.policyName];
  const [qe, qn] = table.Q[sim.ns][sim.ew];
  $("qState").textContent = `(ns ${sim.ns}, ew ${sim.ew})`;
  // best action = full bar; the other shrinks with its relative gap (exaggerated so small gaps show)
  const best = Math.max(qe, qn), span = (x) => `${Math.max(8, 100 * Math.pow(Math.abs(best) / Math.max(Math.abs(x), 1e-9), 4))}%`;
  $("qbEW").style.width = span(qe); $("qbNS").style.width = span(qn);
  $("qbEW").classList.toggle("best", qe >= qn); $("qbNS").classList.toggle("best", qn > qe);
  $("qvEW").textContent = qe.toFixed(2); $("qvNS").textContent = qn.toFixed(2);
  $("qNote").textContent = own
    ? "Q-values: expected discounted reward of each action (closer to zero is better)."
    : "This controller follows a fixed rule. Shown for reference: the optimal policy's Q-values.";
}

function drawSpark() {
  const c = $("spark"), dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth, h = c.clientHeight;
  c.width = w * dpr; c.height = h * dpr;
  const g = c.getContext("2d");
  g.scale(dpr, dpr);
  g.clearRect(0, 0, w, h);
  const max = NS_MAX + EW_MAX, n = 60;
  g.strokeStyle = "rgba(125,138,160,.18)"; g.lineWidth = 1;
  for (const v of [0, 4, 8]) { const y = h - 4 - (v / max) * (h - 10); g.beginPath(); g.moveTo(0, y); g.lineTo(w, y); g.stroke(); }
  const hs = sim.history;
  if (hs.length < 2) return;
  const x = (i) => w - (hs.length - 1 - i) * (w / (n - 1));
  const y = (v) => h - 4 - (v / max) * (h - 10);
  const col = COLORS[sim.policyName] || "#94a3b8";
  const grad = g.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, col + "55"); grad.addColorStop(1, col + "00");
  g.beginPath(); g.moveTo(x(0), h);
  hs.forEach((v, i) => g.lineTo(x(i), y(v)));
  g.lineTo(x(hs.length - 1), h); g.closePath(); g.fillStyle = grad; g.fill();
  g.beginPath(); hs.forEach((v, i) => (i ? g.lineTo(x(i), y(v)) : g.moveTo(x(i), y(v))));
  g.strokeStyle = col; g.lineWidth = 2; g.lineJoin = "round"; g.stroke();
  $("sparkMax").textContent = `max ${Math.max(...hs)} · now ${hs[hs.length - 1]}`;
}

function placeLabels() {
  for (const [road, cap, id] of [["ns", NS_MAX, "qlabelNS"], ["ew", EW_MAX, "qlabelEW"]]) {
    const p = scene.project(scene.queueAnchor(road, cap));
    const el = $(id);
    const W = $("stage").clientWidth, H = $("stage").clientHeight;
    el.style.left = `${Math.min(W - 52, Math.max(52, p.x))}px`;
    el.style.top = `${Math.min(H - 12, Math.max(64, p.y))}px`;
    el.style.display = p.visible ? "" : "none";
  }
}

function dropTag(worldPos) {
  const p = scene.project(worldPos);
  const el = document.createElement("div");
  el.className = "float-tag"; el.textContent = "turned away";
  el.style.left = `${p.x}px`; el.style.top = `${p.y}px`;
  $("toasts").appendChild(el);
  setTimeout(() => el.remove(), 1700);
}

// ------------------------------------------------------------------ results
function renderResults(res) {
  if (!res?.baselines) return;
  const b = res.baselines, ex = res.experiments;
  const q = ex.baseline.q_learning, mc = ex.baseline.monte_carlo;
  const fixed = b["Fixed-time (2-step cycle)"].mean, opt = b["Optimal (value iteration)"].mean;
  document.querySelector('[data-kpi="gain_fixed"]').textContent = `${Math.round((1 - q.mean_waiting / fixed) * 100)}%`;
  document.querySelector('[data-kpi="gap_opt"]').textContent = `+${(q.mean_waiting - opt).toFixed(3)}`;
  document.querySelector('[data-kpi="agree"]').textContent = `${Math.round(q.policy_agreement_with_optimal * 100)}%`;

  const colorOf = { "Optimal (value iteration)": COLORS.optimal, "Optimal, overflow-aware": COLORS.optimal_overflow_aware };
  const rows = Object.entries(b).map(([k, v]) => ({ name: k, m: v.mean, ci: `±${v.ci95.toFixed(3)}`, d: v.dropped, ag: "–", c: colorOf[k] }));
  rows.push({ name: "Q-learning (mean of 5 seeds)", m: q.mean_waiting, ci: `sd ${q.std_across_seeds.toFixed(3)}`, d: q.mean_dropped, ag: `${Math.round(q.policy_agreement_with_optimal * 100)}%`, c: COLORS.q_learning });
  rows.push({ name: "Monte Carlo (mean of 5 seeds)", m: mc.mean_waiting, ci: `sd ${mc.std_across_seeds.toFixed(3)}`, d: mc.mean_dropped, ag: `${Math.round(mc.policy_agreement_with_optimal * 100)}%`, c: COLORS.monte_carlo });
  rows.sort((a, z) => a.m - z.m);
  $("resultsTable").tBodies[0].innerHTML = rows.map((r) =>
    `<tr><td><span class="key" style="${r.c ? `background:${r.c}` : ""}"></span>${r.name}</td><td class="num">${r.m.toFixed(3)}</td><td class="num">${r.ci}</td><td class="num">${r.d.toFixed(3)}</td><td class="num">${r.ag}</td></tr>`).join("");

  const names = { baseline: "Report settings", exp1: "Experiment 1", exp2: "Experiment 2", overflow: "Overflow-aware reward" };
  const algo = { q_learning: "Q-learning", monte_carlo: "Monte Carlo" };
  $("expTable").tBodies[0].innerHTML = Object.entries(ex).flatMap(([k, d]) => ["q_learning", "monte_carlo"].map((a) => {
    const hp = d.hyperparams, e = d[a];
    return `<tr><td>${names[k] || k}</td><td>${algo[a]}</td><td class="num">${hp.alpha}</td><td class="num">${hp.gamma}</td><td class="num">${hp.epsilon} → ${hp.epsilon_min}</td><td class="num">${hp.epsilon_decay}</td><td class="num">${hp.episodes.toLocaleString()}</td><td class="num">${e.mean_waiting.toFixed(3)} ± ${e.std_across_seeds.toFixed(3)}</td><td class="num">${Math.round(e.policy_agreement_with_optimal * 100)}%</td></tr>`;
  })).join("");
}

// ------------------------------------------------------------------ benchmark
function runBenchmark() {
  const names = ["optimal", "q_learning", "monte_carlo", "optimal_overflow_aware", "longest_queue", "random", "fixed_time"];
  const EPIS = 400, LEN = 50, out = [];
  for (const name of names) {
    let sum = 0;
    const polRng = mulberry32(99);
    const pol = makePolicy(name, polRng);
    for (let i = 0; i < EPIS; i++) {
      const rng = mulberry32(1000 + i);
      let ns = Math.floor(rng() * 4), ew = Math.floor(rng() * 6), w = 0;
      for (let t = 0; t < LEN; t++) {
        const r = envStep(ns, ew, pol(ns, ew, t), rng);
        ns = r.ns; ew = r.ew; w += r.waiting;
      }
      sum += w / LEN;
    }
    out.push({ name, v: sum / EPIS });
  }
  out.sort((a, b) => a.v - b.v);
  const max = Math.max(...out.map((o) => o.v));
  $("bench").innerHTML = out.map((o) =>
    `<div class="brow"><span>${LABELS[o.name]}</span><div><div class="bar" data-w="${(o.v / max) * 100}" style="background:${COLORS[o.name] || "#556274"}"></div></div><code>${o.v.toFixed(2)}</code></div>`).join("")
    + `<p class="muted small" style="margin:6px 0 0">Average cars waiting per tick · lower is better.</p>`;
  requestAnimationFrame(() => document.querySelectorAll("#bench .bar").forEach((b) => (b.style.width = `${b.dataset.w}%`)));
}

// ------------------------------------------------------------------ ask the API
const ask = { ns: 1, ew: 3 };
async function askAgent() {
  const policy = TABLES[sim.policyName] ? sim.policyName : "q_learning";
  const btn = $("askBtn");
  btn.disabled = true;
  try {
    const r = await fetch(`/api/decide?ns=${ask.ns}&ew=${ask.ew}&policy=${policy}`);
    const j = await r.json();
    $("answer").hidden = false;
    $("ansJson").textContent = JSON.stringify(j, null, 2);
    if (!r.ok) { $("ansAction").textContent = "Error"; $("ansMatch").textContent = ""; return; }
    $("ansAction").textContent = j.action === "NS_GREEN" ? "North–South green" : "East–West green";
    $("ansMatch").textContent = j.matches_optimal ? "✓ same as the optimal policy" : "✗ optimal would choose " + (j.optimal_action === "NS_GREEN" ? "NS" : "EW");
    resetSim(ask.ns, ask.ew);
  } catch (e) {
    $("answer").hidden = false; $("ansAction").textContent = "Offline"; $("ansJson").textContent = String(e);
  } finally { btn.disabled = false; }
}

// ------------------------------------------------------------------ boot
async function boot() {
  document.querySelectorAll(".origin").forEach((el) => (el.textContent = location.origin));
  scene = new IntersectionScene($("scene"), $("stage"));
  scene.onFrame = placeLabels;
  scene.onDrop = dropTag;

  const [pol, res] = await Promise.all([
    fetch("/api/policies").then((r) => r.json()),
    fetch("/api/results").then((r) => r.json()).catch(() => null),
  ]);
  TABLES = pol.policies;
  renderResults(res);

  setPolicy("q_learning");
  resetSim(1, 3);
  scene.setLights(EW);
  setPhase(EW, false);
  $("loading").classList.add("done");

  $("policy").addEventListener("change", (e) => setPolicy(e.target.value));
  $("play").addEventListener("click", () => {
    sim.playing = !sim.playing;
    $("icoPause").hidden = !sim.playing; $("icoPlay").hidden = sim.playing;
    $("play").setAttribute("aria-label", sim.playing ? "Pause" : "Play");
    document.querySelector(".stage-hud.top-left").classList.toggle("paused", !sim.playing);
    schedule();
  });
  $("stepBtn").addEventListener("click", () => { if (!sim.playing) tick(); });
  $("resetBtn").addEventListener("click", () => resetSim());
  $("speed").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    sim.speed = +b.dataset.speed;
    $("speed").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
  });
  document.querySelectorAll(".stepper button").forEach((b) => b.addEventListener("click", () => {
    const k = b.dataset.t, max = k === "ns" ? NS_MAX : EW_MAX;
    ask[k] = Math.min(max, Math.max(0, ask[k] + +b.dataset.d));
    $(k === "ns" ? "askNS" : "askEW").textContent = ask[k];
  }));
  $("askBtn").addEventListener("click", askAgent);
  $("benchBtn").addEventListener("click", () => { $("benchBtn").disabled = true; setTimeout(() => { runBenchmark(); $("benchBtn").disabled = false; }, 30); });
  window.addEventListener("resize", drawSpark);

  // pause the loop when the tab is hidden
  document.addEventListener("visibilitychange", () => { if (document.hidden) clearTimeout(sim.timer); else schedule(); });
  schedule();
}

boot().catch((e) => {
  console.error(e);
  $("loading").innerHTML = `<span>Couldn't start the simulator: ${e.message}</span>`;
});
