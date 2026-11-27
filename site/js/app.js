// Interactive explorer: feature browser + steering playground, computed live from exported weights.
//
// Local: loads data/manifest.json (written by `python run.py site`), builds the model and SAEs
// (model.js), caches resid_mid / mlp_post for all 113*113 inputs, then renders:
//   - a heatmap of one SAE feature over the (a, b) grid with hover tooltips,
//   - its tuning curve against the mechanism angle psi_k (analytic A+B coordinates),
//   - a steering playground: signed single-feature steering, ring transport by delta and the
//     antipodal swap (rings + calibration exported from the Python pipeline), with live logits.
// Global context: these are the same interventions as src/modsae/steering.py (spliced SAE writes,
// full-residual replacement unless "MLP input only" is ticked).
import { ModAddModel, JumpReLUSAE, argmax } from "./model.js";

const P = 113;
const css = (name) => getComputedStyle(document.querySelector(".viz-root")).getPropertyValue(name).trim();
const tooltip = document.getElementById("tooltip");
const showTip = (ev, html) => { tooltip.innerHTML = html; tooltip.style.display = "block"; tooltip.style.left = `${ev.clientX + 12}px`; tooltip.style.top = `${ev.clientY + 12}px`; };
const hideTip = () => { tooltip.style.display = "none"; };

async function getJSON(path) { const r = await fetch(path); if (!r.ok) throw new Error(`${path}: ${r.status}`); return r.json(); }

// sequential blue ramp (dataviz reference steps 100 -> 700)
const RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"];
function rampColor(t) { const x = Math.max(0, Math.min(0.9999, t)) * (RAMP.length - 1); const i = Math.floor(x); return mix(RAMP[i], RAMP[i + 1] || RAMP[i], x - i); }
function mix(c1, c2, f) { const a = parseInt(c1.slice(1), 16), b = parseInt(c2.slice(1), 16); const ch = (s) => Math.round(((a >> s) & 255) * (1 - f) + ((b >> s) & 255) * f); return `rgb(${ch(16)},${ch(8)},${ch(0)})`; }

function mechCoords(a, b, k) {
  const w = (2 * Math.PI * k) / P;
  const re = Math.cos(w * a) + Math.cos(w * b), im = Math.sin(w * a) + Math.sin(w * b);
  return { psi: Math.atan2(im, re), r: Math.hypot(re, im) };
}

const state = { model: null, saes: {}, cache: {}, manifest: null };

async function main() {
  const manifest = await getJSON("data/manifest.json");
  state.manifest = manifest;
  document.getElementById("build-date").textContent = manifest.date;
  document.getElementById("status").innerHTML = manifest.status_html || "";
  state.model = new ModAddModel(await getJSON(`data/${manifest.model}`));
  for (const s of manifest.saes) state.saes[s.id] = { sae: new JumpReLUSAE(await getJSON(`data/${s.file}`)), info: s };
  // cache activations for the whole grid
  const rm = [], mp = [];
  for (let a = 0; a < P; a++) for (let b = 0; b < P; b++) { const r = state.model.residMid(a, b); rm.push(r); mp.push(state.model.mlpPost(r)); }
  state.cache.resid_mid = rm; state.cache.mlp_post = mp;
  setupBrowser();
  setupSteering();
  document.getElementById("tables").innerHTML = manifest.tables_html || "";
}

// ---------------------------------------------------------------- feature browser
function setupBrowser() {
  const saeSel = document.getElementById("sae-select"), featSel = document.getElementById("feature-select");
  for (const id of Object.keys(state.saes)) saeSel.add(new Option(id.replace("sae_", ""), id));
  const fillFeatures = () => {
    const { sae } = state.saes[saeSel.value];
    const f = sae.meta.features;
    featSel.innerHTML = "";
    const order = f.alive_idx.map((j, n) => n).sort((x, y) => (f.r2_key_best[y] ?? -1) - (f.r2_key_best[x] ?? -1));
    for (const n of order) featSel.add(new Option(`#${f.alive_idx[n]}  k=${f.k_key[n]}  R²=${(f.r2_key_best[n] ?? 0).toFixed(2)}${f.ring_mask[n] ? "  ring" : ""}`, n));
    drawFeature();
  };
  saeSel.onchange = fillFeatures; featSel.onchange = drawFeature;
  fillFeatures();
}

function featureActivations(entry, j) {
  const { sae, info } = entry;
  const X = state.cache[info.site];
  const { W_enc, b_enc, b_dec, threshold } = sae.w;
  const out = new Float32Array(P * P);
  for (let n = 0; n < X.length; n++) {
    let h = b_enc.data[j];
    for (let i = 0; i < sae.dIn; i++) h += (X[n][i] - b_dec.data[i]) * W_enc.data[i * sae.dSae + j];
    out[n] = h > threshold.data[j] && h > 0 ? h : 0;
  }
  return out;
}

function drawFeature() {
  const entry = state.saes[document.getElementById("sae-select").value];
  const n = +document.getElementById("feature-select").value;
  const f = entry.sae.meta.features, j = f.alive_idx[n], k = f.k_key[n], gamma = f.gamma[n] ?? 0;
  const acts = featureActivations(entry, j);
  const max = Math.max(...acts) || 1;
  const cv = document.getElementById("heatmap"), ctx = cv.getContext("2d"), cell = cv.width / P;
  ctx.fillStyle = css("--surface-1"); ctx.fillRect(0, 0, cv.width, cv.height);
  for (let a = 0; a < P; a++) for (let b = 0; b < P; b++) {
    const v = acts[a * P + b];
    if (v > 0) { ctx.fillStyle = rampColor(v / max); ctx.fillRect(b * cell, (P - 1 - a) * cell, cell, cell); }
  }
  cv.onmousemove = (ev) => {
    const rect = cv.getBoundingClientRect(), b = Math.floor((ev.clientX - rect.left) / cell), a = P - 1 - Math.floor((ev.clientY - rect.top) / cell);
    if (a >= 0 && a < P && b >= 0 && b < P) showTip(ev, `a=${a}, b=${b}, a+b mod 113=${(a + b) % P}<br>activation ${acts[a * P + b].toFixed(3)}`);
  };
  cv.onmouseleave = hideTip;
  // tuning curve
  const svg = document.getElementById("tuning"), W = +svg.getAttribute("width"), H = +svg.getAttribute("height"), m = 34;
  const pts = [];
  let ymax = 1e-9;
  for (let a = 0; a < P; a += 2) for (let b = 0; b < P; b += 2) {
    const v = acts[a * P + b]; if (v <= 0) continue;
    const { psi, r } = mechCoords(a, b, k); const y = v / Math.max(r, 1e-3) ** gamma; ymax = Math.max(ymax, y); pts.push([psi, y]);
  }
  const sx = (x) => m + ((x + Math.PI) / (2 * Math.PI)) * (W - 2 * m), sy = (y) => H - m - (y / ymax) * (H - 2 * m);
  svg.innerHTML = `<line x1="${m}" y1="${H - m}" x2="${W - m}" y2="${H - m}" stroke="${css("--grid")}"/>` +
    pts.map(([x, y]) => `<circle cx="${sx(x).toFixed(1)}" cy="${sy(y).toFixed(1)}" r="1.6" fill="${css("--series-1")}" opacity="0.6"/>`).join("") +
    `<text x="${W / 2}" y="${H - 6}" text-anchor="middle">ψ_k (radians, −π … π), k = ${k}</text>` +
    `<text x="10" y="${H / 2}" transform="rotate(-90 10 ${H / 2})" text-anchor="middle">activation / r^${gamma}</text>`;
  document.getElementById("feature-info").innerHTML =
    `feature #${j}: key frequency k = ${k}, held-out arc-model R² = ${(f.r2_key_best[n] ?? NaN).toFixed(3)}, γ = ${gamma}, ` +
    `tuning width = ${f.width[n] == null ? "–" : (f.width[n] * 100).toFixed(0) + "% of the circle"}, ring = ${f.ring_label[n]}, ` +
    `active on ${((acts.filter((v) => v > 0).length / acts.length) * 100).toFixed(1)}% of inputs.`;
}

// ---------------------------------------------------------------- steering
function setupSteering() {
  const steerable = Object.values(state.saes).find((e) => e.info.site === "resid_mid");
  if (!steerable) return;
  state.steer = steerable;
  const f = steerable.sae.meta.features, sel = document.getElementById("steer-feature");
  sel.add(new Option("(none)", -1));
  f.alive_idx.forEach((j, n) => sel.add(new Option(`#${j} k=${f.k_key[n]}${f.ring_mask[n] ? " ring" : ""}`, n)));
  state.antipodal = false;
  for (const id of ["in-a", "in-b", "steer-feature", "steer-c", "mlp-only", "transport-d"]) document.getElementById(id).oninput = drawSteering;
  document.getElementById("antipodal").onclick = () => { state.antipodal = !state.antipodal; drawSteering(); };
  document.getElementById("reset").onclick = () => { document.getElementById("steer-c").value = 0; document.getElementById("transport-d").value = 0; state.antipodal = false; drawSteering(); };
  drawSteering();
}

function transport(z, shiftByRing) {
  const { sae, info } = state.steer, f = sae.meta.features;
  const label = new Array(sae.dSae).fill(-1), angle = new Array(sae.dSae).fill(null);
  f.alive_idx.forEach((j, n) => { label[j] = f.ring_label[n]; angle[j] = f.ring_angle[n]; });
  const out = new Float32Array(z.length);
  for (let j = 0; j < z.length; j++) {
    if (z[j] === 0) continue;
    const ring = label[j], shift = shiftByRing[ring];
    if (ring < 0 || shift === undefined || angle[j] == null) { out[j] += z[j]; continue; }
    let best = j, bestD = Infinity;
    for (let i = 0; i < z.length; i++) {
      if (label[i] !== ring || angle[i] == null) continue;
      const d = Math.abs(Math.atan2(Math.sin(angle[i] - angle[j] - shift), Math.cos(angle[i] - angle[j] - shift)));
      if (d < bestD) { bestD = d; best = i; }
    }
    out[best] += z[j];
  }
  return out;
}

function drawSteering() {
  const a = Math.max(0, Math.min(112, +document.getElementById("in-a").value | 0)), b = Math.max(0, Math.min(112, +document.getElementById("in-b").value | 0));
  const c = +document.getElementById("steer-c").value, n = +document.getElementById("steer-feature").value;
  const delta = +document.getElementById("transport-d").value, mlpOnly = document.getElementById("mlp-only").checked;
  document.getElementById("c-val").textContent = c; document.getElementById("d-val").textContent = delta;
  const { sae, info } = state.steer, model = state.model;
  const x0 = model.residMid(a, b);
  let x = x0, note = "clean model";
  const calib = info.calibration || {};
  if (delta !== 0 || state.antipodal) {
    const shifts = {};
    for (const [ring, cal] of Object.entries(calib)) if (cal.R >= 0.5) shifts[+ring] = (state.antipodal ? Math.PI : 0) + cal.slope * delta;
    x = sae.decode(transport(sae.encode(x0), shifts));
    note = `SAE ring transport (δ=${delta}${state.antipodal ? ", + antipodal swap" : ""}); target ${(a + b + delta) % P}`;
  }
  if (n >= 0 && c !== 0) {
    const d = sae.decoderRow(sae.meta.features.alive_idx[n]);
    x = x.map((v, i) => v + c * info.resid_norm * d[i]);
    note += `; + ${c}·‖x̄‖·d(#${sae.meta.features.alive_idx[n]})`;
  }
  const logits = model.logitsFromResidMid(x, mlpOnly ? x0 : null);
  const pred = argmax(logits);
  const svg = document.getElementById("logits"), W = +svg.getAttribute("width"), H = +svg.getAttribute("height"), m = 30;
  const lo = Math.min(...logits), hi = Math.max(...logits), bw = (W - 2 * m) / P;
  const sy = (v) => H - m - ((v - lo) / (hi - lo || 1)) * (H - 2 * m);
  const truth = (a + b) % P, target = (a + b + delta) % P;
  let html = "";
  logits.forEach((v, i) => {
    const color = i === pred ? css("--series-2") : css("--series-1");
    html += `<rect data-i="${i}" x="${(m + i * bw + 0.5).toFixed(1)}" y="${sy(Math.max(v, lo)).toFixed(1)}" width="${Math.max(bw - 1, 1).toFixed(1)}" height="${(H - m - sy(Math.max(v, lo))).toFixed(1)}" fill="${color}" rx="1"/>`;
  });
  html += `<text x="${m}" y="${H - 8}">output c = 0 … 112 (orange bar = argmax)</text>`;
  svg.innerHTML = html;
  svg.querySelectorAll("rect").forEach((r) => { r.onmousemove = (ev) => showTip(ev, `c=${r.dataset.i}: logit ${logits[+r.dataset.i].toFixed(2)}`); r.onmouseleave = hideTip; });
  document.getElementById("steer-info").innerHTML = `${a} + ${b} mod 113 = <strong>${truth}</strong> · model output <strong>${pred}</strong> · ${note}` +
    (delta !== 0 ? ` · ${pred === target ? "✓ hit a+b+δ" : "✗ missed a+b+δ"}` : "");
}

main().catch((e) => { document.getElementById("status").textContent = `Could not load data: ${e.message}. Run \`python run.py site\` first.`; });
