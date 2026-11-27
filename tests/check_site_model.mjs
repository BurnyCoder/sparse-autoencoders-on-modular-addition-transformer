// Node check used by tests/test_site_export.py: the JS forward pass and SAE must match Python.
// Usage: node tests/check_site_model.mjs <model.json> <reference.json> [<sae.json>]
import { readFileSync } from "node:fs";
import { ModAddModel, JumpReLUSAE, unpack } from "../site/js/model.js";

const [modelPath, refPath, saePath] = process.argv.slice(2);
const model = new ModAddModel(JSON.parse(readFileSync(modelPath, "utf8")));
const ref = JSON.parse(readFileSync(refPath, "utf8"));
const rm = unpack(ref.resid_mid), lg = unpack(ref.logits);
const p = model.p, d = rm.shape[1];
let maxResid = 0, maxLogit = 0;
for (const [n, idx] of ref.idx.entries()) {
  const a = Math.floor(idx / p), b = idx % p;
  const r = model.residMid(a, b), l = model.logits(a, b);
  for (let i = 0; i < d; i++) maxResid = Math.max(maxResid, Math.abs(r[i] - rm.data[n * d + i]));
  for (let i = 0; i < p; i++) maxLogit = Math.max(maxLogit, Math.abs(l[i] - lg.data[n * p + i]));
}
const out = { maxResid, maxLogit };
if (saePath) {
  const sae = new JumpReLUSAE(JSON.parse(readFileSync(saePath, "utf8")));
  const x = rm.data.slice(0, d);
  out.saeRecon = Array.from(sae.decode(sae.encode(x)).slice(0, 4));
  out.saeL0 = sae.encode(x).filter((v) => v > 0).length;
}
console.log(JSON.stringify(out));
