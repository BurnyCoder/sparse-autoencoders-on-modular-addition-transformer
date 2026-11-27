// Exact client-side forward pass of the 1-layer modular-addition transformer and its JumpReLU SAE.
//
// Local: rebuilds Float32Arrays from site/data/*.json (base64 little-endian float32, weights in
// nn.Linear (out, in) layout) and computes, for the '=' position of input [a, b, p]:
//   resid_pre_j = E[tok_j] + P[j]; attention (4 heads, scores / sqrt(d_head), causal softmax);
//   resid_mid = resid_pre_2 + O(z); mlp_post = ReLU(W_in resid_mid + b_in);
//   logits = U(resid_mid + W_out mlp_post + b_out) + b_U.
// SAE (SAELens JumpReLU inference): h = (x - b_dec) W_enc + b_enc; z = ReLU(h) * [h > threshold];
// x_hat = z W_dec + b_dec.
// Global context: the steering playground edits resid_mid (full residual stream, as in the paper)
// and calls logitsFromResidMid, so every number on the page is computed from the real weights.
// Equality with the Python pipeline is checked by tests/test_site_export.py (via Node).

export function unpack(obj) {
  const bin = typeof atob === "function" ? atob(obj.b64) : Buffer.from(obj.b64, "base64").toString("binary");
  const buf = new ArrayBuffer(bin.length);
  const bytes = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return { data: new Float32Array(buf), shape: obj.shape };
}

// y = W x + b for W of shape (out, in) stored row-major.
function linear(W, b, x) {
  const [out, inn] = W.shape;
  const y = new Float32Array(out);
  for (let o = 0; o < out; o++) {
    let s = b ? b.data[o] : 0;
    const row = o * inn;
    for (let i = 0; i < inn; i++) s += W.data[row + i] * x[i];
    y[o] = s;
  }
  return y;
}

export class ModAddModel {
  constructor(json) {
    this.p = json.p;
    this.nHeads = json.n_heads;
    this.dHead = json.d_head;
    this.w = {};
    for (const [k, v] of Object.entries(json.weights)) this.w[k] = unpack(v);
    this.dModel = this.w["embed.weight"].shape[1];
  }

  residPre(tok, pos) {
    const d = this.dModel, E = this.w["embed.weight"].data, P = this.w["pos_embed.weight"].data;
    const x = new Float32Array(d);
    for (let i = 0; i < d; i++) x[i] = E[tok * d + i] + P[pos * d + i];
    return x;
  }

  residMid(a, b) {
    const w = this.w, H = this.nHeads, dh = this.dHead;
    const xs = [this.residPre(a, 0), this.residPre(b, 1), this.residPre(this.p, 2)];
    const q = linear(w["blocks.0.attn.q.weight"], w["blocks.0.attn.q.bias"], xs[2]);
    const ks = xs.map((x) => linear(w["blocks.0.attn.k.weight"], w["blocks.0.attn.k.bias"], x));
    const vs = xs.map((x) => linear(w["blocks.0.attn.v.weight"], w["blocks.0.attn.v.bias"], x));
    const z = new Float32Array(H * dh);
    for (let h = 0; h < H; h++) {
      const scores = ks.map((k) => {
        let s = 0;
        for (let i = 0; i < dh; i++) s += q[h * dh + i] * k[h * dh + i];
        return s / Math.sqrt(dh);
      });
      const m = Math.max(...scores);
      const e = scores.map((s) => Math.exp(s - m));
      const tot = e.reduce((u, v) => u + v, 0);
      for (let j = 0; j < 3; j++) for (let i = 0; i < dh; i++) z[h * dh + i] += (e[j] / tot) * vs[j][h * dh + i];
    }
    const attnOut = linear(w["blocks.0.attn.o.weight"], w["blocks.0.attn.o.bias"], z);
    return xs[2].map((v, i) => v + attnOut[i]);
  }

  mlpPost(residMid) {
    return linear(this.w["blocks.0.mlp.in.weight"], this.w["blocks.0.mlp.in.bias"], residMid).map((v) => Math.max(v, 0));
  }

  // skip = residual carried around the MLP (defaults to resid_mid itself: full-residual intervention)
  logitsFromResidMid(residMid, skip = null) {
    const h = this.mlpPost(residMid);
    const mlpOut = linear(this.w["blocks.0.mlp.out.weight"], this.w["blocks.0.mlp.out.bias"], h);
    const s = skip || residMid;
    const post = s.map((v, i) => v + mlpOut[i]);
    return linear(this.w["unembed.weight"], this.w["unembed.bias"], post);
  }

  logits(a, b) {
    return this.logitsFromResidMid(this.residMid(a, b));
  }
}

export class JumpReLUSAE {
  constructor(json) {
    this.meta = json;
    this.w = {};
    for (const [k, v] of Object.entries(json.weights)) this.w[k] = unpack(v);
    [this.dIn, this.dSae] = this.w["W_enc"].shape;
  }

  encode(x) {
    const { W_enc, b_enc, b_dec, threshold } = this.w;
    const z = new Float32Array(this.dSae);
    for (let j = 0; j < this.dSae; j++) {
      let h = b_enc.data[j];
      for (let i = 0; i < this.dIn; i++) h += (x[i] - b_dec.data[i]) * W_enc.data[i * this.dSae + j];
      z[j] = h > threshold.data[j] && h > 0 ? h : 0;
    }
    return z;
  }

  decode(z) {
    const { W_dec, b_dec } = this.w;
    const x = Float32Array.from(b_dec.data);
    for (let j = 0; j < this.dSae; j++) {
      if (z[j] === 0) continue;
      for (let i = 0; i < this.dIn; i++) x[i] += z[j] * W_dec.data[j * this.dIn + i];
    }
    return x;
  }

  decoderRow(j) {
    return this.w["W_dec"].data.slice(j * this.dIn, (j + 1) * this.dIn);
  }
}

export function argmax(arr) {
  let best = 0;
  for (let i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i;
  return best;
}
