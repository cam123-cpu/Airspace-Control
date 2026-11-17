/* pixelModel: in-browser drone / drone+payload / bird / clutter classifier.
 * Trained by train_browser_model.py on 32x32 grayscale crops taken from the sim's own PTZ camera.
 * The same cropFromCanvas() is used for capture and at runtime, so inputs match exactly.
 * Works in browsers and in Node (classify/load only; cropFromCanvas needs a canvas).
 */
(function (root) {
  "use strict";
  const CLASSES = ["drone", "drone_payload", "bird", "clutter"];
  const B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  const LUT = new Int16Array(128).fill(-1);
  for (let i = 0; i < 64; i++) LUT[B64.charCodeAt(i)] = i;

  function b64bytes(s) {
    s = String(s).replace(/[^A-Za-z0-9+/]/g, "");
    const n = Math.floor((s.length * 3) / 4), out = new Uint8Array(n);
    let o = 0;
    for (let i = 0; i < s.length; i += 4) {
      const a = LUT[s.charCodeAt(i)], b = LUT[s.charCodeAt(i + 1)];
      const c = i + 2 < s.length ? LUT[s.charCodeAt(i + 2)] : 0, d = i + 3 < s.length ? LUT[s.charCodeAt(i + 3)] : 0;
      const v = (a << 18) | (b << 12) | (c << 6) | d;
      if (o < n) out[o++] = (v >> 16) & 255;
      if (o < n && i + 2 < s.length) out[o++] = (v >> 8) & 255;
      if (o < n && i + 3 < s.length) out[o++] = v & 255;
    }
    return out;
  }

  const pixelModel = { CLASSES, model: null };

  /** Load the parsed model.json object. Validates shapes; throws on a malformed model. */
  pixelModel.load = function (m) {
    if (!m || m.format !== "pixelmlp-int8-v1") throw new Error("pixelModel: unknown model format");
    const layers = m.layers.map((L, k) => {
      const wb = b64bytes(L.w), bb = b64bytes(L.b);
      if (wb.length !== L.nin * L.nout) throw new Error(`pixelModel: layer ${k} weight size mismatch`);
      if (bb.length !== L.nout * 4) throw new Error(`pixelModel: layer ${k} bias size mismatch`);
      const W = new Int8Array(wb.buffer, wb.byteOffset, wb.length);
      const dv = new DataView(bb.buffer, bb.byteOffset, bb.length), b = new Float64Array(L.nout);
      for (let j = 0; j < L.nout; j++) b[j] = dv.getFloat32(j * 4, true);
      if (!(L.scale > 0) || !isFinite(L.scale)) throw new Error(`pixelModel: layer ${k} bad scale`);
      return { nin: L.nin, nout: L.nout, W, b, scale: L.scale, relu: L.act === "relu" };
    });
    const features = m.input.features || "pixels";
    const nin = (features === "hog" ? 0 : m.input.size * m.input.size) + (features === "pixels" ? 0 : 324);
    if (layers[0].nin !== nin) throw new Error("pixelModel: input size mismatch");
    this.model = { layers, T: m.temperature || 1, stdFloor: m.input.std_floor, size: m.input.size, features, hogScale: m.input.hog_scale || 1, meta: m };
    return this;
  };

  /** Per-crop standardization, identical to the Python training code. gray: 1024 values 0..255. */
  pixelModel.normalize = function (gray) {
    const n = gray.length, x = new Float64Array(n);
    let mean = 0;
    for (let i = 0; i < n; i++) { x[i] = gray[i] / 255; mean += x[i]; }
    mean /= n;
    let v = 0;
    for (let i = 0; i < n; i++) { const d = x[i] - mean; v += d * d; }
    const sd = Math.max(Math.sqrt(v / n), this.model.stdFloor);
    for (let i = 0; i < n; i++) x[i] = (x[i] - mean) / sd;
    return x;
  };

  /** hogLite: edge-orientation histograms on the normalised 32x32 crop.
   *  Central differences (clamped borders), 9 unsigned orientation bins with linear interpolation,
   *  8x8-pixel cells (4x4 grid), 2x2-cell blocks (3x3 grid), each block L2-normalised -> 324 values. */
  pixelModel.hogLite = function (x) {
    const S = 32, NB = 9, cells = new Float64Array(16 * NB);
    for (let i = 0; i < S; i++) for (let j = 0; j < S; j++) {
      const gx = x[i * S + Math.min(j + 1, S - 1)] - x[i * S + Math.max(j - 1, 0)];
      const gy = x[Math.min(i + 1, S - 1) * S + j] - x[Math.max(i - 1, 0) * S + j];
      const mag = Math.sqrt(gx * gx + gy * gy);
      if (mag === 0) continue;
      let a = Math.atan2(gy, gx);
      if (a < 0) a += Math.PI;
      if (a >= Math.PI) a -= Math.PI;
      const b = a / (Math.PI / NB) - 0.5, b0 = Math.floor(b), w1 = b - b0;
      const c = ((i >> 3) * 4 + (j >> 3)) * NB;
      cells[c + ((b0 % NB) + NB) % NB] += mag * (1 - w1);
      cells[c + (((b0 + 1) % NB) + NB) % NB] += mag * w1;
    }
    const out = new Float64Array(9 * 36);
    let o = 0;
    for (let bi = 0; bi < 3; bi++) for (let bj = 0; bj < 3; bj++) {
      const start = o;
      let ss = 0;
      for (const [ci, cj] of [[bi, bj], [bi, bj + 1], [bi + 1, bj], [bi + 1, bj + 1]])
        for (let k = 0; k < NB; k++) { const v = cells[(ci * 4 + cj) * NB + k]; out[o++] = v; ss += v * v; }
      const n = Math.sqrt(ss + 1e-2);
      for (let k = start; k < o; k++) out[k] /= n;
    }
    return out;
  };

  /** Feature vector the network expects (pixels, hogLite, or both), per model.json input.features. */
  pixelModel.features = function (gray) {
    const x = this.normalize(gray), f = this.model.features;
    if (f === "pixels") return x;
    const h = this.hogLite(x), k = this.model.hogScale;
    if (f === "hog") { for (let i = 0; i < h.length; i++) h[i] *= k; return h; }
    const v = new Float64Array(x.length + h.length);
    v.set(x, 0);
    for (let i = 0; i < h.length; i++) v[x.length + i] = h[i] * k;
    return v;
  };

  /** Classify a 32x32 grayscale crop (array-like of 1024 values, 0..255). */
  pixelModel.classify = function (gray) {
    if (!this.model) throw new Error("pixelModel: call load() first");
    let h = this.features(gray);
    for (const L of this.model.layers) {
      const z = new Float64Array(L.nout);
      for (let i = 0; i < L.nin; i++) {
        const hi = h[i];
        if (hi === 0) continue;
        const row = i * L.nout;
        for (let j = 0; j < L.nout; j++) z[j] += hi * L.W[row + j];
      }
      for (let j = 0; j < L.nout; j++) { z[j] = z[j] * L.scale + L.b[j]; if (L.relu && z[j] < 0) z[j] = 0; }
      h = z;
    }
    const T = this.model.T;
    let mx = -Infinity;
    for (let j = 0; j < h.length; j++) mx = Math.max(mx, h[j] / T);
    let s = 0;
    const p = new Array(h.length);
    for (let j = 0; j < h.length; j++) { p[j] = Math.exp(h[j] / T - mx); s += p[j]; }
    let cls = 0;
    for (let j = 0; j < p.length; j++) { p[j] /= s; if (p[j] > p[cls]) cls = j; }
    return { probs: p, cls, label: CLASSES[cls], conf: p[cls] };
  };

  let tmp = null, tctx = null;
  /** Square crop of side `size` px centred on (cx,cy), shifted to stay inside the canvas,
   *  resampled to 32x32 and converted to grayscale (0.299R+0.587G+0.114B, rounded). */
  pixelModel.cropFromCanvas = function (ctx, cx, cy, size) {
    const cv = ctx.canvas, N = 32;
    let s = Math.round(Math.max(8, size));
    s = Math.min(s, cv.width, cv.height);
    const sx = Math.min(Math.max(Math.round(cx - s / 2), 0), cv.width - s);
    const sy = Math.min(Math.max(Math.round(cy - s / 2), 0), cv.height - s);
    if (!tmp) {
      tmp = (typeof OffscreenCanvas !== "undefined") ? new OffscreenCanvas(N, N) : Object.assign(document.createElement("canvas"), { width: N, height: N });
      tctx = tmp.getContext("2d", { willReadFrequently: true });
    }
    tctx.imageSmoothingEnabled = true;
    tctx.imageSmoothingQuality = "high";
    tctx.clearRect(0, 0, N, N);
    tctx.drawImage(cv, sx, sy, s, s, 0, 0, N, N);
    const d = tctx.getImageData(0, 0, N, N).data, g = new Uint8Array(N * N);
    for (let i = 0; i < N * N; i++) g[i] = Math.round(0.299 * d[4 * i] + 0.587 * d[4 * i + 1] + 0.114 * d[4 * i + 2]);
    return g;
  };

  /** Recommended crop side for a target expected to be `px` pixels across. */
  pixelModel.cropSize = function (px) { return Math.max(12, Math.min(220, 1.6 * px)); };

  root.pixelModel = pixelModel;
  if (typeof module !== "undefined" && module.exports) module.exports = pixelModel;
})(typeof globalThis !== "undefined" ? globalThis : this);
