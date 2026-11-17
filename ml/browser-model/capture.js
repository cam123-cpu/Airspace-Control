/* Injected into the sim page by train_browser_model.py. Uses the page's own globals
 * (W, step, restart, scenSel, $, UI, camBasis, proj, drawCam, cctx, camC, CW, CH, PTZv, LIGHTS, ROADLAMPS, mulberry32)
 * and pixelModel.cropFromCanvas from infer.js, so capture and runtime crop identically. */
globalThis.__capture = function (cfg) {
  const R = mulberry32((cfg.seed * 7919 + cfg.salt) >>> 0);
  const rnd = (a, b) => a + (b - a) * R();
  const pick = a => a[Math.floor(R() * a.length)];
  const D = Math.PI / 180;
  // Suppress HUD / overlays on the composited camera canvas: the model sees the raw sensor picture.
  if (!globalThis.__origLine) globalThis.__origLine = line;
  line = function (ctx, a, b) { if (ctx === cctx) return; globalThis.__origLine(ctx, a, b); };
  cctx.fillText = function () {}; cctx.strokeRect = function () {};
  playing = false;
  scenSel.value = cfg.scen; $("seed").value = cfg.seed; restart();
  UI.labels = false;
  const crops = [], meta = [], counts = [0, 0, 0, 0];
  const az = p => (Math.atan2(p[0] - PTZv[0], p[1] - PTZv[1]) / D + 360) % 360;
  const el = p => Math.asin((p[2] - PTZv[2]) / Math.hypot(p[0] - PTZv[0], p[1] - PTZv[1], p[2] - PTZv[2])) / D;
  const rng = p => Math.hypot(p[0] - PTZv[0], p[1] - PTZv[1], p[2] - PTZv[2]);
  function env() {
    UI.band = R() < cfg.thermalFrac ? "thermal" : "vis";
    UI.wind = Math.round(rnd(0, 12)); UI.clutter = Math.round(rnd(0, 8));
    if ("tod" in UI) { const r = R(); UI.tod = r < 0.5 ? 0 : r < 0.75 ? 0.45 : 1; }
    if ("wx" in UI) { const r = R(); UI.wx = r < 0.6 ? "clear" : r < 0.8 ? "fog" : "rain"; }
    W.cam.det = null; W.cam.cue = null; W.cam.lock = false;
    W.cam.rate = R() < 0.7 ? 0 : rnd(0, 40);
    W.insect = null;
  }
  function aim(p, span, s, err) {
    const z = Math.max(5, rng(p));
    const fov = Math.min(50, Math.max(0.25, 2 * Math.atan((CW * span) / (2 * s * z)) / D));
    W.cam.fov = fov;
    W.cam.az = (az(p) + (err ? rnd(-0.3, 0.3) * fov : 0) + 360) % 360;
    W.cam.el = el(p) + (err ? rnd(-0.22, 0.22) * fov : 0);
    W.cam.range = z;
  }
  const overlaps = (x0, y0, x1, y1) => (W.labels || []).some(l => x0 < l.x1 && x1 > l.x0 && y0 < l.y1 && y1 > l.y0);
  function keep(cls, cx, cy, size, px, kind) {
    const g = pixelModel.cropFromCanvas(cctx, cx, cy, size);
    crops.push(g); counts[cls]++;
    meta.push({ cls, px: Math.round(px * 10) / 10, band: UI.band, tod: UI.tod ?? 0, wx: UI.wx ?? "clear", scen: cfg.scen, seed: cfg.seed, t: Math.round(W.t), kind, crop: Math.round(size) });
  }
  function shotTarget() {
    const ents = W.ents.filter(e => e.alive && !(e.type === "aircraft" && e.wait > 0) && e.type !== "car" &&
      !(e.type === "drone" && (e.phase === "wait" || e.phase === "done" || e.pos[2] < 2)) && !(e.type === "payload" && e.landed));
    const byCls = { 0: [], 1: [], 2: [], 3: [] };
    for (const e of ents) {
      if (e.type === "drone") { byCls[0].push(e); byCls[1].push(e); } // payload is toggled per shot below
      else if (e.type === "bird") byCls[2].push(e);
      else byCls[3].push(e);
    }
    const avail = [0, 1, 2, 3].filter(c => byCls[c].length);
    if (!avail.length) return false;
    avail.sort((a, b) => counts[a] - counts[b]);
    const c = R() < 0.75 ? avail[0] : pick(avail);
    const e = pick(byCls[c]);
    const s = Math.exp(rnd(Math.log(6), Math.log(80)));
    env();
    aim(e.pos, e.span || 0.35, s, true);
    const hadPayload = e.hasPayload;
    if (e.type === "drone") e.hasPayload = (c === 1); // render the same airframe with or without a hanging package
    drawCam();
    if (e.type === "drone") e.hasPayload = hadPayload;
    const B = camBasis(W.cam), p = proj(B, e.pos);
    if (!p) return false;
    let best = null, bd = 1e9;
    for (const l of W.labels) {
      const dx = Math.max(l.x0 - p[0], 0, p[0] - l.x1), dy = Math.max(l.y0 - p[1], 0, p[1] - l.y1), d = Math.hypot(dx, dy);
      if (d < bd) { bd = d; best = l; }
    }
    if (!best || bd > 4) return false;
    const m = Math.max(best.x1 - best.x0, best.y1 - best.y0);
    const size = Math.max(10, m * rnd(1.25, 2.1));
    const cx = (best.x0 + best.x1) / 2 + rnd(-0.15, 0.15) * size, cy = (best.y0 + best.y1) / 2 + rnd(-0.15, 0.15) * size;
    keep(best.cls, cx, cy, size, m, e.type === "bird" ? "bird:" + e.kind : e.type);
    return true;
  }
  function shotEmptyTrack() { // where a radar clutter track would cue the camera: nothing there
    env();
    const p = [rnd(-900, 900), rnd(-900, 700), rnd(5, 150)], s = rnd(8, 60);
    aim(p, 0.35, s, false); drawCam();
    const size = pixelModel.cropSize(s) * rnd(0.8, 1.3), cx = CW / 2 + rnd(-0.1, 0.1) * size, cy = CH / 2 + rnd(-0.1, 0.1) * size;
    if (overlaps(cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2)) return false;
    keep(3, cx, cy, size, 0, "empty"); return true;
  }
  function shotPatch() {
    env();
    W.cam.az = rnd(0, 360); W.cam.el = rnd(-8, 30); W.cam.fov = Math.exp(rnd(Math.log(1), Math.log(40))); drawCam();
    const size = rnd(12, 120), cx = rnd(size / 2, CW - size / 2), cy = rnd(size / 2, CH - size / 2);
    if (overlaps(cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2)) return false;
    keep(3, cx, cy, size, 0, "patch"); return true;
  }
  function shotLight() {
    env();
    const L = pick(R() < 0.6 ? LIGHTS : ROADLAMPS);
    W.cam.fov = Math.exp(rnd(Math.log(1), Math.log(30)));
    W.cam.az = (az(L) + rnd(-0.2, 0.2) * W.cam.fov + 360) % 360; W.cam.el = el(L) + rnd(-0.15, 0.15) * W.cam.fov; drawCam();
    const p = proj(camBasis(W.cam), L); if (!p || p[0] < 0 || p[0] > CW || p[1] < 0 || p[1] > CH) return false;
    const size = rnd(12, 80);
    if (overlaps(p[0] - size / 2, p[1] - size / 2, p[0] + size / 2, p[1] + size / 2)) return false;
    keep(3, p[0] + rnd(-0.2, 0.2) * size, p[1] + rnd(-0.2, 0.2) * size, size, 0, "light"); return true;
  }
  function shotInsect() {
    env(); UI.band = "vis"; UI.clutter = Math.max(1, UI.clutter);
    W.cam.az = rnd(0, 360); W.cam.el = rnd(0, 25); W.cam.fov = rnd(2, 30);
    W.insect = { x: rnd(60, CW - 60), y: rnd(60, CH - 60), vx: 0, vy: 0, life: 0.6 };
    drawCam();
    const l = (W.labels || []).find(q => /insect/.test(q.name)); W.insect = null;
    if (!l) return false;
    const m = Math.max(l.x1 - l.x0, l.y1 - l.y0), size = m * rnd(0.6, 1.6);
    keep(3, (l.x0 + l.x1) / 2 + rnd(-0.15, 0.15) * size, (l.y0 + l.y1) / 2 + rnd(-0.15, 0.15) * size, size, m, "insect"); return true;
  }
  while (W.t < cfg.tmax) {
    const adv = Math.floor(rnd(30, 140));
    for (let i = 0; i < adv; i++) step(0.05);
    for (let k = 0; k < cfg.shots; k++) {
      const r = R();
      if (r < 0.64) shotTarget();
      else if (r < 0.76) shotEmptyTrack();
      else if (r < 0.87) shotPatch();
      else if (r < 0.94) shotLight();
      else shotInsect();
    }
  }
  // pack crops as base64
  const all = new Uint8Array(crops.length * 1024);
  crops.forEach((g, i) => all.set(g, i * 1024));
  let bin = "";
  for (let i = 0; i < all.length; i += 0x8000) bin += String.fromCharCode.apply(null, all.subarray(i, i + 0x8000));
  return { b64: btoa(bin), meta, counts };
};
