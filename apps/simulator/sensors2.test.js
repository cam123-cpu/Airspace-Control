// node sensors2.test.js: sanity and calibration checks for sensors2.js
const S = require("./sensors2.js");
let pass = 0, fail = 0;
function ok(cond, msg) { if (cond) { pass++; } else { fail++; console.log("FAIL:", msg); } }
function near(a, b, tol, msg) { ok(Math.abs(a - b) <= tol, `${msg}: got ${a.toFixed(4)}, want ${b} ± ${tol}`); }
function seeded(seed) { let a = seed >>> 0; return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
const R = S.radar, T = S.thermal, F = S.rf, A = S.acoustic;

// ---- radar calibration: Pd = 0.9 at datasheet range for datasheet RCS
near(R.pd(0.02, 3100, { preset: "iris" }), 0.9, 0.005, "IRIS Pd at 3100 m, 0.02 m²");
near(R.pd(0.032, 1000, { preset: "echoguard" }), 0.9, 0.005, "EchoGuard Pd at 1000 m, Phantom 4");
near(R.swerlingISnrFor(0.9, 1e-6), 130.1, 0.5, "Swerling I SNR for Pd .9 Pfa 1e-6");
// monotonic in range, and fourth-root range scaling with RCS
let prev = 1; for (let r = 200; r <= 6000; r += 200) { const p = R.pd(0.01, r); ok(p <= prev + 1e-12, "radar Pd monotonic at " + r); prev = p; }
const r90 = rcs => { let lo = 10, hi = 20000; for (let i = 0; i < 60; i++) { const m = (lo + hi) / 2; if (R.pd(rcs, m) > 0.9) lo = m; else hi = m; } return lo; };
near(r90(0.02) / r90(0.0002), Math.pow(100, 0.25), 0.01, "range ∝ RCS^¼");
ok(R.pd(R.RCS.pigeon, 1500) < R.pd(R.RCS.phantom, 1500), "pigeon harder to see than Phantom");
ok(R.pd(0.01, 1500, { rainMmH: 25, preset: "echoguard" }) < R.pd(0.01, 1500, { preset: "echoguard" }), "rain hurts K-band");
ok(R.rainAttenDbPerKm(24.55, 10) > R.rainAttenDbPerKm(9.25, 10) * 5, "K-band rain attenuation ≫ X-band");
ok(R.pd(0.01, 10) === 0, "inside minimum range → 0");
ok(R.pd(0.01, 500, { elDeg: 80 }) === 0, "outside elevation coverage → 0");
// σ floors at datasheet accuracy, grows at low SNR
const sg = R.sigma(40), sgl = R.sigma(5);
near(sg.azDeg, 0.7, 1e-9, "az σ floor (IRIS 0.6–0.75°)"); near(sg.rangeM, 0.6, 1e-9, "range σ floor 0.6 m");
ok(sgl.azDeg >= sg.azDeg, "σ grows at low SNR");
// multipath: masking for low targets
ok(R.multipathLossDb(9.25, 28, 5, 1500) >= 4, "low target at grazing angle is masked");
// micro-Doppler: Phantom 4 blade flash 140 Hz, bandwidth ≈ 7–8 kHz at 9.85 GHz
const md = R.microDoppler({ type: "drone", model: "phantom" }, 9.85);
near(md.bladeFlashHz, 140, 1e-9, "Phantom blade flash"); ok(md.bandwidthHz > 6500 && md.bandwidthHz < 8500, "Phantom md bandwidth " + md.bandwidthHz.toFixed(0));
const mb = R.microDoppler({ type: "bird", kind: "gull" }, 9.85);
ok(mb.bandwidthHz > 150 && mb.bandwidthHz < 300, "bird wing md bandwidth within 150–300 Hz: " + mb.bandwidthHz.toFixed(0));
ok(R.microDopplerPc(40, 0.5, { type: "drone" }) > 0.93, "high SNR long dwell → ~0.95");
ok(R.microDopplerPc(0, 0.5, { type: "drone" }) < 0.6, "body SNR 0 dB → md near chance");
ok(R.microDopplerPc(40, 0.002, { type: "drone" }) < 0.6, "too short dwell → near chance");
{ const rng = seeded(3); let n = 0, right = 0; for (let i = 0; i < 4000; i++) { const l = R.microDopplerLLR(35, 0.3, { type: "drone" }, rng); n++; if (l > 0) right++; } near(right / n, R.microDopplerPc(35, 0.3, { type: "drone" }), 0.02, "LLR sign matches Pc"); }
{ const c = R.microDopplerColumn({ type: "drone" }, 0.0001, 5, 30, 128, 60, seeded(1)); ok(c.length === 128 && Math.max(...c) >= 29, "spectrogram has body line"); }

// ---- thermal: DSIAC calibration (Phantom 4 + Lepton over ground at night, limit ≈ 51 m)
const ph = { type: "drone", span: 0.35 };
const p50 = r => T.detect(ph, r, { camera: "lepton", background: "ground" }).pDetect;
ok(p50(30) > 0.7, "Lepton sees Phantom at 30 m: " + p50(30).toFixed(2));
ok(p50(51) > 0.25 && p50(51) < 0.75, "Lepton ≈ 50 % near 51 m: " + p50(51).toFixed(2));
ok(p50(100) < 0.1, "Lepton fails by 100 m: " + p50(100).toFixed(2));
{ let prev=1; for(let r=5;r<=300;r+=5){const p=p50(r); ok(p<=prev+1e-9,"thermal Pd monotonic at "+r); prev=p;} }
// against a clear sky the same drone is far easier (cold sky) and a C-UAS camera reaches much further
const sky = r => T.detect(ph, r, { camera: "uncooled640", background: "sky", sky: "clear", elevDeg: 10 }).pDetect;
ok(sky(1500) > 0.9, "uncooled 640 / 100 mm sees drone vs clear sky at 1.5 km: " + sky(1500).toFixed(2));
// fog: LWIR ≈ 4× visible at V = 610 m, ≈ equal at V ≤ 305 m (FLIR)
near(T.extinctionPerKm("visible", { visibilityKm: 0.61 }) / T.extinctionPerKm("lwir", { visibilityKm: 0.61 }), 4, 0.05, "LWIR 4× better at Cat II fog");
near(T.extinctionPerKm("visible", { visibilityKm: 0.3 }) / T.extinctionPerKm("lwir", { visibilityKm: 0.3 }), 1, 0.05, "no LWIR advantage at Cat IIIa");
near(T.transmission(1000, "visible", { visibilityKm: 1 }), 0.02, 0.001, "Koschmieder: 2 % contrast transmission at V");
ok(T.ttpf(4, 4) > 0.49 && T.ttpf(4, 4) < 0.51, "TTPF = 0.5 at N50");
// Johnson recognition range < detection range
ok(T.detect(ph, 800, { background: "sky", elevDeg: 10 }).pRecognize < T.detect(ph, 800, { background: "sky", elevDeg: 10 }).pDetect, "recognition harder than detection");

// ---- RF
ok(F.link("ocusync", "drone", 500).pd > 0.99, "OcuSync drone link obvious at 500 m");
let pr = 1; for (let r = 100; r <= 20000; r += 500) { const p = F.link("ocusync", "controller", r).pd; ok(p <= pr + 1e-12, "RF pd monotonic " + r); pr = p; }
ok(F.link("ocusync", "controller", 1200).bearingSigmaDeg < 6, "controller DF at 1.2 km (5.8 GHz) < 6°: " + F.link("ocusync", "controller", 1200).bearingSigmaDeg.toFixed(2));
near(F.pathLossDb(2400, 1), 40.05, 0.1, "FSPL at 1 m, 2.4 GHz ≈ 40 dB");
near(F.remoteIdPd(1524 * 0.8, "wifi"), 0.5, 1e-9, "RID Wi-Fi R50");
ok(F.remoteIdPd(300, "bt4") > 0.8 && F.remoteIdPd(800, "bt4") < 0.1, "BT legacy RID ~ few hundred m");
{ const c = F.spectrumColumn(0.123, [{ protocol: "wifi", rangeM: 300, id: "a" }, { protocol: "elrs24", side: "controller", rangeM: 800, id: "b" }], [2400, 2483.5], 128, seeded(2)); ok(c.length === 128, "spectrum column length"); }

// ---- acoustic
near(A.splAt(1, 83), 72, 0.01, "Lp at 1 m = Lw − 11");
near(A.splAt(2, 83) - A.splAt(4, 83), 6.02, 0.02, "−6 dB per doubling (plus tiny absorption)");
const r50 = opts => { let lo = 1, hi = 5000; for (let i = 0; i < 60; i++) { const m = (lo + hi) / 2; if (A.pd(m, opts) > 0.5) lo = m; else hi = m; } return lo; };
const single = r50({ mics: 1 });
ok(single > 110 && single < 190, "single mic quiet-site 50 % range ≈150 m: " + single.toFixed(0));
ok(r50({ mics: 8, windMps: 10 }) < r50({ mics: 8 }), "wind shortens acoustic range");
ok(r50({ mics: 8, urban: true }) < r50({ mics: 8 }), "urban background shortens range");
ok(A.bearingSigmaDeg(10) < A.bearingSigmaDeg(-10), "bearing σ worse at low SNR");
near(A.bladePassHz({ model: "phantom" }), 140, 1e-9, "Phantom blade-pass 140 Hz");
near(A.droneSoundPowerDbA(1800), 83 + 18.5 * Math.log10(2), 1e-9, "EU 945 C2 mass term");
{ const c = A.spectrogramColumn(1, [{ type: "drone", rangeM: 80, model: "phantom" }, { type: "bird", rangeM: 20 }], {}, 128, 2000, seeded(5)); const b = Math.round(140 / (2000 / 128)); ok(c[b] > 5, "blade-pass line visible at 80 m: " + c[b].toFixed(1)); }

console.log(`${pass} passed, ${fail} failed`);
// key numbers for the report
const fmt = x => x.toFixed(0);
console.log("IRIS 90% range: phantom", fmt(r90(R.RCS.phantom)), "m; pigeon", fmt(r90(R.RCS.pigeon)), "m; gull", fmt(r90(R.RCS.gull)), "m");
console.log("Acoustic 50% range, 8 mics: quiet", fmt(r50({ mics: 8 })), "m; wind 8 m/s", fmt(r50({ mics: 8, windMps: 8 })), "m; urban", fmt(r50({ mics: 8, urban: true })), "m");
let rr = 0; while (F.link("ocusync", "controller", rr + 100).pd > 0.5) rr += 100; console.log("OcuSync controller RF 50% range ≈", rr, "m");
let tr = 0; while (sky(tr + 50) > 0.5) tr += 50; console.log("Thermal (640/100 mm) vs clear sky 50% detect ≈", tr, "m; Lepton/ground ≈", (() => { let x = 0; while (p50(x + 1) > 0.5) x++; return x; })(), "m");
process.exit(fail ? 1 : 0);

