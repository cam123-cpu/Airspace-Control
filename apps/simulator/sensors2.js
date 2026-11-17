/* sensors2.js: physics-grounded sensor models for the Prison Airspace Sim.
 * Plain script (browser global SENS2, or module.exports in Node). Pure functions, no DOM.
 * Every constant carries its source, or "assumed" when no primary number was found.
 * Units: metres, seconds, degrees (angles in public API), dB / dBm / dBsm, kelvin differences.
 * Sources are listed in SENSOR_MODELS.md; tags in comments: [IRIS], [ECHO], [RR19], [RR18], [MDP], [CHAMBER],
 * [DSIAC], [FLIR-FOG], [KOSCH], [ITU838], [ITU-FSO], [DJI-O4], [ELRS], [RID-T], [EU945], [ISO9613], [EHAWK], [JOHNSON].
 */
(function (root) {
  "use strict";
  const DEG = Math.PI / 180;
  const C = 299792458;
  const db = x => 10 * Math.log10(x);
  const lin = d => Math.pow(10, d / 10);
  const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);
  const logistic = x => 1 / (1 + Math.exp(-x));

  // ------------------------------------------------------------------ RADAR
  // Two reference radars. Detection range is used as the Pd=0.9 point at the stated RCS (assumption: vendors quote
  // ranges near Pd 0.9 / Pfa 1e-6; most do not state Pd).
  const RADAR_PRESETS = {
    // Robin Radar IRIS (X-band, 360° rotating, 1 s update) [IRIS]: micro class (C1, 0.9 kg) 3100 m; range acc 0.6 m;
    // azimuth acc 0.6–0.75°; elevation acc 1°; elevation coverage 60°. RCS of a C1 (Mavic-class) ≈ -17 dBsm [CHAMBER].
    iris: { name: "X-band 360° surveillance (IRIS-class)", fGHz: 9.25, refRangeM: 3100, refRcsM2: 0.02, refPd: 0.9, pfa: 1e-6,
      updateS: 1.0, sigRangeFloorM: 0.6, sigAzFloorDeg: 0.7, sigElFloorDeg: 1.0, beamAzDeg: 3.0 /*assumed*/, beamElDeg: 10 /*assumed*/,
      elCoverDeg: [-5, 55] /* 60° coverage [IRIS], split assumed */, rangeResM: 3 /*assumed*/, minRangeM: 30 /*assumed*/ },
    // Echodyne EchoGuard (K-band 24.45–24.65 GHz, ESA panel 120°x80°, 10 Hz track update) [ECHO]: Phantom 4 >1 km;
    // track accuracy <1° az, <1.5° el. Phantom 4 RCS ≈ -15 dBsm at K-band [CHAMBER].
    echoguard: { name: "K-band ESA panel (EchoGuard-class)", fGHz: 24.55, refRangeM: 1000, refRcsM2: 0.032, refPd: 0.9, pfa: 1e-6,
      updateS: 0.1, sigRangeFloorM: 1.0 /*assumed, not stated*/, sigAzFloorDeg: 1.0, sigElFloorDeg: 1.5, beamAzDeg: 2.0 /*assumed*/,
      beamElDeg: 4 /*assumed*/, elCoverDeg: [-40, 40], azFovDeg: 120, rangeResM: 2 /*assumed*/, minRangeM: 10 /*assumed*/ }
  };
  // Typical in-flight / chamber RCS (linear m²), used as entity defaults when the sim has none.
  const RCS = {
    phantom: 0.01,   // Phantom 3 ≈ -20 dBsm at 24 GHz in flight, 10–90% -29.5…-12 dBsm [RR19]; Phantom 4 median 0.020 m² @9 GHz [CHAMBER]
    mavic: 0.02,     // Mavic Pro ≈ -17 dBsm Ku/K [CHAMBER]
    mini: 0.005,     // sub-250 g, plastic frames ≈7 dB below carbon [CHAMBER] → assumed
    hexa: 0.16,      // S900 hexacopter -8 dBsm @24 GHz [RR19]
    pigeon: 0.0015,  // pigeon -28.2 dBsm X-band [RR19-review]
    gull: 0.006,     // assumed between pigeon and large raptor (-22…-19 dBsm) [RR19]
    crow: 0.003,     // assumed
    kestrel: 0.002,  // assumed (small raptor)
    car: 10, aircraft: 10, bag: 0.0005, payload: 0.003
  };

  function swerlingIPd(snrLin, pfa) { return Math.pow(pfa, 1 / (1 + snrLin)); }           // Swerling I/II single look
  function swerlingISnrFor(pd, pfa) { return Math.log(pfa) / Math.log(pd) - 1; }           // inverse of the above
  function marcumApproxPd(snrLin, pfa) {                                                    // non-fluctuating (Swerling 0), Albersheim-style
    // North's approximation: Pd ≈ 0.5·erfc(√(-ln Pfa) − √(SNR + 0.5))
    return 0.5 * erfc(Math.sqrt(-Math.log(pfa)) - Math.sqrt(snrLin + 0.5));
  }
  function erfc(x) { // Numerical Recipes erfc, |rel err| < 1.2e-7
    const z = Math.abs(x), t = 1 / (1 + 0.5 * z);
    const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
    return x >= 0 ? r : 2 - r;
  }
  // ITU-R P.838-3 specific rain attenuation γ = k·R^α (dB/km), horizontal pol. Values at the two preset bands [ITU838] (approximate table values).
  const ITU838 = { 9.25: { k: 0.0101, a: 1.276 }, 24.55: { k: 0.1286, a: 1.0214 } }; // approximate; verify against P.838-3 tables before relying on them
  function rainAttenDbPerKm(fGHz, mmPerH) {
    if (!mmPerH) return 0;
    const key = Math.abs(fGHz - 9.25) < Math.abs(fGHz - 24.55) ? 9.25 : 24.55;
    const c = ITU838[key];
    return c.k * Math.pow(mmPerH, c.a);
  }
  // Two-ray multipath propagation factor F⁴ over ground (flat earth), reflection coefficient ρ (assumed −0.5 rough ground),
  // plus terrain/vegetation masking below 0.5° grazing for low targets (assumed 10 dB).
  function multipathLossDb(fGHz, hRadar, hTarget, rangeM) {
    const lam = C / (fGHz * 1e9), rho = 0.5;
    const phi = 4 * Math.PI * hRadar * Math.max(hTarget, 0.5) / (lam * Math.max(rangeM, 1));
    const re = 1 - rho * Math.cos(phi), im = rho * Math.sin(phi);           // 1 + ρe^{iφ} with ρ = −0.5
    const F2 = re * re + im * im;                                           // |F|²
    let L = -db(F2 * F2);                                                   // two-way: |F|⁴
    const grazing = Math.atan2(hTarget - hRadar, rangeM) / DEG;
    if (hTarget < 15 && grazing < 0.5) L += 10;                             // assumed masking by trees/terrain
    return clamp(L, -6, 30);                                                // lobing peaks capped at +6 dB gain (assumed)
  }
  /** Single-look SNR (dB) for a target of given RCS at range, scaled from the preset's reference point. */
  function radarSnrDb(rcsM2, rangeM, opts = {}) {
    const P = RADAR_PRESETS[opts.preset || "iris"];
    const snrRef = swerlingISnrFor(P.refPd, P.pfa);                         // ≈130 (21.1 dB) for Pd .9 / Pfa 1e-6
    let s = db(snrRef) + db(Math.max(rcsM2, 1e-7) / P.refRcsM2) + 40 * Math.log10(P.refRangeM / Math.max(rangeM, 1));
    if (opts.rainMmH) s -= 2 * rainAttenDbPerKm(P.fGHz, opts.rainMmH) * rangeM / 1000;
    if (opts.fogExtraDb) s -= opts.fogExtraDb;                             // fog is ~transparent at X/K band (assumed ≈0)
    if (opts.hRadar != null && opts.hTarget != null) s -= multipathLossDb(P.fGHz, opts.hRadar, opts.hTarget, rangeM);
    return s;
  }
  /** Probability of detection on one look (Swerling I by default, rcs fluctuates scan to scan per [RR19]/[NATO-RCS]). */
  function radarPd(rcsM2, rangeM, opts = {}) {
    const P = RADAR_PRESETS[opts.preset || "iris"];
    if (rangeM < P.minRangeM) return 0;
    const el = opts.elDeg;
    if (el != null && (el < P.elCoverDeg[0] || el > P.elCoverDeg[1])) return 0;
    const snr = lin(radarSnrDb(rcsM2, rangeM, opts));
    return opts.swerling0 ? marcumApproxPd(snr, P.pfa) : swerlingIPd(snr, P.pfa);
  }
  /** Measurement σ vs SNR: CRLB-like θ3dB/(1.6·√(2·SNR)) (Skolnik), floored at the datasheet accuracy. */
  function radarSigma(snrDb, opts = {}) {
    const P = RADAR_PRESETS[opts.preset || "iris"], s = Math.max(lin(snrDb), 1);
    return {
      rangeM: Math.max(P.sigRangeFloorM, P.rangeResM / Math.sqrt(2 * s)),
      azDeg: Math.max(P.sigAzFloorDeg, P.beamAzDeg / (1.6 * Math.sqrt(2 * s))),
      elDeg: Math.max(P.sigElFloorDeg, P.beamElDeg / (1.6 * Math.sqrt(2 * s)))
    };
  }
  /** Expected rain/insect volume-clutter false plots per scan (assumed linear in rain rate). */
  function radarClutterRate(baseRate, rainMmH) { return baseRate + 0.35 * (rainMmH || 0); }
  /** One stochastic look: returns null (miss) or {range, az, el, snrDb} with noise. rng() uniform, gauss() normal. */
  function radarLook(target, radar, opts, rng, gauss) {
    const dx = target.pos[0] - radar.x, dy = target.pos[1] - radar.y, dz = target.pos[2] - radar.z;
    const r = Math.hypot(dx, dy, dz), el = Math.asin(dz / Math.max(r, 1e-6)) / DEG, az = (Math.atan2(dx, dy) / DEG + 360) % 360;
    // RCS fluctuation: Swerling I (exponential) handled inside Pd; aspect-dependent mean ±6 dB (assumed from 18 dB chamber spread [CHAMBER])
    const o = Object.assign({}, opts, { elDeg: el, hRadar: radar.z, hTarget: target.pos[2] });
    const snr = radarSnrDb(target.rcs, r, o);
    if (rng() > radarPd(target.rcs, r, o)) return null;
    const sg = radarSigma(snr, o);
    return { range: r + gauss() * sg.rangeM, az: az + gauss() * sg.azDeg, el: el + gauss() * sg.elDeg, snrDb: snr };
  }

  // ---- micro-Doppler
  // Rotor parameters: Phantom 4 ≈70 rev/s, 2 blades → 140 Hz blade flash measured; M300 100 rev/s, r=0.266 m; FIMI X8 80 rev/s, r=0.105 m [MDP].
  // Bird wingbeat 4–6 Hz typical small/medium birds [RR18]; gull ≈3 Hz (assumed); wing micro-Doppler bandwidth 150–300 Hz at 9.85 GHz [MDP].
  const ROTOR = {
    phantom: { rps: 70, blades: 2, radiusM: 0.12 },   // 9.4" props → r ≈ 0.12 m (assumed from prop size)
    mavic: { rps: 95, blades: 2, radiusM: 0.105 },    // assumed (smaller, faster props)
    mini: { rps: 130, blades: 2, radiusM: 0.06 },     // assumed
    hexa: { rps: 60, blades: 2, radiusM: 0.19 }       // assumed
  };
  // tipMps = RCS-weighted effective wing velocity, set so total bandwidth 4v/λ falls in the measured 150–300 Hz at 9.85 GHz [MDP]
  const WING = { gull: { hz: 3.0, tipMps: 2.1 }, crow: { hz: 4.2, tipMps: 1.8 }, kestrel: { hz: 5.5, tipMps: 1.4 }, pigeon: { hz: 5.5, tipMps: 1.6 } };
  /** Micro-Doppler descriptor for an entity. ent: {type:'drone'|'bird', model?, kind?, rotor?} */
  function microDoppler(ent, fGHz = 9.25) {
    const lam = C / (fGHz * 1e9);
    if (ent.type === "drone") {
      const R = ent.rotor || ROTOR[ent.model || "phantom"];
      const vTip = 2 * Math.PI * R.radiusM * R.rps;
      return { kind: "rotor", bladeFlashHz: R.blades * R.rps, tipSpeedMps: vTip, dopplerSpreadMps: vTip, bandwidthHz: 4 * vTip / lam, wingbeatHz: 0,
        featureVector: [R.blades * R.rps, vTip, 0, 1] };
    }
    if (ent.type === "bird") {
      const W = WING[ent.kind] || WING.pigeon;
      return { kind: "wing", bladeFlashHz: 0, tipSpeedMps: W.tipMps, dopplerSpreadMps: W.tipMps, bandwidthHz: 4 * W.tipMps / lam, wingbeatHz: W.hz,
        featureVector: [0, W.tipMps, W.hz, 0] };
    }
    return { kind: "none", bladeFlashHz: 0, tipSpeedMps: 0, dopplerSpreadMps: 0, bandwidthHz: 0, wingbeatHz: 0, featureVector: [0, 0, 0, 0] };
  }
  /** Probability that the micro-Doppler classifier labels the target correctly.
   *  Props are 20–25 dB weaker than the body [RR19-review]; lab CNN accuracy 94.4% (4-class) / 98.3% (drone vs not) [RR18-review].
   *  Needs ≥ ~2 blade-flash periods (drone) or ≥ ~1 wingbeat (bird) of dwell. Field derating to 0.95 max (assumed). */
  function microDopplerPc(snrDb, dwellS, ent) {
    const mdSnr = snrDb - 22.5;                                                 // micro-Doppler component SNR
    const md = microDoppler(ent);
    const need = md.kind === "rotor" ? 2 / md.bladeFlashHz : md.kind === "wing" ? 1 / md.wingbeatHz : 1;
    const dwellF = clamp(dwellS / need, 0, 1) * (1 - Math.exp(-3 * dwellS / need));
    const snrF = logistic((mdSnr + 3) / 2.5);                                   // 50% of the way at −3 dB md-SNR (assumed)
    return 0.5 + 0.45 * snrF * dwellF;                                          // 0.5 = coin flip, max 0.95
  }
  /** Log-likelihood ratio (drone vs bird) the fusion should add for one micro-Doppler look.
   *  Draws the classifier output with probability pc of being right; returns signed LLR. */
  function microDopplerLLR(snrDb, dwellS, ent, rng) {
    // Hypothesis test on the rotor blade-flash signature: a rotating radar resolves blade flash within one beam dwell,
    // so at high SNR its ABSENCE is evidence too. Detection reliability is set by the rotor hypothesis for any target.
    const pc = microDopplerPc(snrDb, dwellS, { type: "drone" });
    const truthDrone = ent.type === "drone";
    const saysDrone = rng() < pc ? truthDrone : !truthDrone;
    const llr = Math.log(pc / (1 - pc));
    return saysDrone ? llr : -llr;
  }
  /** One column of a micro-Doppler spectrogram (radial velocity axis, ±vMax), values in dB re noise.
   *  t: time (s); vRadial: bulk radial velocity (m/s); snrDb: body SNR. Deterministic given (t, rngSeedNoise). */
  function microDopplerColumn(ent, t, vRadial, snrDb, nBins = 128, vMax = 60, rng = Math.random) {
    const col = new Float32Array(nBins), md = microDoppler(ent);
    const bin = v => Math.round((v + vMax) / (2 * vMax) * (nBins - 1));
    for (let i = 0; i < nBins; i++) col[i] = -3 + 3 * (rng() - 0.5);          // noise floor ≈ 0 dB with ±1.5 dB grain
    const put = (v, dB) => { const b = bin(v); if (b >= 0 && b < nBins) col[b] = Math.max(col[b], dB); };
    if (snrDb < -5) return col;
    put(vRadial, snrDb); put(vRadial - 0.5, snrDb - 6); put(vRadial + 0.5, snrDb - 6);
    if (md.kind === "rotor") {
      const flashPeriod = 1 / md.bladeFlashHz, phase = (t % flashPeriod) / flashPeriod;
      const mdDb = snrDb - 22.5;
      // HERM lines (continuous, faint) at body ± k·(rotation-rate) spacing in velocity terms
      const herm = md.tipSpeedMps / 6;
      for (let k = 1; k <= 6; k++) { put(vRadial + k * herm, mdDb - 4 - k); put(vRadial - k * herm, mdDb - 4 - k); }
      // blade flash: a vertical streak across ±tip speed lasting ~5% of the flash period
      if (phase < 0.06) for (let v = -md.tipSpeedMps; v <= md.tipSpeedMps; v += (2 * vMax) / nBins) put(vRadial + v, mdDb + 3);
    } else if (md.kind === "wing") {
      const s = Math.sin(2 * Math.PI * md.wingbeatHz * t), mdDb = snrDb - 12;    // wings are a large fraction of bird RCS (assumed)
      for (let f = 0; f <= 1; f += 0.25) put(vRadial + s * md.tipSpeedMps * f, mdDb - 6 * f);
    }
    return col;
  }

  // ------------------------------------------------------------------ THERMAL / VISIBLE
  const CAMERAS = {
    // FLIR Lepton (DSIAC field test): 80×60, 25° HFOV → 0.3125°/px = 5.45 mrad; NETD 50 mK (Lepton class, assumed) [DSIAC]
    lepton: { ifovRad: 0.3125 * DEG, netdK: 0.05, ee: 0.45 },
    // C-UAS uncooled LWIR PTZ: 640×512, 12 µm pitch, 100 mm lens → 0.12 mrad; NETD 40 mK (typical uncooled VOx, assumed)
    uncooled640: { ifovRad: 12e-6 / 0.1, netdK: 0.04, ee: 0.45 },
    // Cooled MWIR long-range: 1280×1024, 10 µm, 300 mm → 0.033 mrad; NETD 20 mK (assumed)
    cooledMwir: { ifovRad: 10e-6 / 0.3, netdK: 0.02, ee: 0.5 }
  };
  // Apparent temperature contrast of the target against its background (K).
  // Battery is the main heat source; motors are air-cooled [DSIAC]. Body +5 K night / +8 K sunlit (assumed).
  // Clear-sky LWIR brightness temperature well below ambient at elevation (assumed Δzenith 35 K clear, 4 K overcast, 1 K fog).
  function skyDepressionK(elevDeg, sky) {
    const dz = sky === "clear" ? 35 : sky === "overcast" ? 4 : sky === "fog" ? 1 : 15;
    return dz * (1 - Math.exp(-Math.max(elevDeg, 0) / 30)); // slow rise: low-elevation paths cross humid near-ground air (assumed)
  }
  function apparentDeltaT(ent, opts = {}) {
    const day = opts.day ? 1 : 0;
    let dT = ent.type === "drone" ? 5 + 3 * day : ent.type === "bird" ? 6 : ent.type === "aircraft" ? 30 : ent.type === "payload" ? 0.5 : ent.type === "bag" ? 0.3 : 2;
    if (ent.type === "drone" && ent.phase === "wait") dT = 1;              // powered but parked (assumed)
    if (opts.background === "ground") return dT;                           // ground ≈ ambient (assumed)
    return dT + skyDepressionK(opts.elevDeg || 0, opts.sky || "clear");
  }
  /** Background clutter σ (K) to add in quadrature with NETD (assumed): sky is smooth, ground is cluttered. */
  function clutterSigmaK(bg, sky) { return bg === "ground" ? 0.5 : sky === "clear" ? 0.1 : 0.2; } // cloud structure + scintillation (assumed)
  // Atmospheric extinction (1/km). Visible: Koschmieder σ = 3.912/V [KOSCH]. LWIR: clear-air water vapour 0.15/km (assumed humid);
  // in fog LWIR outranges visible ≈4× at Cat II (V≈610 m) and ≈1× at Cat IIIa/c (V≤305 m) [FLIR-FOG] → ratio interpolated in log V.
  // Rain (both bands): FSO empirical 1.076·R^0.67 dB/km [ITU-FSO].
  function extinctionPerKm(band, weather = {}) {
    const V = weather.visibilityKm != null ? weather.visibilityKm : weather.fog ? 0.6 : 20;
    const sVis = 3.912 / V;
    let s;
    if (band === "visible") s = sVis;
    else {
      let ratio;
      if (V <= 0.305) ratio = 1; else if (V >= 2) ratio = 6; else if (V <= 0.61) ratio = 1 + 3 * Math.log(V / 0.305) / Math.log(2); else ratio = 4 + 2 * Math.log(V / 0.61) / Math.log(2 / 0.61);
      s = Math.max(0.15, sVis / ratio);
    }
    if (weather.rainMmH) s += (1.076 * Math.pow(weather.rainMmH, 0.67)) / 4.343;
    return s;
  }
  function transmission(rangeM, band, weather) { return Math.exp(-extinctionPerKm(band, weather) * rangeM / 1000); }
  function pixelsOnTarget(sizeM, rangeM, ifovRad) { return sizeM / (Math.max(rangeM, 1) * ifovRad); }
  // Johnson / NVESD TTPF: P(N) = (N/N50)^E / (1 + (N/N50)^E), E = 2.7 + 0.7·N/N50. N50: detection 1.0, recognition 4.0, identification 6.4 cycles [JOHNSON].
  function ttpf(cycles, n50) { const x = cycles / n50, E = 2.7 + 0.7 * x; const p = Math.pow(x, E); return p / (1 + p); }
  /** Thermal detection: point-target SNR (works for sub-pixel hot targets) OR resolved-target Johnson detection, whichever is higher.
   *  Calibrated so a Phantom 4 over a ground background with a Lepton reaches Pd≈0.5 near ~50 m (DSIAC field test: 51 m). */
  function thermalDetect(ent, rangeM, opts = {}) {
    const cam = CAMERAS[opts.camera || "uncooled640"];
    const size = ent.span || 0.35, px = pixelsOnTarget(size, rangeM, cam.ifovRad);
    const dT = apparentDeltaT(ent, opts) * transmission(rangeM, "lwir", opts.weather || {});
    const fill = Math.min(1, px * px) * (cam.ee + (1 - cam.ee) * clamp((px - 1) / 2, 0, 1)); // ensquared energy ramps from EE(1 px) to 1 at 3 px
    const sigma = Math.hypot(cam.netdK, clutterSigmaK(opts.background || "sky", opts.sky || "clear"));
    const snr = dT * fill / sigma;
    const pPoint = logistic((snr - 5) / 1.2);                                   // threshold SNR 5 (assumed CFAR-like)
    const cycles = px / 2, contrastOK = logistic((dT / sigma - 3) / 1);
    const pDetect = Math.max(pPoint, ttpf(cycles, 1.0) * contrastOK);
    const pRecognize = ttpf(cycles, 4.0) * contrastOK;
    const pIdentify = ttpf(cycles, 6.4) * contrastOK;
    return { pixels: px, deltaTK: dT, snr, pDetect, pRecognize, pIdentify };
  }
  /** Visible-band detection at night/day: contrast C0 attenuated by Koschmieder, threshold contrast 0.02 (assumed) × size factor. */
  function visibleDetect(ent, rangeM, opts = {}) {
    const ifov = opts.ifovRad || 0.05e-3, px = pixelsOnTarget(ent.span || 0.35, rangeM, ifov);
    const C0 = opts.day ? 0.6 : (opts.illum != null ? 0.05 + 0.5 * opts.illum : 0.08);   // silhouette vs sky by day; floodlit vs dark at night (assumed)
    const Cr = C0 * transmission(rangeM, "visible", opts.weather || {});
    const fill = Math.min(1, px * px) * (0.45 + 0.55 * clamp((px - 1) / 2, 0, 1));
    const snr = Cr * fill / (opts.day ? 0.004 : 0.012);                         // sensor noise in contrast units (assumed)
    return { pixels: px, contrast: Cr, pDetect: Math.max(logistic((snr - 6) / 1.2), ttpf(px / 2, 1.0) * logistic((Cr / 0.02 - 1) * 3)), pRecognize: ttpf(px / 2, 4.0) * logistic((Cr / 0.02 - 1) * 3) };
  }

  // ------------------------------------------------------------------ RF
  // Control/video link table. EIRP: DJI O4 5.8 GHz FCC <33 dBm (Pro) / <30 dBm, CE <14 dBm; 5.1 GHz CE <23 dBm [DJI-O4].
  // ELRS 2.4 GHz / 900 MHz up to 1 W, packet rate ≤1000 Hz (2.4) / ≤200 Hz (900); Crossfire ≤150 Hz [ELRS].
  const RF_PROTOCOLS = {
    ocusync: { name: "OcuSync/O4-class video+control", bandsMHz: [[5725, 5850], [2400, 2483.5]], droneEirpDbm: 30, ctrlEirpDbm: 26 /*assumed*/, bwMHz: 20, hopsPerS: 100 /*assumed*/, duty: 0.9 },
    wifi: { name: "Wi-Fi drone (2.4/5 GHz)", bandsMHz: [[2400, 2483.5]], droneEirpDbm: 20 /*assumed*/, ctrlEirpDbm: 18, bwMHz: 20, hopsPerS: 0, duty: 0.5 },
    elrs24: { name: "ExpressLRS 2.4 GHz", bandsMHz: [[2400, 2480]], droneEirpDbm: 17 /*telemetry, assumed*/, ctrlEirpDbm: 27 /*500 mW typical, ≤30 dBm [ELRS]*/, bwMHz: 0.8, hopsPerS: 500, duty: 0.2 },
    elrs900: { name: "ExpressLRS / Crossfire 900 MHz", bandsMHz: [[902, 928]], droneEirpDbm: 17, ctrlEirpDbm: 27, bwMHz: 0.5, hopsPerS: 100, duty: 0.2 },
    rid: { name: "Remote ID broadcast", bandsMHz: [[2402, 2480]], droneEirpDbm: 10 /*assumed BT/Wi-Fi beacon*/, ctrlEirpDbm: -99, bwMHz: 2, hopsPerS: 3, duty: 0.02 }
  };
  // Log-distance path loss PL = FSPL(1 m) + 10·n·log10(d). n = 2.0 air-to-sensor LOS; 2.7 for a ground controller (assumed).
  // Two-slope: free space to a 100 m breakpoint, then n = 2.0 (airborne, LOS) or 2.7 (ground-level controller) beyond (assumed).
  function pathLossDb(freqMHz, rangeM, groundLevel = false) {
    const fspl = d => 20 * Math.log10(freqMHz) - 27.55 + 20 * Math.log10(Math.max(d, 1)), bp = 100;
    if (!groundLevel || rangeM <= bp) return fspl(rangeM);
    return fspl(bp) + 27 * Math.log10(rangeM / bp);
  }
  function receivedPowerDbm(eirpDbm, freqMHz, rangeM, groundLevel = false, rxGainDbi = 3 /*assumed*/) { return eirpDbm + rxGainDbi - pathLossDb(freqMHz, rangeM, groundLevel); }
  function rfNoiseDbm(bwMHz, nfDb = 6 /*assumed*/) { return -174 + 10 * Math.log10(bwMHz * 1e6) + nfDb; }
  /** Detection probability of an emitter: SNR threshold 10 dB for protocol-signature detection (assumed), 2 dB slope. */
  function rfPd(snrDb) { return logistic((snrDb - 10) / 2); }
  /** DF bearing σ (deg): floor 2° (vendor class, assumed) ⊕ 30°/√SNR (assumed amplitude/phase DF noise). */
  function rfBearingSigmaDeg(snrDb) { return Math.hypot(2, 30 / Math.sqrt(Math.max(lin(snrDb), 1))); }
  function rfLink(protocol, side, rangeM) {
    const P = RF_PROTOCOLS[protocol], f = (P.bandsMHz[0][0] + P.bandsMHz[0][1]) / 2;
    const eirp = side === "controller" ? P.ctrlEirpDbm : P.droneEirpDbm;
    const pr = receivedPowerDbm(eirp, f, rangeM, side === "controller");
    const snr = pr - rfNoiseDbm(P.bwMHz) + 10 /* protocol-signature processing gain over many bursts (assumed) */ + db(Math.max(P.duty, 0.01)) * 0.5;  // duty-cycle penalty (assumed half in dB)
    return { rxDbm: pr, snrDb: snr, pd: rfPd(snr), bearingSigmaDeg: rfBearingSigmaDeg(snr) };
  }
  // Remote ID reception: best measured ranges: Wi-Fi beacon module 1524 m, BT+WLAN 895 m, BT 500 m / 104 m [RID-T].
  // Model: logistic in range with R50 = 0.8× best measured (assumed), 15% width.
  function remoteIdPd(rangeM, tech = "wifi") { const R50 = (tech === "wifi" ? 1524 : tech === "bt5" ? 895 : 500) * 0.8; return logistic((R50 - rangeM) / (0.15 * R50)); }
  /** One waterfall column: power (dBm per bin) across a band at time t for a list of emitters
   *  emitters: [{protocol, rangeM, side:'drone'|'controller', id}]; band: [fLoMHz, fHiMHz]. */
  function rfSpectrumColumn(t, emitters, band = [2400, 2483.5], nBins = 128, rng = Math.random) {
    const [lo, hi] = band, df = (hi - lo) / nBins, col = new Float32Array(nBins);
    const nf = rfNoiseDbm(df);
    for (let i = 0; i < nBins; i++) col[i] = nf + 2 * (rng() - 0.5);
    const add = (fc, bw, pDbm) => { const a = Math.floor((fc - bw / 2 - lo) / df), b = Math.ceil((fc + bw / 2 - lo) / df); const per = pDbm - db(Math.max(1, b - a)); for (let i = Math.max(0, a); i < Math.min(nBins, b); i++) col[i] = Math.max(col[i], per + (rng() - 0.5)); };
    for (const e of emitters) {
      const P = RF_PROTOCOLS[e.protocol]; if (!P) continue;
      const B = P.bandsMHz.find(b => b[0] < hi && b[1] > lo); if (!B) continue;
      const fMid = (B[0] + B[1]) / 2, l = rfLink(e.protocol, e.side || "drone", e.rangeM);
      const seed = hash(e.id || e.protocol);
      if (P.hopsPerS > 0) {
        const slot = Math.floor(t * P.hopsPerS), u = frac(Math.sin((slot + seed) * 12.9898) * 43758.5453);
        if (frac(Math.sin((slot + seed) * 78.233) * 9631.7) < P.duty) add(B[0] + P.bwMHz / 2 + u * (B[1] - B[0] - P.bwMHz), P.bwMHz, l.rxDbm);
      } else if (frac(Math.sin((Math.floor(t * 200) + seed) * 3.17) * 1e4) < P.duty) add(fMid, P.bwMHz, l.rxDbm);
    }
    return col;
  }
  const frac = x => x - Math.floor(x);
  function hash(s) { let h = 0; for (const ch of String(s)) h = (h * 31 + ch.charCodeAt(0)) | 0; return (h >>> 0) % 997; }

  // ------------------------------------------------------------------ ACOUSTIC
  // Sound power: EU 2019/945 C1 limit LWA 85 → 83 → 81 dB(A); C2 adds 18.5·log10(m/900 g) [EU945]. Default 83 dB(A) for a 0.9 kg class (assumed at limit).
  // Point source: Lp = Lw − 20·log10(r) − 11 − α·r [ISO9613]. α (dB/km) per octave at 10 °C, 70 % RH (a cool night) [ISO9613].
  const ALPHA_DBKM = { 63: 0.1, 125: 0.4, 250: 1.0, 500: 1.9, 1000: 3.7, 2000: 9.7, 4000: 32.8, 8000: 117 };
  function droneSoundPowerDbA(massG = 900) { return massG <= 900 ? 83 : 83 + 18.5 * Math.log10(Math.min(massG, 4000) / 900); }
  function splAt(rangeM, lwDbA = 83, bandHz = 250) { const a = ALPHA_DBKM[bandHz] != null ? ALPHA_DBKM[bandHz] : 1.9; return lwDbA - 20 * Math.log10(Math.max(rangeM, 1)) - 11 - a * rangeM / 1000; }
  // Background: rural night ≈ 35 dB(A) LAeq measured at a flight test site [FA2023]; urban day ≈ 55 (assumed). Wind noise on a
  // wind-screened mic 40 + 20·log10(v/5) dB(A) for v > 2 m/s (assumed). Rain adds 10 dB (assumed).
  function backgroundDbA(opts = {}) {
    const base = opts.urban ? 55 : 35, v = opts.windMps || 0;
    const wind = v > 2 ? 40 + 20 * Math.log10(v / 5) : 0, rain = opts.rain ? 45 : 0;
    return 10 * Math.log10(lin(base) + (wind ? lin(wind) : 0) + (rain ? lin(rain) : 0));
  }
  // Effective SNR = Lp − background + harmonic (narrowband) gain 10 dB (assumed) + array gain 10·log10(N mics).
  // Calibration: single mic, quiet urban/rural, ≈150 m limit (no spectral evidence beyond 150 m) [EMPIRICAL-DNN]; threshold set to SNR_eff 3 dB.
  function acousticSnrDb(rangeM, opts = {}) {
    const Lp = splAt(rangeM, opts.lwDbA != null ? opts.lwDbA : droneSoundPowerDbA(opts.massG));
    return Lp - backgroundDbA(opts) + 10 + 10 * Math.log10(opts.mics || 8);
  }
  function acousticPd(rangeM, opts = {}) { return logistic((acousticSnrDb(rangeM, opts) - 3) / 1.5); }
  // Bearing σ: synthetic 8-mic UCA MUSIC ≈0.3° @+10 dB, ≈0.9° @0 dB, ≈2.3° @−10 dB [EHAWK]; ×3 field derating (assumed) + 1° floor.
  function acousticBearingSigmaDeg(snrDb) { const s = snrDb >= 10 ? 0.3 : snrDb >= 0 ? 0.9 - 0.06 * snrDb : 0.9 + 0.14 * (-snrDb); return Math.hypot(1, 3 * Math.min(s, 6)); }
  /** Blade-pass frequency f0 = blades × rev/s (Phantom ≈140 Hz [MDP]; M300 BPF ≈100 Hz [FA2023]). */
  function bladePassHz(ent) { const R = (ent && ent.rotor) || ROTOR[(ent && ent.model) || "phantom"]; return R.blades * R.rps; }
  /** One spectrogram column 0..fMax Hz (dB re background) at a node, summing drone sources (birds are acoustically silent here). */
  function acousticSpectrogramColumn(t, sources, opts = {}, nBins = 128, fMax = 2000, rng = Math.random) {
    const col = new Float32Array(nBins), df = fMax / nBins;
    for (let i = 0; i < nBins; i++) col[i] = 2 * (rng() - 0.5) + (i < 6 ? 6 : 0);  // low-frequency wind/traffic rumble (assumed)
    for (const s of sources) {
      if (s.type !== "drone") continue;
      const snr = acousticSnrDb(s.rangeM, opts) - 10 * Math.log10(opts.mics || 8);   // per-mic picture
      const f0 = bladePassHz(s) * (1 + 0.01 * Math.sin(t * 0.7 + (s.id || 0)));      // throttle wander ±1 % (assumed)
      for (let k = 1; k * f0 < fMax; k++) {
        const lvl = snr - 3 * (k - 1);                                                 // −3 dB per harmonic (assumed)
        for (let r = 0; r < 4; r++) {                                                   // four rotors at slightly different rpm → beating
          const f = k * f0 * (1 + (r - 1.5) * 0.004), b = Math.round(f / df);
          if (b >= 0 && b < nBins) col[b] = Math.max(col[b], lvl - 6 + 3 * Math.sin(2 * Math.PI * 1.3 * t + r));
        }
      }
      for (let i = Math.round(500 / df); i < nBins; i++) col[i] = Math.max(col[i], snr - 18 + 2 * (rng() - 0.5)); // broadband above ~500 Hz [FA2023]
    }
    return col;
  }

  const SENS2 = {
    version: "1.0.0",
    radar: { PRESETS: RADAR_PRESETS, RCS, snrDb: radarSnrDb, pd: radarPd, sigma: radarSigma, look: radarLook, swerlingIPd, swerlingISnrFor, marcumApproxPd,
      rainAttenDbPerKm, multipathLossDb, clutterRate: radarClutterRate, microDoppler, microDopplerPc, microDopplerLLR, microDopplerColumn, ROTOR, WING },
    thermal: { CAMERAS, apparentDeltaT, skyDepressionK, clutterSigmaK, extinctionPerKm, transmission, pixelsOnTarget, ttpf, detect: thermalDetect, visibleDetect },
    rf: { PROTOCOLS: RF_PROTOCOLS, pathLossDb, receivedPowerDbm, noiseDbm: rfNoiseDbm, pd: rfPd, bearingSigmaDeg: rfBearingSigmaDeg, link: rfLink, remoteIdPd, spectrumColumn: rfSpectrumColumn },
    acoustic: { ALPHA_DBKM, droneSoundPowerDbA, splAt, backgroundDbA, snrDb: acousticSnrDb, pd: acousticPd, bearingSigmaDeg: acousticBearingSigmaDeg, bladePassHz, spectrogramColumn: acousticSpectrogramColumn },
    util: { db, lin, logistic, erfc }
  };
  if (typeof module !== "undefined" && module.exports) module.exports = SENS2; else root.SENS2 = SENS2;
})(typeof window !== "undefined" ? window : globalThis);
