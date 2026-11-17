# Sensor models v2 (`sensors2.js`)

Physics-based detection models for the simulator's radar, thermal/visible camera, RF direction-finding and acoustic sensors. Defaults are calibrated to published specs and measurements where they exist. Anything without a primary number is marked **assumed** here and in the code comments.

Tests: `node sensors2.test.js` gives **175 passed, 0 failed**.

## Headline numbers the models now produce

| Quantity | Model output | Calibration anchor |
|---|---|---|
| Radar (IRIS-class X-band), 90 % Pd range, Phantom (0.01 m²) | 2,607 m | Pd = 0.9 at 3,100 m for 0.02 m² (IRIS C1 micro-class range; Mavic RCS −17 dBsm) |
| Radar 90 % Pd range, pigeon (0.0015 m²) / gull (0.006 m², assumed) | 1,622 m / 2,294 m | Pigeon −28.2 dBsm at X-band |
| Radar (EchoGuard-class K-band), Phantom 4 | Pd = 0.90 at 1,000 m | EchoGuard: "Phantom 4 > 1 km"; Phantom 4 ≈ −15 dBsm at K-band |
| Micro-Doppler, Phantom 4 at 9.85 GHz | 140 Hz blade flash, ~6.9 kHz bandwidth | Measured 140 Hz and 7.1–7.5 kHz |
| Micro-Doppler, birds | 150–300 Hz bandwidth, 3–5.5 Hz wingbeat | Measured 150–300 Hz bandwidth; wingbeat 4–6 Hz |
| Thermal, FLIR Lepton vs Phantom 4 over ground at night | 50 % Pd at ~53 m | DSIAC field test: detection limit 51 m |
| Thermal, 640×512 / 100 mm uncooled vs clear sky at 10° elevation | 50 % Pd at ~6.3 km | **Assumption-driven** and probably optimistic (see weaknesses) |
| Fog: LWIR vs visible extinction | 4× at V = 610 m; 1× at V ≤ 305 m | FLIR fog study, ICAO Cat II and Cat IIIa |
| RF, OcuSync-class controller on the ground (5.8 GHz) | 50 % Pd at ~2.1 km; DF σ ≈ 4.7° at 1.2 km | DJI O4 EIRP < 30–33 dBm (FCC); the rest assumed |
| Remote ID reception R50 | Wi-Fi beacon 1.22 km; BT legacy 400 m | Best measured ranges: 1,524 m (Wi-Fi), 500 m (BT) |
| Acoustic, single mic, quiet site | 50 % Pd at ~150 m | "No spectral evidence beyond 150 m" (field recordings) |
| Acoustic, 8-mic array | 427 m quiet; 146 m in 8 m/s wind; 45 m urban | Same anchor plus array gain (assumed) |

## Equations

**Radar**
- **SNR, scaled from the reference spec point:** `SNR(dB) = SNR_ref + 10·log10(σ/σ_ref) + 40·log10(R_ref/R) − L_rain − L_multipath`. `SNR_ref` is the single-look Swerling I SNR for Pd 0.9 at Pfa 1e-6, which is 130 (21.1 dB).
- **Pd, Swerling I/II:** `Pd = Pfa^(1/(1+SNR))`. Swerling 0 (North's approximation) is also available.
- **Measurement noise:** `σθ = max(spec, θ3dB / (1.6·√(2·SNR)))`, and the same form for range.
- **Rain:** ITU-R P.838 `γ = k·R^α` dB/km, applied over the two-way path.
- **Multipath:** two-ray `|1 + ρe^{iφ}|⁴` with ρ = −0.5, plus 10 dB masking below 0.5° grazing for targets under 15 m (both **assumed**).
- **Micro-Doppler:**
  - Rotor: tip speed `v = 2π·r·rps`, bandwidth `4v/λ`, blade flash `= blades × rps`.
  - Classifier accuracy: `Pc = 0.5 + 0.45·logistic((SNR−22.5+3)/2.5)·dwellFactor`. The 22.5 dB offset is the midpoint of the 20–25 dB by which props are weaker than the body.
  - The 0.95 cap derates lab accuracies of 94–98 % for the field (**assumed**).

**Thermal and visible**
- **Point-target SNR:** `ΔT·τ(R)·fill/√(NETD² + σclutter²)`, where `fill = min(1, px²)·EE(px)`. Threshold SNR 5 (**assumed**).
- **Resolved targets:** Johnson / NVESD TTPF with N50 = 1.0 (detection), 4.0 (recognition) and 6.4 (identification) cycles.
- **Sky:** clear-sky depression `35·(1−e^{−elev/30°})` K (**assumed**).
- **Transmission:**
  - Visible: Koschmieder `σ = 3.912/V`.
  - LWIR: visible extinction divided by a fog ratio interpolated from FLIR's anchors, with a 0.15 /km water-vapour floor.
  - Rain: FSO `1.076·R^0.67` dB/km.

**RF**
- **Path loss:** free space to a 100 m breakpoint, then `n = 2.7` for ground-level controllers (**assumed**).
- **SNR:** `Pr − (−174 + 10log B + NF) + 10 dB` processing gain (**assumed**).
- **Detection and bearing:** `Pd = logistic((SNR−10)/2)`; `σbearing = √(2² + (30/√SNR)²)` degrees (**assumed**).

**Acoustic**
- **Level at range:** ISO 9613 point source `Lp = Lw − 20log r − 11 − α·r`, with α from the 10 °C / 70 % RH octave table.
- **Source level:** `Lw` = 83 dB(A) for a 0.9 kg drone, at the EU C1 limit. Heavier drones add `18.5·log10(m/900)` (EU 2019/945 C2 formula).
- **Effective SNR:** `Lp − background + 10 dB` harmonic gain `+ 10·log10(Nmics)`, with Pd = 0.5 at 3 dB.
- **Bearing σ:** from the EchoHawk MUSIC table, multiplied by 3 for field conditions (**assumed**).

## Parameter table

| Parameter | Value | Source / status |
|---|---|---|
| IRIS band, update, accuracy | X-band 8.8–9.8 GHz; 1 s; range 0.6 m, az 0.6–0.75°, el 1°; elevation coverage 60° | [Robin IRIS datasheet](https://www.robinradar.com/hubfs/IRIS%20product%20sheet%20042026.pdf) |
| IRIS micro (C1) range | 3,100 m (column meaning ambiguous; treated as Pd 0.9) | Same; interpretation **assumed** |
| EchoGuard band, FOV, update, accuracy | 24.45–24.65 GHz; 120°×80°; 10 Hz; az < 1°, el < 1.5°; Phantom 4 > 1 km | [Echodyne EchoGuard sheet](https://www.echodyne.com/media/zixmcptm/ts-echoguard-echodyne_6ja.pdf) |
| Drone RCS | Phantom 3 ≈ −20 dBsm (10–90 %: −29.5 to −12) at 24 GHz in flight; Phantom 4 median 0.020 m² at 9 GHz, 18 dB aspect spread; Mavic Pro −17 dBsm; S900 −8 dBsm; plastic frames ≈ 7 dB lower | Rahman & Robertson, *IET RSN* 2019, plus chamber studies, via [review](https://migflug.com/afterburner/why-radar-cannot-detect-small-drones-rcs-micro-doppler-birds/) |
| Bird RCS | Pigeon −28.2 dBsm (X-band); raptors −22 to −19 dBsm | Same review |
| Props vs body | Props 20–25 dB weaker | Same review |
| Micro-Doppler classifier accuracy (lab) | 94.4 % (4-class), 98.3 % (drone / not drone) | Same review (CNN studies) |
| Rotor parameters | Phantom 4 70 rev/s, 140 Hz flash, 7.1–7.5 kHz bandwidth at 9.85 GHz; M300 100 rev/s, r = 26.6 cm; FIMI X8 80 rev/s, r = 10.5 cm; bird bandwidth 150–300 Hz | [SciELO JMOe micro-Doppler paper](https://www.scielo.br/j/jmoea/a/bKyX54S8ZjFSX3vYVkgDqdf/?lang=en) |
| Phantom prop radius 0.12 m; Mavic / mini / hexa rotor values | Derived or **assumed** | n/a |
| Rain attenuation k, α | X ≈ 0.0101 / 1.276; K ≈ 0.129 / 1.02 | ITU-R P.838-3 (approximate; **verify against the tables**) |
| Thermal field anchor | Lepton 80×60, 25° HFOV (5.45 mrad); Phantom 4 detection limit 51 m, Parrot 41 m, over ground at night; battery is the main heat source | [DSIAC sUAV thermal imaging](https://dsiac.dtic.mil/technical-inquiries/notable/thermal-imaging-of-small-unmanned-aerial-vehicles-suavs) |
| Drone ΔT +5 K night / +8 K day; bird +6 K; NETD 40–50 mK; sky depression; clutter σ | **Assumed** | n/a |
| Fog: LWIR vs visible | ~4× at Cat II (610 m); no advantage at Cat IIIa/c; rain drop-off at 100–500 m | [FLIR: can thermal see through fog and rain](https://www.flir.com/discover/rd-science/can-thermal-imaging-see-through-fog-and-rain/) |
| Johnson criteria | N50 = 1.0 / 4.0 / 6.4 cycles; TTPF exponent 2.7 + 0.7·N/N50 | Standard NVESD (textbook) |
| DJI O4 EIRP | 5.8 GHz FCC < 33 dBm (Pro) / < 30 dBm; CE < 14 dBm; 5.1 GHz CE < 23 dBm; 60 MHz max bandwidth | [DJI O4 Air Unit specs](https://www.dji.com/au/o4-air-unit/specs) |
| ELRS / Crossfire | 2.4 GHz ≤ 1,000 Hz packet rate, 900 MHz ≤ 200 Hz, Crossfire ≤ 150 Hz; 10 mW–1 W; community range tests up to 30 km at 900 MHz / 1 W | [Oscar Liang on ExpressLRS](https://oscarliang.com/expresslrs) |
| RF detector sensitivity, NF, processing gain, DF floor 2° | **Assumed**. Vendor pages (e.g. [DroneShield RfOne](https://www.droneshield.com/rfone-mk2)) don't publish them | n/a |
| Remote ID reception | Wi-Fi module 1,524 m; BT + WLAN 895 m; BT 500 m and 104 m (best per module) | [DroneXL module test](https://dronexl.co/2023/08/31/remote-id-module-complete-test/) |
| Drone sound power | EU C1 limit 85 → 83 → 81 dB(A); C2 formula `+18.5·log10(m/900)` | EU 2019/945, via [EU Drone Port summary](https://eudroneport.com/blog/sound-power-drones/) |
| BPF and background | M300 BPF ≈ 100 Hz with harmonics; broadband above ~500 Hz; site background LAeq ≈ 35 dB | [Forum Acusticum 2023 paper](https://dael.euracoustics.org/confs/fa2023/data/articles/000248.pdf) |
| Single-mic limit | No spectral evidence beyond 150 m; harmonics below 1.5 kHz | [Jeon et al., arXiv 1701.05779](https://arxiv.org/pdf/1701.05779) |
| Acoustic DOA accuracy | 8-mic UCA MUSIC RMS ≈ 0.3° (+10 dB), 0.9° (0 dB), 2.3° (−10 dB), synthetic | [EchoHawk, arXiv 2606.29589](https://arxiv.org/pdf/2606.29589) |
| Wind-noise law, urban background, harmonic gain, array gain | **Assumed** | n/a |

## Integration into `index.html`

All code below goes in `index.html`. Load the model with `<script src="sensors2.js"></script>` before the main script, or inline it. It defines `SENS2`. The sim's RNG and Gaussian helpers are `RNG` and `gauss`.

**1. Entity RCS (`newWorld` / `newBird`).** The current bird RCS values (0.008–0.02 m²) are too large; measured birds are smaller than drones. Use `SENS2.radar.RCS`: `rcs={gull:0.006, crow:0.003, kestrel:0.002}`. Drones stay at 0.01 (Phantom); a decoy or mini can use 0.005.

**2. `radarScan(t)`.** Replace the `r50` / `pd` / masking / noise block with:
```js
const look=SENS2.radar.look(e,RADAR,{preset:"iris",rainMmH:UI.wx==="rain"?10:0},RNG,gauss);
if(!look)continue;
meas.push(sph(R,look.range,look.az,look.el)); meas[meas.length-1].snr=look.snrDb; meas[meas.length-1].ent=e;
```
- Keep `RADAR.range` as an instrumented cap.
- Change the scan period from 0.5 s to `SENS2.radar.PRESETS.iris.updateS` (1 s), or select `"echoguard"` (0.1 s) for a K-band panel.
- **Clutter:** use `poisson(SENS2.radar.clutterRate(UI.clutter*0.8, rainMmH))`.
- `meas.ent` exists only to drive micro-Doppler truth. Fusion must not read it otherwise.

**3. Micro-Doppler evidence.** This is new and gives the biggest gain. In `updTrack`, keep each track's last associated measurement's `snr`. In `fusionTick`, for confirmed tracks whose truth is a drone or bird:
```js
tr.eMD = clamp((tr.eMD||0)*0.9 + 0.5*SENS2.radar.microDopplerLLR(tr.lastSnr, SENS2.radar.PRESETS.iris.updateS*0.05 /*dwell per scan on a rotating radar ≈ beam dwell, assumed*/, tr.truth, RNG), -4, 4);
```
Then add `tr.eMD` to the evidence sum `L` and show an "M" source chip.

Expected LLR per look:
- **Near zero** at body SNR < 20 dB: the micro-Doppler component is under the noise.
- **About ±1.5** at 25 dB.
- **About ±2.9** at 35 dB and up, the 0.95 cap.

Lower `0.5` if alerts become too eager. For a rotating radar with short dwell (≈50 ms per beam), dwell limits Pc. A staring K-band panel can dwell 0.2–0.5 s and gets near-full Pc.

**4. `rfScan(t)`.** Replace `dist<3500 && RNG()<0.92` and `gauss()*3` / `gauss()*4` with:
```js
const L=SENS2.rf.link("ocusync","drone",dist3(s,d.pos));       if(RNG()<L.pd){b=azOf(s,d.pos)+gauss()*L.bearingSigmaDeg;...}
const Lc=SENS2.rf.link("ocusync","controller",dist3(s,pp));    if(RNG()<Lc.pd){b=azOf(s,pp)+gauss()*Lc.bearingSigmaDeg;...}
```
- **Remote ID:** use `RNG()<SENS2.rf.remoteIdPd(dist,"wifi")` per sensor.
- **Waterfall (optional):** build a panel from `SENS2.rf.spectrumColumn(t, emitters, [2400,2483.5])` or `[5725,5850]`, one column per frame.

**5. `acScan(t)`.** Replace the linear `p` with:
```js
const o={mics:8,windMps:UI.wind,rain:UI.wx==="rain",massG:900};
p=SENS2.acoustic.pd(r,o); bearing σ = SENS2.acoustic.bearingSigmaDeg(SENS2.acoustic.snrDb(r,o))
```
For a spectrogram panel, use `SENS2.acoustic.spectrogramColumn(t, sourcesNearNode, o)`.

**6. `visibility(e)` and `classifyStep(t)`.** For the PTZ camera, replace the `sigmoid((px-4.5)/1.3)*visibility()` detection term with:
- **Thermal band:**
  ```js
  SENS2.thermal.detect(tgt, range, {camera:"uncooled640", background: el>2?"sky":"ground", sky: UI.wx==="fog"?"fog":UI.wx==="rain"?"overcast":"clear", elevDeg: el, day: ENV().day>0.5, weather:{fog:UI.wx==="fog", rainMmH: UI.wx==="rain"?10:0}}).pDetect
  ```
- **Visible band:** `SENS2.thermal.visibleDetect(tgt, range, {ifovRad: (c.fov*D2R)/CW, day: ENV().day>0.5, illum: <existing floodlight term>, weather:{...}}).pDetect`.
- **Recognition accuracy:** use `.pRecognize`, mapped as `acc = 0.5 + 0.48*pRecognize`, in place of the hand-made `acc` curve.
- **Fog:** keep the current fog overlay rendering, but take the detection numbers from `transmission()`.

## Honest weaknesses

- **Radar reference ranges:** vendors rarely state the Pd/Pfa behind a quoted range. Treating it as Pd 0.9 at Pfa 1e-6 is an assumption, and the IRIS sheet's two-column ranges are ambiguous.
- **Clear-sky thermal range is driven by assumptions:** the 6.3 km figure for a 100 mm uncooled camera rests on assumed sky depression, clutter σ and drone ΔT. Field reports usually give low single-digit km for small drones. Before quoting performance, derate sky depression or ΔT, or calibrate on your own footage. The ground-background case is anchored to DSIAC.
- **RF numbers are mostly assumed:** detection, processing gain and DF accuracy are assumed because vendors don't publish sensitivity or DF accuracy. Re-anchor them to the datasheet of the RF sensor you actually buy.
- **Acoustic background and wind:** levels and the wind-noise law are assumed. Urban detection range (45 m) shows that acoustics is a short-range confirmation sensor, which matches field experience.
- **Micro-Doppler is a probabilistic classifier, not a signal simulation:** `microDopplerColumn` is for visualization only. It is qualitatively right (body line, HERM lines, blade flashes; sinusoidal wing modulation) but is not an electromagnetic simulation.
- **ITU-R P.838 coefficients:** approximate table values; check them against the official tables.
