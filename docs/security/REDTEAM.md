# Red-team report

18 attack scenarios × 20 seeds each, plus 20 attack-free control nights (~34 bird / aircraft / bag / truck events per night). Alert threshold 0.9. Every fusion generation faces the same seeds, sensors and stand-in camera classifier. Reproduce with `python -m redteam.run_redteam` (v1 vs H1) and `python -m redteam.compare_v3 --seeds 20 --nights 20` (H2, H3).

| Generation | What changed |
|---|---|
| **v1** | Original fusion (`hardened=False`) |
| **H1** | Hardening pass: Remote ID verified against RF direction-finding (DF) and radar; Remote ID alone never confirms a track; emitter-level RF; sensor-health monitoring; camera cueing under saturation; formation and dive cues; P-1, P-2 and P-3 security patches; per-sensor circular-mean triangulation (F-15) |
| **H2** | H1 plus radar micro-Doppler (blade flash present or absent); RF evidence never attached to a track that radar or camera calls a bird; ground-vehicle class road-gated, recomputed every second and overridden by drone signatures |
| **H3** | H2 plus a remote RF DF mast about 1.1 km from the fence corners (`rf_remote_mast=True`) |

## Results

| Scenario | Alerted before drop v1 / H1 / **H2** / H3 | False alerts per run v1 / H1 / **H2** / H3 | Median takeoff→alert H1 → **H2** | Pilot error, median v1 / H1 / H2 / **H3** | Attack flagged H1 → H2 |
|---|---|---|---|---|---|
| baseline_rf | 85% / 95% / **100%** / 100% | 0.20 / 0.05 / **0.00** / 0.00 | 68 s → **14 s** | 252 / 127 / 118 / **43 m** | n/a |
| baseline_dark | 85% / 85% / **100%** / 100% | 0 / 0 / **0** / 0 | 127 s → **70 s** | – | n/a |
| rid_ghost | – | 4.50 / 0.00 / **0.00** / 0.00 | – | – | 100% → 100% |
| rid_op_misdirect | 100% / 100% / **100%** / 100% | 0.25 / 0.05 / **0.00** / 0.05 | 8 s → **5 s** | 761 / 117 / 91 / **42 m** | 100% → 100% |
| rid_pos_offset | 85% / 95% / **100%** / 100% | 21.30 / 0.05 / **0.10** / 0.00 | 78 s → **15 s** | 801 / 98 / 118 / **48 m** | 100% → 100% |
| rf_decoy_downlink | 85% / 85% / **95%** / 100% | 0.45 / 0.10 / **0.05** / 0.05 | 120 s → **72 s** | – | 70% → **100%** |
| ctrl_clone | 85% / 95% / **100%** / 100% | 0.20 / 0.15 / **0.00** / 0.00 | 60 s → **13 s** | 946 / 328 / 335 / **288 m** | 75% → 75% |
| decoy_saturation | 85% / 75% / **100%** / 100% | 0.50 / 0.15 / **0.05** / 0.05 | 104 s → **70 s** | – | n/a |
| swarm | 85% / 89% / **99%** / 100% | 0 / 0.05 / **0** / 0 | 108 s → **39 s** | – | n/a |
| terrain_masked | 50% / 60% / **95%** / 100% | 0 / 0 / **0** / 0 | 134 s → **91 s** | – | n/a |
| building_shadow | 0% / 0% / **55%** / 55% | 0 / 0 / **0** / 0 | – → **44 s** | – | 10% → 5% |
| bird_mimicry | 0% / 0% / **85%** / 85% | 0 / 0 / **0** / 0 | – → **108 s** | – | n/a |
| dive_drop | 70% / 85% / **100%** / 100% | 0 / 0 / **0** / 0.05 | 83 s → **41 s** | – | n/a |
| tether_drop | 90% / 100% / **100%** / 100% | 0.10 / 0.05 / **0.00** / 0.00 | 64 s → **12 s** | 196 / 141 / 173 / **43 m** | n/a |
| acoustic_flood | 85% / 85% / **100%** / 100% | 0 / 0 / **0** / 0 | 133 s → **67 s** | – | 100% → 100% |
| camera_dazzle | 0% / 90% / **100%** / 100% | 0 / 0 / **0.05** / 0 | 126 s → **86 s** | – | 100% → 100% |
| radar_surge | 85% / 85% / **100%** / 100% | 0 / 0 / **0** / 0 | 125 s → **72 s** | – | 100% → 100% |
| storm | 85% / 85% / **100%** / 100% | 0.05 / 0 / **0** / 0 | 87 s → **20 s** | 290 / 102 / 99 / **49 m** | n/a |

**Control nights (no attack):** H2 and H3 both 0.00 false drone alerts, 0.00 sensor-degraded alarms, 0.00 spoof indicators per night. H1 was the same.

**False-alarm sources (H2):** all birds, 5 in 360 attack runs (rid_pos_offset 2, rf_decoy_downlink 1, decoy_saturation 1, camera_dazzle 1). Decoy drones in decoy_saturation are real drones, so alerting on them (98%) is correct.

## Why H2 moved so much

- **Micro-Doppler.** A rotating radar's beam dwell (~25 ms) covers two blade-flash periods of a Phantom-class rotor (140 Hz) but not a bird's wingbeat. So the radar answers "blade flash: yes or no" on every plot of a strong target, and absence counts against "drone".
  - It is independent of flight path, which is why bird_mimicry went from 0% to 85% and dive, tether and terrain-masked runs alert much earlier.
  - Model in `microdoppler.py`, ported from `apps/simulator/sensors2.js` and anchored to published blade-flash and rotor-SNR figures (see `apps/simulator/SENSOR_MODELS.md`).
- **Bird-gated RF.** A drone's RF bearing can no longer attach to a gull flying on the same line. This removed the RF-driven bird false alarms. A geometric match on a withheld track still suppresses the "RF emitter with no airborne target" decoy indicator, so real drones are not reported as decoys.
- **Overridable ground class.** Radar elevation noise makes a drone at 10 m look like a car. Blade flash or a confident camera now overrides the road-gated "vehicle" label.
- **Remote DF mast (H3).** Fence-corner nodes 400–500 m apart triangulate a pilot 1+ km away along a thin wedge. A fourth node about 1.1 km out cuts the median pilot error by roughly 60% (about 100–120 m → 42–49 m) wherever the pilot's controller is on air.

## What remains weak

1. **building_shadow (55%).** A drone that stays in the radar's shadow behind a cell block, and gives off no RF, relies on acoustic and camera only. Fix: a second radar, or a staring K-band panel, covering the shadowed sector. Also add a policy that treats the shadow sectors as higher-risk.
2. **ctrl_clone pilot location (288 m).** A cloned controller signature splits the DF bearings between two transmitters. The clone is flagged 75% of the time, but the pilot fix stays ambiguous. Fix: RF fingerprinting (transmitter hardware features) and wider DF baselines.
3. **Micro-Doppler is modelled, not measured.** The benefit rests on an assumed 22.5 dB rotor-to-body ratio, a 0.95 classifier ceiling and a 25 ms dwell. Plastic-prop mini drones and slow, large-prop builds will be weaker. Validate with the actual radar before quoting H2 numbers. A radar without a micro-Doppler output reverts to H1 behaviour.
4. **Camera model attacks.**
   - Laser dazzle drives recall to ~0%, so the system treats it as "camera degraded" and falls back to the other sensors.
   - Sky-coloured camouflage and bird-shaped shells cut the reference detector's recall by 20–30 points.
   - The hardening augmentation recipe did **not** help the small CPU reference detector (`redteam/results/vision_robustness.md`). Re-test it on the GPU-trained YOLO before relying on it.
5. **Remote ID is unauthenticated by design.** Verification against DF and radar catches every ghost and offset spoof here. A spoofer who is physically co-located with a real drone would pass; treat Remote ID as a hint, never as identity.

## Recommendations

- **Radar:** buy one with a micro-Doppler or classification output, and ask for its blade-flash detection range at your drone RCS.
- **Sites with shadowed sectors:** add a staring panel radar.
- **RF:** place at least one RF DF node 1 km or more from the fence (H3). It is the cheapest large gain in pilot location.
- **Operator display:** keep "sensor degraded" and "spoof/decoy indicator" as separate alarm classes from "drone".
- **Remote ID pilot locations:** only dispatch police on those marked `remote_id` (verified), never on `remote_id_unverified`.
- **Re-run cadence:** run this suite after every fusion or model change, and before every release.

## Files

- `redteam/attacks.py`: scenario builders
- `redteam/run_redteam.py`: v1 vs H1
- `redteam/compare_v3.py`: H2 and H3
- `redteam/vision_robustness.py`: camera attacks
- `microdoppler.py`: micro-Doppler observation model
- `redteam/tests/`: tests
- `redteam/results/`: raw JSON and charts
