# Counter-drone sensor-fusion simulator + synthetic PTZ data factory

A first working simulation of the prison counter-drone product: four sensors (RF direction finding with Remote ID, radar, acoustic, cued PTZ camera) fused into one alert with a predicted drop zone and pilot location. It also produces labelled camera frames for training the vision model.

Dependencies: numpy, scipy, opencv-python, Pillow, pandas, matplotlib only. No GPU is needed.

```
python generate_dataset.py --n 2000 --out data/ --mode mixed      # training frames + labels
python evaluate.py --nights 100 --out results/                    # Monte Carlo over simulated nights
python evaluate.py --export-sample sample_run.json                # one recorded run for the browser replay
```

## Files

| File | What it does |
|---|---|
| `scenario.py` | Site geometry (local metres, origin at the yard centre, 400 × 300 m fence) and sensor placement: radar + PTZ on a 20 m tower, 3 RF direction-finding nodes at fence corners, 4 acoustic nodes on the fence. Trajectory generators for: drone delivery runs (pilot 1–1.5 km out → climb to 60–120 m → ingress at 8–15 m/s → descend to 20–40 m → hover 10–40 s → drop → exit and land); birds (transit, circling, and foraging/loitering over the yard like gulls and crows, plus occasional wind-hover); flocks; aircraft; drifting plastic bags; trucks on the perimeter road. |
| `sensors.py` | Noise models. **Radar:** 2 Hz scans; range/azimuth/elevation noise; detection probability falls with range and radar cross-section; Poisson clutter. Birds are detected about as well as drones. **RF:** bearings to the drone and to the controller, with per-node bias plus noise, ~3 km range. 80% of drones have a live link; 25% broadcast Remote ID, which includes drone and operator position. **Acoustic:** detection range ~250 m, shrinking with wind; trucks and wind cause false detections. **PTZ:** slew-rate limited; zoom set from track range and uncertainty (1.5–30° FOV); grabs a frame once settled and scores whatever is nearest boresight. `StandInClassifier` stands in for the trained model; `RenderedModelClassifier` plugs in a real one. |
| `fusion.py` | Constant-velocity Kalman filter tracker with greedy nearest-neighbour Mahalanobis gating, 3-of-5 confirmation and track deletion. Per-track log-odds evidence (prior 1%): kinematics (hover, inside the fence, altitude, speed, path tortuosity), RF bearing matches, Remote ID, acoustic and camera. Alerts fire when P(drone) ≥ threshold within 1.5 km. Drop zone = hover centroid inside the fence, otherwise the chord of the extrapolated path through the fence. Pilot = Remote ID operator field if present, otherwise least-squares triangulation of controller bearings from ≥2 nodes. The camera is pointed at the most uncertain confirmed track (each look lasts 3 s). |
| `sim.py` | Runs one episode: truth → sensors → fusion → scoring. It can record a 2 Hz replay. |
| `render.py` | Procedural 640×512 PTZ frames in two modes. **Night visible:** sodium skyglow, clouds, fence/razor-wire and light-pole silhouettes, floodlight glare, LEDs. **Thermal (white-hot):** hot motors, warm birds, warm rooflines. Targets are 3D models projected at random yaw, pitch and viewing angle: quad with rotor discs, gimbal, skids and optional payload on a line; bird with flapping/gliding wings; bag, aircraft lights, or an insect at the lens. Pixel size comes from physical size, range and FOV. Degradations: rolling-shutter skew, motion blur, defocus, shot/read noise, column noise, vignetting, JPEG at quality 40–90. Boxes are taken from the final alpha mask, and objects with contrast < 0.03 are left unlabelled because they are genuinely invisible. |
| `generate_dataset.py` | Writes `images/`, YOLO `labels/`, `coco.json`, `meta.csv` (range, FOV, pixels on target, contrast, mode), `data.yaml`, and `contact_sheet.png`. Classes: `drone`, `drone_payload`, `bird`, `other`. |
| `evaluate.py` | Monte Carlo over N nights, each built from independent events. Per night: 0–2 drone runs, ~25 bird passes, ~4 aircraft, ~1 bag, ~4 trucks. Reports detection rate, alert before drop, takeoff→alert time, drop-zone error, pilot error, and false alarms per site-night. Each is split by drone type: RF-only, Remote ID, or dark. Also sweeps the alert threshold and the classifier strength. Outputs `report.md`, `summary.json` and 3 charts. |
| `sample_run.json` | One recorded run at 2 Hz: an RF-link drone with no Remote ID, plus birds. Contains site geometry, truth, radar plots, RF bearings, acoustic hits, camera pointing and scores, fused tracks with evidence breakdown, and alerts. Built for the browser replay. |

## Swapping in a real vision model

1. Generate data with `python generate_dataset.py --n 20000 --out data/`. Mix in real footage before trusting any number.
2. Train any detector on `data/data.yaml`. For example, with Ultralytics: `yolo detect train data=data/data.yaml imgsz=640 model=yolov8n.pt`. Hold out **real** frames for validation, from public drone-vs-bird and anti-UAV datasets plus your own night flights.
3. Close the loop in `evaluate.py` / `sim.py`:
   ```python
   from sensors import RenderedModelClassifier
   def predict(img):                     # BGR uint8 640x512
       r = model(img, verbose=False)[0]
       conf = [b.conf.item() for b in r.boxes if int(b.cls) in (0, 1)]  # drone, drone_payload
       return max(conf, default=0.02)
   params = {"classifier": RenderedModelClassifier(predict)}
   ```
   Each camera look then renders the object actually in view, at its simulated pixel size, and runs your model on it. The false-alarm and alert-before-drop numbers then reflect the real model rather than the stand-in.

## What is approximated (read before quoting numbers)

- **Calibration.** False-alarm rates depend mostly on assumptions that need field data: how often birds hover over the yard, the classifier's 5% "consistently wrong about this object" rate, and the clutter rate. The headline results come from the stand-in classifier and show how the system behaves. They are not a performance claim.
- **Quiet time.** Nights are simulated as independent ~2–8 min events. Clutter-only time between events is not simulated, which slightly under-counts clutter-driven false alarms.
- **Physics.** No terrain, buildings, multipath or occlusion. Radar has no Doppler or micro-Doppler, which real systems use heavily to separate rotors from wings and would make the kinematics evidence stronger. RF is bearing-only; there is no protocol decoding.
- **Camera.** The camera has no autonomous visual tracker. Pointing follows the fused track, so "empty frame" looks happen when the track is off.
- **Evidence weights.** The log-odds weights in `fusion.py` are hand-set. Once real labelled tracks exist, replace them with weights learned by logistic regression on the same features.
- **Rendering.** Sprites are simple geometric models, not CAD meshes. Before training a production model, add real-footage backgrounds (cut-paste onto recorded night sky) and Unreal/Isaac renders.
- **Pilot location.** Without Remote ID, triangulation from 3 fence-corner nodes gives about 170–250 m median error at 1–1.5 km. That is a search area, not an address. Wider baselines or more nodes would cut it substantially.
