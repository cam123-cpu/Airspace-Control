# Prison Airspace Sim

An interactive digital twin of a correctional facility's airspace. Drones, birds, traffic and aircraft move through a 3D site. Four sensor types detect them through physics models, a fusion engine turns the raw detections into tracks and alerts, and a PTZ camera inspects the most suspicious track.

```bash
python3 -m http.server 8000   # then open http://localhost:8000
```

## Files

| File | What it does |
|---|---|
| `index.html` | App shell, world model, tracker, fusion engine, PTZ camera, panels, 2D map, scenarios |
| `gl3d.js` | Readable source of the inline WebGL2 renderer: HDR, ACES tone mapping, shadow maps, sky, terrain, floodlights, weather |
| `sensors2.js` | Physics sensor models (radar equation + Swerling, micro-Doppler, LWIR/visible Johnson criteria, RF link budget, ISO 9613 acoustics) |
| `sensors2.test.js` | 175 calibration tests: `node sensors2.test.js` |
| `views.js` | Operator views: radar PPI, range-Doppler + micro-Doppler, RF waterfall, acoustic spectrogram |
| `pixelmodel.js`, `pixelmodel_weights.js` | In-browser camera classifier (trained on this sim's frames; see `ml/browser-model`) |

## Things to try

- **Scenarios:** *Night drop*, then the **red-team** set: spoofed Remote ID, RF decoy, a 4-drone swarm with decoys, and a low-altitude masked approach.
- **Sensors:** turn sensors off and watch fusion degrade gracefully. *Birds only* with the camera off shows why all four sensors matter.
- **Conditions:** Night / Dusk / Day, Clear / Fog / Rain, wind and clutter. They change sensor performance, not just the look.
- **View menu:** guard tower, camera-turret POV (true-scale zoom), top-down, pilot's position, drone's view, chase.
- **Training labels:** **Copy YOLO labels** exports auto-labels for the current camera frame.

Security: strict CSP (own origin only, no network calls); every dynamic string is escaped before it's rendered.
