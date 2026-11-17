# Drone Watch Live

On-device drone vs bird detection from any camera: phone, laptop webcam, USB camera, OBS virtual camera, or a video file. Video is processed in the browser and never uploaded.

## What it does

1. **Motion detection on the sky**: adaptive background model with camera-shake cancellation; textured regions (trees, buildings, ground) are ignored by default.
2. **Tracking**: each object keeps an ID. A hovering object stays tracked through a local-contrast probe after its motion stops.
3. **Classification without a trained model**:
   - **Wingbeat**: a bird's outline flickers at 2× its wingbeat (typically 6–20 Hz). Drone rotors spin too fast for a camera to see, so a drone's outline stays rigid.
   - **Hover**: the ability to hover is a strong drone cue.
   - **Straight, rigid flight**: also counts toward "drone".
4. **Optional trained model**: load the `.onnx` YOLO model exported by `training/run_pipeline.sh`. It runs through ONNX Runtime Web (WebGPU, falling back to WASM) on every 4th frame. Its detections add evidence to tracks and can also spot objects that aren't moving.
5. **Alerts and evidence**: an alert fires after 1.5 s of consistent "drone" classification and is logged with a snapshot.
6. **Labeling**: click a box to correct its class. Mark events as drone, bird, other or false alarm, save full frames, then export JSON.

## Security

- No third-party code at runtime: run `./fetch_vendor.sh` once before hosting (needs Node/npm). It vendors ONNX Runtime Web into `vendor/`; npm checks the tarball against the registry's sha512 integrity, and the script writes `vendor/SHA256SUMS`.
- Strict Content-Security-Policy: only this app's own files; no outbound network calls.
- Models: paste the SHA-256 from the training pipeline's model card into **Expected SHA-256**. A model that doesn't match is refused. Models are ONNX (no pickle, no code execution on load) and capped at 300 MB.
- Video, frames and labels never leave the device unless you export them yourself.
- Host behind HTTPS with HSTS. Add `Permissions-Policy: camera=(self)` and `X-Content-Type-Options: nosniff` headers on your host.

## Run it

| Where | How | Camera works? |
|---|---|---|
| Laptop | `python3 -m http.server 8000` in this folder, open `http://localhost:8000` | Yes (localhost counts as secure) |
| Phone or anywhere | Host the folder on any HTTPS static host (below) and open the URL | Yes |
| Claude artifact preview | Demo sky, video and photo files | No: the preview sandbox blocks cameras and ONNX Runtime |

**Free HTTPS hosting:**
- **GitHub Pages:** push this folder to a repo, then Settings → Pages → Deploy from branch.
- **Netlify Drop:** drag the folder onto app.netlify.com/drop.
- **Cloudflare Pages:** `npx wrangler pages deploy .`

On a phone, open the URL, tap **Use camera**, and allow camera access. The rear camera is used by default; pick another from the dropdown. Use "Add to Home Screen" to install it as an app that works offline.

**IP cameras / CCTV (RTSP):** browsers can't open RTSP directly. Either:
- open the stream in OBS and start **Virtual Camera**; it then appears in the camera list, or
- run [MediaMTX](https://github.com/bluenviron/mediamtx) and open its WebRTC page as a source in OBS.

## Field-test protocol (to see if it works)

1. Pick a clear-sky site where you can legally fly. In the US, recreational flyers need TRUST; check airspace with B4UFLY.
2. Mount the phone on a tripod. Start with 1080p at 30 fps.
3. Fly passes at 30, 60, 100 and 150 m distance, plus hovers. Log the times.
4. Record birds at the same site with no drone flying.
5. For every pass, note: first-detection time, alert time, misses, and false alerts per hour.
6. Export the labels, correct them, and convert them into training data:
   `python3 -m training.import_liveapp export.json --out /data/raw/own_footage`.
   Then retrain. Your own sites are the data that matters most.

## Limits (honest)

- The rule-based classifier was tuned on synthetic sky video. Expect weaker results on real footage until a model trained on real data is loaded.
- **Pixels on target decide everything.** A 30 cm drone at 150 m covers about 4 px on a phone's wide lens at 1080p. Use the phone's 3× or 5× lens for range.
- Fast camera panning, heavy cloud motion and insects near the lens cause false tracks. Use a tripod and the "ignore textured areas" option.
