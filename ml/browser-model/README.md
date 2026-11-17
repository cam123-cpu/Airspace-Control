# In-browser pixel classifier for the PTZ camera

A small neural network (MLP, int8) that looks at a 32×32 grayscale crop (plus edge-orientation features computed from it) from the sim's PTZ camera and says **drone**, **drone carrying a package**, **bird**, or **clutter** (sky, lights, buildings, insects, aircraft, bags, empty track positions).
It replaces the sim's stand-in classifier, which guessed from target size and a dice roll.

## Re-run

```
python3 train_browser_model.py --html ../../apps/simulator/index.html --out model.json
```
Captures fresh crops from whatever renderer the page currently has, retrains, exports and parity-checks. `--skip-capture` reuses `cache/`. Full run: about 3 min capture (30k crops) plus 10–15 min training on 2 cores; `--features hog` alone trains in about 3 min.

In-browser check on a never-seen seed: `python3 browser_check.py ../../apps/simulator/index.html 303` (latest: 91.7% accuracy on 1,824 crops, drone recall 92.8%, 0.09 ms per classification).

## Results on held-out simulation seeds (int8 model as shipped)

- Crops: 30010 total (drone 5645, drone_payload 5631, bird 7633, clutter 11101); test set 7656 from seeds never used in training.
- Overall accuracy: **89.6%** (float 89.7%; int8 quantisation cost 0.14 pts).
- Drone (with or without package) vs everything else: recall 92.5%, precision 91.9%; birds called drones 6.6%, clutter called drones 3.7%.
- Model choice on validation: mlp on hog (90.6%) vs skimage HOG + logistic regression baseline 81.1% (HOG+LR on test: 77.7%).
- model.json: 49 KB. JS vs Python (dequantised) max probability difference on 200 test crops: 4.77e-15; argmax agreement 100.0%.
- Temperature scaling T = 4.55 (fitted on the validation seed).

| Class | Precision | Recall | n |
|---|---|---|---|
| drone | 87.6% | 88.6% | 1438 |
| drone_payload | 93.0% | 93.1% | 1431 |
| bird | 88.0% | 84.3% | 1946 |
| clutter | 89.8% | 91.8% | 2841 |

Confusion matrix (rows = truth, columns = prediction; order drone, drone_payload, bird, clutter):

```
 1274     25     73     66
   24   1332     23     52
  113     15   1641    177
   44     61    127   2609
```

| Slice | Accuracy | n |
|---|---|---|
| Target size: 06-12 px | 74.6% | 650 |
| Target size: 12-24 px | 82.8% | 1161 |
| Target size: 24-48 px | 90.4% | 1353 |
| Target size: 48+ px | 93.6% | 2047 |
| Target size: n/a (empty/patch/light) | 92.9% | 2445 |
| Band: thermal | 94.6% | 2473 |
| Band: vis | 87.1% | 5183 |
| Light: day | 89.7% | 1773 |
| Light: dusk | 87.5% | 1960 |
| Light: night | 90.5% | 3923 |
| Weather: clear | 92.2% | 4615 |
| Weather: fog | 83.6% | 1556 |
| Weather: rain | 87.4% | 1485 |

## Integration (browser)

```js
// once: load infer.js (inline it), then
pixelModel.load(MODEL_JSON);            // parsed model.json object
// each classification: crop the composited camera frame BEFORE any HUD / boxes / text are drawn
const g = pixelModel.cropFromCanvas(cctx, sx, sy, pixelModel.cropSize(expectedPx));
const r = pixelModel.classify(g);       // {probs:[drone, drone_payload, bird, clutter], cls, label, conf}
```

- `sx, sy`: where the target is expected in the frame (projected track estimate, or video-lock position).
- `expectedPx`: expected target size in pixels (span × focal / range). The crop side is ~1.6× that, min 12 px.
- Training crops were taken after noise + vignette, with no HUD, labels or detector boxes on the canvas. Crops containing the crosshair or overlay text are out of distribution.
- Fusion: P(drone) = probs[0] + probs[1]; payload when probs[1] dominates.

## Honest limits

- Trained and tested **only on this simulator's renders**. High accuracy here means the model learned the sim's drawing style; it says nothing about real cameras. Real-world accuracy needs the GPU pipeline in `sim-core/training/` with real footage, and a real-only test set.
- The sim draws drones and birds as simple shapes; real birds, drones and backgrounds vary far more.
- 32×32 grayscale throws away colour and fine detail; a production detector should be a full CNN/YOLO on full-resolution frames, plus a temporal (multi-frame) model.
- Browser canvas resampling can differ slightly between browsers; capture and parity checks used Chrome.
