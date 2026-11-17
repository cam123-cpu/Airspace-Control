#!/usr/bin/env python3
"""Capture crops from the sim's own PTZ camera, train a small pixel classifier, export int8 model.json.

  python3 train_browser_model.py --html ../../apps/simulator/index.html --out model.json

Stages (all re-runnable):
  1. capture  - headless Chrome runs the sim page, aims the PTZ camera at drones / birds / clutter and
                crops 32x32 grayscale patches with pixelModel.cropFromCanvas (the same code used at runtime).
  2. train    - per-crop standardised pixels -> sklearn MLP; compared against HOG + logistic regression.
                Train/validation/test are split by simulation seed (no frame leakage between splits).
  3. export   - int8 weights (per-layer scale), float32 biases, temperature; parity-checked against infer.js in Node.
"""
import argparse, asyncio, base64, csv, json, os, subprocess, sys, time, warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CLASSES = ["drone", "drone_payload", "bird", "clutter"]
STD_FLOOR = 0.02
N = 32


# ---------------------------------------------------------------- capture
async def _capture(html: Path, seeds_train, seeds_test, shots, tmax, thermal_frac, chrome):
    from playwright.async_api import async_playwright
    infer_src = (HERE / "infer.js").read_text()
    cap_src = (HERE / "capture.js").read_text()
    X, meta = [], []
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path=chrome, args=["--no-sandbox"])
        pg = await b.new_page(viewport={"width": 1400, "height": 900})
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        await pg.route("**/fonts.*/**", lambda r: r.abort())
        await pg.goto(html.resolve().as_uri())
        await pg.wait_for_timeout(300)
        await pg.evaluate("() => { playing = false; }")
        await pg.evaluate("(src) => { (0, eval)(src); }", infer_src)
        await pg.evaluate("(src) => { (0, eval)(src); }", cap_src)
        scens = await pg.evaluate("() => Object.keys(SCEN)")
        runs = [(s, sd, "train") for sd in seeds_train for s in scens] + [(s, sd, "test") for sd in seeds_test for s in scens]
        t0 = time.time()
        for i, (scen, seed, split) in enumerate(runs):
            r = await pg.evaluate("(cfg) => __capture(cfg)", {"scen": scen, "seed": seed, "salt": 17, "shots": shots, "tmax": tmax, "thermalFrac": thermal_frac})
            arr = np.frombuffer(base64.b64decode(r["b64"]), dtype=np.uint8).reshape(-1, N * N)
            X.append(arr)
            for m in r["meta"]:
                m["split"] = split
                meta.append(m)
            print(f"  [{i+1}/{len(runs)}] {scen:8s} seed {seed:3d} {split:5s}: {len(arr):4d} crops  counts={r['counts']}  ({time.time()-t0:.0f}s)", flush=True)
        if errs:
            print("page errors:", errs[:3])
        await b.close()
    return np.concatenate(X), meta


def capture(args):
    seeds_train = list(range(1, args.train_seeds + 1))
    seeds_test = list(range(101, 101 + args.test_seeds))
    X, meta = asyncio.run(_capture(Path(args.html), seeds_train, seeds_test, args.shots, args.tmax, args.thermal_frac, args.chrome))
    args.cache.mkdir(parents=True, exist_ok=True)
    np.save(args.cache / "crops.npy", X)
    with open(args.cache / "meta.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(meta[0].keys()))
        w.writeheader(); w.writerows(meta)
    return X, meta


def load_cache(cache: Path):
    X = np.load(cache / "crops.npy")
    with open(cache / "meta.csv") as f:
        meta = list(csv.DictReader(f))
    for m in meta:
        m["cls"] = int(m["cls"]); m["px"] = float(m["px"]); m["seed"] = int(m["seed"])
    return X, meta


# ---------------------------------------------------------------- features / models
def normalize(gray_u8: np.ndarray) -> np.ndarray:
    """Per-crop standardisation; identical to pixelModel.normalize in infer.js."""
    x = gray_u8.astype(np.float64) / 255.0
    mean = x.mean(axis=1, keepdims=True)
    sd = np.sqrt(((x - mean) ** 2).mean(axis=1, keepdims=True))
    return (x - mean) / np.maximum(sd, STD_FLOOR)


HOG_SCALE = 4.0


def hog_lite(x: np.ndarray, chunk: int = 4000) -> np.ndarray:
    """Same as pixelModel.hogLite in infer.js. x: (n, 1024) normalised crops -> (n, 324). Chunked to bound memory."""
    if x.shape[0] > chunk:
        return np.concatenate([hog_lite(x[i:i + chunk], chunk) for i in range(0, x.shape[0], chunk)])
    n = x.shape[0]
    im = x.reshape(n, N, N)
    jp = np.minimum(np.arange(N) + 1, N - 1); jm = np.maximum(np.arange(N) - 1, 0)
    gx = im[:, :, jp] - im[:, :, jm]
    gy = im[:, jp, :] - im[:, jm, :]
    mag = np.sqrt(gx * gx + gy * gy)
    a = np.arctan2(gy, gx)
    a = np.where(a < 0, a + np.pi, a)
    a = np.where(a >= np.pi, a - np.pi, a)
    b = a / (np.pi / 9) - 0.5
    b0 = np.floor(b); w1 = b - b0
    i0 = (b0.astype(int) % 9); i1 = ((b0.astype(int) + 1) % 9)
    cell = (np.arange(N)[:, None] // 8) * 4 + (np.arange(N)[None, :] // 8)
    base = (np.arange(n)[:, None, None] * 16 + cell[None]) * 9
    flat = np.bincount((base + i0).ravel(), weights=(mag * (1 - w1)).ravel(), minlength=n * 144)
    flat += np.bincount((base + i1).ravel(), weights=(mag * w1).ravel(), minlength=n * 144)
    cells = flat.reshape(n, 4, 4, 9)
    out = []
    for bi in range(3):
        for bj in range(3):
            v = np.concatenate([cells[:, bi, bj], cells[:, bi, bj + 1], cells[:, bi + 1, bj], cells[:, bi + 1, bj + 1]], axis=1)
            out.append(v / np.sqrt((v * v).sum(axis=1, keepdims=True) + 1e-2))
    return np.concatenate(out, axis=1)


def features(gray_u8: np.ndarray, kind: str) -> np.ndarray:
    x = normalize(gray_u8)
    if kind == "pixels":
        return x
    h = hog_lite(x) * HOG_SCALE
    return h if kind == "hog" else np.concatenate([x, h], axis=1)  # float64 kept for exact parity with infer.js


def hflip(g):
    return g.reshape(-1, N, N)[:, :, ::-1].reshape(-1, N * N)


def hog_feats(g):
    from skimage.feature import hog
    return np.stack([hog(im.reshape(N, N), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)) for im in g])


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def mlp_logits(layers, x):
    h = x
    for k, (Wm, b) in enumerate(layers):
        h = h @ Wm + b
        if k < len(layers) - 1:
            h = np.maximum(h, 0)
    return h


def fit_temperature(logits, y):
    best = (1.0, 1e9)
    for T in np.linspace(0.4, 12.0, 233):
        p = softmax(logits / T)
        nll = -np.mean(np.log(p[np.arange(len(y)), y] + 1e-12))
        if nll < best[1]:
            best = (float(T), float(nll))
    return best[0]


def quantize(layers):
    q = []
    for Wm, b in layers:
        scale = float(np.abs(Wm).max() / 127.0) or 1e-8
        qi = np.clip(np.round(Wm / scale), -127, 127).astype(np.int8)
        q.append((qi, scale, b.astype(np.float32)))
    return q


def dequant_layers(q):
    return [(qi.astype(np.float64) * s, b.astype(np.float64)) for qi, s, b in q]


def bucket(px):
    if px <= 0:
        return "n/a (empty/patch/light)"
    if px < 12:
        return "06-12 px"
    if px < 24:
        return "12-24 px"
    if px < 48:
        return "24-48 px"
    return "48+ px"


def report(y, pred, meta_rows, title):
    from sklearn.metrics import classification_report, confusion_matrix
    out = {"title": title, "n": int(len(y)), "accuracy": float((y == pred).mean())}
    out["per_class"] = classification_report(y, pred, labels=range(4), target_names=CLASSES, output_dict=True, zero_division=0)
    out["confusion"] = confusion_matrix(y, pred, labels=range(4)).tolist()
    def group(key):
        g = {}
        for i, m in enumerate(meta_rows):
            k = key(m)
            g.setdefault(k, [0, 0])
            g[k][0] += int(y[i] == pred[i]); g[k][1] += 1
        return {k: {"acc": round(v[0] / v[1], 4), "n": v[1]} for k, v in sorted(g.items())}
    out["by_px"] = group(lambda m: bucket(m["px"]))
    out["by_band"] = group(lambda m: m["band"])
    out["by_light"] = group(lambda m: {"0": "night", "0.45": "dusk", "1": "day"}.get(str(m["tod"]), str(m["tod"])))
    out["by_weather"] = group(lambda m: m["wx"])
    out["by_kind"] = group(lambda m: m["kind"])
    # drone (any) vs not: the decision fusion actually uses
    yd, pd = np.isin(y, [0, 1]), np.isin(pred, [0, 1])
    out["drone_vs_rest"] = {"recall": float((yd & pd).sum() / max(1, yd.sum())), "precision": float((yd & pd).sum() / max(1, pd.sum())),
                            "bird_called_drone_rate": float(((y == 2) & pd).sum() / max(1, (y == 2).sum())),
                            "clutter_called_drone_rate": float(((y == 3) & pd).sum() / max(1, (y == 3).sum()))}
    return out


def train(X, meta, args):
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import LogisticRegression
    y = np.array([m["cls"] for m in meta])
    split = np.array([m["split"] for m in meta])
    seeds = np.array([m["seed"] for m in meta])
    train_seeds = sorted(set(seeds[split == "train"]))
    val_seed = train_seeds[-1]
    tr = (split == "train") & (seeds != val_seed)
    va = (split == "train") & (seeds == val_seed)
    te = split == "test"
    print(f"split: train {tr.sum()} / val {va.sum()} (seed {val_seed}) / test {te.sum()} (seeds {sorted(set(seeds[te]))})")
    print("class counts train:", np.bincount(y[tr], minlength=4).tolist(), " test:", np.bincount(y[te], minlength=4).tolist())
    # balance training classes by oversampling to the largest class, then add horizontal flips
    rng = np.random.default_rng(0)
    idx = np.where(tr)[0]
    cnt = np.bincount(y[idx], minlength=4)
    bal = np.concatenate([rng.choice(idx[y[idx] == c], cnt.max(), replace=cnt[c] < cnt.max()) if cnt[c] else np.array([], int) for c in range(4)])
    Xtr_g = np.concatenate([X[bal], hflip(X[bal])])
    ytr = np.concatenate([y[bal], y[bal]])
    results = {}
    # --- candidate A: MLPs on pixels / hogLite / both (all implemented in infer.js)
    hidden = tuple(int(h) for h in args.hidden.split(","))
    best_mlp, best_va, best_kind = None, -1, None
    for kind in args.features:
        Ftr, Fva = features(Xtr_g, kind), features(X[va], kind)
        for alpha in args.alphas:
            mlp = MLPClassifier(hidden_layer_sizes=hidden, activation="relu", alpha=alpha, batch_size=256, learning_rate_init=1e-3,
                                max_iter=args.max_iter, early_stopping=True, validation_fraction=0.1, n_iter_no_change=12, random_state=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mlp.fit(Ftr, ytr)
            acc = mlp.score(Fva, y[va])
            results[f"mlp[{kind},alpha={alpha}]"] = round(acc, 4)
            print(f"  MLP {hidden} on {kind:10s} alpha={alpha}: val acc {acc:.4f} ({mlp.n_iter_} iters)", flush=True)
            if acc > best_va:
                best_mlp, best_va, best_kind = mlp, acc, kind
    results["mlp"] = best_va
    Xva, Xte = features(X[va], best_kind), features(X[te], best_kind)
    # --- candidate B: HOG + logistic regression
    lr = LogisticRegression(C=1.0, max_iter=3000)
    Htr, Hva = hog_feats(Xtr_g), hog_feats(X[va])
    lr.fit(Htr, ytr)
    results["hog_lr"] = lr.score(Hva, y[va])
    print(f"  HOG(skimage)+LR baseline: val acc {results['hog_lr']:.4f}")
    print(f"  chosen: MLP on {best_kind} (val {best_va:.4f})")
    layers = [(Wm.astype(np.float64), b.astype(np.float64)) for Wm, b in zip(best_mlp.coefs_, best_mlp.intercepts_)]
    T = fit_temperature(mlp_logits(layers, Xva), y[va])
    q = quantize(layers)
    dq = dequant_layers(q)
    p_float = softmax(mlp_logits(layers, Xte) / T)
    p_q = softmax(mlp_logits(dq, Xte) / T)
    meta_te = [m for m, k in zip(meta, te) if k]
    rep = {"chosen": f"mlp on {best_kind}", "features": best_kind, "hidden": list(hidden), "candidates_val_acc": results, "temperature": T,
           "float": report(y[te], p_float.argmax(1), meta_te, "held-out test (float weights)"),
           "int8": report(y[te], p_q.argmax(1), meta_te, "held-out test (int8 weights, as shipped)")}
    rep["hog_lr_test_acc"] = float(lr.score(hog_feats(X[te]), y[te]))
    rep["quantization_acc_drop"] = rep["float"]["accuracy"] - rep["int8"]["accuracy"]
    rep["counts"] = {"total": int(len(y)), "train_balanced_with_flips": int(len(ytr)), "val": int(va.sum()), "test": int(te.sum()),
                     "by_class_all": np.bincount(y, minlength=4).tolist()}
    return q, T, rep, (X[te], p_q)


def export(q, T, rep, out: Path):
    layers = []
    for k, (qi, s, b) in enumerate(q):
        layers.append({"nin": int(qi.shape[0]), "nout": int(qi.shape[1]), "scale": s, "act": "relu" if k < len(q) - 1 else "linear",
                       "w": base64.b64encode(qi.tobytes(order="C")).decode(), "b": base64.b64encode(b.astype("<f4").tobytes()).decode()})
    m = {"format": "pixelmlp-int8-v1", "classes": CLASSES, "temperature": T,
         "input": {"size": N, "features": rep["features"], "hog_scale": HOG_SCALE, "gray": "round(0.299R+0.587G+0.114B) via pixelModel.cropFromCanvas", "std_floor": STD_FLOOR,
                   "norm": "per-crop: x=g/255; (x-mean)/max(std,std_floor)", "crop_side": "about 1.6x expected target px, min 12"},
         "trained_on": "sim PTZ renders only (night/dusk/day, clear/fog/rain, visible + thermal)",
         "test_metrics": {"accuracy": round(rep["int8"]["accuracy"], 4), "drone_vs_rest": rep["int8"]["drone_vs_rest"], "by_px": rep["int8"]["by_px"]},
         "layers": layers}
    out.write_text(json.dumps(m, separators=(",", ":")))
    return m


def parity(Xte_u8, p_q, model_path: Path, n=200):
    sel = Xte_u8[:n]
    tmp = HERE / ".parity_in.json"
    tmp.write_text(json.dumps({"x": sel.tolist()}))
    js = f"""
const pm=require({json.dumps(str(HERE / 'infer.js'))});const fs=require('fs');
pm.load(JSON.parse(fs.readFileSync({json.dumps(str(model_path))},'utf8')));
const inp=JSON.parse(fs.readFileSync({json.dumps(str(tmp))},'utf8')).x;
process.stdout.write(JSON.stringify(inp.map(g=>pm.classify(g).probs)));"""
    node = os.environ.get("NODE", "node")
    r = subprocess.run([node, "-e", js], capture_output=True, text=True, timeout=120)
    tmp.unlink(missing_ok=True)
    if r.returncode != 0:
        raise RuntimeError("node parity run failed: " + r.stderr[:500])
    pj = np.array(json.loads(r.stdout))
    diff = float(np.abs(pj - p_q[:n]).max())
    agree = float((pj.argmax(1) == p_q[:n].argmax(1)).mean())
    return diff, agree


def write_readme(rep, model_meta, size_kb, diff, agree, path: Path):
    i8 = rep["int8"]
    pc = i8["per_class"]
    lines = ["# In-browser pixel classifier for the PTZ camera", "",
             "A small neural network (MLP, int8) that looks at a 32×32 grayscale crop (plus edge-orientation features computed from it) from the sim's PTZ camera and says "
             "**drone**, **drone carrying a package**, **bird**, or **clutter** (sky, lights, buildings, insects, aircraft, bags, empty track positions).",
             "It replaces the sim's stand-in classifier, which guessed from target size and a dice roll.", "",
             "## Re-run", "", "```", "python3 train_browser_model.py --html ../../apps/simulator/index.html --out model.json", "```",
             "Captures fresh crops from whatever renderer the page currently has, retrains, exports and parity-checks. "
             "`--skip-capture` reuses `cache/`.", "",
             "## Results on held-out simulation seeds (int8 model as shipped)", "",
             f"- Crops: {rep['counts']['total']} total ({', '.join(f'{c} {n}' for c, n in zip(CLASSES, rep['counts']['by_class_all']))}); "
             f"test set {rep['counts']['test']} from seeds never used in training.",
             f"- Overall accuracy: **{i8['accuracy']:.1%}** (float {rep['float']['accuracy']:.1%}; int8 quantisation cost {rep['quantization_acc_drop']*100:.2f} pts).",
             f"- Drone (with or without package) vs everything else: recall {i8['drone_vs_rest']['recall']:.1%}, precision {i8['drone_vs_rest']['precision']:.1%}; "
             f"birds called drones {i8['drone_vs_rest']['bird_called_drone_rate']:.1%}, clutter called drones {i8['drone_vs_rest']['clutter_called_drone_rate']:.1%}.",
             f"- Model choice on validation: {rep['chosen']} ({rep['candidates_val_acc']['mlp']:.1%}) vs skimage HOG + logistic regression baseline {rep['candidates_val_acc']['hog_lr']:.1%} "
             f"(HOG+LR on test: {rep['hog_lr_test_acc']:.1%}).",
             f"- model.json: {size_kb:.0f} KB. JS vs Python (dequantised) max probability difference on 200 test crops: {diff:.2e}; argmax agreement {agree:.1%}.",
             f"- Temperature scaling T = {rep['temperature']:.2f} (fitted on the validation seed).", "",
             "| Class | Precision | Recall | n |", "|---|---|---|---|"]
    for c in CLASSES:
        lines.append(f"| {c} | {pc[c]['precision']:.1%} | {pc[c]['recall']:.1%} | {int(pc[c]['support'])} |")
    lines += ["", "Confusion matrix (rows = truth, columns = prediction; order " + ", ".join(CLASSES) + "):", "", "```"]
    lines += ["  ".join(f"{v:5d}" for v in row) for row in i8["confusion"]]
    lines += ["```", "", "| Slice | Accuracy | n |", "|---|---|---|"]
    for name, key in [("Target size", "by_px"), ("Band", "by_band"), ("Light", "by_light"), ("Weather", "by_weather")]:
        for k, v in i8[key].items():
            lines.append(f"| {name}: {k} | {v['acc']:.1%} | {v['n']} |")
    lines += ["", "## Integration (browser)", "",
              "```js", "// once: load infer.js (inline it), then", "pixelModel.load(MODEL_JSON);            // parsed model.json object",
              "// each classification: crop the composited camera frame BEFORE any HUD / boxes / text are drawn",
              "const g = pixelModel.cropFromCanvas(cctx, sx, sy, pixelModel.cropSize(expectedPx));",
              "const r = pixelModel.classify(g);       // {probs:[drone, drone_payload, bird, clutter], cls, label, conf}", "```", "",
              "- `sx, sy`: where the target is expected in the frame (projected track estimate, or video-lock position).",
              "- `expectedPx`: expected target size in pixels (span × focal / range). The crop side is ~1.6× that, min 12 px.",
              "- Training crops were taken after noise + vignette, with no HUD, labels or detector boxes on the canvas. "
              "Crops containing the crosshair or overlay text are out of distribution.",
              "- Fusion: P(drone) = probs[0] + probs[1]; payload when probs[1] dominates.", "",
              "## Honest limits", "",
              "- Trained and tested **only on this simulator's renders**. High accuracy here means the model learned the sim's drawing style; "
              "it says nothing about real cameras. Real-world accuracy needs the GPU pipeline in `sim-core/training/` with real footage, "
              "and a real-only test set.",
              "- The sim draws drones and birds as simple shapes; real birds, drones and backgrounds vary far more.",
              "- 32×32 grayscale throws away colour and fine detail; a production detector should be a full CNN/YOLO on full-resolution frames, plus a temporal (multi-frame) model.",
              "- Browser canvas resampling can differ slightly between browsers; capture and parity checks used Chrome."]
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--html", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", type=Path, default=HERE / "cache")
    ap.add_argument("--skip-capture", action="store_true")
    ap.add_argument("--train-seeds", type=int, default=5)
    ap.add_argument("--test-seeds", type=int, default=2)
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--tmax", type=float, default=300)
    ap.add_argument("--thermal-frac", type=float, default=0.35)
    ap.add_argument("--hidden", default="96,48")
    ap.add_argument("--features", nargs="+", default=["hog", "pixels+hog", "pixels"], choices=["pixels", "hog", "pixels+hog"])
    ap.add_argument("--alphas", type=float, nargs="+", default=[1e-3, 1e-2])
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--chrome", default="/usr/bin/google-chrome")
    args = ap.parse_args()
    if not (1 <= args.train_seeds <= 50 and 1 <= args.test_seeds <= 20 and 1 <= args.shots <= 50 and 10 <= args.tmax <= 3600):
        sys.exit("argument out of range")
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.skip_capture and (args.cache / "crops.npy").exists():
        X, meta = load_cache(args.cache)
        print(f"loaded {len(X)} cached crops")
    else:
        print("capturing crops from", args.html)
        X, meta = capture(args)
        X, meta = load_cache(args.cache)
    print(f"total crops: {len(X)}  by class: {np.bincount([m['cls'] for m in meta], minlength=4).tolist()}")
    q, T, rep, (Xte, p_q) = train(X, meta, args)
    model_meta = export(q, T, rep, out)
    size_kb = out.stat().st_size / 1024
    diff, agree = parity(Xte, p_q, out)
    rep["parity"] = {"max_prob_diff": diff, "argmax_agreement": agree, "n": min(200, len(Xte))}
    rep["model_json_kb"] = round(size_kb, 1)
    (HERE / "report.json").write_text(json.dumps(rep, indent=1))
    write_readme(rep, model_meta, size_kb, diff, agree, HERE / "README.md")
    i8 = rep["int8"]
    print(f"\nheld-out int8 accuracy {i8['accuracy']:.4f} (float {rep['float']['accuracy']:.4f}); drone-vs-rest {i8['drone_vs_rest']}")
    print("by px:", i8["by_px"]); print("by band:", i8["by_band"])
    print(f"model.json {size_kb:.1f} KB; parity max diff {diff:.2e}, argmax agreement {agree:.3f}")
    if diff >= 1e-3:
        sys.exit("PARITY CHECK FAILED")


if __name__ == "__main__":
    main()
