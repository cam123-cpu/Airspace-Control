"""Vision-model robustness benchmark + hardening recipe.

What it does
  1. Renders labelled PTZ frames with render.py (imported, not modified).
  2. Applies attacks / corruptions at severity 1-5:
       pixel-level : glare, laser_dazzle, fog, rain, snow, jpeg, motion_blur, defocus, low_light_noise,
                     ir_bloom (thermal), hot_decoys (thermal flares)
       object-level: camo_sky (sky-matched paint), bird_shell (bird-shaped shroud),
                     adv_patch_placeholder (high-contrast texture patch; NOT an optimised patch)
  3. Scores any detector behind the `Detector` interface: drone recall, false positives per
     frame and mean drone confidence, per corruption x severity, relative to clean frames.
  4. Ships `harden_augment()` + AUGMENT_RECIPE: the training-time augmentation to close the gaps.
  5. Optional gradient attacks (FGSM / PGD, L-inf) as an *evaluation harness*, enabled only when
     torch is installed (GPU box). They measure how far the trained model's drone score drops
     under small pixel perturbations, the standard robustness check before deployment.

Detectors
  OnnxYoloDetector   : the trained YOLO model exported to ONNX (needs onnxruntime). Product path.
  ReferenceDetector  : CPU-only classical baseline (residual blobs + gradient-boosted crop
                       classifier) so the benchmark runs end to end on a laptop. It is a
                       yardstick for the benchmark, not the product model.

Usage
  python -m redteam.vision_robustness --frames 40 --train 500            # reference detector, CPU
  python -m redteam.vision_robustness --onnx best.onnx --frames 200       # trained model on GPU box
  python -m redteam.vision_robustness --onnx best.onnx --pgd --torch-weights best.pt
"""
import argparse, json, os, sys, time, zlib
os.environ.setdefault("OMP_NUM_THREADS", "1")   # avoid OpenMP oversubscription in sklearn
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import render as R  # noqa: E402

W, H = R.W, R.H
DRONE = {"drone", "drone_payload"}
PIXEL_ATTACKS = ["glare", "laser_dazzle", "fog", "rain", "snow", "jpeg", "motion_blur", "defocus", "low_light_noise",
                 "ir_bloom", "hot_decoys"]
OBJECT_ATTACKS = ["camo_sky", "bird_shell", "adv_patch_placeholder"]
THERMAL_ONLY = {"ir_bloom", "hot_decoys"}
VISIBLE_ONLY = {"rain", "snow", "laser_dazzle", "camo_sky"}


# ======================================================================== frame synthesis
def make_scene(rng, mode=None, cls=None, px=None):
    """Clean scene before sensor finishing. Returns dict with float RGB image, background,
    object mask and label list (main object + optional birds)."""
    mode = mode or ("thermal" if rng.random() < 0.4 else "night")
    cls = cls or rng.choice(["drone", "drone_payload", "bird", "other"], p=[0.35, 0.25, 0.3, 0.1])
    px = px or float(np.exp(rng.uniform(np.log(8), np.log(70))))
    img, meta = R.background(rng, mode)
    bg = img.copy()
    cx, cy = W / 2 + rng.normal(0, 60), H * 0.42 + rng.normal(0, 50)
    objs = []
    box, sub, con = R.render_object(rng, cls, px, mode, img, cx, cy)
    if box is not None and con >= R.MIN_CONTRAST:
        objs.append({"cls": cls, "box": [int(v) for v in box], "px": px})
    if rng.random() < 0.3:
        b2 = img.copy()
        bx, _, c2 = R.render_object(rng, "bird", px * rng.uniform(0.5, 1.4), mode, img, cx + rng.normal(0, 150), cy + rng.normal(0, 80))
        if bx is not None and c2 >= R.MIN_CONTRAST:
            objs.append({"cls": "bird", "box": [int(v) for v in bx], "px": px})
    mask = (np.abs(img - bg).max(axis=2) > 0.01).astype(np.float32)
    return {"img": img, "bg": bg, "mask": mask, "objs": objs, "mode": mode, "main": cls, "px": px, "c": (cx, cy)}


def finish(rng, img, mode):
    out, _ = R.sensor_finish(rng, img.astype(np.float32), mode)
    return out


# ======================================================================== attacks (float RGB in, float RGB out)
def _blob(h, w, cx, cy, sigma):
    yy, xx = np.mgrid[0:h, 0:w]
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)).astype(np.float32)


def a_glare(rng, s, sc):
    cx, cy = sc["c"]
    g = _blob(H, W, cx + rng.normal(0, 80), cy + rng.normal(0, 60), 40 + 25 * s)
    col = np.array([1.0, 0.8, 0.55]) if sc["mode"] == "night" else np.array([1, 1, 1])
    out = sc["img"] + g[..., None] * col * (0.15 * s)
    for k in range(int(s)):   # lens-flare ghosts along the axis through the centre
        t = rng.uniform(-1, 1)
        out += _blob(H, W, W / 2 + t * (cx - W / 2), H / 2 + t * (cy - H / 2), 6 + 4 * s)[..., None] * col * 0.05 * s
    return out


def a_laser_dazzle(rng, s, sc):
    """Green laser pointed at the lens: saturated core, veiling halo, diffraction streaks."""
    cx, cy = sc["c"]
    px, py = cx + rng.normal(0, 30 / s), cy + rng.normal(0, 30 / s)
    core = _blob(H, W, px, py, 6 + 10 * s)
    halo = _blob(H, W, px, py, 40 + 50 * s)
    streak = np.zeros((H, W), np.float32)
    for ang in rng.uniform(0, np.pi, 3 + int(s)):
        d = np.array([np.cos(ang), np.sin(ang)]) * 900
        cv2.line(streak, (int(px - d[0]), int(py - d[1])), (int(px + d[0]), int(py + d[1])), 1.0, 1 + int(s))
    streak = cv2.GaussianBlur(streak, (0, 0), 2 + s)
    lum = 2.5 * core + 0.35 * s * halo + 0.25 * s * streak * halo ** 0.3
    return sc["img"] + lum[..., None] * np.array([0.35, 1.0, 0.35])


def a_fog(rng, s, sc):
    k = 0.12 * s
    air = sc["img"].mean(axis=(0, 1)) * 0.6 + 0.08
    return sc["img"] * (1 - k) + air * k


def a_rain(rng, s, sc):
    lay = np.zeros((H, W), np.float32)
    ang = rng.uniform(-0.3, 0.3)
    for _ in range(int(300 * s)):
        x, y = rng.uniform(0, W), rng.uniform(0, H)
        L = rng.uniform(8, 20)
        cv2.line(lay, (int(x), int(y)), (int(x + L * np.sin(ang)), int(y + L * np.cos(ang))), rng.uniform(0.1, 0.35), 1)
    lay = cv2.GaussianBlur(lay, (0, 0), 0.8)
    return a_fog(rng, s * 0.4, sc) + lay[..., None] * 0.6


def a_snow(rng, s, sc):
    lay = np.zeros((H, W), np.float32)
    for _ in range(int(400 * s)):
        cv2.circle(lay, (int(rng.uniform(0, W)), int(rng.uniform(0, H))), int(rng.integers(1, 3)), float(rng.uniform(0.2, 0.7)), -1)
    lay = cv2.GaussianBlur(lay, (0, 0), 1.0)
    return a_fog(rng, s * 0.5, sc) + lay[..., None]


def a_motion_blur(rng, s, sc):
    L = int(5 + 9 * s)
    k = np.zeros((L, L), np.float32); k[L // 2, :] = 1
    M = cv2.getRotationMatrix2D((L / 2 - 0.5, L / 2 - 0.5), rng.uniform(0, 180), 1)
    k = cv2.warpAffine(k, M, (L, L)); k /= max(k.sum(), 1e-6)
    return cv2.filter2D(sc["img"], -1, k)


def a_defocus(rng, s, sc):
    r = 1 + int(1.6 * s)
    k = np.zeros((2 * r + 1, 2 * r + 1), np.float32); cv2.circle(k, (r, r), r, 1, -1); k /= k.sum()
    return cv2.filter2D(sc["img"], -1, k)


def a_low_light_noise(rng, s, sc):
    gain = 1 + 0.6 * s
    img = sc["img"] * gain
    return img + rng.normal(0, 0.02 * s, img.shape).astype(np.float32) * gain


def a_ir_bloom(rng, s, sc):
    img = sc["img"].copy()
    hot = np.clip((img[..., 0] - np.percentile(img[..., 0], 99)) * 6, 0, 1)
    b = cv2.GaussianBlur(hot, (0, 0), 2 + 3 * s)
    return img + b[..., None] * 0.35 * s


def a_hot_decoys(rng, s, sc):
    """Flares / hand warmers on balloons: hot points around the target (thermal)."""
    img = sc["img"].copy()
    cx, cy = sc["c"]
    for _ in range(int(s) + 1):
        x, y = cx + rng.normal(0, 90), cy + rng.normal(0, 70)
        img += _blob(H, W, x, y, rng.uniform(1.5, 4))[..., None] * rng.uniform(0.6, 1.0)
    return img


def a_camo_sky(rng, s, sc):
    """Sky-matched paint: pull the target's pixels toward the local sky colour."""
    a = min(0.95, 0.18 * s)
    m = sc["mask"][..., None] * a
    return sc["img"] * (1 - m) + sc["bg"] * m


def a_bird_shell(rng, s, sc):
    """A bird-shaped shroud over the drone: render a bird 1.2-2x the drone at the same spot."""
    if sc["main"] not in DRONE:
        return sc["img"]
    img = sc["img"].copy()
    cx, cy = sc["c"]
    R.render_object(rng, "bird", sc["px"] * (1.1 + 0.2 * s), sc["mode"], img, cx, cy)
    return img


def a_adv_patch_placeholder(rng, s, sc):
    """Placeholder for an optimised adversarial patch: a high-contrast random texture covering
    (10*s)% of the target box. An *optimised* patch needs model gradients: see pgd_eval()."""
    if not sc["objs"]:
        return sc["img"]
    img = sc["img"].copy()
    x0, y0, x1, y1 = sc["objs"][0]["box"]
    w, h = max(2, int((x1 - x0) * np.sqrt(0.1 * s))), max(2, int((y1 - y0) * np.sqrt(0.1 * s)))
    px0, py0 = int((x0 + x1 - w) / 2), int((y0 + y1 - h) / 2)
    tex = (rng.random((max(1, h // 2), max(1, w // 2), 3)) > 0.5).astype(np.float32)
    tex = cv2.resize(tex, (w, h), interpolation=cv2.INTER_NEAREST)
    img[py0:py0 + h, px0:px0 + w] = tex[: max(0, min(h, H - py0)), : max(0, min(w, W - px0))]
    return img


ATTACK_FN = {"glare": a_glare, "laser_dazzle": a_laser_dazzle, "fog": a_fog, "rain": a_rain, "snow": a_snow,
             "motion_blur": a_motion_blur, "defocus": a_defocus, "low_light_noise": a_low_light_noise,
             "ir_bloom": a_ir_bloom, "hot_decoys": a_hot_decoys, "camo_sky": a_camo_sky, "bird_shell": a_bird_shell,
             "adv_patch_placeholder": a_adv_patch_placeholder, "jpeg": None}


def attacked_frame(rng, sc, name, s):
    """Returns BGR uint8 frame with attack `name` at severity s (0 = clean)."""
    if s == 0 or name == "clean":
        return finish(rng, sc["img"], sc["mode"])
    if name == "jpeg":
        out = finish(rng, sc["img"], sc["mode"])
        q = int([30, 20, 12, 8, 5][int(s) - 1])
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return finish(rng, np.clip(ATTACK_FN[name](rng, s, sc), 0, 4), sc["mode"])


# ======================================================================== hardening recipe
AUGMENT_RECIPE = {
    "why": "Each entry closes a gap the benchmark measures. Apply on top of the usual YOLO augmentations "
           "(mosaic, HSV jitter, flips) to synthetic AND real frames. Keep 30-40% of frames clean.",
    "p_any": 0.6,
    "severity_range": [1, 3],          # train at 1-3, evaluate at 1-5 so severity 4-5 tests generalisation
    "weights": {"glare": 1.0, "laser_dazzle": 0.5, "fog": 1.0, "rain": 0.8, "snow": 0.4, "jpeg": 1.0,
                "motion_blur": 1.0, "defocus": 0.8, "low_light_noise": 1.0, "ir_bloom": 0.6, "hot_decoys": 0.6,
                "camo_sky": 0.8, "bird_shell": 0.4},
    "hard_negatives": "add birds, insects near the lens, bags and thermal flares as labelled negatives / class 'other'",
    "ultralytics_hyp": {"hsv_h": 0.01, "hsv_s": 0.5, "hsv_v": 0.5, "degrees": 5, "scale": 0.6, "mosaic": 1.0,
                        "mixup": 0.1, "copy_paste": 0.2, "close_mosaic": 10},
    "adversarial_training": "after the model converges, fine-tune 5-10 epochs on PGD examples (eps 2/255, 7 steps) "
                            "mixed 1:1 with clean frames; re-run pgd_eval to confirm the drop shrinks",
}


def harden_augment(rng, scene, recipe=AUGMENT_RECIPE):
    """Training-time augmentation: returns a BGR frame (labels unchanged). Call with a scene from
    make_scene(); for real frames use the pixel-level attacks on the float image directly."""
    if rng.random() > recipe["p_any"]:
        return attacked_frame(rng, scene, "clean", 0)
    names = [n for n in recipe["weights"] if not (n in THERMAL_ONLY and scene["mode"] != "thermal")
             and not (n in VISIBLE_ONLY and scene["mode"] == "thermal")]
    w = np.array([recipe["weights"][n] for n in names]); w /= w.sum()
    name = rng.choice(names, p=w)
    return attacked_frame(rng, scene, name, rng.integers(recipe["severity_range"][0], recipe["severity_range"][1] + 1))


# ======================================================================== detectors
class Detector:
    """Interface: detect(bgr uint8 HxWx3) -> list of {"cls": str, "conf": float, "box": [x0,y0,x1,y1]}."""

    def detect(self, bgr):
        raise NotImplementedError


class OnnxYoloDetector(Detector):
    """YOLOv8/YOLO11-style ONNX export (output 1 x (4+nc) x N). Needs onnxruntime."""

    def __init__(self, path, classes=R.CLASSES, imgsz=640, conf=0.25, iou=0.5):
        import onnxruntime as ort   # noqa: deferred import, GPU box only
        self.s = ort.InferenceSession(path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.inp = self.s.get_inputs()[0].name
        self.classes, self.imgsz, self.conf, self.iou = classes, imgsz, conf, iou

    def detect(self, bgr):
        h, w = bgr.shape[:2]; r = self.imgsz / max(h, w)
        nh, nw = int(round(h * r)), int(round(w * r))
        pad = np.full((self.imgsz, self.imgsz, 3), 114, np.uint8)
        pad[:nh, :nw] = cv2.resize(bgr, (nw, nh))
        x = pad[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.s.run(None, {self.inp: x})[0][0]          # (4+nc, N)
        boxes, scores = out[:4].T, out[4:].T
        cls = scores.argmax(1); conf = scores.max(1)
        k = conf >= self.conf
        boxes, cls, conf = boxes[k], cls[k], conf[k]
        xyxy = np.c_[boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2,
                     boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2] / r
        idx = cv2.dnn.NMSBoxes([[float(a), float(b), float(c - a), float(d - b)] for a, b, c, d in xyxy],
                               conf.tolist(), self.conf, self.iou)
        idx = np.array(idx).reshape(-1)
        return [{"cls": self.classes[int(cls[i])], "conf": float(conf[i]), "box": xyxy[i].tolist()} for i in idx]


class ReferenceDetector(Detector):
    """CPU baseline: proposals = local-contrast blobs, classifier = gradient boosting on crop features."""

    def __init__(self):
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.clf = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08)
        self.labels = ["background", "drone", "drone_payload", "bird", "other"]

    @staticmethod
    def proposals(bgr, k=5.0, max_n=12):
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        base = cv2.medianBlur((g * 255).astype(np.uint8), 31).astype(np.float32) / 255
        res = cv2.GaussianBlur(np.abs(g - base), (0, 0), 1.2)
        thr = np.median(res) + k * 1.4826 * np.median(np.abs(res - np.median(res))) + 0.01
        n, lab, st, _ = cv2.connectedComponentsWithStats((res > thr).astype(np.uint8), 8)
        boxes = []
        for i in np.argsort(-st[1:, cv2.CC_STAT_AREA])[:max_n] + 1:
            x, y, w, h, a = st[i]
            if a < 3 or w > W * 0.6:
                continue
            m = max(2, int(0.2 * max(w, h)))
            boxes.append([max(0, x - m), max(0, y - m), min(W, x + w + m), min(H, y + h + m)])
        return boxes, g, res

    @staticmethod
    def features(g, res, b):
        x0, y0, x1, y1 = [int(v) for v in b]
        c = g[y0:y1, x0:x1]; r = res[y0:y1, x0:x1]
        p = cv2.resize(c, (12, 12), interpolation=cv2.INTER_AREA).ravel()
        p = (p - p.mean()) / (p.std() + 1e-3)
        gx, gy = cv2.Sobel(c, cv2.CV_32F, 1, 0), cv2.Sobel(c, cv2.CV_32F, 0, 1)
        ang = (np.arctan2(gy, gx) % np.pi); mag = np.hypot(gx, gy)
        hog = np.histogram(ang, bins=8, range=(0, np.pi), weights=mag)[0]; hog = hog / (hog.sum() + 1e-6)
        w, h = x1 - x0, y1 - y0
        return np.r_[p, hog, w, h, w / max(h, 1), r.max(), r.mean(), c.mean(), c.std(), (c > c.mean() + 2 * c.std()).mean()]

    def _xy(self, frames):
        X, y = [], []
        for bgr, objs in frames:
            boxes, g, res = self.proposals(bgr)
            for b in boxes:
                lab = "background"
                for o in objs:
                    if iou(b, o["box"]) > 0.1 or _centre_in(o["box"], b):
                        lab = o["cls"]; break
                X.append(self.features(g, res, b)); y.append(self.labels.index(lab))
        return np.array(X), np.array(y)

    def fit(self, frames):
        X, y = self._xy(frames)
        self.clf.fit(X, y)
        return self

    def detect(self, bgr):
        boxes, g, res = self.proposals(bgr)
        if not boxes:
            return []
        P = self.clf.predict_proba(np.array([self.features(g, res, b) for b in boxes]))
        cl = list(self.clf.classes_)
        out = []
        for b, p in zip(boxes, P):
            full = np.zeros(len(self.labels)); full[cl] = p
            k = int(full[1:].argmax()) + 1
            if full[k] > 0.3:
                out.append({"cls": self.labels[k], "conf": float(full[1] + full[2]) if k in (1, 2) else float(full[k]), "box": b})
        return out


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def _centre_in(gt, b):
    cx, cy = (gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2
    return b[0] <= cx <= b[2] and b[1] <= cy <= b[3]


# ======================================================================== benchmark
def score_frame(dets, objs):
    """Small-target matching: IoU >= 0.1 or GT centre inside the detection box."""
    gts = [o for o in objs if o["cls"] in DRONE]
    hit, conf = 0, []
    used = set()
    for o in gts:
        best = None
        for i, d in enumerate(dets):
            if i in used or d["cls"] not in DRONE:
                continue
            if iou(d["box"], o["box"]) >= 0.1 or _centre_in(o["box"], d["box"]):
                if best is None or d["conf"] > dets[best]["conf"]:
                    best = i
        if best is not None:
            used.add(best); hit += 1; conf.append(dets[best]["conf"])
        else:
            conf.append(0.0)
    fp = sum(1 for i, d in enumerate(dets) if d["cls"] in DRONE and i not in used
             and not any(iou(d["box"], o["box"]) >= 0.1 or _centre_in(o["box"], d["box"]) for o in gts))
    return hit, len(gts), fp, conf


def run_benchmark(det, n_frames, seed=7, severities=(1, 2, 3, 4, 5), attacks=None):
    attacks = attacks or ["clean"] + PIXEL_ATTACKS + OBJECT_ATTACKS
    attacks = list(dict.fromkeys(attacks))
    res = {}
    scenes = []
    srng = np.random.default_rng(seed)
    for i in range(n_frames):
        mode = "thermal" if i % 3 == 0 else "night"
        scenes.append(make_scene(srng, mode=mode, cls="drone" if i % 2 == 0 else "drone_payload"))
    for name in attacks:
        for s in ([0] if name == "clean" else severities):
            rng = np.random.default_rng([seed, zlib.crc32(name.encode()) % 10_000, s])
            H_, N_, FP_, C_, nf = 0, 0, 0, [], 0
            for sc in scenes:
                if (name in THERMAL_ONLY and sc["mode"] != "thermal") or (name in VISIBLE_ONLY and sc["mode"] == "thermal"):
                    continue
                img = attacked_frame(rng, sc, name, s)
                h_, n_, fp, c = score_frame(det.detect(img), sc["objs"])
                H_ += h_; N_ += n_; FP_ += fp; C_ += c; nf += 1
            res[f"{name}@{s}"] = {"recall": H_ / max(N_, 1), "fp_per_frame": FP_ / max(nf, 1),
                                  "mean_conf": float(np.mean(C_)) if C_ else 0.0, "frames": nf}
    return res


# ======================================================================== gradient attacks (GPU box)
def pgd_eval(score_fn, images, eps_list=(1 / 255, 2 / 255, 4 / 255, 8 / 255), steps=10, alpha=None, fgsm=False):
    """Standard L-inf robustness evaluation. score_fn: torch (B,3,H,W) in [0,1] -> (B,) drone score
    (e.g. max drone-class confidence from the model's raw head). Returns mean score per eps after an
    evasion attack that tries to lower the drone score. Requires torch; returns None otherwise."""
    try:
        import torch
    except ImportError:
        print("pgd_eval: torch not installed; run on the GPU box (see training/).")
        return None
    x0 = torch.as_tensor(np.stack(images)).float()
    out = {"clean": float(score_fn(x0).mean())}
    for eps in eps_list:
        a = alpha or (eps if fgsm else eps / 4)
        x = x0.clone()
        for _ in range(1 if fgsm else steps):
            x.requires_grad_(True)
            loss = score_fn(x).sum()
            g, = torch.autograd.grad(loss, x)
            x = (x.detach() - a * g.sign()).clamp(x0 - eps, x0 + eps).clamp(0, 1)
        out[f"eps={eps * 255:.0f}/255"] = float(score_fn(x).mean())
    return out


# ======================================================================== report
def write_report(results, out_dir, meta):
    os.makedirs(out_dir, exist_ok=True)
    json.dump({"meta": meta, "results": results}, open(os.path.join(out_dir, "vision_robustness.json"), "w"), indent=1)
    names = [k for k in next(iter(results.values())) if k != "clean@0"]
    attacks = list(dict.fromkeys(k.split("@")[0] for k in names))
    L = [f"# Vision robustness benchmark ({meta['detector']})", "",
         f"{meta['frames']} drone frames per cell (1/3 thermal), rendered by render.py; match = IoU≥0.1 or centre-in-box. "
         f"Recall = share of drones still detected. {meta['note']}", ""]
    for tag, res in results.items():
        L += [f"## {tag}", "", f"Clean recall {res['clean@0']['recall']:.0%}, false drone boxes/frame {res['clean@0']['fp_per_frame']:.2f}", "",
              "| Attack | sev 1 | sev 2 | sev 3 | sev 4 | sev 5 |", "|---|---|---|---|---|---|"]
        for a in attacks:
            L.append(f"| {a} | " + " | ".join(f"{res[f'{a}@{s}']['recall']:.0%}" for s in range(1, 6)) + " |")
        L.append("")
    if len(results) == 2:
        (ta, ra), (tb, rb) = list(results.items())
        L += ["## Effect of the hardening recipe (recall, mean over severities 1-5)", "", f"| Attack | {ta} | {tb} | change |", "|---|---|---|---|"]
        for a in attacks:
            ma = np.mean([ra[f"{a}@{s}"]["recall"] for s in range(1, 6)]); mb = np.mean([rb[f"{a}@{s}"]["recall"] for s in range(1, 6)])
            L.append(f"| {a} | {ma:.0%} | {mb:.0%} | {100 * (mb - ma):+.0f} pts |")
    open(os.path.join(out_dir, "vision_robustness.md"), "w").write("\n".join(L) + "\n")
    _chart(results, attacks, out_dir)


def _chart(results, attacks, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(attacks)); wdt = 0.8 / len(results)
    for i, (tag, res) in enumerate(results.items()):
        y = [np.mean([res[f"{a}@{s}"]["recall"] for s in (3, 4, 5)]) * 100 for a in attacks]
        ax.bar(x + i * wdt - 0.4 + wdt / 2, y, wdt, label=tag, color=["#9aa7b4", "#2a6fdb"][i % 2])
        ax.axhline(res["clean@0"]["recall"] * 100, color=["#9aa7b4", "#2a6fdb"][i % 2], ls="--", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(attacks, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Drone recall at severity 3-5 (%)"); ax.set_ylim(0, 100)
    ax.set_title("Vision robustness: recall under attack (dashed = clean)")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "vision_robustness.png"), dpi=130); plt.close(fig)


def save_examples(out_dir, seed=3):
    """One contact sheet showing each attack at severity 3 on the same scene."""
    rng = np.random.default_rng(seed)
    tiles = []
    for mode in ("night", "thermal"):
        sc = make_scene(rng, mode=mode, cls="drone_payload", px=40)
        names = ["clean"] + [n for n in PIXEL_ATTACKS + OBJECT_ATTACKS
                             if not (n in THERMAL_ONLY and mode != "thermal") and not (n in VISIBLE_ONLY and mode == "thermal")]
        for n in names:
            im = attacked_frame(np.random.default_rng(1), sc, n, 0 if n == "clean" else 3)
            if sc["objs"]:
                x0, y0, x1, y1 = sc["objs"][0]["box"]
                cxy = ((x0 + x1) // 2, (y0 + y1) // 2)
                crop = im[max(0, cxy[1] - 96):cxy[1] + 96, max(0, cxy[0] - 128):cxy[0] + 128]
                crop = cv2.resize(crop, (256, 192))
            else:
                crop = cv2.resize(im, (256, 192))
            cv2.putText(crop, f"{mode[:3]} {n}", (6, 186), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(crop)
    cols = 6
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(os.path.join(out_dir, "vision_attacks_examples.png"), np.vstack(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--train", type=int, default=500, help="frames to train the reference detector on")
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    if not (1 <= a.frames <= 5000 and 10 <= a.train <= 20000):
        ap.error("frames/train out of range")
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    save_examples(a.out)
    if a.onnx:
        results = {"trained model": run_benchmark(OnnxYoloDetector(a.onnx), a.frames, a.seed)}
        meta = {"detector": f"ONNX {os.path.basename(a.onnx)}", "frames": a.frames, "note": ""}
    else:
        rng = np.random.default_rng(a.seed + 100)
        clean, aug = [], []
        for i in range(a.train):
            sc = make_scene(rng)
            clean.append((attacked_frame(rng, sc, "clean", 0), sc["objs"]))
            aug.append((harden_augment(rng, sc), sc["objs"]))
        print(f"rendered {2 * a.train} training frames in {time.time() - t0:.0f} s"); sys.stdout.flush()
        d_clean = ReferenceDetector().fit(clean)
        d_aug = ReferenceDetector().fit(aug)
        results = {"reference, clean training": run_benchmark(d_clean, a.frames, a.seed),
                   "reference, hardening recipe": run_benchmark(d_aug, a.frames, a.seed)}
        meta = {"detector": "CPU reference detector (not the product model)", "frames": a.frames,
                "note": f"Reference detector trained on {a.train} rendered frames, once clean and once with harden_augment(). "
                        "Absolute numbers say little about a trained YOLO; the per-attack drops and the recipe's effect are the point."}
    write_report(results, a.out, meta)
    print(open(os.path.join(a.out, "vision_robustness.md")).read())
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
