"""Score a detector on the REAL-only test split built by build_mix.py.

    python -m training.evaluate_real --mix /data/mix_rgb --weights runs/rgb/weights/best.pt
    python -m training.evaluate_real --mix /data/mix_rgb --onnx models/rgb.onnx
    python -m training.evaluate_real --mix /data/mix_rgb --predictions preds.jsonl

Reports AP@0.5 per class and for "any drone" (drone + drone_payload merged), recall at 90%
precision, AP broken out by target size in pixels (the number that matters for a PTZ at
range), per source/context, and how often an empty frame produces a drone detection.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .common import CLASSES
from .imgops import iou, yolo_to_xyxy

SIZE_BINS = [(0, 8), (8, 16), (16, 32), (32, 64), (64, 1e9)]
DRONE_IDS = {0, 1}


def load_split(mix: Path, split: str = "test"):
    meta = {}
    sp = mix / "splits.jsonl"
    if sp.is_file():
        for line in sp.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            if d.get("split") == split:
                meta[Path(d["out_image"]).stem] = d
    items = []
    for ip in sorted((mix / "images" / split).glob("*.jpg")):
        lp = mix / "labels" / split / (ip.stem + ".txt")
        with Image.open(ip) as im:
            W, H = im.size
        gts = yolo_to_xyxy(lp.read_text(encoding="utf-8").splitlines() if lp.exists() else [])
        m = meta.get(ip.stem, {})
        items.append({"stem": ip.stem, "path": ip, "W": W, "H": H, "gts": gts,
                      "source": m.get("source", "?"), "context": m.get("context", "")})
    return items


def ap_from(tps: list[tuple[float, int]], n_gt: int) -> tuple[float, float]:
    """All-point interpolated AP and recall at precision >= 0.9."""
    if n_gt == 0:
        return float("nan"), float("nan")
    if not tps:
        return 0.0, 0.0
    tps = sorted(tps, key=lambda x: -x[0])
    tp = np.cumsum([t for _, t in tps])
    fp = np.cumsum([1 - t for _, t in tps])
    rec = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1e-9)
    mrec = np.concatenate([[0], rec, [rec[-1]]])
    mpre = np.concatenate([[1], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    r90 = float(rec[prec >= 0.9].max()) if np.any(prec >= 0.9) else 0.0
    return ap, r90


def match(items, preds, cls_set: set[int], iou_thr=0.5, size_bin=None, item_filter=None):
    """Greedy matching per image for the classes in cls_set (treated as one class)."""
    tps, n_gt = [], 0
    for it in items:
        if item_filter and not item_filter(it):
            continue
        W, H = it["W"], it["H"]
        gts = [g for g in it["gts"] if g[0] in cls_set]

        def px(b):
            return max((b[2] - b[0]) * W, (b[3] - b[1]) * H)
        in_bin = [size_bin is None or size_bin[0] <= px(g[1:]) < size_bin[1] for g in gts]
        n_gt += sum(in_bin)
        dets = sorted([d for d in preds.get(it["stem"], []) if d[0] in cls_set], key=lambda d: -d[1])
        used = set()
        for d in dets:
            best, bj = 0.0, -1
            for j, g in enumerate(gts):
                if j in used:
                    continue
                v = iou(d[2:6], g[1:])
                if v > best:
                    best, bj = v, j
            if best >= iou_thr:
                used.add(bj)
                if in_bin[bj]:
                    tps.append((d[1], 1))
            elif size_bin is None or size_bin[0] <= px(d[2:6]) < size_bin[1]:
                tps.append((d[1], 0))
    return tps, n_gt


def evaluate(items, preds, conf_alarm: float = 0.5) -> dict:
    res = {"n_images": len(items), "per_class": {}, "by_size_px": {}, "by_source": {}, "by_context": {}}
    for i, c in enumerate(CLASSES):
        ap, r90 = ap_from(*match(items, preds, {i}))
        res["per_class"][c] = {"AP50": ap, "recall_at_p90": r90}
    ap, r90 = ap_from(*match(items, preds, DRONE_IDS))
    res["any_drone"] = {"AP50": ap, "recall_at_p90": r90}
    for lo, hi in SIZE_BINS:
        tp, n = match(items, preds, DRONE_IDS, size_bin=(lo, hi))
        ap, r90 = ap_from(tp, n)
        res["by_size_px"][f"{lo}-{'inf' if hi > 1e8 else int(hi)}"] = {"AP50": ap, "recall_at_p90": r90, "n_gt": n}
    for key in ("source", "context"):
        for v in sorted({it[key] for it in items}):
            tp, n = match(items, preds, DRONE_IDS, item_filter=lambda it, v=v, k=key: it[k] == v)
            ap, _ = ap_from(tp, n)
            res[f"by_{key}"][v or "(none)"] = {"AP50": ap, "n_gt": n}
    empty = [it for it in items if not any(g[0] in DRONE_IDS for g in it["gts"])]
    alarms = sum(1 for it in empty if any(d[0] in DRONE_IDS and d[1] >= conf_alarm for d in preds.get(it["stem"], [])))
    birds_only = [it for it in empty if any(g[0] == 2 for g in it["gts"])]
    bird_alarms = sum(1 for it in birds_only if any(d[0] in DRONE_IDS and d[1] >= conf_alarm
                                                   for d in preds.get(it["stem"], [])))
    res["false_drone_on_no_drone_frames"] = {"conf": conf_alarm, "frames": len(empty),
                                             "rate": alarms / len(empty) if empty else float("nan"),
                                             "bird_frames": len(birds_only),
                                             "bird_frame_rate": bird_alarms / len(birds_only) if birds_only else float("nan")}
    return res


def to_markdown(res: dict, title: str) -> str:
    f = lambda v: "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"
    L = [f"# {title}", "", f"Real-only test frames: {res['n_images']}", "",
         "| Class | AP@0.5 | Recall @ P≥0.9 |", "|---|---|---|"]
    for c, v in res["per_class"].items():
        L.append(f"| {c} | {f(v['AP50'])} | {f(v['recall_at_p90'])} |")
    L.append(f"| **any drone** | **{f(res['any_drone']['AP50'])}** | **{f(res['any_drone']['recall_at_p90'])}** |")
    L += ["", "Drone AP by target size (largest side, pixels):", "", "| Size px | GT | AP@0.5 | Recall @ P≥0.9 |",
          "|---|---|---|---|"]
    for k, v in res["by_size_px"].items():
        L.append(f"| {k} | {v['n_gt']} | {f(v['AP50'])} | {f(v['recall_at_p90'])} |")
    L += ["", "| Source | GT drones | AP@0.5 |", "|---|---|---|"]
    for k, v in res["by_source"].items():
        L.append(f"| {k} | {v['n_gt']} | {f(v['AP50'])} |")
    fa = res["false_drone_on_no_drone_frames"]
    L += ["", f"Frames with no drone that still produced a drone detection ≥ {fa['conf']}: {f(fa['rate'])} "
              f"of {fa['frames']}; on frames containing birds: {f(fa['bird_frame_rate'])} of {fa['bird_frames']}."]
    return "\n".join(L) + "\n"


def predict_ultralytics(weights: Path, items, imgsz: int, sahi_tile: int = 0):
    from ultralytics import YOLO  # imported lazily: not needed for --predictions / --onnx
    model = YOLO(str(weights))
    preds = {}
    if sahi_tile:
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction
        dm = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=str(weights), confidence_threshold=0.01)
        for it in items:
            r = get_sliced_prediction(str(it["path"]), dm, slice_height=sahi_tile, slice_width=sahi_tile,
                                      overlap_height_ratio=0.2, overlap_width_ratio=0.2, verbose=0)
            preds[it["stem"]] = [(o.category.id, o.score.value, o.bbox.minx / it["W"], o.bbox.miny / it["H"],
                                  o.bbox.maxx / it["W"], o.bbox.maxy / it["H"]) for o in r.object_prediction_list]
        return preds
    for it in items:
        r = model.predict(str(it["path"]), imgsz=imgsz, conf=0.01, verbose=False)[0]
        preds[it["stem"]] = [(int(c), float(s), *map(float, b)) for c, s, b in
                             zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxyn.tolist())]
    return preds


def predict_onnx(onnx: Path, items, imgsz: int, tile: int = 0):
    import cv2
    from .onnx_detector import OnnxYoloDetector
    det = OnnxYoloDetector(onnx, imgsz=imgsz, conf=0.01, tile=tile)
    out = {}
    for it in items:
        img = cv2.imread(str(it["path"]), cv2.IMREAD_COLOR)
        out[it["stem"]] = [(c, s, x0 / it["W"], y0 / it["H"], x1 / it["W"], y1 / it["H"])
                           for c, s, (x0, y0, x1, y1) in det.detect(img)]
    return out


def load_predictions(path: Path):
    preds = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)  # {"stem":..., "cls":0, "conf":0.9, "xyxyn":[...]}
            preds[d["stem"]].append((int(d["cls"]), float(d["conf"]), *map(float, d["xyxyn"])))
    return dict(preds)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mix", type=Path, required=True)
    ap.add_argument("--split", default="test")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--weights", type=Path)
    g.add_argument("--onnx", type=Path)
    g.add_argument("--predictions", type=Path)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--tile", type=int, default=0, help="sliced inference tile size for large frames")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    items = load_split(a.mix, a.split)
    if not items:
        raise SystemExit(f"no images in {a.mix}/images/{a.split}")
    if a.weights:
        preds = predict_ultralytics(a.weights, items, a.imgsz, a.tile)
    elif a.onnx:
        preds = predict_onnx(a.onnx, items, a.imgsz, a.tile)
    else:
        preds = load_predictions(a.predictions)
    res = evaluate(items, preds)
    out = a.out or a.mix / f"eval_{a.split}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    md = to_markdown(res, f"Real-only {a.split} evaluation")
    (out / "metrics.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
