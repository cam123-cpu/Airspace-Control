"""ONNX inference for an exported Ultralytics YOLO detector, plus the adapter that plugs it
into the simulator's camera-classifier interface.

The simulator (sensors.py) expects a classifier with
    score(rng, kind, px, has_payload=False, obj_id=None) -> P(drone) in [0, 1]
and ships RenderedModelClassifier(predict_fn), where predict_fn(bgr_image) -> P(drone).

    from training.onnx_detector import make_sim_classifier
    params = {"classifier": make_sim_classifier("models/rgb.onnx")}
    # pass params into sim / evaluate as described in README "Swapping in a real vision model"

Works with onnxruntime (CPU or GPU). If onnxruntime is missing, constructing the detector
raises an ImportError that says how to install it; nothing else in the package needs it.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

DRONE_CLASSES = (0, 1)        # drone, drone_payload
FLOOR = 0.02                  # P(drone) reported when nothing is detected


def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    out = np.full((size, size, 3), 114, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    out[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, r, (left, top)


def nms(boxes: np.ndarray, scores: np.ndarray, thr: float) -> list[int]:
    if len(boxes) == 0:
        return []
    x0, y0, x1, y1 = boxes.T
    areas = (x1 - x0) * (y1 - y0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx0, yy0 = np.maximum(x0[i], x0[order[1:]]), np.maximum(y0[i], y0[order[1:]])
        xx1, yy1 = np.minimum(x1[i], x1[order[1:]]), np.minimum(y1[i], y1[order[1:]])
        inter = np.clip(xx1 - xx0, 0, None) * np.clip(yy1 - yy0, 0, None)
        ov = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][ov < thr]
    return keep


def decode_yolov8(out: np.ndarray, conf: float, iou_thr: float, nc: int = 4) -> list[tuple[int, float, np.ndarray]]:
    """out: (1, 4 + nc, N) raw head output (cx, cy, w, h in input pixels, then class scores)."""
    p = np.asarray(out)
    if p.ndim == 3:
        p = p[0]
    if p.shape[0] != 4 + nc and p.shape[1] == 4 + nc:  # some exports are (N, 4+nc)
        p = p.T
    if p.shape[0] != 4 + nc:
        raise ValueError(f"unexpected head output shape {p.shape} for {nc} classes")
    xywh, cls_scores = p[:4].T, p[4:].T
    cls = cls_scores.argmax(1)
    sc = cls_scores[np.arange(len(cls)), cls]
    m = sc >= conf
    if not np.any(m):
        return []
    xywh, cls, sc = xywh[m], cls[m], sc[m]
    boxes = np.stack([xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2,
                      xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2], 1)
    out_ = []
    for c in np.unique(cls):  # class-wise NMS
        idx = np.where(cls == c)[0]
        for k in nms(boxes[idx], sc[idx], iou_thr):
            j = idx[k]
            out_.append((int(c), float(sc[j]), boxes[j]))
    return sorted(out_, key=lambda d: -d[1])


class OnnxYoloDetector:
    def __init__(self, model_path: str | Path, imgsz: int = 1280, conf: float = 0.25, iou: float = 0.5,
                 tile: int = 0, providers: list[str] | None = None, session=None):
        self.imgsz, self.conf, self.iou, self.tile = int(imgsz), float(conf), float(iou), int(tile)
        if session is not None:          # injected (tests, custom runtimes)
            self.sess = session
        else:
            p = Path(model_path)
            if not p.is_file() or p.suffix.lower() != ".onnx":
                raise FileNotFoundError(f"ONNX model not found: {p}")
            try:
                import onnxruntime as ort
            except ImportError:
                raise ImportError("pip install onnxruntime (or onnxruntime-gpu) to run the ONNX detector") from None
            prov = providers or [x for x in ("CUDAExecutionProvider", "CPUExecutionProvider")
                                 if x in ort.get_available_providers()]
            self.sess = ort.InferenceSession(str(p), providers=prov)
        self.input_name = self.sess.get_inputs()[0].name

    def _run(self, img: np.ndarray):
        lb, r, (dx, dy) = letterbox(img, self.imgsz)
        x = lb[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.sess.run(None, {self.input_name: np.ascontiguousarray(x)})[0]
        dets = []
        for c, s, b in decode_yolov8(out, self.conf, self.iou):
            b = (b - np.array([dx, dy, dx, dy])) / r
            dets.append((c, s, tuple(float(v) for v in b)))
        return dets

    def detect(self, img: np.ndarray) -> list[tuple[int, float, tuple[float, float, float, float]]]:
        """BGR uint8 image -> [(class_id, confidence, (x0, y0, x1, y1) pixels)]."""
        if img is None or img.ndim != 3:
            raise ValueError("expected a BGR uint8 image")
        H, W = img.shape[:2]
        if not self.tile or max(H, W) <= self.tile * 1.25:
            return self._run(img)
        step = int(self.tile * 0.8)
        dets = []
        for y in sorted(set(list(range(0, max(1, H - self.tile), step)) + [max(0, H - self.tile)])):
            for x in sorted(set(list(range(0, max(1, W - self.tile), step)) + [max(0, W - self.tile)])):
                for c, s, b in self._run(img[y:y + self.tile, x:x + self.tile]):
                    dets.append((c, s, (b[0] + x, b[1] + y, b[2] + x, b[3] + y)))
        if not dets:
            return []
        B = np.array([d[2] for d in dets])
        S = np.array([d[1] for d in dets])
        return [dets[i] for i in nms(B, S, self.iou)]  # merge across tile overlaps

    def p_drone(self, img: np.ndarray) -> float:
        conf = [s for c, s, _ in self.detect(img) if c in DRONE_CLASSES]
        return float(max(conf, default=FLOOR))

    __call__ = p_drone


def make_sim_classifier(model_path: str | Path, mode: str | None = None, **kw):
    """Return the simulator's RenderedModelClassifier wired to this ONNX model."""
    import sys
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from sensors import RenderedModelClassifier
    det = OnnxYoloDetector(model_path, **kw)
    return RenderedModelClassifier(det.p_drone, mode=mode)
