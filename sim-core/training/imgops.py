"""Image utilities used by build_mix: perceptual hashing for de-duplication, SAHI-style
tiling for tiny targets, and photometric augmentations for night / IR / sensor effects.
Only numpy + OpenCV, so they run (and are tested) anywhere."""
from __future__ import annotations

import numpy as np
import cv2

# ------------------------------------------------------------------ dedupe


def dhash(img: np.ndarray, size: int = 8) -> int:
    """Difference hash: 64-bit for size=8. Robust to compression/brightness, not to content change."""
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = cv2.resize(g, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (s[:, 1:] > s[:, :-1]).flatten()
    return int(sum(1 << i for i, b in enumerate(bits) if b))


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def iou(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def yolo_to_xyxy(lines: list[str]) -> list[tuple[int, float, float, float, float]]:
    out = []
    for ln in lines:
        t = ln.split()
        if len(t) >= 5:
            c, cx, cy, w, h = int(float(t[0])), *map(float, t[1:5])
            out.append((c, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return out


def same_labels(a: list[str], b: list[str], thr: float = 0.7) -> bool:
    A, B = yolo_to_xyxy(a), yolo_to_xyxy(b)
    if len(A) != len(B):
        return False
    used = set()
    for ca, *ba in A:
        best = max(((iou(ba, bb), j) for j, (cb, *bb) in enumerate(B) if cb == ca and j not in used),
                   default=(0.0, -1))
        if best[0] < thr:
            return False
        used.add(best[1])
    return True


# ------------------------------------------------------------------ tiling


def tile_image(img: np.ndarray, labels: list[str], tile: int = 1280, overlap: float = 0.2,
               min_visible: float = 0.5, keep_empty: float = 0.1, rng: np.random.Generator | None = None):
    """Slice a large frame into overlapping tiles. Yields (tile_img, yolo_lines, (x0, y0)).
    Boxes cut by a tile edge are kept if at least min_visible of their area is inside."""
    rng = rng or np.random.default_rng(0)
    H, W = img.shape[:2]
    if max(H, W) <= tile * 1.25:
        yield img, labels, (0, 0)
        return
    step = max(1, int(tile * (1 - overlap)))
    xs = list(range(0, max(1, W - tile), step)) + [max(0, W - tile)]
    ys = list(range(0, max(1, H - tile), step)) + [max(0, H - tile)]
    boxes = [(c, x0 * W, y0 * H, x1 * W, y1 * H) for c, x0, y0, x1, y1 in yolo_to_xyxy(labels)]
    for ty in sorted(set(ys)):
        for tx in sorted(set(xs)):
            tw, th = min(tile, W - tx), min(tile, H - ty)
            out = []
            for c, x0, y0, x1, y1 in boxes:
                ix0, iy0, ix1, iy1 = max(x0, tx), max(y0, ty), min(x1, tx + tw), min(y1, ty + th)
                if ix1 <= ix0 or iy1 <= iy0:
                    continue
                area = (x1 - x0) * (y1 - y0)
                if area <= 0 or (ix1 - ix0) * (iy1 - iy0) / area < min_visible:
                    continue
                cx, cy = ((ix0 + ix1) / 2 - tx) / tw, ((iy0 + iy1) / 2 - ty) / th
                out.append(f"{c} {cx:.6f} {cy:.6f} {(ix1 - ix0) / tw:.6f} {(iy1 - iy0) / th:.6f}")
            if out or rng.random() < keep_empty:
                yield img[ty:ty + th, tx:tx + tw], out, (tx, ty)


# ------------------------------------------------------------------ augmentation


def aug_night(img, rng):
    f = img.astype(np.float32) / 255.0
    f = np.power(f, rng.uniform(1.8, 3.0)) * rng.uniform(0.35, 0.7)
    f[..., 0] *= rng.uniform(1.0, 1.25)  # slight blue cast (BGR)
    f += rng.normal(0, rng.uniform(0.01, 0.04), f.shape)
    return np.clip(f * 255, 0, 255).astype(np.uint8)


def aug_ir(img, rng):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = cv2.GaussianBlur(g, (0, 0), rng.uniform(0.6, 1.4))
    if rng.random() < 0.5:
        g = 255 - g  # white-hot vs black-hot
    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
    g = np.clip(g.astype(np.float32) + rng.normal(0, 4, g.shape), 0, 255).astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def aug_motion_blur(img, rng):
    k = int(rng.integers(3, 11))
    ker = np.zeros((k, k), np.float32)
    ker[k // 2, :] = 1.0 / k
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), float(rng.uniform(0, 180)), 1.0)
    ker = cv2.warpAffine(ker, M, (k, k))
    ker /= max(ker.sum(), 1e-6)
    return cv2.filter2D(img, -1, ker)


def aug_noise(img, rng):
    return np.clip(img.astype(np.float32) + rng.normal(0, rng.uniform(4, 14), img.shape), 0, 255).astype(np.uint8)


def aug_glare(img, rng):
    H, W = img.shape[:2]
    cx, cy = rng.uniform(0, W), rng.uniform(0, H)
    r = rng.uniform(0.08, 0.3) * max(H, W)
    yy, xx = np.mgrid[0:H, 0:W]
    g = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r)))[..., None]
    tint = np.array([0.55, 0.8, 1.0], np.float32)  # sodium floodlight, BGR
    out = img.astype(np.float32) + g * tint * rng.uniform(80, 200)
    return np.clip(out, 0, 255).astype(np.uint8)


def aug_jpeg(img, rng):
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(20, 60))])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else img


AUGS = {"night": aug_night, "ir": aug_ir, "motion_blur": aug_motion_blur, "noise": aug_noise,
        "glare": aug_glare, "jpeg": aug_jpeg}


def augment(img: np.ndarray, rng: np.random.Generator, n_ops: int = 2, ops: list[str] | None = None):
    names = ops or list(AUGS)
    chosen = list(rng.choice(names, size=min(n_ops, len(names)), replace=False))
    out = img
    for name in chosen:
        out = AUGS[name](out, rng)
    return out, chosen
