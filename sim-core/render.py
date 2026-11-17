"""Procedural PTZ-camera frame renderer (640x512) with exact labels.

Modes: 'night' (low-light visible with sodium floodlight skyglow) and 'thermal'
(white-hot LWIR look). Targets are drawn as simple 3D models (projected), at a
pixel size derived from physical size, range and the camera's field of view,
then degraded with rolling-shutter skew, motion blur, defocus, sensor noise and
JPEG compression. Boxes are computed from the final rendered alpha mask, so
they stay aligned after blur.
"""
import numpy as np
import cv2

W, H = 640, 512
CLASSES = ["drone", "drone_payload", "bird", "other"]
SS = 4  # supersampling factor for sprite drawing


# ------------------------------------------------------------------ geometry
def _rot(yaw, pitch, roll=0.0):
    cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Ry = np.array([[cr, 0, sr], [0, 1, 0], [-sr, 0, cr]])
    return Rz @ Rx @ Ry


def _proj(pts, elev, ppm, c):
    """Orthographic view along +y tilted up by elev. Returns (N,2) pixel coords."""
    u = pts[:, 0]
    v = pts[:, 2] * np.cos(elev) - pts[:, 1] * np.sin(elev)
    return np.c_[c[0] + u * ppm, c[1] - v * ppm]


def _circle3(center, r, n=18, normal_z=True):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return center + np.c_[r * np.cos(a), r * np.sin(a), np.zeros(n)]


def _ellipsoid(a, b, c, n=80, rng=None):
    rng = rng or np.random.default_rng(0)
    v = rng.normal(size=(n, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v * np.array([a, b, c])


# ------------------------------------------------------------------ sprite canvas
class Canvas:
    """Supersampled premultiplied RGBA patch plus an additive light layer."""

    def __init__(self, size):
        self.n = size * SS
        self.P = np.zeros((self.n, self.n, 3), np.float32)
        self.A = np.zeros((self.n, self.n), np.float32)
        self.L = np.zeros((self.n, self.n, 3), np.float32)

    def _apply(self, mask, col, alpha):
        m = mask.astype(np.float32) / 255.0 * alpha
        self.P = self.P * (1 - m[..., None]) + np.asarray(col, np.float32) * m[..., None]
        self.A = self.A * (1 - m) + m

    def poly(self, pts, col, alpha=1.0, hull=False):
        p = np.round(pts * 16).astype(np.int32)
        if hull:
            p = cv2.convexHull(p).reshape(-1, 2)
        m = np.zeros((self.n, self.n), np.uint8)
        cv2.fillPoly(m, [p], 255, cv2.LINE_AA, shift=4)
        self._apply(m, col, alpha)

    def line(self, a, b, width, col, alpha=1.0):
        m = np.zeros((self.n, self.n), np.uint8)
        cv2.line(m, tuple(np.round(a * 16).astype(int)), tuple(np.round(b * 16).astype(int)), 255,
                 max(1, int(round(width))), cv2.LINE_AA, shift=4)
        self._apply(m, col, alpha)

    def light(self, pt, col, radius):
        m = np.zeros((self.n, self.n), np.float32)
        cv2.circle(m, tuple(np.round(pt).astype(int)), max(1, int(radius)), 1.0, -1, cv2.LINE_AA)
        m = cv2.GaussianBlur(m, (0, 0), max(1.0, radius * 1.5))
        m /= max(m.max(), 1e-6)
        self.L += m[..., None] * np.asarray(col, np.float32)

    def finish(self):
        k = self.n // SS
        P = cv2.resize(self.P, (k, k), interpolation=cv2.INTER_AREA)
        A = cv2.resize(self.A, (k, k), interpolation=cv2.INTER_AREA)
        L = cv2.resize(self.L, (k, k), interpolation=cv2.INTER_AREA)
        return P, A, L


# ------------------------------------------------------------------ appearance palettes
def _palette(rng, mode, kind):
    if mode == "thermal":
        g = lambda v: (v, v, v)
        if kind in ("drone", "drone_payload"):
            return {"body": g(rng.uniform(0.45, 0.6)), "motor": g(rng.uniform(0.85, 1.0)),
                    "rotor": g(rng.uniform(0.3, 0.45)), "payload": g(rng.uniform(0.25, 0.4)), "leds": False}
        if kind == "bird":
            return {"body": g(rng.uniform(0.7, 0.9)), "wing": g(rng.uniform(0.4, 0.6))}
        return {"body": g(rng.uniform(0.15, 0.35))}
    lit = rng.random() < 0.35  # underside lit by yard floodlights
    base = np.array([1.0, 0.62, 0.3]) * rng.uniform(0.15, 0.5) if lit else np.full(3, rng.uniform(0.01, 0.05))
    if kind in ("drone", "drone_payload"):
        return {"body": tuple(base), "motor": tuple(base * 0.8), "rotor": tuple(base * 1.3 + 0.02),
                "payload": tuple(base * rng.uniform(0.8, 1.6)), "leds": rng.random() < 0.6}
    if kind == "bird":
        return {"body": tuple(base * rng.uniform(0.9, 1.4)), "wing": tuple(base)}
    return {"body": tuple(np.array([0.9, 0.85, 0.75]) * rng.uniform(0.2, 0.7))}


# ------------------------------------------------------------------ targets
def draw_drone(rng, cv, px, c, mode, payload):
    D = rng.uniform(0.25, 0.9)                     # motor-to-motor diagonal (m)
    ppm = px / D * SS                               # supersampled pixels per metre
    R = _rot(rng.uniform(0, 2 * np.pi), np.radians(rng.uniform(-20, 20)), np.radians(rng.uniform(-10, 10)))
    elev = np.radians(rng.uniform(-5, 40))
    pal = _palette(rng, mode, "drone")
    motors = np.array([[np.cos(a), np.sin(a), 0.0] for a in np.radians([45, 135, 225, 315])]) * D / 2
    P = lambda pts: _proj(np.atleast_2d(pts) @ R.T, elev, ppm, c)
    for m in motors:                                # arms
        a, b = P(np.zeros(3))[0], P(m)[0]
        cv.line(a, b, max(1, 0.05 * D * ppm), pal["body"])
    box = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-0.35, 0.35)]) * 0.14 * D
    cv.poly(P(box), pal["body"], hull=True)
    cv.poly(P(_ellipsoid(0.05 * D, 0.06 * D, 0.05 * D) + [0, 0.06 * D, -0.07 * D]), pal["body"], hull=True)  # gimbal
    for s in (-1, 1):                               # skids
        a, b = P([s * 0.1 * D, 0.12 * D, -0.04 * D])[0], P([s * 0.12 * D, 0.1 * D, -0.16 * D])[0]
        cv.line(a, b, max(1, 0.02 * D * ppm), pal["body"])
    for m in motors:
        cv.poly(P(_circle3(m + [0, 0, 0.03 * D], 0.24 * D)), pal["rotor"], alpha=0.28)
        cv.poly(P(_circle3(m, 0.04 * D, 10)), pal["motor"])
    ext = []
    if payload:
        L = min(rng.uniform(0.4, 2.0), 3.2 * D); sw = rng.normal(0, 0.15, 2)
        top = np.array([0, 0, -0.06 * D]) @ R.T
        bot = top + np.r_[sw * L, -L]
        s = rng.uniform(0.12, 0.3)
        cv.line(_proj(top[None], elev, ppm, c)[0], _proj(bot[None], elev, ppm, c)[0], max(1, 0.01 * ppm), pal["body"])
        cube = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 0)]) * s / 2 * [1, 1, 2] + bot
        cv.poly(_proj(cube, elev, ppm, c), pal["payload"], hull=True)
    if pal.get("leds"):
        for m, col in zip(motors, [(0.1, 1.0, 0.2), (1.0, 0.1, 0.05), (1.0, 0.1, 0.05), (0.1, 1.0, 0.2)]):
            cv.light(P(m - [0, 0, 0.03 * D])[0], np.array(col) * rng.uniform(0.6, 1.2), max(1.5, 0.02 * D * ppm))
        if rng.random() < 0.5:
            cv.light(P([0, -0.1 * D, 0.05 * D])[0], np.array([1, 1, 1]) * 1.5, max(2, 0.03 * D * ppm))
    return D


def draw_bird(rng, cv, px, c, mode):
    Wb = rng.uniform(0.3, 1.3)
    ppm = px / Wb * SS
    R = _rot(rng.uniform(0, 2 * np.pi), np.radians(rng.uniform(-25, 25)), np.radians(rng.uniform(-20, 20)))
    elev = np.radians(rng.uniform(-5, 40))
    pal = _palette(rng, mode, "bird")
    P = lambda pts: _proj(np.atleast_2d(pts) @ R.T, elev, ppm, c)
    phi = np.radians(rng.choice([rng.uniform(-40, 50), rng.uniform(0, 10)]))  # flapping vs gliding
    half = Wb / 2
    for s in (-1, 1):
        wing = np.array([[s * 0.04, 0.06, 0], [s * 0.5 * half, 0.06, 0.5 * half * np.sin(phi * 0.6)],
                         [s * half * np.cos(phi), -0.02, half * np.sin(phi)],
                         [s * 0.6 * half, -0.09, 0.6 * half * np.sin(phi * 0.6)], [s * 0.04, -0.07, 0]]) * [1, Wb, 1]
        cv.poly(P(wing), pal["wing"])
    cv.poly(P(_ellipsoid(0.05 * Wb, 0.2 * Wb, 0.05 * Wb, rng=rng)), pal["body"], hull=True)
    cv.poly(P(np.array([[0, -0.15, 0], [-0.07, -0.32, 0], [0.07, -0.32, 0]]) * Wb), pal["wing"])
    cv.poly(P(_ellipsoid(0.035 * Wb, 0.04 * Wb, 0.035 * Wb) + [0, 0.2 * Wb, 0.01 * Wb]), pal["body"], hull=True)
    return Wb


def draw_other(rng, cv, px, c, mode):
    sub = rng.choice(["bag", "aircraft"] if mode == "thermal" else ["bag", "aircraft", "insect"], p=None)
    pal = _palette(rng, mode, "other")
    if sub == "bag":
        n = rng.integers(7, 13); a = np.sort(rng.uniform(0, 2 * np.pi, n)); r = px / 2 * SS * rng.uniform(0.5, 1.0, n)
        cv.poly(np.c_[c[0] + r * np.cos(a), c[1] + r * np.sin(a) * rng.uniform(0.5, 1)], pal["body"], alpha=0.85)
    elif sub == "aircraft":
        sp = px / 2 * SS
        hot = 1.6 if mode == "thermal" else 1.2
        cols = [(1, 0.1, 0.05), (0.1, 1, 0.2), (1, 1, 1)] if mode != "thermal" else [(1, 1, 1)] * 3
        offs = [(-sp, 0), (sp, 0), (0, -0.2 * sp)]
        for (dx, dy), col in zip(offs, cols):
            cv.light(np.array([c[0] + dx, c[1] + dy]), np.array(col) * hot, max(2, 0.06 * sp))
            cv.poly(np.array([[c[0] + dx - 2, c[1] + dy - 2], [c[0] + dx + 2, c[1] + dy - 2],
                              [c[0] + dx + 2, c[1] + dy + 2], [c[0] + dx - 2, c[1] + dy + 2]]), (0.5,) * 3, alpha=0.05)
    else:  # insect very close to the lens, lit by IR illuminator: big soft blob
        r = px / 2 * SS
        m = np.zeros((cv.n, cv.n), np.uint8)
        cv2.circle(m, tuple(np.round(c).astype(int)), int(r * 0.5), 255, -1)
        cv._apply(m, np.array([0.85, 0.8, 0.7]) * rng.uniform(0.4, 0.9), 0.7)
        cv.A = cv2.GaussianBlur(cv.A, (0, 0), r * 0.25); cv.P = cv2.GaussianBlur(cv.P, (0, 0), r * 0.25)
    return sub


# ------------------------------------------------------------------ degradations
def _degrade_patch(rng, P, A, L, px):
    k = P.shape[0]
    shear = rng.uniform(-0.15, 0.15) if rng.random() < 0.5 else 0.0  # rolling shutter
    if shear:
        M = np.float32([[1, shear, -shear * k / 2], [0, 1, 0]])
        P, A, L = (cv2.warpAffine(x, M, (k, k)) for x in (P, A, L))
    Lm = int(rng.uniform(0, 0.25 * px + 3)) if rng.random() < 0.6 else 0  # motion blur
    if Lm >= 2:
        ker = np.zeros((Lm * 2 + 1, Lm * 2 + 1), np.float32)
        a = rng.uniform(0, np.pi)
        cv2.line(ker, (Lm - int(Lm * np.cos(a)), Lm - int(Lm * np.sin(a))),
                 (Lm + int(Lm * np.cos(a)), Lm + int(Lm * np.sin(a))), 1.0, 1)
        ker /= ker.sum()
        P, A, L = (cv2.filter2D(x, -1, ker) for x in (P, A, L))
    s = rng.uniform(0, 1.3)  # defocus
    if s > 0.3:
        P, A, L = (cv2.GaussianBlur(x, (0, 0), s) for x in (P, A, L))
    return P, A, L


# ------------------------------------------------------------------ backgrounds
def _lowfreq(rng, h, w, cells=(6, 8)):
    return cv2.resize(rng.random(cells).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)


def background(rng, mode):
    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    cloud = np.clip(_lowfreq(rng, H, W, (5, 7)) * 1.4 - 0.4, 0, 1)[..., None] * rng.uniform(0, 1)
    if mode == "thermal":
        sky = rng.uniform(0.06, 0.18) + 0.08 * yy + cloud * rng.uniform(0.05, 0.15)
        img = np.repeat(sky, 3, axis=2) * np.ones((H, W, 3), np.float32)
        if rng.random() < 0.35:  # warm roofline at the bottom
            top = int(rng.uniform(0.75, 0.95) * H)
            prof = (top + 15 * _lowfreq(rng, 1, W, (1, 6))[0] * rng.random()).astype(int)
            v = rng.uniform(0.3, 0.55)
            for x in range(W):
                img[prof[x]:, x] = v
        return img, {"structures": True}
    glow = np.array([1.0, 0.6, 0.32], np.float32) * rng.uniform(0.03, 0.22)
    navy = np.array([0.05, 0.06, 0.12], np.float32) * rng.uniform(0.2, 1.0)
    img = (navy * (1 - yy) + glow * (0.4 + 0.6 * yy)) * (1 + 0.8 * cloud)
    img = img * np.ones((H, W, 3), np.float32)
    if cloud.mean() < 0.15:
        for _ in range(rng.integers(0, 25)):
            img[rng.integers(0, H), rng.integers(0, W)] += rng.uniform(0.1, 0.5)
    meta = {"structures": False}
    if rng.random() < 0.3:   # fence + light pole silhouette along the bottom
        meta["structures"] = True
        top = int(rng.uniform(0.8, 0.95) * H)
        img[top:] *= 0.15
        for x in range(0, W, int(rng.uniform(40, 90))):
            cv2.line(img, (x, top), (x, top - 25), (0.01, 0.01, 0.01), 2)
        wire = (top - 30 + 6 * np.sin(np.arange(W) / 5.0)).astype(int)
        for x in range(W):
            img[wire[x]:wire[x] + 2, x] = 0.02
        if rng.random() < 0.6:
            px = int(rng.uniform(40, W - 40)); ly = int(top - rng.uniform(60, 150))
            cv2.line(img, (px, top), (px, ly), (0.02, 0.02, 0.02), 3)
            g = np.zeros((H, W), np.float32); cv2.circle(g, (px, ly), 4, 1.0, -1)
            g = cv2.GaussianBlur(g, (0, 0), rng.uniform(15, 40)); g /= g.max()
            img += g[..., None] * np.array([1.0, 0.65, 0.3]) * rng.uniform(0.4, 1.0)
    if rng.random() < 0.25:  # off-frame floodlight glare
        g = np.zeros((H, W), np.float32)
        cv2.circle(g, (int(rng.choice([-40, W + 40])), int(rng.uniform(H * 0.5, H + 60))), 30, 1.0, -1)
        g = cv2.GaussianBlur(g, (0, 0), 120); g /= max(g.max(), 1e-6)
        img += g[..., None] * np.array([1.0, 0.7, 0.4]) * rng.uniform(0.2, 0.6)
    return img, meta


def sensor_finish(rng, img, mode):
    if mode == "thermal":
        g = img[..., 0]
        g = g + rng.normal(0, 0.006, (1, W)).astype(np.float32) + rng.normal(0, rng.uniform(0.008, 0.02), (H, W))
        img = np.repeat(g[..., None], 3, axis=2)
    else:
        s0 = rng.uniform(0.006, 0.03)
        img = img + rng.normal(0, 1, img.shape).astype(np.float32) * (s0 + 0.04 * np.sqrt(np.clip(img, 0, 1)))
        img = img + rng.normal(0, s0 * 0.5, (H, W, 1))
    yy, xx = np.mgrid[0:H, 0:W]
    vig = 1 - 0.25 * (((xx - W / 2) / W) ** 2 + ((yy - H / 2) / H) ** 2) * 2
    img = np.clip(img * vig[..., None], 0, 1)
    out = (img * 255).astype(np.uint8)[..., ::-1]  # RGB -> BGR for cv2
    q = int(rng.uniform(40, 90))
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR), q


# ------------------------------------------------------------------ frame assembly
def sample_px(rng, size, fov_min=np.radians(1.5), fov_max=np.radians(30)):
    for _ in range(50):
        R = np.exp(rng.uniform(np.log(60), np.log(1500)))
        fov = np.clip(0.5 / R * 16 * rng.uniform(0.6, 1.8), fov_min, fov_max)
        px = size / R / fov * W
        if 3 <= px <= 160:
            return R, fov, px
    return 300.0, np.radians(3), 20.0


def place(rng, img, P, A, L, cx, cy):
    """Composite patch centred at (cx,cy). Returns box (x0,y0,x1,y1) or None."""
    k = P.shape[0]
    x0, y0 = int(round(cx - k / 2)), int(round(cy - k / 2))
    sx0, sy0 = max(0, -x0), max(0, -y0)
    sx1, sy1 = min(k, W - x0), min(k, H - y0)
    if sx1 <= sx0 or sy1 <= sy0:
        return None, 0.0
    a = A[sy0:sy1, sx0:sx1, None]
    reg = img[y0 + sy0:y0 + sy1, x0 + sx0:x0 + sx1]
    before = reg.copy()
    reg[:] = reg * (1 - a) + P[sy0:sy1, sx0:sx1] + L[sy0:sy1, sx0:sx1]
    vis = (A[sy0:sy1, sx0:sx1] > 0.12) | (L[sy0:sy1, sx0:sx1].max(axis=2) > 0.15)
    ys, xs = np.nonzero(vis)
    if len(xs) == 0:
        return None, 0.0
    # contrast: mean change the object makes to the 6 most-changed % of its pixels
    d = np.abs(reg - before).max(axis=2)[vis]
    contrast = float(np.mean(np.sort(d)[-max(1, len(d) // 16):]))
    return (x0 + sx0 + xs.min(), y0 + sy0 + ys.min(), x0 + sx0 + xs.max() + 1, y0 + sy0 + ys.max() + 1), contrast


# Resource limits (security audit F-01). The sprite canvas is supersampled (SS^2) and holds
# ~7 float32 planes, and the payload line makes it 8x the target size; an unclamped px of 200
# needed 2.4 GB / 21 s and px 800 was OOM-killed. Targets bigger than MAX_PX are drawn at
# MAX_PX: the frame is only 640 px wide, so nothing realistic is lost.
MAX_PX = 128
MAX_CANVAS = 512  # base pixels per side before supersampling (~117 MB per plane set)


def render_object(rng, cls, px, mode, img, cx, cy):
    px = float(min(px, MAX_PX))
    size = min(int(np.ceil(px * (8.0 if cls == "drone_payload" else 3.0))) + 16, MAX_CANVAS)
    cv = Canvas(size)
    c = np.array([cv.n / 2, cv.n / 2])
    if cls in ("drone", "drone_payload"):
        draw_drone(rng, cv, px, c, mode, payload=(cls == "drone_payload")); sub = cls
    elif cls == "bird":
        draw_bird(rng, cv, px, c, mode); sub = "bird"
    else:
        sub = draw_other(rng, cv, px, c, mode)
    P, A, L = cv.finish()
    P, A, L = _degrade_patch(rng, P, A, L, px)
    box, contrast = place(rng, img, P, A, L, cx, cy)
    return box, sub, contrast


MIN_CONTRAST = 0.03  # objects that change the image less than this (0-1 scale) are not labelled

SIZES = {"drone": (0.25, 0.9), "drone_payload": (0.25, 0.9), "bird": (0.3, 1.3), "other": (0.3, 0.6)}


def render_frame(rng, mode=None, main_cls=None, px=None):
    """Render one labelled frame. Returns (bgr image, objects list, frame meta)."""
    # Input validation (security audit F-01/F-07): this is reachable from RenderedModelClassifier
    # with simulator-derived values, so reject garbage instead of rendering it.
    if mode not in (None, "night", "thermal"):
        raise ValueError(f"mode must be None, 'night' or 'thermal', got {mode!r}")
    if main_cls is not None and main_cls not in CLASSES:
        raise ValueError(f"main_cls must be one of {CLASSES} or None, got {main_cls!r}")
    if px is not None:
        px = float(px)
        if not np.isfinite(px) or px <= 0:
            raise ValueError(f"px must be a positive finite number, got {px!r}")
        px = min(px, MAX_PX)
    mode = mode or ("thermal" if rng.random() < 0.4 else "night")
    if main_cls is None:
        main_cls = rng.choice(["drone", "drone_payload", "bird", "other", None], p=[0.3, 0.2, 0.3, 0.1, 0.1])
    img, bmeta = background(rng, mode)
    objs = []
    todo = []
    if main_cls is not None:
        if px is None:
            R, fov, px = sample_px(rng, rng.uniform(*SIZES[main_cls]))
        else:
            R, fov = None, None
        if main_cls == "other" and mode == "night" and rng.random() < 0.3:
            px = rng.uniform(20, 120)  # insect near lens: big
        if rng.random() < 0.7:
            cx, cy = W / 2 + rng.normal(0, 40), H / 2 + rng.normal(0, 40)
        else:
            cx, cy = rng.uniform(30, W - 30), rng.uniform(30, H * 0.8)
        if bmeta.get("structures"):
            cy = min(cy, H * 0.62)  # keep targets in the sky, above rooflines / fence
        todo.append((main_cls, px, cx, cy, R, fov))
        if rng.random() < 0.25:  # a few birds sharing the frame
            for _ in range(rng.integers(1, 4)):
                todo.append(("bird", px * rng.uniform(0.5, 1.5), cx + rng.normal(0, 120),
                             min(cy + rng.normal(0, 80), H * 0.7), None, None))
    for cls, p_, cx, cy, R, fov in todo:
        res = render_object(rng, cls, p_, mode, img, cx, cy)
        box, sub, contrast = res
        if box is None or contrast < MIN_CONTRAST:
            continue  # rendered but effectively invisible (dark target on dark sky): leave unlabelled
        x0, y0, x1, y1 = box
        if (x1 - x0) * (y1 - y0) < 4:
            continue
        objs.append({"cls": cls, "sub": sub, "box": [int(x0), int(y0), int(x1), int(y1)], "px": float(p_),
                     "range_m": None if R is None else float(R), "fov_deg": None if fov is None else float(np.degrees(fov)), "contrast": round(contrast, 3)})
    out, q = sensor_finish(rng, img, mode)
    return out, objs, {"mode": mode, "jpeg_q": q, **bmeta}
