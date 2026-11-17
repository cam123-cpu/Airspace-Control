"""STRETCH / EXPERIMENTAL: track-level drone-vs-bird classifier over several frames.

A single frame of a 10-pixel target is often ambiguous. Over 1-3 seconds a bird's flapping
makes its box size and brightness oscillate at 2-8 Hz and its path wander, while a
multirotor holds a steady size and moves smoothly (and can hover). This module turns a
short track (per-frame boxes + mean target brightness) into features and fits a small
logistic regression in numpy, so it needs no extra dependencies.

Input: JSONL, one track per line
    {"label": "drone"|"bird", "fps": 25, "boxes": [[x0,y0,x1,y1], ...], "intensity": [...]}
Tracks can come from the tracking datasets (Anti-UAV, CST, DUT tracking, FBD-SV with
track_id) or from your own footage.

    python -m training.temporal fit --tracks tracks.jsonl --out temporal.json
    python -m training.temporal score --model temporal.json --tracks new.jsonl

Not yet validated on real tracks; treat its output as one more evidence term for fusion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FEATURES = ["size_cv", "aspect_cv", "osc_power_2_8hz", "intensity_cv", "speed_cv", "turn_rate", "mean_size"]


def features(boxes, intensity, fps: float) -> np.ndarray:
    b = np.asarray(boxes, float)
    if len(b) < 8:
        raise ValueError("need at least 8 frames per track")
    w, h = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
    size = np.sqrt(np.clip(w * h, 1e-6, None))
    asp = w / np.clip(h, 1e-6, None)
    cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
    v = np.stack([np.diff(cx), np.diff(cy)], 1) * fps
    spd = np.linalg.norm(v, axis=1)
    ang = np.unwrap(np.arctan2(v[:, 1], v[:, 0] + 1e-9))
    turn = float(np.mean(np.abs(np.diff(ang))) * fps) if len(ang) > 2 else 0.0
    s = (size - size.mean()) / (size.std() + 1e-9)
    spec = np.abs(np.fft.rfft(s)) ** 2
    freqs = np.fft.rfftfreq(len(s), 1 / fps)
    band = (freqs >= 2) & (freqs <= 8)
    osc = float(spec[band].sum() / (spec[1:].sum() + 1e-9)) if band.any() else 0.0
    inten = np.asarray(intensity if intensity is not None else np.ones(len(b)), float)
    cv = lambda x: float(np.std(x) / (abs(np.mean(x)) + 1e-9))
    return np.array([cv(size), cv(asp), osc, cv(inten), cv(spd + 1e-3), turn, float(size.mean())])


def fit(X: np.ndarray, y: np.ndarray, l2: float = 1e-2, iters: int = 2000, lr: float = 0.1) -> dict:
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ w))
        g = Z.T @ (p - y) / len(y) + l2 * np.r_[0, w[1:]]
        w -= lr * g
    return {"features": FEATURES, "mu": mu.tolist(), "sd": sd.tolist(), "w": w.tolist(), "positive": "drone"}


def predict(model: dict, X: np.ndarray) -> np.ndarray:
    Z = np.c_[np.ones(len(X)), (X - np.array(model["mu"])) / np.array(model["sd"])]
    return 1 / (1 + np.exp(-Z @ np.array(model["w"])))


def _load(path: Path):
    X, y = [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            X.append(features(d["boxes"], d.get("intensity"), float(d.get("fps", 25))))
            y.append(1.0 if d.get("label") == "drone" else 0.0)
    return np.array(X), np.array(y)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--tracks", type=Path, required=True)
    f.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("score")
    s.add_argument("--model", type=Path, required=True)
    s.add_argument("--tracks", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "fit":
        X, y = _load(a.tracks)
        m = fit(X, y)
        acc = float(((predict(m, X) > 0.5) == y).mean())
        a.out.write_text(json.dumps(m, indent=2), encoding="utf-8")
        print(f"fitted on {len(y)} tracks, training accuracy {acc:.3f} (use a held-out set before trusting it)")
    else:
        m = json.loads(a.model.read_text(encoding="utf-8"))
        X, _ = _load(a.tracks)
        for p in predict(m, X):
            print(f"{p:.3f}")


if __name__ == "__main__":
    main()
