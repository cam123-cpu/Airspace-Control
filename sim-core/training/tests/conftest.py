import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]  # sim-core/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sky(w=160, h=120, seed=0, val=40):
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), val, np.uint8)
    img = np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    return img


def draw_target(img, x0, y0, x1, y1, val=230):
    img[int(y0):int(y1), int(x0):int(x1)] = val
    return img


def write_video(path: Path, frames):
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()
    return path


@pytest.fixture
def ds_yes():
    return {"id": "usc_drone", "name": "t", "license": "permissive", "commercial_ok": "yes", "access": "direct",
            "modality": ["rgb"], "contexts": ["sky"]}


@pytest.fixture
def ds_no():
    return {"id": "drone_vs_bird", "name": "t", "license": "research", "commercial_ok": "no", "access": "request",
            "modality": ["rgb"]}
