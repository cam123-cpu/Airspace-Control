"""Common converter interface.

Every converter module exposes

    convert(src: Path, out_root: Path, ds: dict, **opts) -> list[ManifestRow]

`src` is the extracted dataset folder, `ds` is the catalog entry (license is copied onto
every row), and output follows the YOLO layout

    out_root/images/<source>/<sequence>/<frame>.jpg
    out_root/labels/<source>/<sequence>/<frame>.txt   (class cx cy w h, normalised)

Converters raise SchemaError with a precise message when the files do not look like the
documented format, rather than silently producing empty labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from ..common import CLASS_ID, ManifestRow, map_class, safe_join, safe_name

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


class SchemaError(ValueError):
    pass


@dataclass
class Box:
    cls: str            # training class name
    x0: float
    y0: float
    x1: float
    y1: float

    def clip(self, w: int, h: int) -> "Box | None":
        x0, y0 = max(0.0, self.x0), max(0.0, self.y0)
        x1, y1 = min(float(w), self.x1), min(float(h), self.y1)
        if x1 - x0 < 1.0 or y1 - y0 < 1.0:
            return None
        return Box(self.cls, x0, y0, x1, y1)

    @staticmethod
    def from_xywh(cls: str, x, y, w, h) -> "Box":
        return Box(cls, float(x), float(y), float(x) + float(w), float(y) + float(h))

    @staticmethod
    def from_cxcywh(cls: str, cx, cy, w, h) -> "Box":
        cx, cy, w, h = map(float, (cx, cy, w, h))
        return Box(cls, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def yolo_lines(boxes: list[Box], w: int, h: int) -> list[str]:
    out = []
    for b in boxes:
        c = b.clip(w, h)
        if c is None:
            continue
        cx, cy = (c.x0 + c.x1) / 2 / w, (c.y0 + c.y1) / 2 / h
        bw, bh = (c.x1 - c.x0) / w, (c.y1 - c.y0) / h
        out.append(f"{CLASS_ID[c.cls]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return out


class Writer:
    """Writes images + YOLO labels and builds manifest rows for one source."""

    def __init__(self, out_root: Path, ds: dict, modality: str | None = None, context: str = "",
                 jpeg_quality: int = 95, keep_empty: bool = True):
        self.root = Path(out_root)
        self.ds = ds
        self.source = safe_name(ds["id"])
        mods = ds.get("modality") or ["rgb"]
        self.modality = modality or (mods[0] if isinstance(mods, list) else str(mods))
        self.context = context or ", ".join(ds.get("contexts", [])[:3])
        self.q = int(jpeg_quality)
        self.keep_empty = keep_empty
        self.rows: list[ManifestRow] = []

    def add(self, img: np.ndarray, boxes: list[Box], sequence: str, frame: int, name: str | None = None,
            extra: dict | None = None) -> ManifestRow | None:
        if img is None or img.ndim not in (2, 3):
            raise SchemaError(f"unreadable image for {sequence}:{frame}")
        h, w = img.shape[:2]
        lines = yolo_lines(boxes, w, h)
        if not lines and not self.keep_empty:
            return None
        seq = safe_name(sequence)
        stem = safe_name(name) if name else f"{frame:07d}"
        ip = safe_join(self.root, "images", self.source, seq, stem + ".jpg")
        lp = safe_join(self.root, "labels", self.source, seq, stem + ".txt")
        ip.parent.mkdir(parents=True, exist_ok=True)
        lp.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(ip), img, [cv2.IMWRITE_JPEG_QUALITY, self.q]):
            raise OSError(f"could not write {ip}")
        lp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        row = ManifestRow(image=str(ip.relative_to(self.root)), label=str(lp.relative_to(self.root)),
                          source=self.ds["id"], license=str(self.ds["license"]),
                          commercial_ok=str(self.ds["commercial_ok"]), sequence=f"{self.source}/{seq}",
                          modality=self.modality, context=self.context, frame=int(frame),
                          extra={"n_boxes": len(lines), **(extra or {})})
        self.rows.append(row)
        return row


def read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise SchemaError(f"cannot read image {path}")
    return img


def iter_video(path: Path, frame_step: int = 1, wanted: set[int] | None = None,
               start_index: int = 0) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_index, BGR frame). frame_index counts from start_index (0 or 1)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SchemaError(f"cannot open video {path}")
    i = start_index
    try:
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if (wanted is None or i in wanted) and (i - start_index) % max(1, frame_step) == 0:
                yield i, fr
            i += 1
    finally:
        cap.release()


def index_images(root: Path) -> dict[str, Path]:
    """Map file name and stem -> path for every image under root (first one wins)."""
    idx: dict[str, Path] = {}
    for p in sorted(Path(root).rglob("*")):
        if p.suffix.lower() in IMG_EXT and p.is_file():
            idx.setdefault(p.name, p)
            idx.setdefault(p.stem, p)
    return idx


def cls_of(name: str, class_map: dict | None) -> str:
    return map_class(name, class_map)
