"""Already-YOLO datasets (Roboflow exports, own footage labelled in CVAT, synthetic data).

Class names come from data.yaml `names`, or classes.txt / obj.names, and are remapped
into the 4 training classes. Sequence = image folder relative to images/."""
from __future__ import annotations

from pathlib import Path

import yaml

from .base import IMG_EXT, Box, SchemaError, Writer, cls_of, read_image


def _names(src: Path) -> list[str]:
    for y in sorted(src.rglob("data.yaml")) + sorted(src.rglob("*.yaml")):
        try:
            d = yaml.safe_load(y.read_text(encoding="utf-8"))
        except yaml.YAMLError:
            continue
        if isinstance(d, dict) and "names" in d:
            n = d["names"]
            return [n[k] for k in sorted(n)] if isinstance(n, dict) else list(n)
    for t in ("classes.txt", "obj.names"):
        for p in src.rglob(t):
            return [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    raise SchemaError(f"no class names (data.yaml names / classes.txt / obj.names) under {src}")


def _label_for(img: Path) -> Path:
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    return img.with_suffix(".txt")


def convert(src: Path, out_root: Path, ds: dict, class_map: dict | None = None, frame_step: int = 1,
            keep_empty: bool = True, names: list[str] | None = None, **_) -> list:
    src = Path(src)
    names = names or _names(src)
    imgs = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT and p.is_file())
    if not imgs:
        raise SchemaError(f"no images under {src}")
    w = Writer(out_root, ds, keep_empty=keep_empty)
    for k, ip in enumerate(imgs):
        if k % max(1, frame_step):
            continue
        img = read_image(ip)
        H, W = img.shape[:2]
        boxes = []
        lp = _label_for(ip)
        if lp.is_file():
            for ln, line in enumerate(lp.read_text(encoding="utf-8").splitlines(), 1):
                t = line.split()
                if not t:
                    continue
                if len(t) < 5:
                    raise SchemaError(f"{lp}:{ln}: expected 'cls cx cy w h'")
                ci = int(float(t[0]))
                if not 0 <= ci < len(names):
                    raise SchemaError(f"{lp}:{ln}: class id {ci} outside names list")
                cx, cy, bw, bh = (float(v) for v in t[1:5])
                boxes.append(Box.from_cxcywh(cls_of(names[ci], class_map), cx * W, cy * H, bw * W, bh * H))
        rel = ip.parent.relative_to(src).as_posix()
        seq = rel.replace("images/", "").replace("images", "") or "root"
        w.add(img, boxes, seq, k, name=ip.stem)
    return w.rows
